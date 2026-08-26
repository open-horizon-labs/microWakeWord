#!/usr/bin/env python3
"""Freeze teacher logits and hard targets for a reproducible student run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from microwakeword.kizz_teacher import (
    NegativeSource,
    TeacherBatchSequence,
    build_teacher,
)
from microwakeword.ordered_state import OrderedStateTopology
from microwakeword.wake_phrase import HI_FI_KIZZ, WAKE_PHRASES, get_wake_phrase


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_source(value: str) -> NegativeSource:
    if "=" not in value:
        raise argparse.ArgumentTypeError("negative source must be ID=PATH")
    source_id, raw_path = value.split("=", 1)
    path = Path(raw_path).resolve()
    if not source_id or not (path.is_dir() or (path.is_file() and path.suffix == ".npy")):
        raise argparse.ArgumentTypeError(
            "source must be ID=existing RaggedMmap directory or .npy cache"
        )
    return NegativeSource(source_id, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-weights", type=Path, required=True)
    parser.add_argument("--positive-features", type=Path, required=True)
    parser.add_argument("--positive-targets", type=Path, required=True)
    parser.add_argument("--negative-source", type=parse_source, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=24104)
    parser.add_argument(
        "--alignment-offset",
        type=int,
        default=21,
        help="Teacher 30-ms frames to skip before the student's first output.",
    )
    parser.add_argument("--student-output-frames", type=int, default=66)
    parser.add_argument(
        "--phrase-id",
        choices=tuple(sorted(WAKE_PHRASES)),
        default=HI_FI_KIZZ.phrase_id,
    )
    parser.add_argument("--states-per-phone", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args(argv)
    if args.steps < 1 or args.batch_size < 2 or args.batch_size % 2:
        parser.error("steps must be positive and batch-size must be even")

    positive_target_width = int(
        np.load(args.positive_targets, mmap_mode="r").shape[1]
    )
    if args.alignment_offset < 0 or (
        args.alignment_offset + args.student_output_frames > positive_target_width
    ):
        parser.error("teacher alignment slice does not fit the teacher timeline")
    phrase_spec = get_wake_phrase(args.phrase_id)
    topology = OrderedStateTopology(phrase_spec.phones, args.states_per_phone)
    teacher = build_teacher(output_frames=positive_target_width, topology=topology)
    teacher.load_weights(args.teacher_weights)
    sequence = TeacherBatchSequence(
        args.positive_features,
        args.positive_targets,
        args.negative_source,
        batch_size=args.batch_size,
        seed=args.seed,
        steps_per_epoch=args.steps,
    )
    total = args.steps * args.batch_size
    features = np.empty((total, 260, 40), dtype=np.float16)
    targets = np.empty((total, args.student_output_frames), dtype=np.int8)
    labels = np.empty((total,), dtype=np.float16)
    logits = np.empty(
        (total, args.student_output_frames, topology.state_count), dtype=np.float16
    )
    for step in range(args.steps):
        x, batch = sequence[step]
        start = step * args.batch_size
        end = start + args.batch_size
        features[start:end] = x
        targets[start:end] = batch["states"][:,
            args.alignment_offset : args.alignment_offset + args.student_output_frames
        ]
        labels[start:end] = batch["label"]
        teacher_logits = teacher.predict(x, verbose=0)
        logits[start:end] = teacher_logits[
            :, args.alignment_offset : args.alignment_offset + args.student_output_frames
        ]
        if (step + 1) % 50 == 0 or step == 0:
            print(json.dumps({"step": step + 1, "total": args.steps}), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output.with_name("features.npy"), features)
    np.save(args.output.with_name("targets.npy"), targets)
    np.save(args.output.with_name("labels.npy"), labels)
    np.save(args.output.with_name("teacher_logits.npy"), logits)
    metadata = {
        "schema_version": 1,
        "sample_count": total,
        "feature_shape": list(features.shape),
        "target_shape": list(targets.shape),
        "teacher_logit_shape": list(logits.shape),
        "teacher_weights": str(args.teacher_weights.resolve()),
        "teacher_weights_sha256": sha256_file(args.teacher_weights),
        "positive_features": str(args.positive_features.resolve()),
        "positive_features_sha256": sha256_file(args.positive_features),
        "positive_targets": str(args.positive_targets.resolve()),
        "positive_targets_sha256": sha256_file(args.positive_targets),
        "negative_sources": [
            {"id": source.source_id, "path": str(source.path)}
            for source in args.negative_source
        ],
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "teacher_output_frames": positive_target_width,
        "student_output_frames": args.student_output_frames,
        "alignment_offset_frames": args.alignment_offset,
        "alignment_basis": "student_valid_receptive_field_offset_64_frames_div_3",
        "topology": {
            "phrase_id": phrase_spec.phrase_id,
            "text": phrase_spec.text,
            "phones": list(phrase_spec.phones),
            "states_per_phone": topology.states_per_phone,
            "state_count": topology.state_count,
        },
    }
    metadata["cache_sha256"] = hashlib.sha256(
        b"".join(
            sha256_file(args.output.with_name(name)).encode()
            for name in ("features.npy", "targets.npy", "labels.npy", "teacher_logits.npy")
        )
    ).hexdigest()
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
