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
    args = parser.parse_args(argv)
    if args.steps < 1 or args.batch_size < 2 or args.batch_size % 2:
        parser.error("steps must be positive and batch-size must be even")

    teacher = build_teacher()
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
    targets = np.empty((total, 66), dtype=np.int8)
    labels = np.empty((total,), dtype=np.float16)
    logits = np.empty((total, 66, 23), dtype=np.float16)
    for step in range(args.steps):
        x, batch = sequence[step]
        start = step * args.batch_size
        end = start + args.batch_size
        features[start:end] = x
        targets[start:end] = batch["states"]
        labels[start:end] = batch["label"]
        logits[start:end] = teacher.predict(x, verbose=0)
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
