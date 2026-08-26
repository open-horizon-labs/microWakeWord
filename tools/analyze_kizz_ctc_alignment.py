#!/usr/bin/env python3
"""Measure bounded teacher/student CTC timing mismatch on positive examples."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from microwakeword.ctc_occupancy import ctc_state_occupation_log_probs
from microwakeword.phoneme_student import (
    compact_phone_contract,
    resample_log_posteriors,
    student_output_times_seconds,
)
from microwakeword.ordered_state_model import model as build_student
from tools.cache_kizz_phoneme_teacher_posteriors import load_cache
from tools.distill_kizz_phoneme_student import OUTPUT_FRAMES, INPUT_SHAPE, sha256_file
from tools.distill_kizz_student import student_flags


def _log_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def delay_cross_entropies(
    student_logits: np.ndarray, teacher_log_occupation: np.ndarray, max_delay: int
) -> np.ndarray:
    """Return CE for causal student delays 0..max_delay."""
    if student_logits.shape != teacher_log_occupation.shape:
        raise ValueError("student and teacher occupation shapes differ")
    if student_logits.ndim != 2 or not 0 <= max_delay < len(student_logits):
        raise ValueError("invalid sequence rank or maximum delay")
    student = _log_softmax(np.asarray(student_logits, dtype=np.float64))
    teacher = np.exp(np.asarray(teacher_log_occupation, dtype=np.float64))
    losses = []
    for delay in range(max_delay + 1):
        usable = len(student) - delay
        losses.append(
            -np.sum(teacher[:usable] * student[delay : delay + usable], axis=-1).mean()
        )
    return np.asarray(losses, dtype=np.float64)


def _summary(records: list[dict]) -> dict:
    delays = np.asarray([row["best_delay_frames"] for row in records], dtype=np.int64)
    fixed = np.asarray([row["fixed_cross_entropy"] for row in records])
    best = np.asarray([row["best_cross_entropy"] for row in records])
    return {
        "examples": len(records),
        "best_delay_histogram": {
            str(key): int(value) for key, value in sorted(Counter(delays).items())
        },
        "median_best_delay_frames": float(np.median(delays)),
        "p90_best_delay_frames": float(np.quantile(delays, 0.9)),
        "mean_fixed_cross_entropy": float(fixed.mean()),
        "mean_best_cross_entropy": float(best.mean()),
        "mean_relative_cross_entropy_reduction": float(
            np.mean(
                np.divide(
                    fixed - best, fixed, out=np.zeros_like(fixed), where=fixed > 0
                )
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--posterior-cache", type=Path, required=True)
    parser.add_argument("--student-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-delay-frames", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    corpus_path = args.corpus / "corpus.json"
    corpus = json.loads(corpus_path.read_text())
    rows = corpus["examples"]
    contract = compact_phone_contract()
    if corpus.get("compact_phone_contract") != contract:
        parser.error("corpus compact-phone contract differs")
    features = np.load(args.corpus / "features.npy", mmap_mode="r")
    if len(features) != len(rows):
        parser.error("corpus feature count differs")
    declared_cache = json.loads(args.posterior_cache.with_suffix(".json").read_text())
    cache_model = declared_cache.get("model", {})
    metadata, arrays = load_cache(
        args.posterior_cache,
        expected_model_revision=cache_model.get("revision"),
        expected_weights_sha256=cache_model.get("weights_sha256"),
    )
    if metadata.get("vocabulary", {}).get("tokens") != contract["tokens"]:
        parser.error("posterior cache vocabulary differs")
    offsets = arrays["offsets"]
    if len(offsets) != len(rows) + 1:
        parser.error("posterior cache row count differs")

    model = build_student(student_flags(len(contract["tokens"])), INPUT_SHAPE, None)
    model.load_weights(args.student_weights)
    positive_indexes = [
        index for index, row in enumerate(rows) if int(row["label"]) == 1
    ]
    predictions = model.predict(
        np.asarray(features[positive_indexes], dtype=np.float32),
        batch_size=args.batch_size,
        verbose=0,
    )
    times = student_output_times_seconds(
        student_flags(len(contract["tokens"])), OUTPUT_FRAMES
    )
    timing = metadata["timing"]
    records = []
    for prediction, index in zip(predictions, positive_indexes):
        teacher_frames = arrays["log_posteriors"][offsets[index] : offsets[index + 1]]
        occupation = ctc_state_occupation_log_probs(
            teacher_frames, contract["canonical_path"], int(contract["blank_id"])
        )
        occupation = resample_log_posteriors(
            occupation,
            teacher_frame_center_seconds=float(timing["frame_center_seconds"]),
            teacher_frame_stride_seconds=float(timing["frame_stride_seconds"]),
            student_times_seconds=times,
        )
        losses = delay_cross_entropies(prediction, occupation, args.max_delay_frames)
        best_delay = int(np.argmin(losses))
        row = rows[index]
        records.append(
            {
                "index": index,
                "source_id": row.get("source_id"),
                "split": row.get("split"),
                "provider": row.get("provider"),
                "variant": row.get("source_group", "clean"),
                "best_delay_frames": best_delay,
                "best_delay_ms": best_delay * 30,
                "fixed_cross_entropy": float(losses[0]),
                "best_cross_entropy": float(losses[best_delay]),
            }
        )

    buckets = defaultdict(list)
    for row in records:
        buckets[f"split:{row['split']}"].append(row)
        buckets[f"provider:{row['provider']}"].append(row)
    report = {
        "schema_version": 1,
        "kind": "kizz_teacher_student_ctc_alignment_diagnostic",
        "inputs": {
            "corpus": str(corpus_path.resolve()),
            "corpus_sha256": sha256_file(corpus_path),
            "posterior_cache": str(args.posterior_cache.resolve()),
            "posterior_cache_sha256": metadata.get("cache_sha256"),
            "student_weights": str(args.student_weights.resolve()),
            "student_weights_sha256": sha256_file(args.student_weights),
        },
        "delay_contract": {
            "direction": "teacher_frame_t_matches_student_frame_t_plus_delay",
            "frame_ms": 30,
            "maximum_delay_frames": args.max_delay_frames,
        },
        "overall": _summary(records),
        "buckets": {key: _summary(value) for key, value in sorted(buckets.items())},
        "examples": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["overall"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
