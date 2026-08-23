#!/usr/bin/env python3
"""Evaluate a model against real device attempts, including detector misses."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from microwakeword.device_corpus import captures_for, validate_device_corpus
from microwakeword.inference import Model

if __package__:
    from tools.evaluate_recipe_model import peak_probability
else:
    from evaluate_recipe_model import peak_probability


def summarize(peaks: list[float], cutoff: float) -> dict:
    accepted = sum(peak > cutoff for peak in peaks)
    return {
        "attempts": len(peaks),
        "accepted": accepted,
        "acceptance_rate": accepted / len(peaks) if peaks else 0.0,
        "median_peak_probability": float(np.median(peaks)) if peaks else 0.0,
        "minimum_peak_probability": min(peaks, default=0.0),
        "maximum_peak_probability": max(peaks, default=0.0),
    }


def capture_dimensions(item: dict, truth: str) -> list[tuple[str, str]]:
    outcome = "provisional_detected" if item["detected"] else "provisional_missed"
    dimensions = [
        ("truth", truth),
        ("device_profile", item["device_profile"]),
        ("device_profile_by_truth", f'{item["device_profile"]}:{truth}'),
        ("speaker_id", item["speaker_id"]),
        ("speaker_id_by_truth", f'{item["speaker_id"]}:{truth}'),
        ("session_id", item["session_id"]),
        ("session_id_by_truth", f'{item["session_id"]}:{truth}'),
        ("phrase", item["phrase"]),
        ("source_detector_outcome", outcome),
        ("truth_by_source_detector_outcome", f"{truth}:{outcome}"),
    ]
    pronunciation = item.get("pronunciation")
    if pronunciation:
        dimensions.append(("pronunciation", pronunciation))
    return dimensions


def evaluate(
    corpus: Path,
    manifest: dict,
    model_path: Path,
    split: str | None,
    cutoff: float,
    sliding_window: int,
    ignore_initial: int,
    clip_duration_ms: int,
) -> dict:
    model = Model(str(model_path), stride=3)
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for truth in ("positive", "hard_negative", "ambient_negative"):
        for item, path in captures_for(corpus, manifest, truth, split):
            peak = peak_probability(
                model, path, sliding_window, ignore_initial, clip_duration_ms
            )
            for dimension in capture_dimensions(item, truth):
                groups[dimension].append(peak)
    result: dict[str, dict] = defaultdict(dict)
    for (dimension, label), peaks in sorted(groups.items()):
        result[dimension][label] = summarize(peaks, cutoff)
    return dict(result)


def qualification_scope(
    manifest: dict, split: str, required_age_groups: tuple[str, ...] = ()
) -> dict:
    selected = (
        manifest["captures"]
        if split == "all"
        else [item for item in manifest["captures"] if item["split"] == split]
    )
    counts = {
        truth: sum(item["truth"] == truth for item in selected)
        for truth in ("positive", "hard_negative", "ambient_negative")
    }
    speaker_ids = sorted(
        {
            item["speaker_id"]
            for item in selected
            if manifest["speakers"][item["speaker_id"]]["kind"] == "human"
        }
    )
    age_groups = sorted(
        {manifest["speakers"][speaker_id]["age_group"] for speaker_id in speaker_ids}
    )
    issues = []
    if split != "test":
        issues.append("qualification requires the test split")
    for truth, count in counts.items():
        if count == 0:
            issues.append(f"test split has no {truth} captures")
    if not speaker_ids:
        issues.append("test split has no registered human speakers")
    for age_group in required_age_groups:
        if age_group not in age_groups:
            issues.append(f"test split has no registered {age_group} human speaker")
    return {
        "includes_training_data": split in {"all", "train"},
        "qualification_eligible": not issues,
        "issues": issues,
        "capture_counts": counts,
        "human_speaker_ids": speaker_ids,
        "human_age_groups": age_groups,
        "session_ids": sorted({item["session_id"] for item in selected}),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("all", "train", "validation", "test"), default="test"
    )
    parser.add_argument("--cutoff", type=float, required=True)
    parser.add_argument("--sliding-window", type=int, default=5)
    parser.add_argument("--ignore-initial", type=int, default=25)
    parser.add_argument(
        "--clip-duration-ms",
        type=int,
        default=0,
        help="Optionally crop clips to this duration; 0 evaluates the full recording",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--qualification",
        action="store_true",
        help="Fail unless this is a complete, held-out physical test corpus",
    )
    parser.add_argument(
        "--required-age-group",
        choices=("adult", "child"),
        action="append",
        default=[],
        help="Require this human age cohort for qualification; repeatable",
    )
    args = parser.parse_args()
    manifest = validate_device_corpus(args.corpus)
    scope = qualification_scope(manifest, args.split, tuple(args.required_age_group))
    if args.qualification and not scope["qualification_eligible"]:
        parser.error("; ".join(scope["issues"]))
    report = {
        "corpus_id": manifest["corpus_id"],
        "model": str(args.model),
        "split": args.split,
        "cutoff": args.cutoff,
        "scope": scope,
        "metrics": evaluate(
            args.corpus,
            manifest,
            args.model,
            None if args.split == "all" else args.split,
            args.cutoff,
            args.sliding_window,
            args.ignore_initial,
            args.clip_duration_ms,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
