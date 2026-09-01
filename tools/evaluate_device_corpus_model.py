#!/usr/bin/env python3
"""Evaluate a model against real device attempts, including detector misses."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from microwakeword.device_corpus import validate_device_corpus
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


def score_sequence(
    entries: list[tuple[dict, Path]],
    scorer,
    cutoff: float,
    state_mode: str,
) -> list[dict]:
    """Score captures in manifest order with explicit streaming-state policy."""
    reset_next = True
    results = []
    for item, path in entries:
        reset_state = state_mode == "reset_per_capture" or reset_next
        peak = scorer(path, reset_state)
        accepted = peak > cutoff
        results.append(
            {
                "capture_id": item["capture_id"],
                "truth": item["truth"],
                "detected": item["detected"],
                "accepted": accepted,
                "peak_probability": peak,
                "reset_before_capture": reset_state,
            }
        )
        reset_next = accepted
    return results


def selected_captures(
    corpus: Path, manifest: dict, split: str | None
) -> list[tuple[dict, Path]]:
    return [
        (item, corpus / item["path"])
        for item in manifest["captures"]
        if split is None or item["split"] == split
    ]


def evaluate(
    corpus: Path,
    manifest: dict,
    model_path: Path,
    split: str | None,
    cutoff: float,
    sliding_window: int,
    ignore_initial: int,
    clip_duration_ms: int,
    state_mode: str = "reset_per_capture",
) -> tuple[dict, list[dict]]:
    model = Model(str(model_path), stride=3)
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    entries = selected_captures(corpus, manifest, split)

    def scorer(path: Path, reset_state: bool) -> float:
        return peak_probability(
            model,
            path,
            sliding_window,
            ignore_initial if reset_state else 0,
            clip_duration_ms,
            reset_state=reset_state,
        )

    capture_results = score_sequence(entries, scorer, cutoff, state_mode)
    by_id = {item["capture_id"]: item for item, _ in entries}
    for scored in capture_results:
        item = by_id[scored["capture_id"]]
        for dimension in capture_dimensions(item, item["truth"]):
            groups[dimension].append(scored["peak_probability"])
    result: dict[str, dict] = defaultdict(dict)
    for (dimension, label), peaks in sorted(groups.items()):
        result[dimension][label] = summarize(peaks, cutoff)
    return dict(result), capture_results


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
        "--state-mode",
        choices=("reset_per_capture", "carry_until_detection"),
        default="reset_per_capture",
        help=(
            "Reset streaming state for every clip, or preserve it across misses "
            "and reset only after a modeled detection. The latter better exposes "
            "runtime state sensitivity but does not recreate unrecorded gaps."
        ),
    )
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
    metrics, capture_results = evaluate(
        args.corpus,
        manifest,
        args.model,
        None if args.split == "all" else args.split,
        args.cutoff,
        args.sliding_window,
        args.ignore_initial,
        args.clip_duration_ms,
        args.state_mode,
    )
    provisional_matches = sum(
        result["detected"] == result["accepted"] for result in capture_results
    )
    report = {
        "corpus_id": manifest["corpus_id"],
        "model": str(args.model),
        "split": args.split,
        "cutoff": args.cutoff,
        "state_mode": args.state_mode,
        "scope": scope,
        "provisional_detector_agreement": {
            "matches": provisional_matches,
            "attempts": len(capture_results),
            "rate": provisional_matches / len(capture_results)
            if capture_results
            else 0.0,
        },
        "metrics": metrics,
        "captures": capture_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
