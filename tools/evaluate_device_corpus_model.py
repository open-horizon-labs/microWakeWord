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
from tools.evaluate_recipe_model import peak_probability


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


def evaluate(
    corpus: Path,
    manifest: dict,
    model_path: Path,
    split: str,
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
            groups[("truth", truth)].append(peak)
            groups[("device_profile", item["device_profile"])].append(peak)
            groups[
                ("device_profile_by_truth", f'{item["device_profile"]}:{truth}')
            ].append(peak)
            groups[("phrase", item["phrase"])].append(peak)
            pronunciation = item.get("pronunciation")
            if pronunciation:
                groups[("pronunciation", pronunciation)].append(peak)
            # This cohort proves provisional detector misses remain first-class data.
            outcome = (
                "provisional_detected" if item["detected"] else "provisional_missed"
            )
            groups[("source_detector_outcome", outcome)].append(peak)
            groups[("truth_by_source_detector_outcome", f"{truth}:{outcome}")].append(
                peak
            )
    result: dict[str, dict] = defaultdict(dict)
    for (dimension, label), peaks in sorted(groups.items()):
        result[dimension][label] = summarize(peaks, cutoff)
    return dict(result)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--cutoff", type=float, required=True)
    parser.add_argument("--sliding-window", type=int, default=5)
    parser.add_argument("--ignore-initial", type=int, default=25)
    parser.add_argument("--clip-duration-ms", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = validate_device_corpus(args.corpus)
    report = {
        "corpus_id": manifest["corpus_id"],
        "model": str(args.model),
        "split": args.split,
        "cutoff": args.cutoff,
        "metrics": evaluate(
            args.corpus,
            manifest,
            args.model,
            args.split,
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
