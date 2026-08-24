#!/usr/bin/env python3
"""Select a wake cutoff from validation without opening the test split."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import median

import numpy as np
import yaml

from microwakeword.inference import Model
from microwakeword.synthetic_quality import load_quality_mask

try:
    from tools.evaluate_recipe_model import clip_probabilities
except ModuleNotFoundError:  # Direct execution from the tools directory.
    from evaluate_recipe_model import clip_probabilities


def cutoff_for_false_accept_rate(
    negative_peaks: list[float], maximum_false_accept_rate: float
) -> float:
    """Return the lowest cutoff that stays within the false-accept budget."""
    if not negative_peaks:
        raise ValueError("cutoff selection requires validation hard negatives")
    if not 0 <= maximum_false_accept_rate < 1:
        raise ValueError("maximum false-accept rate must be in [0, 1)")
    allowed = math.floor(len(negative_peaks) * maximum_false_accept_rate)
    descending = sorted(negative_peaks, reverse=True)
    return descending[allowed]


def summarize(peaks: list[float], cutoff: float) -> dict:
    ordered = sorted(peaks)

    def quantile(probability: float) -> float:
        if not ordered:
            return 0.0
        position = probability * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        fraction = position - lower
        return ordered[lower] * (1 - fraction) + ordered[upper] * fraction

    accepted = sum(peak > cutoff for peak in peaks)
    return {
        "clips": len(peaks),
        "accepted": accepted,
        "acceptance_rate": accepted / len(peaks) if peaks else 0.0,
        "minimum_peak_probability": min(peaks, default=0.0),
        "median_peak_probability": median(peaks) if peaks else 0.0,
        "maximum_peak_probability": max(peaks, default=0.0),
        "peak_probability_quantiles": {
            "p05": quantile(0.05),
            "p25": quantile(0.25),
            "p50": quantile(0.50),
            "p75": quantile(0.75),
            "p95": quantile(0.95),
        },
    }


def piper_speakers(output: Path) -> dict[str, str]:
    """Return stable speaker-pair labels from Piper synthesis metadata."""
    metadata_path = output / "synthesis-metadata.jsonl"
    if not metadata_path.exists():
        return {}
    speakers = {}
    for line in metadata_path.read_text().splitlines():
        item = json.loads(line)
        filename = item.get("file")
        if filename is None:
            continue
        first = item.get("speaker_1")
        second = item.get("speaker_2")
        if first is None:
            continue
        speakers[filename] = (
            f"piper:{first}" if second is None else f"piper:{first}+{second}"
        )
    return speakers


def validation_records(
    generated: Path, generation_manifest: dict, rejected: set[Path]
) -> list[dict]:
    if generation_manifest.get("schema_version") != 2:
        raise ValueError("validation cutoff selection requires generation schema 2")
    records = []
    for item in generation_manifest.get("plan", []):
        if item.get("split") != "validation":
            continue
        output = Path(item["output"])
        provider = item.get("provider", "piper")
        piper_voice_by_file = piper_speakers(output) if provider == "piper" else {}
        for wav_path in sorted(output.glob("*.wav")):
            if wav_path.resolve() in rejected:
                continue
            records.append(
                {
                    "path": wav_path,
                    "truth": item["class"],
                    "phrase": item.get("text")
                    or item.get("text_source")
                    or "unlabeled",
                    "age_group": item.get("age_group", "unknown"),
                    "provider": provider,
                    "speaker": item.get("speaker_name")
                    or item.get("speaker_id")
                    or piper_voice_by_file.get(wav_path.name, "unknown"),
                }
            )
    if not records:
        raise ValueError("generation manifest contains no eligible validation clips")
    return records


def score_records(
    model_path: Path,
    records: list[dict],
    sliding_windows: list[int],
    ignore_initial: int,
    clip_duration_ms: int,
) -> dict[int, list[dict]]:
    model = Model(str(model_path), stride=3)
    scored = {window: [] for window in sliding_windows}
    for record in records:
        probabilities = clip_probabilities(
            model,
            record["path"],
            ignore_initial,
            clip_duration_ms if record["truth"] == "positive" else 0,
        )
        for window in sliding_windows:
            peak = 0.0
            if probabilities.size >= window:
                moving_average = np.convolve(
                    probabilities,
                    np.ones(window) / window,
                    mode="valid",
                )
                peak = float(np.max(moving_average))
            scored[window].append({**record, "peak": peak})
    return scored


def select_window(results: dict[int, dict]) -> int | None:
    """Choose on validation recall, preferring more smoothing on an exact tie."""
    eligible = [
        window
        for window, result in results.items()
        if result.get("qualification_eligible") is True
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda window: (
            results[window]["selected"]["positive"]["acceptance_rate"],
            window,
        ),
    )


def grouped_summary(records: list[dict], cutoff: float, dimension: str) -> dict:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        grouped[record[dimension]].append(record["peak"])
    return {label: summarize(peaks, cutoff) for label, peaks in sorted(grouped.items())}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--quality-mask", type=Path, required=True)
    parser.add_argument(
        "--sliding-window",
        type=int,
        action="append",
        default=[],
        help="Detector averaging window to compare; repeatable (default: 5)",
    )
    parser.add_argument("--ignore-initial", type=int, default=25)
    parser.add_argument(
        "--clip-duration-ms",
        type=int,
        help="Override the recipe clip duration",
    )
    parser.add_argument(
        "--maximum-false-accept-rate",
        type=float,
        default=0.001,
        help="Validation hard-negative acceptance budget used for selection",
    )
    parser.add_argument(
        "--frontier-rate",
        type=float,
        action="append",
        default=[],
        help="Additional validation false-accept budget to report; repeatable",
    )
    parser.add_argument(
        "--maximum-deployable-cutoff",
        type=float,
        default=0.99,
        help="Reject a selected operating point above the firmware limit",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.maximum_deployable_cutoff < 1:
        parser.error("maximum deployable cutoff must be between zero and one")

    manifest_path = args.generated / "generation-manifest.json"
    generation_manifest = json.loads(manifest_path.read_text())
    recipe = yaml.safe_load(args.recipe.read_text())
    clip_duration_ms = args.clip_duration_ms
    if clip_duration_ms is None:
        clip_duration_ms = int(recipe["clip_duration_ms"])
    mask = load_quality_mask(args.quality_mask, args.recipe, manifest_path)
    rejected = {(args.generated / relative).resolve() for relative in mask["rejected"]}
    records = validation_records(args.generated, generation_manifest, rejected)
    sliding_windows = sorted(set(args.sliding_window or [5]))
    if any(window < 1 for window in sliding_windows):
        parser.error("sliding windows must be positive")
    scored_by_window = score_records(
        args.model,
        records,
        sliding_windows,
        args.ignore_initial,
        clip_duration_ms,
    )
    rates = sorted(
        set(
            (
                0.0,
                args.maximum_false_accept_rate,
                0.005,
                0.01,
                *args.frontier_rate,
            )
        )
    )
    window_results = {}
    for window, scored in scored_by_window.items():
        positive_peaks = [
            record["peak"] for record in scored if record["truth"] == "positive"
        ]
        negative_peaks = [
            record["peak"] for record in scored if record["truth"] == "hard_negative"
        ]
        frontier = []
        for rate in rates:
            cutoff = cutoff_for_false_accept_rate(negative_peaks, rate)
            frontier.append(
                {
                    "maximum_false_accept_rate": rate,
                    "cutoff": cutoff,
                    "positive": summarize(positive_peaks, cutoff),
                    "hard_negative": summarize(negative_peaks, cutoff),
                }
            )
        selected_cutoff = cutoff_for_false_accept_rate(
            negative_peaks, args.maximum_false_accept_rate
        )
        selected_summary = {
            "positive": summarize(positive_peaks, selected_cutoff),
            "hard_negative": summarize(negative_peaks, selected_cutoff),
            "cohorts": {
                truth: {
                    "by_phrase": grouped_summary(
                        [record for record in scored if record["truth"] == truth],
                        selected_cutoff,
                        "phrase",
                    ),
                    "by_age_group": grouped_summary(
                        [record for record in scored if record["truth"] == truth],
                        selected_cutoff,
                        "age_group",
                    ),
                    "by_provider": grouped_summary(
                        [record for record in scored if record["truth"] == truth],
                        selected_cutoff,
                        "provider",
                    ),
                    "by_speaker": grouped_summary(
                        [record for record in scored if record["truth"] == truth],
                        selected_cutoff,
                        "speaker",
                    ),
                }
                for truth in ("positive", "hard_negative")
            },
        }
        issues = []
        if selected_cutoff > args.maximum_deployable_cutoff:
            issues.append(
                f"derived cutoff {selected_cutoff:.6f} exceeds deployable maximum "
                f"{args.maximum_deployable_cutoff:.6f}"
            )
        if selected_summary["positive"]["accepted"] == 0:
            issues.append("selected operating point accepts no validation positives")
        window_results[window] = {
            "selected_cutoff": selected_cutoff,
            "frontier": frontier,
            "selected": selected_summary,
            "qualification_eligible": not issues,
            "qualification_issues": issues,
        }
    selected_window = select_window(window_results)
    selected_result = (
        window_results[selected_window] if selected_window is not None else None
    )
    qualification_issues = (
        []
        if selected_result is not None
        else ["no window has a deployable, nonzero-recall validation operating point"]
    )
    result = {
        "model": str(args.model),
        "model_sha256": sha256(args.model),
        "selection_split": "validation",
        "selected_cutoff": (
            selected_result["selected_cutoff"] if selected_result is not None else None
        ),
        "maximum_false_accept_rate": args.maximum_false_accept_rate,
        "maximum_deployable_cutoff": args.maximum_deployable_cutoff,
        "qualification_eligible": selected_result is not None,
        "qualification_issues": qualification_issues,
        "frontier": selected_result["frontier"] if selected_result is not None else [],
        "selected": (
            selected_result["selected"] if selected_result is not None else None
        ),
        "sliding_window": selected_window,
        "window_selection_policy": (
            "maximum validation positive acceptance at each window's independently "
            "derived false-accept budget; prefer more smoothing on an exact tie"
        ),
        "window_comparison": {
            str(window): value for window, value in sorted(window_results.items())
        },
        "ignore_initial": args.ignore_initial,
        "clip_duration_ms": clip_duration_ms,
        "quality_mask": str(args.quality_mask),
        "quality_mask_sha256": sha256(args.quality_mask),
        "recipe": str(args.recipe),
        "recipe_sha256": sha256(args.recipe),
        "generation_manifest": str(manifest_path),
        "generation_manifest_sha256": sha256(manifest_path),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["qualification_eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
