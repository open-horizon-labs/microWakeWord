#!/usr/bin/env python3
"""Score quarantined wake observations after validation freezes a cutoff."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np

from microwakeword.inference import Model

try:
    from tools.evaluate_recipe_model import peak_probability
except ModuleNotFoundError:  # Direct execution from the tools directory.
    from evaluate_recipe_model import peak_probability


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cutoff_contract(report_path: Path, model_path: Path) -> dict:
    report = json.loads(report_path.read_text())
    if report.get("selection_split") != "validation":
        raise ValueError("cutoff report was not selected on validation")
    if report.get("model_sha256") != sha256(model_path):
        raise ValueError("cutoff report model does not match evaluated model")
    cutoff = report.get("selected_cutoff")
    if not isinstance(cutoff, (int, float)):
        raise ValueError("cutoff report has no numeric selected_cutoff")
    return {
        "cutoff": float(cutoff),
        "sliding_window": int(report["sliding_window"]),
        "ignore_initial": int(report["ignore_initial"]),
        "clip_duration_ms": int(report["clip_duration_ms"]),
        "report_sha256": sha256(report_path),
    }


def observation_records(manifest_path: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("observation manifest requires schema_version 1")
    if manifest.get("training_eligible") is not False:
        raise ValueError("observation manifest must be explicitly training-ineligible")
    root = Path(
        manifest.get(
            "snapshot_root",
            manifest.get("source_corpus", manifest_path.parent),
        )
    ).resolve()
    if root != manifest_path.parent.resolve():
        raise ValueError("observation snapshot must be self-contained")
    records = []
    for item in manifest.get("observations", []):
        relative = Path(str(item.get("path", "")))
        path = (root / relative).resolve()
        if (
            relative.is_absolute()
            or not path.is_relative_to(root)
            or not path.is_file()
        ):
            raise ValueError(f"invalid observation audio path: {relative}")
        if sha256(path) != item.get("audio_sha256"):
            raise ValueError(f"observation audio hash mismatch: {relative}")
        records.append({**item, "resolved_path": path})
    if not records:
        raise ValueError("observation manifest contains no recordings")
    return manifest, records


def summarize(peaks: list[float], cutoff: float) -> dict:
    accepted = sum(peak > cutoff for peak in peaks)
    return {
        "recordings": len(peaks),
        "accepted": accepted,
        "acceptance_rate": accepted / len(peaks) if peaks else 0.0,
        "minimum_peak_probability": min(peaks, default=0.0),
        "median_peak_probability": float(np.median(peaks)) if peaks else 0.0,
        "maximum_peak_probability": max(peaks, default=0.0),
    }


def evaluate_records(
    records: list[dict], cutoff: float, scorer: Callable[[Path], float]
) -> tuple[dict, list[dict]]:
    by_label: dict[str, list[float]] = defaultdict(list)
    scored = []
    for record in records:
        peak = float(scorer(record["resolved_path"]))
        label = str(record.get("weak_label", "unknown"))
        by_label[label].append(peak)
        scored.append(
            {
                "observation_id": record.get("observation_id"),
                "weak_label": label,
                "human_reviewed": bool(record.get("review")),
                "accepted": peak > cutoff,
                "peak_probability": peak,
            }
        )
    return {
        label: summarize(peaks, cutoff) for label, peaks in sorted(by_label.items())
    }, scored


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cutoff-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contract = cutoff_contract(args.cutoff_report, args.model)
    manifest, records = observation_records(args.manifest)
    model = Model(str(args.model), stride=3)

    def scorer(path: Path) -> float:
        return peak_probability(
            model,
            path,
            contract["sliding_window"],
            contract["ignore_initial"],
            0,
        )

    metrics, scored = evaluate_records(records, contract["cutoff"], scorer)
    report = {
        "model": str(args.model),
        "model_sha256": sha256(args.model),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "training_eligible": False,
        "cutoff_source": {
            "path": str(args.cutoff_report),
            **contract,
        },
        "interpretation": {
            "false_wake_no_command": "reviewed negative deployment anchor",
            "speech_unconfirmed": "descriptive score only; not corpus truth",
            "audio_window": "score the complete preserved stream, including pre-roll",
        },
        "metrics_by_weak_label": metrics,
        "observations": scored,
        "source": manifest.get("source"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
