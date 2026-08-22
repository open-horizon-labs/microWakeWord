#!/usr/bin/env python3
"""Compare generated speech with recorded spans and emit a training mask."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import yaml

from microwakeword.device_corpus import captures_for, validate_device_corpus
from microwakeword.synthetic_quality import (
    audio_metrics,
    quality_reasons,
    reference_bounds,
    sha256,
)


def summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    percentiles = np.quantile(values, [0.05, 0.5, 0.95])
    return {
        "count": len(values),
        "minimum": min(values),
        "p05": float(percentiles[0]),
        "median": float(percentiles[1]),
        "p95": float(percentiles[2]),
        "maximum": max(values),
    }


def recorded_positive_spans(corpus: Path, manifest: dict) -> list[float]:
    spans = []
    for split in ("train", "validation", "test"):
        for item, _ in captures_for(corpus, manifest, "positive", split):
            span = item.get("phrase_span")
            if span is not None and item.get("source") == "human":
                spans.append(float(span["end_ms"] - span["start_ms"]))
    return spans


def build_report(
    recipe_path: Path,
    generated: Path,
    corpus: Path,
    maximum_jitter_ms: int,
    minimum_span_ratio: float = 0.75,
    maximum_span_ratio: float = 1.25,
) -> dict:
    recipe = yaml.safe_load(recipe_path.read_text())
    generation_path = generated / "generation-manifest.json"
    generation = json.loads(generation_path.read_text())
    if generation.get("recipe_sha256") != sha256(recipe_path):
        raise ValueError("generation manifest does not match the recipe")
    device_manifest = validate_device_corpus(corpus)
    recorded_spans = recorded_positive_spans(corpus, device_manifest)
    bounds = reference_bounds(
        recorded_spans,
        clip_duration_ms=int(recipe["clip_duration_ms"]),
        maximum_jitter_ms=maximum_jitter_ms,
        minimum_span_ratio=minimum_span_ratio,
        maximum_span_ratio=maximum_span_ratio,
    )

    rejected = {}
    group_metrics: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    group_reasons: dict[str, Counter] = defaultdict(Counter)
    class_metrics: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    group_totals = Counter()
    group_rejected = Counter()
    for item in generation.get("plan", []):
        truth = item["class"]
        group = f'{truth}:{item["text"]}'
        for path in sorted(Path(item["output"]).glob("*.wav")):
            metrics = audio_metrics(path)
            reasons = quality_reasons(metrics, truth, bounds)
            relative = path.resolve().relative_to(generated.resolve()).as_posix()
            group_totals[group] += 1
            for name in ("duration_ms", "speech_span_ms", "rms_dbfs"):
                group_metrics[group][name].append(metrics[name])
                class_metrics[truth][name].append(metrics[name])
            if reasons:
                rejected[relative] = reasons
                group_reasons[group].update(reasons)
                group_rejected[group] += 1

    groups = {}
    for group, total in sorted(group_totals.items()):
        groups[group] = {
            "clips": total,
            "accepted": total - group_rejected[group],
            "rejected_clips": group_rejected[group],
            "reasons": dict(sorted(group_reasons[group].items())),
            "duration_ms": summarize(group_metrics[group]["duration_ms"]),
            "speech_span_ms": summarize(group_metrics[group]["speech_span_ms"]),
            "rms_dbfs": summarize(group_metrics[group]["rms_dbfs"]),
        }
    return {
        "schema_version": 1,
        "recipe_sha256": sha256(recipe_path),
        "generation_manifest_sha256": sha256(generation_path),
        "reference_corpus_id": device_manifest["corpus_id"],
        "reference_corpus_manifest_sha256": sha256(corpus / "device-corpus.json"),
        "reference_positive_spans_ms": summarize(recorded_spans),
        "reference_span_policy": {
            "minimum_quantile": 0.05,
            "maximum_quantile": 0.95,
            "minimum_span_ratio": minimum_span_ratio,
            "maximum_span_ratio": maximum_span_ratio,
            "maximum_jitter_ms": maximum_jitter_ms,
        },
        "synthetic_by_truth": {
            truth: {
                name: summarize(values) for name, values in sorted(measurements.items())
            }
            for truth, measurements in sorted(class_metrics.items())
        },
        "policy": bounds.to_dict(),
        "groups": groups,
        "accepted_clips": sum(group["accepted"] for group in groups.values()),
        "rejected_clips": len(rejected),
        "rejected": dict(sorted(rejected.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--reference-corpus", type=Path, required=True)
    parser.add_argument("--maximum-jitter-ms", type=int, default=300)
    parser.add_argument("--minimum-span-ratio", type=float, default=0.75)
    parser.add_argument("--maximum-span-ratio", type=float, default=1.25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        args.recipe,
        args.generated,
        args.reference_corpus,
        args.maximum_jitter_ms,
        args.minimum_span_ratio,
        args.maximum_span_ratio,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "accepted_clips": report["accepted_clips"],
                "rejected_clips": report["rejected_clips"],
                "reference_positive_spans_ms": report["reference_positive_spans_ms"],
                "synthetic_positive_spans_ms": report["synthetic_by_truth"]
                .get("positive", {})
                .get("speech_span_ms", {"count": 0}),
                "policy": report["policy"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
