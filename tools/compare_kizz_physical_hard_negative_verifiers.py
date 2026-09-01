#!/usr/bin/env python3
"""Compare deployable INT8 verifiers on frozen evaluation and physical failures."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.trace_kizz_candidate_verifier import (
    Int8Verifier,
    _atomic_json,
    _corpus_array_paths,
    _load_object,
    _require_sha,
    sha256_file,
)


def _model(metadata_path: Path) -> tuple[Path, dict[str, Any]]:
    metadata = _load_object(metadata_path, "verifier metadata")
    if (
        metadata.get("schema_version") != 1
        or metadata.get("kind") != "kizz_control_candidate_verifier_fixed_window_int8"
        or metadata.get("candidate_conditioned") is not True
    ):
        raise ValueError("unsupported verifier metadata")
    artifact = metadata.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ValueError("verifier artifact binding is required")
    filename = artifact.get("filename")
    if not isinstance(filename, str) or not filename:
        raise ValueError("verifier artifact filename is required")
    path = metadata_path.parent / filename
    if not path.is_file() or sha256_file(path) != _require_sha(
        artifact.get("sha256"), "verifier artifact"
    ):
        raise ValueError("verifier artifact hash drift")
    return path, metadata


def summarize_scores(
    rows: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    physical_families: Mapping[str, str],
    *,
    threshold: float,
) -> dict[str, Any]:
    if len(rows) != len(scores):
        raise ValueError("row/score count drift")
    buckets: dict[str, list[float]] = {
        "validation_positive": [],
        "validation_negative": [],
        "test_positive": [],
        "test_negative": [],
        "physical_hard_negative": [],
    }
    family_total: Counter[str] = Counter()
    family_accepted: Counter[str] = Counter()
    for row, raw_score in zip(rows, scores):
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError("verifier emitted a non-finite score")
        split = str(row.get("split"))
        label = int(row.get("label", -1))
        if split in {"validation", "test"} and label in {0, 1}:
            buckets[f"{split}_{'positive' if label else 'negative'}"].append(score)
        capture_id = str(row.get("capture_id", ""))
        if capture_id.startswith("hardneg-"):
            if split != "train" or label != 0:
                raise ValueError("physical hard negative contract drift")
            family = physical_families.get(capture_id)
            if family is None:
                raise ValueError(f"physical capture is unbound: {capture_id}")
            buckets["physical_hard_negative"].append(score)
            family_total[family] += 1
            if score >= threshold:
                family_accepted[family] += 1

    def metrics(values: list[float], *, positive: bool) -> dict[str, Any]:
        accepted = sum(value >= threshold for value in values)
        return {
            "count": len(values),
            "accepted": accepted,
            "rejected": len(values) - accepted,
            "recall": accepted / len(values) if positive and values else None,
            "minimum": min(values) if values else None,
            "maximum": max(values) if values else None,
        }

    result = {
        key: metrics(values, positive=key.endswith("positive"))
        for key, values in buckets.items()
    }
    result["physical_by_family"] = {
        family: {
            "count": family_total[family],
            "accepted": family_accepted[family],
            "rejected": family_total[family] - family_accepted[family],
        }
        for family in sorted(family_total)
    }
    return result


def compare(
    candidate_corpus: Path,
    device_corpus: Path,
    verifier_specs: Sequence[tuple[str, Path]],
    *,
    threshold: float,
    scorer_factory: Callable[[Path], Callable[[np.ndarray], float]] = Int8Verifier,
) -> dict[str, Any]:
    candidate_corpus = candidate_corpus.expanduser().resolve()
    device_corpus = device_corpus.expanduser().resolve()
    corpus = _load_object(candidate_corpus, "candidate corpus")
    rows = corpus.get("examples")
    if not isinstance(rows, list):
        raise ValueError("candidate examples are required")
    feature_path = _corpus_array_paths(candidate_corpus, corpus)["features.npy"]
    features = np.load(feature_path, mmap_mode="r")
    if features.shape != (len(rows), 260, 40):
        raise ValueError("candidate feature array shape drift")

    device = _load_object(device_corpus, "device corpus")
    captures = device.get("captures")
    if not isinstance(captures, list):
        raise ValueError("device captures are required")
    physical_families: dict[str, str] = {}
    for capture in captures:
        capture_id = str(capture.get("capture_id", ""))
        conditions = capture.get("conditions")
        family = (
            conditions.get("physical_failure_family")
            if isinstance(conditions, Mapping)
            else None
        )
        if not capture_id or not isinstance(family, str) or not family:
            raise ValueError("device capture family binding is incomplete")
        physical_families[capture_id] = family

    hard_ids = {
        str(row.get("capture_id"))
        for row in rows
        if str(row.get("capture_id", "")).startswith("hardneg-")
    }
    if not hard_ids or not hard_ids.issubset(physical_families):
        raise ValueError("candidate physical hard-negative binding drift")

    models: dict[str, Any] = {}
    for name, raw_metadata_path in verifier_specs:
        if name in models:
            raise ValueError(f"duplicate verifier name: {name}")
        metadata_path = raw_metadata_path.expanduser().resolve()
        model_path, _ = _model(metadata_path)
        scorer = scorer_factory(model_path)
        scores = [float(scorer(features[index])) for index in range(len(rows))]
        models[name] = {
            "metadata": {
                "path": str(metadata_path),
                "sha256": sha256_file(metadata_path),
            },
            "artifact": {"path": str(model_path), "sha256": sha256_file(model_path)},
            "metrics": summarize_scores(
                rows, scores, physical_families, threshold=threshold
            ),
        }
    return {
        "schema_version": 1,
        "kind": "kizz_control_physical_hard_negative_verifier_comparison",
        "deployment_qualification": False,
        "threshold": threshold,
        "candidate_corpus": {
            "path": str(candidate_corpus),
            "sha256": sha256_file(candidate_corpus),
        },
        "candidate_features": {
            "path": str(feature_path),
            "sha256": sha256_file(feature_path),
        },
        "device_corpus": {
            "path": str(device_corpus),
            "sha256": sha256_file(device_corpus),
        },
        "models": models,
    }


def _verifier_spec(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("verifier must be NAME=METADATA_PATH")
    return name, Path(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-corpus", type=Path, required=True)
    parser.add_argument("--device-corpus", type=Path, required=True)
    parser.add_argument(
        "--verifier", action="append", type=_verifier_spec, required=True
    )
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = compare(
        args.candidate_corpus,
        args.device_corpus,
        args.verifier,
        threshold=args.threshold,
    )
    _atomic_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
