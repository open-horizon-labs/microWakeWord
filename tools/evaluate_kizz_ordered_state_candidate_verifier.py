#!/usr/bin/env python3
"""Evaluate a detector-conditioned ordered-state verifier without test leakage."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from microwakeword.ordered_state import OrderedStateTopology
from microwakeword.ordered_state_model import model as build_student
try:
    from tools.distill_kizz_student import student_flags
    from tools.evaluate_kizz_ordered_state_detector import (
        _metrics,
        _score,
        _select_threshold,
        _validate_distillation,
        _write_json_atomic,
        sha256_file,
    )
except ModuleNotFoundError:  # Direct ``python tools/...py`` invocation.
    from distill_kizz_student import student_flags
    from evaluate_kizz_ordered_state_detector import (
        _metrics,
        _score,
        _select_threshold,
        _validate_distillation,
        _write_json_atomic,
        sha256_file,
    )


EVALUATION_KIND = "kizz_control_float_ordered_state_candidate_verifier"
SPECIALIZATIONS = {
    "detector_conditioned_ordered_state_verifier_train_only_v1",
    "detector_conditioned_ordered_state_verifier_train_only_v2",
}


def _binding(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path)}


def _load_candidate_rows(
    corpus_path: Path, features_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    corpus_path = corpus_path.expanduser().resolve()
    features_path = features_path.expanduser().resolve()
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    rows = corpus.get("examples")
    if not isinstance(rows, list) or not rows:
        raise ValueError("candidate corpus examples are missing")
    features = np.load(features_path, mmap_mode="r", allow_pickle=False)
    if features.ndim != 3 or tuple(features.shape[1:]) != (260, 40):
        raise ValueError("candidate features must have shape [N,260,40]")
    if len(rows) != len(features):
        raise ValueError("candidate corpus and feature counts differ")
    declared = corpus.get("array_sha256", {}).get(features_path.name)
    if declared != sha256_file(features_path):
        raise ValueError("candidate feature-array hash drift")
    heldout: list[dict[str, Any]] = []
    indexes: list[int] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ValueError("candidate row is not an object")
        split = raw.get("split")
        label = raw.get("label")
        if split not in {"train", "validation", "test"} or label not in (0, 1, False, True):
            raise ValueError("candidate split/label identity is invalid")
        if split in {"validation", "test"}:
            heldout.append(dict(raw))
            indexes.append(index)
    if not heldout or {row["split"] for row in heldout} != {"validation", "test"}:
        raise ValueError("validation and test candidates are both required")
    return corpus, heldout, np.asarray(features[indexes], dtype=np.float32)


def _validate_training_specialization(
    metadata: Mapping[str, Any], candidate_corpus: Path
) -> dict[str, Any]:
    cache_prefix = metadata.get("cache_prefix")
    if not isinstance(cache_prefix, str):
        raise ValueError("distillation metadata has no cache prefix")
    cache_path = Path(cache_prefix).expanduser().resolve().with_suffix(".json")
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    if cache.get("cache_specialization") not in SPECIALIZATIONS:
        raise ValueError("distillation cache is not a candidate verifier specialization")
    split_policy = cache.get("split_policy", {})
    if (
        split_policy.get("included") != ["train"]
        or split_policy.get("excluded") != ["validation", "test"]
        or split_policy.get("test_used_for_training") is not False
    ):
        raise ValueError("candidate verifier cache is not train-only")
    source = cache.get("source_candidate_corpus", {})
    if (
        Path(str(source.get("path", ""))).expanduser().resolve()
        != candidate_corpus.expanduser().resolve()
        or source.get("sha256") != sha256_file(candidate_corpus)
    ):
        raise ValueError("candidate verifier cache binds a different corpus")
    return cache


def evaluate_candidate_verifier(
    distillation_metadata: Path,
    weights: Path,
    candidate_corpus: Path,
    candidate_features: Path,
    *,
    minimum_recall: float = 1.0,
    maximum_false_candidate_fraction: float = 1.0,
    batch_size: int = 128,
    model_factory: Callable[[OrderedStateTopology], Any] | None = None,
) -> dict[str, Any]:
    if minimum_recall != 1.0 or not 0 <= maximum_false_candidate_fraction <= 1:
        raise ValueError("candidate verifier conversion requires a 100% recall floor")
    metadata, topology = _validate_distillation(
        distillation_metadata.expanduser().resolve(), weights.expanduser().resolve()
    )
    _validate_training_specialization(metadata, candidate_corpus)
    _, rows, features = _load_candidate_rows(candidate_corpus, candidate_features)
    model = (
        model_factory(topology)
        if model_factory is not None
        else build_student(
            student_flags(
                topology.state_count,
                metadata.get("student_architecture", "control_mixconv"),
            ),
            (260, 40),
            None,
        )
    )
    if hasattr(model, "load_weights"):
        model.load_weights(weights)
    scores = _score(model, features, topology, batch_size)
    splits = np.asarray([row["split"] for row in rows])
    labels = np.asarray([int(row["label"]) for row in rows])
    validation = splits == "validation"
    test = splits == "test"
    selection = _select_threshold(
        scores[validation & (labels == 1)],
        scores[validation & (labels == 0)],
        minimum_recall,
    )
    test_metrics = _metrics(
        scores[test & (labels == 1)],
        scores[test & (labels == 0)],
        float(selection["threshold"]),
    )
    reasons: list[str] = []
    if selection["opportunity_recall"] < minimum_recall:
        reasons.append("validation_recall_below_floor")
    if test_metrics["opportunity_recall"] < minimum_recall:
        reasons.append("test_recall_below_floor")
    if selection["false_candidate_fraction"] > maximum_false_candidate_fraction:
        reasons.append("validation_false_candidate_fraction_above_limit")
    if test_metrics["false_candidate_fraction"] > maximum_false_candidate_fraction:
        reasons.append("test_false_candidate_fraction_above_limit")
    return {
        "schema_version": 1,
        "evaluation": EVALUATION_KIND,
        "qualified_for_candidate_verifier_conversion": not reasons,
        "deployment_qualification": False,
        "failure_reasons": reasons,
        "model": {
            "distillation_metadata": str(distillation_metadata.expanduser().resolve()),
            "distillation_metadata_sha256": sha256_file(distillation_metadata),
            "weights": str(weights.expanduser().resolve()),
            "weights_sha256": sha256_file(weights),
        },
        "candidate_corpus": _binding(candidate_corpus),
        "candidate_features": _binding(candidate_features),
        "topology": metadata["topology"],
        "threshold_selection": {
            "fit_split": "validation",
            "test_used_for_selection": False,
            "minimum_recall": minimum_recall,
            "maximum_false_candidate_fraction": maximum_false_candidate_fraction,
            **selection,
        },
        "test": test_metrics,
        "counts": {
            "validation": int(np.sum(validation)),
            "test": int(np.sum(test)),
        },
        "limitations": [
            "candidate-conditioned fixed-window evidence only",
            "not a standalone continuous-audio detector qualification",
            "integer cascade and StackChan performance remain unmeasured",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distillation-metadata", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--candidate-corpus", type=Path, required=True)
    parser.add_argument("--candidate-features", type=Path, required=True)
    parser.add_argument("--minimum-recall", type=float, default=1.0)
    parser.add_argument("--maximum-false-candidate-fraction", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = evaluate_candidate_verifier(
            args.distillation_metadata,
            args.weights,
            args.candidate_corpus,
            args.candidate_features,
            minimum_recall=args.minimum_recall,
            maximum_false_candidate_fraction=args.maximum_false_candidate_fraction,
            batch_size=args.batch_size,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        parser.error(str(error))
    _write_json_atomic(args.output, report)
    print(json.dumps({
        "qualified": report["qualified_for_candidate_verifier_conversion"],
        "validation": report["threshold_selection"],
        "test": report["test"],
    }, indent=2, sort_keys=True))
    return 0 if report["qualified_for_candidate_verifier_conversion"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
