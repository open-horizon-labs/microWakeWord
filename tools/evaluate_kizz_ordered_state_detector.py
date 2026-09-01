#!/usr/bin/env python3
"""Evaluate a float ordered-state detector at a validation-frozen recall floor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np

from microwakeword.ordered_state import (
    OrderedStateTopology,
    ordered_state_sequence_score_numpy,
)
from microwakeword.ordered_state_model import model as build_student
if __package__:
    from tools.distill_kizz_student import student_flags
else:
    from distill_kizz_student import student_flags


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("negative source must be GROUP=PATH")
    group, raw_path = value.split("=", 1)
    path = Path(raw_path).expanduser().resolve()
    if not group or not path.is_file() or path.suffix != ".npy":
        raise argparse.ArgumentTypeError("negative source must be GROUP=existing.npy")
    return group, path


def _load_features(path: Path) -> np.ndarray:
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.ndim != 3 or tuple(values.shape[1:]) != (260, 40):
        raise ValueError(f"{path}: expected [N,260,40] features")
    if len(values) < 1 or not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"{path}: feature array must be nonempty numeric data")
    return values


def _verify_file_binding(binding: Mapping[str, Any], label: str) -> Path:
    raw_path = binding.get("path")
    expected = binding.get("sha256")
    if not isinstance(raw_path, str) or not isinstance(expected, str):
        raise ValueError(f"{label} binding is incomplete")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"{label} hash drift")
    return path


def _validate_distillation(path: Path, weights: Path) -> tuple[dict, OrderedStateTopology]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if (
        metadata.get("schema_version") != 1
        or metadata.get("student_role") != "permissive_detector_candidate_generator"
        or metadata.get("deployment_qualification") is not False
    ):
        raise ValueError("distillation metadata is not a detector-role student")
    gate_path = metadata.get("detector_teacher_gate")
    gate_hash = metadata.get("detector_teacher_gate_sha256")
    if not isinstance(gate_path, str) or not isinstance(gate_hash, str):
        raise ValueError("distillation metadata does not bind its detector gate")
    gate = Path(gate_path).expanduser().resolve()
    if not gate.is_file() or sha256_file(gate) != gate_hash:
        raise ValueError("distillation detector gate hash drift")
    for name, expected in metadata.get("cache_files_sha256", {}).items():
        cache_file = Path(metadata["cache_prefix"]).resolve().with_name(name)
        if not cache_file.is_file() or sha256_file(cache_file) != expected:
            raise ValueError(f"distillation cache {name} hash drift")
    if not weights.is_file():
        raise FileNotFoundError(weights)
    topology_payload = metadata.get("topology", {})
    topology = OrderedStateTopology(
        tuple(str(phone) for phone in topology_payload.get("phones", ())),
        int(topology_payload.get("states_per_phone", 0)),
    )
    if topology_payload.get("state_count") != topology.state_count:
        raise ValueError("distillation topology state count drift")
    return metadata, topology


def _validate_feature_provenance(
    path: Path,
    *,
    validation_positive_count: int,
    test_positive_count: int,
    validation_negative_counts: Mapping[str, int],
    test_negative_counts: Mapping[str, int],
    topology: OrderedStateTopology,
) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("schema_version") != 3
        or report.get("recipe") != "kizz_aligned_teacher_features_v3"
        or report.get("state_count") != topology.state_count
        or report.get("states_per_phone") != topology.states_per_phone
    ):
        raise ValueError("feature provenance differs from detector topology")
    expected_positive = report.get("positive_counts", {})
    if (
        expected_positive.get("validation") != validation_positive_count
        or expected_positive.get("test") != test_positive_count
    ):
        raise ValueError("positive evaluation counts differ from feature provenance")
    expected_negative = report.get("negative_counts", {})
    if expected_negative.get("validation") != dict(validation_negative_counts):
        raise ValueError("validation negative counts differ from feature provenance")
    if expected_negative.get("test") != dict(test_negative_counts):
        raise ValueError("test negative counts differ from feature provenance")
    for row in report.get("examples", []):
        if row.get("split") in {"validation", "test"} and (
            row.get("variant") != "clean" or row.get("augmentation") is not None
        ):
            raise ValueError("evaluation positives must remain clean")
    return report


def _score(model: Any, values: np.ndarray, topology: OrderedStateTopology, batch_size: int) -> np.ndarray:
    scores = []
    for start in range(0, len(values), batch_size):
        logits = np.asarray(
            model.predict(
                np.asarray(values[start : start + batch_size], dtype=np.float32),
                verbose=0,
            )
        )
        if logits.ndim != 3 or logits.shape[-1] != topology.state_count:
            raise ValueError("student logits differ from ordered-state topology")
        scores.append(ordered_state_sequence_score_numpy(logits, topology))
    return np.concatenate(scores)


def _select_threshold(
    positives: np.ndarray, negatives: np.ndarray, minimum_recall: float
) -> dict[str, Any]:
    thresholds = np.unique(np.concatenate([positives, negatives]))
    points = []
    for threshold in thresholds:
        detected = int(np.sum(positives >= threshold))
        false_candidates = int(np.sum(negatives >= threshold))
        recall = detected / len(positives)
        points.append(
            {
                "threshold": float(threshold),
                "positive_opportunities": len(positives),
                "detected_opportunities": detected,
                "opportunity_recall": recall,
                "negative_windows": len(negatives),
                "false_candidates": false_candidates,
                "false_candidate_fraction": false_candidates / len(negatives),
            }
        )
    selected = max(
        points,
        key=lambda point: (
            point["opportunity_recall"] >= minimum_recall,
            -point["false_candidates"],
            point["opportunity_recall"],
            point["threshold"],
        ),
    )
    return {**selected, "minimum_recall": minimum_recall}


def _metrics(positives: np.ndarray, negatives: np.ndarray, threshold: float) -> dict[str, Any]:
    true_accepts = int(np.sum(positives >= threshold))
    false_candidates = int(np.sum(negatives >= threshold))
    accepted = true_accepts + false_candidates
    return {
        "positive_opportunities": len(positives),
        "detected_opportunities": true_accepts,
        "opportunity_recall": true_accepts / len(positives),
        "negative_windows": len(negatives),
        "false_candidates": false_candidates,
        "false_candidate_fraction": false_candidates / len(negatives),
        "window_set_precision": true_accepts / accepted if accepted else 0.0,
    }


def _qualification_reasons(
    validation: Mapping[str, Any],
    test: Mapping[str, Any],
    *,
    minimum_recall: float,
    maximum_false_candidate_fraction: float,
) -> list[str]:
    reasons = []
    if (
        validation["opportunity_recall"] < minimum_recall
        or test["opportunity_recall"] < minimum_recall
    ):
        reasons.append("detector_recall_below_floor")
    if (
        validation["false_candidate_fraction"] > maximum_false_candidate_fraction
        or test["false_candidate_fraction"] > maximum_false_candidate_fraction
    ):
        reasons.append("detector_false_candidate_fraction_above_limit")
    return reasons


def evaluate_detector(
    distillation_metadata: Path,
    weights: Path,
    feature_provenance: Path,
    validation_positive: Path,
    validation_negative: Sequence[tuple[str, Path]],
    test_positive: Path,
    test_negative: Sequence[tuple[str, Path]],
    *,
    minimum_recall: float = 0.95,
    threshold_selection_recall: float | None = None,
    maximum_false_candidate_fraction: float = 0.20,
    batch_size: int = 128,
    model_factory: Callable[[OrderedStateTopology], Any] | None = None,
) -> dict[str, Any]:
    if (
        not 0 < minimum_recall <= 1
        or (
            threshold_selection_recall is not None
            and not 0 < threshold_selection_recall <= 1
        )
        or not 0 <= maximum_false_candidate_fraction <= 1
        or batch_size < 1
    ):
        raise ValueError("invalid evaluation recall or batch size")
    distillation_metadata = distillation_metadata.resolve()
    weights = weights.resolve()
    metadata, topology = _validate_distillation(distillation_metadata, weights)
    validation_positive_values = _load_features(validation_positive)
    test_positive_values = _load_features(test_positive)
    validation_negative_values = {
        group: _load_features(path) for group, path in validation_negative
    }
    test_negative_values = {group: _load_features(path) for group, path in test_negative}
    if len(validation_negative_values) != len(validation_negative):
        raise ValueError("validation negative groups must be unique")
    if len(test_negative_values) != len(test_negative):
        raise ValueError("test negative groups must be unique")
    _validate_feature_provenance(
        feature_provenance,
        validation_positive_count=len(validation_positive_values),
        test_positive_count=len(test_positive_values),
        validation_negative_counts={k: len(v) for k, v in validation_negative_values.items()},
        test_negative_counts={k: len(v) for k, v in test_negative_values.items()},
        topology=topology,
    )
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
    validation_positive_scores = _score(
        model, validation_positive_values, topology, batch_size
    )
    validation_negative_scores = np.concatenate(
        [_score(model, values, topology, batch_size) for values in validation_negative_values.values()]
    )
    selection_recall = (
        minimum_recall
        if threshold_selection_recall is None
        else threshold_selection_recall
    )
    selection = _select_threshold(
        validation_positive_scores, validation_negative_scores, selection_recall
    )
    test_positive_scores = _score(model, test_positive_values, topology, batch_size)
    test_negative_scores = np.concatenate(
        [_score(model, values, topology, batch_size) for values in test_negative_values.values()]
    )
    test = _metrics(test_positive_scores, test_negative_scores, selection["threshold"])
    failure_reasons = _qualification_reasons(
        selection,
        test,
        minimum_recall=minimum_recall,
        maximum_false_candidate_fraction=maximum_false_candidate_fraction,
    )
    return {
        "schema_version": 1,
        "evaluation": "kizz_control_float_ordered_state_detector",
        "qualified_for_detector_conversion": not failure_reasons,
        "deployment_qualification": False,
        "failure_reasons": failure_reasons,
        "model": {
            "distillation_metadata": str(distillation_metadata),
            "distillation_metadata_sha256": sha256_file(distillation_metadata),
            "weights": str(weights),
            "weights_sha256": sha256_file(weights),
        },
        "feature_provenance": {
            "path": str(feature_provenance.resolve()),
            "sha256": sha256_file(feature_provenance),
        },
        "topology": metadata["topology"],
        "threshold_selection": {
            "fit_split": "validation",
            "test_used_for_selection": False,
            "qualification_minimum_recall": minimum_recall,
            "maximum_false_candidate_fraction": maximum_false_candidate_fraction,
            **selection,
        },
        "test": test,
        "inputs": {
            "validation_positive": {"path": str(validation_positive.resolve()), "sha256": sha256_file(validation_positive)},
            "validation_negative": {group: {"path": str(path.resolve()), "sha256": sha256_file(path)} for group, path in validation_negative},
            "test_positive": {"path": str(test_positive.resolve()), "sha256": sha256_file(test_positive)},
            "test_negative": {group: {"path": str(path.resolve()), "sha256": sha256_file(path)} for group, path in test_negative},
        },
        "limitations": [
            "fixed 2.6-second feature windows are not continuous-audio FAPH",
            "threshold remains provisional until causal INT8 and cascade evaluation",
            "no StackChan target-channel evidence",
        ],
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distillation-metadata", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--feature-provenance", type=Path, required=True)
    parser.add_argument("--validation-positive", type=Path, required=True)
    parser.add_argument("--validation-negative", type=parse_source, action="append", required=True)
    parser.add_argument("--test-positive", type=Path, required=True)
    parser.add_argument("--test-negative", type=parse_source, action="append", required=True)
    parser.add_argument("--minimum-recall", type=float, default=0.95)
    parser.add_argument(
        "--threshold-selection-recall",
        type=float,
        help=(
            "Validation recall floor used to choose the detector threshold; "
            "defaults to --minimum-recall."
        ),
    )
    parser.add_argument(
        "--maximum-false-candidate-fraction", type=float, default=0.20
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = evaluate_detector(
            args.distillation_metadata,
            args.weights,
            args.feature_provenance,
            args.validation_positive,
            args.validation_negative,
            args.test_positive,
            args.test_negative,
            minimum_recall=args.minimum_recall,
            threshold_selection_recall=args.threshold_selection_recall,
            maximum_false_candidate_fraction=args.maximum_false_candidate_fraction,
            batch_size=args.batch_size,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        parser.error(str(error))
    _write_json_atomic(args.output, report)
    print(json.dumps({"qualified": report["qualified_for_detector_conversion"], "validation": report["threshold_selection"], "test": report["test"]}, indent=2, sort_keys=True))
    return 0 if report["qualified_for_detector_conversion"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
