#!/usr/bin/env python3
"""Bind a high-recall synthetic teacher checkpoint for detector distillation.

This is deliberately not a deployment qualification.  It proves only that a
detector-oriented checkpoint was selected from validation at a declared recall
floor and that every referenced training artifact still has its recorded hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selection_key(item: Mapping[str, Any], recall_floor: float) -> tuple:
    return (
        float(item["opportunity_recall"]) >= recall_floor,
        -int(item["false_accepts"]),
        float(item["opportunity_recall"]),
        float(item["separation"]),
        -float(item["validation_loss"]),
        float(item["threshold"]),
    )


def _verified_reference(report: Mapping[str, Any], name: str) -> dict[str, Any]:
    raw_path = report.get(name)
    expected = report.get(f"{name}_sha256")
    if not isinstance(raw_path, str) or not isinstance(expected, str):
        raise ValueError(f"teacher training report does not bind {name}")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"teacher training {name} hash drift")
    return {"path": str(path), "sha256": expected}


def bind_detector_teacher(
    training_report_path: Path,
    *,
    minimum_recall: float = 0.95,
    maximum_false_candidate_fraction: float = 0.20,
) -> dict[str, Any]:
    if not 0 < minimum_recall <= 1:
        raise ValueError("minimum_recall must be in (0, 1]")
    if not 0 <= maximum_false_candidate_fraction <= 1:
        raise ValueError("maximum_false_candidate_fraction must be in [0, 1]")
    training_report_path = training_report_path.expanduser().resolve()
    report = json.loads(training_report_path.read_text(encoding="utf-8"))
    if (
        report.get("schema_version") != 1
        or report.get("checkpoint_selection")
        != "validation_min_false_accepts_subject_to_recall_floor"
    ):
        raise ValueError("teacher was not trained with detector checkpoint selection")
    selection_floor = report.get("selection_min_recall")
    if not isinstance(selection_floor, (int, float)) or float(selection_floor) < minimum_recall:
        raise ValueError("teacher selection recall floor is too permissive")
    ledger = report.get("checkpoint_selection_ledger")
    if not isinstance(ledger, list) or not ledger:
        raise ValueError("teacher training report has no checkpoint selection ledger")
    for entry in ledger:
        if not isinstance(entry, dict) or not isinstance(entry.get("selected"), dict):
            raise ValueError("teacher checkpoint selection ledger is malformed")
    best = max(
        ledger,
        key=lambda item: _selection_key(item["selected"], float(selection_floor)),
    )
    if best["selected"] != report.get("best_validation"):
        raise ValueError("teacher best validation does not match its selection ledger")
    step = best.get("step")
    if not isinstance(step, int) or step < 1:
        raise ValueError("selected teacher checkpoint step is invalid")
    checkpoint = training_report_path.parent / f"checkpoint-{step:06d}.weights.h5"
    best_weights = training_report_path.parent / "best.weights.h5"
    if not checkpoint.is_file() or not best_weights.is_file():
        raise ValueError("selected teacher checkpoint files are missing")
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != sha256_file(best_weights):
        raise ValueError("best teacher weights differ from the selected checkpoint")
    positive_count = best.get("positive_count")
    negative_count = best.get("negative_count")
    selected = best["selected"]
    if not isinstance(positive_count, int) or positive_count < 1:
        raise ValueError("selected checkpoint has no validation positives")
    if not isinstance(negative_count, int) or negative_count < 1:
        raise ValueError("selected checkpoint has no validation negatives")
    recall = float(selected.get("opportunity_recall", -1))
    false_accepts = int(selected.get("false_accepts", -1))
    false_candidate_fraction = false_accepts / negative_count
    failure_reasons = []
    if recall < minimum_recall:
        failure_reasons.append("validation_detector_recall_below_floor")
    if false_candidate_fraction > maximum_false_candidate_fraction:
        failure_reasons.append("validation_false_candidate_fraction_above_limit")
    bindings = {
        name: _verified_reference(report, name)
        for name in (
            "feature_provenance",
            "balance_manifest",
            "balance_report",
            "batch_mixture_recipe",
            "batch_mixture_ledger",
            "positive_source_balance_report",
            "positive_features",
            "positive_targets",
        )
    }
    return {
        "schema_version": 1,
        "gate_scope": "teacher_detector_synthetic_bootstrap_prequalification",
        "qualified": not failure_reasons,
        "failure_reasons": failure_reasons,
        "eligible_for_detector_distillation": not failure_reasons,
        "deployment_qualification": False,
        "eligible_for_final_deployment": False,
        "training_report": {
            "path": str(training_report_path),
            "sha256": sha256_file(training_report_path),
        },
        "selected_checkpoint": {
            "step": step,
            "path": str(checkpoint.resolve()),
            "sha256": checkpoint_hash,
            "best_weights_path": str(best_weights.resolve()),
            "best_weights_sha256": checkpoint_hash,
        },
        "selection": {
            "split": "validation",
            "objective": "minimum_false_accepts_subject_to_recall_floor",
            "minimum_recall": minimum_recall,
            "training_selection_recall_floor": float(selection_floor),
            "maximum_false_candidate_fraction": maximum_false_candidate_fraction,
            "positive_opportunities": positive_count,
            "negative_windows": negative_count,
            "opportunity_recall": recall,
            "false_candidates": false_accepts,
            "false_candidate_fraction": false_candidate_fraction,
            "threshold": float(selected["threshold"]),
        },
        "topology": report.get("topology"),
        "bindings": bindings,
        "limitations": [
            "synthetic provider positives only",
            "window-level candidate pressure is not continuous-audio FAPH",
            "no StackChan target-channel qualification",
            "joint detector-verifier and hardware gates remain mandatory",
        ],
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
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
    parser.add_argument("--teacher-training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-recall", type=float, default=0.95)
    parser.add_argument("--maximum-false-candidate-fraction", type=float, default=0.20)
    args = parser.parse_args(argv)
    result = bind_detector_teacher(
        args.teacher_training,
        minimum_recall=args.minimum_recall,
        maximum_false_candidate_fraction=args.maximum_false_candidate_fraction,
    )
    _write_json_atomic(args.output, result)
    print(json.dumps(result["selection"], sort_keys=True))
    return 0 if result["qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
