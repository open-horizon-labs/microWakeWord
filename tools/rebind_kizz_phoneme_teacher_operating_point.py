#!/usr/bin/env python3
"""Bind an adapted teacher's precommitted threshold to stored qualification scores.

This operation is deliberately monotonic: it may only make an already-scored
detector stricter.  It never reruns inference, selects a threshold from the
continuous corpus, or changes model weights.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from microwakeword.kizz_continuous_evaluation import poisson_upper_95
from microwakeword.kizz_phoneme_teacher import sha256_file


def _selected_detector(adaptation: Mapping[str, Any]) -> tuple[int, Mapping[str, Any]]:
    selection = adaptation.get("checkpoint_selection", {})
    step = int(selection.get("selected_step", -1))
    entries = [
        row
        for row in adaptation.get("validation_ledger", [])
        if int(row.get("step", -2)) == step
    ]
    if len(entries) != 1:
        raise ValueError("adaptation report does not contain one selected validation ledger entry")
    detector = entries[0].get("detector_selection", {})
    metrics = detector.get("metrics", {})
    threshold = metrics.get("threshold")
    if threshold is None or not math.isfinite(float(threshold)):
        raise ValueError("selected adaptation checkpoint has no finite detector threshold")
    if metrics.get("qualified_clean_operating_point") is not True:
        raise ValueError("selected adaptation checkpoint did not qualify")
    return step, detector


def _validate_model_lineage(
    clip: Mapping[str, Any], adaptation: Mapping[str, Any], detector: Mapping[str, Any]
) -> None:
    if adaptation.get("kind") != "kizz_phoneme_teacher_adaptation":
        raise ValueError("unexpected adaptation report kind")
    if adaptation.get("wake_phrase", {}).get("phrase_id") != "kizz-control":
        raise ValueError("adaptation report is for the wrong phrase")
    clip_model = clip.get("model", {})
    selected = adaptation.get("checkpoints", {}).get("best", {})
    detector_checkpoint = detector.get("checkpoint", {})
    expected = clip_model.get("weights_sha256")
    if not expected or selected.get("file_sha256") != expected:
        raise ValueError("adaptation best checkpoint differs from qualified teacher weights")
    if detector_checkpoint.get("file_sha256") != expected:
        raise ValueError("selected validation ledger checkpoint differs from teacher weights")


def _accepted(item: Mapping[str, Any], *, threshold: float, beta: float) -> bool:
    score = item.get("score")
    return bool(
        score is not None
        and math.isfinite(float(score))
        and float(item.get("collision_margin", -math.inf)) >= beta
        and float(score) >= threshold
    )


def _retag(items: Sequence[dict[str, Any]], *, threshold: float, beta: float) -> None:
    for item in items:
        reasons = [
            reason
            for reason in item.get("failure_reasons", [])
            if reason not in ("below_validation_threshold", "no_qualifying_validation_threshold")
        ]
        item["accepted"] = _accepted(item, threshold=threshold, beta=beta)
        if item.get("score") is not None and not item["accepted"]:
            reasons.append("below_bound_adaptation_threshold")
        item["failure_reasons"] = reasons


def rebind_clip_report(
    source_clip: Path,
    adaptation_report: Path,
    *,
    min_natural_recall: float,
) -> dict[str, Any]:
    clip = json.loads(source_clip.read_text())
    adaptation = json.loads(adaptation_report.read_text())
    if clip.get("gate_scope") != "teacher_clip_and_anchor_prequalification":
        raise ValueError("unexpected teacher clip gate scope")
    step, detector = _selected_detector(adaptation)
    _validate_model_lineage(clip, adaptation, detector)
    source_threshold = float(clip.get("scoring", {}).get("threshold"))
    threshold = float(detector["metrics"]["threshold"])
    if threshold < source_threshold:
        raise ValueError("operating-point rebinding may only tighten the threshold")
    if not 0 < min_natural_recall <= 1:
        raise ValueError("min_natural_recall must be in (0, 1]")

    report = copy.deepcopy(clip)
    beta = float(report.get("scoring", {}).get("collision_margin_beta", 0.0))
    results = report["results"]
    for name in ("aligned", "validation_negative", "natural_positive", "false_wake_anchors"):
        _retag(results[name], threshold=threshold, beta=beta)

    validation = [
        item for item in results["aligned"]
        if item.get("split") == "validation" and int(item.get("label", -1)) == 1
    ]
    test = [
        item for item in results["aligned"]
        if item.get("split") == "test" and int(item.get("label", -1)) == 1
    ]
    negatives = results["validation_negative"]
    natural = results["natural_positive"]
    false_wakes = results["false_wake_anchors"]
    validation_accepts = sum(bool(item["accepted"]) for item in validation)
    test_accepts = sum(bool(item["accepted"]) for item in test)
    negative_accepts = sum(bool(item["accepted"]) for item in negatives)
    natural_accepts = sum(bool(item["accepted"]) for item in natural)
    false_accepts = sum(bool(item["accepted"]) for item in false_wakes)
    exposure_seconds = float(report["counts"]["validation_negative_exposure_seconds"])
    exposure_hours = max(exposure_seconds / 3600.0, 1e-12)
    min_recall = float(report["limits"]["min_recall"])
    max_faph = float(report["limits"]["max_faph"])
    validation_recall = validation_accepts / max(1, len(validation))
    test_recall = test_accepts / max(1, len(test))
    natural_recall = natural_accepts / max(1, len(natural))
    faph = negative_accepts / exposure_hours

    reasons: list[str] = []
    if validation_recall < min_recall or faph > max_faph:
        reasons.append("bound_validation_operating_point_not_qualified")
    if test_recall < min_recall:
        reasons.append("aligned_test_recall_below_minimum")
    if natural_recall < min_natural_recall:
        reasons.append("natural_positive_recall_below_minimum")
    if len(natural) < int(report["limits"].get("minimum_natural_positives", 0)):
        reasons.append("insufficient_natural_positive_evidence")
    if len(false_wakes) != 62:
        reasons.append("false_wake_anchor_count_not_62")
    if false_accepts:
        reasons.append("quarantined_false_wake_accepted")
    if any(
        not item.get("wake_context", {}).get("best_window_is_pre_wake", False)
        for item in false_wakes
    ):
        reasons.append("false_wake_trigger_context_not_proven")

    report["schema_version"] = max(2, int(report.get("schema_version", 1)))
    report["qualified"] = not reasons
    report["failure_reasons"] = reasons
    report["limits"]["min_natural_recall"] = min_natural_recall
    report["scoring"]["threshold"] = threshold
    report["scoring"]["threshold_selection"] = "adaptation_checkpoint_validation_only"
    report["validation_operating_point"] = {
        "qualified": validation_recall >= min_recall and faph <= max_faph,
        "threshold": threshold,
        "recall": validation_recall,
        "false_accepts": negative_accepts,
        "faph": faph,
        "source": "adaptation selected-checkpoint validation ledger",
    }
    report["counts"].update(
        aligned_validation_accepted=validation_accepts,
        aligned_test_accepted=test_accepts,
        natural_positive_accepted=natural_accepts,
        false_wake_accepted=false_accepts,
    )
    report["operating_point_rebinding"] = {
        "algorithm": "monotonic_adaptation_validation_threshold_v1",
        "source_teacher_qualification": {
            "path": str(source_clip.resolve()),
            "sha256": sha256_file(source_clip),
        },
        "adaptation_report": {
            "path": str(adaptation_report.resolve()),
            "sha256": sha256_file(adaptation_report),
            "selected_step": step,
        },
        "source_threshold": source_threshold,
        "bound_threshold": threshold,
        "monotonic_tightening": True,
        "heldout_recall": {
            "aligned_test": test_recall,
            "natural_device": natural_recall,
        },
    }
    return report


def rebind_continuous_report(
    source_continuous: Path,
    source_clip: Path,
    rebound_clip_path: Path,
    rebound_clip: Mapping[str, Any],
) -> dict[str, Any]:
    continuous = json.loads(source_continuous.read_text())
    if continuous.get("gate_scope") != "untouched_continuous_qualification":
        raise ValueError("unexpected continuous gate scope")
    embedded_source = continuous.get("teacher_qualification", {}).get("report_sha256")
    if embedded_source != sha256_file(source_clip):
        raise ValueError("continuous report is not bound to the source clip report")
    source_threshold = float(continuous.get("scoring", {}).get("threshold"))
    threshold = float(rebound_clip["scoring"]["threshold"])
    if threshold < source_threshold:
        raise ValueError("continuous rebinding may only tighten the threshold")
    if continuous.get("model", {}).get("weights_sha256") != rebound_clip.get("model", {}).get("weights_sha256"):
        raise ValueError("continuous and rebound clip reports use different teacher weights")

    report = copy.deepcopy(continuous)
    categories: dict[str, int] = {}
    false_accepts = 0
    for member in report["members"]:
        kept = [
            event for event in member.get("events", [])
            if float(event.get("peak_score", -math.inf)) >= threshold
        ]
        member["events"] = kept
        count = len(kept)
        categories[member["category"]] = categories.get(member["category"], 0) + count
        false_accepts += count
    for name, values in report["categories"].items():
        values["events"] = categories.get(name, 0)
    exposure_hours = float(report["counts"]["exposure_hours"])
    faph = false_accepts / max(exposure_hours, 1e-12)
    upper = poisson_upper_95(false_accepts, exposure_hours)
    min_hours = float(report["limits"]["min_exposure_hours"])
    max_upper = float(report["limits"]["max_faph_upper_95"])
    reasons = []
    if exposure_hours < min_hours:
        reasons.append(f"exposure {exposure_hours:.4f} hours is below {min_hours:.4f}")
    if upper > max_upper:
        reasons.append(f"FAPH upper bound {upper:.4f} exceeds {max_upper:.4f}")
    report["schema_version"] = max(2, int(report.get("schema_version", 1)))
    report["qualified"] = not reasons
    report["failure_reasons"] = reasons
    report["scoring"]["threshold"] = threshold
    report["counts"].update(false_accepts=false_accepts, faph=faph, faph_upper_95=upper)
    report["teacher_qualification"].update(
        qualified=bool(rebound_clip["qualified"]),
        report_path=str(rebound_clip_path.resolve()),
        report_sha256=sha256_file(rebound_clip_path),
    )
    report["operating_point_rebinding"] = {
        "algorithm": "monotonic_filter_of_stored_event_peaks_v1",
        "source_continuous_report": {
            "path": str(source_continuous.resolve()),
            "sha256": sha256_file(source_continuous),
        },
        "source_threshold": source_threshold,
        "bound_threshold": threshold,
        "monotonic_tightening": True,
        "proof": "events below the looser source threshold cannot cross the stricter bound threshold",
    }
    return report


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-clip", type=Path, required=True)
    parser.add_argument("--source-continuous", type=Path, required=True)
    parser.add_argument("--adaptation-report", type=Path, required=True)
    parser.add_argument("--output-clip", type=Path, required=True)
    parser.add_argument("--output-continuous", type=Path, required=True)
    parser.add_argument("--min-natural-recall", type=float, default=0.875)
    args = parser.parse_args(argv)
    if args.output_clip.resolve() in (args.source_clip.resolve(), args.source_continuous.resolve()):
        parser.error("output clip must not overwrite source evidence")
    if args.output_continuous.resolve() in (args.source_clip.resolve(), args.source_continuous.resolve()):
        parser.error("output continuous report must not overwrite source evidence")
    clip = rebind_clip_report(
        args.source_clip,
        args.adaptation_report,
        min_natural_recall=args.min_natural_recall,
    )
    _write(args.output_clip, clip)
    continuous = rebind_continuous_report(
        args.source_continuous,
        args.source_clip,
        args.output_clip,
        clip,
    )
    _write(args.output_continuous, continuous)
    print(
        json.dumps(
            {
                "clip_qualified": clip["qualified"],
                "continuous_qualified": continuous["qualified"],
                "threshold": clip["scoring"]["threshold"],
                "clip_counts": clip["counts"],
                "continuous_counts": continuous["counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if clip["qualified"] and continuous["qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
