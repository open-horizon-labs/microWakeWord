#!/usr/bin/env python3
"""Package a provenance-bound three-stage Kizz Control firmware handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _all_sha256_values(value: Any) -> set[str]:
    values: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "sha256" and isinstance(child, str) and len(child) == 64:
                values.add(child)
            values.update(_all_sha256_values(child))
    elif isinstance(value, list):
        for child in value:
            values.update(_all_sha256_values(child))
    return values


def _copy_bound(path: Path, destination: Path) -> dict[str, Any]:
    digest = sha256_file(path)
    shutil.copyfile(path, destination)
    return {
        "filename": destination.name,
        "bytes": destination.stat().st_size,
        "sha256": digest,
    }


def package(args: argparse.Namespace) -> dict[str, Any]:
    model_paths = {
        "detector": args.detector_model.resolve(),
        "compact_verifier": args.compact_model.resolve(),
        "ordered_verifier": args.ordered_model.resolve(),
    }
    metadata_paths = {
        "detector": args.detector_metadata.resolve(),
        "compact_verifier": args.compact_metadata.resolve(),
        "ordered_verifier": args.ordered_metadata.resolve(),
    }
    threshold_paths = {
        "detector": args.detector_threshold.resolve(),
        "compact_verifier": args.compact_threshold.resolve(),
        "ordered_verifier": args.ordered_threshold.resolve(),
    }
    positive = load_json(args.positive_report.resolve())
    continuous = load_json(args.continuous_report.resolve())
    metrics = continuous.get("metrics", {})
    policy = continuous.get("policy", {})
    physical = continuous.get("physical_hardware_proof", {})
    if continuous.get("kind") != "kizz_control_int8_continuous_negative_cascade_v1":
        raise ValueError("continuous report has the wrong kind")
    if positive.get("test_scored_after_threshold_frozen") is not True:
        raise ValueError("positive test was not scored after threshold freeze")
    if positive.get("locked_audio_used_for_tuning") is not False:
        raise ValueError("positive report does not prove locked audio stayed out of tuning")
    test = positive.get("test", {})
    if not test.get("threshold_frozen_before_audio_access"):
        raise ValueError("fresh test audio was opened before threshold freeze")
    recall = float(test.get("recall", -1.0))
    if recall < args.minimum_test_recall:
        raise ValueError(
            f"fresh physical-channel recall {recall:.6f} is below {args.minimum_test_recall:.6f}"
        )
    exposure_hours = float(metrics.get("exposure_hours", 0.0))
    if exposure_hours < args.minimum_negative_hours:
        raise ValueError(
            f"continuous exposure {exposure_hours:.3f}h is below {args.minimum_negative_hours:.3f}h"
        )
    observed_faph = float(metrics.get("accepted_false_wakes_per_hour", float("inf")))
    if observed_faph > args.accepted_practical_faph:
        raise ValueError(
            f"observed {observed_faph:.6f} false wakes/hour exceeds practical ceiling "
            f"{args.accepted_practical_faph:.6f}"
        )
    if observed_faph > args.formal_faph and not args.accept_observed_operating_point:
        raise ValueError(
            "packaging an operating point above the formal qualification gate requires "
            "--accept-observed-operating-point"
        )
    if physical.get("present") is not False:
        raise ValueError("host report must keep physical hardware proof separate")

    model_hashes = {role: sha256_file(path) for role, path in model_paths.items()}
    continuous_bindings = _all_sha256_values(continuous)
    for role, digest in model_hashes.items():
        if digest not in continuous_bindings:
            raise ValueError(f"continuous report is not bound to the {role} model")
    positive_bindings = _all_sha256_values(positive)
    if model_hashes["compact_verifier"] not in positive_bindings:
        raise ValueError("positive report is not bound to the compact_verifier model")
    positive_detector_threshold = float(
        positive.get("detector_threshold", {}).get("value", float("nan"))
    )
    if positive_detector_threshold != float(policy["detector_threshold"]):
        raise ValueError(
            "positive and continuous reports disagree on the detector threshold"
        )

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing package: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        names = {
            "detector": "kizz_control_detector.tflite",
            "compact_verifier": "kizz_control_compact_verifier_int8.tflite",
            "ordered_verifier": "kizz_control_ordered_verifier_int8.tflite",
        }
        packaged_models = {
            role: _copy_bound(path, temporary / names[role])
            for role, path in model_paths.items()
        }
        packaged_metadata = {
            role: _copy_bound(path, temporary / f"{role}.metadata.json")
            for role, path in metadata_paths.items()
        }
        packaged_thresholds = {
            role: _copy_bound(path, temporary / f"{role}.threshold.json")
            for role, path in threshold_paths.items()
        }
        reports = {
            "fresh_device_test": _copy_bound(
                args.positive_report.resolve(), temporary / "fresh-device-test.report.json"
            ),
            "locked_continuous_negative": _copy_bound(
                args.continuous_report.resolve(), temporary / "continuous-negative.report.json"
            ),
        }
        manifest = {
            "schema_version": 1,
            "kind": "kizz_control_three_stage_firmware_handoff",
            "phrase": "Kizz Control",
            "models": packaged_models,
            "metadata": packaged_metadata,
            "threshold_reports": packaged_thresholds,
            "thresholds": {
                "detector": float(policy["detector_threshold"]),
                "compact_verifier": float(policy["verifier_logit_threshold"]),
                "ordered_verifier": float(policy["ordered_verifier_score_threshold"]),
            },
            "positive_evidence": {
                "fresh_device_test_recall": recall,
                "accepted": int(test.get("accepted_candidates", 0)),
                "detector_candidates": int(test.get("detector_candidates", 0)),
                "threshold_frozen_before_audio_access": True,
                "compact_model_directly_bound": True,
                "detector_model_directly_bound": False,
                "detector_threshold_matches_continuous_report": True,
            },
            "continuous_negative_evidence": {
                "exposure_hours": exposure_hours,
                "detector_candidates": int(metrics.get("detector_candidates", 0)),
                "compact_accepts": int(metrics.get("compact_verifier_accepts", 0)),
                "compact_acceptance_fraction": float(
                    metrics.get("compact_verifier_acceptance_fraction", 0.0)
                ),
                "accepted_false_wakes": int(metrics.get("accepted_false_wakes", 0)),
                "accepted_false_wakes_per_hour": observed_faph,
                "one_sided_upper_95_per_hour": float(
                    metrics["accepted_false_wake_rate_confidence"]["one_sided_upper_95_per_hour"]
                ),
            },
            "qualification": {
                "formal_false_wakes_per_hour_ceiling": args.formal_faph,
                "formal_gate_passed": observed_faph <= args.formal_faph,
                "practical_false_wakes_per_hour_ceiling": args.accepted_practical_faph,
                "practical_operating_point_explicitly_accepted": bool(
                    args.accept_observed_operating_point
                ),
                "host_cascade_evidence_passed": True,
                "physical_hardware_qualified": False,
                "remaining_hardware_evidence": physical.get("remaining", []),
            },
            "firmware_contract": {
                "execution": "fixed AOT schedules with static arenas and ESP-NN kernels",
                "candidate_window_feature_frames": 260,
                "pre_context_feature_frames": 220,
                "post_context_feature_frames": 39,
                "startup_aot_equivalence_required": True,
                "exact_artifact_flash_and_physical_soak_required": True,
            },
            "limitations": [
                "The historical fresh-device report binds the compact model, frozen validation report, and detector threshold, but not the detector model hash directly; the locked continuous report binds all three model hashes.",
                "Host evidence does not establish ESP32-S3 latency, memory, power, thermal, or soak behavior.",
            ],
            "reports": reports,
        }
        (temporary / "cascade.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ("detector", "compact", "ordered"):
        parser.add_argument(f"--{role}-metadata", type=Path, required=True)
        parser.add_argument(f"--{role}-model", type=Path, required=True)
        parser.add_argument(f"--{role}-threshold", type=Path, required=True)
    parser.add_argument("--positive-report", type=Path, required=True)
    parser.add_argument("--continuous-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-test-recall", type=float, default=0.95)
    parser.add_argument("--minimum-negative-hours", type=float, default=100.0)
    parser.add_argument("--formal-faph", type=float, default=0.1)
    parser.add_argument("--accepted-practical-faph", type=float, required=True)
    parser.add_argument("--accept-observed-operating-point", action="store_true")
    return parser


def main() -> None:
    manifest = package(build_parser().parse_args())
    print(json.dumps(manifest["qualification"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
