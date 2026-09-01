#!/usr/bin/env python3
"""Apply a frozen device-validation threshold once to reserved device test audio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.evaluate_kizz_int8_continuous_cascade import TFLiteRuntime
from tools.extend_kizz_candidate_verifier_with_device_corpus import (
    _load_manifest,
    _validate_capture_metadata,
    _validate_train_audio,
)
from tools.mine_kizz_librispeech_hard_negatives import _binding, _mine_file, sha256_file
from tools.score_kizz_candidate_verifier_device_validation import _load_object
from tools.simulate_kizz_int8_cascade import load_firmware_artifact
from tools.trace_kizz_candidate_verifier import Int8Verifier
from tools.trace_kizz_ordered_state_detector import _threshold_from_report, _validate_artifact


def _bound_path(value: object, anchor: Path, label: str) -> Path:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} binding is missing")
    raw = value.get("path")
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} path is missing")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = anchor / path
    path = path.resolve()
    if not path.is_file() or sha256_file(path) != value.get("sha256"):
        raise ValueError(f"{label} binding drift")
    return path


def validate_reserved_evidence(
    captures: Sequence[Mapping[str, Any]], evidence: Mapping[str, Any]
) -> None:
    if (
        evidence.get("kind")
        != "kizz_control_voice_stratified_device_replay_qualification_evidence"
        or evidence.get("training_eligible") is not False
    ):
        raise ValueError("reserved device evidence has the wrong contract")
    examples = evidence.get("examples")
    if not isinstance(examples, list):
        raise ValueError("reserved device evidence examples are missing")
    capture_hashes = {str(row.get("sha256", "")) for row in captures}
    evidence_hashes = {str(row.get("audio_sha256", "")) for row in examples}
    if (
        len(capture_hashes) != len(captures)
        or len(evidence_hashes) != len(examples)
        or capture_hashes != evidence_hashes
    ):
        raise ValueError("reserved evidence does not exactly cover device captures")
    if any(
        row.get("split") != "test" or row.get("training_eligible") is not False
        for row in examples
    ):
        raise ValueError("reserved evidence is not test-only")


def finalize(
    validation_report_path: Path,
    test_corpus_path: Path,
    test_evidence_path: Path,
) -> dict[str, Any]:
    validation_report_path = validation_report_path.expanduser().resolve()
    test_corpus_path = test_corpus_path.expanduser().resolve()
    test_evidence_path = test_evidence_path.expanduser().resolve()
    validation = _load_object(validation_report_path, "device validation report")
    if (
        validation.get("kind")
        != "kizz_control_candidate_verifier_device_validation_threshold"
        or validation.get("test_audio_opened") is not False
        or validation.get("selection", {}).get("test_used_for_selection") is not False
        or validation.get("selection", {}).get("meets_minimum_recall") is not True
    ):
        raise ValueError("device validation threshold report is not qualified")
    threshold = float(validation["selection"]["threshold"])
    bindings = validation.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("device validation bindings are missing")
    anchor = validation_report_path.parent
    detector_metadata = _bound_path(bindings.get("detector_config"), anchor, "detector config")
    detector_model = _bound_path(bindings.get("detector_artifact"), anchor, "detector artifact")
    detector_threshold_report = _bound_path(
        bindings.get("detector_threshold_report"), anchor, "detector threshold"
    )
    verifier_metadata = _bound_path(bindings.get("verifier_config"), anchor, "verifier config")
    verifier_model = _bound_path(bindings.get("verifier_artifact"), anchor, "verifier artifact")

    manifest = _load_manifest(test_corpus_path)
    captures = manifest["captures"]
    evidence = _load_object(test_evidence_path, "reserved device evidence")
    validate_reserved_evidence(captures, evidence)
    normalized = []
    for index, raw in enumerate(captures):
        if not isinstance(raw, dict):
            raise ValueError(f"capture[{index}] must be an object")
        capture_id, truth, path, declared_hash, samples, duration = (
            _validate_capture_metadata(
                raw, index=index, manifest_root=test_corpus_path.parent.resolve()
            )
        )
        if (
            raw.get("split") != "test"
            or truth != "positive"
            or raw.get("conditions", {}).get("evidence_role")
            != "reserved_target_channel_positive"
        ):
            raise ValueError(f"{capture_id}: reserved capture contract drift")
        normalized.append(
            {
                **raw,
                "path": str(path),
                "declared_sha256": declared_hash,
                "declared_samples": samples,
                "declared_duration_seconds": duration,
            }
        )

    _, topology, detector_contract = _validate_artifact(detector_metadata, detector_model)
    detector_threshold, detector_threshold_provenance = _threshold_from_report(
        detector_threshold_report, topology
    )
    detector_artifact = load_firmware_artifact(detector_metadata, "detector")
    detector = TFLiteRuntime(detector_model, detector_artifact)
    verifier = Int8Verifier(verifier_model)
    results = []
    detector_candidates = 0
    accepted = 0
    for capture in sorted(normalized, key=lambda row: str(row["capture_id"])):
        audio_hash, duration = _validate_train_audio(capture)
        candidates, frame_count, hop_count = _mine_file(
            Path(capture["path"]),
            Path(capture["path"]).parent,
            detector,
            topology,
            detector_contract,
            detector_threshold,
            top_k=1,
        )
        verifier_score = None
        detector_score = None
        trigger = None
        if candidates:
            detector_candidates += 1
            detector_score, trigger, feature = candidates[0]
            verifier_score = float(verifier(np.asarray(feature, dtype=np.float32)))
            accepted += int(verifier_score >= threshold)
        results.append(
            {
                "capture_id": capture["capture_id"],
                "audio_sha256": audio_hash,
                "duration_seconds": duration,
                "provider": capture.get("conditions", {}).get("source_provider"),
                "voice": capture.get("conditions", {}).get("source_voice"),
                "detector_candidate": bool(candidates),
                "detector_score": float(detector_score) if detector_score is not None else None,
                "detector_feature_frame": int(trigger) if trigger is not None else None,
                "verifier_logit": verifier_score,
                "accepted": verifier_score is not None and verifier_score >= threshold,
                "frontend_feature_frames": int(frame_count),
                "detector_hops": int(hop_count),
            }
        )
    full_test_recall = detector_candidates == len(normalized) and accepted == len(
        normalized
    )

    validation_selection = validation["selection"]
    return {
        "schema_version": 1,
        "kind": "kizz_control_candidate_verifier_physical_recall_threshold",
        "deployment_qualification": False,
        "locked_audio_used_for_tuning": False,
        "test_scored_after_threshold_frozen": True,
        "threshold": threshold,
        "selection": {
            "qualified": full_test_recall,
            "fit_split": "physical_microphone_replay",
            "fit_subset": "voice_disjoint_device_validation",
            "test_used_for_selection": False,
            "meets_minimum_recall": full_test_recall,
            "threshold": threshold,
            "detector_candidates": validation_selection["detector_candidates"],
            "accepted_candidates": validation_selection["accepted_candidates"],
        },
        "bindings": {
            "artifact": _binding(verifier_model),
            "config": _binding(verifier_metadata),
            "validation_report": _binding(validation_report_path),
            "test_corpus": _binding(test_corpus_path),
            "test_evidence": _binding(test_evidence_path),
        },
        "detector_threshold": {
            "value": detector_threshold,
            "provenance": detector_threshold_provenance,
        },
        "physical_replay": {
            "transport": "recorded StackChan ESP32-S3 microphone captures",
            "validation_captures": validation_selection["detector_candidates"],
            "validation_compact_accepts": validation_selection["accepted_candidates"],
            "distinct_held_out_test_captures": len(normalized),
            "test_detector_candidates": detector_candidates,
            "test_compact_accepts": accepted,
        },
        "test": {
            "threshold_frozen_before_audio_access": True,
            "detector_candidates": detector_candidates,
            "accepted_candidates": accepted,
            "recall": accepted / len(normalized),
            "detector_recall": detector_candidates / len(normalized),
            "meets_full_recall_gate": full_test_recall,
            "results": results,
        },
        "limitations": [
            "Threshold fitting used synthesized speech replayed through the target microphone path.",
            "Continuous-negative precision is evaluated separately on the locked 100-hour corpus.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--test-corpus", type=Path, required=True)
    parser.add_argument("--test-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = finalize(args.validation_report, args.test_corpus, args.test_evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"selection": report["selection"], "test": {key: value for key, value in report["test"].items() if key != "results"}}, sort_keys=True))
    return 0 if report["selection"]["qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
