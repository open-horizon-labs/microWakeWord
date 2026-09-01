#!/usr/bin/env python3
"""Freeze compact-verifier recall threshold on qualified device validation replays.

Only ``validation`` captures are eligible.  Test audio is rejected before any
audio file is opened, keeping the reserved device set unavailable for model or
threshold selection.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.extend_kizz_candidate_verifier_with_device_corpus import (
    _load_manifest,
    _validate_capture_metadata,
    _validate_train_audio,
)
from tools.mine_kizz_librispeech_hard_negatives import _binding, _mine_file, sha256_file
from tools.simulate_kizz_int8_cascade import load_firmware_artifact
from tools.trace_kizz_candidate_verifier import Int8Verifier
from tools.trace_kizz_ordered_state_detector import _threshold_from_report, _validate_artifact
from tools.evaluate_kizz_int8_continuous_cascade import TFLiteRuntime


def select_full_recall_threshold(scores: Sequence[float]) -> float:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("validation verifier scores must be nonempty and finite")
    return float(np.min(values))


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not readable JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _qualified_validation_ids(
    quality_path: Path, corpus_path: Path, capture_ids: set[str]
) -> set[str]:
    quality = _load_object(quality_path, "validation quality report")
    if (
        quality.get("kind") != "kizz_control_teacher_adaptation_device_replay_quality"
        or quality.get("gate_scope")
        != "validation_only_target_channel_positive_quality"
        or quality.get("expected_split") != "validation"
    ):
        raise ValueError("device validation quality report has the wrong contract")
    inputs = quality.get("inputs")
    if not isinstance(inputs, Mapping) or inputs.get("corpus_sha256") != sha256_file(
        corpus_path
    ):
        raise ValueError("quality report is not bound to the device validation corpus")
    results = quality.get("results")
    if not isinstance(results, list):
        raise ValueError("quality report results are missing")
    decisions: dict[str, bool] = {}
    for result in results:
        if not isinstance(result, Mapping):
            raise ValueError("quality report result is malformed")
        capture_id = result.get("capture_id")
        qualified = result.get("qualified")
        if not isinstance(capture_id, str) or not isinstance(qualified, bool):
            raise ValueError("quality report decision is malformed")
        if capture_id in decisions:
            raise ValueError("quality report duplicates a capture")
        decisions[capture_id] = qualified
    if set(decisions) != capture_ids:
        raise ValueError("quality report does not cover the validation corpus")
    selected = {capture_id for capture_id, qualified in decisions.items() if qualified}
    if not selected:
        raise ValueError("quality report accepted no validation captures")
    return selected


def _verifier_model(metadata_path: Path) -> tuple[Path, dict[str, Any]]:
    metadata = _load_object(metadata_path, "verifier metadata")
    if (
        metadata.get("schema_version") != 1
        or metadata.get("kind") != "kizz_control_candidate_verifier_fixed_window_int8"
        or metadata.get("candidate_conditioned") is not True
    ):
        raise ValueError("unsupported compact verifier metadata")
    artifact = metadata.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ValueError("verifier artifact binding is missing")
    filename = artifact.get("filename")
    if not isinstance(filename, str) or not filename:
        raise ValueError("verifier artifact filename is missing")
    path = metadata_path.parent / filename
    if not path.is_file() or sha256_file(path) != artifact.get("sha256"):
        raise ValueError("verifier artifact binding drift")
    return path, metadata


def score_validation(
    corpus_path: Path,
    quality_path: Path,
    detector_metadata: Path,
    detector_model: Path,
    detector_threshold_report: Path,
    verifier_metadata: Path,
) -> dict[str, Any]:
    corpus_path = corpus_path.expanduser().resolve()
    quality_path = quality_path.expanduser().resolve()
    verifier_metadata = verifier_metadata.expanduser().resolve()
    manifest = _load_manifest(corpus_path)
    captures = manifest["captures"]
    normalized: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(captures):
        if not isinstance(raw, dict):
            raise ValueError(f"capture[{index}] must be an object")
        capture_id, truth, path, declared_hash, samples, duration = (
            _validate_capture_metadata(
                raw, index=index, manifest_root=corpus_path.parent.resolve()
            )
        )
        if raw.get("split") != "validation" or truth != "positive":
            raise ValueError(
                f"{capture_id}: only positive validation captures may be scored"
            )
        normalized[capture_id] = {
            **raw,
            "path": str(path),
            "declared_sha256": declared_hash,
            "declared_samples": samples,
            "declared_duration_seconds": duration,
        }
    qualified_ids = _qualified_validation_ids(
        quality_path, corpus_path, set(normalized)
    )

    detector_metadata = detector_metadata.expanduser().resolve()
    detector_model = detector_model.expanduser().resolve()
    detector_threshold_report = detector_threshold_report.expanduser().resolve()
    _, topology, detector_contract = _validate_artifact(
        detector_metadata, detector_model
    )
    detector_threshold, threshold_provenance = _threshold_from_report(
        detector_threshold_report, topology
    )
    detector_artifact = load_firmware_artifact(detector_metadata, "detector")
    detector = TFLiteRuntime(detector_model, detector_artifact)
    verifier_model, verifier_config = _verifier_model(verifier_metadata)
    verifier = Int8Verifier(verifier_model)

    results = []
    scores: list[float] = []
    provider_counts: Counter[str] = Counter()
    for capture_id in sorted(qualified_ids):
        capture = normalized[capture_id]
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
        score = None
        detector_score = None
        trigger = None
        if candidates:
            detector_score, trigger, feature = candidates[0]
            score = float(verifier(np.asarray(feature, dtype=np.float32)))
            if not math.isfinite(score):
                raise ValueError(f"{capture_id}: verifier score is not finite")
            scores.append(score)
        provider = str(capture.get("conditions", {}).get("source_provider", ""))
        provider_counts[provider] += 1
        results.append(
            {
                "capture_id": capture_id,
                "provider": provider,
                "voice": capture.get("conditions", {}).get("source_voice"),
                "audio_sha256": audio_hash,
                "duration_seconds": duration,
                "detector_candidate": bool(candidates),
                "detector_score": float(detector_score) if detector_score is not None else None,
                "detector_feature_frame": int(trigger) if trigger is not None else None,
                "verifier_logit": score,
                "frontend_feature_frames": int(frame_count),
                "detector_hops": int(hop_count),
            }
        )
    if len(scores) != len(qualified_ids):
        raise ValueError("detector missed a qualified device validation positive")
    threshold = select_full_recall_threshold(scores)
    accepted = sum(score >= threshold for score in scores)
    artifact = verifier_config["artifact"]
    return {
        "schema_version": 1,
        "kind": "kizz_control_candidate_verifier_device_validation_threshold",
        "deployment_qualification": False,
        "test_audio_opened": False,
        "selection": {
            "fit_split": "physical_microphone_validation_replay",
            "test_used_for_selection": False,
            "minimum_recall": 1.0,
            "threshold": threshold,
            "detector_candidates": len(scores),
            "accepted_candidates": accepted,
            "meets_minimum_recall": accepted == len(scores),
        },
        "counts": {
            "corpus_captures": len(captures),
            "quality_qualified": len(qualified_ids),
            "quality_rejected": len(captures) - len(qualified_ids),
            "providers": dict(sorted(provider_counts.items())),
        },
        "bindings": {
            "device_corpus": _binding(corpus_path),
            "quality_report": _binding(quality_path),
            "detector_config": _binding(detector_metadata),
            "detector_artifact": _binding(detector_model),
            "detector_threshold_report": _binding(detector_threshold_report),
            "verifier_config": _binding(verifier_metadata),
            "verifier_artifact": {
                "path": str(verifier_model),
                "sha256": artifact["sha256"],
                "bytes": verifier_model.stat().st_size,
            },
        },
        "detector_threshold": {
            "value": detector_threshold,
            "provenance": threshold_provenance,
        },
        "results": results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--quality-report", type=Path, required=True)
    parser.add_argument("--detector-metadata", type=Path, required=True)
    parser.add_argument("--detector-model", type=Path, required=True)
    parser.add_argument("--detector-threshold-report", type=Path, required=True)
    parser.add_argument("--verifier-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = score_validation(
        args.corpus,
        args.quality_report,
        args.detector_metadata,
        args.detector_model,
        args.detector_threshold_report,
        args.verifier_metadata,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"selection": report["selection"], "counts": report["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
