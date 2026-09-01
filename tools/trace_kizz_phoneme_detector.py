#!/usr/bin/env python3
"""Trace a frozen deployed Kizz phoneme detector over fixed features.

The default scorer delegates to ``qualify_kizz_phoneme_student`` so the trace
uses the same stateful TFLite stream, causal-window slicing, and suffix CTC
decoder as deployment qualification.  This module owns only provenance,
geometry, event serialization, and deterministic output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


SPLITS = ("train", "validation", "test")
FEATURE_BINS = 40
FEATURE_HOP_MS = 10.0
WINDOW_FRAMES = 260
STRIDE_FRAMES = 3
OUTPUT_FRAMES = 66
WINDOW_LENGTHS = (19, 23, 27, 32, 39, 47, 54)
SCORE_FLOOR = float(np.finfo(np.float32).min)
SUPPORTED_DECODERS = {
    "forward_sum_ctc": {
        "type": "deterministic_suffix_forward_sum_ctc",
        "implementation": "microwakeword.ctc_forward.exhaustive_sliding_forward_score",
    },
    "max_add_ctc_viterbi": {
        "type": "deterministic_suffix_viterbi_ctc",
        "implementation": "microwakeword.kizz_viterbi_decoder.exhaustive_suffix_score",
    },
}
PRESERVED_FIELDS = (
    "path",
    "audio_sha256",
    "sha256",
    "source_audio_sha256",
    "parent_source_audio_sha256",
    "feature_sha256",
    "parent_source_id",
    "speaker_id",
    "session_id",
    "ancestry_id",
    "ancestry_ids",
    "ancestry_sha256",
    "duration_seconds",
    "source_group",
    "provider",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _feature_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes(order="C")).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _load_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return payload


def _examples(payload: Mapping[str, Any], path: Path) -> list[dict[str, Any]]:
    rows = payload.get("examples", payload.get("records"))
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"{path}: expected examples or records list")
    return [dict(row) for row in rows]


def _binding(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path)}


def _required_hash(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    return value.lower()


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _resolve_declared_file(raw: Any, *, parent: Path, name: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{name} path is required")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = parent / candidate
    return candidate.resolve()


def validate_model_config(
    artifact: Path, config_path: Path, payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the converter's immutable causal TFLite/decoder contract."""
    artifact = artifact.resolve()
    config_path = config_path.resolve()
    if int(payload.get("schema_version", 0)) < 2:
        raise ValueError("model config schema is too old")
    artifact_info = payload.get("artifact")
    if not isinstance(artifact_info, dict):
        raise ValueError("model config requires artifact provenance")
    declared_artifact = _resolve_declared_file(
        artifact_info.get("path", artifact_info.get("filename")),
        parent=config_path.parent,
        name="artifact",
    )
    artifact_hash = sha256_file(artifact)
    if (
        declared_artifact != artifact
        or _required_hash(artifact_info.get("sha256"), "artifact sha256")
        != artifact_hash
        or int(artifact_info.get("bytes", -1)) != artifact.stat().st_size
    ):
        raise ValueError("model config artifact binding drifted")

    contract = payload.get("compact_phone_contract")
    if not isinstance(contract, dict) or not isinstance(contract.get("tokens"), list):
        raise ValueError("model config lacks compact phone contract")
    token_count = len(contract["tokens"])
    if token_count < 2 or contract.get("blank_id") != 0:
        raise ValueError("compact phone contract is invalid")

    architecture = payload.get("architecture")
    if not isinstance(architecture, dict) or architecture.get("architecture_id") not in {
        "control_mixconv",
        "temporal_residual",
    }:
        raise ValueError("unsupported student architecture")
    input_contract = payload.get("input")
    output_contract = payload.get("output")
    if (
        not isinstance(input_contract, dict)
        or input_contract.get("shape") != [1, STRIDE_FRAMES, FEATURE_BINS]
        or input_contract.get("dtype") != "int8"
    ):
        raise ValueError("deployed input must be int8[1,3,40]")
    if (
        not isinstance(output_contract, dict)
        or output_contract.get("shape") != [1, 1, token_count]
        or output_contract.get("dtype") != "uint8"
    ):
        raise ValueError("deployed output contract differs from phone vocabulary")

    timeline = payload.get("timeline")
    if not isinstance(timeline, dict):
        raise ValueError("model config lacks causal timeline")
    phase_offset = timeline.get("stream_phase_offset_frames")
    if (
        not math.isclose(_finite(timeline.get("feature_step_seconds"), "feature step"), 0.010)
        or int(timeline.get("output_frames", -1)) != OUTPUT_FRAMES
        or not isinstance(phase_offset, int)
        or not 0 <= phase_offset < STRIDE_FRAMES
        or timeline.get("stream_phase_priming")
        != "zero_prefix_then_observed_prefix"
        or timeline.get("causal_warmup_derived") is not True
    ):
        raise ValueError("model config causal timeline drifted")

    decoder = payload.get("decoder")
    if not isinstance(decoder, dict):
        raise ValueError("model config lacks decoder contract")
    decoder_contract = decoder.get("distillation_decoder_contract")
    if not isinstance(decoder_contract, dict):
        raise ValueError("model config lacks distillation decoder contract")
    algorithm = decoder_contract.get("algorithm")
    expected = SUPPORTED_DECODERS.get(str(algorithm))
    if expected is None:
        raise ValueError("unsupported student decoder algorithm")
    contract_hash = _canonical_hash(decoder_contract)
    if (
        decoder.get("algorithm") != algorithm
        or decoder.get("type") != expected["type"]
        or decoder_contract.get("type") != "kizz_ctc_phone_decoder"
        or decoder_contract.get("implementation") != expected["implementation"]
        or decoder_contract.get("window_lengths_frames") != list(WINDOW_LENGTHS)
        or _finite(decoder_contract.get("beta"), "decoder beta") != 0.0
        or decoder_contract.get("compact_phone_contract_sha256")
        != _canonical_hash(contract)
        or _required_hash(decoder.get("contract_sha256"), "decoder contract sha256")
        != contract_hash
        or decoder.get("distillation_decoder_contract_sha256") != contract_hash
    ):
        raise ValueError("model config decoder contract drifted")
    reference = _resolve_declared_file(
        decoder.get("reference_module"), parent=config_path.parent, name="decoder reference"
    )
    if (
        not reference.is_file()
        or _required_hash(
            decoder.get("reference_module_sha256"), "decoder reference sha256"
        )
        != sha256_file(reference)
    ):
        raise ValueError("model config decoder reference drifted")
    return {
        "artifact_sha256": artifact_hash,
        "contract": contract,
        "architecture_id": architecture["architecture_id"],
        "output_frames": OUTPUT_FRAMES,
        "stream_phase_offset_frames": phase_offset,
        "decoder_algorithm": algorithm,
        "decoder_contract_sha256": contract_hash,
        "beta": 0.0,
    }


def validate_threshold(
    threshold_path: Path,
    payload: Mapping[str, Any],
    *,
    artifact_hash: str,
    config_hash: str,
    decoder_contract_hash: str,
) -> float:
    """Read a validation-selected threshold bound to this exact detector."""
    point = payload.get("threshold")
    if isinstance(point, dict):
        value = point.get("threshold")
        selection = point.get("selection")
    else:
        value = point
        selection = payload.get("selection", payload.get("threshold_selection"))
    threshold = _finite(value, "detector threshold")
    if threshold <= SCORE_FLOOR:
        raise ValueError("detector threshold must exceed serialized score floor")
    if selection != "validation_only":
        raise ValueError("detector threshold must be selected from validation only")

    artifact_metadata = payload.get("artifact_metadata")
    bound_artifact = payload.get("artifact_sha256")
    bound_config = payload.get("config_sha256")
    if isinstance(artifact_metadata, dict):
        bound_artifact = artifact_metadata.get("artifact_sha256", bound_artifact)
        bound_config = artifact_metadata.get("sha256", bound_config)
    decoder = payload.get("decoder")
    bound_decoder = payload.get("decoder_contract_sha256")
    if isinstance(decoder, dict):
        bound_decoder = decoder.get("contract_sha256", bound_decoder)
    if (
        _required_hash(bound_artifact, "threshold artifact sha256") != artifact_hash
        or _required_hash(bound_config, "threshold config sha256") != config_hash
        or _required_hash(bound_decoder, "threshold decoder contract sha256")
        != decoder_contract_hash
    ):
        raise ValueError(f"threshold provenance drifted: {threshold_path}")
    return threshold


def _identity_values(row: Mapping[str, Any]) -> set[str]:
    values = {
        str(row[key])
        for key in ("source_id", "parent_source_id", "speaker_id", "session_id", "ancestry_id")
        if row.get(key) not in (None, "")
    }
    for key in ("ancestry_ids", "parent_source_ids"):
        raw = row.get(key, [])
        if isinstance(raw, list):
            values.update(str(value) for value in raw if value not in (None, ""))
    return values


def _source_hash_values(row: Mapping[str, Any]) -> set[str]:
    return {
        str(row[key])
        for key in (
            "audio_sha256",
            "sha256",
            "source_audio_sha256",
            "parent_source_audio_sha256",
        )
        if row.get(key) not in (None, "")
    }


def _reject_split_leakage(rows: Sequence[Mapping[str, Any]]) -> None:
    identities: dict[str, set[str]] = defaultdict(set)
    hashes: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split = str(row["split"])
        identities[split].update(_identity_values(row))
        hashes[split].update(_source_hash_values(row))
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            if identities[left] & identities[right]:
                raise ValueError(f"{left}/{right} source identity overlap")
            if hashes[left] & hashes[right]:
                raise ValueError(f"{left}/{right} source hash overlap")


def validate_sources(
    manifest_path: Path,
    payload: Mapping[str, Any],
    feature_path: Path,
    features: np.ndarray,
) -> list[dict[str, Any]]:
    if features.ndim != 3 or features.shape[0] < 1 or features.shape[2] != FEATURE_BINS:
        raise ValueError("source features must be [examples,frames,40]")
    if features.shape[1] < 1 or not np.issubdtype(features.dtype, np.number):
        raise ValueError("source features must be nonempty numeric values")
    feature_file_hash = sha256_file(feature_path)
    declared = payload.get("array_sha256")
    if not isinstance(declared, dict) or declared.get(feature_path.name) != feature_file_hash:
        raise ValueError("source manifest feature-array SHA-256 is missing or drifted")
    rows = _examples(payload, manifest_path)
    if len(rows) != len(features):
        raise ValueError("source manifest and feature array example counts differ")
    seen_ids: set[str] = set()
    seen_indexes: set[int] = set()
    for ordinal, row in enumerate(rows):
        source_id = row.get("source_id")
        split = row.get("split")
        label = row.get("label")
        index = row.get("feature_index", ordinal)
        if not isinstance(source_id, str) or not source_id or source_id in seen_ids:
            raise ValueError("source IDs must be unique nonempty strings")
        if split not in SPLITS or label not in (0, 1, False, True):
            raise ValueError(f"{source_id}: invalid split or binary label")
        if not isinstance(index, int) or not 0 <= index < len(features) or index in seen_indexes:
            raise ValueError(f"{source_id}: invalid or duplicate feature_index")
        values = np.asarray(features[index])
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{source_id}: source features contain non-finite values")
        observed_feature_hash = _feature_hash(values)
        declared_feature_hash = row.get("feature_sha256")
        if declared_feature_hash is not None and (
            _required_hash(declared_feature_hash, f"{source_id} feature_sha256")
            != observed_feature_hash
        ):
            raise ValueError(f"{source_id}: feature hash drift")
        audio_hash = row.get("audio_sha256", row.get("source_audio_sha256", row.get("sha256")))
        audio_hash = _required_hash(audio_hash, f"{source_id} audio sha256")
        audio_path = _resolve_declared_file(
            row.get("path"), parent=manifest_path.parent, name=f"{source_id} audio"
        )
        if not audio_path.is_file() or sha256_file(audio_path) != audio_hash:
            raise ValueError(f"{source_id}: audio hash drift")
        row["feature_index"] = index
        row["path"] = str(audio_path)
        row["_observed_feature_sha256"] = observed_feature_hash
        seen_ids.add(source_id)
        seen_indexes.add(index)
    if seen_indexes != set(range(len(features))):
        raise ValueError("source manifest does not map every feature row exactly once")
    _reject_split_leakage(rows)
    return rows


def threshold_region_events(
    scores: Sequence[float], feature_frame_indexes: Sequence[int], threshold: float
) -> list[dict[str, Any]]:
    values = np.asarray(scores, dtype=np.float64)
    indexes = list(feature_frame_indexes)
    if values.ndim != 1 or len(values) != len(indexes) or not np.all(np.isfinite(values)):
        raise ValueError("scores and feature indexes must be aligned finite vectors")
    events: list[dict[str, Any]] = []
    start = 0
    while start < len(values):
        if values[start] < threshold:
            start += 1
            continue
        stop = start + 1
        while stop < len(values) and values[stop] >= threshold:
            stop += 1
        peak = start + int(np.argmax(values[start:stop]))
        events.append(
            {
                "score_frame_index": peak,
                "feature_frame_index": indexes[peak],
                "score": float(values[peak]),
                "threshold_region_start_score_frame_index": start,
                "threshold_region_end_score_frame_index": stop - 1,
                "threshold_region_start_feature_frame_index": indexes[start],
                "threshold_region_end_feature_frame_index": indexes[stop - 1],
            }
        )
        start = stop
    return events


class QualifierScorer:
    """Thin adapter around the deployed qualification implementation."""

    def __init__(self, artifact: Path, model_contract: Mapping[str, Any]) -> None:
        # Keep TensorFlow and its startup cost out of provenance-only unit tests.
        if __package__:
            from tools.distill_kizz_phoneme_student import (
                student_flags_for_architecture,
            )
            from tools.qualify_kizz_phoneme_student import DeployedStudent
        else:
            from distill_kizz_phoneme_student import student_flags_for_architecture
            from qualify_kizz_phoneme_student import DeployedStudent

        flags = student_flags_for_architecture(
            str(model_contract["architecture_id"]),
            len(model_contract["contract"]["tokens"]),
        )
        if int(flags.stride) != STRIDE_FRAMES:
            raise ValueError("deployed architecture stride differs from trace geometry")
        self.model = DeployedStudent(
            artifact,
            output_frames=int(model_contract["output_frames"]),
            stream_phase_offset_frames=int(model_contract["stream_phase_offset_frames"]),
            flags=flags,
        )
        self.contract = dict(model_contract["contract"])
        self.algorithm = str(model_contract["decoder_algorithm"])
        self.beta = float(model_contract["beta"])

    def score(self, features: np.ndarray) -> Sequence[float]:
        if __package__:
            from tools.qualify_kizz_phoneme_student import _stream_window_scores
        else:
            from qualify_kizz_phoneme_student import _stream_window_scores

        scores, timestamps = _stream_window_scores(
            self.model,
            features,
            self.contract,
            beta=self.beta,
            decoder_algorithm=self.algorithm,
        )
        expected_timestamps = [
            index * STRIDE_FRAMES * FEATURE_HOP_MS / 1000.0
            for index in range(len(scores))
        ]
        if not np.allclose(timestamps, expected_timestamps, rtol=0.0, atol=1e-12):
            raise ValueError("qualifier score timestamps differ from deployed geometry")
        return scores


def _score_one(scorer: Any, features: np.ndarray) -> list[float]:
    raw = scorer.score(features) if hasattr(scorer, "score") else scorer(features)
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("detector scorer must emit a one-dimensional score vector")
    padded_frames = max(len(features), WINDOW_FRAMES)
    expected_count = (padded_frames - WINDOW_FRAMES) // STRIDE_FRAMES + 1
    if len(values) != expected_count:
        raise ValueError(
            f"detector score geometry mismatch: got {len(values)}, expected {expected_count}"
        )
    if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
        raise ValueError("detector scorer emitted NaN or positive infinity")
    values = np.where(np.isneginf(values), SCORE_FLOOR, values)
    if not np.all(np.isfinite(values)):
        raise ValueError("detector scorer emitted an unsupported non-finite score")
    return [float(value) for value in values]


def generate_detector_traces(
    artifact: Path,
    model_config: Path,
    threshold_file: Path,
    source_manifest: Path,
    source_features: Path,
    output: Path,
    *,
    scorer_factory: Callable[[Path, Mapping[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Generate and atomically write a verifier-builder-compatible trace."""
    paths = [artifact, model_config, threshold_file, source_manifest, source_features]
    artifact, model_config, threshold_file, source_manifest, source_features = (
        path.expanduser().resolve() for path in paths
    )
    config_payload = _load_object(model_config)
    model_contract = validate_model_config(artifact, model_config, config_payload)
    config_hash = sha256_file(model_config)
    threshold_payload = _load_object(threshold_file)
    threshold = validate_threshold(
        threshold_file,
        threshold_payload,
        artifact_hash=model_contract["artifact_sha256"],
        config_hash=config_hash,
        decoder_contract_hash=model_contract["decoder_contract_sha256"],
    )
    source_payload = _load_object(source_manifest)
    features = np.load(source_features, mmap_mode="r", allow_pickle=False)
    rows = validate_sources(
        source_manifest, source_payload, source_features, features
    )
    factory = scorer_factory or QualifierScorer
    scorer = factory(artifact, model_contract)

    traced: list[dict[str, Any]] = []
    negative_infinity_count = 0
    event_count = 0
    for row in sorted(rows, key=lambda item: str(item["source_id"])):
        values = np.asarray(features[int(row["feature_index"])])
        scores = _score_one(scorer, values)
        negative_infinity_count += sum(score == SCORE_FLOOR for score in scores)
        feature_indexes = [
            WINDOW_FRAMES - 1 + index * STRIDE_FRAMES for index in range(len(scores))
        ]
        events = threshold_region_events(scores, feature_indexes, threshold)
        event_count += len(events)
        item: dict[str, Any] = {
            "source_id": row["source_id"],
            "split": row["split"],
            "label": int(row["label"]),
            "feature_index": int(row["feature_index"]),
            **{key: row[key] for key in PRESERVED_FIELDS if key in row},
            "source_feature_sha256": row["_observed_feature_sha256"],
            "source_feature_shape": list(values.shape),
            "source_feature_dtype": str(values.dtype),
            "scores": scores,
            "feature_frame_indexes": feature_indexes,
            "events": events,
        }
        traced.append(item)

    report: dict[str, Any] = {
        "schema_version": 1,
        "recipe": "kizz_control_deployed_phoneme_detector_trace_v1",
        "source_manifest": _binding(source_manifest),
        "source_features": _binding(source_features),
        "detector": {
            "artifact": _binding(artifact),
            "config": _binding(model_config),
            "threshold": {**_binding(threshold_file), "value": threshold},
            "event_policy": "recorded_events",
            "score_geometry": {
                "feature_stride_frames": STRIDE_FRAMES,
                "feature_offset_frames": WINDOW_FRAMES - 1,
                "feature_hop_ms": FEATURE_HOP_MS,
                "window_frames": WINDOW_FRAMES,
                "score_position": "causal_window_final_evidence_frame",
                "short_source_padding": "right_zero_to_window_frames",
            },
            "streaming": {
                "implementation": "tools.qualify_kizz_phoneme_student._stream_window_scores",
                "model": "tools.qualify_kizz_phoneme_student.DeployedStudent",
                "one_stateful_interpreter_per_source": True,
                "output_frames_per_window": OUTPUT_FRAMES,
                "stream_phase_offset_frames": model_contract[
                    "stream_phase_offset_frames"
                ],
            },
            "decoder": {
                "algorithm": model_contract["decoder_algorithm"],
                "contract_sha256": model_contract["decoder_contract_sha256"],
                "beta": model_contract["beta"],
                "window_lengths_frames": list(WINDOW_LENGTHS),
            },
            "score_serialization": {
                "negative_infinity": "float32_min",
                "finite_floor": SCORE_FLOOR,
            },
        },
        "counts": {
            "sources": len(traced),
            "score_frames": sum(len(item["scores"]) for item in traced),
            "threshold_region_events": event_count,
            "negative_infinity_scores_serialized": negative_infinity_count,
            "by_split": {
                split: sum(item["split"] == split for item in traced)
                for split in SPLITS
            },
        },
        "examples": traced,
    }
    _atomic_json(output, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--threshold-file", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = generate_detector_traces(
        args.artifact,
        args.model_config,
        args.threshold_file,
        args.source_manifest,
        args.source_features,
        args.output,
    )
    print(json.dumps(report["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
