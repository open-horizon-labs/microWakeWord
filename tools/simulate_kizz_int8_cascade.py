#!/usr/bin/env python3
"""Replay and benchmark a provenance-bound Kizz Control INT8 cascade on a host.

This is a host simulator, not ESP32-S3 qualification.  It replays the exact
streaming detector hops and frozen detector-conditioned verifier candidates,
without selecting or changing either threshold.  Host latency is useful for
regression detection and scheduler arithmetic only; physical StackChan
telemetry remains mandatory for any hardware-performance claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

try:
    import resource
except ImportError:  # pragma: no cover - resource is POSIX-only.
    resource = None  # type: ignore[assignment]


FEATURE_BINS = 40
SOURCE_FRAMES = 260
FEATURE_HOP_SECONDS = 0.010
ESP_HOP_MS = 30.0
SPLITS = ("train", "validation", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not readable JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _resolve(raw: object, anchor: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} path is required")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = anchor / path
    return path.resolve()


def _verify_binding(
    value: object,
    *,
    anchor: Path,
    label: str,
    expected_path: Path | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} binding is required")
    path = _resolve(value.get("path"), anchor, label)
    expected_hash = _sha256(value.get("sha256"), f"{label} hash")
    if expected_path is not None and path != expected_path.resolve():
        raise ValueError(f"{label} path drift: expected {expected_path}, got {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"{label} hash drift: expected {expected_hash}, got {actual_hash}"
        )
    declared_bytes = value.get("bytes")
    if declared_bytes is not None and declared_bytes != path.stat().st_size:
        raise ValueError(f"{label} byte-size drift")
    return {
        "path": str(path),
        "sha256": actual_hash,
        "bytes": path.stat().st_size,
    }


def _verify_nested_bindings(
    value: object, *, anchor: Path, label: str
) -> list[dict[str, Any]]:
    """Verify every nested conventional object containing both path and hash."""
    verified: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        has_path = "path" in value
        has_hash = "sha256" in value
        if has_path != has_hash:
            raise ValueError(f"{label} has an incomplete path/hash binding")
        if has_path:
            verified.append(
                {"label": label, **_verify_binding(value, anchor=anchor, label=label)}
            )
        for key, child in value.items():
            if key not in {"path", "sha256", "bytes"}:
                verified.extend(
                    _verify_nested_bindings(
                        child, anchor=anchor, label=f"{label}.{key}"
                    )
                )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            verified.extend(
                _verify_nested_bindings(
                    child, anchor=anchor, label=f"{label}[{index}]"
                )
            )
    return verified


def _tensor_contract(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} tensor contract is required")
    shape = value.get("shape")
    dtype = value.get("dtype")
    quantization = value.get("quantization")
    if isinstance(quantization, Mapping):
        quantization = (quantization.get("scale"), quantization.get("zero_point"))
    if (
        not isinstance(shape, list)
        or not shape
        or not all(isinstance(item, int) and item > 0 for item in shape)
        or dtype not in {"int8", "int16", "uint8"}
        or not isinstance(quantization, (list, tuple))
        or len(quantization) != 2
    ):
        raise ValueError(f"{label} tensor contract is invalid")
    scale = float(quantization[0])
    zero = int(quantization[1])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"{label} tensor quantization scale is invalid")
    limits = np.iinfo(np.dtype(str(dtype)))
    if not limits.min <= zero <= limits.max:
        raise ValueError(f"{label} tensor quantization zero point is invalid")
    return {
        "shape": [int(item) for item in shape],
        "dtype": str(dtype),
        "quantization": [scale, zero],
    }


@dataclass(frozen=True)
class FirmwareArtifact:
    role: str
    metadata_path: Path
    metadata_sha256: str
    metadata: dict[str, Any]
    artifact_path: Path
    artifact_sha256: str
    input_contract: dict[str, Any]
    output_contract: dict[str, Any]
    verified_bindings: tuple[dict[str, Any], ...]


def _artifact_path(metadata: Mapping[str, Any], metadata_path: Path, label: str) -> Path:
    artifact = metadata.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ValueError(f"{label} metadata has no artifact contract")
    raw = artifact.get("path", artifact.get("filename"))
    return _resolve(raw, metadata_path.parent, f"{label} artifact")


def load_firmware_artifact(metadata_path: Path, role: str) -> FirmwareArtifact:
    metadata_path = metadata_path.expanduser().resolve()
    metadata = _load_object(metadata_path, f"{role} firmware metadata")
    expected_schema = 2 if role == "detector" else 1
    if metadata.get("schema_version") != expected_schema:
        raise ValueError(f"unsupported {role} firmware metadata schema")
    if metadata.get("deployment_qualification") is not False:
        raise ValueError(f"{role} firmware metadata must remain non-deployment-qualified")
    artifact_path = _artifact_path(metadata, metadata_path, role)
    artifact = metadata.get("artifact")
    assert isinstance(artifact, Mapping)
    expected_hash = _sha256(artifact.get("sha256"), f"{role} artifact hash")
    if not artifact_path.is_file() or sha256_file(artifact_path) != expected_hash:
        raise ValueError(f"{role} TFLite artifact hash drift")
    if artifact.get("bytes") != artifact_path.stat().st_size:
        raise ValueError(f"{role} TFLite artifact byte-size drift")
    tensors = metadata.get("tensor_contracts")
    if not isinstance(tensors, Mapping):
        raise ValueError(f"{role} firmware metadata lacks tensor contracts")
    input_contract = _tensor_contract(tensors.get("input"), f"{role} input")
    output_contract = _tensor_contract(tensors.get("output"), f"{role} output")

    if role == "detector":
        if (
            metadata.get("kind")
            != "kizz_control_ordered_state_detector_streaming_int8"
            or metadata.get("student_role")
            != "permissive_detector_candidate_generator"
        ):
            raise ValueError("detector student role drift")
        timeline = metadata.get("timeline")
        topology = metadata.get("topology")
        if not isinstance(timeline, Mapping) or not isinstance(topology, Mapping):
            raise ValueError("detector timeline/topology contract is required")
        stride = timeline.get("stream_input_frames_per_call")
        if input_contract["shape"] != [1, stride, FEATURE_BINS]:
            raise ValueError("detector input shape differs from streaming timeline")
        if input_contract["dtype"] != "int8":
            raise ValueError("detector input must be INT8")
        if output_contract["shape"] != [1, 1, topology.get("state_count")]:
            raise ValueError("detector output shape differs from topology")
        hop = float(timeline.get("stream_hop_seconds", -1))
        if not math.isclose(hop, float(stride) * FEATURE_HOP_SECONDS):
            raise ValueError("detector streaming hop contract drift")
        phase = timeline.get("stream_phase_offset_frames")
        if not isinstance(phase, int) or not 0 <= phase < int(stride):
            raise ValueError("detector phase offset is invalid")
    elif role == "verifier":
        kind = str(metadata.get("kind", metadata.get("artifact_kind", "")))
        if (
            kind != "kizz_control_candidate_verifier_fixed_window_int8"
            or metadata.get("model_role")
            != "detector_conditioned_candidate_verifier"
            or metadata.get("candidate_conditioned") is not True
        ):
            raise ValueError("verifier firmware artifact kind is invalid")
        valid_inputs = ([1, SOURCE_FRAMES, FEATURE_BINS], [1, SOURCE_FRAMES, FEATURE_BINS, 1])
        if (
            input_contract["shape"] not in valid_inputs
            or input_contract["dtype"] not in {"int8", "int16"}
        ):
            raise ValueError(
                "verifier input must be integer [1,260,40] or [1,260,40,1]"
            )
        if output_contract["dtype"] != input_contract["dtype"]:
            raise ValueError("verifier input/output integer dtypes must match")
        if output_contract["shape"] not in ([1], [1, 1]):
            raise ValueError("verifier output must be a single quantized score")
        threshold = metadata.get("threshold_contract")
        if (
            not isinstance(threshold, Mapping)
            or threshold.get("fit_split") != "validation"
            or threshold.get("test_used_for_selection") is not False
            or threshold.get("int8_threshold_retuning_performed") is not False
            or not isinstance(threshold.get("deployed_logit_threshold"), (int, float))
            or not math.isfinite(float(threshold["deployed_logit_threshold"]))
        ):
            raise ValueError("verifier requires a frozen validation-only threshold")
    else:
        raise ValueError(f"unsupported firmware role: {role}")

    binding_subtrees = (
        (("source", metadata.get("source", {})),)
        if role == "detector"
        else (("inputs", metadata.get("inputs", {})),)
    )
    nested: list[dict[str, Any]] = []
    for subtree_name, subtree in (*binding_subtrees, ("provenance", metadata.get("provenance", {}))):
        nested.extend(
            _verify_nested_bindings(
                subtree,
                anchor=metadata_path.parent,
                label=f"{role}.{subtree_name}",
            )
        )
    return FirmwareArtifact(
        role=role,
        metadata_path=metadata_path,
        metadata_sha256=sha256_file(metadata_path),
        metadata=metadata,
        artifact_path=artifact_path,
        artifact_sha256=expected_hash,
        input_contract=input_contract,
        output_contract=output_contract,
        verified_bindings=tuple(nested),
    )


def _feature_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


@dataclass(frozen=True)
class SourceCorpus:
    manifest_path: Path
    manifest_sha256: str
    features_path: Path
    features_sha256: str
    rows: tuple[dict[str, Any], ...]
    features: np.ndarray


def load_source_corpus(manifest_path: Path, features_path: Path) -> SourceCorpus:
    manifest_path = manifest_path.expanduser().resolve()
    features_path = features_path.expanduser().resolve()
    payload = _load_object(manifest_path, "canonical source corpus")
    if payload.get("recipe") != "kizz_detector_scoring_corpus_v1":
        raise ValueError("unsupported canonical source corpus recipe")
    rows = payload.get("examples", payload.get("records"))
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("canonical source corpus requires nonempty examples")
    features_hash = sha256_file(features_path)
    declared = payload.get("array_sha256", {}).get(features_path.name)
    outputs = payload.get("outputs", {}).get("source_features", {})
    declared = declared if declared is not None else outputs.get("sha256")
    if declared != features_hash:
        raise ValueError("canonical source feature-array hash drift")
    _verify_nested_bindings(
        payload.get("inputs", {}),
        anchor=manifest_path.parent,
        label="source.inputs",
    )
    features = np.load(features_path, mmap_mode="r", allow_pickle=False)
    if features.shape != (len(rows), SOURCE_FRAMES, FEATURE_BINS):
        raise ValueError("canonical source features must be [N,260,40]")
    if not np.issubdtype(features.dtype, np.number) or not np.all(np.isfinite(features)):
        raise ValueError("canonical source features must be finite numeric values")
    seen_ids: set[str] = set()
    seen_indexes: set[int] = set()
    copied: list[dict[str, Any]] = []
    for ordinal, original in enumerate(rows):
        row = dict(original)
        source_id = row.get("source_id")
        index = row.get("feature_index", ordinal)
        if not isinstance(source_id, str) or not source_id or source_id in seen_ids:
            raise ValueError("canonical source_id values must be unique")
        if row.get("split") not in SPLITS or row.get("label") not in (0, 1, False, True):
            raise ValueError(f"{source_id}: invalid split/label")
        if not isinstance(index, int) or not 0 <= index < len(rows) or index in seen_indexes:
            raise ValueError(f"{source_id}: invalid feature index")
        observed = _feature_hash(np.asarray(features[index]))
        if row.get("feature_sha256") != observed:
            raise ValueError(f"{source_id}: source feature hash drift")
        row["feature_index"] = index
        audio_path = row.get("path")
        audio_hash = row.get("audio_sha256", row.get("sha256"))
        if audio_path is not None or audio_hash is not None:
            if audio_path is None or audio_hash is None:
                raise ValueError(f"{source_id}: incomplete source-audio binding")
            _verify_binding(
                {"path": audio_path, "sha256": audio_hash},
                anchor=manifest_path.parent,
                label=f"source {source_id} audio",
            )
        copied.append(row)
        seen_ids.add(source_id)
        seen_indexes.add(index)
    return SourceCorpus(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        features_path=features_path,
        features_sha256=features_hash,
        rows=tuple(copied),
        features=features,
    )


@dataclass(frozen=True)
class DetectorTrace:
    path: Path
    sha256: str
    payload: dict[str, Any]
    rows: tuple[dict[str, Any], ...]
    event_count: int


def load_detector_trace(
    path: Path, corpus: SourceCorpus, detector: FirmwareArtifact
) -> DetectorTrace:
    path = path.expanduser().resolve()
    payload = _load_object(path, "detector traces")
    if (
        payload.get("schema_version") != 1
        or payload.get("recipe")
        != "kizz_control_ordered_state_deployed_int8_trace_v1"
        or payload.get("deployment_qualification") is not False
    ):
        raise ValueError("unsupported detector trace contract")
    _verify_binding(
        payload.get("source_manifest"),
        anchor=path.parent,
        label="trace source manifest",
        expected_path=corpus.manifest_path,
    )
    _verify_binding(
        payload.get("source_features"),
        anchor=path.parent,
        label="trace source features",
        expected_path=corpus.features_path,
    )
    provenance = payload.get("detector")
    if not isinstance(provenance, Mapping):
        raise ValueError("detector traces lack detector provenance")
    _verify_binding(
        provenance.get("artifact"),
        anchor=path.parent,
        label="trace detector artifact",
        expected_path=detector.artifact_path,
    )
    _verify_binding(
        provenance.get("config"),
        anchor=path.parent,
        label="trace detector config",
        expected_path=detector.metadata_path,
    )
    threshold_binding = provenance.get("threshold")
    _verify_binding(
        threshold_binding,
        anchor=path.parent,
        label="trace detector threshold",
    )
    if not isinstance(threshold_binding, Mapping) or not isinstance(
        threshold_binding.get("value"), (int, float)
    ):
        raise ValueError("trace detector threshold value is missing")
    _verify_nested_bindings(
        provenance, anchor=path.parent, label="trace.detector"
    )
    _verify_nested_bindings(
        payload.get("arrays", {}), anchor=path.parent, label="trace.arrays"
    )
    rows = payload.get("examples", payload.get("records"))
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("detector traces require examples")
    source_by_id = {str(row["source_id"]): row for row in corpus.rows}
    traced: list[dict[str, Any]] = []
    seen: set[str] = set()
    event_count = 0
    for trace in rows:
        source_id = trace.get("source_id")
        if not isinstance(source_id, str) or source_id in seen or source_id not in source_by_id:
            raise ValueError("detector trace source identity drift")
        source = source_by_id[source_id]
        expected = {
            "feature_index": source["feature_index"],
            "split": source["split"],
            "label": int(source["label"]),
            "source_feature_sha256": source["feature_sha256"],
        }
        for key, value in expected.items():
            if trace.get(key) != value:
                raise ValueError(f"{source_id}: detector trace {key} drift")
        events = trace.get("events")
        if not isinstance(events, list):
            raise ValueError(f"{source_id}: detector trace events are missing")
        for event in events:
            if (
                not isinstance(event, Mapping)
                or not isinstance(event.get("score_frame_index"), int)
                or not isinstance(event.get("score"), (int, float))
                or not math.isfinite(float(event["score"]))
            ):
                raise ValueError(f"{source_id}: malformed detector event")
        event_count += len(events)
        traced.append(dict(trace))
        seen.add(source_id)
    if seen != set(source_by_id):
        raise ValueError("detector traces do not cover the canonical source corpus")
    counts = payload.get("counts")
    if not isinstance(counts, Mapping):
        raise ValueError("detector trace count contract is missing")
    declared_events = counts.get("candidate_events", counts.get("threshold_region_events"))
    if declared_events != event_count:
        raise ValueError("detector trace candidate-event count drift")
    evaluation = payload.get("evaluation")
    if (
        not isinstance(evaluation, Mapping)
        or evaluation.get("threshold_frozen_before_test_scoring") is not True
        or evaluation.get("test_used_for_selection") is not False
    ):
        raise ValueError("detector trace violates frozen-threshold/test-leakage policy")
    return DetectorTrace(path, sha256_file(path), payload, tuple(traced), event_count)


@dataclass(frozen=True)
class CandidateDataset:
    corpus_path: Path
    corpus_sha256: str
    corpus: dict[str, Any]
    rows: tuple[dict[str, Any], ...]
    features: np.ndarray
    features_path: Path
    features_sha256: str


def load_candidate_dataset(
    root: Path,
    trace: DetectorTrace,
    corpus: SourceCorpus,
    detector: FirmwareArtifact,
    verifier: FirmwareArtifact,
) -> CandidateDataset:
    root = root.expanduser().resolve()
    corpus_path = root / "corpus.json"
    payload = _load_object(corpus_path, "candidate dataset")
    if (
        payload.get("schema_version") != 1
        or payload.get("recipe") != "kizz_control_candidate_conditioned_verifier_v1"
        or payload.get("candidate_condition") != "frozen_detector_trigger_only"
    ):
        raise ValueError("unsupported candidate dataset contract")
    corpus_hash = sha256_file(corpus_path)
    verifier_inputs = verifier.metadata.get("inputs")
    if not isinstance(verifier_inputs, Mapping):
        raise ValueError("verifier metadata lacks input bindings")
    _verify_binding(
        verifier_inputs.get("candidate_corpus"),
        anchor=verifier.metadata_path.parent,
        label="verifier candidate dataset",
        expected_path=corpus_path,
    )
    bindings = payload.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("candidate dataset lacks input bindings")
    _verify_binding(
        bindings.get("source_manifest"),
        anchor=corpus_path.parent,
        label="candidate source manifest",
        expected_path=corpus.manifest_path,
    )
    _verify_binding(
        bindings.get("source_features"),
        anchor=corpus_path.parent,
        label="candidate source features",
        expected_path=corpus.features_path,
    )
    _verify_binding(
        bindings.get("detector_traces"),
        anchor=corpus_path.parent,
        label="candidate detector traces",
        expected_path=trace.path,
    )
    detector_provenance = payload.get("detector")
    if not isinstance(detector_provenance, Mapping):
        raise ValueError("candidate dataset lacks detector provenance")
    _verify_binding(
        detector_provenance.get("artifact"),
        anchor=corpus_path.parent,
        label="candidate detector artifact",
        expected_path=detector.artifact_path,
    )
    _verify_binding(
        detector_provenance.get("config"),
        anchor=corpus_path.parent,
        label="candidate detector config",
        expected_path=detector.metadata_path,
    )
    _verify_nested_bindings(
        {"bindings": bindings, "detector": detector_provenance},
        anchor=corpus_path.parent,
        label="candidate",
    )

    hashes = payload.get("array_sha256")
    if not isinstance(hashes, Mapping) or "features.npy" not in hashes:
        raise ValueError("candidate dataset lacks feature-array binding")
    for name, expected in hashes.items():
        array_path = root / str(name)
        if sha256_file(array_path) != _sha256(expected, f"candidate {name} hash"):
            raise ValueError(f"candidate {name} hash drift")
    features_path = root / "features.npy"
    features = np.load(features_path, mmap_mode="r", allow_pickle=False)
    rows = payload.get("examples")
    if (
        not isinstance(rows, list)
        or not rows
        or features.shape != (len(rows), SOURCE_FRAMES, FEATURE_BINS)
    ):
        raise ValueError("candidate features must be [N,260,40]")
    if payload.get("counts", {}).get("selected_candidates") != len(rows):
        raise ValueError("candidate selected-example count drift")
    for index, row in enumerate(rows):
        if (
            not isinstance(row, Mapping)
            or row.get("feature_index") != index
            or row.get("detector_conditioned") is not True
        ):
            raise ValueError("candidate row order/conditioning drift")
        if row.get("candidate_feature_sha256") != _feature_hash(np.asarray(features[index])):
            raise ValueError(f"candidate {index} feature hash drift")
    return CandidateDataset(
        corpus_path,
        corpus_hash,
        payload,
        tuple(dict(row) for row in rows),
        features,
        features_path,
        sha256_file(features_path),
    )


def replay_verifier_windows(
    trace: DetectorTrace,
    corpus: SourceCorpus,
    candidates: CandidateDataset,
) -> tuple[np.ndarray, tuple[dict[str, Any], ...]]:
    """Reconstruct every real detector-triggered verifier invocation."""
    contract = candidates.corpus.get("window_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("candidate dataset lacks verifier window contract")
    pre = contract.get("pre_context_frames")
    trigger = contract.get("trigger_frames")
    post = contract.get("post_context_frames")
    if (
        not isinstance(pre, int)
        or pre < 0
        or trigger != 1
        or not isinstance(post, int)
        or post < 0
        or pre + trigger + post != SOURCE_FRAMES
        or contract.get("padding") != "zero"
    ):
        raise ValueError("candidate verifier window contract drift")
    source_by_id = {str(row["source_id"]): row for row in corpus.rows}
    geometry = trace.payload.get("detector", {}).get("score_geometry", {})
    stride = geometry.get("feature_stride_frames")
    offset = geometry.get("feature_offset_frames")
    if not isinstance(stride, int) or stride < 1 or not isinstance(offset, int) or offset < 0:
        raise ValueError("detector trace score geometry is invalid")

    windows: list[np.ndarray] = []
    invocations: list[dict[str, Any]] = []
    by_event: dict[tuple[str, int], np.ndarray] = {}
    for traced in trace.rows:
        source_id = str(traced["source_id"])
        source = source_by_id[source_id]
        values = np.asarray(corpus.features[int(source["feature_index"])], dtype=np.float32)
        for event_ordinal, event in enumerate(traced["events"]):
            score_frame = int(event["score_frame_index"])
            feature_frame = event.get("feature_frame_index")
            if feature_frame is None:
                feature_frame = offset + score_frame * stride
            if (
                not isinstance(feature_frame, int)
                or feature_frame < 0
                or feature_frame >= len(values)
            ):
                raise ValueError(f"{source_id}: detector event feature frame is invalid")
            start = feature_frame - pre
            stop = feature_frame + post + 1
            window = np.zeros((SOURCE_FRAMES, FEATURE_BINS), dtype=np.float32)
            source_start = max(0, start)
            source_stop = min(len(values), stop)
            if source_stop > source_start:
                destination_start = source_start - start
                window[
                    destination_start : destination_start + source_stop - source_start
                ] = values[source_start:source_stop]
            key = (source_id, score_frame)
            if key in by_event:
                raise ValueError(f"{source_id}: duplicate detector event frame")
            by_event[key] = window
            windows.append(window)
            invocations.append(
                {
                    "source_id": source_id,
                    "split": traced["split"],
                    "label": int(traced["label"]),
                    "event_ordinal": event_ordinal,
                    "score_frame_index": score_frame,
                    "feature_frame_index": feature_frame,
                }
            )
    if len(windows) != trace.event_count:
        raise ValueError("reconstructed verifier invocation count drift")
    matched_event_keys: set[tuple[str, int]] = set()
    for row in candidates.rows:
        parent = row.get("parent_source_id")
        score_frame = row.get("detector_score_frame_index")
        if not isinstance(parent, str) or not isinstance(score_frame, int):
            raise ValueError("candidate row lacks exact detector event identity")
        key = (parent, score_frame)
        expected = by_event.get(key)
        if expected is None:
            # A verifier training corpus may be a provenance-bound superset of the
            # current detector trace (for example, retained legacy hard negatives
            # or consumed physical positives).  Those supplemental rows are never
            # simulation invocations and must remain training-only.
            if row.get("split") != "train":
                raise ValueError("candidate row is not tied to a recorded detector event")
            continue
        if key in matched_event_keys:
            raise ValueError("candidate dataset duplicates a recorded detector event")
        selected = np.asarray(candidates.features[int(row["feature_index"])])
        if not np.array_equal(selected, expected.astype(selected.dtype)):
            raise ValueError("candidate feature differs from reconstructed event window")
        matched_event_keys.add(key)
    if matched_event_keys != set(by_event):
        raise ValueError("candidate dataset does not cover every recorded detector event")
    if not windows:
        raise ValueError("detector trace contains no verifier invocations")
    return np.stack(windows), tuple(invocations)


class InferenceRuntime(Protocol):
    tensor_bytes: int
    tensor_count: int

    def reset(self) -> None: ...

    def invoke(self, values: np.ndarray) -> np.ndarray: ...


class TFLiteRuntime:
    """One allocated TFLite interpreter with quantization at its input seam."""

    def __init__(self, artifact: Path, contract: FirmwareArtifact) -> None:
        try:
            import tensorflow as tf

            interpreter_type = tf.lite.Interpreter
        except ImportError:
            try:
                from tflite_runtime.interpreter import Interpreter as interpreter_type
            except ImportError as error:
                raise RuntimeError(
                    "TensorFlow or tflite-runtime is required for a real simulation"
                ) from error
        self._interpreter_type = interpreter_type
        self._artifact = artifact
        self._contract = contract
        self._create()

    def _create(self) -> None:
        self._runner = self._interpreter_type(model_path=str(self._artifact))
        self._runner.allocate_tensors()
        inputs = self._runner.get_input_details()
        outputs = self._runner.get_output_details()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("cascade models must expose exactly one input and output")
        self._input = inputs[0]
        self._output = outputs[0]
        observed_input = [int(value) for value in self._input["shape"]]
        observed_output = [int(value) for value in self._output["shape"]]
        if observed_input != self._contract.input_contract["shape"]:
            raise ValueError("runtime input tensor differs from firmware metadata")
        if observed_output != self._contract.output_contract["shape"]:
            raise ValueError("runtime output tensor differs from firmware metadata")
        if np.dtype(self._input["dtype"]).name != self._contract.input_contract["dtype"]:
            raise ValueError("runtime input dtype differs from firmware metadata")
        if np.dtype(self._output["dtype"]).name != self._contract.output_contract["dtype"]:
            raise ValueError("runtime output dtype differs from firmware metadata")
        for label, detail, expected in (
            ("input", self._input, self._contract.input_contract),
            ("output", self._output, self._contract.output_contract),
        ):
            observed_scale, observed_zero = detail.get("quantization", (0.0, 0))
            expected_scale, expected_zero = expected["quantization"]
            if not math.isclose(
                float(observed_scale), float(expected_scale), rel_tol=0.0, abs_tol=1e-12
            ) or int(observed_zero) != int(expected_zero):
                raise ValueError(
                    f"runtime {label} quantization differs from firmware metadata"
                )
        details = self._runner.get_tensor_details()
        self.tensor_count = len(details)
        self.tensor_bytes = sum(
            int(np.prod(detail.get("shape", ()))) * np.dtype(detail["dtype"]).itemsize
            for detail in details
            if all(int(value) >= 0 for value in detail.get("shape", ()))
        )

    def reset(self) -> None:
        reset = getattr(self._runner, "reset_all_variables", None)
        if callable(reset):
            reset()
        else:
            self._create()

    def invoke(self, values: np.ndarray) -> np.ndarray:
        expected_shape = tuple(self._contract.input_contract["shape"])
        sample = np.asarray(values, dtype=np.float32)
        if sample.shape == expected_shape[1:]:
            sample = sample[None, ...]
        if sample.shape != expected_shape:
            raise ValueError(f"runtime input shape {sample.shape} != {expected_shape}")
        scale, zero = self._input.get("quantization", (0.0, 0))
        if scale <= 0:
            raise ValueError("runtime input quantization is invalid")
        info = np.iinfo(self._input["dtype"])
        quantized = np.clip(np.rint(sample / scale + zero), info.min, info.max).astype(
            self._input["dtype"]
        )
        self._runner.set_tensor(self._input["index"], quantized)
        self._runner.invoke()
        return np.array(self._runner.get_tensor(self._output["index"]), copy=True)


def _default_runtime_factory(
    role: str, artifact: Path, contract: FirmwareArtifact
) -> InferenceRuntime:
    del role
    return TFLiteRuntime(artifact, contract)


def stream_chunks(features: np.ndarray, stride: int, phase_offset: int):
    values = np.asarray(features, dtype=np.float32)
    if values.shape != (SOURCE_FRAMES, FEATURE_BINS):
        raise ValueError("detector source feature shape drift")
    if not 0 <= phase_offset < stride:
        raise ValueError("invalid stream phase offset")
    if phase_offset:
        primer = np.zeros((stride, FEATURE_BINS), dtype=np.float32)
        primer[-phase_offset:] = values[:phase_offset]
        yield primer
    for offset in range(phase_offset, len(values) - stride + 1, stride):
        yield values[offset : offset + stride]


def _quantiles(durations_ns: Sequence[int]) -> dict[str, Any]:
    if not durations_ns:
        return {"count": 0, "total_ms": 0.0, "p50_ms": None, "p95_ms": None, "p99_ms": None, "max_ms": None}
    values = np.asarray(durations_ns, dtype=np.float64) / 1_000_000.0
    try:
        percentiles = np.percentile(values, [50, 95, 99], method="higher")
    except TypeError:  # NumPy < 1.22
        percentiles = np.percentile(values, [50, 95, 99], interpolation="higher")
    return {
        "count": len(values),
        "total_ms": float(np.sum(values)),
        "p50_ms": float(percentiles[0]),
        "p95_ms": float(percentiles[1]),
        "p99_ms": float(percentiles[2]),
        "max_ms": float(np.max(values)),
    }


def _rss() -> dict[str, Any]:
    if resource is None:
        return {"available": False, "max_rss_bytes": None, "source": None}
    try:
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (AttributeError, OSError):
        return {"available": False, "max_rss_bytes": None, "source": None}
    multiplier = 1 if sys.platform == "darwin" else 1024
    return {
        "available": True,
        "max_rss_bytes": int(raw * multiplier),
        "source": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
        "platform_units_normalized": "bytes",
    }


def _digest_update(digest: Any, role: str, index: int, output: np.ndarray) -> None:
    values = np.ascontiguousarray(output)
    digest.update(role.encode("ascii"))
    digest.update(index.to_bytes(8, "little"))
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())


def _warmup(
    runtime: InferenceRuntime,
    sample: np.ndarray,
    count: int,
) -> None:
    if count < 0:
        raise ValueError("warmup counts must be nonnegative")
    for _ in range(count):
        runtime.reset()
        runtime.invoke(sample)


def _artifact_report(contract: FirmwareArtifact, runtime: InferenceRuntime) -> dict[str, Any]:
    memory = contract.metadata.get(
        "static_memory_contract", contract.metadata.get("static_memory_audit", {})
    )
    audit = memory.get("tensor_audit", {}) if isinstance(memory, Mapping) else {}
    return {
        "metadata": {
            "path": str(contract.metadata_path),
            "sha256": contract.metadata_sha256,
            "bytes": contract.metadata_path.stat().st_size,
        },
        "tflite": {
            "path": str(contract.artifact_path),
            "sha256": contract.artifact_sha256,
            "bytes": contract.artifact_path.stat().st_size,
        },
        "input_tensor": contract.input_contract,
        "output_tensor": contract.output_contract,
        "runtime_tensor_count": int(runtime.tensor_count),
        "runtime_declared_tensor_bytes_sum": int(runtime.tensor_bytes),
        "converter_tensor_audit": audit,
    }


def simulate_cascade(
    detector_metadata: Path,
    verifier_metadata: Path,
    source_manifest: Path,
    source_features: Path,
    detector_traces: Path,
    candidate_dataset: Path,
    *,
    warmup_detector_hops: int = 50,
    warmup_verifier_candidates: int = 10,
    runtime_factory: Callable[[str, Path, FirmwareArtifact], InferenceRuntime] | None = None,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    elapsed_clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, Any]:
    """Validate, replay, and benchmark the frozen cascade without threshold fitting."""
    detector = load_firmware_artifact(detector_metadata, "detector")
    verifier = load_firmware_artifact(verifier_metadata, "verifier")
    corpus = load_source_corpus(source_manifest, source_features)
    traces = load_detector_trace(detector_traces, corpus, detector)
    candidates = load_candidate_dataset(
        candidate_dataset, traces, corpus, detector, verifier
    )
    verifier_windows, verifier_invocations = replay_verifier_windows(
        traces, corpus, candidates
    )
    factory = runtime_factory or _default_runtime_factory
    detector_runtime = factory("detector", detector.artifact_path, detector)
    verifier_runtime = factory("verifier", verifier.artifact_path, verifier)

    timeline = detector.metadata["timeline"]
    stride = int(timeline["stream_input_frames_per_call"])
    phase = int(timeline["stream_phase_offset_frames"])
    first_source = np.asarray(corpus.features[int(corpus.rows[0]["feature_index"])])
    first_chunk = next(iter(stream_chunks(first_source, stride, phase)))
    first_candidate = np.asarray(verifier_windows[0], dtype=np.float32)
    if len(verifier.input_contract["shape"]) == 4:
        first_candidate = first_candidate[..., None]
    _warmup(detector_runtime, first_chunk, warmup_detector_hops)
    _warmup(verifier_runtime, first_candidate, warmup_verifier_candidates)

    detector_times: list[int] = []
    verifier_times: list[int] = []
    detector_digest = hashlib.sha256()
    verifier_digest = hashlib.sha256()
    detector_resets = 0
    verifier_resets = 0
    benchmark_start = elapsed_clock_ns()
    invocation_index = 0
    for row in corpus.rows:
        detector_runtime.reset()
        detector_resets += 1
        features = np.asarray(corpus.features[int(row["feature_index"])])
        for chunk in stream_chunks(features, stride, phase):
            start = clock_ns()
            output = detector_runtime.invoke(chunk)
            stop = clock_ns()
            if stop < start:
                raise ValueError("monotonic clock moved backwards")
            detector_times.append(stop - start)
            _digest_update(detector_digest, "detector", invocation_index, output)
            invocation_index += 1

    for index, feature in enumerate(verifier_windows):
        verifier_runtime.reset()
        verifier_resets += 1
        values = np.asarray(feature, dtype=np.float32)
        if len(verifier.input_contract["shape"]) == 4:
            values = values[..., None]
        start = clock_ns()
        output = verifier_runtime.invoke(values)
        stop = clock_ns()
        if stop < start:
            raise ValueError("monotonic clock moved backwards")
        verifier_times.append(stop - start)
        _digest_update(verifier_digest, "verifier", index, output)
    benchmark_stop = elapsed_clock_ns()
    if benchmark_stop < benchmark_start:
        raise ValueError("elapsed monotonic clock moved backwards")

    detector_latency = _quantiles(detector_times)
    verifier_latency = _quantiles(verifier_times)
    audio_seconds = sum(
        float(row.get("duration_seconds", SOURCE_FRAMES * FEATURE_HOP_SECONDS))
        for row in corpus.rows
    )
    if audio_seconds <= 0:
        raise ValueError("canonical corpus exposure must be positive")
    candidate_count = traces.event_count
    candidate_per_hop = candidate_count / len(detector_times)
    measured_inference_ms = detector_latency["total_ms"] + verifier_latency["total_ms"]
    e2e_seconds = (benchmark_stop - benchmark_start) / 1_000_000_000.0
    detector_p95 = float(detector_latency["p95_ms"] or 0.0)
    verifier_p95 = float(verifier_latency["p95_ms"] or 0.0)
    amortized_verifier_ms = candidate_per_hop * verifier_p95
    host_proxy_ms = detector_p95 + amortized_verifier_ms

    report = {
        "schema_version": 1,
        "benchmark": "kizz_control_int8_host_cascade_replay_v1",
        "deployment_qualification": False,
        "qualification_scope": "host_functional_replay_and_nontransferable_timing_only",
        "threshold_policy": {
            "selection_performed": False,
            "test_used_for_selection": False,
            "detector_events": "replayed_exactly_from_provenance-bound_trace",
            "verifier_threshold": {
                "value": float(
                    verifier.metadata["threshold_contract"]["deployed_logit_threshold"]
                ),
                "semantics": "deployed_scalar_logit",
                "fit_split": "validation",
                "test_used_for_selection": False,
            },
        },
        "bindings": {
            "source_manifest": {
                "path": str(corpus.manifest_path),
                "sha256": corpus.manifest_sha256,
            },
            "source_features": {
                "path": str(corpus.features_path),
                "sha256": corpus.features_sha256,
            },
            "detector_traces": {"path": str(traces.path), "sha256": traces.sha256},
            "candidate_dataset": {
                "path": str(candidates.corpus_path),
                "sha256": candidates.corpus_sha256,
            },
        },
        "artifacts": {
            "detector": _artifact_report(detector, detector_runtime),
            "verifier": _artifact_report(verifier, verifier_runtime),
        },
        "functional_replay": {
            "source_examples": len(corpus.rows),
            "detector_state_resets": detector_resets,
            "detector_hops": len(detector_times),
            "recorded_detector_candidates": candidate_count,
            "verifier_candidate_invocations": len(verifier_invocations),
            "candidate_dataset_selected_examples": len(candidates.rows),
            "candidate_dataset_supplemental_training_examples": (
                len(candidates.rows) - len(verifier_invocations)
            ),
            "verifier_state_resets": verifier_resets,
            "detector_raw_output_sha256": detector_digest.hexdigest(),
            "verifier_raw_output_sha256": verifier_digest.hexdigest(),
            "functional_outputs_deterministic_for_fixed_runtime_and_inputs": True,
        },
        "warmup": {
            "excluded_from_all_latency_and_throughput_measurements": True,
            "detector_hops": warmup_detector_hops,
            "verifier_candidates": warmup_verifier_candidates,
        },
        "host_timing": {
            "clock": "time.perf_counter_ns",
            "detector_hop": detector_latency,
            "verifier_candidate": verifier_latency,
            "benchmark_elapsed_seconds": e2e_seconds,
            "measured_inference_seconds": measured_inference_ms / 1000.0,
            "end_to_end_audio_replay_x_realtime": (
                audio_seconds / e2e_seconds if e2e_seconds > 0 else None
            ),
            "inference_only_audio_replay_x_realtime": (
                audio_seconds / (measured_inference_ms / 1000.0)
                if measured_inference_ms > 0
                else None
            ),
            "portable_to_esp32_s3": False,
            "warning": "Host TFLite timings are non-transferable to ESP32-S3/ESP-NN.",
        },
        "rates_and_duty_cycle": {
            "audio_exposure_seconds": audio_seconds,
            "recorded_candidates_per_second": candidate_count / audio_seconds,
            "recorded_candidates_per_hour": candidate_count * 3600.0 / audio_seconds,
            "candidate_probability_per_detector_hop": candidate_per_hop,
            "detector_host_inference_duty_fraction": (
                detector_latency["total_ms"] / 1000.0 / audio_seconds
            ),
            "verifier_host_inference_duty_fraction_on_benchmarked_candidates": (
                verifier_latency["total_ms"] / 1000.0 / audio_seconds
            ),
            "candidate_dataset_is_exact_benchmark_invocation_set": (
                candidate_count == len(candidates.rows)
            ),
        },
        "analytical_scheduler_budget": {
            "esp_hop_period_ms": ESP_HOP_MS,
            "measured_invocation_counts": {
                "detector_hops": len(detector_times),
                "recorded_detector_candidates": candidate_count,
                "benchmarked_verifier_candidates": len(verifier_invocations),
            },
            "host_p95_proxy": {
                "detector_ms_per_hop": detector_p95,
                "verifier_ms_per_candidate": verifier_p95,
                "amortized_verifier_ms_per_hop": amortized_verifier_ms,
                "combined_ms_per_hop": host_proxy_ms,
                "fraction_of_30ms_hop": host_proxy_ms / ESP_HOP_MS,
                "headroom_ms": ESP_HOP_MS - host_proxy_ms,
            },
            "interpretation": (
                "Invocation-count scheduler arithmetic only; host latency values are "
                "not estimates of ESP32-S3 latency."
            ),
            "hardware_qualification": False,
            "required_hardware_evidence": [
                "exact StackChan artifact flash and sustained boot",
                "on-device detector and verifier p50/p95/p99/max latency",
                "30 ms deadline misses and audio queue high-water",
                "tensor arena and heap high-water",
                "30-minute continuous soak",
            ],
        },
        "process_memory": _rss(),
        "limitations": [
            "host timings are non-transferable to ESP32-S3/ESP-NN",
            "no threshold selection or deployment qualification is performed",
            "candidate dataset may intentionally filter training hard negatives",
            "physical StackChan telemetry is required for every hardware claim",
        ],
    }
    return report


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(_canonical_bytes(payload))
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-metadata", type=Path, required=True)
    parser.add_argument("--verifier-metadata", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-features", type=Path, required=True)
    parser.add_argument("--detector-traces", type=Path, required=True)
    parser.add_argument("--candidate-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-detector-hops", type=int, default=50)
    parser.add_argument("--warmup-verifier-candidates", type=int, default=10)
    args = parser.parse_args(argv)
    report = simulate_cascade(
        args.detector_metadata,
        args.verifier_metadata,
        args.source_manifest,
        args.source_features,
        args.detector_traces,
        args.candidate_dataset,
        warmup_detector_hops=args.warmup_detector_hops,
        warmup_verifier_candidates=args.warmup_verifier_candidates,
    )
    _atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "detector_hop": report["host_timing"]["detector_hop"],
                "verifier_candidate": report["host_timing"]["verifier_candidate"],
                "deployment_qualification": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
