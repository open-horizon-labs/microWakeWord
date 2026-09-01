#!/usr/bin/env python3
"""Evaluate the deployed Kizz INT8 cascade on locked continuous negatives.

This evaluator is deliberately a one-way qualification pass.  It accepts only
the schema-v2, pre-scoring lock, uses already-frozen validation thresholds, and
never exposes locked-audio scores or candidates for threshold selection.  Work
is transactionally checkpointed by whole audio file and can be deterministically
sharded; the final merge revalidates the complete lock and rejects missing,
duplicate, or drifted shard evidence.

Host execution proves the model/frontend/cascade data path and continuous false
wake rate.  It does not prove ESP32-S3 latency, memory, thermal, power, or soak
stability; those remain separate physical-hardware evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np
import soundfile as sf

from microwakeword.kizz_continuous_evaluation import poisson_upper_95
from microwakeword.ordered_state import OrderedStateDecoder, OrderedStateTopology
if __package__:
    from tools.simulate_kizz_int8_cascade import (
        FirmwareArtifact,
        InferenceRuntime,
        TFLiteRuntime,
        load_firmware_artifact,
    )
    from tools.trace_kizz_ordered_state_detector import (
        _threshold_from_report,
        _validate_artifact,
    )
else:  # Direct ``python tools/evaluate_kizz_int8_continuous_cascade.py``.
    from simulate_kizz_int8_cascade import (  # type: ignore[no-redef]
        FirmwareArtifact,
        InferenceRuntime,
        TFLiteRuntime,
        load_firmware_artifact,
    )
    from trace_kizz_ordered_state_detector import (  # type: ignore[no-redef]
        _threshold_from_report,
        _validate_artifact,
    )


SCHEMA_VERSION = 1
LOCK_SCHEMA_VERSION = 2
LOCK_SCOPE = "locked_untouched_continuous_negative_corpus"
REPORT_KIND = "kizz_control_int8_continuous_negative_cascade_v1"
CHECKPOINT_KIND = "kizz_control_int8_continuous_negative_checkpoint_v1"
FEATURE_BINS = 40
FEATURE_SAMPLES = 160
SAMPLE_RATE = 16_000
MINIMUM_EXPOSURE_HOURS = 100.0
CONFIDENCE_LEVEL = 0.95


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(_canonical_bytes(value))
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


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


def _finite(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _binding(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _resolve(raw: object, anchor: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} path is required")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = anchor / path
    return path.resolve()


@dataclass(frozen=True)
class LockedAudio:
    source_id: str
    path: Path
    sha256: str
    duration_seconds: float
    category: str
    source: str
    manifest_row: dict[str, Any]


@dataclass(frozen=True)
class LockedManifest:
    path: Path
    sha256: str
    payload: dict[str, Any]
    rows: tuple[LockedAudio, ...]
    exposure_seconds: float


def load_locked_manifest(
    path: Path, *, minimum_exposure_hours: float = MINIMUM_EXPOSURE_HOURS
) -> LockedManifest:
    """Load and fully verify the immutable schema-v2 continuous-negative lock."""
    path = path.expanduser().resolve()
    payload = _load_object(path, "locked continuous-negative manifest")
    if (
        payload.get("schema_version") != LOCK_SCHEMA_VERSION
        or payload.get("gate_scope") != LOCK_SCOPE
        or payload.get("locked_before_scoring") is not True
        or payload.get("training_eligible") is not False
    ):
        raise ValueError("manifest is not the locked schema-v2 continuous-negative corpus")
    examples = payload.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("locked manifest must contain non-empty examples")

    rows: list[LockedAudio] = []
    source_ids: set[str] = set()
    hashes: set[str] = set()
    for index, raw in enumerate(examples):
        if not isinstance(raw, Mapping):
            raise ValueError(f"locked example {index} must be an object")
        source_id = str(raw.get("source_id") or "").strip()
        if not source_id or source_id in source_ids:
            raise ValueError("locked source IDs must be unique and non-empty")
        source_ids.add(source_id)
        expected_hash = _sha256(
            raw.get("audio_sha256", raw.get("sha256")), f"{source_id} audio"
        )
        if expected_hash in hashes:
            raise ValueError("locked manifest contains duplicate audio hashes")
        hashes.add(expected_hash)
        audio = _resolve(raw.get("path"), path.parent, f"{source_id} audio")
        if not audio.is_file():
            raise FileNotFoundError(audio)
        observed_hash = sha256_file(audio)
        if observed_hash != expected_hash:
            raise ValueError(
                f"{source_id} audio hash drift: expected {expected_hash}, got {observed_hash}"
            )
        duration = _finite(raw.get("duration_seconds"), f"{source_id} duration")
        if duration <= 0:
            raise ValueError(f"{source_id} duration must be positive")
        info = sf.info(audio)
        if info.samplerate != SAMPLE_RATE:
            raise ValueError(
                f"{source_id} sample rate {info.samplerate} is not the exact 16 kHz frontend contract"
            )
        if info.frames <= 0 or info.channels <= 0:
            raise ValueError(f"{source_id} audio metadata is invalid")
        observed_duration = info.frames / info.samplerate
        if not math.isclose(
            duration, observed_duration, rel_tol=0.0, abs_tol=1.0 / info.samplerate
        ):
            raise ValueError(
                f"{source_id} duration drift: declared {duration}, got {observed_duration}"
            )
        category = str(raw.get("category") or "").strip()
        source = str(raw.get("source") or "").strip()
        if not category or not source:
            raise ValueError(f"{source_id} requires category and source provenance")
        rows.append(
            LockedAudio(
                source_id,
                audio,
                expected_hash,
                observed_duration,
                category,
                source,
                dict(raw),
            )
        )

    exposure = math.fsum(row.duration_seconds for row in rows)
    counts = payload.get("counts")
    if not isinstance(counts, Mapping):
        raise ValueError("locked manifest counts are missing")
    if counts.get("files") != len(rows):
        raise ValueError("locked manifest file count drift")
    declared_seconds = _finite(counts.get("exposure_seconds"), "declared exposure")
    declared_hours = _finite(counts.get("exposure_hours"), "declared exposure hours")
    if not math.isclose(declared_seconds, exposure, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("locked manifest exposure-seconds drift")
    if not math.isclose(declared_hours, exposure / 3600.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("locked manifest exposure-hours drift")
    expected_categories = dict(Counter(row.category for row in rows))
    if counts.get("categories") != expected_categories:
        raise ValueError("locked manifest category counts drift")
    if exposure + 1e-9 < minimum_exposure_hours * 3600.0:
        raise ValueError(
            f"locked continuous-negative exposure is incomplete: "
            f"{exposure / 3600.0:.6f}h < {minimum_exposure_hours:.6f}h"
        )
    return LockedManifest(path, sha256_file(path), payload, tuple(rows), exposure)


@dataclass(frozen=True)
class EvaluationInputs:
    locked: LockedManifest
    detector_metadata: Path
    detector_model: Path
    detector: FirmwareArtifact
    topology: OrderedStateTopology
    detector_contract: dict[str, Any]
    detector_threshold_report: Path
    detector_threshold: float
    detector_threshold_provenance: dict[str, Any]
    verifier_metadata: Path
    verifier_model: Path
    verifier: FirmwareArtifact
    verifier_threshold: float
    verifier_threshold_report: Path | None
    ordered_verifier_metadata: Path | None
    ordered_verifier_model: Path | None
    ordered_verifier: FirmwareArtifact | None
    ordered_verifier_topology: OrderedStateTopology | None
    ordered_verifier_contract: dict[str, Any] | None
    ordered_verifier_threshold: float | None
    ordered_verifier_threshold_report: Path | None
    candidate_window: dict[str, int]
    bindings: dict[str, Any]
    run_compact_verifier: bool = True


def _same_binding(
    value: object, expected_path: Path, expected_hash: str, anchor: Path, label: str
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} binding is missing")
    path = _resolve(value.get("path"), anchor, label)
    digest = _sha256(value.get("sha256"), f"{label} hash")
    if path != expected_path.resolve() or digest != expected_hash:
        raise ValueError(f"{label} binding disagrees with the deployed artifact")


def _verifier_tensor_contract(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} tensor contract is missing")
    shape = value.get("shape")
    dtype = value.get("dtype")
    quantization = value.get("quantization")
    if isinstance(quantization, Mapping):
        scale = quantization.get("scale")
        zero = quantization.get("zero_point")
    elif isinstance(quantization, (list, tuple)) and len(quantization) == 2:
        scale, zero = quantization
    else:
        raise ValueError(f"{label} quantization contract is invalid")
    scale = _finite(scale, f"{label} quantization scale")
    dtype_limits = {
        "int8": (-128, 127),
        "int16": (-32768, 32767),
        "uint8": (0, 255),
    }
    limits = dtype_limits.get(str(dtype))
    if (
        not isinstance(shape, list)
        or not all(isinstance(item, int) and item > 0 for item in shape)
        or limits is None
        or scale <= 0
        or not isinstance(zero, int)
        or not limits[0] <= zero <= limits[1]
    ):
        raise ValueError(f"{label} tensor contract is invalid")
    return {"shape": list(shape), "dtype": dtype, "quantization": [scale, zero]}


def _verify_nested_file_bindings(value: object, anchor: Path, label: str) -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        has_path = "path" in value
        has_hash = "sha256" in value
        if has_path != has_hash:
            raise ValueError(f"{label} has an incomplete path/hash binding")
        if has_path:
            path = _resolve(value.get("path"), anchor, label)
            expected = _sha256(value.get("sha256"), f"{label} hash")
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"{label} hash drift")
            if "bytes" in value and value.get("bytes") != path.stat().st_size:
                raise ValueError(f"{label} byte-size drift")
            verified.append(
                {"label": label, "path": str(path), "sha256": expected, "bytes": path.stat().st_size}
            )
        for key, child in value.items():
            if key not in {"path", "sha256", "bytes"}:
                verified.extend(
                    _verify_nested_file_bindings(child, anchor, f"{label}.{key}")
                )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            verified.extend(
                _verify_nested_file_bindings(child, anchor, f"{label}[{index}]")
            )
    return verified


def _load_verifier_artifact(metadata_path: Path, model_path: Path) -> FirmwareArtifact:
    metadata = _load_object(metadata_path, "verifier firmware metadata")
    if (
        metadata.get("schema_version") != 1
        or metadata.get("kind") != "kizz_control_candidate_verifier_fixed_window_int8"
        or metadata.get("model_role") != "detector_conditioned_candidate_verifier"
        or metadata.get("candidate_conditioned") is not True
        or metadata.get("deployment_qualification") is not False
    ):
        raise ValueError("verifier firmware artifact kind/role drift")
    artifact = metadata.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ValueError("verifier artifact binding is missing")
    resolved_model = _resolve(
        artifact.get("path", artifact.get("filename")), metadata_path.parent, "verifier model"
    )
    expected_hash = _sha256(artifact.get("sha256"), "verifier artifact hash")
    if resolved_model != model_path or not model_path.is_file() or sha256_file(model_path) != expected_hash:
        raise ValueError("verifier TFLite artifact hash drift")
    if artifact.get("bytes") != model_path.stat().st_size:
        raise ValueError("verifier TFLite artifact byte-size drift")
    tensors = metadata.get("tensor_contracts")
    if not isinstance(tensors, Mapping) or not (
        tensors.get("fully_integer") is True or tensors.get("fully_int8") is True
    ):
        raise ValueError("verifier fully-integer tensor contract is missing")
    input_contract = _verifier_tensor_contract(tensors.get("input"), "verifier input")
    output_contract = _verifier_tensor_contract(tensors.get("output"), "verifier output")
    if input_contract["shape"] != [1, 260, 40, 1] or output_contract["shape"] != [1, 1]:
        raise ValueError("verifier fixed-window tensor shape drift")
    output_semantics = tensors.get("output_semantics")
    if output_semantics not in {
        "unnormalized_candidate_verifier_logit",
        "monotonic_bounded_candidate_verifier_logit",
    }:
        raise ValueError("verifier output semantics drift")
    if output_semantics == "monotonic_bounded_candidate_verifier_logit":
        model = metadata.get("model")
        transform = (
            model.get("deployment_logit_transform")
            if isinstance(model, Mapping)
            else None
        )
        threshold_contract = metadata.get("threshold_contract")
        if (
            not isinstance(transform, Mapping)
            or transform.get("kind") != "bound_times_tanh_logit_over_bound"
            or transform.get("monotonic") is not True
            or not isinstance(threshold_contract, Mapping)
        ):
            raise ValueError("verifier bounded-logit transform contract drift")
        transform_bound = _finite(
            transform.get("bound"), "verifier bounded-logit transform bound"
        )
        threshold_bound = _finite(
            threshold_contract.get("deployment_logit_bound"),
            "verifier threshold deployment logit bound",
        )
        if transform_bound <= 0 or not math.isclose(
            transform_bound, threshold_bound, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("verifier bounded-logit transform contract drift")
    verified = []
    for name in ("inputs", "provenance"):
        verified.extend(
            _verify_nested_file_bindings(
                metadata.get(name, {}), metadata_path.parent, f"verifier.{name}"
            )
        )
    return FirmwareArtifact(
        "verifier",
        metadata_path,
        sha256_file(metadata_path),
        metadata,
        model_path,
        expected_hash,
        input_contract,
        output_contract,
        tuple(verified),
    )


def _load_ordered_verifier_artifact(
    metadata_path: Path, model_path: Path
) -> tuple[FirmwareArtifact, OrderedStateTopology, dict[str, Any], float]:
    metadata, topology, contract = _validate_artifact(metadata_path, model_path)
    if (
        metadata.get("kind")
        != "kizz_control_ordered_state_candidate_verifier_streaming_int8"
        or metadata.get("student_role")
        != "detector_conditioned_ordered_state_candidate_verifier"
        or metadata.get("candidate_conditioned") is not True
        or metadata.get("deployment_qualification") is not False
    ):
        raise ValueError("ordered verifier artifact kind/role drift")
    artifact = metadata.get("artifact")
    tensors = metadata.get("tensor_contracts")
    if not isinstance(artifact, Mapping) or not isinstance(tensors, Mapping):
        raise ValueError("ordered verifier artifact/tensor binding is missing")
    resolved_model = _resolve(
        artifact.get("path", artifact.get("filename")),
        metadata_path.parent,
        "ordered verifier model",
    )
    expected_hash = _sha256(artifact.get("sha256"), "ordered verifier artifact hash")
    if (
        resolved_model != model_path
        or not model_path.is_file()
        or sha256_file(model_path) != expected_hash
        or artifact.get("bytes") != model_path.stat().st_size
    ):
        raise ValueError("ordered verifier TFLite artifact hash/size drift")
    input_contract = _verifier_tensor_contract(
        tensors.get("input"), "ordered verifier input"
    )
    output_contract = _verifier_tensor_contract(
        tensors.get("output"), "ordered verifier output"
    )
    if (
        input_contract["shape"] != [1, 3, 40]
        or input_contract["dtype"] != "int8"
        or output_contract["shape"] != [1, 1, topology.state_count]
        or output_contract["dtype"] != "uint8"
    ):
        raise ValueError("ordered verifier streaming tensor contract drift")
    decoder = metadata.get("decoder")
    provisional = (
        decoder.get("provisional_float_threshold")
        if isinstance(decoder, Mapping)
        else None
    )
    if (
        not isinstance(provisional, Mapping)
        or provisional.get("fit_split") != "validation"
        or provisional.get("test_used_for_selection") is not False
    ):
        raise ValueError("ordered verifier lacks a frozen validation threshold")
    threshold = _finite(provisional.get("threshold"), "ordered verifier threshold")
    return (
        FirmwareArtifact(
            "ordered_verifier",
            metadata_path,
            sha256_file(metadata_path),
            metadata,
            model_path,
            expected_hash,
            input_contract,
            output_contract,
            tuple(),
        ),
        topology,
        contract,
        threshold,
    )


def _validate_detector_threshold_binding(
    report_path: Path,
    detector_metadata: Path,
    detector_model: Path,
    detector_metadata_hash: str,
    detector_model_hash: str,
    detector_contract_hash: str,
) -> None:
    report = _load_object(report_path, "detector threshold report")
    report_kind = report.get("kind")
    if (
        report.get("schema_version") != 1
        or report_kind
        not in {
            "kizz_control_deployed_int8_validation_threshold",
            "kizz_control_recovered_deployed_int8_validation_threshold",
        }
        or report.get("deployment_qualification") is not False
    ):
        raise ValueError("detector threshold is not the frozen deployed-INT8 report")
    selection = report.get("selection")
    if not isinstance(selection, Mapping) or selection.get("qualified") is not True:
        raise ValueError("detector validation-only threshold did not qualify")
    bindings = report.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("detector threshold bindings are missing")
    _same_binding(
        bindings.get("artifact"),
        detector_model,
        detector_model_hash,
        report_path.parent,
        "detector threshold artifact",
    )
    _same_binding(
        bindings.get("config"),
        detector_metadata,
        detector_metadata_hash,
        report_path.parent,
        "detector threshold config",
    )
    if report.get("decoder_contract_sha256") != detector_contract_hash:
        raise ValueError("detector threshold decoder contract drift")
    if report_kind == "kizz_control_recovered_deployed_int8_validation_threshold":
        recovery = report.get("recovery")
        provenance_binding = (
            recovery.get("deployed_cascade_provenance")
            if isinstance(recovery, Mapping)
            else None
        )
        if (
            report.get("original_selection_report_recovered") is not False
            or not isinstance(provenance_binding, Mapping)
        ):
            raise ValueError("recovered detector threshold provenance is incomplete")
        provenance_path = _resolve(
            provenance_binding.get("path"),
            report_path.parent,
            "recovered deployed cascade provenance",
        )
        provenance_hash = _sha256(
            provenance_binding.get("sha256"),
            "recovered deployed cascade provenance hash",
        )
        if (
            not provenance_path.is_file()
            or sha256_file(provenance_path) != provenance_hash
        ):
            raise ValueError("recovered deployed cascade provenance drift")
        provenance = _load_object(
            provenance_path, "recovered deployed cascade provenance"
        )
        detector_record = (
            provenance.get("models", {}).get("detector")
            if isinstance(provenance.get("models"), Mapping)
            else None
        )
        if (
            not isinstance(detector_record, Mapping)
            or detector_record.get("sha256") != detector_model_hash
            or not math.isclose(
                _finite(detector_record.get("threshold"), "recovered detector threshold"),
                _finite(report.get("threshold"), "detector threshold report value"),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("recovered detector threshold disagrees with deployment provenance")


def _ordered_threshold_from_report(
    report_path: Path,
    metadata_path: Path,
    model_path: Path,
    metadata_hash: str,
    model_hash: str,
    decoder_contract_hash: str,
) -> float:
    report = _load_object(report_path, "ordered verifier threshold report")
    if (
        report.get("schema_version") != 1
        or report.get("kind")
        != "kizz_control_ordered_state_candidate_verifier_int8_validation_threshold"
        or report.get("deployment_qualification") is not False
        or report.get("test_scored_after_threshold_frozen") is not True
    ):
        raise ValueError("ordered verifier threshold is not the frozen deployed-INT8 report")
    selection = report.get("selection")
    if (
        not isinstance(selection, Mapping)
        or selection.get("qualified") is not True
        or selection.get("fit_split") != "validation"
        or selection.get("test_used_for_selection") is not False
        or selection.get("meets_minimum_recall") is not True
    ):
        raise ValueError("ordered verifier validation-only threshold did not qualify")
    bindings = report.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("ordered verifier threshold bindings are missing")
    _same_binding(
        bindings.get("artifact"), model_path, model_hash, report_path.parent,
        "ordered verifier threshold artifact",
    )
    _same_binding(
        bindings.get("config"), metadata_path, metadata_hash, report_path.parent,
        "ordered verifier threshold config",
    )
    if report.get("decoder_contract_sha256") != decoder_contract_hash:
        raise ValueError("ordered verifier threshold decoder contract drift")
    threshold = _finite(report.get("threshold"), "ordered verifier INT8 threshold")
    if not math.isclose(
        threshold,
        _finite(selection.get("threshold"), "ordered verifier selection threshold"),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("ordered verifier threshold report disagrees with selection")
    return threshold


def _verifier_candidate_window(
    verifier: FirmwareArtifact,
    detector_metadata: Path,
    detector_model: Path,
) -> dict[str, int]:
    inputs = verifier.metadata.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("verifier input provenance is missing")
    candidate_binding = inputs.get("candidate_corpus")
    if not isinstance(candidate_binding, Mapping):
        raise ValueError("verifier candidate-corpus binding is missing")
    candidate_path = _resolve(
        candidate_binding.get("path"), verifier.metadata_path.parent, "candidate corpus"
    )
    corpus = _load_object(candidate_path, "candidate corpus")
    if (
        corpus.get("schema_version") != 1
        or corpus.get("recipe") != "kizz_control_candidate_conditioned_verifier_v1"
        or corpus.get("candidate_condition") != "frozen_detector_trigger_only"
    ):
        raise ValueError("verifier candidate corpus contract drift")
    window = corpus.get("window_contract")
    if not isinstance(window, Mapping):
        raise ValueError("verifier candidate window contract is missing")
    pre = window.get("pre_context_frames")
    trigger = window.get("trigger_frames")
    post = window.get("post_context_frames")
    if (
        not isinstance(pre, int)
        or not isinstance(post, int)
        or pre < 0
        or post < 0
        or trigger != 1
        or window.get("padding") != "zero"
        or pre + trigger + post != 260
    ):
        raise ValueError("verifier fixed 260-frame candidate window contract drift")
    detector = corpus.get("detector")
    if not isinstance(detector, Mapping):
        raise ValueError("candidate corpus detector binding is missing")
    _same_binding(
        detector.get("artifact"),
        detector_model,
        sha256_file(detector_model),
        candidate_path.parent,
        "candidate corpus detector artifact",
    )
    _same_binding(
        detector.get("config", detector.get("metadata")),
        detector_metadata,
        sha256_file(detector_metadata),
        candidate_path.parent,
        "candidate corpus detector config",
    )
    return {"pre_context_frames": pre, "trigger_frames": trigger, "post_context_frames": post}


def _verifier_frozen_threshold(verifier: FirmwareArtifact) -> float:
    contract = verifier.metadata.get("threshold_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("verifier threshold contract is missing")
    threshold = _finite(
        contract.get("deployed_logit_threshold"), "verifier deployed logit threshold"
    )
    training_probability = _finite(
        contract.get("training_probability_threshold"),
        "verifier training probability threshold",
    )
    if not 0.0 < training_probability < 1.0:
        raise ValueError("verifier training probability threshold must be within (0,1)")
    training_logit = math.log(training_probability / (1.0 - training_probability))
    if not math.isclose(
        _finite(contract.get("training_logit_threshold"), "training logit threshold"),
        training_logit,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("verifier training probability/logit thresholds disagree")
    bound = contract.get("deployment_logit_bound")
    if bound is None:
        expected_deployed = training_logit
    else:
        bound = _finite(bound, "verifier deployment logit bound")
        if bound <= 0:
            raise ValueError("verifier deployment logit bound must be positive")
        expected_deployed = bound * math.tanh(training_logit / bound)
    quantization_margin = _finite(
        contract.get("quantization_logit_safety_margin", 0.0),
        "verifier quantization logit safety margin",
    )
    if quantization_margin < 0:
        raise ValueError("verifier quantization logit safety margin must be non-negative")
    expected_deployed -= quantization_margin
    if not math.isclose(threshold, expected_deployed, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("verifier deployed threshold transform disagrees")
    inputs = verifier.metadata.get("inputs")
    report_binding = inputs.get("training_report") if isinstance(inputs, Mapping) else None
    if not isinstance(report_binding, Mapping):
        raise ValueError("verifier training-report binding is missing")
    report_path = _resolve(
        report_binding.get("path"), verifier.metadata_path.parent, "verifier training report"
    )
    report = _load_object(report_path, "verifier training report")
    selection = report.get("selection_contract")
    winner = report.get("winner")
    if (
        not isinstance(selection, Mapping)
        or selection.get("selection_split") != "validation"
        or selection.get("test_used_for_selection") is not False
        or not isinstance(winner, Mapping)
        or not math.isclose(
            _finite(winner.get("frozen_threshold"), "training winner threshold"),
            training_probability,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("verifier frozen threshold is not validation-only provenance")
    equivalence = verifier.metadata.get("equivalence")
    if not isinstance(equivalence, Mapping) or equivalence.get("passed") is not True:
        raise ValueError("verifier INT8 equivalence evidence is missing or failed")
    audit = verifier.metadata.get("static_memory_audit")
    tensor_audit = audit.get("tensor_audit") if isinstance(audit, Mapping) else None
    if (
        not isinstance(audit, Mapping)
        or audit.get("fixed_input_shape") is not True
        or audit.get("fixed_output_shape") is not True
        or audit.get("dynamic_tensor_shapes_forbidden") is not True
        or not isinstance(tensor_audit, Mapping)
        or tensor_audit.get("dynamic_shape_tensor_count") != 0
        or tensor_audit.get("variable_tensor_count") != 0
    ):
        raise ValueError("verifier static-memory/stateless contract drift")
    return threshold


def _verifier_threshold_from_report(
    report_path: Path,
    metadata_path: Path,
    model_path: Path,
    metadata_hash: str,
    model_hash: str,
) -> float:
    report = _load_object(report_path, "verifier threshold report")
    report_kind = report.get("kind")
    if (
        report.get("schema_version") != 1
        or report_kind not in {
            "kizz_control_candidate_verifier_int8_joint_validation_threshold",
            "kizz_control_candidate_verifier_physical_recall_threshold",
        }
        or report.get("deployment_qualification") is not False
        or report.get("test_scored_after_threshold_frozen") is not True
    ):
        raise ValueError("verifier threshold is not the frozen deployed-INT8 report")
    selection = report.get("selection")
    expected_fit_split = (
        "physical_microphone_replay"
        if report_kind == "kizz_control_candidate_verifier_physical_recall_threshold"
        else "validation"
    )
    if (
        not isinstance(selection, Mapping)
        or selection.get("qualified") is not True
        or selection.get("fit_split") != expected_fit_split
        or selection.get("test_used_for_selection") is not False
        or selection.get("meets_minimum_recall") is not True
    ):
        raise ValueError("verifier validation-only threshold did not qualify")
    if report_kind == "kizz_control_candidate_verifier_physical_recall_threshold":
        detector_candidates = selection.get("detector_candidates")
        accepted_candidates = selection.get("accepted_candidates")
        if (
            report.get("locked_audio_used_for_tuning") is not False
            or not isinstance(detector_candidates, int)
            or detector_candidates <= 0
            or accepted_candidates != detector_candidates
        ):
            raise ValueError("physical verifier threshold evidence is incomplete")
    bindings = report.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("verifier threshold bindings are missing")
    _same_binding(
        bindings.get("artifact"), model_path, model_hash, report_path.parent,
        "verifier threshold artifact",
    )
    _same_binding(
        bindings.get("config"), metadata_path, metadata_hash, report_path.parent,
        "verifier threshold config",
    )
    threshold = _finite(report.get("threshold"), "verifier INT8 threshold")
    if not math.isclose(
        threshold,
        _finite(selection.get("threshold"), "verifier selection threshold"),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("verifier threshold report disagrees with selection")
    return threshold


def validate_inputs(
    locked_manifest: Path,
    detector_metadata: Path,
    detector_model: Path,
    detector_threshold_report: Path,
    verifier_metadata: Path,
    verifier_model: Path,
    ordered_verifier_metadata: Path | None = None,
    ordered_verifier_model: Path | None = None,
    *,
    verifier_threshold_report: Path | None = None,
    ordered_verifier_threshold_report: Path | None = None,
    run_compact_verifier: bool = True,
    minimum_exposure_hours: float = MINIMUM_EXPOSURE_HOURS,
) -> EvaluationInputs:
    locked = load_locked_manifest(
        locked_manifest, minimum_exposure_hours=minimum_exposure_hours
    )
    detector_metadata = detector_metadata.expanduser().resolve()
    detector_model = detector_model.expanduser().resolve()
    detector_threshold_report = detector_threshold_report.expanduser().resolve()
    verifier_metadata = verifier_metadata.expanduser().resolve()
    verifier_model = verifier_model.expanduser().resolve()
    if verifier_threshold_report is not None:
        verifier_threshold_report = verifier_threshold_report.expanduser().resolve()
    if (ordered_verifier_metadata is None) != (ordered_verifier_model is None):
        raise ValueError("ordered verifier metadata and model must be supplied together")
    if ordered_verifier_metadata is not None:
        ordered_verifier_metadata = ordered_verifier_metadata.expanduser().resolve()
        assert ordered_verifier_model is not None
        ordered_verifier_model = ordered_verifier_model.expanduser().resolve()
    if ordered_verifier_threshold_report is not None:
        if ordered_verifier_metadata is None:
            raise ValueError("ordered verifier threshold report requires an ordered verifier")
        ordered_verifier_threshold_report = (
            ordered_verifier_threshold_report.expanduser().resolve()
        )
    if not run_compact_verifier and ordered_verifier_metadata is None:
        raise ValueError("skipping the compact verifier requires an ordered verifier")

    detector_meta, topology, detector_contract = _validate_artifact(
        detector_metadata, detector_model
    )
    detector = load_firmware_artifact(detector_metadata, "detector")
    if detector.artifact_path != detector_model:
        raise ValueError("detector metadata resolved a different TFLite artifact")
    detector_threshold, threshold_provenance = _threshold_from_report(
        detector_threshold_report, topology
    )
    _validate_detector_threshold_binding(
        detector_threshold_report,
        detector_metadata,
        detector_model,
        detector.metadata_sha256,
        detector.artifact_sha256,
        str(detector_meta["decoder"]["contract_sha256"]),
    )

    verifier = _load_verifier_artifact(verifier_metadata, verifier_model)
    verifier_threshold = _verifier_frozen_threshold(verifier)
    if verifier_threshold_report is not None:
        verifier_threshold = _verifier_threshold_from_report(
            verifier_threshold_report,
            verifier_metadata,
            verifier_model,
            verifier.metadata_sha256,
            verifier.artifact_sha256,
        )
    candidate_window = _verifier_candidate_window(
        verifier, detector_metadata, detector_model
    )
    ordered_verifier = None
    ordered_topology = None
    ordered_contract = None
    ordered_threshold = None
    if ordered_verifier_metadata is not None and ordered_verifier_model is not None:
        (
            ordered_verifier,
            ordered_topology,
            ordered_contract,
            ordered_threshold,
        ) = _load_ordered_verifier_artifact(
            ordered_verifier_metadata, ordered_verifier_model
        )
        if ordered_verifier_threshold_report is not None:
            ordered_threshold = _ordered_threshold_from_report(
                ordered_verifier_threshold_report,
                ordered_verifier_metadata,
                ordered_verifier_model,
                ordered_verifier.metadata_sha256,
                ordered_verifier.artifact_sha256,
                str(ordered_verifier.metadata["decoder"]["contract_sha256"]),
            )
        timeline = ordered_verifier.metadata.get("timeline")
        if (
            not isinstance(timeline, Mapping)
            or timeline.get("offline_input_frames") != sum(candidate_window.values())
            or ordered_contract.get("calls") != 87
            or ordered_contract.get("warmup") != 21
            or ordered_contract.get("stride") != 3
            or ordered_contract.get("phase_offset") != 2
        ):
            raise ValueError("ordered verifier candidate timeline contract drift")
    bindings = {
        "locked_manifest": _binding(locked.path),
        "detector_metadata": _binding(detector_metadata),
        "detector_model": _binding(detector_model),
        "detector_threshold_report": _binding(detector_threshold_report),
        "verifier_metadata": _binding(verifier_metadata),
        "verifier_model": _binding(verifier_model),
    }
    if verifier_threshold_report is not None:
        bindings["verifier_threshold_report"] = _binding(verifier_threshold_report)
    if ordered_verifier_metadata is not None and ordered_verifier_model is not None:
        bindings["ordered_verifier_metadata"] = _binding(
            ordered_verifier_metadata
        )
        bindings["ordered_verifier_model"] = _binding(ordered_verifier_model)
        if ordered_verifier_threshold_report is not None:
            bindings["ordered_verifier_threshold_report"] = _binding(
                ordered_verifier_threshold_report
            )
    return EvaluationInputs(
        locked,
        detector_metadata,
        detector_model,
        detector,
        topology,
        detector_contract,
        detector_threshold_report,
        detector_threshold,
        threshold_provenance,
        verifier_metadata,
        verifier_model,
        verifier,
        verifier_threshold,
        verifier_threshold_report,
        ordered_verifier_metadata,
        ordered_verifier_model,
        ordered_verifier,
        ordered_topology,
        ordered_contract,
        ordered_threshold,
        ordered_verifier_threshold_report,
        candidate_window,
        bindings,
        run_compact_verifier,
    )


class CausalScorer(Protocol):
    def step(self, logits: np.ndarray) -> float: ...


class OrderedStateCausalScorer:
    """Current-time score from the exact artifact-bound ordered-state recurrence."""

    def __init__(self, topology: OrderedStateTopology, contract: Mapping[str, Any]):
        arguments = dict(contract["decoder_arguments"])
        self.decoder = OrderedStateDecoder(
            topology,
            completion_margin=math.inf,
            from_logits=bool(arguments["from_logits"]),
            state_evidence_floor=arguments["state_evidence_floor"],
            self_loop_probability=float(arguments["self_loop_probability"]),
            next_state_probability=float(arguments["next_state_probability"]),
        )

    def step(self, logits: np.ndarray) -> float:
        self.decoder.step(logits)
        return self.decoder.current_completion_score


def _default_runtime_factory(
    role: str, artifact: Path, contract: FirmwareArtifact
) -> InferenceRuntime:
    del role
    return TFLiteRuntime(artifact, contract)


def _default_scorer_factory(
    topology: OrderedStateTopology, contract: Mapping[str, Any]
) -> CausalScorer:
    return OrderedStateCausalScorer(topology, contract)


def stream_repository_frontend(path: Path) -> Iterable[np.ndarray]:
    """Yield exact repository MicroFrontend frames without loading a file at once."""
    from microwakeword.audio.audio_utils import MicroFrontend

    frontend = MicroFrontend()
    process = getattr(frontend, "process_samples", None) or frontend.ProcessSamples
    pending = bytearray()
    with sf.SoundFile(path) as audio:
        if audio.samplerate != SAMPLE_RATE:
            raise ValueError("continuous audio violates the 16 kHz frontend contract")
        while True:
            block = audio.read(FEATURE_SAMPLES, dtype="float32", always_2d=True)
            if not len(block):
                break
            mono = np.mean(np.asarray(block, dtype=np.float32), axis=1)
            pcm = np.clip(mono * 32768.0, -32768, 32767).astype("<i2")
            pending.extend(pcm.tobytes())
            while len(pending) >= FEATURE_SAMPLES * 2:
                result = process(bytes(pending[: FEATURE_SAMPLES * 2]))
                used = int(getattr(result, "samples_read", FEATURE_SAMPLES))
                if used <= 0 or used > FEATURE_SAMPLES:
                    raise ValueError("C MicroFrontend made invalid progress")
                del pending[: used * 2]
                if result.features:
                    values = np.asarray(result.features, dtype=np.float32)
                    if values.size % FEATURE_BINS:
                        raise ValueError("C MicroFrontend feature width drift")
                    for frame in values.reshape(-1, FEATURE_BINS):
                        if not np.all(np.isfinite(frame)):
                            raise ValueError("C MicroFrontend emitted non-finite features")
                        yield np.asarray(frame, dtype=np.float32)


def _dequantize(output: np.ndarray, artifact: FirmwareArtifact, label: str) -> np.ndarray:
    values = np.asarray(output)
    expected_shape = tuple(artifact.output_contract["shape"])
    expected_dtype = np.dtype(artifact.output_contract["dtype"])
    if values.shape != expected_shape or values.dtype != expected_dtype:
        raise ValueError(f"{label} runtime output violates its tensor contract")
    scale, zero = artifact.output_contract["quantization"]
    result = (values.astype(np.float32) - int(zero)) * float(scale)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} runtime output is non-finite after dequantization")
    return result


def _sigmoid(logit: float) -> float:
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)


@dataclass
class CandidateWindow:
    trigger_frame: int
    detector_score: float
    values: np.ndarray
    post_context_frames: int
    last_filled_frame: int

    @property
    def ready(self) -> bool:
        return self.last_filled_frame >= self.trigger_frame + self.post_context_frames

    def add(self, frame_index: int, frame: np.ndarray, pre_context_frames: int) -> None:
        if frame_index <= self.trigger_frame or frame_index > self.trigger_frame + self.post_context_frames:
            return
        position = frame_index - (self.trigger_frame - pre_context_frames)
        self.values[position] = frame
        self.last_filled_frame = max(self.last_filled_frame, frame_index)


def _snapshot_candidate(
    ring: Sequence[tuple[int, np.ndarray]],
    trigger_frame: int,
    detector_score: float,
    *,
    pre_context_frames: int,
    post_context_frames: int,
) -> CandidateWindow:
    values = np.zeros(
        (pre_context_frames + 1 + post_context_frames, FEATURE_BINS), dtype=np.float32
    )
    start = trigger_frame - pre_context_frames
    for frame_index, frame in ring:
        position = frame_index - start
        if 0 <= position <= pre_context_frames:
            values[position] = frame
    return CandidateWindow(
        trigger_frame,
        detector_score,
        values,
        post_context_frames,
        trigger_frame,
    )


def _score_ordered_verifier(
    values: np.ndarray,
    inputs: EvaluationInputs,
    runtime: InferenceRuntime,
) -> tuple[bool, float, int]:
    if (
        inputs.ordered_verifier is None
        or inputs.ordered_verifier_topology is None
        or inputs.ordered_verifier_contract is None
        or inputs.ordered_verifier_threshold is None
    ):
        raise ValueError("ordered verifier inputs are incomplete")
    runtime.reset()
    contract = inputs.ordered_verifier_contract
    stride = int(contract["stride"])
    phase = int(contract["phase_offset"])
    warmup = int(contract["warmup"])
    expected_calls = int(contract["calls"])
    scorer = OrderedStateCausalScorer(
        inputs.ordered_verifier_topology, contract
    )
    chunks: list[np.ndarray] = []
    if phase:
        primer = np.zeros((stride, FEATURE_BINS), dtype=np.float32)
        primer[-phase:] = values[:phase]
        chunks.append(primer)
    for offset in range(phase, len(values) - stride + 1, stride):
        chunks.append(np.asarray(values[offset : offset + stride]))
    if len(chunks) != expected_calls:
        raise ValueError("ordered verifier stream-call geometry drift")
    maximum = -math.inf
    for call, chunk in enumerate(chunks):
        raw = runtime.invoke(chunk)
        logits = _dequantize(
            raw, inputs.ordered_verifier, "ordered verifier"
        ).reshape(-1)
        if call < warmup:
            continue
        maximum = max(maximum, float(scorer.step(logits)))
        if maximum >= inputs.ordered_verifier_threshold:
            return True, maximum, call + 1
    return False, maximum, len(chunks)


def _record_hash(record: Mapping[str, Any]) -> str:
    value = dict(record)
    value.pop("record_sha256", None)
    return sha256_json(value)


def _score_file(
    row: LockedAudio,
    inputs: EvaluationInputs,
    detector_runtime: InferenceRuntime,
    verifier_runtime: InferenceRuntime | None,
    ordered_verifier_runtime: InferenceRuntime | None,
    feature_stream_factory: Callable[[Path], Iterable[np.ndarray]],
    scorer_factory: Callable[[OrderedStateTopology, Mapping[str, Any]], CausalScorer],
) -> dict[str, Any]:
    before_hash = sha256_file(row.path)
    if before_hash != row.sha256:
        raise ValueError(f"{row.source_id} audio changed before scoring")
    detector_runtime.reset()
    if verifier_runtime is not None:
        verifier_runtime.reset()
    if ordered_verifier_runtime is not None:
        ordered_verifier_runtime.reset()
    scorer = scorer_factory(inputs.topology, inputs.detector_contract)
    stride = int(inputs.detector_contract["stride"])
    phase = int(inputs.detector_contract["phase_offset"])
    warmup = int(inputs.detector_contract["warmup"])
    pre = inputs.candidate_window["pre_context_frames"]
    post = inputs.candidate_window["post_context_frames"]
    ring: deque[tuple[int, np.ndarray]] = deque(maxlen=pre + 1)
    group: list[np.ndarray] = []
    pending: list[CandidateWindow] = []
    active: CandidateWindow | None = None
    detector_hops = 0
    aligned_hops = 0
    feature_frames = 0
    candidates = 0
    accepted = 0
    verifier_invocations = 0
    compact_accepts = 0
    ordered_verifier_runs = 0
    ordered_verifier_invocations = 0

    def invoke_verifier(candidate: CandidateWindow) -> None:
        nonlocal accepted, verifier_invocations, compact_accepts
        nonlocal ordered_verifier_runs, ordered_verifier_invocations
        values = candidate.values
        if inputs.run_compact_verifier:
            if verifier_runtime is None:
                raise AssertionError("compact verifier runtime is missing")
            verifier_runtime.reset()
            compact_values = values
            if len(inputs.verifier.input_contract["shape"]) == 4:
                compact_values = compact_values[..., None]
            raw = verifier_runtime.invoke(compact_values)
            logit = float(
                _dequantize(raw, inputs.verifier, "verifier").reshape(-1)[0]
            )
            verifier_invocations += 1
            if logit < inputs.verifier_threshold:
                return
            compact_accepts += 1
        if inputs.ordered_verifier is None:
            accepted += 1
            return
        if ordered_verifier_runtime is None:
            raise AssertionError("ordered verifier runtime is missing")
        ordered_verifier_runs += 1
        ordered_accepted, _, calls = _score_ordered_verifier(
            candidate.values, inputs, ordered_verifier_runtime
        )
        ordered_verifier_invocations += calls
        if ordered_accepted:
            accepted += 1

    def finish_ready(*, eof: bool = False) -> None:
        nonlocal pending
        ready = pending if eof else [candidate for candidate in pending if candidate.ready]
        pending = [] if eof else [candidate for candidate in pending if not candidate.ready]
        for candidate in ready:
            invoke_verifier(candidate)

    def observe_feature(index: int, frame: np.ndarray) -> None:
        if active is not None:
            active.add(index, frame, pre)
        for candidate in pending:
            candidate.add(index, frame, pre)
        ring.append((index, frame.copy()))
        finish_ready()

    def observe_score(score: float, trigger_frame: int) -> None:
        nonlocal active, candidates
        above = math.isfinite(score) and score >= inputs.detector_threshold
        if above:
            if active is None or score > active.detector_score:
                active = _snapshot_candidate(
                    tuple(ring),
                    trigger_frame,
                    score,
                    pre_context_frames=pre,
                    post_context_frames=post,
                )
            return
        if active is not None:
            pending.append(active)
            candidates += 1
            active = None
            finish_ready()

    def invoke_detector(chunk: np.ndarray, trigger_frame: int) -> None:
        nonlocal detector_hops, aligned_hops
        raw = detector_runtime.invoke(chunk)
        detector_hops += 1
        logits = _dequantize(raw, inputs.detector, "detector").reshape(-1)
        if detector_hops <= warmup:
            return
        aligned_hops += 1
        observe_score(float(scorer.step(logits)), trigger_frame)

    for raw_frame in feature_stream_factory(row.path):
        frame = np.asarray(raw_frame, dtype=np.float32)
        if frame.shape != (FEATURE_BINS,) or not np.all(np.isfinite(frame)):
            raise ValueError(f"{row.source_id} frontend emitted an invalid feature frame")
        frame_index = feature_frames
        feature_frames += 1
        observe_feature(frame_index, frame)
        if phase and frame_index + 1 == phase:
            primer = np.zeros((stride, FEATURE_BINS), dtype=np.float32)
            primer[-phase:] = np.stack([value for _, value in ring][-phase:])
            invoke_detector(primer, frame_index)
        if frame_index >= phase:
            group.append(frame)
            if len(group) == stride:
                invoke_detector(np.stack(group), frame_index)
                group.clear()

    if active is not None:
        pending.append(active)
        candidates += 1
    finish_ready(eof=True)  # Unseen post-context is the bound zero-padding policy.
    if inputs.run_compact_verifier:
        if verifier_invocations != candidates:
            raise AssertionError(
                "every deduplicated detector candidate must reach the compact verifier"
            )
    elif ordered_verifier_runs != candidates:
        raise AssertionError(
            "every deduplicated detector candidate must reach the ordered verifier"
        )
    after_hash = sha256_file(row.path)
    if after_hash != before_hash:
        raise ValueError(f"{row.source_id} audio changed while it was being scored")
    record: dict[str, Any] = {
        "source_id": row.source_id,
        "path": str(row.path),
        "audio_sha256": row.sha256,
        "duration_seconds": row.duration_seconds,
        "category": row.category,
        "source": row.source,
        "frontend_feature_frames": feature_frames,
        "detector_hops": detector_hops,
        "detector_aligned_hops": aligned_hops,
        "detector_candidates": candidates,
        "verifier_invocations": verifier_invocations,
        "compact_verifier_accepts": compact_accepts,
        "ordered_verifier_runs": ordered_verifier_runs,
        "ordered_verifier_invocations": ordered_verifier_invocations,
        "accepted_false_wakes": accepted,
        "state_reset": {
            "frontend": "fresh MicroFrontend instance per file",
            "detector": "runtime reset before file",
            "ordered_state_decoder": "fresh recurrence per file",
            "verifier": (
                "runtime reset before file and each candidate"
                if inputs.run_compact_verifier
                else "not executed (ordered verifier is the sole candidate gate)"
            ),
            "ordered_verifier": (
                (
                    "runtime reset before file and each compact-gate pass"
                    if inputs.run_compact_verifier
                    else "runtime reset before file and each detector candidate"
                )
                if inputs.ordered_verifier is not None
                else "not configured"
            ),
        },
    }
    record["record_sha256"] = _record_hash(record)
    return record


def _shard_index(row: LockedAudio, shard_count: int) -> int:
    material = f"{row.source_id}\0{row.sha256}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % shard_count


def _expected_ids(rows: Sequence[LockedAudio]) -> list[str]:
    return sorted(row.source_id for row in rows)


def _evaluation_fingerprint(
    inputs: EvaluationInputs, *, acceptance_ceiling: float
) -> tuple[str, dict[str, Any]]:
    policy = {
        "report_kind": REPORT_KIND,
        "locked_manifest_sha256": inputs.locked.sha256,
        "bindings": inputs.bindings,
        "detector_threshold": inputs.detector_threshold,
        "detector_threshold_fit_split": "validation",
        "run_compact_verifier": inputs.run_compact_verifier,
        "verifier_logit_threshold": (
            inputs.verifier_threshold if inputs.run_compact_verifier else None
        ),
        "verifier_threshold_fit_split": (
            "validation" if inputs.run_compact_verifier else None
        ),
        "ordered_verifier_score_threshold": inputs.ordered_verifier_threshold,
        "ordered_verifier_threshold_fit_split": (
            "validation" if inputs.ordered_verifier is not None else None
        ),
        "frontend": {
            "implementation": "repository pymicro-features MicroFrontend",
            "sample_rate_hz": SAMPLE_RATE,
            "feature_hop_samples": FEATURE_SAMPLES,
            "feature_bins": FEATURE_BINS,
        },
        "detector_hop_seconds": 0.03,
        "candidate_window": inputs.candidate_window,
        "detector_event_policy": "one_peak_per_contiguous_threshold_region",
        "acceptance_ceiling_faph_upper_95": acceptance_ceiling,
        "minimum_exposure_hours": MINIMUM_EXPOSURE_HOURS,
        "evaluator": _binding(Path(__file__).resolve()),
    }
    return sha256_json(policy), policy


def _validate_checkpoint(
    path: Path,
    *,
    evaluation_fingerprint: str,
    shard_count: int,
    shard_index: int,
    assigned: Sequence[LockedAudio],
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = _load_object(path, "continuous-evaluation checkpoint")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("kind") != CHECKPOINT_KIND
        or payload.get("evaluation_fingerprint") != evaluation_fingerprint
        or payload.get("shard_count") != shard_count
        or payload.get("shard_index") != shard_index
    ):
        raise ValueError("resume checkpoint provenance drift")
    records = payload.get("completed_files")
    if not isinstance(records, list):
        raise ValueError("resume checkpoint completed-files ledger is invalid")
    allowed = {row.source_id: row for row in assigned}
    result: dict[str, dict[str, Any]] = {}
    for raw in records:
        if not isinstance(raw, Mapping):
            raise ValueError("resume checkpoint record is invalid")
        record = dict(raw)
        source_id = str(record.get("source_id") or "")
        if source_id not in allowed or source_id in result:
            raise ValueError("resume checkpoint contains duplicate or foreign source")
        source = allowed[source_id]
        if (
            record.get("audio_sha256") != source.sha256
            or record.get("path") != str(source.path)
            or record.get("record_sha256") != _record_hash(record)
        ):
            raise ValueError("resume checkpoint file evidence drift")
        result[source_id] = record
    return result


def _checkpoint_payload(
    evaluation_fingerprint: str,
    shard_count: int,
    shard_index: int,
    assigned: Sequence[LockedAudio],
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": CHECKPOINT_KIND,
        "evaluation_fingerprint": evaluation_fingerprint,
        "shard_count": shard_count,
        "shard_index": shard_index,
        "assigned_source_ids_sha256": sha256_json(_expected_ids(assigned)),
        "completed_files": [dict(records[key]) for key in sorted(records)],
        "complete": len(records) == len(assigned),
    }


def _poisson_interval(events: int, exposure_hours: float) -> dict[str, float]:
    if events < 0 or exposure_hours <= 0:
        raise ValueError("Poisson interval needs non-negative events and positive exposure")
    from scipy.stats import chi2

    alpha = 1.0 - CONFIDENCE_LEVEL
    lower_count = 0.0 if events == 0 else 0.5 * float(chi2.ppf(alpha / 2.0, 2 * events))
    upper_count = 0.5 * float(chi2.ppf(1.0 - alpha / 2.0, 2 * (events + 1)))
    return {
        "confidence_level": CONFIDENCE_LEVEL,
        "method": "exact_two_sided_poisson_chi_square",
        "lower_per_hour": lower_count / exposure_hours,
        "upper_per_hour": upper_count / exposure_hours,
        "one_sided_upper_95_per_hour": poisson_upper_95(events, exposure_hours),
    }


def _binomial_interval(successes: int, trials: int) -> dict[str, Any]:
    if successes < 0 or trials < 0 or successes > trials:
        raise ValueError("binomial interval counts are invalid")
    if trials == 0:
        return {
            "confidence_level": CONFIDENCE_LEVEL,
            "method": "exact_clopper_pearson",
            "lower": 0.0,
            "upper": 1.0,
            "trials": 0,
        }
    from scipy.stats import beta

    alpha = 1.0 - CONFIDENCE_LEVEL
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    return {
        "confidence_level": CONFIDENCE_LEVEL,
        "method": "exact_clopper_pearson",
        "lower": lower,
        "upper": upper,
        "trials": trials,
    }


def _breakdown(records: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record[field])].append(record)
    result: dict[str, Any] = {}
    for name in sorted(grouped):
        rows = grouped[name]
        seconds = math.fsum(float(row["duration_seconds"]) for row in rows)
        candidates = sum(int(row["detector_candidates"]) for row in rows)
        accepted = sum(int(row["accepted_false_wakes"]) for row in rows)
        result[name] = {
            "files": len(rows),
            "exposure_seconds": seconds,
            "exposure_hours": seconds / 3600.0,
            "detector_candidates": candidates,
            "detector_candidates_per_hour": candidates * 3600.0 / seconds,
            "accepted_false_wakes": accepted,
            "accepted_false_wakes_per_hour": accepted * 3600.0 / seconds,
        }
    return result


def _metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    seconds = math.fsum(float(row["duration_seconds"]) for row in records)
    if not records:
        return {
            "files": 0,
            "exposure_seconds": 0.0,
            "exposure_hours": 0.0,
            "detector_candidates": 0,
            "detector_candidates_per_hour": None,
            "detector_candidate_rate_confidence": None,
            "verifier_invocations": 0,
            "compact_verifier_accepts": 0,
            "ordered_verifier_runs": 0,
            "ordered_verifier_invocations": 0,
            "accepted_false_wakes": 0,
            "accepted_false_wakes_per_hour": None,
            "accepted_false_wake_rate_confidence": None,
            "verifier_acceptance_fraction": 0.0,
            "verifier_acceptance_fraction_confidence": _binomial_interval(0, 0),
            "by_family": {},
            "by_source": {},
        }
    if seconds <= 0:
        raise ValueError("completed continuous-negative exposure must be positive")
    hours = seconds / 3600.0
    candidates = sum(int(row["detector_candidates"]) for row in records)
    accepted = sum(int(row["accepted_false_wakes"]) for row in records)
    compact_accepts = sum(int(row.get("compact_verifier_accepts", 0)) for row in records)
    ordered_runs = sum(int(row.get("ordered_verifier_runs", 0)) for row in records)
    return {
        "files": len(records),
        "exposure_seconds": seconds,
        "exposure_hours": hours,
        "detector_candidates": candidates,
        "detector_candidates_per_hour": candidates / hours,
        "detector_candidate_rate_confidence": _poisson_interval(candidates, hours),
        "verifier_invocations": sum(int(row["verifier_invocations"]) for row in records),
        "compact_verifier_accepts": compact_accepts,
        "compact_verifier_acceptance_fraction": (
            compact_accepts / candidates if candidates else 0.0
        ),
        "ordered_verifier_runs": ordered_runs,
        "ordered_verifier_invocations": sum(
            int(row.get("ordered_verifier_invocations", 0)) for row in records
        ),
        "accepted_false_wakes": accepted,
        "accepted_false_wakes_per_hour": accepted / hours,
        "accepted_false_wake_rate_confidence": _poisson_interval(accepted, hours),
        "verifier_acceptance_fraction": accepted / candidates if candidates else 0.0,
        "verifier_acceptance_fraction_confidence": _binomial_interval(accepted, candidates),
        "by_family": _breakdown(records, "category"),
        "by_source": _breakdown(records, "source"),
    }


def _report(
    inputs: EvaluationInputs,
    evaluation_fingerprint: str,
    policy: Mapping[str, Any],
    shard_count: int,
    shard_index: int | None,
    assigned_ids: Sequence[str],
    records: Sequence[Mapping[str, Any]],
    *,
    acceptance_ceiling: float,
    complete: bool,
    merged_shards: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    metrics = _metrics(records)
    all_evidence = complete and len(records) == len(inputs.locked.rows)
    exposure_complete = (
        all_evidence
        and metrics["exposure_hours"] + 1e-12 >= MINIMUM_EXPOSURE_HOURS
        and math.isclose(
            metrics["exposure_seconds"],
            inputs.locked.exposure_seconds,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    )
    confidence = metrics["accepted_false_wake_rate_confidence"]
    upper = (
        confidence["one_sided_upper_95_per_hour"]
        if isinstance(confidence, Mapping)
        else None
    )
    ceiling_met = upper is not None and upper <= acceptance_ceiling
    qualified = exposure_complete and ceiling_met
    reasons = []
    if not all_evidence:
        reasons.append("not_all_locked_files_are_present")
    if not exposure_complete:
        reasons.append("locked_negative_exposure_below_complete_100h_gate")
    if not ceiling_met:
        reasons.append("accepted_false_wake_upper_95_exceeds_ceiling")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "deployment_qualification": qualified,
        "qualification_scope": "locked_continuous_negative_host_evidence_only",
        "failure_reasons": reasons,
        "evaluation_fingerprint": evaluation_fingerprint,
        "policy": dict(policy),
        "threshold_policy": {
            "selection_performed": False,
            "locked_audio_used_for_tuning": False,
            "detector": {
                "value": inputs.detector_threshold,
                "fit_split": "validation",
                "report": inputs.bindings["detector_threshold_report"],
            },
            "verifier": (
                {
                    "value": inputs.verifier_threshold,
                    "score_transform": "dequantized_raw_logit",
                    "fit_split": (
                        "physical_microphone_replay"
                        if inputs.verifier_threshold_report is not None
                        and _load_object(
                            inputs.verifier_threshold_report,
                            "verifier threshold report",
                        ).get("kind")
                        == "kizz_control_candidate_verifier_physical_recall_threshold"
                        else "validation"
                    ),
                    "test_used_for_selection": False,
                }
                if inputs.run_compact_verifier
                else None
            ),
            "ordered_verifier": (
                {
                    "value": inputs.ordered_verifier_threshold,
                    "score_transform": "maximum_complete_ordered_state_path",
                    "fit_split": "validation",
                    "test_used_for_selection": False,
                }
                if inputs.ordered_verifier is not None
                else None
            ),
        },
        "bindings": inputs.bindings,
        "lock": {
            "schema_version": LOCK_SCHEMA_VERSION,
            "locked_before_scoring": True,
            "expected_files": len(inputs.locked.rows),
            "expected_exposure_seconds": inputs.locked.exposure_seconds,
            "expected_source_ids_sha256": sha256_json(_expected_ids(inputs.locked.rows)),
        },
        "shard": {
            "count": shard_count,
            "index": shard_index,
            "assigned_source_ids_sha256": sha256_json(sorted(assigned_ids)),
            "complete": complete,
        },
        "merged_shards": list(merged_shards or []),
        "files": [dict(record) for record in sorted(records, key=lambda item: str(item["source_id"]))],
        "metrics": metrics,
        "acceptance": {
            "ceiling_faph_upper_95": acceptance_ceiling,
            "observed_faph_upper_95": upper,
            "ceiling_met": ceiling_met,
            "complete_100h_evidence": exposure_complete,
        },
        "physical_hardware_proof": {
            "required_for_final_StackChan_deployment": True,
            "present": False,
            "separate_from_host_continuous_negative_qualification": True,
            "remaining": [
                "exact StackChan ESP32-S3 latency and duty cycle",
                "tensor-arena and heap high-water",
                "audio-drop and queue telemetry",
                "thermal and power evidence",
                "30-minute physical soak",
            ],
        },
    }


def evaluate_shard(
    locked_manifest: Path,
    detector_metadata: Path,
    detector_model: Path,
    detector_threshold_report: Path,
    verifier_metadata: Path,
    verifier_model: Path,
    checkpoint: Path,
    output: Path,
    *,
    ordered_verifier_metadata: Path | None = None,
    ordered_verifier_model: Path | None = None,
    verifier_threshold_report: Path | None = None,
    ordered_verifier_threshold_report: Path | None = None,
    run_compact_verifier: bool = True,
    shard_count: int = 1,
    shard_index: int = 0,
    acceptance_ceiling: float = 0.1,
    runtime_factory: Callable[[str, Path, FirmwareArtifact], InferenceRuntime] | None = None,
    feature_stream_factory: Callable[[Path], Iterable[np.ndarray]] | None = None,
    scorer_factory: Callable[[OrderedStateTopology, Mapping[str, Any]], CausalScorer] | None = None,
    _minimum_exposure_hours: float = MINIMUM_EXPOSURE_HOURS,
) -> dict[str, Any]:
    """Evaluate one deterministic shard, resuming only exact whole-file evidence."""
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard index/count are invalid")
    if not math.isfinite(acceptance_ceiling) or acceptance_ceiling < 0:
        raise ValueError("acceptance ceiling must be finite and non-negative")
    inputs = validate_inputs(
        locked_manifest,
        detector_metadata,
        detector_model,
        detector_threshold_report,
        verifier_metadata,
        verifier_model,
        ordered_verifier_metadata,
        ordered_verifier_model,
        verifier_threshold_report=verifier_threshold_report,
        ordered_verifier_threshold_report=ordered_verifier_threshold_report,
        run_compact_verifier=run_compact_verifier,
        minimum_exposure_hours=_minimum_exposure_hours,
    )
    evaluation_fingerprint, policy = _evaluation_fingerprint(
        inputs, acceptance_ceiling=acceptance_ceiling
    )
    assigned = tuple(
        row for row in inputs.locked.rows if _shard_index(row, shard_count) == shard_index
    )
    checkpoint = checkpoint.expanduser().resolve()
    output = output.expanduser().resolve()
    records = _validate_checkpoint(
        checkpoint,
        evaluation_fingerprint=evaluation_fingerprint,
        shard_count=shard_count,
        shard_index=shard_index,
        assigned=assigned,
    )
    factory = runtime_factory or _default_runtime_factory
    detector_runtime = factory("detector", inputs.detector_model, inputs.detector)
    verifier_runtime = (
        factory("verifier", inputs.verifier_model, inputs.verifier)
        if inputs.run_compact_verifier
        else None
    )
    ordered_verifier_runtime = (
        factory(
            "ordered_verifier",
            inputs.ordered_verifier_model,
            inputs.ordered_verifier,
        )
        if inputs.ordered_verifier_model is not None
        and inputs.ordered_verifier is not None
        else None
    )
    feature_factory = feature_stream_factory or stream_repository_frontend
    score_factory = scorer_factory or _default_scorer_factory

    for row in sorted(assigned, key=lambda item: item.source_id):
        if row.source_id in records:
            continue
        record = _score_file(
            row,
            inputs,
            detector_runtime,
            verifier_runtime,
            ordered_verifier_runtime,
            feature_factory,
            score_factory,
        )
        records[row.source_id] = record
        _atomic_json(
            checkpoint,
            _checkpoint_payload(
                evaluation_fingerprint,
                shard_count,
                shard_index,
                assigned,
                records,
            ),
        )

    # Detect model/config/threshold drift that occurred during a long shard run.
    final_inputs = validate_inputs(
        locked_manifest,
        detector_metadata,
        detector_model,
        detector_threshold_report,
        verifier_metadata,
        verifier_model,
        ordered_verifier_metadata,
        ordered_verifier_model,
        verifier_threshold_report=verifier_threshold_report,
        ordered_verifier_threshold_report=ordered_verifier_threshold_report,
        run_compact_verifier=run_compact_verifier,
        minimum_exposure_hours=_minimum_exposure_hours,
    )
    final_fingerprint, _ = _evaluation_fingerprint(
        final_inputs, acceptance_ceiling=acceptance_ceiling
    )
    if final_fingerprint != evaluation_fingerprint:
        raise ValueError("evaluation inputs drifted while shard was running")
    complete = len(records) == len(assigned)
    _atomic_json(
        checkpoint,
        _checkpoint_payload(
            evaluation_fingerprint,
            shard_count,
            shard_index,
            assigned,
            records,
        ),
    )
    report = _report(
        inputs,
        evaluation_fingerprint,
        policy,
        shard_count,
        shard_index,
        _expected_ids(assigned),
        list(records.values()),
        acceptance_ceiling=acceptance_ceiling,
        complete=complete,
    )
    _atomic_json(output, report)
    return report


def merge_shards(
    shards: Sequence[Path],
    output: Path,
    *,
    _minimum_exposure_hours: float = MINIMUM_EXPOSURE_HOURS,
) -> dict[str, Any]:
    """Merge a complete deterministic shard set and reject any provenance gap."""
    if not shards:
        raise ValueError("at least one shard report is required")
    loaded: list[tuple[Path, str, dict[str, Any]]] = []
    for path in shards:
        resolved = path.expanduser().resolve()
        loaded.append((resolved, sha256_file(resolved), _load_object(resolved, "shard report")))
    first = loaded[0][2]
    if first.get("kind") != REPORT_KIND:
        raise ValueError("merge input is not a continuous cascade shard")
    fingerprint = str(first.get("evaluation_fingerprint") or "")
    policy = first.get("policy")
    bindings = first.get("bindings")
    shard_meta = first.get("shard")
    acceptance = first.get("acceptance")
    if not all(isinstance(value, Mapping) for value in (policy, bindings, shard_meta, acceptance)):
        raise ValueError("shard report provenance is incomplete")
    count = int(shard_meta["count"])
    if count < 1 or len(loaded) != count:
        raise ValueError("missing shard reports")
    indexes: set[int] = set()
    records: dict[str, dict[str, Any]] = {}
    shard_bindings: list[dict[str, Any]] = []
    for path, digest, report in loaded:
        meta = report.get("shard")
        if (
            report.get("schema_version") != SCHEMA_VERSION
            or report.get("kind") != REPORT_KIND
            or report.get("evaluation_fingerprint") != fingerprint
            or report.get("policy") != policy
            or report.get("bindings") != bindings
            or not isinstance(meta, Mapping)
            or meta.get("count") != count
            or meta.get("complete") is not True
        ):
            raise ValueError("drifted or incomplete shard report")
        index = meta.get("index")
        if not isinstance(index, int) or not 0 <= index < count or index in indexes:
            raise ValueError("duplicate or invalid shard index")
        indexes.add(index)
        file_rows = report.get("files")
        if not isinstance(file_rows, list):
            raise ValueError("shard file ledger is missing")
        for raw in file_rows:
            if not isinstance(raw, Mapping):
                raise ValueError("shard file evidence is invalid")
            record = dict(raw)
            source_id = str(record.get("source_id") or "")
            if source_id in records:
                raise ValueError("duplicate source evidence across shards")
            if record.get("record_sha256") != _record_hash(record):
                raise ValueError("shard file evidence hash drift")
            records[source_id] = record
        shard_bindings.append({"path": str(path), "sha256": digest, "index": index})
    if indexes != set(range(count)):
        raise ValueError("missing shard indexes")

    manifest_binding = bindings.get("locked_manifest")
    if not isinstance(manifest_binding, Mapping):
        raise ValueError("locked manifest binding is missing from shards")
    manifest_path = _resolve(manifest_binding.get("path"), Path.cwd(), "locked manifest")
    if sha256_file(manifest_path) != manifest_binding.get("sha256"):
        raise ValueError("locked manifest changed before shard merge")
    # Reconstruct all strict inputs from the common path bindings.
    def bound_path(name: str) -> Path:
        value = bindings.get(name)
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} binding is missing")
        return _resolve(value.get("path"), Path.cwd(), name)

    inputs = validate_inputs(
        manifest_path,
        bound_path("detector_metadata"),
        bound_path("detector_model"),
        bound_path("detector_threshold_report"),
        bound_path("verifier_metadata"),
        bound_path("verifier_model"),
        (
            bound_path("ordered_verifier_metadata")
            if "ordered_verifier_metadata" in bindings
            else None
        ),
        (
            bound_path("ordered_verifier_model")
            if "ordered_verifier_model" in bindings
            else None
        ),
        ordered_verifier_threshold_report=(
            bound_path("ordered_verifier_threshold_report")
            if "ordered_verifier_threshold_report" in bindings
            else None
        ),
        verifier_threshold_report=(
            bound_path("verifier_threshold_report")
            if "verifier_threshold_report" in bindings
            else None
        ),
        run_compact_verifier=bool(policy.get("run_compact_verifier", True)),
        minimum_exposure_hours=_minimum_exposure_hours,
    )
    ceiling = _finite(acceptance.get("ceiling_faph_upper_95"), "acceptance ceiling")
    expected_fingerprint, expected_policy = _evaluation_fingerprint(
        inputs, acceptance_ceiling=ceiling
    )
    if expected_fingerprint != fingerprint or expected_policy != policy:
        raise ValueError("shard evaluation provenance drift")
    expected_ids = set(_expected_ids(inputs.locked.rows))
    if set(records) != expected_ids:
        raise ValueError("final merge has missing or foreign locked files")
    for row in inputs.locked.rows:
        expected_index = _shard_index(row, count)
        matching = [
            report
            for _, _, report in loaded
            if report["shard"]["index"] == expected_index
        ][0]
        shard_ids = {str(item["source_id"]) for item in matching["files"]}
        if row.source_id not in shard_ids:
            raise ValueError("source was evaluated by the wrong deterministic shard")
    for _, _, report in loaded:
        ids = sorted(str(item["source_id"]) for item in report["files"])
        if report["shard"]["assigned_source_ids_sha256"] != sha256_json(ids):
            raise ValueError("shard assignment ledger drift")

    merged = _report(
        inputs,
        fingerprint,
        policy,
        count,
        None,
        _expected_ids(inputs.locked.rows),
        list(records.values()),
        acceptance_ceiling=ceiling,
        complete=True,
        merged_shards=sorted(shard_bindings, key=lambda item: int(item["index"])),
    )
    _atomic_json(output, merged)
    return merged


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merge-shards", type=Path, nargs="+")
    parser.add_argument("--locked-manifest", type=Path)
    parser.add_argument("--detector-metadata", type=Path)
    parser.add_argument("--detector-model", type=Path)
    parser.add_argument("--detector-threshold-report", type=Path)
    parser.add_argument("--verifier-metadata", type=Path)
    parser.add_argument("--verifier-model", type=Path)
    parser.add_argument("--verifier-threshold-report", type=Path)
    parser.add_argument("--ordered-verifier-metadata", type=Path)
    parser.add_argument("--ordered-verifier-model", type=Path)
    parser.add_argument("--ordered-verifier-threshold-report", type=Path)
    parser.add_argument(
        "--skip-compact-verifier",
        action="store_true",
        help="evaluate detector candidates directly with the ordered verifier",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--acceptance-ceiling", type=float, default=0.1)
    args = parser.parse_args(argv)
    if args.merge_shards:
        forbidden = (
            args.locked_manifest,
            args.detector_metadata,
            args.detector_model,
            args.detector_threshold_report,
            args.verifier_metadata,
            args.verifier_model,
            args.verifier_threshold_report,
            args.ordered_verifier_metadata,
            args.ordered_verifier_model,
            args.ordered_verifier_threshold_report,
            args.checkpoint,
            True if args.skip_compact_verifier else None,
        )
        if any(value is not None for value in forbidden):
            parser.error("merge mode accepts only --merge-shards and --output")
        report = merge_shards(args.merge_shards, args.output)
    else:
        required = {
            "--locked-manifest": args.locked_manifest,
            "--detector-metadata": args.detector_metadata,
            "--detector-model": args.detector_model,
            "--detector-threshold-report": args.detector_threshold_report,
            "--verifier-metadata": args.verifier_metadata,
            "--verifier-model": args.verifier_model,
            "--checkpoint": args.checkpoint,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error(f"evaluation mode requires: {', '.join(missing)}")
        report = evaluate_shard(
            args.locked_manifest,
            args.detector_metadata,
            args.detector_model,
            args.detector_threshold_report,
            args.verifier_metadata,
            args.verifier_model,
            args.checkpoint,
            args.output,
            verifier_threshold_report=args.verifier_threshold_report,
            ordered_verifier_metadata=args.ordered_verifier_metadata,
            ordered_verifier_model=args.ordered_verifier_model,
            ordered_verifier_threshold_report=args.ordered_verifier_threshold_report,
            run_compact_verifier=not args.skip_compact_verifier,
            shard_count=args.shard_count,
            shard_index=args.shard_index,
            acceptance_ceiling=args.acceptance_ceiling,
        )
    print(json.dumps({"deployment_qualification": report["deployment_qualification"], "metrics": report["metrics"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
