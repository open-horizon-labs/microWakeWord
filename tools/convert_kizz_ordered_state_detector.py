#!/usr/bin/env python3
"""Convert a provenance-bound Kizz ordered-state detector to streaming INT8.

This is deliberately a detector-role conversion boundary, not a deployment
qualification boundary.  It accepts only the permissive candidate-generator
student and preserves ``deployment_qualification=false`` in the emitted
schema-v2 artifact metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from microwakeword.ordered_state import (
    OrderedStateTopology,
    ordered_state_sequence_score_numpy,
)
from microwakeword.wake_phrase import KIZZ_CONTROL


SCHEMA_VERSION = 2
STUDENT_ROLE = "permissive_detector_candidate_generator"
MODEL_KIND = "ordered_state_causal_student_distilled"
ARTIFACT_KIND = "kizz_control_ordered_state_detector_streaming_int8"
ARTIFACT_FILENAME = "kizz_control_detector_ordered_state_streaming_int8.tflite"
VERIFIER_ARTIFACT_KIND = (
    "kizz_control_ordered_state_candidate_verifier_streaming_int8"
)
VERIFIER_ARTIFACT_FILENAME = (
    "kizz_control_ordered_state_candidate_verifier_streaming_int8.tflite"
)
VERIFIER_STUDENT_ROLE = "detector_conditioned_ordered_state_candidate_verifier"
METADATA_FILENAME = "firmware-artifact.json"
INPUT_FRAMES = 260
FEATURE_BINS = 40
OUTPUT_FRAMES = 66
FEATURE_STEP_SECONDS = 0.010
FRONTEND_WINDOW_SECONDS = 0.030
EXPECTED_CACHE_FILES = (
    "features.npy",
    "targets.npy",
    "labels.npy",
    "teacher_logits.npy",
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ORDERED_STATE_MODULE = REPOSITORY_ROOT / "microwakeword/ordered_state.py"
ORDERED_STATE_MODEL_MODULE = REPOSITORY_ROOT / "microwakeword/ordered_state_model.py"
DISTILLATION_MODULE = REPOSITORY_ROOT / "tools/distill_kizz_student.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not readable JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _resolve_path(raw: object, anchor: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} has no path")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = anchor / path
    return path.resolve()


def _require_sha256(raw: object, label: str) -> str:
    if (
        not isinstance(raw, str)
        or len(raw) != 64
        or any(character not in "0123456789abcdef" for character in raw)
    ):
        raise ValueError(f"{label} has no canonical SHA-256")
    return raw


@dataclass(frozen=True)
class FileBinding:
    label: str
    path: Path
    sha256: str

    def verify(self) -> None:
        if not self.path.is_file():
            raise ValueError(f"{self.label} does not exist: {self.path}")
        actual = sha256_file(self.path)
        if actual != self.sha256:
            raise ValueError(
                f"{self.label} hash drift: expected {self.sha256}, got {actual}"
            )


def _binding(
    value: Mapping[str, Any], anchor: Path, label: str
) -> FileBinding:
    result = FileBinding(
        label=label,
        path=_resolve_path(value.get("path"), anchor, label),
        sha256=_require_sha256(value.get("sha256"), label),
    )
    result.verify()
    return result


def _walk_path_hash_bindings(
    value: object,
    anchor: Path,
    label: str,
) -> list[FileBinding]:
    """Verify every conventional ``{path, sha256}`` binding recursively."""
    results: list[FileBinding] = []
    if isinstance(value, Mapping):
        if "path" in value or "sha256" in value:
            if "path" not in value or "sha256" not in value:
                raise ValueError(f"{label} has an incomplete path/hash binding")
            results.append(_binding(value, anchor, label))
        for key, child in value.items():
            if key not in {"path", "sha256"}:
                results.extend(
                    _walk_path_hash_bindings(child, anchor, f"{label}.{key}")
                )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            results.extend(
                _walk_path_hash_bindings(child, anchor, f"{label}[{index}]")
            )
    return results


def _topology_payload(
    value: object,
    label: str,
    *,
    require_state_count: bool,
) -> OrderedStateTopology:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} topology is missing")
    phones = value.get("phones")
    states_per_phone = value.get("states_per_phone")
    if not isinstance(phones, list) or not all(
        isinstance(phone, str) and phone for phone in phones
    ):
        raise ValueError(f"{label} topology phones are invalid")
    try:
        topology = OrderedStateTopology(tuple(phones), int(states_per_phone))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} topology is invalid: {error}") from error
    if topology.phones != tuple(KIZZ_CONTROL.phones):
        raise ValueError(f"{label} topology is not Kizz Control")
    if topology.states_per_phone != 1:
        raise ValueError(f"{label} topology must use one ordered state per phone")
    if require_state_count and value.get("state_count") != topology.state_count:
        raise ValueError(f"{label} topology state_count drift")
    if "state_count" in value and value.get("state_count") != topology.state_count:
        raise ValueError(f"{label} topology state_count drift")
    if "state_names" in value and tuple(value.get("state_names", ())) != topology.state_names:
        raise ValueError(f"{label} topology state_names drift")
    return topology


def _require_same_topology(
    value: object, topology: OrderedStateTopology, label: str
) -> None:
    candidate = _topology_payload(value, label, require_state_count=False)
    if candidate != topology:
        raise ValueError(f"{label} topology differs from distillation topology")


def _selected_weight_bindings(
    metadata: Mapping[str, Any], anchor: Path
) -> list[FileBinding]:
    candidates: list[tuple[str, object]] = []
    if "selected_weights" in metadata:
        candidates.append(("distillation.selected_weights", metadata["selected_weights"]))
    student = metadata.get("student")
    if isinstance(student, Mapping) and (
        "weights" in student or "weights_sha256" in student
    ):
        candidates.append(
            (
                "distillation.student.selected_weights",
                {
                    "path": student.get("weights"),
                    "sha256": student.get("weights_sha256"),
                },
            )
        )
        if not isinstance(student.get("selected_checkpoint"), str) or not student.get(
            "selected_checkpoint"
        ):
            raise ValueError("distillation student has no selected checkpoint identity")
    if not candidates:
        raise ValueError("distillation metadata does not bind selected student weights")
    results = []
    for label, value in candidates:
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} is malformed")
        results.append(_binding(value, anchor, label))
    return results


def _verify_gate(
    gate_path: Path,
    expected_sha256: str,
    topology: OrderedStateTopology,
) -> tuple[dict[str, Any], list[FileBinding]]:
    FileBinding("detector teacher gate", gate_path, expected_sha256).verify()
    gate = _load_json(gate_path, "detector teacher gate")
    if (
        gate.get("gate_scope")
        != "teacher_detector_synthetic_bootstrap_prequalification"
        or gate.get("qualified") is not True
        or gate.get("eligible_for_detector_distillation") is not True
    ):
        raise ValueError("detector teacher gate does not permit detector distillation")
    if (
        gate.get("deployment_qualification") is not False
        or gate.get("eligible_for_final_deployment") is not False
    ):
        raise ValueError("detector teacher gate is not explicitly non-deployment")
    selection = gate.get("selection")
    if (
        not isinstance(selection, Mapping)
        or selection.get("split") != "validation"
        or float(selection.get("minimum_recall", 0.0)) < 0.95
        or float(selection.get("opportunity_recall", 0.0)) < 0.95
    ):
        raise ValueError("detector teacher gate recall contract drift")
    _require_same_topology(gate.get("topology"), topology, "detector gate")
    bindings = _walk_path_hash_bindings(gate, gate_path.parent, "detector_gate")
    checkpoint = gate.get("selected_checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("detector teacher gate has no selected checkpoint")
    best = FileBinding(
        "detector_gate.selected_checkpoint.best_weights",
        _resolve_path(
            checkpoint.get("best_weights_path"),
            gate_path.parent,
            "detector gate best weights",
        ),
        _require_sha256(
            checkpoint.get("best_weights_sha256"), "detector gate best weights"
        ),
    )
    best.verify()
    bindings.append(best)
    if "path" in checkpoint and "sha256" in checkpoint:
        selected = _binding(
            checkpoint, gate_path.parent, "detector_gate.selected_checkpoint"
        )
        if selected.sha256 != best.sha256:
            raise ValueError("detector gate selected and best checkpoint hashes differ")
    return gate, bindings


def _verify_cache(
    metadata: Mapping[str, Any],
    training_path: Path,
    topology: OrderedStateTopology,
) -> tuple[Path, dict[str, Any], list[FileBinding]]:
    prefix = _resolve_path(
        metadata.get("cache_prefix"), training_path.parent, "distillation cache_prefix"
    )
    cache_path = prefix.with_suffix(".json")
    cache = _load_json(cache_path, "detector distillation cache")
    if (
        cache.get("schema_version") != 2
        or cache.get("cache_role") != "detector_student_distillation"
        or cache.get("deployment_qualification") is not False
    ):
        raise ValueError("cache is not a non-deployment detector distillation cache")
    _require_same_topology(cache.get("topology"), topology, "detector cache")
    cache_hashes = metadata.get("cache_files_sha256")
    outputs = cache.get("outputs")
    if not isinstance(cache_hashes, Mapping) or set(cache_hashes) != set(
        EXPECTED_CACHE_FILES
    ):
        raise ValueError("distillation metadata does not bind every cache array")
    if not isinstance(outputs, Mapping) or set(outputs) != {
        name.removesuffix(".npy") for name in EXPECTED_CACHE_FILES
    }:
        raise ValueError("detector cache output bindings are incomplete")
    bindings: list[FileBinding] = []
    for filename in EXPECTED_CACHE_FILES:
        key = filename.removesuffix(".npy")
        expected = _require_sha256(
            cache_hashes.get(filename), f"distillation cache {filename}"
        )
        declared = outputs.get(key)
        if not isinstance(declared, Mapping):
            raise ValueError(f"detector cache has no {key} binding")
        output = _binding(declared, cache_path.parent, f"detector_cache.outputs.{key}")
        expected_path = prefix.with_name(filename).resolve()
        if output.path != expected_path or output.sha256 != expected:
            raise ValueError(f"detector cache {filename} binding differs from training")
        bindings.append(output)
    expected_cache_hash = _require_sha256(cache.get("cache_sha256"), "detector cache")
    if sha256_json(outputs) != expected_cache_hash:
        raise ValueError("detector cache output-binding digest drift")
    bindings.extend(
        _walk_path_hash_bindings(cache, cache_path.parent, "detector_cache")
    )
    return cache_path, cache, bindings


@dataclass(frozen=True)
class ValidatedInputs:
    training_path: Path
    training_sha256: str
    metadata: dict[str, Any]
    weights: FileBinding
    representative_features: FileBinding
    topology: OrderedStateTopology
    cache_path: Path
    cache_sha256: str
    cache: dict[str, Any]
    gate_path: Path
    gate_sha256: str
    gate: dict[str, Any]
    float_qualification_path: Path
    float_qualification_sha256: str
    float_qualification: dict[str, Any]
    provisional_float_threshold: float
    all_bindings: tuple[FileBinding, ...]

    def revalidate(self) -> None:
        if sha256_file(self.training_path) != self.training_sha256:
            raise ValueError("distillation-training metadata changed during conversion")
        if sha256_file(self.cache_path) != self.cache_sha256:
            raise ValueError("detector cache metadata changed during conversion")
        if (
            sha256_file(self.float_qualification_path)
            != self.float_qualification_sha256
        ):
            raise ValueError("float detector qualification changed during conversion")
        for binding in self.all_bindings:
            binding.verify()


def _verify_float_qualification(
    report_path: Path,
    training_path: Path,
    training_sha256: str,
    weights: FileBinding,
    topology: OrderedStateTopology,
    artifact_role: str = "detector",
) -> tuple[dict[str, Any], float, list[FileBinding]]:
    report = _load_json(report_path, "float detector qualification")
    detector_role = artifact_role == "detector"
    expected_evaluation = (
        "kizz_control_float_ordered_state_detector"
        if detector_role
        else "kizz_control_float_ordered_state_candidate_verifier"
    )
    qualification_key = (
        "qualified_for_detector_conversion"
        if detector_role
        else "qualified_for_candidate_verifier_conversion"
    )
    if (
        report.get("schema_version") != 1
        or report.get("evaluation") != expected_evaluation
        or report.get(qualification_key) is not True
        or report.get("deployment_qualification") is not False
        or report.get("failure_reasons") != []
    ):
        raise ValueError(
            "float detector report is not qualified for non-deployment conversion"
        )
    model = report.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("float detector report has no model binding")
    declared_training = _resolve_path(
        model.get("distillation_metadata"), report_path.parent, "float report model"
    )
    if (
        declared_training != training_path
        or _require_sha256(
            model.get("distillation_metadata_sha256"), "float report model"
        )
        != training_sha256
    ):
        raise ValueError("float detector report binds different distillation metadata")
    declared_weights = _resolve_path(
        model.get("weights"), report_path.parent, "float report weights"
    )
    if (
        declared_weights != weights.path
        or _require_sha256(model.get("weights_sha256"), "float report weights")
        != weights.sha256
    ):
        raise ValueError("float detector report binds different selected weights")
    _require_same_topology(
        report.get("topology"), topology, "float detector report"
    )
    selection = report.get("threshold_selection")
    if not isinstance(selection, Mapping):
        raise ValueError("float detector report has no threshold_selection")
    expected_recall = 0.95 if detector_role else 1.0
    expected_false_fraction = 0.20 if detector_role else 1.0
    try:
        selection_recall = float(selection.get("minimum_recall", -1.0))
        qualification_recall = float(
            selection.get("qualification_minimum_recall", selection_recall)
        )
    except (TypeError, ValueError) as error:
        raise ValueError("float detector recall contract is invalid") from error
    if (
        selection.get("fit_split") != "validation"
        or selection.get("test_used_for_selection") is not False
        or qualification_recall != expected_recall
        or not expected_recall <= selection_recall <= 1.0
        or float(selection.get("maximum_false_candidate_fraction", -1.0))
        != expected_false_fraction
    ):
        raise ValueError("float detector threshold-selection contract drift")
    try:
        threshold = float(selection["threshold"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("float detector report has no numeric threshold") from error
    if not math.isfinite(threshold):
        raise ValueError("float detector threshold must be finite")
    if (
        float(selection.get("opportunity_recall", 0.0)) < selection_recall
        or float(selection.get("false_candidate_fraction", math.inf))
        > expected_false_fraction
    ):
        raise ValueError("float detector validation metrics violate their gate")
    test = report.get("test")
    if (
        not isinstance(test, Mapping)
        or float(test.get("opportunity_recall", 0.0)) < expected_recall
        or float(test.get("false_candidate_fraction", math.inf))
        > expected_false_fraction
    ):
        raise ValueError("float detector held-out metrics violate their gate")
    bindings = _walk_path_hash_bindings(
        report, report_path.parent, "float_detector_qualification"
    )
    return report, threshold, bindings


def validate_inputs(
    training_path: Path,
    weights_path: Path,
    representative_features: Path,
    float_qualification: Path,
    artifact_role: str = "detector",
) -> ValidatedInputs:
    training_path = training_path.expanduser().resolve()
    weights_path = weights_path.expanduser().resolve()
    representative_features = representative_features.expanduser().resolve()
    float_qualification = float_qualification.expanduser().resolve()
    metadata = _load_json(training_path, "distillation-training metadata")
    if metadata.get("student_role") != STUDENT_ROLE:
        raise ValueError(f"distillation student_role must be {STUDENT_ROLE}")
    if metadata.get("deployment_qualification") is not False:
        raise ValueError("detector distillation must say deployment_qualification=false")
    if metadata.get("model") != MODEL_KIND:
        raise ValueError("distillation model kind is not the ordered-state student")
    if metadata.get("input_shape") != [INPUT_FRAMES, FEATURE_BINS]:
        raise ValueError("distillation input shape must be [260, 40]")
    topology = _topology_payload(
        metadata.get("topology"), "distillation", require_state_count=True
    )
    if metadata.get("output_shape") != [OUTPUT_FRAMES, topology.state_count]:
        raise ValueError("distillation output shape differs from ordered-state topology")

    selected_weights = _selected_weight_bindings(metadata, training_path.parent)
    allowed_weights = list(selected_weights)
    last_weights = metadata.get("last_weights")
    if isinstance(last_weights, Mapping):
        allowed_weights.append(
            _binding(last_weights, training_path.parent, "distillation.last_weights")
        )
    actual_weight_hash = sha256_file(weights_path) if weights_path.is_file() else None
    if not any(
        binding.path == weights_path and binding.sha256 == actual_weight_hash
        for binding in allowed_weights
    ):
        raise ValueError("weights input is not a metadata-bound checkpoint")
    weight_binding = FileBinding(
        "selected detector student weights", weights_path, str(actual_weight_hash)
    )
    weight_binding.verify()
    training_hash = sha256_file(training_path)

    cache_path, cache, cache_bindings = _verify_cache(
        metadata, training_path, topology
    )
    if artifact_role == "candidate-verifier":
        if (
            cache.get("cache_specialization")
            not in {
                "detector_conditioned_ordered_state_verifier_train_only_v1",
                "detector_conditioned_ordered_state_verifier_train_only_v2",
            }
            or cache.get("split_policy")
            != {
                "included": ["train"],
                "excluded": ["validation", "test"],
                "test_used_for_training": False,
            }
        ):
            raise ValueError("candidate verifier conversion requires a train-only specialized cache")
    gate_path = _resolve_path(
        metadata.get("detector_teacher_gate"),
        training_path.parent,
        "distillation detector_teacher_gate",
    )
    gate_hash = _require_sha256(
        metadata.get("detector_teacher_gate_sha256"),
        "distillation detector_teacher_gate",
    )
    gate, gate_bindings = _verify_gate(gate_path, gate_hash, topology)
    if metadata.get("teacher_qualification") is not None or metadata.get(
        "continuous_qualification"
    ) is not None:
        raise ValueError("detector distillation must not carry single-stage qualification")
    if metadata.get("teacher_qualification_sha256") is not None or metadata.get(
        "continuous_qualification_sha256"
    ) is not None:
        raise ValueError("detector distillation has stray single-stage hashes")

    selected_teacher = cache.get("selected_teacher", {}).get("best_weights", {})
    checkpoint = gate.get("selected_checkpoint", {})
    if (
        not isinstance(selected_teacher, Mapping)
        or selected_teacher.get("sha256") != checkpoint.get("best_weights_sha256")
    ):
        raise ValueError("detector cache and gate select different teacher weights")
    cache_training = cache.get("teacher_training")
    gate_training = gate.get("training_report")
    if (
        not isinstance(cache_training, Mapping)
        or not isinstance(gate_training, Mapping)
        or cache_training.get("sha256") != gate_training.get("sha256")
    ):
        raise ValueError("detector cache and gate bind different teacher training")

    representative_binding = FileBinding(
        "representative features",
        representative_features,
        sha256_file(representative_features)
        if representative_features.is_file()
        else "missing",
    )
    representative_binding.verify()
    qualification_report, provisional_threshold, qualification_bindings = (
        _verify_float_qualification(
            float_qualification,
            training_path,
            training_hash,
            weight_binding,
            topology,
            artifact_role,
        )
    )
    qualification_hash = sha256_file(float_qualification)
    all_bindings = [
        weight_binding,
        representative_binding,
        FileBinding("detector teacher gate", gate_path, gate_hash),
        FileBinding(
            "float detector qualification",
            float_qualification,
            qualification_hash,
        ),
        *allowed_weights,
        *cache_bindings,
        *gate_bindings,
        *qualification_bindings,
    ]
    # Deduplicate repeated recursive bindings while preserving deterministic order.
    unique: dict[tuple[str, str], FileBinding] = {}
    for binding in all_bindings:
        unique[(str(binding.path), binding.sha256)] = binding
    return ValidatedInputs(
        training_path=training_path,
        training_sha256=training_hash,
        metadata=metadata,
        weights=weight_binding,
        representative_features=representative_binding,
        topology=topology,
        cache_path=cache_path,
        cache_sha256=sha256_file(cache_path),
        cache=cache,
        gate_path=gate_path,
        gate_sha256=gate_hash,
        gate=gate,
        float_qualification_path=float_qualification,
        float_qualification_sha256=qualification_hash,
        float_qualification=qualification_report,
        provisional_float_threshold=provisional_threshold,
        all_bindings=tuple(unique.values()),
    )


def _spread_indices(length: int, count: int) -> np.ndarray:
    if length < 1 or count < 1:
        raise ValueError("sample length and count must be positive")
    return np.unique(np.linspace(0, length - 1, min(length, count), dtype=np.int64))


def _stream_input_chunks(
    features: np.ndarray, stride: int, phase_offset: int
):
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or stride < 1 or not 0 <= phase_offset < stride:
        raise ValueError("invalid streaming feature geometry")
    if phase_offset:
        if len(values) < phase_offset:
            raise ValueError("stream is shorter than its phase primer")
        primer = np.zeros((stride, values.shape[1]), dtype=np.float32)
        primer[-phase_offset:] = values[:phase_offset]
        yield primer
    for offset in range(phase_offset, len(values) - stride + 1, stride):
        yield values[offset : offset + stride]


def _representative_dataset(
    features: np.ndarray,
    indices: Sequence[int],
    stride: int,
    phase_offset: int,
):
    for index in indices:
        spectrogram = np.array(features[int(index)], dtype=np.float32, copy=True)
        if spectrogram.shape != (INPUT_FRAMES, FEATURE_BINS):
            raise ValueError("representative feature shape must be [260, 40]")
        if np.any(~np.isfinite(spectrogram)):
            raise ValueError("representative features contain non-finite values")
        # Preserve the calibration-extrema convention used by the original
        # 7724269 streaming converter.
        spectrogram[0, 0] = 0.0
        spectrogram[0, 1] = 26.0
        for chunk in _stream_input_chunks(spectrogram, stride, phase_offset):
            yield [chunk]


def _as_sequence(value: object, state_count: int, label: str) -> np.ndarray:
    values = np.asarray(value, dtype=np.float32)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2 or values.shape[-1] != state_count:
        raise ValueError(
            f"{label} output must be [time, {state_count}], got {values.shape}"
        )
    if np.any(~np.isfinite(values)):
        raise ValueError(f"{label} output contains non-finite logits")
    return values


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str
    quantization: tuple[float, int]

    def as_json(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "quantization": [float(self.quantization[0]), int(self.quantization[1])],
        }


@dataclass
class ConversionRuntime:
    artifact: bytes
    input_spec: TensorSpec
    output_spec: TensorSpec
    run_float: Callable[[np.ndarray], np.ndarray]
    run_streaming_float: Callable[[np.ndarray], np.ndarray]
    run_streaming_int8: Callable[[np.ndarray], np.ndarray]
    tensor_audit: dict[str, Any]
    model_parameters: int
    framework: dict[str, str]


class ConversionBackend(Protocol):
    def convert(
        self,
        *,
        flags: SimpleNamespace,
        topology: OrderedStateTopology,
        weights: Path,
        representative_features: np.ndarray,
        calibration_indices: np.ndarray,
        phase_offset: int,
        work_dir: Path,
    ) -> ConversionRuntime: ...


def _reset_keras_internal_state(model: object, tf: Any) -> None:
    flatten = getattr(model, "_flatten_layers", None)
    layers = (
        flatten(include_self=False, recursive=True)
        if callable(flatten)
        else getattr(model, "layers", ())
    )
    for layer in layers:
        state = getattr(layer, "states", None)
        if state is not None and hasattr(state, "assign"):
            state.assign(tf.zeros_like(state))


class TensorFlowConversionBackend:
    """Real Keras/SavedModel/TFLite conversion seam."""

    def convert(
        self,
        *,
        flags: SimpleNamespace,
        topology: OrderedStateTopology,
        weights: Path,
        representative_features: np.ndarray,
        calibration_indices: np.ndarray,
        phase_offset: int,
        work_dir: Path,
    ) -> ConversionRuntime:
        # TensorFlow remains lazy so provenance and focused seam tests do not pay
        # the conversion/runtime cost.
        import tensorflow as tf

        from microwakeword import utils
        from microwakeword.layers import modes
        from microwakeword.ordered_state_model import model as build_student

        offline_model = build_student(flags, (INPUT_FRAMES, FEATURE_BINS), None)
        offline_model.load_weights(weights)
        stream_source = build_student(flags, (INPUT_FRAMES, FEATURE_BINS), None)
        stream_source.load_weights(weights)
        config = {
            "train_dir": str(work_dir),
            "spectrogram_length": INPUT_FRAMES,
            "stride": int(flags.stride),
        }
        streaming_model = utils.convert_model_saved(
            stream_source,
            config,
            folder="stream_state_internal",
            mode=modes.Modes.STREAM_INTERNAL_STATE_INFERENCE,
        )
        saved_model = work_dir / "stream_state_internal"
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model))
        converter.optimizations = {tf.lite.Optimize.DEFAULT}
        converter._experimental_variable_quantization = True
        converter.target_spec.supported_ops = {
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        }
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.uint8
        converter.representative_dataset = tf.lite.RepresentativeDataset(
            lambda: _representative_dataset(
                representative_features,
                calibration_indices,
                int(flags.stride),
                phase_offset,
            )
        )
        artifact = converter.convert()
        interpreter = tf.lite.Interpreter(model_content=artifact)
        interpreter.allocate_tensors()
        inputs = interpreter.get_input_details()
        outputs = interpreter.get_output_details()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("streaming detector must expose exactly one input and output")

        def tensor_spec(detail: Mapping[str, Any]) -> TensorSpec:
            scale, zero = detail.get("quantization", (0.0, 0))
            return TensorSpec(
                tuple(int(value) for value in detail["shape"]),
                np.dtype(detail["dtype"]).name,
                (float(scale), int(zero)),
            )

        tensor_details = interpreter.get_tensor_details()
        tensor_bytes = 0
        dynamic = 0
        variable = 0
        for detail in tensor_details:
            shape = tuple(int(value) for value in detail.get("shape", ()))
            signature = tuple(
                int(value) for value in detail.get("shape_signature", shape)
            )
            if any(value < 0 for value in signature):
                dynamic += 1
            elif shape:
                tensor_bytes += int(np.prod(shape)) * np.dtype(detail["dtype"]).itemsize
            variable += int(bool(detail.get("is_variable", False)))

        def run_float(sample: np.ndarray) -> np.ndarray:
            return np.asarray(
                offline_model(sample[None, ...], training=False), dtype=np.float32
            )

        def run_streaming_float(sample: np.ndarray) -> np.ndarray:
            _reset_keras_internal_state(streaming_model, tf)
            emitted = []
            for chunk in _stream_input_chunks(
                sample, int(flags.stride), phase_offset
            ):
                emitted.extend(
                    _as_sequence(
                        streaming_model(chunk[None, ...], training=False),
                        topology.state_count,
                        "float streaming",
                    )
                )
            return np.asarray(emitted, dtype=np.float32)

        def run_streaming_int8(sample: np.ndarray) -> np.ndarray:
            runner = tf.lite.Interpreter(model_content=artifact)
            runner.allocate_tensors()
            input_detail = runner.get_input_details()[0]
            output_detail = runner.get_output_details()[0]
            input_scale, input_zero = input_detail["quantization"]
            output_scale, output_zero = output_detail["quantization"]
            if input_scale <= 0 or output_scale <= 0:
                raise ValueError("INT8 detector quantization scales must be positive")
            info = np.iinfo(input_detail["dtype"])
            emitted = []
            for chunk in _stream_input_chunks(
                sample, int(flags.stride), phase_offset
            ):
                quantized = np.clip(
                    np.rint(chunk / input_scale + input_zero), info.min, info.max
                ).astype(input_detail["dtype"])
                runner.set_tensor(input_detail["index"], quantized[None, ...])
                runner.invoke()
                raw = runner.get_tensor(output_detail["index"])
                logits = (np.asarray(raw, dtype=np.float32) - output_zero) * output_scale
                emitted.extend(
                    _as_sequence(logits, topology.state_count, "INT8 streaming")
                )
            return np.asarray(emitted, dtype=np.float32)

        return ConversionRuntime(
            artifact=artifact,
            input_spec=tensor_spec(inputs[0]),
            output_spec=tensor_spec(outputs[0]),
            run_float=run_float,
            run_streaming_float=run_streaming_float,
            run_streaming_int8=run_streaming_int8,
            tensor_audit={
                "tensor_count": len(tensor_details),
                "declared_tensor_bytes_sum": tensor_bytes,
                "dynamic_shape_tensor_count": dynamic,
                "variable_tensor_count": variable,
                "input_count": len(inputs),
                "output_count": len(outputs),
            },
            model_parameters=int(offline_model.count_params()),
            framework={"tensorflow": str(tf.__version__)},
        )


def _student_flags(
    state_count: int, architecture: str = "control_mixconv"
) -> SimpleNamespace:
    # Import the authoritative training flags rather than copying architecture
    # values into this converter.
    if __package__:
        from tools.distill_kizz_student import student_flags
    else:
        from distill_kizz_student import student_flags

    return student_flags(state_count, architecture)


def _flags_contract(flags: SimpleNamespace) -> dict[str, Any]:
    names = (
        "pointwise_filters",
        "residual_connection",
        "repeat_in_block",
        "mixconv_kernel_sizes",
        "first_conv_filters",
        "first_conv_kernel_size",
        "stride",
        "num_states",
    )
    return {name: getattr(flags, name) for name in names}


def _phase_offset(flags: SimpleNamespace) -> int:
    from microwakeword.phoneme_student import student_stream_phase_offset_frames

    return int(student_stream_phase_offset_frames(flags))


def _output_times(flags: SimpleNamespace) -> list[float]:
    from microwakeword.phoneme_student import student_output_times_seconds

    return [float(value) for value in student_output_times_seconds(flags, OUTPUT_FRAMES)]


def _validate_tensor_contracts(
    runtime: ConversionRuntime, topology: OrderedStateTopology, stride: int
) -> None:
    if runtime.input_spec.shape != (1, stride, FEATURE_BINS):
        raise ValueError("INT8 artifact input shape must be [1, 3, 40]")
    if runtime.input_spec.dtype != "int8":
        raise ValueError("INT8 artifact input dtype must be int8")
    if runtime.output_spec.shape != (1, 1, topology.state_count):
        raise ValueError(
            "INT8 artifact output shape must be [1, 1, state_count]"
        )
    if runtime.output_spec.dtype != "uint8":
        raise ValueError("INT8 artifact output dtype must be uint8")
    for label, spec in (
        ("input", runtime.input_spec),
        ("output", runtime.output_spec),
    ):
        scale = float(spec.quantization[0])
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"INT8 artifact {label} quantization scale is invalid")
    audit = runtime.tensor_audit
    if audit.get("input_count") != 1 or audit.get("output_count") != 1:
        raise ValueError("static-memory contract requires one fixed input and output")
    if int(audit.get("dynamic_shape_tensor_count", -1)) != 0:
        raise ValueError("static-memory contract forbids dynamic tensor shapes")


def _aligned_stream(
    value: object,
    topology: OrderedStateTopology,
    expected_calls: int,
    label: str,
) -> np.ndarray:
    sequence = _as_sequence(value, topology.state_count, label)
    if len(sequence) != expected_calls:
        raise ValueError(
            f"{label} emitted {len(sequence)} calls; expected {expected_calls}"
        )
    if expected_calls < OUTPUT_FRAMES:
        raise ValueError("stream emits fewer frames than the offline timeline")
    return sequence[expected_calls - OUTPUT_FRAMES :]


def _path_evidence(
    baseline: np.ndarray,
    candidate: np.ndarray,
    topology: OrderedStateTopology,
) -> dict[str, Any]:
    if candidate.shape != baseline.shape:
        raise ValueError("equivalence tensors have different shapes")
    delta = np.abs(baseline - candidate)
    baseline_score = float(
        ordered_state_sequence_score_numpy(baseline[None, ...], topology)[0]
    )
    score = float(
        ordered_state_sequence_score_numpy(candidate[None, ...], topology)[0]
    )
    if not math.isfinite(baseline_score) or not math.isfinite(score):
        raise ValueError("ordered-state equivalence score is non-finite")
    mismatches = int(
        np.count_nonzero(np.argmax(baseline, axis=-1) != np.argmax(candidate, axis=-1))
    )
    return {
        "max_logit_abs_error": float(np.max(delta)),
        "mean_logit_abs_error": float(np.mean(delta)),
        "frame_argmax_mismatch_count": mismatches,
        "frame_argmax_mismatch_rate": float(mismatches / len(baseline)),
        "baseline_sequence_score": baseline_score,
        "sequence_score": score,
        "sequence_score_abs_error": abs(baseline_score - score),
    }


def build_equivalence_evidence(
    runtime: ConversionRuntime,
    features: np.ndarray,
    indices: np.ndarray,
    topology: OrderedStateTopology,
    *,
    stride: int,
    phase_offset: int,
    limits: Mapping[str, float],
) -> dict[str, Any]:
    expected_calls = len(
        list(
            _stream_input_chunks(
                np.zeros((INPUT_FRAMES, FEATURE_BINS), dtype=np.float32),
                stride,
                phase_offset,
            )
        )
    )
    warmup = expected_calls - OUTPUT_FRAMES
    if warmup < 0:
        raise ValueError("streaming timeline has no complete offline tail")
    examples = []
    for raw_index in indices:
        index = int(raw_index)
        sample = np.asarray(features[index], dtype=np.float32)
        if sample.shape != (INPUT_FRAMES, FEATURE_BINS):
            raise ValueError("equivalence feature shape must be [260, 40]")
        if np.any(~np.isfinite(sample)):
            raise ValueError("equivalence features contain non-finite values")
        offline = _as_sequence(
            runtime.run_float(sample), topology.state_count, "float offline"
        )
        if offline.shape != (OUTPUT_FRAMES, topology.state_count):
            raise ValueError("float model output shape differs from training metadata")
        streaming = _aligned_stream(
            runtime.run_streaming_float(sample),
            topology,
            expected_calls,
            "float streaming",
        )
        quantized = _aligned_stream(
            runtime.run_streaming_int8(sample),
            topology,
            expected_calls,
            "INT8 streaming",
        )
        examples.append(
            {
                "representative_index": index,
                "float_streaming": _path_evidence(offline, streaming, topology),
                "int8_streaming": _path_evidence(offline, quantized, topology),
            }
        )
    if not examples:
        raise ValueError("equivalence evidence set is empty")

    paths: dict[str, Any] = {}
    for path_name in ("float_streaming", "int8_streaming"):
        rows = [example[path_name] for example in examples]
        paths[path_name] = {
            "max_logit_abs_error": max(row["max_logit_abs_error"] for row in rows),
            "mean_logit_abs_error": float(
                np.mean([row["mean_logit_abs_error"] for row in rows])
            ),
            "max_sequence_score_abs_error": max(
                row["sequence_score_abs_error"] for row in rows
            ),
            "frame_argmax_mismatch_count": sum(
                row["frame_argmax_mismatch_count"] for row in rows
            ),
            "frame_argmax_mismatch_rate": float(
                np.mean([row["frame_argmax_mismatch_rate"] for row in rows])
            ),
        }

    checks = {
        "float_streaming": {
            "max_logit_abs_error": limits["float_max_abs"],
            "mean_logit_abs_error": limits["float_mean_abs"],
            "max_sequence_score_abs_error": limits["float_score_abs"],
            "frame_argmax_mismatch_rate": limits["float_argmax_mismatch"],
        },
        "int8_streaming": {
            "max_logit_abs_error": limits["int8_max_abs"],
            "mean_logit_abs_error": limits["int8_mean_abs"],
            "max_sequence_score_abs_error": limits["int8_score_abs"],
            "frame_argmax_mismatch_rate": limits["int8_argmax_mismatch"],
        },
    }
    for path_name, maximums in checks.items():
        for metric, maximum in maximums.items():
            if paths[path_name][metric] > maximum:
                raise ValueError(
                    f"{path_name} failed {metric} equivalence: "
                    f"{paths[path_name][metric]} > {maximum}"
                )
    report = {
        "algorithm": "generic_ordered_state_sequence_score_numpy_v1",
        "from_logits": True,
        "example_count": len(examples),
        "representative_indices": [int(value) for value in indices],
        "offline_shape": [OUTPUT_FRAMES, topology.state_count],
        "streaming_calls_per_example": expected_calls,
        "streaming_warmup_outputs_discarded": warmup,
        "limits": dict(limits),
        "paths": paths,
        "examples": examples,
    }
    report["evidence_sha256"] = sha256_json(report)
    return report


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _limits(args: argparse.Namespace) -> dict[str, float]:
    result = {
        "float_max_abs": float(getattr(args, "float_max_abs", 1e-4)),
        "float_mean_abs": float(getattr(args, "float_mean_abs", 1e-5)),
        "float_score_abs": float(getattr(args, "float_score_abs", 1e-3)),
        "float_argmax_mismatch": float(
            getattr(args, "float_argmax_mismatch", 0.0)
        ),
        "int8_max_abs": float(getattr(args, "int8_max_abs", 2.0)),
        "int8_mean_abs": float(getattr(args, "int8_mean_abs", 0.20)),
        "int8_score_abs": float(getattr(args, "int8_score_abs", 12.0)),
        "int8_argmax_mismatch": float(
            getattr(args, "int8_argmax_mismatch", 0.25)
        ),
    }
    if any(not math.isfinite(value) or value < 0 for value in result.values()):
        raise ValueError("equivalence limits must be finite and non-negative")
    return result


def convert(
    args: argparse.Namespace,
    *,
    backend: ConversionBackend | None = None,
) -> dict[str, Any]:
    artifact_role = str(getattr(args, "artifact_role", "detector"))
    if artifact_role not in {"detector", "candidate-verifier"}:
        raise ValueError("unsupported ordered-state artifact role")
    detector_role = artifact_role == "detector"
    validated = validate_inputs(
        args.distillation_training,
        args.weights,
        args.representative_features,
        args.float_qualification,
        artifact_role,
    )
    features = np.load(validated.representative_features.path, mmap_mode="r")
    if features.ndim != 3 or tuple(features.shape[1:]) != (
        INPUT_FRAMES,
        FEATURE_BINS,
    ):
        raise ValueError("representative features must have shape [N, 260, 40]")
    if len(features) < 1:
        raise ValueError("representative features are empty")
    equivalence_count = int(getattr(args, "equivalence_examples", 32))
    calibration_count = int(getattr(args, "calibration_examples", 500))
    if equivalence_count < 1 or calibration_count < 1:
        raise ValueError("representative sample counts must be positive")
    equivalence_indices = _spread_indices(len(features), equivalence_count)
    calibration_indices = _spread_indices(len(features), calibration_count)
    flags = _student_flags(
        validated.topology.state_count,
        validated.metadata.get("student_architecture", "control_mixconv"),
    )
    flags_contract = _flags_contract(flags)
    if flags_contract["stride"] != 3 or flags_contract["num_states"] != validated.topology.state_count:
        raise ValueError("current ordered-state student flags drifted from detector contract")
    phase_offset = _phase_offset(flags)
    backend = backend or TensorFlowConversionBackend()
    with tempfile.TemporaryDirectory(prefix="kizz-detector-convert-") as temporary:
        runtime = backend.convert(
            flags=flags,
            topology=validated.topology,
            weights=validated.weights.path,
            representative_features=features,
            calibration_indices=calibration_indices,
            phase_offset=phase_offset,
            work_dir=Path(temporary),
        )
        _validate_tensor_contracts(runtime, validated.topology, int(flags.stride))
        equivalence = build_equivalence_evidence(
            runtime,
            features,
            equivalence_indices,
            validated.topology,
            stride=int(flags.stride),
            phase_offset=phase_offset,
            limits=_limits(args),
        )
        artifact = bytes(runtime.artifact)
    if not artifact:
        raise ValueError("TFLite conversion produced an empty artifact")
    validated.revalidate()
    if sha256_file(validated.representative_features.path) != validated.representative_features.sha256:
        raise ValueError("representative features changed during conversion")

    output_times = _output_times(flags)
    decoder_arguments = {
        "from_logits": True,
        "state_evidence_floor": None,
        "self_loop_probability": 0.6,
        "next_state_probability": 0.4,
    }
    topology_contract = {
        "phrase_id": KIZZ_CONTROL.phrase_id,
        "text": KIZZ_CONTROL.text,
        "phones": list(validated.topology.phones),
        "states_per_phone": validated.topology.states_per_phone,
        "state_count": validated.topology.state_count,
        "state_names": list(validated.topology.state_names),
        "background_index": validated.topology.background_index,
        "silence_index": validated.topology.silence_index,
        "first_ordered_state_index": validated.topology.first_ordered_state_index,
    }
    artifact_kind = ARTIFACT_KIND if detector_role else VERIFIER_ARTIFACT_KIND
    artifact_filename = (
        ARTIFACT_FILENAME if detector_role else VERIFIER_ARTIFACT_FILENAME
    )
    student_role = STUDENT_ROLE if detector_role else VERIFIER_STUDENT_ROLE
    source_cache_key = (
        "detector_cache_metadata"
        if detector_role
        else "candidate_verifier_cache_metadata"
    )
    source_qualification_key = (
        "float_detector_qualification"
        if detector_role
        else "float_candidate_verifier_qualification"
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "kind": artifact_kind,
        "student_role": student_role,
        "model_role": (
            "permissive_continuous_candidate_detector"
            if detector_role
            else "detector_conditioned_ordered_state_candidate_verifier"
        ),
        "candidate_conditioned": not detector_role,
        "deployment_qualification": False,
        "qualification_scope": "conversion_and_numeric_equivalence_only",
        "artifact": {
            "filename": artifact_filename,
            "bytes": len(artifact),
            "sha256": hashlib.sha256(artifact).hexdigest(),
        },
        "source": {
            "distillation_training": {
                "path": str(validated.training_path),
                "sha256": validated.training_sha256,
            },
            "selected_weights": {
                "path": str(validated.weights.path),
                "sha256": validated.weights.sha256,
            },
            "representative_features": {
                "path": str(validated.representative_features.path),
                "sha256": validated.representative_features.sha256,
                "shape": [int(value) for value in features.shape],
                "dtype": str(features.dtype),
                "calibration_indices": [int(value) for value in calibration_indices],
                "calibration_indices_sha256": sha256_json(
                    [int(value) for value in calibration_indices]
                ),
            },
            source_cache_key: {
                "path": str(validated.cache_path),
                "sha256": validated.cache_sha256,
                "cache_sha256": validated.cache["cache_sha256"],
            },
            "detector_teacher_gate": {
                "path": str(validated.gate_path),
                "sha256": validated.gate_sha256,
            },
            source_qualification_key: {
                "path": str(validated.float_qualification_path),
                "sha256": validated.float_qualification_sha256,
            },
        },
        "model": {
            "keras_builder": "microwakeword.ordered_state_model.model",
            "student_flags_factory": "tools.distill_kizz_student.student_flags",
            "student_flags": flags_contract,
            "parameter_count": int(runtime.model_parameters),
            "framework": dict(sorted(runtime.framework.items())),
            "stream_conversion_mode": "STREAM_INTERNAL_STATE_INFERENCE",
        },
        "topology": topology_contract,
        "decoder": {
            "algorithm": "ordered_state_sequence_score_numpy",
            "contract_version": 1,
            "arguments": decoder_arguments,
            "score_semantics": "maximum_complete_left_to_right_log_odds_path",
            "threshold_binding": "selected_later_by_validation_only_joint_cascade_sweep",
            "provisional_float_threshold": {
                "threshold": validated.provisional_float_threshold,
                "fit_split": "validation",
                "test_used_for_selection": False,
                "minimum_recall": 0.95 if detector_role else 1.0,
                "maximum_false_candidate_fraction": 0.20 if detector_role else 1.0,
                "qualification_report_sha256": validated.float_qualification_sha256,
                "applies_to": "float_offline_ordered_state_score",
                "deployment_qualification": False,
            },
            "reference_module": str(ORDERED_STATE_MODULE),
            "reference_module_sha256": sha256_file(ORDERED_STATE_MODULE),
            "contract_sha256": sha256_json(
                {
                    "topology": topology_contract,
                    "algorithm": "ordered_state_sequence_score_numpy",
                    "arguments": decoder_arguments,
                }
            ),
        },
        "timeline": {
            "frontend_feature_step_seconds": FEATURE_STEP_SECONDS,
            "frontend_window_seconds": FRONTEND_WINDOW_SECONDS,
            "offline_input_frames": INPUT_FRAMES,
            "offline_output_frames": OUTPUT_FRAMES,
            "stream_input_frames_per_call": int(flags.stride),
            "stream_hop_seconds": int(flags.stride) * FEATURE_STEP_SECONDS,
            "stream_phase_offset_frames": phase_offset,
            "stream_phase_priming": "zero_prefix_then_observed_prefix",
            "streaming_calls_per_260_frame_example": equivalence[
                "streaming_calls_per_example"
            ],
            "streaming_warmup_outputs_discarded": equivalence[
                "streaming_warmup_outputs_discarded"
            ],
            "offline_output_times_seconds": output_times,
            "causal_tail_alignment": "derived_from_calls_minus_offline_output_frames",
        },
        "tensor_contracts": {
            "input": runtime.input_spec.as_json(),
            "output": runtime.output_spec.as_json(),
            "output_semantics": "unnormalized_ordered_state_logits",
        },
        "static_memory_contract": {
            "batch_size": 1,
            "fixed_input_shape": True,
            "fixed_output_shape": True,
            "dynamic_tensor_shapes_forbidden": True,
            "external_state_tensor_count": 0,
            "persistent_state": "internal_tflite_variables",
            "interpreter_allocation": "allocate_once_then_reuse_without_resize",
            "simultaneous_model_instances": 1,
            "activation_workspace": "single_invocation_reusable",
            "tensor_audit": runtime.tensor_audit,
            "tensor_arena_bytes": None,
            "hardware_high_water_measurement_required": True,
        },
        "equivalence": equivalence,
        "provenance": {
            "verified_binding_count": len(validated.all_bindings),
            "ordered_state_model_module_sha256": sha256_file(
                ORDERED_STATE_MODEL_MODULE
            ),
            "student_flags_module_sha256": sha256_file(DISTILLATION_MODULE),
            "converter_module_sha256": sha256_file(Path(__file__).resolve()),
        },
        "limitations": [
            (
                "synthetic-bootstrap detector evidence only"
                if detector_role
                else "candidate-conditioned verifier evidence only"
            ),
            "not qualified for final deployment",
            "joint detector-verifier thresholds and continuous FAPH remain unmeasured",
            "ESP32-S3 latency, tensor arena, and sustained-run stability remain unmeasured",
        ],
    }

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifact_path = output / artifact_filename
    metadata_path = output / METADATA_FILENAME
    _atomic_write(artifact_path, artifact)
    _atomic_write(
        metadata_path,
        (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distillation-training", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--representative-features", type=Path, required=True)
    parser.add_argument("--float-qualification", type=Path, required=True)
    parser.add_argument(
        "--artifact-role",
        choices=("detector", "candidate-verifier"),
        default="detector",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-examples", type=int, default=500)
    parser.add_argument("--equivalence-examples", type=int, default=32)
    parser.add_argument("--float-max-abs", type=float, default=1e-4)
    parser.add_argument("--float-mean-abs", type=float, default=1e-5)
    parser.add_argument("--float-score-abs", type=float, default=1e-3)
    parser.add_argument("--float-argmax-mismatch", type=float, default=0.0)
    parser.add_argument("--int8-max-abs", type=float, default=2.0)
    parser.add_argument("--int8-mean-abs", type=float, default=0.20)
    parser.add_argument("--int8-score-abs", type=float, default=12.0)
    parser.add_argument("--int8-argmax-mismatch", type=float, default=0.25)
    args = parser.parse_args(argv)
    try:
        metadata = convert(args)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
