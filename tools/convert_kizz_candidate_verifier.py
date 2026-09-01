#!/usr/bin/env python3
"""Package the fixed-window Kizz candidate verifier as a fully INT8 TFLite model.

The converter is intentionally not a cascade qualification step.  It accepts only
the provenance-bound output of ``train_kizz_candidate_verifier.py``, calibrates on
training candidates, checks numeric equivalence on validation candidates, and
never scores or tunes against the test split.
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
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

if __package__:
    from tools.train_kizz_candidate_verifier import (
        MODEL_JSON_PROVENANCE_KEY,
        MODEL_VARIANT_CHANNELS,
        INPUT_SHAPE,
        TensorFlowVerifierBackend,
        dscnn_spec,
        estimate_dscnn_cost,
        load_verified_dataset,
        model_topology_sha256,
    )
else:  # Direct ``python tools/convert_kizz_candidate_verifier.py`` execution.
    from train_kizz_candidate_verifier import (  # type: ignore[no-redef]
        MODEL_JSON_PROVENANCE_KEY,
        MODEL_VARIANT_CHANNELS,
        INPUT_SHAPE,
        TensorFlowVerifierBackend,
        dscnn_spec,
        estimate_dscnn_cost,
        load_verified_dataset,
        model_topology_sha256,
    )


SCHEMA_VERSION = 1
TRAINING_RECIPE = "kizz_control_candidate_conditioned_dscnn_verifier_v1"
ARTIFACT_KIND = "kizz_control_candidate_verifier_fixed_window_int8"
ARTIFACT_FILENAME = "kizz_control_candidate_verifier_int8.tflite"
METADATA_FILENAME = "firmware-artifact.json"
MODEL_ROLE = "detector_conditioned_candidate_verifier"
EXPECTED_INPUT_SHAPE = (1, 260, 40, 1)
EXPECTED_OUTPUT_SHAPE = (1, 1)
EXPECTED_OP_COUNTS = {
    "CONV_2D": 5,
    "DEPTHWISE_CONV_2D": 4,
    "FULLY_CONNECTED": 1,
}
ALLOWED_OPS = frozenset((*EXPECTED_OP_COUNTS, "RESHAPE", "MUL", "TANH"))
CONVERTER_MODULE = Path(__file__).resolve()
TRAINER_MODULE = CONVERTER_MODULE.with_name("train_kizz_candidate_verifier.py")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
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


def _require_sha256(value: object, label: str) -> str:
    digest = str(value or "")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return digest


def _resolve_path(value: object, anchor: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} requires a path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = anchor / path
    return path.resolve()


@dataclass(frozen=True)
class FileBinding:
    label: str
    path: Path
    sha256: str
    bytes: int

    @classmethod
    def from_value(
        cls, value: object, *, anchor: Path, label: str
    ) -> "FileBinding":
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} binding is missing")
        path = _resolve_path(value.get("path"), anchor, label)
        expected = _require_sha256(value.get("sha256"), f"{label} hash")
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(
                f"{label} hash drift: expected {expected}, got {observed}"
            )
        size = path.stat().st_size
        if "bytes" in value and value.get("bytes") != size:
            raise ValueError(f"{label} byte count drift")
        return cls(label, path, observed, size)

    def as_json(self) -> dict[str, Any]:
        return {"path": str(self.path), "sha256": self.sha256, "bytes": self.bytes}

    def revalidate(self) -> None:
        if not self.path.is_file() or sha256_file(self.path) != self.sha256:
            raise ValueError(f"{self.label} changed during conversion")
        if self.path.stat().st_size != self.bytes:
            raise ValueError(f"{self.label} byte count changed during conversion")


def _binding_identity(binding: FileBinding) -> tuple[str, str, int]:
    return str(binding.path), binding.sha256, binding.bytes


def _same_binding(left: FileBinding, right: FileBinding, label: str) -> None:
    if _binding_identity(left) != _binding_identity(right):
        raise ValueError(f"{label} binding disagreement")


def _expected_architecture(model_variant: str) -> dict[str, Any]:
    cost = estimate_dscnn_cost(INPUT_SHAPE, model_variant=model_variant)
    # JSON normalization converts the tuple-valued fixed specification to the
    # representation emitted in training-report.json.
    return json.loads(json.dumps(cost, sort_keys=True))


def _model_topology_sha256(serialized: str) -> str:
    return model_topology_sha256(serialized)


def _verify_model_json_provenance(
    serialized: str,
    *,
    model_variant: str,
    expected_cost: Mapping[str, Any],
    allow_legacy_compact: bool,
) -> None:
    try:
        payload = json.loads(serialized)
    except json.JSONDecodeError as error:
        raise ValueError(f"bound model architecture is not valid JSON: {error}") from error
    provenance = payload.get(MODEL_JSON_PROVENANCE_KEY)
    if provenance is None and allow_legacy_compact:
        return
    if not isinstance(provenance, Mapping):
        raise ValueError("bound model JSON provenance is missing")
    expected = {
        "schema_version": 1,
        "name": "fixed_window_dscnn",
        "variant": model_variant,
        "channel_plan": list(MODEL_VARIANT_CHANNELS[model_variant]),
        "input_shape": [*INPUT_SHAPE, 1],
        "output": "one_logit",
        "dscnn_spec": json.loads(json.dumps(dscnn_spec(model_variant))),
        "cost": json.loads(json.dumps(expected_cost)),
        "topology_sha256": _model_topology_sha256(serialized),
    }
    if dict(provenance) != expected:
        raise ValueError("bound model JSON variant/spec/cost/topology provenance drift")


def _verify_architecture(
    report: Mapping[str, Any],
) -> tuple[str, dict[str, Any], bool]:
    architecture = report.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError("training report architecture is missing")
    raw_variant = architecture.get("variant")
    legacy_compact = (
        raw_variant is None
        and "channel_plan" not in architecture
        and "dscnn_spec" not in architecture
        and architecture.get("input_shape") == list(INPUT_SHAPE)
    )
    if legacy_compact:
        # Pre-variant reports are accepted only through their complete, exact
        # compact architecture/cost contract. This is provenance inference,
        # not an unvalidated compact default.
        model_variant = "compact"
    elif raw_variant in MODEL_VARIANT_CHANNELS:
        model_variant = str(raw_variant)
    else:
        raise ValueError("training report model variant is missing or unsupported")
    expected = _expected_architecture(model_variant)
    if architecture.get("name") != "fixed_window_dscnn":
        raise ValueError("training report does not name the locked DS-CNN topology")
    if architecture.get("output") != "one_logit":
        raise ValueError("training report output is not one logit")
    expected_input_shape = list(INPUT_SHAPE) if legacy_compact else [*INPUT_SHAPE, 1]
    if architecture.get("input_shape") != expected_input_shape:
        raise ValueError("training report input shape drift")
    if not legacy_compact:
        if architecture.get("channel_plan") != list(
            MODEL_VARIANT_CHANNELS[model_variant]
        ):
            raise ValueError("training report DS-CNN channel plan drift")
        expected_spec = json.loads(json.dumps(dscnn_spec(model_variant)))
        if architecture.get("dscnn_spec") != expected_spec:
            raise ValueError("training report DS-CNN specification drift")
    expected_ops = [
        "Conv2D",
        "DepthwiseConv2D",
        "pointwise_Conv2D",
        "Flatten",
        "Dense",
    ]
    if architecture.get("int8_friendly_core_ops") != expected_ops:
        raise ValueError("training report INT8 operator contract drift")
    for key in (
        "parameter_estimate",
        "mac_estimate",
        "mac_scope",
        "layers",
    ):
        if architecture.get(key) != expected[key]:
            raise ValueError(f"training report DS-CNN {key} drift")
    if architecture.get("parameter_count") != expected["parameter_estimate"]:
        raise ValueError("training report parameter count drift")
    if not legacy_compact and architecture.get("topology_sha256") is None:
        raise ValueError("training report model topology hash is missing")
    return model_variant, expected, legacy_compact


def _verified_report_binding(
    value: object, *, anchor: Path, label: str
) -> FileBinding:
    return FileBinding.from_value(value, anchor=anchor, label=label)


@dataclass(frozen=True)
class ValidatedInputs:
    training_report_path: Path
    training_report_sha256: str
    report: dict[str, Any]
    dataset: Any
    corpus: FileBinding
    arrays: tuple[FileBinding, ...]
    transitive: tuple[FileBinding, ...]
    model_architecture: FileBinding
    model_topology_sha256: str
    weights: FileBinding
    validation_logits: FileBinding
    model_variant: str
    expected_parameter_count: int
    expected_mac_count: int
    frozen_threshold: float

    @property
    def all_bindings(self) -> tuple[FileBinding, ...]:
        unique: dict[tuple[str, str, int], FileBinding] = {}
        for binding in (
            self.corpus,
            *self.arrays,
            *self.transitive,
            self.model_architecture,
            self.weights,
            self.validation_logits,
        ):
            unique[_binding_identity(binding)] = binding
        return tuple(unique[key] for key in sorted(unique))

    def revalidate(self) -> None:
        if sha256_file(self.training_report_path) != self.training_report_sha256:
            raise ValueError("training report changed during conversion")
        for binding in self.all_bindings:
            binding.revalidate()


def validate_inputs(training_report: Path, weights: Path) -> ValidatedInputs:
    training_report = training_report.expanduser().resolve()
    weights = weights.expanduser().resolve()
    report_sha = sha256_file(training_report)
    report = _load_json(training_report, "training report")
    if report.get("schema_version") != 1 or report.get("recipe") != TRAINING_RECIPE:
        raise ValueError("unsupported candidate-verifier training report")
    if report.get("candidate_conditioned") is not True:
        raise ValueError("training report is not detector-conditioned")
    if report.get("deployment_qualification") is not False:
        raise ValueError("training report must remain non-deployment-qualified")
    model_variant, expected_cost, legacy_compact = _verify_architecture(report)

    selection = report.get("selection_contract")
    if not isinstance(selection, Mapping):
        raise ValueError("training report selection contract is missing")
    if (
        selection.get("selection_split") != "validation"
        or selection.get("test_used_for_selection") is not False
        or selection.get("test_score_passes") != 1
    ):
        raise ValueError("training report violates validation-only selection")
    winner = report.get("winner")
    if not isinstance(winner, Mapping):
        raise ValueError("training report winner is missing")
    try:
        frozen_threshold = float(winner["frozen_threshold"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("training report frozen threshold is invalid") from error
    if not math.isfinite(frozen_threshold):
        raise ValueError("training report frozen threshold is non-finite")

    anchor = training_report.parent
    inputs = report.get("input_bindings")
    outputs = report.get("output_bindings")
    if not isinstance(inputs, Mapping) or not isinstance(outputs, Mapping):
        raise ValueError("training report file bindings are missing")
    corpus = _verified_report_binding(
        inputs.get("corpus"), anchor=anchor, label="candidate corpus"
    )
    dataset = load_verified_dataset(
        corpus.path.parent, expected_corpus_sha256=corpus.sha256
    )
    if dataset.corpus_path != corpus.path:
        raise ValueError("candidate corpus must be corpus.json in the dataset root")

    report_arrays = inputs.get("arrays")
    if not isinstance(report_arrays, Mapping):
        raise ValueError("training report candidate array bindings are missing")
    if set(report_arrays) != set(dataset.array_bindings):
        raise ValueError("training report candidate array binding set drift")
    arrays: list[FileBinding] = []
    for name in sorted(report_arrays):
        bound = _verified_report_binding(
            report_arrays[name], anchor=anchor, label=f"candidate array {name}"
        )
        actual = _verified_report_binding(
            dataset.array_bindings[name],
            anchor=dataset.root,
            label=f"verified candidate array {name}",
        )
        _same_binding(bound, actual, f"candidate array {name}")
        arrays.append(bound)

    report_transitive = inputs.get("transitive")
    if not isinstance(report_transitive, list):
        raise ValueError("training report transitive bindings are missing")
    transitive: list[FileBinding] = []
    for index, value in enumerate(report_transitive):
        transitive.append(
            _verified_report_binding(
                value, anchor=anchor, label=f"transitive binding {index}"
            )
        )
    expected_transitive = {
        (entry["path"], entry["sha256"], int(entry["bytes"]))
        for entry in dataset.transitive_bindings
    }
    actual_transitive = {_binding_identity(binding) for binding in transitive}
    if actual_transitive != expected_transitive:
        raise ValueError("training report transitive candidate bindings drift")

    architecture_binding = _verified_report_binding(
        outputs.get("model_architecture"),
        anchor=anchor,
        label="model architecture",
    )
    serialized_model = architecture_binding.path.read_text(encoding="utf-8")
    model_topology_sha256 = _model_topology_sha256(serialized_model)
    _verify_model_json_provenance(
        serialized_model,
        model_variant=model_variant,
        expected_cost=expected_cost,
        allow_legacy_compact=legacy_compact,
    )
    if (
        not legacy_compact
        and report["architecture"].get("topology_sha256")
        != model_topology_sha256
    ):
        raise ValueError("training report model topology hash drift")
    winner_weights = _verified_report_binding(
        winner.get("best_weights"), anchor=anchor, label="winner best weights"
    )
    output_weights = _verified_report_binding(
        outputs.get("best_weights"), anchor=anchor, label="output best weights"
    )
    _same_binding(winner_weights, output_weights, "selected weights")
    provided_weights = FileBinding(
        "provided selected weights",
        weights,
        sha256_file(weights) if weights.is_file() else "",
        weights.stat().st_size if weights.is_file() else -1,
    )
    if not weights.is_file():
        raise FileNotFoundError(weights)
    _same_binding(winner_weights, provided_weights, "selected weights")
    checkpoint = _verified_report_binding(
        winner.get("checkpoint"), anchor=anchor, label="winner checkpoint"
    )
    if checkpoint.sha256 != winner_weights.sha256:
        raise ValueError("winner checkpoint is not byte-identical to selected weights")
    validation_logits = _verified_report_binding(
        outputs.get("validation_winner_logits"),
        anchor=anchor,
        label="validation winner logits",
    )
    validation_count = sum(row["split"] == "validation" for row in dataset.rows)
    logits = np.load(validation_logits.path, mmap_mode="r", allow_pickle=False)
    if logits.shape != (validation_count,) or not np.all(np.isfinite(logits)):
        raise ValueError("bound validation winner logits shape/content drift")

    return ValidatedInputs(
        training_report,
        report_sha,
        report,
        dataset,
        corpus,
        tuple(arrays),
        tuple(transitive),
        architecture_binding,
        model_topology_sha256,
        winner_weights,
        validation_logits,
        model_variant,
        int(expected_cost["parameter_estimate"]),
        int(expected_cost["mac_estimate"]),
        frozen_threshold,
    )


def _spread_indices(values: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if values.ndim != 1 or len(values) < 1:
        raise ValueError("candidate index set is empty")
    if count < 1:
        raise ValueError("example count must be positive")
    positions = np.linspace(0, len(values) - 1, min(count, len(values)), dtype=np.int64)
    return values[np.unique(positions)]


def _features_for(dataset: Any, indices: np.ndarray) -> np.ndarray:
    values = np.asarray(dataset.features[indices], dtype=np.float32)[..., None]
    if values.shape[1:] != (260, 40, 1) or not np.all(np.isfinite(values)):
        raise ValueError("candidate features must be finite [N,260,40,1] tensors")
    return values


def _representative_dataset(features: np.ndarray):
    for sample in features:
        yield [np.asarray(sample[None, ...], dtype=np.float32)]


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str
    quantization: tuple[float, int]

    def as_json(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "quantization": {
                "scale": self.quantization[0],
                "zero_point": self.quantization[1],
            },
        }


@dataclass(frozen=True)
class ConversionRuntime:
    artifact: bytes
    input_spec: TensorSpec
    output_spec: TensorSpec
    run_float: Callable[[np.ndarray], float]
    run_int8: Callable[[np.ndarray], float]
    tensor_audit: dict[str, Any]
    operator_audit: dict[str, Any]
    parameter_count: int
    model_json_sha256: str
    framework: dict[str, str]


class ConversionBackend(Protocol):
    def convert(
        self,
        *,
        validated: ValidatedInputs,
        calibration_features: np.ndarray,
        work_dir: Path,
    ) -> ConversionRuntime: ...


def _rebuild_bound_training_model(
    validated: ValidatedInputs,
    *,
    builder: Any | None = None,
) -> tuple[Any, Any, str]:
    """Rebuild exactly the variant declared by verified training provenance."""
    training = validated.report.get("training", {})
    builder = builder or TensorFlowVerifierBackend()
    model = builder.build_model(
        learning_rate=float(training.get("learning_rate")),
        seed=int(training.get("seed")),
        model_variant=validated.model_variant,
    )
    rebuilt_json = builder.model_json(model)
    bound_json = validated.model_architecture.path.read_text(encoding="utf-8")
    if _model_topology_sha256(rebuilt_json) != _model_topology_sha256(bound_json):
        raise ValueError("rebuilt DS-CNN model topology differs from bound model JSON")
    builder.load_weights(model, validated.weights.path)
    if int(builder.count_params(model)) != validated.expected_parameter_count:
        raise ValueError("rebuilt DS-CNN parameter count drift")
    return builder, model, rebuilt_json


class TensorFlowConversionBackend:
    """Lazy TensorFlow conversion backend for the locked trainer topology."""

    def __init__(
        self,
        *,
        deployment_logit_bound: float | None = None,
        quantization_mode: str = "int8",
    ):
        self.deployment_logit_bound = deployment_logit_bound
        self.quantization_mode = quantization_mode

    def convert(
        self,
        *,
        validated: ValidatedInputs,
        calibration_features: np.ndarray,
        work_dir: Path,
    ) -> ConversionRuntime:
        del work_dir
        import tensorflow as tf

        builder, model, rebuilt_json = _rebuild_bound_training_model(validated)

        fixed_input = tf.keras.Input(
            batch_shape=EXPECTED_INPUT_SHAPE, name="log_mel_window"
        )
        deployment_model = tf.keras.models.clone_model(
            model, input_tensors=fixed_input
        )
        deployment_model.set_weights(model.get_weights())
        if self.deployment_logit_bound is not None:
            bounded = tf.keras.layers.Rescaling(
                scale=1.0 / self.deployment_logit_bound,
                name="deployment_logit_scale_down",
            )(deployment_model.output)
            bounded = tf.keras.layers.Activation(
                "tanh", name="deployment_logit_tanh"
            )(bounded)
            bounded = tf.keras.layers.Rescaling(
                scale=self.deployment_logit_bound,
                name="deployment_logit",
            )(bounded)
            deployment_model = tf.keras.Model(
                deployment_model.input,
                bounded,
                name="kizz_candidate_verifier_bounded_deployment",
            )
        if (
            tuple(deployment_model.input_shape) != EXPECTED_INPUT_SHAPE
            or tuple(deployment_model.output_shape) != EXPECTED_OUTPUT_SHAPE
        ):
            raise ValueError("fixed-batch DS-CNN clone tensor contract drift")
        if int(deployment_model.count_params()) != validated.expected_parameter_count:
            raise ValueError("fixed-batch DS-CNN clone parameter count drift")

        float_converter = tf.lite.TFLiteConverter.from_keras_model(deployment_model)
        float_artifact = float_converter.convert()
        converter = tf.lite.TFLiteConverter.from_keras_model(deployment_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if self.quantization_mode == "int8":
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        elif self.quantization_mode == "int16-activations":
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
            ]
            converter.inference_input_type = tf.int16
            converter.inference_output_type = tf.int16
        else:  # pragma: no cover - validated by convert().
            raise ValueError(f"unsupported quantization mode: {self.quantization_mode}")
        converter.representative_dataset = tf.lite.RepresentativeDataset(
            lambda: _representative_dataset(calibration_features)
        )
        artifact = converter.convert()
        interpreter_options = {
            "model_content": artifact,
            "experimental_op_resolver_type": (
                tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
            ),
        }
        interpreter = tf.lite.Interpreter(**interpreter_options)
        interpreter.allocate_tensors()
        inputs = interpreter.get_input_details()
        outputs = interpreter.get_output_details()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("candidate verifier must expose exactly one input and output")

        def tensor_spec(detail: Mapping[str, Any]) -> TensorSpec:
            scale, zero = detail.get("quantization", (0.0, 0))
            return TensorSpec(
                tuple(int(value) for value in detail["shape"]),
                np.dtype(detail["dtype"]).name,
                (float(scale), int(zero)),
            )

        details = interpreter.get_tensor_details()
        declared_bytes = 0
        dynamic = 0
        variable = 0
        float_tensors = 0
        tensor_dtypes: dict[str, int] = {}
        for detail in details:
            shape = tuple(int(value) for value in detail.get("shape", ()))
            signature = tuple(
                int(value) for value in detail.get("shape_signature", shape)
            )
            dtype = np.dtype(detail["dtype"])
            tensor_dtypes[dtype.name] = tensor_dtypes.get(dtype.name, 0) + 1
            float_tensors += int(np.issubdtype(dtype, np.floating))
            if any(value < 0 for value in signature):
                dynamic += 1
            elif shape:
                declared_bytes += int(np.prod(shape)) * dtype.itemsize
            variable += int(bool(detail.get("is_variable", False)))

        op_names = [
            str(detail.get("op_name", ""))
            for detail in interpreter._get_ops_details()  # noqa: SLF001
        ]
        op_counts = {name: op_names.count(name) for name in sorted(set(op_names))}

        def run_float(sample: np.ndarray) -> float:
            runner = tf.lite.Interpreter(
                model_content=float_artifact,
                experimental_op_resolver_type=(
                    tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
                ),
            )
            runner.allocate_tensors()
            in_detail = runner.get_input_details()[0]
            out_detail = runner.get_output_details()[0]
            runner.set_tensor(
                in_detail["index"], np.asarray(sample[None, ...], dtype=np.float32)
            )
            runner.invoke()
            values = np.asarray(runner.get_tensor(out_detail["index"]), dtype=np.float32)
            if values.shape != EXPECTED_OUTPUT_SHAPE:
                raise ValueError("float TFLite verifier output shape drift")
            return float(values[0, 0])

        input_detail = inputs[0]
        output_detail = outputs[0]

        def run_int8(sample: np.ndarray) -> float:
            runner = tf.lite.Interpreter(**interpreter_options)
            runner.allocate_tensors()
            in_detail = runner.get_input_details()[0]
            out_detail = runner.get_output_details()[0]
            input_scale, input_zero = in_detail["quantization"]
            output_scale, output_zero = out_detail["quantization"]
            if input_scale <= 0 or output_scale <= 0:
                raise ValueError("INT8 verifier quantization scales must be positive")
            limits = np.iinfo(in_detail["dtype"])
            quantized = np.clip(
                np.rint(sample / input_scale + input_zero), limits.min, limits.max
            ).astype(in_detail["dtype"])
            runner.set_tensor(in_detail["index"], quantized[None, ...])
            runner.invoke()
            raw = runner.get_tensor(out_detail["index"])
            if raw.shape != EXPECTED_OUTPUT_SHAPE:
                raise ValueError("INT8 verifier output shape drift")
            return float((float(raw[0, 0]) - output_zero) * output_scale)

        return ConversionRuntime(
            artifact=bytes(artifact),
            input_spec=tensor_spec(input_detail),
            output_spec=tensor_spec(output_detail),
            run_float=run_float,
            run_int8=run_int8,
            tensor_audit={
                "tensor_count": len(details),
                "declared_tensor_bytes_sum": declared_bytes,
                "dynamic_shape_tensor_count": dynamic,
                "variable_tensor_count": variable,
                "float_tensor_count": float_tensors,
                "tensor_dtypes": tensor_dtypes,
                "input_count": len(inputs),
                "output_count": len(outputs),
            },
            operator_audit={"operators": op_names, "counts": op_counts},
            parameter_count=int(deployment_model.count_params()),
            model_json_sha256=_model_topology_sha256(rebuilt_json),
            framework={"tensorflow": str(tf.__version__)},
        )


def _validate_runtime(
    runtime: ConversionRuntime, expected_parameters: int, quantization_mode: str
) -> None:
    if runtime.input_spec.shape != EXPECTED_INPUT_SHAPE:
        raise ValueError("TFLite input shape must be [1,260,40,1]")
    if runtime.output_spec.shape != EXPECTED_OUTPUT_SHAPE:
        raise ValueError("TFLite scalar-logit output shape must be [1,1]")
    expected_io_dtype = "int8" if quantization_mode == "int8" else "int16"
    for label, spec in (("input", runtime.input_spec), ("output", runtime.output_spec)):
        if spec.dtype != expected_io_dtype:
            raise ValueError(f"TFLite {label} must be {expected_io_dtype}")
        scale, zero = spec.quantization
        limits = np.iinfo(np.dtype(expected_io_dtype))
        if not math.isfinite(scale) or scale <= 0 or not limits.min <= zero <= limits.max:
            raise ValueError(f"TFLite {label} quantization parameters are invalid")
    if runtime.parameter_count != expected_parameters:
        raise ValueError("converted model parameter count drift")
    audit = runtime.tensor_audit
    if audit.get("input_count") != 1 or audit.get("output_count") != 1:
        raise ValueError("static-memory audit requires one input and one output")
    if audit.get("dynamic_shape_tensor_count") != 0:
        raise ValueError("static-memory audit forbids dynamic tensor shapes")
    if audit.get("variable_tensor_count") != 0:
        raise ValueError("fixed-window verifier must not contain variable tensors")
    if audit.get("float_tensor_count") != 0:
        raise ValueError("fully INT8 model contains floating-point tensors")
    dtypes = audit.get("tensor_dtypes")
    allowed_dtypes = (
        {"int8", "int32"}
        if quantization_mode == "int8"
        else {"int8", "int16", "int32", "int64"}
    )
    if not isinstance(dtypes, Mapping) or set(dtypes) - allowed_dtypes:
        raise ValueError("fully INT8 model contains unsupported tensor dtypes")
    counts = runtime.operator_audit.get("counts")
    if not isinstance(counts, Mapping):
        raise ValueError("TFLite operator audit is missing")
    unsupported = set(counts) - ALLOWED_OPS
    if unsupported:
        raise ValueError(f"TFLite contains unsupported operators: {sorted(unsupported)}")
    for name, expected in EXPECTED_OP_COUNTS.items():
        if counts.get(name) != expected:
            raise ValueError(f"TFLite {name} operator count drift")


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    result = np.empty_like(value)
    positive = value >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _apply_logit_bound(value: np.ndarray, bound: float | None) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if bound is None:
        return value
    return bound * np.tanh(value / bound)


def _equivalence_limits(args: argparse.Namespace) -> dict[str, float]:
    limits = {
        "float_reference_max_absolute_error": float(
            getattr(args, "float_reference_max_absolute_error", 1e-4)
        ),
        "float_reference_mean_absolute_error": float(
            getattr(args, "float_reference_mean_absolute_error", 1e-5)
        ),
        "max_logit_absolute_error": float(
            getattr(args, "max_logit_absolute_error", 0.75)
        ),
        "mean_logit_absolute_error": float(
            getattr(args, "mean_logit_absolute_error", 0.15)
        ),
        "max_probability_absolute_error": float(
            getattr(args, "max_probability_absolute_error", 0.05)
        ),
        "threshold_decision_mismatch_fraction": float(
            getattr(args, "threshold_decision_mismatch_fraction", 0.0)
        ),
    }
    if any(not math.isfinite(value) or value < 0 for value in limits.values()):
        raise ValueError("equivalence limits must be finite and non-negative")
    if limits["threshold_decision_mismatch_fraction"] > 1:
        raise ValueError("decision mismatch fraction limit must not exceed one")
    return limits


def build_equivalence_evidence(
    runtime: ConversionRuntime,
    features: np.ndarray,
    indices: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    reference_float_logits: np.ndarray,
    *,
    threshold: float,
    limits: Mapping[str, float],
) -> dict[str, Any]:
    float_logits = np.asarray(
        [runtime.run_float(sample) for sample in features], dtype=np.float64
    )
    int8_logits = np.asarray(
        [runtime.run_int8(sample) for sample in features], dtype=np.float64
    )
    reference_float_logits = np.asarray(reference_float_logits, dtype=np.float64)
    if reference_float_logits.shape != float_logits.shape:
        raise ValueError("bound validation logits do not align with equivalence examples")
    if not (
        np.all(np.isfinite(reference_float_logits))
        and np.all(np.isfinite(float_logits))
        and np.all(np.isfinite(int8_logits))
    ):
        raise ValueError("equivalence logits must be finite")
    reference_errors = np.abs(reference_float_logits - float_logits)
    errors = np.abs(reference_float_logits - int8_logits)
    probability_errors = np.abs(
        _sigmoid(reference_float_logits) - _sigmoid(int8_logits)
    )
    mismatches = np.not_equal(
        reference_float_logits >= threshold, int8_logits >= threshold
    )
    actual = {
        "float_reference_max_absolute_error": float(np.max(reference_errors)),
        "float_reference_mean_absolute_error": float(np.mean(reference_errors)),
        "max_logit_absolute_error": float(np.max(errors)),
        "mean_logit_absolute_error": float(np.mean(errors)),
        "max_probability_absolute_error": float(np.max(probability_errors)),
        "threshold_decision_mismatch_fraction": float(np.mean(mismatches)),
    }
    for metric, limit in limits.items():
        if actual[metric] > limit:
            raise ValueError(
                f"INT8 verifier equivalence failed {metric}: "
                f"{actual[metric]:.9g} > {limit:.9g}"
            )
    return {
        "scope": "bound_validation_candidates_only",
        "float_reference": "bound_validation_winner_logits",
        "float_replay": "locked_topology_float_tflite",
        "test_examples_scored": 0,
        "threshold_source": "training_report.validation_only_frozen_threshold",
        "frozen_deployed_logit_threshold": threshold,
        "threshold_semantics": "deployed_scalar_logit",
        "example_count": len(indices),
        "feature_indices": [int(value) for value in indices],
        "candidate_ids": [str(rows[int(value)]["candidate_id"]) for value in indices],
        "feature_indices_sha256": sha256_json([int(value) for value in indices]),
        "limits": dict(limits),
        "actual": actual,
        "passed": True,
    }


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(fd, "wb") as target:
            target.write(content)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def convert(
    args: argparse.Namespace, *, backend: ConversionBackend | None = None
) -> dict[str, Any]:
    validated = validate_inputs(args.training_report, args.weights)
    raw_bound = float(getattr(args, "deployment_logit_bound", 0.0))
    if not math.isfinite(raw_bound) or raw_bound < 0:
        raise ValueError("deployment logit bound must be finite and non-negative")
    deployment_logit_bound = raw_bound if raw_bound > 0 else None
    quantization_mode = str(getattr(args, "quantization_mode", "int8"))
    if quantization_mode not in {"int8", "int16-activations"}:
        raise ValueError(f"unsupported quantization mode: {quantization_mode}")
    quantization_safety_margin = float(
        getattr(args, "quantization_logit_safety_margin", 0.0)
    )
    if (
        not math.isfinite(quantization_safety_margin)
        or quantization_safety_margin < 0
    ):
        raise ValueError(
            "quantization logit safety margin must be finite and non-negative"
        )
    calibration_count = int(getattr(args, "calibration_examples", 500))
    equivalence_count = int(getattr(args, "equivalence_examples", 128))
    rows = validated.dataset.rows
    train_pool = np.asarray(
        [index for index, row in enumerate(rows) if row["split"] == "train"],
        dtype=np.int64,
    )
    validation_pool = np.asarray(
        [index for index, row in enumerate(rows) if row["split"] == "validation"],
        dtype=np.int64,
    )
    calibration_indices = _spread_indices(train_pool, calibration_count)
    equivalence_indices = _spread_indices(validation_pool, equivalence_count)
    calibration_features = _features_for(validated.dataset, calibration_indices)
    equivalence_features = _features_for(validated.dataset, equivalence_indices)
    all_validation_logits = np.asarray(
        np.load(validated.validation_logits.path, mmap_mode="r", allow_pickle=False),
        dtype=np.float64,
    )
    validation_positions = {
        int(global_index): position
        for position, global_index in enumerate(validation_pool)
    }
    reference_float_logits = np.asarray(
        [all_validation_logits[validation_positions[int(index)]] for index in equivalence_indices],
        dtype=np.float64,
    )
    reference_float_logits = _apply_logit_bound(
        reference_float_logits, deployment_logit_bound
    )
    training_threshold_logit = math.log(
        validated.frozen_threshold / (1.0 - validated.frozen_threshold)
    )
    deployment_threshold_logit = float(
        _apply_logit_bound(
            np.asarray([training_threshold_logit]), deployment_logit_bound
        )[0]
    ) - quantization_safety_margin

    backend = backend or TensorFlowConversionBackend(
        deployment_logit_bound=deployment_logit_bound,
        quantization_mode=quantization_mode,
    )
    with tempfile.TemporaryDirectory(prefix="kizz-verifier-convert-") as temporary:
        runtime = backend.convert(
            validated=validated,
            calibration_features=calibration_features,
            work_dir=Path(temporary),
        )
        _validate_runtime(
            runtime, validated.expected_parameter_count, quantization_mode
        )
        if runtime.model_json_sha256 != validated.model_topology_sha256:
            raise ValueError("rebuilt model topology hash differs from bound topology")
        equivalence = build_equivalence_evidence(
            runtime,
            equivalence_features,
            equivalence_indices,
            rows,
            reference_float_logits,
            threshold=deployment_threshold_logit,
            limits=_equivalence_limits(args),
        )
        artifact = bytes(runtime.artifact)
    if not artifact:
        raise ValueError("TFLite conversion produced an empty artifact")
    if len(artifact) < 8 or artifact[4:8] != b"TFL3":
        raise ValueError("conversion output is not a TFLite FlatBuffer")
    validated.revalidate()

    calibration_ids = [str(rows[int(i)]["candidate_id"]) for i in calibration_indices]
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": ARTIFACT_KIND,
        "model_role": MODEL_ROLE,
        "candidate_conditioned": True,
        "deployment_qualification": False,
        "qualification_scope": "conversion_and_validation_numeric_equivalence_only",
        "cascade_qualification": False,
        "artifact": {
            "filename": ARTIFACT_FILENAME,
            "bytes": len(artifact),
            "sha256": hashlib.sha256(artifact).hexdigest(),
        },
        "inputs": {
            "training_report": {
                "path": str(validated.training_report_path),
                "sha256": validated.training_report_sha256,
                "bytes": validated.training_report_path.stat().st_size,
            },
            "selected_weights": validated.weights.as_json(),
            "model_architecture": validated.model_architecture.as_json(),
            "validation_winner_logits": validated.validation_logits.as_json(),
            "candidate_corpus": validated.corpus.as_json(),
            "candidate_arrays": {
                binding.path.name: binding.as_json() for binding in validated.arrays
            },
            "transitive_candidate_bindings": [
                binding.as_json() for binding in validated.transitive
            ],
        },
        "calibration": {
            "scope": "bound_training_candidates_only",
            "test_examples_used": 0,
            "validation_examples_used": 0,
            "example_count": len(calibration_indices),
            "feature_indices": [int(value) for value in calibration_indices],
            "candidate_ids": calibration_ids,
            "feature_indices_sha256": sha256_json(
                [int(value) for value in calibration_indices]
            ),
            "candidate_ids_sha256": sha256_json(calibration_ids),
            "selection": "deterministic_evenly_spaced_in_corpus_order",
        },
        "model": {
            "topology": "fixed_window_dscnn",
            "variant": validated.model_variant,
            "topology_sha256": validated.model_topology_sha256,
            "input_shape": [260, 40, 1],
            "output": "scalar_logit",
            "deployment_logit_transform": {
                "kind": "identity"
                if deployment_logit_bound is None
                else "bound_times_tanh_logit_over_bound",
                "bound": deployment_logit_bound,
                "monotonic": True,
                "selection_split_reused_without_retuning": True,
            },
            "quantization_mode": quantization_mode,
            "parameter_count": runtime.parameter_count,
            "mac_count": validated.expected_mac_count,
            "channel_plan": list(
                MODEL_VARIANT_CHANNELS[validated.model_variant]
            ),
            "dscnn_spec": json.loads(
                json.dumps(dscnn_spec(validated.model_variant))
            ),
            "framework": dict(sorted(runtime.framework.items())),
        },
        "tensor_contracts": {
            "input": runtime.input_spec.as_json(),
            "output": runtime.output_spec.as_json(),
            "output_semantics": (
                "unnormalized_candidate_verifier_logit"
                if deployment_logit_bound is None
                else "monotonic_bounded_candidate_verifier_logit"
            ),
            "fully_integer": True,
            "quantization_mode": quantization_mode,
        },
        "operator_contract": runtime.operator_audit,
        "static_memory_audit": {
            "batch_size": 1,
            "fixed_input_shape": True,
            "fixed_output_shape": True,
            "dynamic_tensor_shapes_forbidden": True,
            "interpreter_allocation": "allocate_once_then_reuse_without_resize",
            "activation_workspace": "single_verifier_invocation_reusable",
            "tensor_arena_bytes": None,
            "tensor_audit": runtime.tensor_audit,
            "hardware_high_water_measurement_required": True,
            "hardware_target": "exact StackChan ESP32-S3 revision",
        },
        "equivalence": equivalence,
        "threshold_contract": {
            "training_probability_threshold": validated.frozen_threshold,
            "training_logit_threshold": training_threshold_logit,
            "deployed_logit_threshold": deployment_threshold_logit,
            "deployment_logit_bound": deployment_logit_bound,
            "quantization_logit_safety_margin": quantization_safety_margin,
            "fit_split": "validation",
            "test_used_for_selection": False,
            "int8_threshold_retuning_performed": False,
            "joint_cascade_threshold_selection_remaining": True,
        },
        "provenance": {
            "verified_input_binding_count": len(validated.all_bindings) + 1,
            "trainer_module": {
                "path": str(TRAINER_MODULE),
                "sha256": sha256_file(TRAINER_MODULE),
            },
            "converter_module": {
                "path": str(CONVERTER_MODULE),
                "sha256": sha256_file(CONVERTER_MODULE),
            },
        },
        "limitations": [
            "not cascade-qualified",
            "test split was not used for conversion or equivalence",
            "joint detector-verifier thresholds and continuous FAPH remain unmeasured",
            "ESP32-S3 latency, tensor-arena high-water, and sustained stability remain unmeasured",
        ],
    }

    output = args.output.expanduser().resolve()
    artifact_path = output / ARTIFACT_FILENAME
    metadata_path = output / METADATA_FILENAME
    if artifact_path.exists() or metadata_path.exists():
        raise FileExistsError("refusing to overwrite existing verifier package")
    _atomic_write(artifact_path, artifact)
    _atomic_write(
        metadata_path,
        (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-examples", type=int, default=500)
    parser.add_argument("--equivalence-examples", type=int, default=128)
    parser.add_argument(
        "--deployment-logit-bound",
        type=float,
        default=0.0,
        help=(
            "Apply bound*tanh(logit/bound) in the deployment graph; zero "
            "keeps the training logit unchanged"
        ),
    )
    parser.add_argument(
        "--quantization-mode",
        choices=("int8", "int16-activations"),
        default="int8",
    )
    parser.add_argument(
        "--quantization-logit-safety-margin",
        type=float,
        default=0.0,
        help=(
            "Subtract a predeclared deployment-only logit margin to protect "
            "recall against measured integer equivalence error"
        ),
    )
    parser.add_argument(
        "--float-reference-max-absolute-error", type=float, default=1e-4
    )
    parser.add_argument(
        "--float-reference-mean-absolute-error", type=float, default=1e-5
    )
    parser.add_argument("--max-logit-absolute-error", type=float, default=0.75)
    parser.add_argument("--mean-logit-absolute-error", type=float, default=0.15)
    parser.add_argument("--max-probability-absolute-error", type=float, default=0.05)
    parser.add_argument(
        "--threshold-decision-mismatch-fraction", type=float, default=0.0
    )
    args = parser.parse_args(argv)
    try:
        metadata = convert(args)
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
