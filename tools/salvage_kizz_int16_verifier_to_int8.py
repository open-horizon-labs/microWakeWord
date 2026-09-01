#!/usr/bin/env python3
"""Re-encode the qualified Kizz fixed-window verifier for INT8 ESP-NN kernels.

This is a bounded artifact-salvage utility, not a qualification shortcut.  It
rebuilds the exact topology represented by the deployed INT16-activation TFLite
artifact, dequantizes that artifact's learned weights, and converts the rebuilt
graph to full INT8 using deterministic synthetic calibration windows.  The
result must still pass corpus and physical qualification before deployment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


EXPECTED_INPUT_SHAPE = (1, 260, 40, 1)
EXPECTED_OUTPUT_SHAPE = (1, 1)
EXPECTED_CHANNELS = (48, 64, 96, 128, 160)
EXPECTED_OPS = (
    "CONV_2D",
    "DEPTHWISE_CONV_2D",
    "CONV_2D",
    "DEPTHWISE_CONV_2D",
    "CONV_2D",
    "DEPTHWISE_CONV_2D",
    "CONV_2D",
    "DEPTHWISE_CONV_2D",
    "CONV_2D",
    "RESHAPE",
    "FULLY_CONNECTED",
    "TANH",
    "MUL",
)
LOGIT_BOUND = 4.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _interpreter(tf: Any, *, model_path: Path | None = None, model: bytes | None = None) -> Any:
    options: dict[str, Any] = {
        "experimental_op_resolver_type": (
            tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        )
    }
    if model_path is not None:
        options["model_path"] = str(model_path)
    else:
        options["model_content"] = model
    runtime = tf.lite.Interpreter(**options)
    runtime.allocate_tensors()
    return runtime


def _dequantized_tensor(runtime: Any, index: int) -> np.ndarray:
    detail = next(item for item in runtime.get_tensor_details() if item["index"] == index)
    raw = np.asarray(runtime.get_tensor(index))
    quant = detail["quantization_parameters"]
    scales = np.asarray(quant["scales"], dtype=np.float64)
    zeros = np.asarray(quant["zero_points"], dtype=np.float64)
    if not len(scales):
        return raw.astype(np.float32)
    if len(scales) == 1:
        return ((raw.astype(np.float64) - zeros[0]) * scales[0]).astype(np.float32)
    axis = int(quant["quantized_dimension"])
    shape = [1] * raw.ndim
    shape[axis] = len(scales)
    return (
        (raw.astype(np.float64) - zeros.reshape(shape)) * scales.reshape(shape)
    ).astype(np.float32)


def _validate_source(runtime: Any) -> None:
    inputs = runtime.get_input_details()
    outputs = runtime.get_output_details()
    if len(inputs) != 1 or tuple(inputs[0]["shape"]) != EXPECTED_INPUT_SHAPE:
        raise ValueError("source verifier input contract drift")
    if np.dtype(inputs[0]["dtype"]) != np.dtype(np.int16):
        raise ValueError("source verifier input is not int16")
    if len(outputs) != 1 or tuple(outputs[0]["shape"]) != EXPECTED_OUTPUT_SHAPE:
        raise ValueError("source verifier output contract drift")
    if np.dtype(outputs[0]["dtype"]) != np.dtype(np.int16):
        raise ValueError("source verifier output is not int16")
    ops = tuple(item["op_name"] for item in runtime._get_ops_details())  # noqa: SLF001
    if ops != EXPECTED_OPS:
        raise ValueError(f"source verifier topology drift: {ops}")


def _build_recovered_model(tf: Any, source: Any) -> Any:
    inputs = tf.keras.Input(batch_shape=EXPECTED_INPUT_SHAPE, name="log_mel_window")
    value = inputs
    layers: list[tuple[Any, int, int, bool]] = []

    def conv(name: str, filters: int, kernel: int, weight: int, bias: int) -> None:
        nonlocal value
        layer = tf.keras.layers.Conv2D(
            filters,
            (kernel, kernel),
            strides=(2, 2) if kernel != 1 else (1, 1),
            padding="same",
            activation="relu",
            use_bias=True,
            name=name,
        )
        value = layer(value)
        layers.append((layer, weight, bias, False))

    def depthwise(name: str, weight: int, bias: int) -> None:
        nonlocal value
        layer = tf.keras.layers.DepthwiseConv2D(
            (3, 3), strides=(2, 2), padding="same", activation="relu",
            use_bias=True, name=name,
        )
        value = layer(value)
        layers.append((layer, weight, bias, True))

    conv("stem", 48, 5, 21, 22)
    depthwise("ds1_depthwise", 18, 10)
    conv("ds1_pointwise", 64, 1, 17, 9)
    depthwise("ds2_depthwise", 16, 8)
    conv("ds2_pointwise", 96, 1, 15, 7)
    depthwise("ds3_depthwise", 14, 6)
    conv("ds3_pointwise", 128, 1, 13, 5)
    depthwise("ds4_depthwise", 12, 4)
    conv("ds4_pointwise", 160, 1, 11, 3)
    value = tf.keras.layers.Flatten(name="temporal_flatten")(value)
    dense = tf.keras.layers.Dense(1, use_bias=True, name="scaled_verifier_logit")
    value = dense(value)
    value = tf.keras.layers.Activation("tanh", name="deployment_logit_tanh")(value)
    value = tf.keras.layers.Rescaling(LOGIT_BOUND, name="deployment_logit")(value)
    model = tf.keras.Model(inputs, value, name="kizz_verifier_int8_salvage")

    for layer, weight_index, bias_index, is_depthwise in layers:
        weights = _dequantized_tensor(source, weight_index)
        if is_depthwise:
            weights = np.transpose(weights, (1, 2, 3, 0))
        else:
            weights = np.transpose(weights, (1, 2, 3, 0))
        layer.set_weights([weights, _dequantized_tensor(source, bias_index)])
    dense_weights = _dequantized_tensor(source, 2).T
    dense_bias = _dequantized_tensor(source, 1)
    dense.set_weights([dense_weights, dense_bias])

    if tuple(model.output_shape) != EXPECTED_OUTPUT_SHAPE:
        raise ValueError("rebuilt verifier output contract drift")
    if tuple(layer.output.shape[-1] for layer, *_ in layers if not isinstance(
        layer, tf.keras.layers.DepthwiseConv2D
    )) != EXPECTED_CHANNELS:
        raise ValueError("rebuilt verifier channel plan drift")
    return model


def _calibration_windows(count: int, seed: int) -> np.ndarray:
    """Create deterministic nonnegative, speech-like log-mel envelopes."""
    if count < 32:
        raise ValueError("calibration count must be at least 32")
    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, 1.0, 260, dtype=np.float32)
    frequency = np.linspace(0.0, 1.0, 40, dtype=np.float32)
    windows = np.empty((count, 260, 40, 1), dtype=np.float32)
    for index in range(count):
        floor = rng.uniform(0.0, 2.0)
        value = np.full((260, 40), floor, dtype=np.float32)
        for _ in range(int(rng.integers(2, 7))):
            center_t = rng.uniform(0.05, 0.95)
            center_f = rng.uniform(0.0, 0.8)
            width_t = rng.uniform(0.015, 0.20)
            width_f = rng.uniform(0.03, 0.30)
            amplitude = rng.uniform(1.0, 16.0)
            envelope_t = np.exp(-0.5 * ((time - center_t) / width_t) ** 2)
            envelope_f = np.exp(-0.5 * ((frequency - center_f) / width_f) ** 2)
            value += amplitude * np.outer(envelope_t, envelope_f)
        value += rng.normal(0.0, rng.uniform(0.02, 0.45), value.shape)
        windows[index, ..., 0] = np.clip(value, 0.0, 26.0)
    windows[0].fill(0.0)
    windows[1].fill(26.0)
    windows[2, 0, 0, 0] = 0.0
    windows[2, 0, 1, 0] = 26.0
    return windows


def _audio_windows(audio_dir: Path, count: int, seed: int) -> np.ndarray:
    """Build realistic candidate-history windows with the product C frontend."""
    import soundfile as sf

    from microwakeword.audio.audio_utils import generate_features_for_clip

    paths = sorted(audio_dir.expanduser().resolve().glob("*.wav"))
    if not paths:
        raise ValueError(f"calibration audio directory has no WAV files: {audio_dir}")
    rng = np.random.default_rng(seed)
    candidates: list[np.ndarray] = []
    for path in paths:
        samples, rate = sf.read(path, dtype="float32", always_2d=True)
        if rate != 16_000:
            raise ValueError(f"calibration audio must be 16 kHz: {path}")
        mono = np.mean(samples, axis=1)
        for gain in (0.25, 0.5, 1.0):
            for noise_level in (0.0, 0.002, 0.01):
                prefix = rng.normal(0.0, noise_level, 6 * rate).astype(np.float32)
                speech = np.clip(mono * gain, -1.0, 1.0)
                suffix = rng.normal(0.0, noise_level, rate).astype(np.float32)
                audio = np.concatenate((prefix, speech, suffix))
                features = np.asarray(
                    generate_features_for_clip(audio, use_c=True), dtype=np.float32
                ).reshape(-1, 40)
                if len(features) < 260:
                    continue
                speech_start = 6 * 100
                first_end = max(260, speech_start)
                for end in range(first_end, len(features) + 1, 10):
                    candidates.append(features[end - 260 : end, :, None])
    if len(candidates) < count:
        raise ValueError(
            f"calibration audio produced only {len(candidates)} windows, need {count}"
        )
    indexes = rng.choice(len(candidates), size=count, replace=False)
    windows = np.stack([candidates[int(index)] for index in indexes]).astype(np.float32)
    # Preserve the product frontend's declared extrema in calibration.
    windows[0, 0, 0, 0] = 0.0
    windows[0, 0, 1, 0] = 26.0
    return windows


def _representative(windows: np.ndarray) -> Iterable[list[np.ndarray]]:
    for window in windows:
        yield [window[None, ...].astype(np.float32)]


def _run(runtime: Any, sample: np.ndarray) -> float:
    input_detail = runtime.get_input_details()[0]
    output_detail = runtime.get_output_details()[0]
    scale, zero = input_detail["quantization"]
    limits = np.iinfo(input_detail["dtype"])
    quantized = np.clip(np.rint(sample / scale + zero), limits.min, limits.max)
    runtime.set_tensor(input_detail["index"], quantized.astype(input_detail["dtype"])[None])
    runtime.invoke()
    raw = runtime.get_tensor(output_detail["index"])[0, 0]
    output_scale, output_zero = output_detail["quantization"]
    return float((float(raw) - output_zero) * output_scale)


def salvage(
    source_path: Path,
    output: Path,
    *,
    count: int,
    seed: int,
    calibration_audio_dir: Path | None = None,
) -> Mapping[str, Any]:
    import tensorflow as tf

    source_path = source_path.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    source = _interpreter(tf, model_path=source_path)
    _validate_source(source)
    model = _build_recovered_model(tf, source)
    calibration = (
        _audio_windows(calibration_audio_dir, count, seed)
        if calibration_audio_dir is not None
        else _calibration_windows(count, seed)
    )

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.representative_dataset = tf.lite.RepresentativeDataset(
        lambda: _representative(calibration)
    )
    artifact = bytes(converter.convert())
    recovered = _interpreter(tf, model=artifact)
    input_detail = recovered.get_input_details()[0]
    output_detail = recovered.get_output_details()[0]
    ops = tuple(item["op_name"] for item in recovered._get_ops_details())  # noqa: SLF001
    if tuple(input_detail["shape"]) != EXPECTED_INPUT_SHAPE or input_detail["dtype"] != np.int8:
        raise ValueError("recovered verifier input is not static int8")
    if tuple(output_detail["shape"]) != EXPECTED_OUTPUT_SHAPE or output_detail["dtype"] != np.int8:
        raise ValueError("recovered verifier output is not static int8")
    if ops != EXPECTED_OPS:
        raise ValueError(f"recovered verifier topology drift: {ops}")
    if any(np.issubdtype(item["dtype"], np.floating) for item in recovered.get_tensor_details()):
        raise ValueError("recovered verifier contains floating-point tensors")

    validation = (
        _audio_windows(calibration_audio_dir, max(128, count // 2), seed + 1)
        if calibration_audio_dir is not None
        else _calibration_windows(max(128, count // 2), seed + 1)
    )
    source_scores = np.asarray([_run(source, item) for item in validation])
    recovered_scores = np.asarray([_run(recovered, item) for item in validation])
    errors = np.abs(source_scores - recovered_scores)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(artifact)
    report = {
        "schema_version": 1,
        "qualification": False,
        "warning": "salvaged quantization experiment; corpus and hardware requalification required",
        "source": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "bytes": source_path.stat().st_size,
            "tensor_contract": "int16[1,260,40,1] -> int16[1,1]",
        },
        "artifact": {
            "path": str(output),
            "sha256": hashlib.sha256(artifact).hexdigest(),
            "bytes": len(artifact),
            "tensor_contract": "int8[1,260,40,1] -> int8[1,1]",
        },
        "recovery": {
            "learned_parameters": "dequantized from source TFLite constants",
            "topology": "xwide fixed-window DS-CNN with bounded logit head",
            "channel_plan": list(EXPECTED_CHANNELS),
            "calibration": (
                "repository C microfrontend windows from supplied 16 kHz WAV audio"
                if calibration_audio_dir is not None
                else "deterministic synthetic nonnegative speech-like log-mel envelopes"
            ),
            "calibration_audio_dir": (
                str(calibration_audio_dir.expanduser().resolve())
                if calibration_audio_dir is not None
                else None
            ),
            "calibration_windows": count,
            "seed": seed,
        },
        "synthetic_equivalence": {
            "windows": len(validation),
            "mean_absolute_logit_error": float(np.mean(errors)),
            "max_absolute_logit_error": float(np.max(errors)),
            "correlation": float(np.corrcoef(source_scores, recovered_scores)[0, 1]),
            "source_range": [float(np.min(source_scores)), float(np.max(source_scores))],
            "artifact_range": [float(np.min(recovered_scores)), float(np.max(recovered_scores))],
        },
        "framework": {"tensorflow": str(tf.__version__)},
    }
    report_path = output.with_suffix(output.suffix + ".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-windows", type=int, default=512)
    parser.add_argument("--calibration-audio-dir", type=Path)
    parser.add_argument("--seed", type=int, default=248)
    args = parser.parse_args()
    report = salvage(
        args.source,
        args.output,
        count=args.calibration_windows,
        seed=args.seed,
        calibration_audio_dir=args.calibration_audio_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
