#!/usr/bin/env python3
"""Convert and audit the compact Kizz Control student deployment boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import tensorflow as tf

from microwakeword import utils
from microwakeword.ctc_forward import exhaustive_suffix_forward_score
from microwakeword.kizz_viterbi_decoder import exhaustive_suffix_score
from microwakeword.layers import modes
from microwakeword.ordered_state_model import model as build_student
from microwakeword.phoneme_student import (
    compact_phone_contract,
    student_output_times_seconds,
    student_stream_phase_offset_frames,
)
from tools.distill_kizz_phoneme_student import (
    student_decoder_contract,
    student_decoder_contract_hash,
)
from tools.distill_kizz_student import student_flags

INPUT_SHAPE = (260, 40)
FEATURE_STEP_SECONDS = 0.010
OUTPUT_STEP_SECONDS = 0.030
VITERBI_DECODER_MODULE = (
    Path(__file__).resolve().parents[1] / "microwakeword/kizz_viterbi_decoder.py"
)
FORWARD_SUM_DECODER_MODULE = (
    Path(__file__).resolve().parents[1] / "microwakeword/ctc_forward.py"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def architecture_contract(contract: dict) -> dict:
    """Return the exact Keras architecture contract for this phone vocabulary."""
    return {
        "input_shape": list(INPUT_SHAPE), "output_frames": 66,
        "output_count": len(contract["tokens"]),
        "pointwise_filters": [96, 96, 96, 96],
        "residual_connection": [0, 0, 0, 0], "repeat_in_block": [1, 1, 1, 1],
        "mixconv_kernel_sizes": [[3], [5], [7], [9]],
        "first_conv_filters": 48, "first_conv_kernel_size": 5, "stride": 3,
    }


def _flags(contract: dict) -> SimpleNamespace:
    if contract != compact_phone_contract():
        raise ValueError("unsupported or drifted compact phone contract")
    return student_flags(len(contract["tokens"]))


def load_distillation_contract(path: Path, weights: Path) -> tuple[dict, dict, str, str]:
    """Validate immutable distillation/weight binding before building Keras."""
    metadata = json.loads(path.read_text())
    contract = metadata.get("compact_phone_contract")
    if contract != compact_phone_contract():
        raise ValueError("distillation metadata compact phone contract differs")
    if metadata.get("recipe") not in {
        f"kizz_control_compact_ctc_distillation_v{version}"
        for version in range(1, 7)
    }:
        raise ValueError("unsupported distillation recipe")
    architecture = metadata.get("architecture")
    if architecture is None:
        raise ValueError("distillation metadata has no exact architecture contract")
    if architecture != architecture_contract(contract):
        raise ValueError("distillation architecture contract differs")
    decoder = metadata.get("decoder") or {}
    decoder_contract = decoder.get("contract") or {}
    decoder_algorithm = decoder_contract.get("algorithm")
    if (
        decoder_contract != student_decoder_contract(contract, decoder_algorithm)
        or decoder.get("contract_sha256")
        != student_decoder_contract_hash(contract, decoder_algorithm)
    ):
        raise ValueError("distillation metadata decoder contract differs")
    weights_hash = sha256_file(weights)
    declared = metadata.get("student", {}).get("weights_sha256")
    if not declared:
        raise ValueError("distillation metadata does not bind exact student weights")
    if declared != weights_hash:
        raise ValueError("student weights do not match distillation metadata")
    return metadata, contract, sha256_file(path), weights_hash


def _spread_indices(length: int, count: int) -> np.ndarray:
    """Select a deterministic corpus-wide sample instead of a prefix slice."""
    if length < 1 or count < 1:
        raise ValueError("sample length and count must be positive")
    return np.unique(
        np.linspace(0, length - 1, min(count, length), dtype=np.int64)
    )


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


def _representative_dataset(features: np.ndarray, stride: int, phase_offset: int):
    for index in _spread_indices(len(features), 500):
        spectrogram = np.array(features[index], dtype=np.float32, copy=True)
        if spectrogram.shape != INPUT_SHAPE:
            raise ValueError(f"representative feature shape must be {INPUT_SHAPE}")
        spectrogram[0, 0] = 0.0; spectrogram[0, 1] = 26.0
        for chunk in _stream_input_chunks(spectrogram, stride, phase_offset):
            yield [chunk]


def _as_sequence(value: Any, output_count: int) -> np.ndarray:
    values = np.asarray(value, dtype=np.float32)
    if values.ndim == 3: values = values[0]
    if values.ndim != 2 or values.shape[-1] != output_count:
        raise ValueError(f"model output must be [time, {output_count}], got {values.shape}")
    return values


def _stream_keras(
    model: Any,
    features: np.ndarray,
    stride: int,
    expected: int,
    phase_offset: int = 0,
) -> np.ndarray:
    emitted = []
    for chunk in _stream_input_chunks(features, stride, phase_offset):
        result = _as_sequence(model(chunk[None, ...], training=False), expected)
        emitted.extend(result)
    return np.asarray(emitted, dtype=np.float32)


def _reset_streaming_model(model: Any) -> None:
    """Reset internal state between independent held-out examples."""
    for layer in getattr(model, "submodules", model.layers):
        state = getattr(layer, "states", None)
        if state is not None:
            state.assign(tf.zeros_like(state))


def _stream_tflite(
    path: Path,
    features: np.ndarray,
    stride: int,
    expected: int,
    phase_offset: int = 0,
) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_path=str(path)); interpreter.allocate_tensors()
    inputs, outputs = interpreter.get_input_details(), interpreter.get_output_details()
    if len(inputs) != 1 or len(outputs) != 1: raise ValueError("streaming artifact must have one input and output")
    input_detail, output_detail = inputs[0], outputs[0]; result = []
    for chunk in _stream_input_chunks(features, stride, phase_offset):
        values = chunk.astype(np.float32)
        if np.issubdtype(np.dtype(input_detail["dtype"]), np.integer):
            scale, zero = input_detail["quantization"]
            if not np.isfinite(scale) or scale <= 0: raise ValueError("streaming input has invalid quantization")
            info = np.iinfo(input_detail["dtype"]); values = np.clip(np.rint(values / scale + zero), info.min, info.max).astype(input_detail["dtype"])
        interpreter.set_tensor(input_detail["index"], values[None, ...]); interpreter.invoke()
        raw = interpreter.get_tensor(output_detail["index"])
        if np.issubdtype(np.dtype(output_detail["dtype"]), np.integer):
            scale, zero = output_detail["quantization"]
            if not np.isfinite(scale) or scale <= 0: raise ValueError("streaming output has invalid quantization")
            raw = (np.asarray(raw, dtype=np.float32) - zero) * scale
        result.extend(_as_sequence(raw, expected))
    return np.asarray(result, dtype=np.float32)


def _causal_slice(
    values: np.ndarray, flags: Any, output_frames: int, phase_offset: int
) -> np.ndarray:
    """Select output centers matching the offline causal timeline."""
    # A stride-3 internal-state call emits one output per complete 3-frame
    # chunk.  The offline graph emits only after its valid receptive field.
    # Derive the discarded prefix from both geometries; never hand-pick an
    # offset (the old implementation's offset was the source of a real drift).
    total_chunks = (
        int(phase_offset > 0)
        + (INPUT_SHAPE[0] - phase_offset) // int(flags.stride)
    )
    warmup_chunks = total_chunks - output_frames
    expected_first = student_output_times_seconds(flags, output_frames)[0]
    if warmup_chunks < 0 or not np.isfinite(expected_first):
        raise ValueError("invalid causal streaming geometry")
    start = max(0, warmup_chunks); stop = start + output_frames
    if len(values) < stop: raise ValueError(f"streaming model emitted {len(values)} outputs; need {stop}")
    return values[start:stop]


def _decoder_score(
    sequence: np.ndarray, contract: dict, decoder_algorithm: str
):
    scorer = (
        exhaustive_suffix_forward_score
        if decoder_algorithm == "forward_sum_ctc"
        else exhaustive_suffix_score
        if decoder_algorithm == "max_add_ctc_viterbi"
        else None
    )
    if scorer is None:
        raise ValueError("unsupported student decoder algorithm")
    return scorer(
        sequence, contract, window_lengths=(19, 23, 27, 32, 39, 47, 54), beta=0.0
    )


def _decoder_decision(
    sequence: np.ndarray,
    contract: dict,
    decoder_algorithm: str = "forward_sum_ctc",
) -> bool:
    score = _decoder_score(sequence, contract, decoder_algorithm)
    return bool(np.isfinite(score.canonical_fit) and score.collision_margin >= 0.0)


def equivalence_report(offline: np.ndarray, streaming: np.ndarray, quantized: np.ndarray, *, contract: dict | None = None, decoder_algorithm: str = "forward_sum_ctc") -> dict:
    """Report numerical and argmax-decision equivalence for aligned tensors."""
    arrays = [np.asarray(value, dtype=np.float32) for value in (offline, streaming, quantized)]
    if any(value.shape != arrays[0].shape for value in arrays):
        raise ValueError(f"equivalence tensors have different shapes: {[value.shape for value in arrays]}")
    baseline = arrays[0]; report = {"shape": list(baseline.shape), "paths": {}}
    for name, value in zip(("tf_streaming", "int8_tflite"), arrays[1:]):
        delta = np.abs(baseline - value); base_decisions = np.argmax(baseline, axis=-1)
        decisions = np.argmax(value, axis=-1)
        if contract is None:
            mismatch_count = int(np.sum(base_decisions != decisions))
            decision_kind = "frame_argmax"
        else:
            baseline_decision = _decoder_decision(
                baseline, contract, decoder_algorithm
            )
            decision = _decoder_decision(value, contract, decoder_algorithm)
            mismatch_count = int(baseline_decision != decision)
            decision_kind = "suffix_ctc_acceptance"
        report["paths"][name] = {"max_abs": float(np.max(delta)), "mean_abs": float(np.mean(delta)), "decision_kind": decision_kind, "decision_mismatch_count": mismatch_count, "decision_mismatch_rate": float(mismatch_count > 0)}
    return report


def require_equivalence(report: dict, *, max_abs: float, max_mean_abs: float, max_decision_mismatch: float) -> None:
    for name, item in report["paths"].items():
        if item["max_abs"] > max_abs or item["mean_abs"] > max_mean_abs or item["decision_mismatch_rate"] > max_decision_mismatch:
            raise ValueError(f"{name} failed offline/streaming equivalence: {item}")


def convert(args: argparse.Namespace) -> dict:
    metadata, contract, metadata_hash, weights_hash = load_distillation_contract(args.distillation_metadata, args.weights)
    decoder_contract = metadata["decoder"]["contract"]
    decoder_algorithm = decoder_contract["algorithm"]
    decoder_module = (
        FORWARD_SUM_DECODER_MODULE
        if decoder_algorithm == "forward_sum_ctc"
        else VITERBI_DECODER_MODULE
    )
    flags = _flags(contract)
    phase_offset = student_stream_phase_offset_frames(flags)
    model = build_student(flags, INPUT_SHAPE, None); model.load_weights(args.weights)
    output_frames = int(metadata.get("student_output_frames", 66))
    if output_frames != 66: raise ValueError("distillation output timeline must contain 66 frames")
    features = np.load(args.representative_features, mmap_mode="r")
    if len(features) < 1: raise ValueError("representative feature set is empty")
    config = {"train_dir": str(args.output), "spectrogram_length": INPUT_SHAPE[0], "stride": flags.stride}
    # ``convert_model_saved`` mutates/wraps the passed model into its streaming
    # graph. Keep an independent offline graph for the equivalence baseline.
    offline_model = build_student(flags, INPUT_SHAPE, None); offline_model.load_weights(args.weights)
    streaming_model = utils.convert_model_saved(model, config, folder="stream_state_internal", mode=modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
    source = args.output / "stream_state_internal"; converter = tf.lite.TFLiteConverter.from_saved_model(str(source))
    converter.optimizations = {tf.lite.Optimize.DEFAULT}; converter._experimental_variable_quantization = True
    # ESPHome's microWakeWord runtime consumes int8 frontend features and a
    # uint8 quantized output tensor.  Weights and internal activations remain
    # fully integer-quantized; only the output's zero point is unsigned.
    converter.target_spec.supported_ops = {tf.lite.OpsSet.TFLITE_BUILTINS_INT8}; converter.inference_input_type = tf.int8; converter.inference_output_type = tf.uint8
    converter.representative_dataset = tf.lite.RepresentativeDataset(lambda: _representative_dataset(features, flags.stride, phase_offset))
    artifact = converter.convert(); artifact_path = args.output / "kizz_control_student_streaming_int8.tflite"; artifact_path.write_bytes(artifact)
    count = len(contract["tokens"]); per_example = []
    equivalence_examples = int(getattr(args, "equivalence_examples", 32))
    equivalence_indices = _spread_indices(len(features), equivalence_examples)
    for example in np.asarray(features[equivalence_indices]):
        sample = np.asarray(example, dtype=np.float32)
        if sample.shape != INPUT_SHAPE: raise ValueError(f"equivalence feature shape must be {INPUT_SHAPE}")
        offline = _as_sequence(offline_model(sample[None, ...], training=False), count)
        _reset_streaming_model(streaming_model)
        tf_stream = _causal_slice(_stream_keras(streaming_model, sample, flags.stride, count, phase_offset), flags, output_frames, phase_offset)
        int8_stream = _causal_slice(_stream_tflite(artifact_path, sample, flags.stride, count, phase_offset), flags, output_frames, phase_offset)
        per_example.append(
            equivalence_report(
                offline,
                tf_stream,
                int8_stream,
                contract=contract,
                decoder_algorithm=decoder_algorithm,
            )
        )
    if not per_example: raise ValueError("equivalence feature set is empty")
    report = {
        "examples": len(per_example),
        "example_indices": equivalence_indices.tolist(),
        "shape": per_example[0]["shape"],
        "limits": {
            "max_abs": float(args.max_abs),
            "max_mean_abs": float(args.max_mean_abs),
            "max_decision_mismatch": float(args.max_decision_mismatch),
        },
        "paths": {},
    }
    for name in per_example[0]["paths"]:
        items = [item["paths"][name] for item in per_example]
        report["paths"][name] = {"max_abs": max(item["max_abs"] for item in items), "mean_abs": float(np.mean([item["mean_abs"] for item in items])), "decision_kind": items[0]["decision_kind"], "decision_mismatch_count": sum(item["decision_mismatch_count"] for item in items), "decision_mismatch_rate": float(np.mean([item["decision_mismatch_rate"] for item in items]))}
    require_equivalence(report, max_abs=args.max_abs, max_mean_abs=args.max_mean_abs, max_decision_mismatch=args.max_decision_mismatch)
    interpreter = tf.lite.Interpreter(model_path=str(artifact_path)); interpreter.allocate_tensors(); input_detail, output_detail = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    immutable = {"schema_version": 2, "artifact": {"filename": artifact_path.name, "sha256": sha256_file(artifact_path), "bytes": len(artifact)}, "source": {"distillation_metadata": str(args.distillation_metadata.resolve()), "distillation_metadata_sha256": metadata_hash, "weights": str(args.weights.resolve()), "weights_sha256": weights_hash, "representative_features": str(args.representative_features.resolve()), "representative_features_sha256": sha256_file(args.representative_features)}, "compact_phone_contract": contract, "architecture": architecture_contract(contract), "timeline": {"feature_step_seconds": FEATURE_STEP_SECONDS, "output_step_seconds": OUTPUT_STEP_SECONDS, "output_frames": output_frames, "output_times_seconds": student_output_times_seconds(flags, output_frames).tolist(), "stream_phase_offset_frames": phase_offset, "stream_phase_priming": "zero_prefix_then_observed_prefix", "causal_warmup_derived": True}, "input": {"shape": [int(v) for v in input_detail["shape"]], "dtype": np.dtype(input_detail["dtype"]).name, "quantization": [float(v) for v in input_detail["quantization"]]}, "output": {"shape": [int(v) for v in output_detail["shape"]], "dtype": np.dtype(output_detail["dtype"]).name, "quantization": [float(v) for v in output_detail["quantization"]]}, "equivalence": report, "decoder": {"type": ("deterministic_suffix_forward_sum_ctc" if decoder_algorithm == "forward_sum_ctc" else "deterministic_suffix_viterbi_ctc"), "algorithm": decoder_algorithm, "contract_sha256": metadata["decoder"]["contract_sha256"], "distillation_decoder_contract": decoder_contract, "distillation_decoder_contract_sha256": metadata["decoder"]["contract_sha256"], "reference_module": str(decoder_module), "reference_module_sha256": sha256_file(decoder_module)}}
    (args.output / "firmware-artifact.json").write_text(json.dumps(immutable, indent=2, sort_keys=True) + "\n")
    return immutable


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--weights", type=Path, required=True); parser.add_argument("--distillation-metadata", type=Path, required=True); parser.add_argument("--representative-features", type=Path, required=True); parser.add_argument("--equivalence-examples", type=int, default=32); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--max-abs", type=float, default=2.0); parser.add_argument("--max-mean-abs", type=float, default=0.15); parser.add_argument("--max-decision-mismatch", type=float, default=0.10)
    args = parser.parse_args(argv)
    if min(args.max_abs, args.max_mean_abs, args.max_decision_mismatch) < 0 or args.equivalence_examples < 1: parser.error("equivalence limits/examples must be positive")
    args.output.mkdir(parents=True, exist_ok=True); print(json.dumps(convert(args), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
