#!/usr/bin/env python3
"""Build and benchmark the default quantized ordered-state streaming model."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import tensorflow as tf

from microwakeword import ordered_state_model, utils
from microwakeword.layers import modes
from microwakeword.ordered_state import KIZZ_TOPOLOGY


def tensor_bytes(detail: dict[str, Any]) -> int:
    """Return storage bytes for one concrete tensor detail."""
    dtype = np.dtype(detail["dtype"])
    if dtype.hasobject:
        return 0
    return int(np.prod(detail["shape"], dtype=np.int64)) * dtype.itemsize


def persistent_stream_state_bytes(details: Sequence[dict[str, Any]]) -> int:
    """Estimate persistent ring-buffer bytes in the converted TFLite graph."""
    return sum(
        tensor_bytes(detail) for detail in details if "ReadVariableOp" in detail["name"]
    )


def latency_summary(milliseconds: Sequence[float]) -> dict[str, float | int]:
    ordered = sorted(milliseconds)
    p95_index = min(len(ordered) - 1, int(np.ceil(len(ordered) * 0.95)) - 1)
    return {
        "iterations": len(ordered),
        "mean_ms": statistics.fmean(ordered),
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[p95_index],
    }


class _CalibrationData:
    def __init__(self, spectrogram_length: int):
        self.spectrogram_length = spectrogram_length

    def get_data(self, *_args, **_kwargs):
        samples = np.zeros((8, self.spectrogram_length, 40), dtype=np.float32)
        samples[0, 0, 1] = 26.0
        return samples, np.zeros(8), np.ones(8)


def build_report(iterations: int = 1000, spectrogram_length: int = 96) -> dict:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    parser = argparse.ArgumentParser(add_help=False)
    ordered_state_model.model_parameters(parser)
    flags = parser.parse_args([])
    training_model = ordered_state_model.model(
        flags, (spectrogram_length, 40), batch_size=1
    )
    config = {
        "spectrogram_length": spectrogram_length,
        "stride": flags.stride,
        "train_dir": None,
    }
    with tempfile.TemporaryDirectory() as temporary:
        config["train_dir"] = temporary
        saved = Path(temporary) / "stream"
        streaming_model = utils.convert_model_saved(
            training_model,
            config,
            folder="stream",
            mode=modes.Modes.STREAM_INTERNAL_STATE_INFERENCE,
        )
        output_dir = Path(temporary) / "tflite"
        utils.convert_saved_model_to_tflite(
            config,
            _CalibrationData(spectrogram_length),
            str(saved),
            str(output_dir),
            "ordered_state.tflite",
            quantize=True,
        )
        model_path = output_dir / "ordered_state.tflite"
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        input_detail = interpreter.get_input_details()[0]
        output_detail = interpreter.get_output_details()[0]
        interpreter.set_tensor(
            input_detail["index"],
            np.zeros(input_detail["shape"], dtype=input_detail["dtype"]),
        )
        interpreter.invoke()
        durations = []
        for _ in range(iterations):
            started = time.perf_counter_ns()
            interpreter.invoke()
            durations.append((time.perf_counter_ns() - started) / 1_000_000.0)
        details = interpreter.get_tensor_details()
        return {
            "architecture": {
                "training_model_parameters": training_model.count_params(),
                "streaming_model_parameters": streaming_model.count_params(),
                "receptive_field_ms": ordered_state_model.receptive_field_ms(flags),
                "state_count": KIZZ_TOPOLOGY.state_count,
                "stride_ms": flags.stride * 10,
            },
            "artifact": {
                "tflite_bytes": model_path.stat().st_size,
                "input_shape": input_detail["shape"].tolist(),
                "input_dtype": np.dtype(input_detail["dtype"]).name,
                "output_shape": output_detail["shape"].tolist(),
                "output_dtype": np.dtype(output_detail["dtype"]).name,
            },
            "state_memory_estimate": {
                "tflite_persistent_ring_buffers_bytes": (
                    persistent_stream_state_bytes(details)
                ),
                "decoder_float32_scores_and_int32_starts_bytes": (
                    KIZZ_TOPOLOGY.ordered_state_count * 8
                ),
            },
            "desktop_inference": latency_summary(durations),
            "environment": {
                "machine": platform.machine(),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "tensorflow": tf.__version__,
            },
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = build_report(args.iterations)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
