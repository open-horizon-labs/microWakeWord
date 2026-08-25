#!/usr/bin/env python3
"""Convert the distilled causal student to the firmware streaming TFLite form."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import tensorflow as tf

from microwakeword import utils
from microwakeword.layers import modes
from microwakeword.ordered_state_model import model as build_student
from tools.distill_kizz_student import student_flags


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--representative-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    student = build_student(student_flags(), (260, 40), None)
    student.load_weights(args.weights)
    config = {
        "train_dir": str(args.output),
        "spectrogram_length": 260,
        "stride": 3,
    }
    utils.convert_model_saved(
        student,
        config,
        folder="stream_state_internal",
        mode=modes.Modes.STREAM_INTERNAL_STATE_INFERENCE,
    )

    features = np.load(args.representative_features, mmap_mode="r")

    def representative_dataset():
        for index in range(min(500, len(features))):
            # ``features`` is opened as a read-only memmap. Calibration tweaks
            # must operate on a writable per-example copy.
            spectrogram = np.array(features[index], dtype=np.float32, copy=True)
            spectrogram[0, 0] = 0.0
            spectrogram[0, 1] = 26.0
            for offset in range(0, len(spectrogram) - 2, 3):
                yield [spectrogram[offset : offset + 3]]

    source = args.output / "stream_state_internal"
    converter = tf.lite.TFLiteConverter.from_saved_model(str(source))
    converter.optimizations = {tf.lite.Optimize.DEFAULT}
    converter._experimental_variable_quantization = True
    converter.target_spec.supported_ops = {tf.lite.OpsSet.TFLITE_BUILTINS_INT8}
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.uint8
    converter.representative_dataset = tf.lite.RepresentativeDataset(representative_dataset)
    artifact = converter.convert()
    output = args.output / "student_stream_state_internal_quant.tflite"
    output.write_bytes(artifact)

    interpreter = tf.lite.Interpreter(model_path=str(output))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    metadata = {
        "schema_version": 1,
        "artifact": str(output),
        "bytes": len(artifact),
        "input": {
            "shape": [int(value) for value in input_detail["shape"]],
            "dtype": str(input_detail["dtype"]),
            "quantization": [float(value) for value in input_detail["quantization"]],
        },
        "output": {
            "shape": [int(value) for value in output_detail["shape"]],
            "dtype": str(output_detail["dtype"]),
            "quantization": [float(value) for value in output_detail["quantization"]],
        },
    }
    (args.output / "firmware-artifact.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
