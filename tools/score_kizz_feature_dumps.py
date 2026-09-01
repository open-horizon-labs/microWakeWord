#!/usr/bin/env python3
"""Score exact StackChan verifier feature dumps with one or more TFLite models."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


FRAMES = 260
BINS = 40
FRONTEND_SCALE = 26.0 / 256.0
BEGIN = re.compile(r"KIZZ_FEATURE_DUMP begin capture=(\d+)")
FRAME = re.compile(r"KIZZ_FEATURE_DUMP frame=(\d{3}) data=([0-9a-fA-F]{80})")
END = re.compile(r"KIZZ_FEATURE_DUMP end capture=(\d+)")


def parse_captures(path: Path) -> tuple[np.ndarray, list[int]]:
    captures: list[np.ndarray] = []
    identifiers: list[int] = []
    current_id: int | None = None
    current: dict[int, bytes] = {}
    for line in path.read_text(errors="replace").splitlines():
        if match := BEGIN.search(line):
            if current_id is not None:
                raise ValueError(f"capture {current_id} ended implicitly")
            current_id = int(match.group(1))
            current = {}
        elif current_id is not None and (match := FRAME.search(line)):
            frame = int(match.group(1))
            current[frame] = bytes.fromhex(match.group(2))
        elif current_id is not None and (match := END.search(line)):
            end_id = int(match.group(1))
            if end_id != current_id:
                raise ValueError(f"capture id mismatch: {current_id} != {end_id}")
            missing = sorted(set(range(FRAMES)) - current.keys())
            if missing:
                raise ValueError(f"capture {current_id} missing frames: {missing}")
            raw = b"".join(current[frame] for frame in range(FRAMES))
            captures.append(np.frombuffer(raw, dtype=np.int8).reshape(FRAMES, BINS))
            identifiers.append(current_id)
            current_id = None
            current = {}
    if current_id is not None:
        raise ValueError(f"capture {current_id} is incomplete")
    if not captures:
        raise ValueError(f"no complete feature captures in {path}")
    return np.stack(captures), identifiers


def lround(values: np.ndarray) -> np.ndarray:
    return np.where(values >= 0.0, np.floor(values + 0.5), np.ceil(values - 0.5))


def score_model(tf: object, path: Path, features: np.ndarray) -> np.ndarray:
    runtime = tf.lite.Interpreter(
        model_path=str(path),
        experimental_op_resolver_type=(
            tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        ),
    )
    runtime.allocate_tensors()
    input_detail = runtime.get_input_details()[0]
    output_detail = runtime.get_output_details()[0]
    expected = (1, FRAMES, BINS, 1)
    if tuple(input_detail["shape"]) != expected:
        raise ValueError(f"unexpected input shape for {path}: {input_detail['shape']}")
    input_scale, input_zero = input_detail["quantization"]
    output_scale, output_zero = output_detail["quantization"]
    dtype = np.dtype(input_detail["dtype"])
    limits = np.iinfo(dtype)
    real = (features.astype(np.int16) + 128).astype(np.float32) * FRONTEND_SCALE
    quantized = np.clip(
        lround(real / input_scale) + input_zero, limits.min, limits.max
    ).astype(dtype)
    scores: list[float] = []
    for window in quantized:
        runtime.set_tensor(input_detail["index"], window[None, ..., None])
        runtime.invoke()
        raw = int(runtime.get_tensor(output_detail["index"]).reshape(-1)[0])
        scores.append((raw - output_zero) * output_scale)
    return np.asarray(scores, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("models", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threshold", type=float, default=-1.447065643966198)
    args = parser.parse_args()

    import tensorflow as tf

    features, identifiers = parse_captures(args.log)
    results: dict[str, np.ndarray] = {}
    for model in args.models:
        results[model.name] = score_model(tf, model, features)
    print("capture " + " ".join(results))
    for row, identifier in enumerate(identifiers):
        values = " ".join(
            f"{scores[row]: .6f}{'*' if scores[row] >= args.threshold else ''}"
            for scores in results.values()
        )
        print(f"{identifier:7d} {values}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            frontend_int8=features,
            capture_ids=np.asarray(identifiers),
            **results,
        )
        print(f"saved {len(features)} captures to {args.output}")


if __name__ == "__main__":
    main()
