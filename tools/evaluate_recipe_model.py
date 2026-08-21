#!/usr/bin/env python3
"""Report wake acceptance separately for every recipe pronunciation and foil."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from microwakeword.inference import Model


def reset_model(model: Model) -> None:
    reset = getattr(model.model, "reset_all_variables", None)
    if reset is not None:
        reset()


def peak_probability(
    model: Model,
    wav_path: Path,
    sliding_window: int,
    ignore_initial: int,
    clip_duration_ms: int,
) -> float:
    sample_rate, pcm = wavfile.read(wav_path)
    if pcm.dtype != np.int16:
        raise ValueError(f"{wav_path} must be signed-16 PCM")
    if pcm.ndim != 1:
        raise ValueError(f"{wav_path} must be mono")
    if sample_rate != 16000:
        divisor = np.gcd(sample_rate, 16000)
        pcm = resample_poly(
            pcm.astype(np.float32) / 32768.0,
            16000 // divisor,
            sample_rate // divisor,
        ).astype(np.float32)
    target_samples = 16000 * clip_duration_ms // 1000
    if pcm.shape[0] < target_samples:
        pcm = np.pad(pcm, (target_samples - pcm.shape[0], 0))
    elif pcm.shape[0] > target_samples:
        pcm = pcm[-target_samples:]
    reset_model(model)
    probabilities = np.asarray(model.predict_clip(pcm, step_ms=10), dtype=np.float32)
    probabilities = probabilities[ignore_initial:]
    if probabilities.size < sliding_window:
        return 0.0
    moving_average = np.convolve(
        probabilities, np.ones(sliding_window) / sliding_window, mode="valid"
    )
    return float(np.max(moving_average))


def evaluate_group(
    model_path: Path,
    group: Path,
    cutoff: float,
    sliding_window: int,
    ignore_initial: int,
    clip_duration_ms: int,
    limit: int,
) -> dict:
    model = Model(str(model_path), stride=3)
    clips = sorted(group.glob("*.wav"))
    if limit:
        clips = clips[:limit]
    peaks = [
        peak_probability(
            model, clip, sliding_window, ignore_initial, clip_duration_ms
        )
        for clip in clips
    ]
    accepted = sum(peak > cutoff for peak in peaks)
    return {
        "clips": len(peaks),
        "accepted": accepted,
        "acceptance_rate": accepted / len(peaks) if peaks else 0.0,
        "median_peak_probability": float(np.median(peaks)) if peaks else 0.0,
        "minimum_peak_probability": min(peaks, default=0.0),
        "maximum_peak_probability": max(peaks, default=0.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--cutoff", type=float, required=True)
    parser.add_argument("--sliding-window", type=int, default=5)
    parser.add_argument("--ignore-initial", type=int, default=25)
    parser.add_argument("--clip-duration-ms", type=int, default=2000)
    parser.add_argument("--limit-per-phrase", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = {
        "model": str(args.model),
        "cutoff": args.cutoff,
        "sliding_window": args.sliding_window,
        "positive": {},
        "hard_negative": {},
    }
    for truth in ("positive", "hard_negative"):
        for group in sorted((args.generated / truth).iterdir()):
            if group.is_dir():
                result[truth][group.name] = evaluate_group(
                    args.model,
                    group,
                    args.cutoff,
                    args.sliding_window,
                    args.ignore_initial,
                    args.clip_duration_ms,
                    args.limit_per_phrase,
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
