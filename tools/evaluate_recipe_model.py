#!/usr/bin/env python3
"""Report wake acceptance separately for every recipe pronunciation and foil."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import datasets
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from microwakeword.audio.clips import Clips
from microwakeword.inference import Model
from microwakeword.synthetic_quality import load_quality_mask


def reset_model(model: Model) -> None:
    model.reset_states()


def peak_probability(
    model: Model,
    wav_path: Path,
    sliding_window: int,
    ignore_initial: int,
    clip_duration_ms: int,
    reset_state: bool = True,
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
    if clip_duration_ms:
        target_samples = 16000 * clip_duration_ms // 1000
        if pcm.shape[0] < target_samples:
            pcm = np.pad(pcm, (target_samples - pcm.shape[0], 0))
        elif pcm.shape[0] > target_samples:
            pcm = pcm[-target_samples:]
    if reset_state:
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
    clips: list[Path],
    cutoff: float,
    sliding_window: int,
    ignore_initial: int,
    clip_duration_ms: int,
    limit: int,
) -> dict:
    model = Model(str(model_path), stride=3)
    clips = sorted(clips)
    if limit:
        clips = clips[:limit]
    peaks = [
        peak_probability(model, clip, sliding_window, ignore_initial, clip_duration_ms)
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


def clips_by_group(
    root: Path,
    split: str,
    split_seed: int,
    rejected: set[Path] | None = None,
    generation_manifest: dict | None = None,
    class_name: str | None = None,
) -> dict[str, list[Path]]:
    if not root.exists():
        return {}
    rejected = rejected or set()
    if generation_manifest and generation_manifest.get("schema_version") == 2:
        grouped: dict[str, list[Path]] = defaultdict(list)
        for item in generation_manifest.get("plan", []):
            if item.get("class") != class_name:
                continue
            if split != "all" and item.get("split") != split:
                continue
            grouped[item["group"]].extend(
                path
                for path in sorted(Path(item["output"]).glob("*.wav"))
                if path.resolve() not in rejected
            )
        return dict(grouped)
    if split == "all":
        return {
            group.name: [
                path
                for path in sorted(group.glob("*.wav"))
                if path.resolve() not in rejected
            ]
            for group in sorted(root.iterdir())
            if group.is_dir()
        }
    clips = Clips(
        input_directory=str(root),
        file_pattern="**/*.wav",
        max_clip_duration_s=None,
        remove_silence=False,
        random_split_seed=split_seed,
        split_count=0.1,
    )
    grouped: dict[str, list[Path]] = defaultdict(list)
    held_out = clips.split_clips[split].cast_column(
        "audio", datasets.Audio(sampling_rate=16000, decode=False)
    )
    for audio in held_out["audio"]:
        path = Path(audio["path"])
        if path.resolve() not in rejected:
            grouped[path.parent.name].append(path)
    return dict(grouped)


def phrase_labels(generated: Path) -> dict[str, str]:
    manifest_path = generated / "generation-manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text())
    return {
        item.get("group", Path(item["output"]).name): item["text"]
        for item in manifest.get("plan", [])
    }


def clips_by_age_group(
    manifest: dict,
    class_name: str,
    split: str,
    rejected: set[Path],
) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for item in manifest.get("plan", []):
        if item.get("class") != class_name:
            continue
        if split != "all" and item.get("split") != split:
            continue
        age_group = item.get("age_group", "unknown")
        grouped[age_group].extend(
            path
            for path in sorted(Path(item["output"]).glob("*.wav"))
            if path.resolve() not in rejected
        )
    return dict(grouped)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument(
        "--quality-mask",
        type=Path,
        help="Exclude synthetic clips rejected by this provenance-bound mask",
    )
    parser.add_argument("--cutoff", type=float, required=True)
    parser.add_argument("--sliding-window", type=int, default=5)
    parser.add_argument("--ignore-initial", type=int, default=25)
    parser.add_argument(
        "--clip-duration-ms",
        type=int,
        default=0,
        help="Optionally crop clips to this duration; 0 evaluates the full recording",
    )
    parser.add_argument("--limit-per-phrase", type=int, default=0)
    parser.add_argument(
        "--split",
        choices=("all", "test", "validation"),
        default="test",
        help="Evaluate the exact held-out split used during feature generation",
    )
    parser.add_argument("--split-seed", type=int, default=231)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.quality_mask and not args.recipe:
        parser.error("--quality-mask requires --recipe")

    rejected: set[Path] = set()
    if args.quality_mask:
        mask = load_quality_mask(
            args.quality_mask,
            args.recipe,
            args.generated / "generation-manifest.json",
        )
        rejected = {
            (args.generated / relative).resolve() for relative in mask["rejected"]
        }

    generation_manifest_path = args.generated / "generation-manifest.json"
    generation_manifest = (
        json.loads(generation_manifest_path.read_text())
        if generation_manifest_path.exists()
        else None
    )
    result = {
        "model": str(args.model),
        "cutoff": args.cutoff,
        "sliding_window": args.sliding_window,
        "split": args.split,
        "split_seed": args.split_seed,
        "quality_mask": str(args.quality_mask) if args.quality_mask else None,
        "positive": {},
        "hard_negative": {},
        "age_cohorts": {"positive": {}, "hard_negative": {}},
    }
    labels = phrase_labels(args.generated)
    for truth in ("positive", "hard_negative"):
        seed = args.split_seed + (1 if truth == "hard_negative" else 0)
        grouped = clips_by_group(
            args.generated / truth,
            args.split,
            seed,
            rejected,
            generation_manifest,
            truth,
        )
        for name, clips in sorted(grouped.items()):
            result[truth][labels.get(name, name)] = evaluate_group(
                args.model,
                clips,
                args.cutoff,
                args.sliding_window,
                args.ignore_initial,
                args.clip_duration_ms,
                args.limit_per_phrase,
            )
        if generation_manifest and generation_manifest.get("schema_version") == 2:
            for age_group, clips in sorted(
                clips_by_age_group(
                    generation_manifest,
                    truth,
                    args.split,
                    rejected,
                ).items()
            ):
                result["age_cohorts"][truth][age_group] = evaluate_group(
                    args.model,
                    clips,
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
