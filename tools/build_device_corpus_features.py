#!/usr/bin/env python3
"""Build feature archives from explicit, leak-safe device-corpus splits."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import datasets
import numpy as np
from mmap_ninja.ragged import RaggedMmap

from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.clips import Clips
from microwakeword.audio.spectrograms import SpectrogramGeneration
from microwakeword.device_corpus import (
    SPLITS,
    TRUTHS,
    captures_for,
    validate_device_corpus,
)


def explicit_clips(root: Path, manifest: dict, truth: str) -> Clips:
    """Create a Clips source without re-randomizing the manifest's splits."""
    split_paths = {
        split: [str(path) for _, path in captures_for(root, manifest, truth, split)]
        for split in SPLITS
    }
    missing = sorted(split for split, paths in split_paths.items() if not paths)
    if missing:
        raise ValueError(f"{truth} requires captures in every split; missing={missing}")

    clips = Clips.__new__(Clips)
    clips.trim_zeros = False
    clips.trimmed_clip_duration_s = None
    clips.repeat_clip_min_duration_s = 0.0
    clips.remove_silence = False
    clips.split_clips = datasets.DatasetDict(
        {
            split: datasets.Dataset.from_dict({"audio": paths}).cast_column(
                "audio", datasets.Audio(sampling_rate=16000)
            )
            for split, paths in split_paths.items()
        }
    )
    clips.clips = datasets.concatenate_datasets(list(clips.split_clips.values()))
    return clips


def feature_split_directory(truth: str, split: str) -> str:
    if truth == "ambient_negative":
        return {
            "train": "training",
            "validation": "validation_ambient",
            "test": "testing_ambient",
        }[split]
    return {"train": "training", "validation": "validation", "test": "testing"}[split]


def build_truth(root: Path, manifest: dict, truth: str, output: Path) -> None:
    augmenter = Augmentation(
        augmentation_duration_s=2.0,
        augmentation_probabilities={
            "SevenBandParametricEQ": 0.10,
            "TanhDistortion": 0.05,
            "PitchShift": 0.05,
            "BandStopFilter": 0.05,
            "AddColorNoise": 0.10,
            "Gain": 1.0,
            "GainTransition": 0.10,
        },
        min_gain_db=-12,
        max_gain_db=0,
        min_jitter_s=0.05,
        max_jitter_s=0.15,
    )
    clips = explicit_clips(root, manifest, truth)
    for split in sorted(SPLITS):
        spectrograms = SpectrogramGeneration(
            clips=clips,
            augmenter=augmenter,
            slide_frames=1 if split == "test" else 5,
            step_ms=10,
        )
        output_split = feature_split_directory(truth, split)
        destination = output / truth / output_split
        destination.mkdir(parents=True, exist_ok=True)
        RaggedMmap.from_generator(
            out_dir=str(destination / "wakeword_mmap"),
            sample_generator=spectrograms.spectrogram_generator(
                split=split, repeat=2 if split == "train" else 1
            ),
            batch_size=100,
            verbose=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--truth", choices=("all", *sorted(TRUTHS)), default="all")
    args = parser.parse_args()

    manifest = validate_device_corpus(args.corpus)
    seed = int(manifest.get("random_seed", 231))
    random.seed(seed)
    np.random.seed(seed)
    truths = sorted(TRUTHS) if args.truth == "all" else [args.truth]
    for truth in truths:
        build_truth(args.corpus, manifest, truth, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
