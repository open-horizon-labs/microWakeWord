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

PRODUCTION_SOURCES = {
    "positive": {
        "train": {"human", "synthetic_playback"},
        "validation": {"human"},
        "test": {"human"},
    },
    "hard_negative": {
        "train": {"human", "synthetic_playback"},
        "validation": {"human"},
        "test": {"human"},
    },
    "ambient_negative": {
        "train": {"ambient"},
        "validation": {"ambient"},
        "test": {"ambient"},
    },
}


def explicit_clips(
    root: Path,
    manifest: dict,
    truth: str,
    include_sources: set[str] | None = None,
    splits: set[str] | None = None,
) -> Clips:
    """Create a Clips source without re-randomizing the manifest's splits."""
    selected_splits = splits or SPLITS
    split_paths = {
        split: [
            str(path)
            for item, path in captures_for(root, manifest, truth, split)
            if item["source"]
            in (include_sources or PRODUCTION_SOURCES[truth][split])
        ]
        for split in selected_splits
    }
    missing = sorted(split for split, paths in split_paths.items() if not paths)
    if missing:
        raise ValueError(
            f"{truth} requires eligible captures in every split; "
            f"missing={missing}"
        )

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


def build_truth(
    root: Path,
    manifest: dict,
    truth: str,
    output: Path,
    splits: set[str] | None = None,
) -> None:
    augmenter = Augmentation(
        augmentation_duration_s=None,
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
    selected_splits = splits or SPLITS
    clips = explicit_clips(root, manifest, truth, splits=selected_splits)
    for split in sorted(selected_splits):
        is_training = split == "train"
        spectrograms = SpectrogramGeneration(
            clips=clips,
            augmenter=augmenter if is_training else None,
            slide_frames=5 if is_training else None,
            step_ms=10,
        )
        output_split = feature_split_directory(truth, split)
        destination = output / truth / output_split
        destination.mkdir(parents=True, exist_ok=True)
        RaggedMmap.from_generator(
            out_dir=str(destination / "wakeword_mmap"),
            sample_generator=spectrograms.spectrogram_generator(
                split=split, repeat=2 if is_training else 1
            ),
            batch_size=100,
            verbose=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--truth", choices=("all", *sorted(TRUTHS)), default="all")
    parser.add_argument(
        "--split",
        action="append",
        choices=sorted(SPLITS),
        help="Build only the selected manifest split; repeat for multiple splits",
    )
    args = parser.parse_args()

    manifest = validate_device_corpus(args.corpus)
    seed = int(manifest.get("random_seed", 231))
    random.seed(seed)
    np.random.seed(seed)
    truths = sorted(TRUTHS) if args.truth == "all" else [args.truth]
    splits = set(args.split) if args.split else None
    for truth in truths:
        build_truth(args.corpus, manifest, truth, args.output, splits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
