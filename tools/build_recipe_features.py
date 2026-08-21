#!/usr/bin/env python3
"""Convert generated recipe audio into train/validation/test feature mmaps."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import yaml
from mmap_ninja.ragged import RaggedMmap

from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.clips import Clips
from microwakeword.audio.spectrograms import SpectrogramGeneration


def validate_generated_corpus(recipe_path: Path, generated: Path) -> None:
    manifest_path = generated / "generation-manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"missing generation manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    recipe_sha256 = hashlib.sha256(recipe_path.read_bytes()).hexdigest()
    if manifest.get("recipe_sha256") != recipe_sha256:
        raise ValueError(
            "generated corpus recipe hash does not match the requested recipe"
        )

    for class_name in ("positive", "hard_negative"):
        expected = {
            Path(item["output"]).resolve(): int(item["samples"])
            for item in manifest.get("plan", [])
            if item.get("class") == class_name
        }
        class_root = generated / class_name
        actual = {
            path.resolve()
            for path in class_root.iterdir()
            if path.is_dir()
        }
        if actual != set(expected):
            missing = sorted(str(path) for path in set(expected) - actual)
            extra = sorted(str(path) for path in actual - set(expected))
            raise ValueError(
                f"{class_name} corpus does not match manifest; "
                f"missing={missing}, extra={extra}"
            )
        for phrase_dir, expected_count in expected.items():
            actual_count = len(list(phrase_dir.glob("*.wav")))
            if actual_count != expected_count:
                raise ValueError(
                    f"{phrase_dir} has {actual_count} WAVs; "
                    f"manifest requires {expected_count}"
                )


def generate_class_features(
    source: Path,
    destination: Path,
    augmenter: Augmentation,
    split_seed: int,
) -> None:
    clips = Clips(
        input_directory=str(source),
        file_pattern="**/*.wav",
        max_clip_duration_s=None,
        remove_silence=False,
        random_split_seed=split_seed,
        split_count=0.1,
    )
    for split in ("training", "validation", "testing"):
        split_name = {"training": "train", "validation": "validation", "testing": "test"}[split]
        slide_frames = 1 if split == "testing" else 10
        repetition = 2 if split == "training" else 1
        spectrograms = SpectrogramGeneration(
            clips=clips,
            augmenter=augmenter,
            slide_frames=slide_frames,
            step_ms=10,
        )
        out_dir = destination / split
        out_dir.mkdir(parents=True, exist_ok=True)
        RaggedMmap.from_generator(
            out_dir=str(out_dir / "wakeword_mmap"),
            sample_generator=spectrograms.spectrogram_generator(
                split=split_name, repeat=repetition
            ),
            batch_size=100,
            verbose=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--background", type=Path, action="append", default=[])
    parser.add_argument("--impulses", type=Path, action="append", default=[])
    args = parser.parse_args()

    validate_generated_corpus(args.recipe, args.generated)
    recipe = yaml.safe_load(args.recipe.read_text())
    seed = int(recipe["random_seed"])
    random.seed(seed)
    np.random.seed(seed)
    augmenter = Augmentation(
        augmentation_duration_s=recipe["clip_duration_ms"] / 1000,
        augmentation_probabilities={
            "SevenBandParametricEQ": 0.15,
            "TanhDistortion": 0.1,
            "PitchShift": 0.1,
            "BandStopFilter": 0.1,
            "AddColorNoise": 0.35,
            "AddBackgroundNoise": 0.8 if args.background else 0.0,
            "Gain": 1.0,
            "GainTransition": 0.15,
            "RIR": 0.6 if args.impulses else 0.0,
        },
        impulse_paths=[str(path) for path in args.impulses],
        background_paths=[str(path) for path in args.background],
        background_min_snr_db=-8,
        background_max_snr_db=12,
        min_gain_db=-35,
        max_gain_db=0,
        min_jitter_s=0.15,
        max_jitter_s=0.30,
    )
    generate_class_features(
        args.generated / "positive", args.output / "positive", augmenter, seed
    )
    generate_class_features(
        args.generated / "hard_negative",
        args.output / "hard_negative",
        augmenter,
        seed + 1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
