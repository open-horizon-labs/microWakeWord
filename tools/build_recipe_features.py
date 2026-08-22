#!/usr/bin/env python3
"""Convert generated recipe audio into train/validation/test feature mmaps."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import random
from pathlib import Path
import tempfile
from collections.abc import Iterator

import numpy as np
import yaml
from mmap_ninja.ragged import RaggedMmap

from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.clips import Clips
from microwakeword.audio.spectrograms import SpectrogramGeneration
from microwakeword.synthetic_quality import load_quality_mask


def validate_generated_corpus(recipe_path: Path, generated: Path) -> dict:
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
        actual = {path.resolve() for path in class_root.iterdir() if path.is_dir()}
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
    return manifest


def selected_phrase_directories(
    manifest: dict, class_name: str, texts: list[str]
) -> list[Path]:
    by_text = {
        item["text"]: Path(item["output"])
        for item in manifest.get("plan", [])
        if item.get("class") == class_name
    }
    unknown = sorted(set(texts) - set(by_text))
    if unknown:
        raise ValueError(f"unknown {class_name} phrase(s): {unknown}")
    return [by_text[text] for text in texts]


@contextmanager
def staged_clip_source(
    source_dirs: list[Path], rejected: set[Path] | None = None
) -> Iterator[Path]:
    """Expose selected phrase directories as one flat, temporary clip corpus."""
    with tempfile.TemporaryDirectory(prefix="mww-selected-clips-") as temporary:
        root = Path(temporary)
        rejected = rejected or set()
        for source in source_dirs:
            for clip in source.glob("*.wav"):
                if clip.resolve() in rejected:
                    continue
                destination = root / f"{source.name}--{clip.name}"
                os.symlink(clip, destination)
        yield root


def class_directories(manifest: dict, class_name: str) -> list[Path]:
    return [
        Path(item["output"])
        for item in manifest.get("plan", [])
        if item.get("class") == class_name
    ]


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
        split_name = {
            "training": "train",
            "validation": "validation",
            "testing": "test",
        }[split]
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
    parser.add_argument(
        "--quality-mask",
        type=Path,
        help="Exclude generated clips rejected by this provenance-bound mask",
    )
    parser.add_argument(
        "--class-name",
        choices=("both", "positive", "hard_negative"),
        default="both",
        help="Rebuild one class when the other class corpus is unchanged",
    )
    parser.add_argument(
        "--positive-text",
        action="append",
        default=[],
        help="Build positive features from only this exact recipe phrase; repeatable",
    )
    parser.add_argument(
        "--hard-negative-text",
        action="append",
        default=[],
        help="Build hard-negative features from only this exact phrase; repeatable",
    )
    args = parser.parse_args()

    if args.positive_text and args.class_name == "hard_negative":
        parser.error("--positive-text requires positive or both class generation")
    if args.hard_negative_text and args.class_name == "positive":
        parser.error(
            "--hard-negative-text requires hard_negative or both class generation"
        )
    manifest = validate_generated_corpus(args.recipe, args.generated)
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
    if args.class_name in ("both", "positive"):
        if args.positive_text:
            selected = selected_phrase_directories(
                manifest, "positive", args.positive_text
            )
            with staged_clip_source(selected, rejected) as source:
                generate_class_features(
                    source, args.output / "positive", augmenter, seed
                )
        elif rejected:
            with staged_clip_source(
                class_directories(manifest, "positive"), rejected
            ) as source:
                generate_class_features(
                    source, args.output / "positive", augmenter, seed
                )
        else:
            generate_class_features(
                args.generated / "positive",
                args.output / "positive",
                augmenter,
                seed,
            )
    if args.class_name in ("both", "hard_negative"):
        if args.hard_negative_text:
            selected = selected_phrase_directories(
                manifest, "hard_negative", args.hard_negative_text
            )
            with staged_clip_source(selected, rejected) as source:
                generate_class_features(
                    source,
                    args.output / "hard_negative",
                    augmenter,
                    seed + 1,
                )
        elif rejected:
            with staged_clip_source(
                class_directories(manifest, "hard_negative"), rejected
            ) as source:
                generate_class_features(
                    source,
                    args.output / "hard_negative",
                    augmenter,
                    seed + 1,
                )
        else:
            generate_class_features(
                args.generated / "hard_negative",
                args.output / "hard_negative",
                augmenter,
                seed + 1,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
