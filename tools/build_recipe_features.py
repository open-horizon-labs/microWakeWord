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


def _plan_speakers(plan_item: dict) -> set[str]:
    """Return stable speaker identities for either Piper or labeled TTS."""
    if speaker_id := plan_item.get("speaker_id"):
        provider = plan_item.get("provider")
        if not provider:
            raise ValueError("labeled TTS plan item requires provider")
        return {f"{provider}:{speaker_id}"}
    return {
        f"piper:{speaker}"
        for speaker in range(
            int(plan_item["speaker_start"]), int(plan_item["speaker_end"])
        )
    }


def _validate_labeled_voice_coverage(recipe: dict, manifest: dict) -> None:
    requirements = recipe.get("generation", {}).get("labeled_voice_requirements")
    if not requirements:
        return
    required_ages = set(requirements.get("age_groups", []))
    minimums = requirements.get("minimum_voices_per_split", {})
    voices: dict[tuple[str, str], set[str]] = {}
    for item in manifest.get("plan", []):
        if not item.get("speaker_id"):
            continue
        key = (item["split"], item["age_group"])
        voices.setdefault(key, set()).update(_plan_speakers(item))
    missing = []
    for split in ("train", "validation", "test"):
        minimum = int(minimums.get(split, 0))
        for age_group in sorted(required_ages):
            found = len(voices.get((split, age_group), set()))
            if found < minimum:
                missing.append(f"{split}/{age_group}: {found} of {minimum}")
    if missing:
        raise ValueError(
            "generated corpus lacks required labeled voice cohorts: "
            + ", ".join(missing)
        )


def validate_generated_corpus(recipe_path: Path, generated: Path) -> dict:
    manifest_path = generated / "generation-manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"missing generation manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 2:
        raise ValueError("generated corpus requires schema_version 2 speaker cohorts")
    recipe_sha256 = hashlib.sha256(recipe_path.read_bytes()).hexdigest()
    if manifest.get("recipe_sha256") != recipe_sha256:
        raise ValueError(
            "generated corpus recipe hash does not match the requested recipe"
        )

    recipe = yaml.safe_load(recipe_path.read_text())
    for class_name in ("positive", "hard_negative"):
        expected = {
            Path(item["output"]).resolve(): int(item["samples"])
            for item in manifest.get("plan", [])
            if item.get("class") == class_name
        }
        class_root = generated / class_name
        actual = {path.parent.resolve() for path in class_root.rglob("*.wav")}
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
            metadata_path = phrase_dir / "synthesis-metadata.jsonl"
            if not metadata_path.exists():
                raise ValueError(f"missing synthesis provenance: {metadata_path}")
            metadata = [
                json.loads(line)
                for line in metadata_path.read_text().splitlines()
                if line.strip()
            ]
            if len(metadata) != expected_count:
                raise ValueError(
                    f"{metadata_path} has {len(metadata)} records; "
                    f"manifest requires {expected_count}"
                )
            plan_item = next(
                item
                for item in manifest["plan"]
                if Path(item["output"]).resolve() == phrase_dir
            )
            files = set()
            for record in metadata:
                files.add(record.get("file"))
                if plan_item.get("speaker_id"):
                    if (
                        record.get("speaker_id") != plan_item["speaker_id"]
                        or record.get("provider") != plan_item["provider"]
                        or record.get("age_group") != plan_item["age_group"]
                    ):
                        raise ValueError(
                            f"{metadata_path} contradicts its labeled voice cohort"
                        )
                else:
                    speaker_range = range(
                        int(plan_item["speaker_start"]),
                        int(plan_item["speaker_end"]),
                    )
                    if (
                        record.get("speaker_1") not in speaker_range
                        or record.get("speaker_2") not in speaker_range
                    ):
                        raise ValueError(
                            f"{metadata_path} contains a speaker outside its cohort"
                        )
            expected_files = {path.name for path in phrase_dir.glob("*.wav")}
            if files != expected_files:
                raise ValueError(f"{metadata_path} does not describe its WAV files")

    speakers_by_split: dict[str, set[str]] = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }
    for item in manifest.get("plan", []):
        split = item.get("split")
        if split not in speakers_by_split:
            raise ValueError(f"generated plan has invalid split: {split}")
        speakers_by_split[split].update(_plan_speakers(item))
    for first, second in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        if speakers_by_split[first] & speakers_by_split[second]:
            raise ValueError(f"synthetic speakers cross {first} and {second} splits")
    _validate_labeled_voice_coverage(recipe, manifest)
    return manifest


def selected_phrase_directories(
    manifest: dict, class_name: str, texts: list[str], split: str | None = None
) -> list[Path]:
    by_text: dict[str, list[Path]] = {text: [] for text in texts}
    available = {
        item["text"]
        for item in manifest.get("plan", [])
        if item.get("class") == class_name
    }
    unknown = sorted(set(texts) - available)
    if unknown:
        raise ValueError(f"unknown {class_name} phrase(s): {unknown}")
    for item in manifest.get("plan", []):
        if (
            item.get("class") == class_name
            and item.get("text") in by_text
            and (split is None or item.get("split") == split)
        ):
            by_text[item["text"]].append(Path(item["output"]))
    return [path for text in texts for path in by_text[text]]


@contextmanager
def staged_clip_source(
    source_dirs: list[Path], rejected: set[Path] | None = None
) -> Iterator[Path]:
    """Expose selected phrase directories as one flat, temporary clip corpus."""
    with tempfile.TemporaryDirectory(prefix="mww-selected-clips-") as temporary:
        root = Path(temporary)
        rejected = rejected or set()
        for source in source_dirs:
            prefix = (
                source.parent.name
                if source.name in {"train", "validation", "test"}
                else source.name
            )
            for clip in source.glob("*.wav"):
                if clip.resolve() in rejected:
                    continue
                destination = root / f"{prefix}--{clip.name}"
                os.symlink(clip, destination)
        yield root


def class_directories(
    manifest: dict, class_name: str, split: str | None = None
) -> list[Path]:
    return [
        Path(item["output"])
        for item in manifest.get("plan", [])
        if item.get("class") == class_name
        and (split is None or item.get("split") == split)
    ]


def augmentation_for_split(augmenter: Augmentation, feature_split: str):
    """Apply acoustic stress only to training evidence."""
    return augmenter if feature_split == "training" else None


def generate_class_features(
    sources: list[Path],
    destination: Path,
    augmenter: Augmentation,
    split: str,
    rejected: set[Path] | None = None,
) -> None:
    with staged_clip_source(sources, rejected) as source:
        clips = Clips(
            input_directory=str(source),
            file_pattern="**/*.wav",
            max_clip_duration_s=None,
            remove_silence=False,
            random_split_seed=None,
        )
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
                split=None, repeat=repetition
            ),
            batch_size=100,
            verbose=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--background",
        type=Path,
        action="append",
        default=[],
        help="Unclassified background directory (repeatable)",
    )
    parser.add_argument(
        "--background-indoor",
        type=Path,
        action="append",
        default=[],
        help="Indoor background directory (repeatable)",
    )
    parser.add_argument(
        "--background-outdoor",
        type=Path,
        action="append",
        default=[],
        help="Outdoor background directory (repeatable)",
    )
    parser.add_argument("--impulses", type=Path, action="append", default=[])
    parser.add_argument(
        "--background-min-snr-db",
        type=int,
        default=3,
        help="Minimum speech-to-background ratio; positive keeps speech louder",
    )
    parser.add_argument(
        "--background-max-snr-db",
        type=int,
        default=20,
        help="Maximum speech-to-background ratio",
    )
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
        "--feature-split",
        choices=("training", "validation", "testing"),
        action="append",
        help=(
            "Build only this feature split; repeatable. By default all three "
            "splits are built."
        ),
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
    if args.background_min_snr_db < 0:
        parser.error("--background-min-snr-db must keep background below speech")
    if args.background_max_snr_db < args.background_min_snr_db:
        parser.error("background maximum SNR must be at least its minimum")
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
    feature_splits = tuple(args.feature_split or ("training", "validation", "testing"))
    seed = int(recipe["random_seed"])
    random.seed(seed)
    np.random.seed(seed)
    backgrounds = [
        *args.background,
        *args.background_indoor,
        *args.background_outdoor,
    ]
    augmenter = Augmentation(
        augmentation_duration_s=recipe["clip_duration_ms"] / 1000,
        augmentation_probabilities={
            "SevenBandParametricEQ": 0.15,
            "TanhDistortion": 0.1,
            "PitchShift": 0.1,
            "BandStopFilter": 0.1,
            "AddColorNoise": 0.35,
            "AddBackgroundNoise": 0.8 if backgrounds else 0.0,
            "Gain": 1.0,
            "GainTransition": 0.15,
            "RIR": 0.6 if args.impulses else 0.0,
        },
        impulse_paths=[str(path) for path in args.impulses],
        background_paths=[str(path) for path in backgrounds],
        background_min_snr_db=args.background_min_snr_db,
        background_max_snr_db=args.background_max_snr_db,
        min_gain_db=-35,
        max_gain_db=0,
        min_jitter_s=0.15,
        max_jitter_s=0.30,
    )
    split_names = {"training": "train", "validation": "validation", "testing": "test"}
    selections = {
        "positive": args.positive_text,
        "hard_negative": args.hard_negative_text,
    }
    requested_classes = (
        ("positive", "hard_negative")
        if args.class_name == "both"
        else (args.class_name,)
    )
    for class_name in requested_classes:
        for feature_split in feature_splits:
            manifest_split = split_names[feature_split]
            texts = selections[class_name]
            sources = (
                selected_phrase_directories(manifest, class_name, texts, manifest_split)
                if texts
                else class_directories(manifest, class_name, manifest_split)
            )
            generate_class_features(
                sources,
                args.output / class_name,
                augmentation_for_split(augmenter, feature_split),
                feature_split,
                rejected,
            )

    build_manifest = {
        "schema_version": 1,
        "recipe_sha256": hashlib.sha256(args.recipe.read_bytes()).hexdigest(),
        "generation_manifest_sha256": hashlib.sha256(
            (args.generated / "generation-manifest.json").read_bytes()
        ).hexdigest(),
        "quality_mask": str(args.quality_mask) if args.quality_mask else None,
        "training_augmentation": {
            "background_sources": {
                "unclassified": [str(path) for path in args.background],
                "indoor": [str(path) for path in args.background_indoor],
                "outdoor": [str(path) for path in args.background_outdoor],
            },
            "impulse_paths": [str(path) for path in args.impulses],
            "background_min_snr_db": args.background_min_snr_db,
            "background_max_snr_db": args.background_max_snr_db,
            "probabilities": {
                "parametric_eq": 0.15,
                "tanh_distortion": 0.1,
                "pitch_shift": 0.1,
                "band_stop_filter": 0.1,
                "color_noise": 0.35,
                "background_noise": 0.8 if backgrounds else 0.0,
                "gain": 1.0,
                "gain_transition": 0.15,
                "room_impulse_response": 0.6 if args.impulses else 0.0,
            },
            "held_out_audio": "clean",
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "feature-build-manifest.json").write_text(
        json.dumps(build_manifest, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
