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


def _plan_provider(plan_item: dict) -> str:
    """Return the synthesizer provider for filtering feature sources."""
    return plan_item.get("provider", "piper")


def validate_metadata_texts(plan_item: dict, metadata: list[dict], path: Path) -> None:
    """Bind synthesized records to the exact text contract in their plan item."""
    text = plan_item.get("text")
    expected = (
        {text}
        if isinstance(text, str) and text
        else set(plan_item.get("normalized_texts", []))
    )
    if not expected:
        raise ValueError(f"{path} plan item has no expected text contract")
    actual = {record.get("text") for record in metadata}
    unexpected = actual - expected
    missing = expected - actual
    if unexpected or missing:
        raise ValueError(
            f"{path} text contract mismatch; "
            f"unexpected={sorted(str(value) for value in unexpected)}, "
            f"missing={sorted(missing)}"
        )


def phrase_slug(text: str) -> str:
    readable = "_".join(
        "".join(
            character.lower() if character.isalnum() else " " for character in text
        ).split()
    )
    return f"{readable}-{hashlib.sha256(text.encode()).hexdigest()[:8]}"


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
            validate_metadata_texts(plan_item, metadata, metadata_path)
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
    manifest: dict,
    class_name: str,
    texts: list[str],
    split: str | None = None,
    providers: set[str] | None = None,
    age_groups: set[str] | None = None,
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
            and (not providers or _plan_provider(item) in providers)
            and (not age_groups or item.get("age_group") in age_groups)
        ):
            by_text[item["text"]].append(Path(item["output"]))
    return [path for text in texts for path in by_text[text]]


@contextmanager
def staged_clip_source(
    source_dirs: list[Path],
    rejected: set[Path] | None = None,
    max_clips_per_speaker_phrase: int | None = None,
    selection_seed: int = 0,
) -> Iterator[Path]:
    """Expose selected phrase directories as one flat, temporary clip corpus."""
    with tempfile.TemporaryDirectory(prefix="mww-selected-clips-") as temporary:
        root = Path(temporary)
        rejected = rejected or set()
        for source in source_dirs:
            readable_prefix = (
                source.parent.name
                if source.name in {"train", "validation", "test"}
                else source.name
            )
            source_identity = hashlib.sha256(
                source.resolve().as_posix().encode()
            ).hexdigest()[:12]
            prefix = f"{readable_prefix}-{source_identity}"
            clips = sorted(source.glob("*.wav"))
            if max_clips_per_speaker_phrase:
                metadata_path = source / "synthesis-metadata.jsonl"
                metadata = {
                    record["file"]: record
                    for record in (
                        json.loads(line)
                        for line in metadata_path.read_text().splitlines()
                        if line.strip()
                    )
                }
                clips.sort(
                    key=lambda clip: hashlib.sha256(
                        f"{selection_seed}:{source}:{clip.name}".encode()
                    ).digest()
                )
                counts: dict[str, int] = {}
                selected = []
                for clip in clips:
                    if clip.resolve() in rejected:
                        continue
                    record = metadata[clip.name]
                    if speaker_id := record.get("speaker_id"):
                        speakers = [f"{record.get('provider')}:{speaker_id}"]
                    else:
                        speakers = [
                            f"piper:{record[field]}"
                            for field in ("speaker_1", "speaker_2")
                        ]
                    if any(
                        counts.get(speaker, 0) >= max_clips_per_speaker_phrase
                        for speaker in speakers
                    ):
                        continue
                    selected.append(clip)
                    for speaker in speakers:
                        counts[speaker] = counts.get(speaker, 0) + 1
                clips = selected
            for clip in clips:
                if clip.resolve() in rejected:
                    continue
                destination = root / f"{prefix}--{clip.name}"
                os.symlink(clip, destination)
        yield root


def class_directories(
    manifest: dict,
    class_name: str,
    split: str | None = None,
    providers: set[str] | None = None,
    age_groups: set[str] | None = None,
) -> list[Path]:
    return [
        Path(item["output"])
        for item in manifest.get("plan", [])
        if item.get("class") == class_name
        and (split is None or item.get("split") == split)
        and (not providers or _plan_provider(item) in providers)
        and (not age_groups or item.get("age_group") in age_groups)
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
    max_clips_per_speaker_phrase: int | None = None,
    selection_seed: int = 0,
) -> None:
    with staged_clip_source(
        sources,
        rejected,
        max_clips_per_speaker_phrase,
        selection_seed,
    ) as source:
        clips = Clips(
            input_directory=str(source),
            file_pattern="**/*.wav",
            max_clip_duration_s=None,
            remove_silence=False,
            random_split_seed=None,
        )
        slide_frames = 1 if split == "testing" else 10
        repetition = 2 if split == "training" and augmenter is not None else 1
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
        default=None,
        help="Minimum speech-to-background ratio; negative allows louder noise",
    )
    parser.add_argument(
        "--background-max-snr-db",
        type=int,
        default=None,
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
    parser.add_argument(
        "--provider",
        action="append",
        default=[],
        help="Build features from only this labeled provider; repeatable",
    )
    parser.add_argument(
        "--age-group",
        action="append",
        default=[],
        help="Build features from only this labeled age cohort; repeatable",
    )
    parser.add_argument(
        "--max-clips-per-speaker-phrase",
        type=int,
        help="Deterministically cap each synthetic speaker within each phrase",
    )
    parser.add_argument(
        "--separate-by-phrase",
        action="store_true",
        help="Write each phrase to an independent feature source for balanced sampling",
    )
    parser.add_argument(
        "--augmentation-profile",
        choices=("clean", "normal_room", "challenging"),
        default="normal_room",
        help="Build a separately labeled acoustic-condition bank",
    )
    args = parser.parse_args()

    if args.positive_text and args.class_name == "hard_negative":
        parser.error("--positive-text requires positive or both class generation")
    if args.hard_negative_text and args.class_name == "positive":
        parser.error(
            "--hard-negative-text requires hard_negative or both class generation"
        )
    if (
        args.max_clips_per_speaker_phrase is not None
        and args.max_clips_per_speaker_phrase < 1
    ):
        parser.error("--max-clips-per-speaker-phrase must be positive")
    manifest = validate_generated_corpus(args.recipe, args.generated)
    default_snr = {
        "clean": (None, None),
        "normal_room": (3, 20),
        "challenging": (-6, 6),
    }[args.augmentation_profile]
    minimum_snr = (
        args.background_min_snr_db
        if args.background_min_snr_db is not None
        else default_snr[0]
    )
    maximum_snr = (
        args.background_max_snr_db
        if args.background_max_snr_db is not None
        else default_snr[1]
    )
    if args.augmentation_profile != "clean" and maximum_snr < minimum_snr:
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
    probabilities = {
        "SevenBandParametricEQ": 0.15,
        "TanhDistortion": 0.1,
        "PitchShift": 0.1,
        "BandStopFilter": 0.1,
        "AddColorNoise": 0.35,
        "AddBackgroundNoise": 0.8 if backgrounds else 0.0,
        "Gain": 1.0,
        "GainTransition": 0.15,
        "RIR": 0.6 if args.impulses else 0.0,
    }
    augmenter = None
    if args.augmentation_profile != "clean":
        augmenter = Augmentation(
            augmentation_duration_s=recipe["clip_duration_ms"] / 1000,
            augmentation_probabilities=probabilities,
            impulse_paths=[str(path) for path in args.impulses],
            background_paths=[str(path) for path in backgrounds],
            background_min_snr_db=minimum_snr,
            background_max_snr_db=maximum_snr,
            min_gain_db=-35,
            max_gain_db=0,
            min_jitter_s=0.15,
            max_jitter_s=0.30,
        )
    split_names = {"training": "train", "validation": "validation", "testing": "test"}
    providers = set(args.provider)
    age_groups = set(args.age_group)
    selections = {
        "positive": args.positive_text,
        "hard_negative": args.hard_negative_text,
    }
    requested_classes = (
        ("positive", "hard_negative")
        if args.class_name == "both"
        else (args.class_name,)
    )
    built_sources = []
    for class_name in requested_classes:
        for feature_split in feature_splits:
            manifest_split = split_names[feature_split]
            texts = selections[class_name]
            if args.separate_by_phrase:
                phrase_texts = texts or [
                    phrase["text"] for phrase in recipe[f"{class_name}_phrases"]
                ]
                source_groups = [
                    (
                        text,
                        selected_phrase_directories(
                            manifest,
                            class_name,
                            [text],
                            manifest_split,
                            providers,
                            age_groups,
                        ),
                    )
                    for text in phrase_texts
                ]
            else:
                sources = (
                    selected_phrase_directories(
                        manifest,
                        class_name,
                        texts,
                        manifest_split,
                        providers,
                        age_groups,
                    )
                    if texts
                    else class_directories(
                        manifest,
                        class_name,
                        manifest_split,
                        providers,
                        age_groups,
                    )
                )
                source_groups = [(None, sources)]
            for text, sources in source_groups:
                destination = args.output / class_name
                if text is not None:
                    destination /= phrase_slug(text)
                generate_class_features(
                    sources,
                    destination,
                    augmentation_for_split(augmenter, feature_split),
                    feature_split,
                    rejected,
                    args.max_clips_per_speaker_phrase,
                    seed,
                )
                built_sources.append(
                    {
                        "class": class_name,
                        "text": text,
                        "feature_split": feature_split,
                        "features_dir": str(destination),
                    }
                )

    build_manifest = {
        "schema_version": 1,
        "recipe_sha256": hashlib.sha256(args.recipe.read_bytes()).hexdigest(),
        "generation_manifest_sha256": hashlib.sha256(
            (args.generated / "generation-manifest.json").read_bytes()
        ).hexdigest(),
        "quality_mask": str(args.quality_mask) if args.quality_mask else None,
        "selection": {
            "providers": sorted(providers),
            "age_groups": sorted(age_groups),
            "max_clips_per_speaker_phrase": args.max_clips_per_speaker_phrase,
            "seed": seed,
            "separate_by_phrase": args.separate_by_phrase,
        },
        "feature_sources": built_sources,
        "training_augmentation": {
            "profile": args.augmentation_profile,
            "background_sources": {
                "unclassified": [str(path) for path in args.background],
                "indoor": [str(path) for path in args.background_indoor],
                "outdoor": [str(path) for path in args.background_outdoor],
            },
            "impulse_paths": [str(path) for path in args.impulses],
            "background_min_snr_db": minimum_snr,
            "background_max_snr_db": maximum_snr,
            "probabilities": (
                {
                    "parametric_eq": 0.15,
                    "tanh_distortion": 0.1,
                    "pitch_shift": 0.1,
                    "band_stop_filter": 0.1,
                    "color_noise": 0.35,
                    "background_noise": 0.8 if backgrounds else 0.0,
                    "gain": 1.0,
                    "gain_transition": 0.15,
                    "room_impulse_response": 0.6 if args.impulses else 0.0,
                }
                if augmenter is not None
                else {}
            ),
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
