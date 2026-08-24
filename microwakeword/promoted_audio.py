"""Validate and build features from an explicit, reviewed audio manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import wave
from pathlib import Path
from tempfile import TemporaryDirectory

SCHEMA_VERSION = 1
SPLITS = {"train", "validation", "test"}
TRUTHS = {"positive", "hard_negative"}
REQUIRED_ENTRY_FIELDS = {
    "id",
    "wav_path",
    "sha256",
    "truth",
    "split",
    "text",
    "provenance",
    "human_reviewed",
    "training_eligible",
}
QUARANTINED_COMPONENTS = {"observations", "false-wakes", "evidence"}
OUTPUT_MANIFEST_NAME = "promoted-audio-feature-build.json"
RaggedMmap = SpectrogramGeneration = None


def _quarantined_path_components(path: Path) -> list[str]:
    """Return quarantined components from both the named and resolved path."""
    components = {
        component.casefold()
        for candidate in (path, path.resolve(strict=False))
        for component in candidate.parts
    }
    return sorted(components & QUARANTINED_COMPONENTS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _string(entry: dict, key: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"entry {entry.get('id', '<unknown>')} requires non-empty {key}"
        )
    return value


def _validate_wav(path: Path, entry_id: str) -> None:
    try:
        with wave.open(str(path), "rb") as source:
            if (
                source.getnchannels() != 1
                or source.getframerate() != 16000
                or source.getsampwidth() != 2
                or source.getcomptype() != "NONE"
            ):
                raise ValueError(
                    f"entry {entry_id} WAV must be mono 16kHz signed 16-bit PCM"
                )
    except (wave.Error, EOFError) as error:
        raise ValueError(f"entry {entry_id} is not a readable PCM WAV") from error


def validate_manifest(manifest_path: Path) -> dict:
    """Load and validate a schema-v1 promotion manifest."""
    forbidden_manifest_components = _quarantined_path_components(manifest_path)
    if forbidden_manifest_components:
        raise ValueError(
            "promotion manifest uses quarantined path component: "
            f"{sorted(forbidden_manifest_components)}"
        )
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid promotion manifest: {manifest_path}") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("promotion manifest schema_version must be 1")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("promotion manifest requires non-empty entries")

    seen: set[str] = set()
    validated = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each promotion entry must be an object")
        missing = sorted(REQUIRED_ENTRY_FIELDS - set(entry))
        if missing:
            raise ValueError(f"promotion entry is missing required fields: {missing}")
        entry_id = _string(entry, "id")
        if entry_id in seen:
            raise ValueError(f"duplicate promotion entry id: {entry_id}")
        seen.add(entry_id)
        path_text = _string(entry, "wav_path")
        path = Path(path_text)
        if not path.is_absolute():
            raise ValueError(f"entry {entry_id} wav_path must be absolute")
        forbidden = _quarantined_path_components(path)
        if forbidden:
            raise ValueError(
                f"entry {entry_id} uses quarantined path component: {sorted(forbidden)}"
            )
        if not path.is_file():
            raise ValueError(f"entry {entry_id} WAV does not exist: {path}")
        expected_hash = _string(entry, "sha256").lower()
        if len(expected_hash) != 64 or any(
            c not in "0123456789abcdef" for c in expected_hash
        ):
            raise ValueError(f"entry {entry_id} sha256 is malformed")
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"entry {entry_id} sha256 mismatch")
        if entry.get("truth") not in TRUTHS or entry.get("split") not in SPLITS:
            raise ValueError(f"entry {entry_id} has invalid truth or split")
        _string(entry, "text")
        _string(entry, "provenance")
        if entry.get("truth") == "positive":
            span = entry.get("phrase_span")
            if (
                not isinstance(span, dict)
                or not isinstance(span.get("start_ms"), (int, float))
                or not isinstance(span.get("end_ms"), (int, float))
                or not 0 <= span["start_ms"] < span["end_ms"]
            ):
                raise ValueError(
                    f"entry {entry_id} positive requires a valid phrase_span"
                )
        if entry.get("human_reviewed") is not True:
            raise ValueError(f"entry {entry_id} must be human_reviewed")
        if entry.get("training_eligible") is not True:
            raise ValueError(f"entry {entry_id} must be training_eligible")
        _validate_wav(path, entry_id)
        validated.append(dict(entry, wav_path=str(path)))
    return dict(manifest, entries=validated)


def select_entries(manifest: dict, positive_text: str | None = None) -> list[dict]:
    """Apply the exact canonical positive-text filter after validation."""
    if positive_text is None:
        return list(manifest["entries"])
    return [
        entry
        for entry in manifest["entries"]
        if entry["truth"] != "positive" or entry["text"] == positive_text
    ]


def _clips(entries: list[dict]):
    import datasets
    from microwakeword.audio.clips import Clips

    clips = Clips.__new__(Clips)
    clips.trim_zeros = False
    clips.trimmed_clip_duration_s = None
    clips.repeat_clip_min_duration_s = 0.0
    clips.remove_silence = False
    paths = [entry["wav_path"] for entry in entries]
    clips.split_clips = datasets.DatasetDict(
        {
            "selected": datasets.Dataset.from_dict({"audio": paths}).cast_column(
                "audio", datasets.Audio(sampling_rate=16000)
            )
        }
    )
    clips.clips = clips.split_clips["selected"]
    return clips


def phrase_aligned_pcm(
    pcm, start_ms: float, end_ms: float, target_samples: int = 32000
):
    """Return a fixed window that is guaranteed to contain the whole phrase."""
    import numpy as np

    start = round(float(start_ms) * 16)
    end = round(float(end_ms) * 16)
    if not 0 <= start < end <= pcm.size:
        raise ValueError("phrase_span falls outside promoted audio")
    if end - start > target_samples:
        raise ValueError("promoted phrase is longer than the model window")
    if pcm.size >= target_samples:
        lower = max(0, end - target_samples)
        upper = min(start, pcm.size - target_samples)
        if lower > upper:
            raise ValueError("no model window can contain the promoted phrase")
        centered = round((start + end - target_samples) / 2)
        window_start = min(max(centered, lower), upper)
        return np.asarray(pcm[window_start : window_start + target_samples])

    phrase_center = (start + end) // 2
    left_pad = target_samples // 2 - phrase_center
    left_pad = min(max(left_pad, 0), target_samples - pcm.size)
    right_pad = target_samples - pcm.size - left_pad
    return np.pad(pcm, (left_pad, right_pad))


def phrase_aligned_entries(entries: list[dict], output: Path) -> list[dict]:
    """Stage canonical positives as exact two-second phrase-containing WAVs."""
    import numpy as np

    output.mkdir(parents=True, exist_ok=True)
    aligned = []
    for entry in entries:
        if entry["truth"] != "positive":
            aligned.append(entry)
            continue
        source = Path(entry["wav_path"])
        with wave.open(str(source), "rb") as wav:
            pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        span = entry["phrase_span"]
        window = phrase_aligned_pcm(pcm, span["start_ms"], span["end_ms"])
        destination = output / f'{entry["id"]}.wav'
        with wave.open(str(destination), "wb") as wav:
            wav.setparams((1, 2, 16000, window.size, "NONE", "not compressed"))
            wav.writeframes(window.astype("<i2", copy=False).tobytes())
        aligned.append(dict(entry, wav_path=str(destination)))
    return aligned


def _augmenter():
    from microwakeword.audio.augmentation import Augmentation

    return Augmentation(
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


def build_features(
    manifest_path: Path,
    output: Path,
    positive_text: str | None = None,
    truths: set[str] | None = None,
    splits: set[str] | None = None,
    dry_run: bool = False,
) -> dict:
    manifest = validate_manifest(manifest_path)
    selected = select_entries(manifest, positive_text)
    selected = [
        entry
        for entry in selected
        if (not truths or entry["truth"] in truths)
        and (not splits or entry["split"] in splits)
    ]
    report = {
        "schema_version": SCHEMA_VERSION,
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": sha256(manifest_path),
        "positive_text": positive_text,
        "build": {
            "random_seed": 231,
            "training_repeat": 2,
            "training_slide_frames": 5,
            "held_out_audio": "clean",
            "positive_alignment": "phrase-span-centered-2000ms",
        },
        "entries": [
            {
                **{
                    key: entry[key]
                    for key in (
                        "id",
                        "wav_path",
                        "sha256",
                        "truth",
                        "split",
                        "text",
                        "provenance",
                        "human_reviewed",
                        "training_eligible",
                    )
                },
                "phrase_span": entry.get("phrase_span"),
            }
            for entry in selected
        ],
    }
    if dry_run:
        return report
    if not selected:
        raise ValueError("no eligible entries remain after filters")
    random.seed(231)
    global RaggedMmap, SpectrogramGeneration
    try:
        import numpy as np
    except ModuleNotFoundError:
        np = None
    if RaggedMmap is None:
        from mmap_ninja.ragged import RaggedMmap as _RaggedMmap

        RaggedMmap = _RaggedMmap
    if SpectrogramGeneration is None:
        from microwakeword.audio.spectrograms import (
            SpectrogramGeneration as _SpectrogramGeneration,
        )

        SpectrogramGeneration = _SpectrogramGeneration

    if np is not None:
        np.random.seed(231)
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="promoted-audio-") as temporary:
        for truth in sorted({entry["truth"] for entry in selected}):
            for split in ("train", "validation", "test"):
                group = [
                    entry
                    for entry in selected
                    if entry["truth"] == truth and entry["split"] == split
                ]
                if not group:
                    continue
                group = phrase_aligned_entries(group, Path(temporary) / truth / split)
                feature_split = {
                    "train": "training",
                    "validation": "validation",
                    "test": "testing",
                }[split]
                spectrograms = SpectrogramGeneration(
                    clips=_clips(group),
                    augmenter=_augmenter() if split == "train" else None,
                    slide_frames=5 if split == "train" else None,
                    step_ms=10,
                )
                destination = output / truth / feature_split / "wakeword_mmap"
                destination.parent.mkdir(parents=True, exist_ok=True)
                RaggedMmap.from_generator(
                    out_dir=str(destination),
                    sample_generator=spectrograms.spectrogram_generator(
                        split="selected", repeat=2 if split == "train" else 1
                    ),
                    batch_size=100,
                    verbose=True,
                )
    (output / OUTPUT_MANIFEST_NAME).write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--positive-text", help="Exact positive text, e.g. 'Hi-Fi Kizz'"
    )
    parser.add_argument("--truth", action="append", choices=sorted(TRUTHS))
    parser.add_argument("--split", action="append", choices=sorted(SPLITS))
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and report provenance only"
    )
    args = parser.parse_args()
    report = build_features(
        args.manifest,
        args.output,
        args.positive_text,
        set(args.truth) if args.truth else None,
        set(args.split) if args.split else None,
        args.dry_run,
    )
    print(
        json.dumps(
            {"entries": len(report["entries"]), "dry_run": args.dry_run}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
