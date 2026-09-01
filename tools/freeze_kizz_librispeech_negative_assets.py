#!/usr/bin/env python3
"""Freeze a speaker-disjoint LibriSpeech negative manifest for hard mining."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import soundfile as sf

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.mine_kizz_librispeech_hard_negatives import _atomic_json, sha256_file


def _speaker_split(speaker: str, validation_fraction: float) -> str:
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be within (0,0.5)")
    bucket = int.from_bytes(
        hashlib.sha256(f"librispeech-speaker:{speaker}".encode()).digest()[:8],
        "big",
    ) / 2**64
    return "validation" if bucket < validation_fraction else "train"


def _speakers(root: Path) -> set[str]:
    return {
        path.name
        for path in root.expanduser().resolve().iterdir()
        if path.is_dir() and path.name.isdigit()
    }


def freeze(
    root: Path,
    output: Path,
    *,
    validation_fraction: float,
    exclude_speaker_root: Path | None = None,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    files = sorted(root.rglob("*.flac"))
    if not files:
        raise ValueError("LibriSpeech root contains no FLAC files")
    excluded_speakers = (
        _speakers(exclude_speaker_root) if exclude_speaker_root is not None else set()
    )
    examples: list[dict[str, Any]] = []
    counts = {"train": 0, "validation": 0}
    seconds = {"train": 0.0, "validation": 0.0}
    speakers_by_split = {"train": set(), "validation": set()}
    excluded_files = 0
    for path in files:
        relative = path.relative_to(root)
        if len(relative.parts) < 3:
            raise ValueError(f"unexpected LibriSpeech path: {relative}")
        speaker, chapter = relative.parts[0], relative.parts[1]
        if speaker in excluded_speakers:
            excluded_files += 1
            continue
        info = sf.info(path)
        if info.samplerate != 16_000 or info.channels != 1 or info.frames <= 0:
            raise ValueError(f"LibriSpeech audio contract drift: {path}")
        split = _speaker_split(speaker, validation_fraction)
        digest = sha256_file(path)
        duration = info.frames / info.samplerate
        source_id = f"librispeech-train-clean-100:{path.stem}"
        speaker_id = f"librispeech-speaker:{speaker}"
        session_id = f"librispeech-chapter:{speaker}:{chapter}"
        examples.append(
            {
                "source_id": source_id,
                "path": str(path),
                "audio_sha256": digest,
                "duration_seconds": duration,
                "speaker_id": speaker_id,
                "session_id": session_id,
                "ancestry_id": speaker_id,
                "source_group": "public_speech",
                "semantic_label": "non_wake_public_speech",
                "source": "OpenSLR LibriSpeech train-clean-100",
                "provider": "openslr_librispeech",
                "license": "CC BY 4.0",
                "split": split,
                "label": 0,
                "training_eligible": split == "train",
                "locked_holdout": False,
                "locked_deployment_anchor": False,
            }
        )
        counts[split] += 1
        seconds[split] += duration
        speakers_by_split[split].add(speaker_id)
    if not examples or not all(counts.values()):
        raise ValueError("LibriSpeech partition requires train and validation examples")
    if speakers_by_split["train"] & speakers_by_split["validation"]:
        raise AssertionError("speaker partition overlap")
    payload = {
        "schema_version": 1,
        "kind": "kizz_librispeech_train_clean_100_negative_assets",
        "source_root": str(root),
        "source": "OpenSLR SLR12 LibriSpeech train-clean-100",
        "license": "CC BY 4.0",
        "partition": {
            "unit": "speaker_id",
            "algorithm": "sha256(librispeech-speaker:<id>)_first_u64_fraction",
            "validation_fraction": validation_fraction,
            "test_created": False,
        },
        "exclusions": {
            "speaker_root": (
                str(exclude_speaker_root.expanduser().resolve())
                if exclude_speaker_root is not None
                else None
            ),
            "speaker_ids": sorted(excluded_speakers),
            "files": excluded_files,
        },
        "counts": {
            "examples": len(examples),
            "by_split": counts,
            "hours_by_split": {
                split: seconds[split] / 3600 for split in ("train", "validation")
            },
            "speakers_by_split": {
                split: len(speakers_by_split[split])
                for split in ("train", "validation")
            },
        },
        "examples": examples,
    }
    output = output.expanduser().resolve()
    _atomic_json(output, payload)
    return {
        "output": str(output),
        "sha256": sha256_file(output),
        **payload["counts"],
        "excluded_files": excluded_files,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--exclude-speaker-root", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(freeze(**vars(args)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
