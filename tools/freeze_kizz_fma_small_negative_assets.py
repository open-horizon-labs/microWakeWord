#!/usr/bin/env python3
"""Freeze FMA-small train/validation music assets, preserving official test."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.mine_kizz_librispeech_hard_negatives import _atomic_json, _binding, sha256_file


def _columns(groups: Sequence[str], fields: Sequence[str]) -> dict[tuple[str, str], int]:
    if len(groups) != len(fields):
        raise ValueError("FMA metadata header widths differ")
    result = {(group, field): index for index, (group, field) in enumerate(zip(groups, fields))}
    required = {
        ("album", "id"),
        ("artist", "id"),
        ("set", "split"),
        ("set", "subset"),
        ("track", "genre_top"),
        ("track", "license"),
    }
    missing = required - result.keys()
    if missing:
        raise ValueError(f"FMA metadata is missing columns: {sorted(missing)}")
    return result


def freeze(
    tracks_csv: Path,
    audio_root: Path,
    archive: Path,
    output: Path,
) -> dict[str, Any]:
    tracks_csv = tracks_csv.expanduser().resolve()
    audio_root = audio_root.expanduser().resolve()
    archive = archive.expanduser().resolve()
    # Bind large immutable inputs before the expensive full-decode pass.  A
    # disconnected volume then fails before minutes of validation work rather
    # than after the final track has already been scanned.
    tracks_binding = _binding(tracks_csv)
    archive_binding = _binding(archive)

    # FMA's official artist partition is disjoint, but compilation albums can
    # contain multiple artists assigned to different splits.  Treat album as a
    # recording/session identity and exclude every development row from an
    # album touching more than one official split, including the unread test
    # partition.
    albums_by_split = {"train": set(), "validation": set(), "test": set()}
    with tracks_csv.open(newline="", encoding="utf-8") as source:
        reader = csv.reader(source)
        groups = next(reader)
        fields = next(reader)
        index_header = next(reader)
        if not index_header or index_header[0] != "track_id":
            raise ValueError("FMA metadata index header drift")
        columns = _columns(groups, fields)
        for row in reader:
            if not row or row[columns[("set", "subset")]] != "small":
                continue
            official_split = row[columns[("set", "split")]]
            if official_split not in {"training", "validation", "test"}:
                raise ValueError(f"unsupported FMA split: {official_split}")
            split = {
                "training": "train",
                "validation": "validation",
                "test": "test",
            }[official_split]
            album = row[columns[("album", "id")]].strip()
            albums_by_split[split].add(f"fma-album:{album}")
    cross_split_albums = set()
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        cross_split_albums.update(albums_by_split[left] & albums_by_split[right])

    with tracks_csv.open(newline="", encoding="utf-8") as source:
        reader = csv.reader(source)
        groups = next(reader)
        fields = next(reader)
        index_header = next(reader)
        if not index_header or index_header[0] != "track_id":
            raise ValueError("FMA metadata index header drift")
        columns = _columns(groups, fields)
        examples: list[dict[str, Any]] = []
        official_test_tracks = 0
        artists_by_split = {"train": set(), "validation": set(), "test": set()}
        counts = {"train": 0, "validation": 0}
        seconds = {"train": 0.0, "validation": 0.0}
        genres = {"train": set(), "validation": set()}
        excluded_unlicensed = {"train": 0, "validation": 0}
        excluded_cross_split_album = {"train": 0, "validation": 0}
        quarantined_audio: list[dict[str, Any]] = []
        for row in reader:
            if not row or row[columns[("set", "subset")]] != "small":
                continue
            track_id = int(row[0])
            official_split = row[columns[("set", "split")]]
            artist = row[columns[("artist", "id")]].strip()
            album = row[columns[("album", "id")]].strip()
            if official_split not in {"training", "validation", "test"}:
                raise ValueError(f"unsupported FMA split: {official_split}")
            split = {"training": "train", "validation": "validation", "test": "test"}[
                official_split
            ]
            artist_id = f"fma-artist:{artist}"
            artists_by_split[split].add(artist_id)
            if split == "test":
                official_test_tracks += 1
                continue
            album_id = f"fma-album:{album}"
            if album_id in cross_split_albums:
                excluded_cross_split_album[split] += 1
                continue
            license_name = row[columns[("track", "license")]].strip()
            genre = row[columns[("track", "genre_top")]].strip() or "unknown"
            if not license_name:
                excluded_unlicensed[split] += 1
                continue
            stem = f"{track_id:06d}"
            path = audio_root / stem[:3] / f"{stem}.mp3"
            if not path.is_file():
                raise FileNotFoundError(path)
            digest = sha256_file(path)
            try:
                with sf.SoundFile(path) as audio:
                    samplerate = int(audio.samplerate)
                    channels = int(audio.channels)
                    expected_frames = len(audio)
                    decoded_frames = 0
                    finite = True
                    for block in audio.blocks(
                        blocksize=65_536, dtype="float32", always_2d=True
                    ):
                        decoded_frames += len(block)
                        finite = finite and bool(np.all(np.isfinite(block)))
                if (
                    samplerate <= 0
                    or channels <= 0
                    or expected_frames <= 0
                    or decoded_frames != expected_frames
                    or not finite
                ):
                    raise ValueError("full decode contract drift")
            except (sf.LibsndfileError, RuntimeError, ValueError):
                quarantined_audio.append(
                    {
                        "track_id": track_id,
                        "path": str(path),
                        "audio_sha256": digest,
                        "reason": "full_decode_contract_error",
                        "split": split,
                    }
                )
                continue
            duration = decoded_frames / samplerate
            examples.append(
                {
                    "source_id": f"fma-small:{track_id}",
                    "path": str(path),
                    "audio_sha256": digest,
                    "duration_seconds": duration,
                    "speaker_id": artist_id,
                    "session_id": album_id,
                    "ancestry_id": artist_id,
                    "source_group": "music",
                    "semantic_label": "non_wake_music",
                    "category": genre,
                    "source": "Free Music Archive small",
                    "provider": "fma",
                    "license": license_name,
                    "split": split,
                    "label": 0,
                    "training_eligible": split == "train",
                    "locked_holdout": False,
                    "locked_deployment_anchor": False,
                }
            )
            counts[split] += 1
            seconds[split] += duration
            genres[split].add(genre)
    if not examples or not all(counts.values()) or official_test_tracks < 1:
        raise ValueError("FMA-small requires train, validation, and preserved test tracks")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = artists_by_split[left] & artists_by_split[right]
        if overlap:
            raise ValueError(f"FMA official {left}/{right} artist overlap: {sorted(overlap)[:3]}")
    sessions_by_split = {
        split: {row["session_id"] for row in examples if row["split"] == split}
        for split in ("train", "validation")
    }
    session_overlap = sessions_by_split["train"] & sessions_by_split["validation"]
    if session_overlap:
        raise ValueError(
            f"FMA usable train/validation album overlap: {sorted(session_overlap)[:3]}"
        )
    payload = {
        "schema_version": 1,
        "kind": "kizz_fma_small_negative_assets",
        "source": "Free Music Archive small",
        "bindings": {
            "tracks_csv": tracks_binding,
            "archive": archive_binding,
        },
        "partition": {
            "source": "official FMA set.split",
            "identity_unit": "artist.id",
            "official_test_preserved_unread": True,
            "cross_split_album_policy": "exclude_development_rows_if_album_touches_multiple_official_splits",
            "cross_split_albums": len(cross_split_albums),
        },
        "counts": {
            "examples": len(examples),
            "by_split": counts,
            "hours_by_split": {
                split: seconds[split] / 3600 for split in ("train", "validation")
            },
            "artists_by_split": {
                split: len(artists_by_split[split])
                for split in ("train", "validation", "test")
            },
            "genres_by_split": {
                split: sorted(genres[split]) for split in ("train", "validation")
            },
            "preserved_official_test_tracks": official_test_tracks,
            "excluded_unlicensed_by_split": excluded_unlicensed,
            "excluded_cross_split_album_by_split": excluded_cross_split_album,
            "quarantined_audio_files": len(quarantined_audio),
        },
        "quarantine": quarantined_audio,
        "examples": examples,
    }
    output = output.expanduser().resolve()
    _atomic_json(output, payload)
    return {
        "output": str(output),
        "sha256": sha256_file(output),
        **payload["counts"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks-csv", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(freeze(**vars(args)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
