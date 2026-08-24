#!/usr/bin/env python3
"""Freeze quarantined wake observations as training-ineligible evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from math import isfinite
from pathlib import Path
import shutil
import wave
from typing import Any

OBSERVATION_DIRECTORIES = ("wakes", "false-wakes")
UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def parse_utc_timestamp(value: str, argument: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        timestamp = datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError(
            f"{argument} must be an ISO-8601 timestamp with an explicit UTC offset"
        ) from error
    if timestamp.tzinfo is None:
        raise ValueError(f"{argument} must include a UTC offset")
    timestamp = timestamp.astimezone(timezone.utc)
    return timestamp


def canonical_timestamp(timestamp: datetime) -> str:
    return timestamp.astimezone(timezone.utc).strftime(UTC_FORMAT)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def received_at_utc(value: Any, metadata_path: Path) -> datetime:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not isfinite(float(value)):
            raise ValueError(f"{metadata_path}: received_at must be finite")
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        try:
            return parse_utc_timestamp(value, f"{metadata_path}: received_at")
        except ValueError as error:
            raise ValueError(str(error)) from error
    raise ValueError(f"{metadata_path}: received_at must be an epoch or UTC timestamp")


def validate_wav(path: Path) -> None:
    try:
        with wave.open(str(path), "rb") as audio:
            if audio.getnchannels() != 1:
                raise ValueError(f"{path}: WAV must be mono")
            if audio.getframerate() != 16000:
                raise ValueError(f"{path}: WAV sample rate must be 16000 Hz")
            if audio.getsampwidth() != 2:
                raise ValueError(f"{path}: WAV sample width must be 16-bit PCM")
            if audio.getcomptype() != "NONE":
                raise ValueError(f"{path}: WAV compression must be PCM")
    except (wave.Error, EOFError) as error:
        raise ValueError(f"{path}: invalid WAV: {error}") from error


def relative_source_path(path: str, corpus: Path, metadata_path: Path) -> Path:
    if not isinstance(path, str) or not path or Path(path).is_absolute():
        raise ValueError(f"{metadata_path}: path must be a relative corpus path")
    resolved = (corpus / Path(path)).resolve()
    if not path_inside(resolved, corpus):
        raise ValueError(f"{metadata_path}: path escapes corpus: {path}")
    if not resolved.is_file():
        raise ValueError(f"{metadata_path}: referenced audio does not exist: {path}")
    return resolved.relative_to(corpus)


def collect_observations(
    corpus: Path, since: datetime, frozen_at: datetime
) -> list[tuple[Path, Path, dict[str, Any], datetime]]:
    observations: list[tuple[Path, Path, dict[str, Any], datetime]] = []
    seen_ids: set[str] = set()
    for directory_name in OBSERVATION_DIRECTORIES:
        directory = corpus / "observations" / directory_name
        if not directory.is_dir():
            continue
        for metadata_path in sorted(directory.glob("*.json")):
            try:
                metadata = json.loads(metadata_path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"{metadata_path}: invalid JSON: {error}") from error
            if not isinstance(metadata, dict):
                raise ValueError(f"{metadata_path}: metadata must be a JSON object")
            received_at = received_at_utc(metadata.get("received_at"), metadata_path)
            if not since <= received_at <= frozen_at:
                continue
            observation_id = metadata.get("observation_id")
            if not isinstance(observation_id, str) or not observation_id:
                raise ValueError(f"{metadata_path}: missing observation_id")
            if metadata_path.stem != observation_id:
                raise ValueError(
                    f"{metadata_path}: observation_id does not match metadata filename"
                )
            if observation_id in seen_ids:
                raise ValueError(f"duplicate observation_id: {observation_id}")
            seen_ids.add(observation_id)
            audio_relative = relative_source_path(
                metadata.get("path"), corpus, metadata_path
            )
            expected_audio_relative = metadata_path.with_suffix(".wav").relative_to(
                corpus
            )
            if audio_relative != expected_audio_relative:
                raise ValueError(
                    f"{metadata_path}: audio path does not match observation filename"
                )
            audio_path = corpus / audio_relative
            actual_hash = sha256_file(audio_path)
            recorded_hash = metadata.get("sha256")
            if not isinstance(recorded_hash, str) or recorded_hash != actual_hash:
                raise ValueError(
                    f"{metadata_path}: WAV SHA-256 mismatch: "
                    f"metadata={recorded_hash!r} actual={actual_hash}"
                )
            validate_wav(audio_path)
            observations.append((metadata_path, audio_path, metadata, received_at))
    if not observations:
        raise ValueError("selection is empty")
    return observations


def freeze_snapshot(
    corpus: Path,
    output: Path,
    since: datetime,
    frozen_at: datetime,
    reviewer: str,
    review_note: str,
) -> dict[str, Any]:
    corpus = corpus.resolve()
    output = output.resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise ValueError(f"output manifest already exists: {manifest_path}")
    if not corpus.is_dir():
        raise ValueError(f"corpus does not exist or is not a directory: {corpus}")
    if frozen_at < since:
        raise ValueError("--frozen-at must not be earlier than --since")
    if not reviewer.strip() or not review_note.strip():
        raise ValueError("reviewer and review note must be non-empty")
    observations = collect_observations(corpus, since, frozen_at)
    output.mkdir(parents=True, exist_ok=True)

    manifest_observations = []
    copied_destinations: set[Path] = set()
    for metadata_path, audio_path, metadata, received_at in observations:
        metadata_relative = metadata_path.relative_to(corpus)
        audio_relative = audio_path.relative_to(corpus)
        destination_metadata = output / metadata_relative
        destination_audio = output / audio_relative
        if (
            destination_metadata in copied_destinations
            or destination_audio in copied_destinations
        ):
            raise ValueError(f"duplicate snapshot destination for {metadata_path}")
        copied_destinations.update((destination_metadata, destination_audio))
        destination_metadata.parent.mkdir(parents=True, exist_ok=True)
        destination_audio.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(audio_path, destination_audio)
        shutil.copyfile(metadata_path, destination_metadata)
        manifest_observations.append(
            {
                "observation_id": metadata["observation_id"],
                "path": str(audio_relative),
                "audio_path": str(audio_relative),
                "metadata_path": str(metadata_relative),
                "audio_sha256": sha256_file(destination_audio),
                "received_at": canonical_timestamp(received_at),
                "weak_label": "false_wake_no_command",
                "human_review_basis": (
                    "quarantined device wake observation selected for human review; "
                    "not automatically promoted to training"
                ),
                "review": {
                    "reviewer": reviewer,
                    "note": review_note,
                    "basis": "user confirmed the selected wakes were false",
                },
            }
        )

    manifest = {
        "schema_version": 1,
        "source_corpus": str(corpus),
        "snapshot_root": str(output),
        "training_eligible": False,
        "source_provenance": {
            "corpus": str(corpus),
            "observation_directories": [
                f"observations/{name}" for name in OBSERVATION_DIRECTORIES
            ],
            "selection_since": canonical_timestamp(since),
            "frozen_at": canonical_timestamp(frozen_at),
        },
        "selection_timestamp": canonical_timestamp(frozen_at),
        "review": {
            "reviewer": reviewer,
            "note": review_note,
            "basis": "human review required before any promotion to hard negatives",
        },
        "observation_count": len(manifest_observations),
        "observations": manifest_observations,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--since", required=True)
    parser.add_argument("--frozen-at", required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--review-note", required=True)
    args = parser.parse_args()
    manifest = freeze_snapshot(
        args.corpus,
        args.output,
        parse_utc_timestamp(args.since, "--since"),
        parse_utc_timestamp(args.frozen_at, "--frozen-at"),
        args.reviewer,
        args.review_note,
    )
    print(json.dumps({"observation_count": manifest["observation_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
