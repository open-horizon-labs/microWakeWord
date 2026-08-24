#!/usr/bin/env python3
"""Freeze an ordered-state experiment's source inventory and split contract.

The tool deliberately creates metadata only.  It never copies source audio, and
it treats quarantined deployment evidence as an invalid training source rather
than trying to infer whether an individual file is useful.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import wave
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - JSON configs need no optional dep.
    yaml = None

SPLITS = {"train", "validation", "test"}
FORBIDDEN_COMPONENTS = {"observations", "false-wakes", "evidence"}
DEFAULT_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
SCHEMA_VERSION = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _hash_records(records: list[dict]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(_canonical_json(record))
        digest.update(b"\n")
    return digest.hexdigest()


def _forbidden_path(path: Path) -> str | None:
    lowered = {part.casefold() for part in path.resolve().parts}
    for component in FORBIDDEN_COMPONENTS:
        if component in lowered:
            return component
    return None


def _load_config(path: Path) -> dict:
    try:
        raw = path.read_text()
    except OSError as error:
        raise ValueError(f"cannot read config: {path}") from error
    if path.suffix.casefold() in {".yaml", ".yml"}:
        if yaml is None:
            raise ValueError("YAML config requires PyYAML")
        config = yaml.safe_load(raw)
    else:
        config = json.loads(raw)
    if not isinstance(config, dict):
        raise ValueError("experiment config must be an object")
    return config


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    value = float(value)
    if value < 0 or value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{label} must be finite and non-negative")
    return value


def _wav_exposure(path: Path) -> float | None:
    if path.suffix.casefold() != ".wav":
        return None
    try:
        with wave.open(str(path), "rb") as audio:
            rate = audio.getframerate()
            return audio.getnframes() / rate if rate else None
    except (OSError, wave.Error):
        return None


def _source_files(root: Path, extensions: set[str]) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(
                f"source contains a symlink; resolve it explicitly: {path}"
            )
        if path.is_file() and (not extensions or path.suffix.casefold() in extensions):
            files.append(path)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def _validate_identity(
    source: dict, source_index: int
) -> tuple[str, str, str, str, str, str | None, str | None, set[str]]:
    prefix = f"sources[{source_index}]"
    source_id = source.get("source_id")
    split = source.get("split")
    speaker_id = source.get("speaker_id")
    session_id = source.get("session_id")
    truth = source.get("truth", "negative")
    category = source.get("category")
    channel = source.get("channel")
    source_family = source.get("source_family")
    for name, value in (
        ("source_id", source_id),
        ("split", split),
        ("speaker_id", speaker_id),
        ("session_id", session_id),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{prefix} requires non-empty {name}")
    if split not in SPLITS:
        raise ValueError(f"{prefix}.split must be train, validation, or test")
    if not isinstance(truth, str) or not truth:
        raise ValueError(f"{prefix}.truth must be non-empty")
    if not isinstance(category, str) or not category.strip():
        raise ValueError(f"{prefix} requires non-empty category")
    for name, value in (("channel", channel), ("source_family", source_family)):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{prefix}.{name} must be non-empty when provided")
    extensions = source.get("extensions")
    if extensions is None:
        extension_set = DEFAULT_EXTENSIONS
    elif isinstance(extensions, list) and all(
        isinstance(item, str) for item in extensions
    ):
        extension_set = {
            item if item.startswith(".") else f".{item}" for item in extensions
        }
        extension_set = {item.casefold() for item in extension_set}
    else:
        raise ValueError(f"{prefix}.extensions must be a list of suffixes")
    return (
        source_id,
        split,
        speaker_id,
        session_id,
        category,
        channel,
        source_family,
        extension_set,
    )


def prepare(config_path: Path, output_path: Path) -> dict:
    config = _load_config(config_path)
    sources = config.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("experiment config requires a non-empty sources list")
    if config.get("threshold_selection_split", "validation") != "validation":
        raise ValueError("threshold_selection_split must be validation")

    seen_ids: set[str] = set()
    identities: dict[str, dict[str, str]] = {
        key: {} for key in ("source_id", "speaker_id", "session_id")
    }
    source_records = []
    all_files = []
    seen_splits: set[str] = set()
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ValueError(f"sources[{index}] must be an object")
        (
            source_id,
            split,
            speaker_id,
            session_id,
            category,
            channel,
            source_family,
            extensions,
        ) = _validate_identity(source, index)
        if source_id in seen_ids:
            raise ValueError(f"duplicate source_id: {source_id}")
        seen_ids.add(source_id)
        seen_splits.add(split)
        for kind, value in (
            ("source_id", source_id),
            ("speaker_id", speaker_id),
            ("session_id", session_id),
        ):
            prior = identities[kind].get(value)
            if prior is not None and prior != split:
                raise ValueError(
                    f"{kind} {value!r} crosses splits: {prior} and {split}"
                )
            identities[kind][value] = split

        root = Path(str(source.get("path", ""))).expanduser()
        if not root.is_absolute():
            root = (config_path.parent / root).resolve()
        else:
            root = root.resolve()
        forbidden = _forbidden_path(root)
        if forbidden:
            raise ValueError(
                f"source {source_id} is forbidden: path contains {forbidden!r}: {root}"
            )
        if not root.is_dir():
            raise ValueError(f"source {source_id} is not a directory: {root}")
        files = _source_files(root, extensions)
        if not files:
            raise ValueError(f"source {source_id} contains no configured audio files")

        declared_exposure = source.get("exposure_seconds")
        if declared_exposure is not None:
            declared_exposure = _number(
                declared_exposure, f"sources[{index}].exposure_seconds"
            )
        file_records = []
        for path in files:
            relative = path.relative_to(root).as_posix()
            file_exposure = _wav_exposure(path)
            record = {
                "path": relative,
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "exposure_seconds": file_exposure,
                "category": category,
            }
            for name, value in (("channel", channel), ("source_family", source_family)):
                if value is not None:
                    record[name] = value
            file_records.append(record)
            all_files.append(
                {
                    **record,
                    "source_id": source_id,
                    "split": split,
                    "speaker_id": speaker_id,
                    "session_id": session_id,
                    "truth": source.get("truth", "negative"),
                    "category": category,
                    "source_root": str(root),
                }
            )
            for name, value in (
                ("channel", channel),
                ("source_family", source_family),
            ):
                if value is not None:
                    all_files[-1][name] = value
        inferred = sum(
            item["exposure_seconds"]
            for item in file_records
            if item["exposure_seconds"] is not None
        )
        exposure = declared_exposure if declared_exposure is not None else inferred
        if exposure <= 0:
            raise ValueError(
                f"source {source_id} needs positive exposure_seconds or readable WAV files"
            )
        source_record = {
            "source_id": source_id,
            "path": str(root),
            "path_sha256": _hash_records(file_records),
            "split": split,
            "truth": source.get("truth", "negative"),
            "category": category,
            "speaker_id": speaker_id,
            "session_id": session_id,
            "file_count": len(file_records),
            "bytes": sum(item["bytes"] for item in file_records),
            "exposure_seconds": exposure,
            "exposure_declared": declared_exposure is not None,
            "files": file_records,
        }
        for name, value in (("channel", channel), ("source_family", source_family)):
            if value is not None:
                source_record[name] = value
        source_records.append(source_record)

    missing_splits = sorted(SPLITS - seen_splits)
    if missing_splits:
        raise ValueError(
            "experiment requires train, validation, and test sources; "
            f"missing={missing_splits}"
        )

    split_category_counts: dict[str, dict[str, dict[str, float | int]]] = {
        split: {} for split in sorted(SPLITS)
    }
    for source in source_records:
        bucket = split_category_counts[source["split"]].setdefault(
            source["category"], {"sources": 0, "files": 0, "exposure_seconds": 0.0}
        )
        bucket["sources"] += 1
        bucket["files"] += source["file_count"]
        bucket["exposure_seconds"] += source["exposure_seconds"]
    exposure_seconds_by_split_and_category = {
        split: {
            category: values["exposure_seconds"]
            for category, values in sorted(categories.items())
        }
        for split, categories in split_category_counts.items()
    }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": config.get("experiment", "ordered-state-kizz"),
        "config": str(config_path.resolve()),
        "config_sha256": sha256(config_path),
        "training_eligible": True,
        "threshold_selection_split": "validation",
        "source_disjoint": {
            "source_id": True,
            "speaker_id": True,
            "session_id": True,
        },
        "sources": sorted(source_records, key=lambda item: item["source_id"]),
        "files": sorted(all_files, key=lambda item: (item["source_id"], item["path"])),
        "counts": {
            "sources": len(source_records),
            "files": len(all_files),
            "bytes": sum(item["bytes"] for item in all_files),
            "exposure_seconds": sum(
                item["exposure_seconds"] for item in source_records
            ),
            "exposure_seconds_by_split": {
                split: sum(
                    item["exposure_seconds"]
                    for item in source_records
                    if item["split"] == split
                )
                for split in sorted(SPLITS)
            },
            "counts_by_split_and_category": split_category_counts,
            "exposure_seconds_by_split_and_category": exposure_seconds_by_split_and_category,
        },
        "inventory_sha256": _hash_records(all_files),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare(args.config, args.output)
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
