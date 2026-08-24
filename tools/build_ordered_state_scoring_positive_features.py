#!/usr/bin/env python3
"""Build exact-occurrence positive RaggedMmaps for ordered-state scoring.

Phrase coordinates use the product feature timeline: frame ``i`` covers the
10 ms coordinate ``i * 0.01``.  The valid end coordinate is therefore
``N * 0.01`` for an ``[N, 40]`` feature array.  This deliberately does not
use a record's declared audio duration or add a 30 ms frontend tail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from mmap_ninja.ragged import RaggedMmap

FEATURE_STEP_SECONDS = 0.01
FEATURE_BINS = 40
INPUT_SPLITS = frozenset(("validation", "test"))
OUTPUT_SPLITS = {"validation": "validation", "test": "testing"}
FORBIDDEN_PATH_PARTS = frozenset(
    ("quarantine", "observations", "evidence", "false-wakes")
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: Path) -> str:
    if not path.is_dir():
        raise ValueError(f"RaggedMmap output was not created: {path}")
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"RaggedMmap output is empty: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _reject_forbidden_path(path: Path) -> None:
    parts = {part.casefold() for part in path.resolve().parts}
    if parts & FORBIDDEN_PATH_PARTS:
        raise ValueError(f"quarantine/evidence path is not allowed: {path}")


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    _reject_forbidden_path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        value = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if isinstance(value, dict) and isinstance(value.get("records"), list):
        records = value["records"]
    elif isinstance(value, list):
        records = value
    else:
        raise ValueError(f"manifest must contain a record list: {path}")
    if not records:
        raise ValueError(f"manifest is empty: {path}")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"manifest records must be objects: {path}")
    return records


def _finite(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _feature_path(record: Mapping[str, Any], manifest_path: Path) -> Path:
    raw = record.get("features_path")
    if not isinstance(raw, str) or not raw:
        raise ValueError("features_path is required")
    path = Path(raw)
    path = path if path.is_absolute() else manifest_path.parent / path
    _reject_forbidden_path(path)
    if not path.is_file():
        raise ValueError(f"feature file does not exist: {path}")
    return path


def _load_features(path: Path) -> np.ndarray:
    try:
        loaded = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not load features: {path}") from exc
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            if "features" not in loaded.files:
                raise ValueError(f"feature npz must contain features: {path}")
            features = loaded["features"]
        finally:
            # NpzFile owns a zip handle; close it after extracting the array.
            loaded.close()
    else:
        features = loaded
    features = np.asarray(features)
    if features.ndim != 2 or features.shape[1] != FEATURE_BINS:
        raise ValueError(f"features must have shape [N, 40]: {path}")
    if features.shape[0] == 0 or not np.all(np.isfinite(features)):
        raise ValueError(f"features must be non-empty and finite: {path}")
    return features.astype(np.float32, copy=False)


def _validate_record(
    record: Mapping[str, Any], manifest_path: Path
) -> tuple[np.ndarray, dict[str, Any]]:
    source_id = record.get("source_id")
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError("source_id is required")
    split = record.get("split")
    if split == "train":
        raise ValueError(f"train records are not allowed: {source_id}")
    if split not in INPUT_SPLITS:
        raise ValueError(f"record split must be validation or test: {source_id}")
    declared_step = _finite(
        record.get("feature_frame_step_seconds"),
        "feature_frame_step_seconds",
    )
    if not math.isclose(
        declared_step, FEATURE_STEP_SECONDS, rel_tol=1e-6, abs_tol=1e-9
    ):
        raise ValueError(f"feature frame cadence must be 0.01 seconds: {source_id}")
    span = record.get("phrase_span")
    if not isinstance(span, Mapping):
        raise ValueError(f"phrase_span is required: {source_id}")
    start = _finite(span.get("start_s"), "phrase_span.start_s")
    end = _finite(span.get("end_s"), "phrase_span.end_s")
    features_path = _feature_path(record, manifest_path)
    features = _load_features(features_path)
    # Coordinate contract: no undocumented 30 ms tail; end <= N * 10 ms.
    timeline_end = features.shape[0] * FEATURE_STEP_SECONDS
    if not (0.0 <= start < end <= timeline_end + 1e-9):
        raise ValueError(
            f"phrase_span must satisfy 0 <= start < end <= {timeline_end:g}: {source_id}"
        )
    source_group = record.get("source_group")
    if not isinstance(source_group, str) or not source_group.strip():
        raise ValueError(f"source_group is required: {source_id}")
    return features, {
        "source_id": source_id,
        "source_group": source_group,
        "split": split,
        "feature_path": features_path,
        "feature_sha256": sha256_file(features_path),
        "phrase_start_s": start,
        "phrase_end_s": end,
    }


def _records_from_manifests(
    manifest_paths: Iterable[Path],
) -> tuple[dict[str, list[tuple[np.ndarray, dict[str, Any]]]], list[dict[str, str]]]:
    by_split: dict[str, list[tuple[np.ndarray, dict[str, Any]]]] = {
        "validation": [],
        "test": [],
    }
    source_ids: set[str] = set()
    manifest_sources = []
    paths = list(manifest_paths)
    if not paths:
        raise ValueError("at least one frame-supervision manifest is required")
    for manifest_path in paths:
        records = _load_manifest(manifest_path)
        splits = {record.get("split") for record in records}
        if len(splits) != 1:
            raise ValueError(f"manifest has mixed or missing splits: {manifest_path}")
        manifest_split = next(iter(splits))
        if manifest_split not in INPUT_SPLITS:
            raise ValueError(
                f"manifest contains train or invalid split {manifest_split!r}: {manifest_path}"
            )
        manifest_sources.append(
            {"path": str(manifest_path), "sha256": sha256_file(manifest_path)}
        )
        for record in records:
            features, metadata = _validate_record(record, manifest_path)
            source_id = metadata["source_id"]
            if source_id in source_ids:
                raise ValueError(f"duplicate source_id: {source_id}")
            source_ids.add(source_id)
            by_split[metadata["split"]].append((features, metadata))
    missing = sorted(split for split, records in by_split.items() if not records)
    if missing:
        raise ValueError(f"missing required split: {', '.join(missing)}")
    return by_split, manifest_sources


def _write_split(
    output: Path,
    split: str,
    records: list[tuple[np.ndarray, dict[str, Any]]],
) -> tuple[str, list[dict[str, Any]]]:
    destination = output / "positive" / OUTPUT_SPLITS[split] / "wakeword_mmap"

    def samples():
        for features, _ in records:
            yield features

    RaggedMmap.from_generator(
        out_dir=str(destination),
        sample_generator=samples(),
        batch_size=100,
        verbose=False,
    )
    mmap_hash = sha256_directory(destination)
    occurrences = []
    for index, (_, metadata) in enumerate(records):
        occurrences.append(
            {
                "source_id": metadata["source_id"],
                "source_group": metadata["source_group"],
                "split": split,
                "item_index": index,
                "features_path": str(metadata["feature_path"]),
                "feature_sha256": metadata["feature_sha256"],
                "phrase_span": {
                    "start_s": metadata["phrase_start_s"],
                    "end_s": metadata["phrase_end_s"],
                },
            }
        )
    return mmap_hash, occurrences


def build_positive_features(
    manifest_paths: Iterable[Path], output: Path
) -> dict[str, Any]:
    manifest_path = output / "ordered-state-positive-occurrences.json"
    if manifest_path.exists() or any(
        (output / "positive" / name / "wakeword_mmap").exists()
        for name in OUTPUT_SPLITS.values()
    ):
        raise ValueError(f"scoring-positive output already exists: {output}")
    by_split, manifest_sources = _records_from_manifests(manifest_paths)
    output.mkdir(parents=True, exist_ok=True)
    split_hashes = {}
    occurrences = []
    for split in ("validation", "test"):
        mmap_hash, split_occurrences = _write_split(output, split, by_split[split])
        split_hashes[split] = mmap_hash
        occurrences.extend(split_occurrences)
    manifest = {
        "schema_version": 1,
        "coordinate_contract": {
            "feature_frame_step_seconds": FEATURE_STEP_SECONDS,
            "phrase_end_limit": "N * 0.01; no 30ms frontend tail",
        },
        "source_manifests": manifest_sources,
        "ragged_mmap_sha256": split_hashes,
        "occurrences": occurrences,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "manifest": str(manifest_path),
        "validation_examples": len(by_split["validation"]),
        "test_examples": len(by_split["test"]),
        "ragged_mmap_sha256": split_hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(build_positive_features(args.manifest, args.output), sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
