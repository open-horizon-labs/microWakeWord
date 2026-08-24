#!/usr/bin/env python3
"""Write reproducible validation and untouched-test manifests for feature scoring.

The negative inventory is authoritative: this tool carries its declared
exposure and path hash through unchanged.  Positive occurrences are derived
from the stored RaggedMmap item lengths, so they do not depend on a second
alignment or duration source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from mmap_ninja.ragged import RaggedMmap

FEATURE_STEP_SECONDS = 0.01
FROZEN_SCHEMA_VERSION = 1
SCORING_SCHEMA_VERSION = 1
SPLIT_MAP = {"validation": "validation", "test": "testing"}
FORBIDDEN_COMPONENTS = {"observations", "false-wakes", "evidence"}
MIN_NEGATIVE_EXPOSURE_SECONDS = 100.0 * 60.0 * 60.0


def sha256_path(path: Path) -> str:
    """Hash a file or directory with stable relative-path framing."""
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    paths = sorted(item for item in path.rglob("*") if item.is_file())
    for child in paths:
        relative = child.relative_to(path) if path.is_dir() else Path(child.name)
        encoded = relative.as_posix().encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        with child.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _forbidden_component(path: Path) -> str | None:
    components = {part.casefold() for part in path.resolve().parts}
    return next(
        (
            component
            for component in sorted(FORBIDDEN_COMPONENTS)
            if component in components
        ),
        None,
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON input: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON input must be an object: {path}")
    return value


def _positive_sources(
    feature_root: Path,
    target_split: str,
    occurrence_manifest_path: Path | None = None,
) -> tuple[dict[str, Any], str, str, str | None]:
    if target_split not in SPLIT_MAP:
        raise ValueError(f"unsupported scoring split: {target_split}")
    path = (
        feature_root / "positive" / SPLIT_MAP[target_split] / "wakeword_mmap"
    ).resolve()
    forbidden = _forbidden_component(path)
    if forbidden:
        raise ValueError(f"positive feature path is quarantined ({forbidden}): {path}")
    if not path.is_dir():
        raise ValueError(f"positive feature path does not exist: {path}")

    mmap = RaggedMmap(path)
    if len(mmap) == 0:
        raise ValueError(f"positive feature mmap is empty: {path}")
    item_frame_counts = []
    for item_index in range(len(mmap)):
        item = mmap[item_index]
        shape = getattr(item, "shape", None)
        if shape is None or len(shape) != 2 or int(shape[1]) != 40:
            raise ValueError(
                f"positive item {item_index} must have shape [N, 40], got {shape}"
            )
        frame_count = int(shape[0])
        if frame_count <= 0:
            raise ValueError(f"positive item {item_index} is empty")
        item_frame_counts.append(frame_count)

    occurrence_manifest_hash = None
    path_hash = sha256_path(path)
    if occurrence_manifest_path is None:
        geometry = "full_item_proxy"
        occurrences = [
            {
                "id": f"canonical-positive-{target_split}-item-{index:06d}",
                "item_index": index,
                "start_seconds": 0.0,
                "end_seconds": frames * FEATURE_STEP_SECONDS,
            }
            for index, frames in enumerate(item_frame_counts)
        ]
    else:
        occurrence_manifest_path = occurrence_manifest_path.resolve()
        forbidden = _forbidden_component(occurrence_manifest_path)
        if forbidden:
            raise ValueError(
                f"positive occurrence manifest is quarantined ({forbidden})"
            )
        occurrence_manifest = _load_json(occurrence_manifest_path)
        if occurrence_manifest.get("schema_version") != 1:
            raise ValueError("positive occurrence manifest has the wrong schema")
        expected_mmap_hash = occurrence_manifest.get("ragged_mmap_sha256", {}).get(
            target_split
        )
        if expected_mmap_hash != path_hash:
            raise ValueError(
                f"positive {target_split} RaggedMmap hash does not match occurrence manifest"
            )
        records = [
            record
            for record in occurrence_manifest.get("occurrences", [])
            if record.get("split") == target_split
        ]
        by_item = {int(record["item_index"]): record for record in records}
        if len(by_item) != len(records) or set(by_item) != set(range(len(mmap))):
            raise ValueError(
                f"positive {target_split} occurrences must map exactly one per item"
            )
        occurrences = []
        for item_index, frames in enumerate(item_frame_counts):
            record = by_item[item_index]
            span = record.get("phrase_span", {})
            start = float(span["start_s"])
            end = float(span["end_s"])
            if not (0 <= start < end <= frames * FEATURE_STEP_SECONDS + 1e-9):
                raise ValueError(
                    f"positive {target_split} item {item_index} has an invalid exact span"
                )
            occurrences.append(
                {
                    "id": str(record["source_id"]),
                    "item_index": item_index,
                    "start_seconds": start,
                    "end_seconds": end,
                    "source_group": record.get("source_group"),
                }
            )
        geometry = "exact_phrase_span"
        occurrence_manifest_hash = sha256_path(occurrence_manifest_path)

    source_id = f"canonical-positive-{target_split}"
    source = {
        "id": source_id,
        "path": str(path),
        "split": target_split,
        "label": "positive",
        "feature_step_seconds": FEATURE_STEP_SECONDS,
        "occurrences": occurrences,
        "positive_item_count": len(occurrences),
        "expected_path_sha256": path_hash,
        "occurrence_geometry": geometry,
    }
    return source, path_hash, geometry, occurrence_manifest_hash


def _negative_sources(
    frozen: Mapping[str, Any], frozen_path: Path, target_split: str
) -> list[dict[str, Any]]:
    if frozen.get("schema_version") != FROZEN_SCHEMA_VERSION:
        raise ValueError("frozen negative manifest has the wrong schema_version")
    if frozen.get("threshold_selection_split") != "validation":
        raise ValueError(
            "frozen negative manifest must select thresholds on validation"
        )
    raw_sources = frozen.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("frozen negative manifest must contain sources")

    result = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_sources):
        if not isinstance(raw, Mapping):
            raise ValueError(f"frozen sources[{index}] must be an object")
        split = raw.get("split")
        if split == "train":
            continue
        if split not in {"validation", "test"}:
            raise ValueError(f"frozen sources[{index}] has unsupported split: {split}")
        if split != target_split:
            continue
        source_id = raw.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"frozen sources[{index}] needs source_id")
        output_id = f"negative-{source_id}"
        if output_id in seen_ids:
            raise ValueError(f"duplicate source id: {output_id}")
        seen_ids.add(output_id)
        exposure = raw.get("exposure_seconds")
        if (
            isinstance(exposure, bool)
            or not isinstance(exposure, (int, float))
            or not math.isfinite(float(exposure))
            or float(exposure) < 0
        ):
            raise ValueError(f"{source_id}: invalid declared exposure_seconds")
        raw_path = raw.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"{source_id}: missing path")
        path = Path(raw_path)
        if not path.is_absolute():
            path = (frozen_path.parent / path).resolve()
        else:
            path = path.resolve()
        forbidden = _forbidden_component(path)
        if forbidden:
            raise ValueError(
                f"negative feature path is quarantined ({forbidden}): {path}"
            )
        if not path.is_dir():
            raise ValueError(f"{source_id}: feature path does not exist: {path}")
        expected_hash = raw.get("path_sha256")
        if not isinstance(expected_hash, str) or not expected_hash:
            raise ValueError(f"{source_id}: missing frozen path_sha256")
        source = {
            "id": output_id,
            "path": str(path),
            "split": target_split,
            "label": "negative",
            "exposure_seconds": float(exposure),
            "feature_step_seconds": FEATURE_STEP_SECONDS,
            "expected_path_sha256": expected_hash,
            "source_id": source_id,
        }
        for field in (
            "category",
            "channel",
            "session_id",
            "speaker_id",
            "source_family",
            "truth",
        ):
            if field in raw:
                source[field] = raw[field]
        result.append(source)
    result.sort(key=lambda item: item["id"])
    exposure = sum(float(item["exposure_seconds"]) for item in result)
    if exposure < MIN_NEGATIVE_EXPOSURE_SECONDS:
        raise ValueError(
            f"{target_split} negative exposure is below 100 hours: {exposure / 3600:.3f}h"
        )
    return result


def build_manifest(
    frozen_manifest_path: Path,
    positive_feature_root: Path,
    target_split: str,
    positive_occurrence_manifest: Path | None = None,
) -> dict[str, Any]:
    """Build one scoring manifest for ``validation`` or untouched ``test``."""
    frozen_manifest_path = frozen_manifest_path.resolve()
    positive_feature_root = positive_feature_root.resolve()
    frozen = _load_json(frozen_manifest_path)
    negative = _negative_sources(frozen, frozen_manifest_path, target_split)
    positive, positive_hash, geometry, occurrence_hash = _positive_sources(
        positive_feature_root, target_split, positive_occurrence_manifest
    )
    sources = sorted(negative + [positive], key=lambda item: item["id"])
    return {
        "schema_version": SCORING_SCHEMA_VERSION,
        "manifest_type": "ordered-state-feature-scoring",
        "experiment": frozen.get("experiment", "ordered-state-kizz"),
        "target_split": target_split,
        "threshold_selection_split": "validation",
        "threshold_selection_allowed": target_split == "validation",
        "test_is_untouched": target_split == "test",
        "test_semantics": (
            "untouched held-out evaluation" if target_split == "test" else None
        ),
        "feature_step_seconds": FEATURE_STEP_SECONDS,
        "split_mapping": {
            "negative": target_split,
            "positive": SPLIT_MAP[target_split],
        },
        "input_hashes": {
            "frozen_negative_manifest_sha256": sha256_path(frozen_manifest_path),
            "positive_feature_root_sha256": sha256_path(positive_feature_root),
            "positive_feature_mmap_sha256": positive_hash,
        },
        "frozen_negative_manifest": str(frozen_manifest_path),
        "positive_feature_root": str(positive_feature_root),
        "positive_occurrence_geometry": geometry,
        "positive_occurrence_manifest": (
            str(positive_occurrence_manifest.resolve())
            if positive_occurrence_manifest is not None
            else None
        ),
        "positive_occurrence_manifest_sha256": occurrence_hash,
        "sources": sources,
    }


def write_manifests(
    frozen_manifest_path: Path,
    positive_feature_root: Path,
    validation_output: Path,
    test_output: Path,
    positive_occurrence_manifest: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validation = build_manifest(
        frozen_manifest_path,
        positive_feature_root,
        "validation",
        positive_occurrence_manifest,
    )
    test = build_manifest(
        frozen_manifest_path,
        positive_feature_root,
        "test",
        positive_occurrence_manifest,
    )
    for output, manifest in ((validation_output, validation), (test_output, test)):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return validation, test


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-negative-manifest", type=Path, required=True)
    parser.add_argument("--positive-feature-root", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--test-output", type=Path, required=True)
    parser.add_argument("--positive-occurrence-manifest", type=Path, required=True)
    args = parser.parse_args()
    validation, test = write_manifests(
        args.frozen_negative_manifest,
        args.positive_feature_root,
        args.validation_output,
        args.test_output,
        args.positive_occurrence_manifest,
    )
    print(
        json.dumps(
            {
                "validation_sources": len(validation["sources"]),
                "test_sources": len(test["sources"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
