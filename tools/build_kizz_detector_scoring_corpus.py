#!/usr/bin/env python3
"""Build one provenance-bound corpus for deployed Kizz detector scoring.

The aligned-feature builder writes one positive array per split and one
negative array per split/source group.  This tool reconstructs the row
identity of those arrays from the canonical-v3 provenance and its exact bound
negative manifest, then emits the single source array/manifest contract used
by ``trace_kizz_phoneme_detector`` and the candidate-conditioned verifier
builder.

No sorting is applied inside an input array.  Positive rows retain the order
of the provenance ledger after filtering by split.  Negative rows retain the
bound manifest's traversal order after filtering by split and source group,
which is the materialization rule in
``build_kizz_aligned_teacher_features_v3``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SPLITS = ("train", "validation", "test")
INPUT_SHAPE = (260, 40)
CONTEXT_DURATION_SECONDS = 41_920 / 16_000
OUTPUT_FEATURES_NAME = "source-features.npy"
OUTPUT_MANIFEST_NAME = "source-manifest.json"
IDENTITY_FIELDS = (
    "source_id",
    "parent_source_id",
    "speaker_id",
    "session_id",
    "ancestry_id",
    "voice_id",
)
IDENTITY_LIST_FIELDS = ("ancestry_ids", "parent_source_ids")
HASH_FIELDS = (
    "audio_sha256",
    "sha256",
    "source_audio_sha256",
    "parent_source_audio_sha256",
    "ancestry_sha256",
)
PRESERVED_FIELDS = (
    "provider",
    "speaker_id",
    "session_id",
    "ancestry_id",
    "ancestry_ids",
    "ancestry_sha256",
    "voice_id",
    "source_group",
    "parent_source_ids",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _feature_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(values).tobytes(order="C")
    ).hexdigest()


def _canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload).rstrip(b"\n")).hexdigest()


def _required_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    normalized = value.lower()
    if any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    return normalized


def _load_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return payload


def _rows(payload: Mapping[str, Any], path: Path) -> list[dict[str, Any]]:
    values = payload.get("examples", payload.get("records"))
    if not isinstance(values, list) or not all(
        isinstance(value, dict) for value in values
    ):
        raise TypeError(f"{path}: expected an examples or records list")
    return [dict(value) for value in values]


def _resolve_path(raw: Any, *, parent: Path, name: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{name} path is required")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = parent / candidate
    return candidate.resolve()


def _binding(path: Path, **metadata: Any) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "sha256": sha256_file(resolved), **metadata}


def _verify_binding(
    raw: Any, *, parent: Path, name: str
) -> tuple[Path, dict[str, str]]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{name} binding is required")
    path = _resolve_path(raw.get("path"), parent=parent, name=name)
    expected = _required_sha256(raw.get("sha256"), f"{name} sha256")
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"{name} binding drifted: {path}")
    return path, {"path": str(path), "sha256": expected}


def _verify_upstream_bindings(
    provenance: Mapping[str, Any], provenance_path: Path, negative_manifest: Path
) -> tuple[list[Path], dict[str, Any]]:
    parent = provenance_path.parent
    raw_positives = provenance.get("positive_manifests")
    if not isinstance(raw_positives, list) or not raw_positives:
        raise ValueError("feature provenance requires positive_manifests")
    positive_paths: list[Path] = []
    positive_bindings: list[dict[str, str]] = []
    for index, raw in enumerate(raw_positives):
        path, binding = _verify_binding(
            raw, parent=parent, name=f"positive manifest {index}"
        )
        positive_paths.append(path)
        positive_bindings.append(binding)

    declared_negative_path, declared_negative = _verify_binding(
        provenance.get("negative_manifest"),
        parent=parent,
        name="negative manifest",
    )
    if declared_negative_path != negative_manifest:
        raise ValueError("provided negative manifest is not the provenance-bound manifest")

    verified: dict[str, Any] = {
        "positive_manifests": positive_bindings,
        "negative_manifest": declared_negative,
    }
    for key in ("background_manifest", "rir_manifest"):
        raw = provenance.get(key)
        if raw is not None:
            _, binding = _verify_binding(raw, parent=parent, name=key)
            verified[key] = binding

    audit = provenance.get("source_pronunciation_audit")
    if audit is not None:
        if not isinstance(audit, Mapping):
            raise ValueError("source_pronunciation_audit must be an object")
        audit_path = _resolve_path(
            audit.get("path"), parent=parent, name="source pronunciation audit"
        )
        audit_sha = _required_sha256(
            audit.get("sha256"), "source pronunciation audit sha256"
        )
        source_path = _resolve_path(
            audit.get("source_manifest"),
            parent=parent,
            name="source pronunciation manifest",
        )
        source_sha = _required_sha256(
            audit.get("source_manifest_sha256"),
            "source pronunciation manifest sha256",
        )
        if (
            not audit_path.is_file()
            or sha256_file(audit_path) != audit_sha
            or not source_path.is_file()
            or sha256_file(source_path) != source_sha
        ):
            raise ValueError("source pronunciation provenance binding drifted")
        verified["source_pronunciation_audit"] = {
            "path": str(audit_path),
            "sha256": audit_sha,
            "source_manifest": {
                "path": str(source_path),
                "sha256": source_sha,
            },
        }
    return positive_paths, verified


def _audio_binding(
    row: Mapping[str, Any], *, manifest_path: Path, name: str
) -> tuple[Path, str]:
    path = _resolve_path(row.get("path"), parent=manifest_path.parent, name=name)
    expected = _required_sha256(
        row.get("audio_sha256", row.get("sha256")), f"{name} audio sha256"
    )
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"{name} live audio hash drifted: {path}")
    return path, expected


def _positive_sources(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = _load_object(path)
        for ordinal, row in enumerate(_rows(payload, path)):
            source_id = row.get("source_id")
            if not isinstance(source_id, str) or not source_id:
                raise ValueError(f"{path}: positive row {ordinal} lacks source_id")
            if source_id in result:
                raise ValueError(f"duplicate upstream positive source_id: {source_id}")
            if row.get("label") not in (1, True):
                raise ValueError(f"{source_id}: upstream positive label drifted")
            audio_path, audio_sha = _audio_binding(
                row, manifest_path=path, name=f"upstream positive {source_id}"
            )
            item = dict(row)
            item["path"] = str(audio_path)
            item["audio_sha256"] = audio_sha
            item["_manifest_path"] = str(path)
            result[source_id] = item
    return result


def _copy_preserved(
    primary: Mapping[str, Any], fallback: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    fallback = fallback or {}
    for key in PRESERVED_FIELDS:
        value = primary.get(key, fallback.get(key))
        if value not in (None, "", []):
            result[key] = value
    return result


def _source_duration(row: Mapping[str, Any]) -> float | None:
    value = row.get("duration_seconds")
    if value is None:
        return None
    try:
        duration = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("source duration must be finite and positive") from error
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("source duration must be finite and positive")
    return duration


def _validate_identity_match(
    ledger: Mapping[str, Any], upstream: Mapping[str, Any], source_id: str
) -> None:
    for key in ("provider", "speaker_id", "session_id", "ancestry_id", "source_group"):
        left, right = ledger.get(key), upstream.get(key)
        if left not in (None, "") and right not in (None, "") and left != right:
            raise ValueError(f"{source_id}: {key} differs from bound positive manifest")


def _positive_rows_by_split(
    provenance: Mapping[str, Any],
    provenance_path: Path,
    upstream: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    ledger = _rows(provenance, provenance_path)
    result: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    seen: set[str] = set()
    for ordinal, row in enumerate(ledger):
        source_id = row.get("source_id")
        parent_id = row.get("parent_source_id")
        split = row.get("split")
        variant = row.get("variant")
        if not isinstance(source_id, str) or not source_id or source_id in seen:
            raise ValueError("positive provenance source IDs must be unique")
        if split not in SPLITS:
            raise ValueError(f"{source_id}: unsupported positive split")
        if not isinstance(parent_id, str) or parent_id not in upstream:
            raise ValueError(f"{source_id}: parent is absent from bound positive manifests")
        if not isinstance(variant, str) or source_id != f"{parent_id}::{variant}":
            raise ValueError(f"{source_id}: variant identity drifted")
        augmentation = row.get("augmentation")
        if split != "train" and (variant != "clean" or augmentation is not None):
            raise ValueError(f"{source_id}: held-out positive is not clean-only")
        if split == "train":
            if variant == "clean" and augmentation is not None:
                raise ValueError(f"{source_id}: clean training row has augmentation")
            if variant != "clean" and (
                not variant.startswith("overlay-") or not isinstance(augmentation, Mapping)
            ):
                raise ValueError(f"{source_id}: training variant is not a bound overlay")
        upstream_row = upstream[parent_id]
        _validate_identity_match(row, upstream_row, source_id)
        audio_path, audio_sha = _audio_binding(
            row, manifest_path=provenance_path, name=f"positive waveform {source_id}"
        )
        source_audio_sha = _required_sha256(
            row.get("source_audio_sha256"), f"{source_id} source_audio_sha256"
        )
        if source_audio_sha != upstream_row["audio_sha256"]:
            raise ValueError(f"{source_id}: parent source audio binding drifted")
        output = {
            "source_id": source_id,
            "parent_source_id": parent_id,
            "split": split,
            "label": 1,
            "path": str(audio_path),
            "audio_sha256": audio_sha,
            "source_audio_sha256": source_audio_sha,
            "parent_source_audio_sha256": source_audio_sha,
            "parent_source_path": upstream_row["path"],
            "duration_seconds": CONTEXT_DURATION_SECONDS,
            "variant": variant,
            "augmentation": augmentation,
            "positive_provenance_index": ordinal,
            **_copy_preserved(row, upstream_row),
        }
        source_duration = _source_duration(upstream_row)
        if source_duration is not None:
            output["source_duration_seconds"] = source_duration
        result[str(split)].append(output)
        seen.add(source_id)

    expected_counts = provenance.get("positive_counts")
    if not isinstance(expected_counts, Mapping):
        raise ValueError("feature provenance requires positive_counts")
    observed_counts = {split: len(result[split]) for split in SPLITS}
    declared_counts = {
        split: int(expected_counts.get(split, -1)) for split in SPLITS
    }
    if observed_counts != declared_counts or any(value < 1 for value in declared_counts.values()):
        raise ValueError(
            f"positive provenance counts drifted: {observed_counts} != {declared_counts}"
        )
    return result


def _negative_rows_by_split_group(
    provenance: Mapping[str, Any],
    negative_payload: Mapping[str, Any],
    negative_path: Path,
) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, dict[str, int]]]:
    raw_counts = provenance.get("negative_counts")
    if not isinstance(raw_counts, Mapping):
        raise ValueError("feature provenance requires negative_counts")
    declared: dict[str, dict[str, int]] = {}
    allowed_groups: set[str] = set()
    for split in SPLITS:
        split_counts = raw_counts.get(split)
        if not isinstance(split_counts, Mapping) or not split_counts:
            raise ValueError(f"negative_counts requires nonempty {split}")
        declared[split] = {}
        for raw_group, raw_count in split_counts.items():
            group = str(raw_group)
            count = int(raw_count)
            if not group or count < 1:
                raise ValueError(f"invalid negative count for {split}/{group}")
            declared[split][group] = count
            allowed_groups.add(group)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for ordinal, row in enumerate(_rows(negative_payload, negative_path)):
        split = str(row.get("split", ""))
        group = str(row.get("source_group", ""))
        if split not in SPLITS or row.get("label") not in (0, False) or group not in allowed_groups:
            continue
        source_id = row.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"negative manifest row {ordinal} lacks source_id")
        if split != "train" and (
            row.get("augmentation") not in (None, {})
            or row.get("variant") not in (None, "clean")
        ):
            raise ValueError(f"{source_id}: held-out negative is not clean-only")
        audio_path, audio_sha = _audio_binding(
            row, manifest_path=negative_path, name=f"negative source {source_id}"
        )
        output = {
            "source_id": source_id,
            "split": split,
            "label": 0,
            "path": str(audio_path),
            "audio_sha256": audio_sha,
            "source_audio_sha256": audio_sha,
            "duration_seconds": CONTEXT_DURATION_SECONDS,
            "source_group": group,
            "negative_manifest_index": ordinal,
            **_copy_preserved(row),
        }
        if row.get("parent_source_id") not in (None, ""):
            output["parent_source_id"] = row["parent_source_id"]
        source_duration = _source_duration(row)
        if source_duration is not None:
            output["source_duration_seconds"] = source_duration
        grouped[(split, group)].append(output)

    observed = {
        split: {
            group: len(grouped[(split, group)])
            for group in sorted(declared[split])
        }
        for split in SPLITS
    }
    if observed != {
        split: dict(sorted(declared[split].items())) for split in SPLITS
    }:
        raise ValueError(f"negative manifest counts drifted: {observed} != {declared}")
    return grouped, declared


def _load_feature_array(path: Path, expected_count: int) -> tuple[np.ndarray, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.shape != (expected_count, *INPUT_SHAPE) or values.dtype != np.float32:
        raise ValueError(
            f"{path}: expected float32[{expected_count},260,40], got {values.dtype}{values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: feature array contains non-finite values")
    return values, {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "shape": list(values.shape),
        "dtype": str(values.dtype),
    }


def _identity_values(row: Mapping[str, Any]) -> set[str]:
    values = {
        str(row[key])
        for key in IDENTITY_FIELDS
        if row.get(key) not in (None, "")
    }
    for key in IDENTITY_LIST_FIELDS:
        raw = row.get(key, [])
        if isinstance(raw, list):
            values.update(str(value) for value in raw if value not in (None, ""))
    return values


def _hash_values(row: Mapping[str, Any]) -> set[str]:
    return {
        str(row[key])
        for key in HASH_FIELDS
        if row.get(key) not in (None, "")
    }


def _reject_leakage(
    rows: Sequence[Mapping[str, Any]], features: np.ndarray
) -> dict[str, Any]:
    if len(rows) != len(features):
        raise ValueError("leakage audit requires aligned rows and features")
    seen_source_ids: set[str] = set()
    identities: dict[str, set[str]] = defaultdict(set)
    hashes: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        source_id = str(row["source_id"])
        if source_id in seen_source_ids:
            raise ValueError(f"duplicate unified source_id: {source_id}")
        seen_source_ids.add(source_id)
        split = str(row["split"])
        identities[split].update(_identity_values(row))
        hashes[split].update(_hash_values(row))
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            identity_overlap = identities[left] & identities[right]
            if identity_overlap:
                raise ValueError(
                    f"{left}/{right} identity leakage: {sorted(identity_overlap)[:3]}"
                )
            hash_overlap = hashes[left] & hashes[right]
            if hash_overlap:
                raise ValueError(
                    f"{left}/{right} hash leakage: {sorted(hash_overlap)[:3]}"
                )

    # The frontend is intentionally lossy: sufficiently quiet independent audio
    # can collapse to the same all-zero tensor.  That is not source leakage.
    # Any exact nonzero tensor shared by held-out splits is too suspicious to
    # accept, while zero collapses remain explicit evidence in the manifest.
    feature_occurrences: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for index, row in enumerate(rows):
        feature_occurrences[str(row["feature_sha256"])].append(
            (str(row["split"]), index)
        )
    zero_collisions: list[dict[str, Any]] = []
    for feature_hash, occurrences in sorted(feature_occurrences.items()):
        splits = sorted({split for split, _ in occurrences})
        if len(splits) < 2:
            continue
        indexes = [index for _, index in occurrences]
        if any(np.any(features[index] != 0.0) for index in indexes):
            raise ValueError(
                f"nonzero feature hash collision across splits: {feature_hash}"
            )
        zero_collisions.append(
            {
                "feature_sha256": feature_hash,
                "splits": splits,
                "count": len(occurrences),
                "count_by_split": {
                    split: sum(item_split == split for item_split, _ in occurrences)
                    for split in splits
                },
                "classification": "independent_audio_frontend_zero_collapse",
            }
        )
    return {
        "source_identity_overlap_across_splits": 0,
        "source_hash_overlap_across_splits": 0,
        "nonzero_feature_hash_overlap_across_splits": 0,
        "all_zero_feature_collisions": zero_collisions,
    }


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, values, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def build_detector_scoring_corpus(
    feature_provenance: Path,
    feature_directory: Path,
    negative_manifest: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Build and atomically materialize the unified detector scoring corpus."""
    feature_provenance = feature_provenance.expanduser().resolve()
    feature_directory = feature_directory.expanduser().resolve()
    negative_manifest = negative_manifest.expanduser().resolve()
    output_directory = output_directory.expanduser().resolve()
    if not feature_directory.is_dir():
        raise NotADirectoryError(feature_directory)
    if feature_provenance.parent != feature_directory:
        raise ValueError("feature provenance must reside in its feature directory")

    provenance = _load_object(feature_provenance)
    if (
        int(provenance.get("schema_version", 0)) != 3
        or provenance.get("recipe") != "kizz_aligned_teacher_features_v3"
        or provenance.get("input_shape") != list(INPUT_SHAPE)
    ):
        raise ValueError("feature provenance is not the canonical-v3 feature contract")
    positive_paths, upstream_bindings = _verify_upstream_bindings(
        provenance, feature_provenance, negative_manifest
    )
    upstream_positives = _positive_sources(positive_paths)
    positives = _positive_rows_by_split(
        provenance, feature_provenance, upstream_positives
    )
    negative_payload = _load_object(negative_manifest)
    negatives, negative_counts = _negative_rows_by_split_group(
        provenance, negative_payload, negative_manifest
    )

    expected_feature_names = {
        f"positive_features-{split}.npy" for split in SPLITS
    }
    expected_feature_names.update(
        f"negative-{split}-{group}.npy"
        for split in SPLITS
        for group in negative_counts[split]
    )
    observed_feature_names = {
        path.name
        for path in feature_directory.glob("*.npy")
        if path.name.startswith("positive_features-") or path.name.startswith("negative-")
    }
    if observed_feature_names != expected_feature_names:
        raise ValueError(
            "feature directory has missing or stale scoring arrays: "
            f"expected {sorted(expected_feature_names)}, observed {sorted(observed_feature_names)}"
        )

    unified_rows: list[dict[str, Any]] = []
    feature_chunks: list[np.ndarray] = []
    input_arrays: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    output_index = 0

    def append_block(
        *,
        split: str,
        label: int,
        source_group: str | None,
        source_rows: list[dict[str, Any]],
        filename: str,
    ) -> None:
        nonlocal output_index
        path = feature_directory / filename
        values, array_binding = _load_feature_array(path, len(source_rows))
        binding_index = len(input_arrays)
        input_arrays.append(
            {
                **array_binding,
                "split": split,
                "label": label,
                **({"source_group": source_group} if source_group is not None else {}),
            }
        )
        start = output_index
        copied = np.asarray(values, dtype=np.float32).copy()
        for source_index, (row, feature_row) in enumerate(zip(source_rows, copied)):
            feature_hash = _feature_sha256(feature_row)
            item = dict(row)
            item.update(
                {
                    "feature_index": output_index,
                    "feature_sha256": feature_hash,
                    "source_feature": {
                        "input_array_binding_index": binding_index,
                        "filename": filename,
                        "array_sha256": array_binding["sha256"],
                        "index": source_index,
                        "feature_sha256": feature_hash,
                    },
                }
            )
            unified_rows.append(item)
            output_index += 1
        feature_chunks.append(copied)
        blocks.append(
            {
                "split": split,
                "label": label,
                **({"source_group": source_group} if source_group is not None else {}),
                "input_array_binding_index": binding_index,
                "input_filename": filename,
                "output_start_index": start,
                "count": len(source_rows),
            }
        )

    for split in SPLITS:
        append_block(
            split=split,
            label=1,
            source_group=None,
            source_rows=positives[split],
            filename=f"positive_features-{split}.npy",
        )
        for group in sorted(negative_counts[split]):
            append_block(
                split=split,
                label=0,
                source_group=group,
                source_rows=negatives[(split, group)],
                filename=f"negative-{split}-{group}.npy",
            )

    unified = np.concatenate(feature_chunks, axis=0).astype(np.float32, copy=False)
    if unified.shape != (len(unified_rows), *INPUT_SHAPE) or not np.all(np.isfinite(unified)):
        raise ValueError("unified feature array failed final shape/finite validation")
    leakage_audit = _reject_leakage(unified_rows, unified)

    output_features = output_directory / OUTPUT_FEATURES_NAME
    output_manifest = output_directory / OUTPUT_MANIFEST_NAME
    _atomic_npy(output_features, unified)
    output_features_sha = sha256_file(output_features)
    report: dict[str, Any] = {
        "schema_version": 1,
        "recipe": "kizz_detector_scoring_corpus_v1",
        "deployment_qualification": False,
        "input_shape": list(INPUT_SHAPE),
        "context_duration_seconds": CONTEXT_DURATION_SECONDS,
        "ordering": {
            "split_order": list(SPLITS),
            "within_split": "positives_then_lexicographic_negative_source_group",
            "positive_rows": "feature_provenance_examples_traversal_filtered_by_split",
            "negative_rows": "bound_negative_manifest_traversal_filtered_by_split_and_source_group",
            "blocks": blocks,
        },
        "inputs": {
            "feature_provenance": _binding(feature_provenance),
            "feature_arrays": input_arrays,
            "upstream": upstream_bindings,
        },
        "outputs": {
            "source_features": {
                "filename": OUTPUT_FEATURES_NAME,
                "sha256": output_features_sha,
                "shape": list(unified.shape),
                "dtype": str(unified.dtype),
            }
        },
        "array_sha256": {OUTPUT_FEATURES_NAME: output_features_sha},
        "leakage_audit": leakage_audit,
        "counts": {
            "total": len(unified_rows),
            "positive": sum(int(row["label"]) == 1 for row in unified_rows),
            "negative": sum(int(row["label"]) == 0 for row in unified_rows),
            "by_split": {
                split: {
                    "total": sum(row["split"] == split for row in unified_rows),
                    "positive": len(positives[split]),
                    "negative": sum(negative_counts[split].values()),
                    "negative_by_source_group": dict(
                        sorted(negative_counts[split].items())
                    ),
                }
                for split in SPLITS
            },
        },
        "examples": unified_rows,
    }
    report["manifest_payload_sha256"] = _canonical_hash(report)
    _atomic_json(output_manifest, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-provenance", type=Path, required=True)
    parser.add_argument("--feature-directory", type=Path, required=True)
    parser.add_argument("--negative-manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_detector_scoring_corpus(
        args.feature_provenance,
        args.feature_directory,
        args.negative_manifest,
        args.output_directory,
    )
    manifest_path = args.output_directory.expanduser().resolve() / OUTPUT_MANIFEST_NAME
    print(
        json.dumps(
            {
                "counts": report["counts"],
                "source_features_sha256": report["array_sha256"][OUTPUT_FEATURES_NAME],
                "source_manifest_sha256": sha256_file(manifest_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
