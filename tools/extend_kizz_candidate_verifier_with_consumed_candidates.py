#!/usr/bin/env python3
"""Append consumed positives and auxiliary negatives to a verified verifier corpus.

The immutable base remains the complete validation/test authority.  Every row
imported by this utility is retagged as training evidence, and auxiliary
negatives may come from an older detector-conditioned distribution.  Source
corpora are integrity-checked without requiring every split/class combination,
which permits positive-only consumed-evidence corpora.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.train_kizz_candidate_verifier import (
    INPUT_SHAPE,
    SPLITS,
    _atomic_bytes,
    _atomic_npy,
    _canonical_bytes,
    _hash_values,
    _identity_values,
    _require_sha256,
    _verify_hard_negative_policy,
    _verify_split_disjointness,
    _verify_transitive_bindings,
    load_verified_dataset,
    sha256_file,
)


ARRAY_NAMES = (
    "features.npy",
    "labels.npy",
    "detector_scores.npy",
    "detector_feature_frames.npy",
    "detector_score_frames.npy",
)
ROLES = ("consumed_positive", "auxiliary_negative")


@dataclass(frozen=True)
class SourceDataset:
    root: Path
    corpus_path: Path
    corpus_sha256: str
    corpus: dict[str, Any]
    rows: tuple[dict[str, Any], ...]
    arrays: dict[str, np.ndarray]


def _binding(path: Path, *, relative_path: str | None = None) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": relative_path if relative_path is not None else str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _absolute_bindings(value: Any, relative_to: Path) -> Any:
    """Relocate copied binding trees so relative source paths stay verifiable."""
    if isinstance(value, Mapping):
        copied = {
            str(key): _absolute_bindings(child, relative_to)
            for key, child in value.items()
        }
        if "path" in copied and "sha256" in copied:
            path = Path(str(copied["path"])).expanduser()
            if not path.is_absolute():
                path = relative_to / path
            copied["path"] = str(path.resolve())
        return copied
    if isinstance(value, list):
        return [_absolute_bindings(child, relative_to) for child in value]
    return copy.deepcopy(value)


def _load_arrays(root: Path, hashes: Mapping[str, Any], row_count: int) -> dict[str, np.ndarray]:
    missing = set(ARRAY_NAMES) - set(hashes)
    if missing:
        raise ValueError(f"{root}: source lacks required arrays: {sorted(missing)}")
    arrays: dict[str, np.ndarray] = {}
    for name in ARRAY_NAMES:
        expected = _require_sha256(hashes.get(name), f"{root}/{name} hash")
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(
                f"{root}: {name} hash drift: expected {expected}, got {observed}"
            )
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.shape[0] != row_count:
            raise ValueError(f"{root}: {name} does not contain one value per row")
        arrays[name] = values
    if arrays["features.npy"].shape != (row_count, *INPUT_SHAPE):
        raise ValueError(f"{root}: features.npy has the wrong candidate-window shape")
    for name in ARRAY_NAMES[1:]:
        if arrays[name].shape != (row_count,):
            raise ValueError(f"{root}: {name} must be one-dimensional")
    return arrays


def _load_role_source(root: Path, expected_corpus_sha256: str) -> SourceDataset:
    """Verify a role-specific source without requiring a trainable class matrix."""
    root = root.expanduser().resolve()
    corpus_path = root / "corpus.json"
    expected = _require_sha256(expected_corpus_sha256, "source corpus expected hash")
    observed = sha256_file(corpus_path)
    if observed != expected:
        raise ValueError(
            f"{root}: corpus hash drift: expected {expected}, got {observed}"
        )
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    if not isinstance(corpus, dict) or corpus.get("schema_version") != 1:
        raise ValueError(f"{root}: unsupported candidate-verifier corpus schema")
    if corpus.get("recipe") != "kizz_control_candidate_conditioned_verifier_v1":
        raise ValueError(f"{root}: corpus is not the candidate-verifier recipe")
    if corpus.get("candidate_condition") != "frozen_detector_trigger_only":
        raise ValueError(f"{root}: corpus is not conditioned on frozen detector triggers")
    _verify_transitive_bindings(
        {"bindings": corpus.get("bindings"), "detector": corpus.get("detector")},
        relative_to=root,
    )
    rows = corpus.get("examples")
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{root}: corpus requires nonempty examples")
    arrays = _load_arrays(root, corpus.get("array_sha256", {}), len(rows))
    if not np.all(np.isfinite(arrays["features.npy"])):
        raise ValueError(f"{root}: features must be finite")
    if not np.all(np.isfinite(arrays["detector_scores.npy"])):
        raise ValueError(f"{root}: detector scores must be finite")

    seen: set[str] = set()
    for index, row in enumerate(rows):
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in seen:
            raise ValueError(f"{root}: candidate IDs must be unique nonempty strings")
        seen.add(candidate_id)
        if row.get("detector_conditioned") is not True:
            raise ValueError(f"{root}/{candidate_id}: detector_conditioned must be true")
        if row.get("split") not in SPLITS or row.get("label") not in (0, 1, False, True):
            raise ValueError(f"{root}/{candidate_id}: invalid split or label")
        if row.get("feature_index") != index:
            raise ValueError(f"{root}/{candidate_id}: feature_index/order drift")
        if int(arrays["labels.npy"][index]) != int(row["label"]):
            raise ValueError(f"{root}/{candidate_id}: labels.npy differs from corpus")
        if not math.isclose(
            float(arrays["detector_scores.npy"][index]),
            float(row.get("detector_score")),
            rel_tol=1e-6,
            abs_tol=1e-7,
        ):
            raise ValueError(f"{root}/{candidate_id}: detector_scores.npy differs from corpus")
        feature_hash = _require_sha256(
            row.get("candidate_feature_sha256"), f"{candidate_id} feature hash"
        )
        actual_feature_hash = hashlib.sha256(
            np.ascontiguousarray(arrays["features.npy"][index]).tobytes()
        ).hexdigest()
        if feature_hash != actual_feature_hash:
            raise ValueError(f"{root}/{candidate_id}: candidate feature hash drift")
    _verify_split_disjointness(rows)
    _verify_hard_negative_policy(corpus, rows)
    return SourceDataset(
        root=root,
        corpus_path=corpus_path,
        corpus_sha256=observed,
        corpus=corpus,
        rows=tuple(copy.deepcopy(row) for row in rows),
        arrays=arrays,
    )


def _detector_identity(value: Any) -> Any:
    """Canonical detector identity: content bindings and execution semantics, not paths."""
    if isinstance(value, Mapping):
        return {
            str(key): _detector_identity(child)
            for key, child in sorted(value.items())
            if key not in {"path", "bytes"}
        }
    if isinstance(value, list):
        return [_detector_identity(child) for child in value]
    return value


def _detector_sha256(corpus: Mapping[str, Any]) -> str:
    detector = corpus.get("detector")
    if not isinstance(detector, Mapping) or not detector:
        raise ValueError("candidate corpus requires detector identity")
    return hashlib.sha256(_canonical_bytes(_detector_identity(detector))).hexdigest()


def _source_group(row: Mapping[str, Any], group_by: str) -> str:
    key = row.get("parent_source_id") if group_by == "source" else row.get("session_id")
    if key in (None, ""):
        raise ValueError(
            f"{row.get('candidate_id')}: train negative lacks {group_by} identity"
        )
    return str(key)


def _source_key(row: Mapping[str, Any], source_sha256: str) -> tuple[str, str]:
    identity = (
        row.get("parent_source_id")
        or row.get("source_parent_source_id")
        or row.get("source_id")
        or row.get("candidate_id")
    )
    return source_sha256, str(identity)


def _duration(row: Mapping[str, Any]) -> float:
    value = row.get("duration_seconds", 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    result = float(value)
    return result if math.isfinite(result) and result >= 0.0 else 0.0


def _rate(count: int, exposure: float, multiplier: float = 1.0) -> float:
    return count * multiplier / exposure if exposure else 0.0


def _input_binding(source: SourceDataset, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "corpus": _binding(source.corpus_path),
        "arrays": {
            name: _binding(source.root / name) for name in ARRAY_NAMES
        },
        "detector": _absolute_bindings(source.corpus.get("detector"), source.root),
        "bindings": _absolute_bindings(source.corpus.get("bindings"), source.root),
    }


def extend(
    base_dataset: Path,
    base_corpus_sha256: str,
    consumed_positive_sources: Sequence[tuple[Path, str]],
    auxiliary_negative_sources: Sequence[tuple[Path, str]],
    output: Path,
    *,
    consumed_provider: str = "consumed_stackchan_physical",
) -> dict[str, Any]:
    """Create a deterministic train-only extension of an immutable base corpus."""
    if not consumed_positive_sources and not auxiliary_negative_sources:
        raise ValueError("at least one consumed-positive or auxiliary-negative source is required")
    if not isinstance(consumed_provider, str) or not consumed_provider.strip():
        raise ValueError("consumed_provider must be a nonempty string")
    consumed_provider = consumed_provider.strip()
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")

    base = load_verified_dataset(
        base_dataset, expected_corpus_sha256=base_corpus_sha256
    )
    base_hashes = set(base.corpus.get("array_sha256", {}))
    missing_base_arrays = set(ARRAY_NAMES) - base_hashes
    if missing_base_arrays:
        raise ValueError(f"base lacks required arrays: {sorted(missing_base_arrays)}")
    base_arrays = {
        name: np.load(base.root / name, mmap_mode="r", allow_pickle=False)
        for name in ARRAY_NAMES
    }

    specifications = [
        (role, Path(path), digest)
        for role, sources in (
            ("consumed_positive", consumed_positive_sources),
            ("auxiliary_negative", auxiliary_negative_sources),
        )
        for path, digest in sources
    ]
    loaded = [
        (role, _load_role_source(path, digest))
        for role, path, digest in specifications
    ]
    loaded.sort(key=lambda item: (item[1].corpus_sha256, item[0], str(item[1].root)))
    source_hashes = [source.corpus_sha256 for _, source in loaded]
    if len(source_hashes) != len(set(source_hashes)):
        raise ValueError("a source corpus may be supplied only once across all roles")

    base_detector_sha = _detector_sha256(base.corpus)
    source_relations: dict[str, str] = {}
    for role, source in loaded:
        source_detector_sha = _detector_sha256(source.corpus)
        relation = "exact_match" if source_detector_sha == base_detector_sha else "mismatch"
        source_relations[source.corpus_sha256] = relation
        if role == "consumed_positive" and relation != "exact_match":
            raise ValueError(
                f"{source.root}: consumed_positive detector mismatch versus immutable base"
            )

    heldout_rows = [row for row in base.rows if row["split"] in {"validation", "test"}]
    heldout_ids = set().union(*(_identity_values(row) for row in heldout_rows))
    heldout_hashes = set().union(*(_hash_values(row) for row in heldout_rows))
    heldout_candidate_ids = {str(row["candidate_id"]) for row in heldout_rows}

    rows = [copy.deepcopy(row) for row in base.rows]
    array_parts: dict[str, list[np.ndarray]] = {
        name: [np.asarray(base_arrays[name])] for name in ARRAY_NAMES
    }
    known_candidate_ids = {str(row["candidate_id"]) for row in rows}
    known_feature_hashes = {
        str(row["candidate_feature_sha256"]) for row in rows
    }
    selected: list[tuple[str, SourceDataset, int, dict[str, Any]]] = []
    skipped: list[dict[str, str]] = []

    for role, source in loaded:
        for index, original in enumerate(source.rows):
            eligible = (
                int(original["label"]) == 1
                if role == "consumed_positive"
                else original["split"] == "train" and int(original["label"]) == 0
            )
            if not eligible:
                continue
            candidate_id = str(original["candidate_id"])
            identities = _identity_values(original)
            hashes = _hash_values(original)
            if candidate_id in heldout_candidate_ids or identities & heldout_ids:
                raise ValueError(
                    f"{source.root}/{candidate_id}: source identity overlaps base validation/test"
                )
            if hashes & heldout_hashes:
                raise ValueError(
                    f"{source.root}/{candidate_id}: source hash overlaps base validation/test"
                )
            feature_hash = str(original["candidate_feature_sha256"])
            if candidate_id in known_candidate_ids:
                skipped.append(
                    {"role": role, "candidate_id": candidate_id, "reason": "duplicate_candidate_id"}
                )
                continue
            if feature_hash in known_feature_hashes:
                skipped.append(
                    {"role": role, "candidate_id": candidate_id, "reason": "duplicate_feature_hash"}
                )
                continue

            row = copy.deepcopy(original)
            row["split"] = "train"
            row["original_split"] = str(original["split"])
            row["evidence_role"] = role
            row["source_candidate_corpus_sha256"] = source.corpus_sha256
            row["source_candidate_binding"] = {
                "path": str(source.corpus_path),
                "sha256": source.corpus_sha256,
            }
            row["source_detector_relation"] = source_relations[source.corpus_sha256]
            if role == "consumed_positive":
                row["consumed_evidence"] = True
                row["detector_mismatch_permitted"] = False
                row["original_provider"] = original.get("provider")
                row["provider"] = consumed_provider
            else:
                row["candidate_distribution_role"] = "older_detector_auxiliary_negative"
                row["detector_mismatch_permitted"] = True
            known_candidate_ids.add(candidate_id)
            known_feature_hashes.add(feature_hash)
            selected.append((role, source, index, row))

    if not selected:
        raise ValueError("no eligible unique candidates remained after role selection and dedupe")
    for role, source, index, row in selected:
        rows.append(row)
        for name in ARRAY_NAMES:
            array_parts[name].append(np.asarray(source.arrays[name][index : index + 1]))
    for index, row in enumerate(rows):
        row["feature_index"] = index

    arrays: dict[str, np.ndarray] = {}
    for name, parts in array_parts.items():
        base_array = base_arrays[name]
        for part in parts[1:]:
            if part.dtype != base_array.dtype or part.shape[1:] != base_array.shape[1:]:
                raise ValueError(f"source {name} dtype/shape differs from immutable base")
        arrays[name] = np.concatenate(parts, axis=0)
        if not np.array_equal(arrays[name][: len(base.rows)], base_array):
            raise AssertionError(f"immutable base {name} prefix changed")

    _verify_split_disjointness(rows)
    corpus = copy.deepcopy(base.corpus)
    corpus["detector"] = _absolute_bindings(base.corpus.get("detector"), base.root)
    corpus["bindings"] = _absolute_bindings(base.corpus.get("bindings"), base.root)
    corpus["examples"] = rows
    labels = arrays["labels.npy"]
    corpus["counts"]["selected_candidates"] = len(rows)
    corpus["counts"]["selected_positives"] = int(np.sum(labels == 1))
    corpus["counts"]["selected_negatives"] = int(np.sum(labels == 0))

    appended_counts = Counter((role, int(row["label"])) for role, _, _, row in selected)
    train_counts = copy.deepcopy(corpus["counts"]["by_split"]["train"])
    train_rows = [row for row in rows if row["split"] == "train"]
    train_counts["selected_positive_candidates"] = sum(int(row["label"]) == 1 for row in train_rows)
    train_counts["selected_negative_candidates"] = sum(int(row["label"]) == 0 for row in train_rows)
    added_positive = appended_counts[("consumed_positive", 1)]
    added_negative = appended_counts[("auxiliary_negative", 0)]
    train_counts["raw_detector_candidates"] = int(train_counts["raw_detector_candidates"]) + added_positive + added_negative
    train_counts["raw_positive_candidates"] = int(train_counts["raw_positive_candidates"]) + added_positive
    train_counts["raw_negative_candidates"] = int(train_counts["raw_negative_candidates"]) + added_negative

    source_durations: dict[tuple[str, str], float] = {}
    negative_source_durations: dict[tuple[str, str], float] = {}
    for role, source, _, row in selected:
        key = _source_key(row, source.corpus_sha256)
        source_durations[key] = max(source_durations.get(key, 0.0), _duration(row))
        if role == "auxiliary_negative":
            negative_source_durations[key] = max(
                negative_source_durations.get(key, 0.0), _duration(row)
            )
    train_counts["source_examples"] = int(train_counts["source_examples"]) + len(source_durations)
    train_counts["exposure_seconds"] = float(train_counts["exposure_seconds"]) + sum(source_durations.values())
    train_counts["negative_exposure_seconds"] = float(train_counts["negative_exposure_seconds"]) + sum(negative_source_durations.values())
    positive_raw = int(train_counts["raw_positive_candidates"])
    positive_misses = int(train_counts.get("detector_missed_positives", 0))
    train_counts["detector_positive_source_recall"] = (
        positive_raw / (positive_raw + positive_misses)
        if positive_raw + positive_misses
        else None
    )
    exposure = float(train_counts["exposure_seconds"])
    negative_exposure = float(train_counts["negative_exposure_seconds"])
    train_counts["raw_candidate_rate_per_second"] = _rate(int(train_counts["raw_detector_candidates"]), exposure)
    train_counts["raw_candidate_rate_per_hour"] = _rate(int(train_counts["raw_detector_candidates"]), exposure, 3600.0)
    train_counts["raw_negative_candidate_rate_per_hour"] = _rate(int(train_counts["raw_negative_candidates"]), negative_exposure, 3600.0)
    corpus["counts"]["by_split"]["train"] = train_counts

    policy = corpus["hard_negative_selection"]
    original_top_k = int(policy["top_k"])
    group_by = str(policy["group_by"])
    train_negatives = [
        row for row in rows if row["split"] == "train" and int(row["label"]) == 0
    ]
    group_counts = Counter(_source_group(row, group_by) for row in train_negatives)
    actual_top_k = max(group_counts.values(), default=1)
    policy["top_k"] = max(original_top_k, actual_top_k)
    policy["raw_training_count"] = int(policy["raw_training_count"]) + added_negative
    policy["selected_training_count"] = len(train_negatives)
    policy["heldout_candidates_unfiltered"] = sum(
        row["split"] in {"validation", "test"} and int(row["label"]) == 0
        for row in rows
    )

    input_bindings = [_input_binding(source, role) for role, source in loaded]
    provenance = {
        "schema_version": 1,
        "utility": "extend_kizz_candidate_verifier_with_consumed_candidates",
        "ordering": "source_corpus_sha256_then_role_then_resolved_path_then_source_row_order",
        "immutable_base": {
            "corpus": _binding(base.corpus_path),
            "rows_preserved_as_prefix": len(base.rows),
            "all_required_arrays_preserved_as_prefix": True,
            "validation_test_rows_unchanged": True,
        },
        "source_inputs": input_bindings,
        "detector_identity": {
            "base_sha256": base_detector_sha,
            "consumed_positive_requires_exact_match": True,
            "auxiliary_negative_mismatch_permitted": True,
            "relations_by_source_corpus_sha256": source_relations,
        },
        "selection": {
            "roles": {
                "consumed_positive": "label=1 from any original split; retagged train",
                "auxiliary_negative": "original split=train and label=0 only; retagged train",
            },
            "consumed_positive_provider": consumed_provider,
            "selected": {
                "total": len(selected),
                "consumed_positive": added_positive,
                "auxiliary_negative": added_negative,
            },
            "deduplicated": skipped,
            "heldout_overlap_policy": "reject_before_deduplication",
        },
        "hard_negative_selection": {
            "group_by": group_by,
            "original_top_k": original_top_k,
            "actual_maximum_group_count": actual_top_k,
            "output_top_k": int(policy["top_k"]),
            "update_policy": "raise_to_actual_group_maximum_without_removing_immutable_base_rows",
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for name in ARRAY_NAMES:
            _atomic_npy(temporary / name, arrays[name])
        _atomic_bytes(temporary / "provenance.json", _canonical_bytes(provenance))
        corpus.setdefault("bindings", {})["consumed_candidate_extension"] = {
            "immutable_base_candidate_corpus": _binding(base.corpus_path),
            "source_candidate_datasets": input_bindings,
            "provenance": _binding(
                temporary / "provenance.json", relative_path="provenance.json"
            ),
        }
        corpus["consumed_candidate_extension"] = {
            "schema_version": 1,
            "appended_rows": len(selected),
            "base_rows": len(base.rows),
            "roles": dict(provenance["selection"]["selected"]),
            "base_detector_identity_sha256": base_detector_sha,
        }
        corpus["array_sha256"] = {
            name: sha256_file(temporary / name) for name in ARRAY_NAMES
        }
        _verify_hard_negative_policy(corpus, rows)
        _atomic_bytes(temporary / "corpus.json", _canonical_bytes(corpus))
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {output}")
        os.rename(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    result_hash = sha256_file(output / "corpus.json")
    load_verified_dataset(output, expected_corpus_sha256=result_hash)
    return {
        "output": str(output),
        "corpus_sha256": result_hash,
        "provenance_sha256": sha256_file(output / "provenance.json"),
        "base_rows": len(base.rows),
        "appended_rows": len(selected),
        "consumed_positives": added_positive,
        "auxiliary_negatives": added_negative,
        "deduplicated": len(skipped),
        "rows": len(rows),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--base-corpus-sha256", required=True)
    parser.add_argument(
        "--consumed-positive",
        nargs=2,
        action="append",
        default=[],
        metavar=("DATASET", "CORPUS_SHA256"),
        help="verified consumed-positive candidate dataset and exact corpus hash; repeatable",
    )
    parser.add_argument(
        "--auxiliary-negative",
        nargs=2,
        action="append",
        default=[],
        metavar=("DATASET", "CORPUS_SHA256"),
        help="verified train-negative dataset and exact corpus hash; repeatable",
    )
    parser.add_argument(
        "--consumed-provider",
        default="consumed_stackchan_physical",
        help="positive sampling-provider domain assigned to consumed evidence",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = extend(
        args.base_dataset,
        args.base_corpus_sha256,
        [(Path(path), digest) for path, digest in args.consumed_positive],
        [(Path(path), digest) for path, digest in args.auxiliary_negative],
        args.output,
        consumed_provider=args.consumed_provider,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
