#!/usr/bin/env python3
"""Deterministically merge candidate-verifier extensions of one immutable base."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.train_kizz_candidate_verifier import (
    SPLITS,
    _atomic_bytes,
    _atomic_npy,
    _canonical_bytes,
    _verify_hard_negative_policy,
    _verify_split_disjointness,
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
DELTA_COUNT_FIELDS = (
    "source_examples",
    "exposure_seconds",
    "raw_detector_candidates",
    "raw_positive_candidates",
    "raw_negative_candidates",
    "negative_exposure_seconds",
)


def _binding(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _absolute_bindings(value: Any, relative_to: Path) -> Any:
    """Copy binding trees while making every bound path relocation-safe."""
    if isinstance(value, Mapping):
        copied = {key: _absolute_bindings(child, relative_to) for key, child in value.items()}
        if "path" in copied and "sha256" in copied:
            path = Path(str(copied["path"])).expanduser()
            if not path.is_absolute():
                path = relative_to / path
            copied["path"] = str(path.resolve())
        return copied
    if isinstance(value, list):
        return [_absolute_bindings(child, relative_to) for child in value]
    return copy.deepcopy(value)


def _load_arrays(dataset: Any) -> dict[str, np.ndarray]:
    missing = set(ARRAY_NAMES) - set(dataset.array_bindings)
    if missing:
        raise ValueError(
            f"{dataset.root}: extension merger requires all five arrays; "
            f"missing {sorted(missing)}"
        )
    arrays = {
        name: np.load(dataset.root / name, mmap_mode="r", allow_pickle=False)
        for name in ARRAY_NAMES
    }
    row_count = len(dataset.rows)
    if arrays["features.npy"].shape != (row_count, 260, 40):
        raise ValueError(f"{dataset.root}: features.npy shape differs from corpus rows")
    for name in ARRAY_NAMES[1:]:
        if arrays[name].shape != (row_count,):
            raise ValueError(f"{dataset.root}: {name} must contain one value per corpus row")
    return arrays


def _require_base_binding(extension: Any, base: Any) -> None:
    raw = extension.corpus.get("bindings", {}).get("base_candidate_corpus")
    if not isinstance(raw, Mapping):
        raise ValueError(f"{extension.root}: extension lacks base_candidate_corpus binding")
    declared_path = Path(str(raw.get("path", ""))).expanduser()
    if not declared_path.is_absolute():
        declared_path = extension.root / declared_path
    try:
        resolved_path = declared_path.resolve()
    except OSError as error:
        raise ValueError(f"{extension.root}: invalid base corpus binding") from error
    if resolved_path != base.corpus_path or raw.get("sha256") != base.corpus_sha256:
        raise ValueError(
            f"{extension.root}: extension is not bound to the requested immutable base"
        )


def _require_exact_base_prefix(
    base: Any,
    base_arrays: Mapping[str, np.ndarray],
    extension: Any,
    extension_arrays: Mapping[str, np.ndarray],
) -> None:
    base_count = len(base.rows)
    if len(extension.rows) <= base_count:
        raise ValueError(f"{extension.root}: extension has no rows beyond the base")
    if list(extension.rows[:base_count]) != list(base.rows):
        raise ValueError(f"{extension.root}: inherited base rows differ from base")
    for name in ARRAY_NAMES:
        inherited = extension_arrays[name]
        expected = base_arrays[name]
        if (
            inherited.dtype != expected.dtype
            or inherited.shape[0] < base_count
            or inherited.shape[1:] != expected.shape[1:]
            or not np.array_equal(inherited[:base_count], expected)
        ):
            raise ValueError(f"{extension.root}: inherited {name} differs from base")


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return number


def _integer(value: object, label: str) -> int:
    number = _finite_number(value, label)
    if not number.is_integer():
        raise ValueError(f"{label} must be an integer")
    return int(number)


def _split_counts(corpus: Mapping[str, Any], split: str, label: str) -> Mapping[str, Any]:
    raw = corpus.get("counts", {}).get("by_split", {}).get(split)
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} lacks counts.by_split.{split}")
    return raw


def _extension_deltas(base: Any, extension: Any) -> dict[str, dict[str, float | int]]:
    base_count = len(base.rows)
    tail = extension.rows[base_count:]
    if any(int(row["label"]) != 0 for row in tail):
        raise ValueError(f"{extension.root}: extension tail must contain only negatives")
    tail_counts = Counter(str(row["split"]) for row in tail)
    result: dict[str, dict[str, float | int]] = {}
    for split in SPLITS:
        base_split = _split_counts(base.corpus, split, "base corpus")
        extension_split = _split_counts(
            extension.corpus, split, f"extension {extension.root}"
        )
        delta: dict[str, float | int] = {}
        for field in DELTA_COUNT_FIELDS:
            label = f"{extension.root}: counts.by_split.{split}.{field}"
            observed = _finite_number(extension_split.get(field), label)
            original = _finite_number(
                base_split.get(field), f"base counts.by_split.{split}.{field}"
            )
            difference = observed - original
            if difference < -1e-9:
                raise ValueError(f"{label} regressed below the immutable base")
            if field in {
                "source_examples",
                "raw_detector_candidates",
                "raw_positive_candidates",
                "raw_negative_candidates",
            }:
                if not math.isclose(difference, round(difference), abs_tol=1e-9):
                    raise ValueError(f"{label} delta must be integral")
                delta[field] = int(round(difference))
            else:
                delta[field] = float(difference)

        expected = tail_counts[split]
        if delta["raw_detector_candidates"] != expected:
            raise ValueError(
                f"{extension.root}: {split} raw detector delta differs from extension tail"
            )
        if delta["raw_negative_candidates"] != expected:
            raise ValueError(
                f"{extension.root}: {split} raw negative delta differs from extension tail"
            )
        if delta["raw_positive_candidates"] != 0:
            raise ValueError(f"{extension.root}: extension adds raw positive candidates")
        selected_positive = _integer(
            extension_split.get("selected_positive_candidates"),
            f"{extension.root}: {split} selected positives",
        ) - _integer(
            base_split.get("selected_positive_candidates"),
            f"base: {split} selected positives",
        )
        selected_negative = _integer(
            extension_split.get("selected_negative_candidates"),
            f"{extension.root}: {split} selected negatives",
        ) - _integer(
            base_split.get("selected_negative_candidates"),
            f"base: {split} selected negatives",
        )
        if selected_positive != 0 or selected_negative != expected:
            raise ValueError(
                f"{extension.root}: {split} selected-count delta differs from extension tail"
            )
        delta["selected_positive_candidates"] = 0
        delta["selected_negative_candidates"] = expected
        result[split] = delta

    policy = extension.corpus.get("hard_negative_selection")
    base_policy = base.corpus.get("hard_negative_selection")
    if not isinstance(policy, Mapping) or not isinstance(base_policy, Mapping):
        raise ValueError("candidate corpora require hard_negative_selection")
    train_delta = _integer(
        policy.get("raw_training_count"),
        f"{extension.root}: raw training count",
    ) - _integer(base_policy.get("raw_training_count"), "base raw training count")
    if train_delta != tail_counts["train"]:
        raise ValueError(
            f"{extension.root}: raw training hard-negative delta differs from tail"
        )
    return result


def _rate(numerator: int, exposure: float, multiplier: float = 1.0) -> float:
    return numerator * multiplier / exposure if exposure else 0.0


def _transitive_input(extension: Any) -> dict[str, Any]:
    return {
        "corpus": _binding(extension.corpus_path),
        "arrays": {
            name: _binding(extension.root / name) for name in ARRAY_NAMES
        },
        "bindings": _absolute_bindings(
            extension.corpus.get("bindings"), extension.root
        ),
        "detector": _absolute_bindings(
            extension.corpus.get("detector"), extension.root
        ),
    }


def merge_extensions(
    base_dataset: Path,
    base_corpus_sha256: str,
    extensions: Sequence[tuple[Path, str]],
    output: Path,
    *,
    deduplicate_feature_hashes: bool = False,
) -> dict[str, Any]:
    """Merge verified extension tails in corpus-hash order."""
    if len(extensions) < 2:
        raise ValueError("at least two candidate-verifier extensions are required")
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")

    base = load_verified_dataset(
        base_dataset, expected_corpus_sha256=base_corpus_sha256
    )
    base_arrays = _load_arrays(base)
    loaded = [
        load_verified_dataset(path, expected_corpus_sha256=corpus_sha256)
        for path, corpus_sha256 in extensions
    ]
    loaded.sort(key=lambda item: (item.corpus_sha256, str(item.root)))
    if len({item.corpus_sha256 for item in loaded}) != len(loaded):
        raise ValueError("duplicate extension corpus hashes are not allowed")

    extension_arrays: dict[Path, dict[str, np.ndarray]] = {}
    extension_deltas: dict[Path, dict[str, dict[str, float | int]]] = {}
    for extension in loaded:
        _require_base_binding(extension, base)
        arrays = _load_arrays(extension)
        _require_exact_base_prefix(base, base_arrays, extension, arrays)
        extension_arrays[extension.root] = arrays
        extension_deltas[extension.root] = _extension_deltas(base, extension)

    rows = [copy.deepcopy(row) for row in base.rows]
    array_parts: dict[str, list[np.ndarray]] = {
        name: [np.asarray(base_arrays[name])] for name in ARRAY_NAMES
    }
    base_count = len(base.rows)
    seen_feature_hashes = {
        str(row.get("candidate_feature_sha256", "")) for row in rows
    }
    duplicate_feature_hashes_skipped = 0
    for extension in loaded:
        selected_indexes: list[int] = []
        for index, raw in enumerate(extension.rows[base_count:], start=base_count):
            feature_hash = str(raw.get("candidate_feature_sha256", ""))
            if deduplicate_feature_hashes and feature_hash in seen_feature_hashes:
                duplicate_feature_hashes_skipped += 1
                continue
            seen_feature_hashes.add(feature_hash)
            selected_indexes.append(index)
            rows.append(copy.deepcopy(raw))
        for name in ARRAY_NAMES:
            array_parts[name].append(
                np.asarray(extension_arrays[extension.root][name][selected_indexes])
            )

    candidate_ids = [str(row.get("candidate_id", "")) for row in rows]
    if not all(candidate_ids) or len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("duplicate or empty candidate IDs across merged datasets")
    feature_hashes = [str(row.get("candidate_feature_sha256", "")) for row in rows]
    if not all(feature_hashes) or (
        not deduplicate_feature_hashes
        and len(feature_hashes) != len(set(feature_hashes))
    ):
        raise ValueError("duplicate or empty candidate feature hashes across merged datasets")
    _verify_split_disjointness(rows)

    arrays = {
        name: np.concatenate(parts, axis=0) for name, parts in array_parts.items()
    }
    for index, row in enumerate(rows):
        row["feature_index"] = index

    corpus = copy.deepcopy(base.corpus)
    corpus["detector"] = _absolute_bindings(base.corpus.get("detector"), base.root)
    corpus["bindings"] = _absolute_bindings(base.corpus.get("bindings"), base.root)
    corpus["examples"] = rows
    corpus["counts"]["selected_candidates"] = len(rows)
    corpus["counts"]["selected_positives"] = int(np.sum(arrays["labels.npy"] == 1))
    corpus["counts"]["selected_negatives"] = int(np.sum(arrays["labels.npy"] == 0))

    for split in SPLITS:
        base_split = _split_counts(base.corpus, split, "base corpus")
        merged_split = copy.deepcopy(base_split)
        for field in DELTA_COUNT_FIELDS:
            original = _finite_number(
                base_split.get(field), f"base counts.by_split.{split}.{field}"
            )
            increment = sum(
                float(extension_deltas[item.root][split][field]) for item in loaded
            )
            value = original + increment
            merged_split[field] = (
                int(round(value))
                if field
                in {
                    "source_examples",
                    "raw_detector_candidates",
                    "raw_positive_candidates",
                    "raw_negative_candidates",
                }
                else value
            )
        split_rows = [row for row in rows if row["split"] == split]
        merged_split["selected_positive_candidates"] = sum(
            int(row["label"]) == 1 for row in split_rows
        )
        merged_split["selected_negative_candidates"] = sum(
            int(row["label"]) == 0 for row in split_rows
        )
        exposure = float(merged_split["exposure_seconds"])
        negative_exposure = float(merged_split["negative_exposure_seconds"])
        raw_total = int(merged_split["raw_detector_candidates"])
        raw_negative = int(merged_split["raw_negative_candidates"])
        merged_split["raw_candidate_rate_per_second"] = _rate(raw_total, exposure)
        merged_split["raw_candidate_rate_per_hour"] = _rate(
            raw_total, exposure, 3600.0
        )
        merged_split["raw_negative_candidate_rate_per_hour"] = _rate(
            raw_negative, negative_exposure, 3600.0
        )
        corpus["counts"]["by_split"][split] = merged_split

    policy = corpus["hard_negative_selection"]
    policies = [base.corpus["hard_negative_selection"]] + [
        item.corpus["hard_negative_selection"] for item in loaded
    ]
    for field in ("ranking", "group_by", "scope"):
        values = {item.get(field) for item in policies}
        if len(values) != 1:
            raise ValueError(f"extensions disagree on hard-negative {field}")
    train_negatives = [
        row for row in rows if row["split"] == "train" and int(row["label"]) == 0
    ]
    heldout_negatives = [
        row for row in rows if row["split"] in {"validation", "test"} and int(row["label"]) == 0
    ]
    policy["raw_training_count"] = _integer(
        base.corpus["hard_negative_selection"].get("raw_training_count"),
        "base raw training count",
    ) + sum(
        int(extension_deltas[item.root]["train"]["raw_negative_candidates"])
        for item in loaded
    )
    policy["selected_training_count"] = len(train_negatives)
    policy["heldout_candidates_unfiltered"] = len(heldout_negatives)
    policy["top_k"] = max(_integer(item.get("top_k"), "hard-negative top_k") for item in policies)

    corpus.setdefault("bindings", {})["immutable_base_candidate_corpus"] = _binding(
        base.corpus_path
    )
    corpus["bindings"]["candidate_verifier_extension_inputs"] = [
        _transitive_input(item) for item in loaded
    ]
    corpus["candidate_verifier_extension_merge"] = {
        "ordering": "extension_corpus_sha256_then_resolved_path",
        "duplicate_feature_hash_policy": (
            "keep_first_in_merge_order"
            if deduplicate_feature_hashes
            else "reject"
        ),
        "duplicate_feature_hashes_skipped": duplicate_feature_hashes_skipped,
        "immutable_base": _binding(base.corpus_path),
        "extensions": [
            {
                "corpus": _binding(item.corpus_path),
                "base_rows": base_count,
                "appended_rows": len(item.rows) - base_count,
                "deltas_by_split": extension_deltas[item.root],
            }
            for item in loaded
        ],
    }
    _verify_hard_negative_policy(corpus, rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        for name in ARRAY_NAMES:
            _atomic_npy(temporary / name, arrays[name])
        corpus["array_sha256"] = {
            name: sha256_file(temporary / name) for name in ARRAY_NAMES
        }
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
        "base_rows": base_count,
        "extensions": len(loaded),
        "appended_rows": len(rows) - base_count,
        "duplicate_feature_hashes_skipped": duplicate_feature_hashes_skipped,
        "rows": len(rows),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--base-corpus-sha256", required=True)
    parser.add_argument(
        "--extension",
        nargs=2,
        action="append",
        metavar=("DATASET", "CORPUS_SHA256"),
        required=True,
        help="verified extension dataset and its expected corpus hash (repeat at least twice)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--deduplicate-feature-hashes",
        action="store_true",
        help="keep the first provenance-ordered copy of identical candidate windows",
    )
    args = parser.parse_args(argv)
    extensions = [(Path(path), digest) for path, digest in args.extension]
    result = merge_extensions(
        args.base_dataset,
        args.base_corpus_sha256,
        extensions,
        args.output,
        deduplicate_feature_hashes=args.deduplicate_feature_hashes,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
