#!/usr/bin/env python3
"""Compose immutable validation/test feature windows for deployed INT8 tracing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def parse_group_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("source must be GROUP=PATH")
    group, raw_path = value.split("=", 1)
    path = Path(raw_path).expanduser().resolve()
    if not group or not path.is_file() or path.suffix != ".npy":
        raise argparse.ArgumentTypeError("source must be GROUP=existing.npy")
    return group, path


def _load(path: Path) -> np.ndarray:
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.ndim != 3 or tuple(values.shape[1:]) != (260, 40):
        raise ValueError(f"{path}: expected [N,260,40]")
    if len(values) < 1 or not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"{path}: feature array must be nonempty numeric data")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: feature array contains non-finite values")
    return values


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        with open(temporary, "wb") as output:
            np.save(output, values, allow_pickle=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def prepare_bundle(
    provenance_path: Path,
    validation_positive: Path,
    validation_negative: Sequence[tuple[str, Path]],
    test_positive: Path,
    test_negative: Sequence[tuple[str, Path]],
    output: Path,
) -> dict:
    provenance_path = provenance_path.expanduser().resolve()
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("schema_version") != 3 or provenance.get("recipe") != "kizz_aligned_teacher_features_v3":
        raise ValueError("feature provenance is not the aligned Kizz evaluation contract")

    sources = [
        ("validation", 1, "positive", validation_positive.resolve()),
        *(("validation", 0, group, path.resolve()) for group, path in validation_negative),
        ("test", 1, "positive", test_positive.resolve()),
        *(("test", 0, group, path.resolve()) for group, path in test_negative),
    ]
    if len({(split, group) for split, _, group, _ in sources}) != len(sources):
        raise ValueError("split/group identities must be unique")

    expected_positive = provenance.get("positive_counts", {})
    expected_negative = provenance.get("negative_counts", {})
    arrays: list[np.ndarray] = []
    rows: list[dict] = []
    bindings: list[dict] = []
    feature_index = 0
    for split, label, group, path in sources:
        values = _load(path)
        expected = (
            expected_positive.get(split)
            if label == 1
            else expected_negative.get(split, {}).get(group)
        )
        if expected != len(values):
            raise ValueError(f"{split}/{group}: count differs from feature provenance")
        bindings.append(
            {
                "split": split,
                "label": label,
                "group": group,
                "path": str(path),
                "sha256": sha256_file(path),
                "count": len(values),
            }
        )
        for group_index, sample in enumerate(values):
            rows.append(
                {
                    "source_id": f"{split}:{group}:{group_index:06d}",
                    "feature_index": feature_index,
                    "feature_sha256": feature_sha256(sample),
                    "split": split,
                    "label": label,
                    "group": group,
                }
            )
            feature_index += 1
        arrays.append(np.asarray(values))

    output = output.expanduser().resolve()
    feature_path = output / "evaluation-features.npy"
    manifest_path = output / "evaluation-manifest.json"
    combined = np.concatenate(arrays, axis=0)
    _atomic_npy(feature_path, combined)
    payload = {
        "schema_version": 1,
        "recipe": "kizz_control_int8_evaluation_bundle_v1",
        "deployment_qualification": False,
        "feature_provenance": {
            "path": str(provenance_path),
            "sha256": sha256_file(provenance_path),
        },
        "source_arrays": bindings,
        "array_sha256": {feature_path.name: sha256_file(feature_path)},
        "counts": {
            "examples": len(rows),
            "validation": sum(row["split"] == "validation" for row in rows),
            "test": sum(row["split"] == "test" for row in rows),
        },
        "examples": rows,
    }
    _atomic_json(manifest_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-provenance", type=Path, required=True)
    parser.add_argument("--validation-positive", type=Path, required=True)
    parser.add_argument("--validation-negative", type=parse_group_path, action="append", required=True)
    parser.add_argument("--test-positive", type=Path, required=True)
    parser.add_argument("--test-negative", type=parse_group_path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = prepare_bundle(
        args.feature_provenance,
        args.validation_positive,
        args.validation_negative,
        args.test_positive,
        args.test_negative,
        args.output,
    )
    print(json.dumps(payload["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
