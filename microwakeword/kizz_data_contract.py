# coding=utf-8
"""Executable source-balance and split contract for Kizz training data."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fraction(value: int, total: int) -> float:
    return float(value / total) if total else 0.0


def validate_balance_manifest(
    manifest_path: Path,
    contract_path: Path,
    *,
    check_paths: bool = True,
) -> dict[str, Any]:
    """Return a report; ``qualified`` is false for any contract violation."""
    import yaml

    manifest = json.loads(manifest_path.read_text())
    contract = yaml.safe_load(contract_path.read_text())
    examples = manifest.get("examples")
    violations: list[str] = []
    if manifest.get("schema_version") != 2:
        violations.append("manifest schema_version must be 2")
    if not isinstance(examples, list) or not examples:
        violations.append("manifest examples must be a non-empty list")
        examples = []

    paths: dict[str, str] = {}
    split_speakers: dict[str, set[str]] = defaultdict(set)
    split_sessions: dict[str, set[str]] = defaultdict(set)
    split_paths: dict[str, set[str]] = defaultdict(set)
    counts: Counter[tuple[str, int, str]] = Counter()
    for index, item in enumerate(examples):
        prefix = f"examples[{index}]"
        if not isinstance(item, dict):
            violations.append(f"{prefix} must be an object")
            continue
        path = str(item.get("path", ""))
        source_group = str(item.get("source_group", ""))
        split = str(item.get("split", ""))
        label = item.get("label")
        if not path or not source_group or not split or label not in (0, 1):
            violations.append(f"{prefix} lacks path/source_group/split or has invalid label")
            continue
        if check_paths and not Path(path).is_file():
            violations.append(f"{prefix} path does not exist: {path}")
        if path in paths:
            violations.append(f"path appears more than once in manifest: {path}")
        paths[path] = split
        split_paths[split].add(path)
        counts[(split, int(label), source_group)] += 1
        speaker = item.get("speaker_id")
        session = item.get("session_id")
        if speaker:
            split_speakers[split].add(str(speaker))
        if session:
            split_sessions[split].add(str(session))
        for required_field in contract.get("required_metadata", []):
            if not item.get(required_field):
                violations.append(f"{prefix} is missing required metadata: {required_field}")

    split_disjoint = contract.get("split_disjoint", {})
    split_names = tuple(contract.get("splits", ["train", "validation", "test"]))
    required_each_split = tuple(contract.get("require_each_split", []))
    for first_index, first in enumerate(split_names):
        for second in split_names[first_index + 1 :]:
            if split_disjoint.get("speaker_id", True) and split_speakers[first] & split_speakers[second]:
                violations.append(f"speaker IDs overlap {first}/{second}")
            if split_disjoint.get("session_id", True) and split_sessions[first] & split_sessions[second]:
                violations.append(f"session IDs overlap {first}/{second}")

    for split in split_names:
        for class_name in required_each_split:
            label = 1 if class_name == "positive" else 0 if class_name == "negative" else None
            if label is None:
                violations.append(f"unknown required split class: {class_name}")
            elif not any(
                split_key == split and label_key == label
                for split_key, label_key, _source in counts
            ):
                violations.append(f"{split} split has no {class_name} examples")

    train_split = str(contract.get("training_split", "train"))
    overall = contract.get("overall", {})
    train_counts = Counter()
    source_counts: dict[str, Counter[str]] = {"positive": Counter(), "negative": Counter()}
    for (split, label, source_group), count in counts.items():
        if split != train_split:
            continue
        key = "positive" if label else "negative"
        train_counts[key] += count
        source_counts[key][source_group] += count
    total = train_counts["positive"] + train_counts["negative"]
    positive_fraction = _fraction(train_counts["positive"], total)
    minimum_positive = float(overall.get("min_positive_fraction", 0.0))
    maximum_positive = float(overall.get("max_positive_fraction", 1.0))
    if not minimum_positive <= positive_fraction <= maximum_positive:
        violations.append(
            f"training positive fraction {positive_fraction:.4f} is outside "
            f"[{minimum_positive}, {maximum_positive}]"
        )

    class_contract = contract.get("classes", {})
    training_source_report = {}
    for key in ("positive", "negative"):
        source_contract = class_contract.get(key, {})
        values = source_counts[key]
        class_total = sum(values.values())
        fractions = {source: _fraction(count, class_total) for source, count in sorted(values.items())}
        training_source_report[key] = {
            "total": class_total,
            "counts": dict(sorted(values.items())),
            "fractions": fractions,
        }
        min_groups = int(source_contract.get("min_source_groups", 0))
        if len(values) < min_groups:
            violations.append(f"{key} has {len(values)} source groups; requires {min_groups}")
        for required in source_contract.get("required_source_groups", []):
            if not values.get(required):
                violations.append(f"{key} is missing required source group: {required}")
        max_fraction = source_contract.get("max_source_fraction")
        if max_fraction is not None:
            for source, fraction in fractions.items():
                if fraction > float(max_fraction):
                    violations.append(
                        f"{key} source group {source} is {fraction:.4f}; "
                        f"maximum is {float(max_fraction):.4f}"
                    )
        for source, minimum in source_contract.get("min_source_fractions", {}).items():
            fraction = fractions.get(source, 0.0)
            if fraction < float(minimum):
                violations.append(
                    f"{key} source group {source} is {fraction:.4f}; "
                    f"minimum is {float(minimum):.4f}"
                )

    report = {
        "schema_version": 1,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "contract": str(contract_path.resolve()),
        "contract_sha256": sha256_file(contract_path),
        "example_count": len(examples),
        "split_counts": {
            split: {
                "positive": sum(
                    count
                    for (split_key, label_key, _source), count in counts.items()
                    if split_key == split and label_key == 1
                ),
                "negative": sum(
                    count
                    for (split_key, label_key, _source), count in counts.items()
                    if split_key == split and label_key == 0
                ),
            }
            for split in split_names
        },
        "training": {
            "positive_fraction": positive_fraction,
            "source_groups": training_source_report,
        },
        "violations": sorted(set(violations)),
        "qualified": not violations,
    }
    return report


__all__ = ["sha256_file", "validate_balance_manifest"]
