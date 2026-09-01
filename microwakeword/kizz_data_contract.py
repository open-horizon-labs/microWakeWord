# coding=utf-8
"""Executable source-balance and split contract for Kizz training data."""

from __future__ import annotations

import hashlib
import json
import wave
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


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _duration_seconds(item: dict[str, Any], path: Path) -> float | None:
    """Return declared or WAV-derived duration without trusting a bad value."""
    declared = item.get("duration_seconds")
    if declared is not None:
        try:
            duration = float(declared)
        except (TypeError, ValueError):
            return None
        return duration if duration > 0 else None
    try:
        with wave.open(str(path), "rb") as audio:
            rate = audio.getframerate()
            frames = audio.getnframes()
    except (OSError, EOFError, wave.Error):
        return None
    if rate <= 0 or frames <= 0:
        return None
    return frames / rate


def _duration_fraction(values: dict[str, float], total: float) -> dict[str, float]:
    return {
        key: (value / total if total else 0.0) for key, value in sorted(values.items())
    }


def _configured_duration(contract: dict[str, Any]) -> dict[str, Any]:
    duration = contract.get("duration", {})
    if not isinstance(duration, dict):
        return {"enabled": bool(duration)}
    return duration


def _configured_semantics(contract: dict[str, Any]) -> dict[str, Any]:
    semantics = contract.get(
        "semantic_labels", contract.get("semantic_label_policy", {})
    )
    return semantics if isinstance(semantics, dict) else {"enabled": bool(semantics)}


def _configured_provenance(contract: dict[str, Any]) -> dict[str, Any]:
    provenance = contract.get("provenance", {})
    return provenance if isinstance(provenance, dict) else {"enabled": bool(provenance)}


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
    split_domains: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    split_paths: dict[str, set[str]] = defaultdict(set)
    counts: Counter[tuple[str, int, str]] = Counter()
    durations: dict[tuple[str, int, str], float] = defaultdict(float)
    duration_config = _configured_duration(contract)
    duration_enabled = bool(duration_config.get("enabled", False))
    semantics = _configured_semantics(contract)
    semantic_enabled = bool(semantics.get("enabled", False))
    provenance = _configured_provenance(contract)
    provenance_enabled = bool(provenance.get("enabled", False))
    semantic_field = str(semantics.get("field", "semantic_label"))
    canonical_labels = set(
        semantics.get(
            "canonical_labels", [semantics.get("canonical", "canonical_positive")]
        )
    )
    prohibited_labels = set(
        semantics.get("prohibited_labels", semantics.get("prohibited_variants", []))
    )
    provenance_fields = tuple(
        provenance.get(
            "required_fields",
            ["source_id", "provenance_id", "parent_id", "ancestry_id"],
        )
    )
    disjoint = contract.get("split_disjoint", {})

    def disjoint_enabled(field: str) -> bool:
        defaults = {
            "speaker_id": True,
            "session_id": True,
            "source_id": provenance_enabled,
            "provenance_id": provenance_enabled,
            "parent_id": provenance_enabled,
            "ancestry_id": provenance_enabled,
            "source_group": False,
        }
        return bool(disjoint.get(field, defaults.get(field, False)))

    leakage_fields = {
        "source_group": "source_group",
        "source_id": "source_id",
        "provenance_id": "provenance_id",
        "parent_id": "parent_id",
        "ancestry_id": "ancestry_id",
    }
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
            violations.append(
                f"{prefix} lacks path/source_group/split or has invalid label"
            )
            continue
        if check_paths and not Path(path).is_file():
            violations.append(f"{prefix} path does not exist: {path}")
        if path in paths:
            violations.append(f"path appears more than once in manifest: {path}")
        paths[path] = split
        split_paths[split].add(path)
        counts[(split, int(label), source_group)] += 1
        audio_path = Path(path)
        if semantic_enabled and int(label) == 1:
            semantic_label = item.get(semantic_field)
            if not _nonempty_string(semantic_label):
                violations.append(
                    f"{prefix} positive is missing semantic label: {semantic_field}"
                )
            elif semantic_label in prohibited_labels:
                violations.append(
                    f"{prefix} uses prohibited positive semantic label: {semantic_label}"
                )
            elif semantic_label not in canonical_labels:
                violations.append(
                    f"{prefix} positive semantic label is not canonical: {semantic_label}"
                )
        if provenance_enabled:
            for field in provenance_fields:
                if not _nonempty_string(item.get(field)):
                    violations.append(
                        f"{prefix} is missing required provenance field: {field}"
                    )
        if duration_enabled:
            duration = _duration_seconds(item, audio_path)
            if duration is None:
                violations.append(
                    f"{prefix} lacks a valid duration_seconds or readable WAV duration"
                )
            else:
                durations[(split, int(label), source_group)] += duration
                minimum = duration_config.get("min_seconds")
                maximum = duration_config.get("max_seconds")
                if minimum is not None and duration < float(minimum):
                    violations.append(
                        f"{prefix} duration {duration:.6f}s is below minimum {float(minimum):.6f}s"
                    )
                if maximum is not None and duration > float(maximum):
                    violations.append(
                        f"{prefix} duration {duration:.6f}s exceeds maximum {float(maximum):.6f}s"
                    )
        speaker = item.get("speaker_id")
        session = item.get("session_id")
        if speaker:
            split_speakers[split].add(str(speaker))
        if session:
            split_sessions[split].add(str(session))
        for field, domain in leakage_fields.items():
            if disjoint_enabled(field) and _nonempty_string(item.get(domain)):
                split_domains[split][field].add(str(item[domain]))
        for required_field in contract.get("required_metadata", []):
            if not item.get(required_field):
                violations.append(
                    f"{prefix} is missing required metadata: {required_field}"
                )

    split_names = tuple(contract.get("splits", ["train", "validation", "test"]))
    required_each_split = tuple(contract.get("require_each_split", []))
    for first_index, first in enumerate(split_names):
        for second in split_names[first_index + 1 :]:
            if (
                disjoint_enabled("speaker_id")
                and split_speakers[first] & split_speakers[second]
            ):
                violations.append(f"speaker IDs overlap {first}/{second}")
            if (
                disjoint_enabled("session_id")
                and split_sessions[first] & split_sessions[second]
            ):
                violations.append(f"session IDs overlap {first}/{second}")
            for field in leakage_fields:
                if (
                    disjoint_enabled(field)
                    and split_domains[first][field] & split_domains[second][field]
                ):
                    violations.append(f"{field} values overlap {first}/{second}")

    for split in split_names:
        for class_name in required_each_split:
            label = (
                1
                if class_name == "positive"
                else 0 if class_name == "negative" else None
            )
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
    source_counts: dict[str, Counter[str]] = {
        "positive": Counter(),
        "negative": Counter(),
    }
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

    training_duration = {"positive": 0.0, "negative": 0.0}
    for (split, label, _source), duration in durations.items():
        if split == train_split:
            training_duration["positive" if label else "negative"] += duration
    overall_min_duration = overall.get("min_duration_seconds")
    overall_max_duration = overall.get("max_duration_seconds")
    total_training_duration = sum(training_duration.values())
    duration_positive_fraction = (
        training_duration["positive"] / total_training_duration
        if total_training_duration
        else 0.0
    )
    if (
        duration_enabled
        and overall_min_duration is not None
        and total_training_duration < float(overall_min_duration)
    ):
        violations.append(
            f"training duration {total_training_duration:.6f}s is below minimum {float(overall_min_duration):.6f}s"
        )
    if (
        duration_enabled
        and overall_max_duration is not None
        and total_training_duration > float(overall_max_duration)
    ):
        violations.append(
            f"training duration {total_training_duration:.6f}s exceeds maximum {float(overall_max_duration):.6f}s"
        )
    minimum_duration_positive = overall.get("min_positive_duration_fraction")
    maximum_duration_positive = overall.get("max_positive_duration_fraction")
    if (
        duration_enabled
        and minimum_duration_positive is not None
        and duration_positive_fraction < float(minimum_duration_positive)
    ):
        violations.append(
            "training positive duration fraction is below the configured minimum"
        )
    if (
        duration_enabled
        and maximum_duration_positive is not None
        and duration_positive_fraction > float(maximum_duration_positive)
    ):
        violations.append(
            "training positive duration fraction exceeds the configured maximum"
        )

    class_contract = contract.get("classes", {})
    training_source_report = {}
    for key in ("positive", "negative"):
        source_contract = class_contract.get(key, {})
        values = source_counts[key]
        class_total = sum(values.values())
        fractions = {
            source: _fraction(count, class_total)
            for source, count in sorted(values.items())
        }
        duration_values = {
            source: sum(
                duration
                for (split, label, duration_source), duration in durations.items()
                if split == train_split
                and label == (1 if key == "positive" else 0)
                and duration_source == source
            )
            for source in values
        }
        duration_total = sum(duration_values.values())
        duration_fractions = _duration_fraction(duration_values, duration_total)
        training_source_report[key] = {
            "total": class_total,
            "counts": dict(sorted(values.items())),
            "fractions": fractions,
            "duration_seconds": {
                source: round(value, 6)
                for source, value in sorted(duration_values.items())
            },
            "duration_fractions": duration_fractions,
        }
        min_groups = int(source_contract.get("min_source_groups", 0))
        if len(values) < min_groups:
            violations.append(
                f"{key} has {len(values)} source groups; requires {min_groups}"
            )
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
        class_min_duration = source_contract.get("min_duration_seconds")
        class_max_duration = source_contract.get("max_duration_seconds")
        if (
            duration_enabled
            and class_min_duration is not None
            and duration_total < float(class_min_duration)
        ):
            violations.append(
                f"{key} training duration is below the configured minimum"
            )
        if (
            duration_enabled
            and class_max_duration is not None
            and duration_total > float(class_max_duration)
        ):
            violations.append(f"{key} training duration exceeds the configured maximum")
        duration_max_fraction = source_contract.get("max_source_duration_fraction")
        if duration_enabled and duration_max_fraction is not None:
            for source, fraction in duration_fractions.items():
                if fraction > float(duration_max_fraction):
                    violations.append(
                        f"{key} source group {source} duration fraction is {fraction:.4f}; "
                        f"maximum is {float(duration_max_fraction):.4f}"
                    )
        for source, minimum in source_contract.get(
            "min_source_duration_fractions", {}
        ).items():
            fraction = duration_fractions.get(source, 0.0)
            if duration_enabled and fraction < float(minimum):
                violations.append(
                    f"{key} source group {source} duration fraction is {fraction:.4f}; "
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
        "split_duration_seconds": {
            split: {
                "positive": round(
                    sum(
                        duration
                        for (
                            split_key,
                            label_key,
                            _source,
                        ), duration in durations.items()
                        if split_key == split and label_key == 1
                    ),
                    6,
                ),
                "negative": round(
                    sum(
                        duration
                        for (
                            split_key,
                            label_key,
                            _source,
                        ), duration in durations.items()
                        if split_key == split and label_key == 0
                    ),
                    6,
                ),
            }
            for split in split_names
        },
        "training": {
            "positive_fraction": positive_fraction,
            "duration_seconds": {
                key: round(value, 6) for key, value in training_duration.items()
            },
            "total_duration_seconds": round(total_training_duration, 6),
            "positive_duration_fraction": duration_positive_fraction,
            "source_groups": training_source_report,
        },
        "violations": sorted(set(violations)),
        "qualified": not violations,
    }
    return report


__all__ = ["sha256_file", "validate_balance_manifest"]
