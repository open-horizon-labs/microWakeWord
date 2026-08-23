#!/usr/bin/env python3
"""Expand a declarative sampling plan into a microWakeWord training config."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def _feature(entry: dict, features_dir: Path, source_name: str) -> dict:
    return {
        "features_dir": str(features_dir),
        "sampling_weight": float(entry.get("within_group_weight", 1.0)),
        "penalty_weight": float(entry.get("penalty_weight", 1.0)),
        "truth": bool(entry["truth"]),
        "truncation_strategy": entry.get("truncation_strategy", "random"),
        "type": "mmap",
        "sampling_group": entry.get("group"),
        "sampling_source": source_name,
        "evaluation_enabled": bool(entry.get("evaluation_enabled", True)),
    }


def expand_source(entry: dict) -> list[dict]:
    if manifest_path := entry.get("feature_build_manifest"):
        manifest = json.loads(Path(manifest_path).read_text())
        selected_class = entry.get("class")
        selected_split = entry.get("feature_split", "training")
        include_phrases = entry.get("include_phrases")
        exclude_phrases = entry.get("exclude_phrases")
        if include_phrases is not None and exclude_phrases is not None:
            raise ValueError("sampling source cannot combine include_phrases and exclude_phrases")
        for field_name, phrases in (
            ("include_phrases", include_phrases),
            ("exclude_phrases", exclude_phrases),
        ):
            if phrases is not None and (
                not isinstance(phrases, list)
                or not phrases
                or any(not isinstance(phrase, str) or not phrase for phrase in phrases)
            ):
                raise ValueError(f"{field_name} must be a non-empty list of phrases")
        included = set(include_phrases or [])
        excluded = set(exclude_phrases or [])
        sources = [
            source
            for source in manifest.get("feature_sources", [])
            if (selected_class is None or source["class"] == selected_class)
            and source["feature_split"] == selected_split
            and (not included or source.get("text") in included)
            and source.get("text") not in excluded
        ]
        if not sources:
            raise ValueError(f"no matching feature sources in {manifest_path}")
        prefix = entry["source_prefix"]
        return [
            _feature(
                entry,
                Path(source["features_dir"]),
                f"{prefix}:{source.get('text') or 'all'}",
            )
            for source in sources
        ]
    if features_dir := entry.get("features_dir"):
        return [_feature(entry, Path(features_dir), entry["source_name"])]
    raise ValueError("sampling source requires feature_build_manifest or features_dir")


def _stage_value(value: object, stage: int) -> float:
    values = value if isinstance(value, list) else [value]
    if not values or any(not isinstance(item, (int, float)) for item in values):
        raise ValueError("class weights must be numbers or non-empty lists of numbers")
    return float(values[min(stage, len(values) - 1)])


def planned_balance(config: dict, groups: dict[str, float], features: list[dict]) -> dict:
    """Summarize the static class pressure implied by a sampling plan."""
    group_details = {}
    total_group_weight = sum(groups.values())
    for group, group_weight in groups.items():
        active = [
            feature
            for feature in features
            if feature["sampling_group"] == group and feature["sampling_weight"] > 0
        ]
        truths = {feature["truth"] for feature in active}
        if len(truths) != 1:
            raise ValueError(f"sampling group {group!r} must contain one truth class")
        source_weight = sum(feature["sampling_weight"] for feature in active)
        average_penalty = sum(
            feature["sampling_weight"] * feature["penalty_weight"]
            for feature in active
        ) / source_weight
        group_details[group] = {
            "truth": truths.pop(),
            "sampling_share": group_weight / total_group_weight,
            "average_penalty_weight": average_penalty,
        }

    positive_share = sum(
        detail["sampling_share"] for detail in group_details.values() if detail["truth"]
    )
    stage_count = max(
        len(config.get("positive_class_weight", [1])),
        len(config.get("negative_class_weight", [1])),
    )
    pressure_stages = []
    for stage in range(stage_count):
        pressures = {
            truth: sum(
                detail["sampling_share"]
                * detail["average_penalty_weight"]
                * _stage_value(
                    config.get(
                        "positive_class_weight" if truth else "negative_class_weight",
                        [1],
                    ),
                    stage,
                )
                for detail in group_details.values()
                if detail["truth"] is truth
            )
            for truth in (True, False)
        }
        total_pressure = sum(pressures.values())
        pressure_stages.append(
            {
                "stage": stage,
                "positive_share": pressures[True] / total_pressure,
                "negative_share": pressures[False] / total_pressure,
            }
        )
    return {
        "positive_sampling_share": positive_share,
        "negative_sampling_share": 1 - positive_share,
        "weighted_pressure_stages": pressure_stages,
        "groups": group_details,
    }


def enforce_balance_guard(summary: dict, guard: dict) -> None:
    maximum_sampling = guard.get("maximum_negative_sampling_share")
    maximum_pressure = guard.get("maximum_negative_weighted_pressure_share")
    for name, value in (
        ("maximum_negative_sampling_share", maximum_sampling),
        ("maximum_negative_weighted_pressure_share", maximum_pressure),
    ):
        if value is not None and (
            not isinstance(value, (int, float)) or not 0 <= value <= 1
        ):
            raise ValueError(f"{name} must be between zero and one")
    if (
        maximum_sampling is not None
        and summary["negative_sampling_share"] > maximum_sampling
    ):
        raise ValueError("negative sampling share exceeds balance guard")
    if maximum_pressure is not None and any(
        stage["negative_share"] > maximum_pressure
        for stage in summary["weighted_pressure_stages"]
    ):
        raise ValueError("negative weighted pressure share exceeds balance guard")


def stratified_config(base_config: Path, plan_path: Path) -> dict:
    config = yaml.safe_load(base_config.read_text())
    plan = yaml.safe_load(plan_path.read_text())
    if plan.get("schema_version") != 1:
        raise ValueError("sampling plan requires schema_version 1")
    groups = {name: float(weight) for name, weight in plan["sampling_groups"].items()}
    if not groups or any(weight <= 0 for weight in groups.values()):
        raise ValueError("sampling group weights must be positive")
    features = [
        feature
        for entry in plan["sources"]
        for feature in expand_source(entry)
    ]
    active_groups = {
        feature["sampling_group"]
        for feature in features
        if feature["sampling_weight"] > 0
    }
    if active_groups != set(groups):
        raise ValueError(
            "sampling groups and active feature groups differ: "
            f"declared={sorted(groups)}, active={sorted(active_groups)}"
        )
    source_names = [feature["sampling_source"] for feature in features]
    if len(source_names) != len(set(source_names)):
        raise ValueError("sampling source names must be unique")
    config["sampling_groups"] = groups
    config["features"] = features
    config.update(plan.get("config_overrides", {}))
    balance = planned_balance(config, groups, features)
    enforce_balance_guard(balance, plan.get("balance_guard", {}))
    config["sampling_plan"] = {
        "path": str(plan_path),
        "sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "planned_balance": balance,
    }
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--sampling-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = stratified_config(args.base_config, args.sampling_plan)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
