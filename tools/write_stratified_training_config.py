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
        sources = [
            source
            for source in manifest.get("feature_sources", [])
            if (selected_class is None or source["class"] == selected_class)
            and source["feature_split"] == selected_split
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
    config["sampling_plan"] = {
        "path": str(plan_path),
        "sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
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
