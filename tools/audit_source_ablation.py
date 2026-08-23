#!/usr/bin/env python3
"""Audit a training pair that changes source diversity but not class exposure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

try:
    from tools.audit_training_ablation import differences
except ModuleNotFoundError:  # Direct execution from the tools directory.
    from audit_training_ablation import differences


def features_by_source(config: dict) -> dict[str, dict]:
    sources = {}
    for feature in config["features"]:
        source = feature["sampling_source"]
        if source in sources:
            raise ValueError(f"duplicate sampling source: {source}")
        sources[source] = feature
    return sources


def class_exposure(config: dict) -> dict[str, float]:
    group_truths: dict[str, set[bool]] = {}
    for feature in config["features"]:
        group = feature.get("sampling_group")
        if group is None or feature.get("sampling_weight", 0) <= 0:
            continue
        group_truths.setdefault(group, set()).add(bool(feature["truth"]))
    exposure = {"positive": 0.0, "negative": 0.0}
    for group, weight in config["sampling_groups"].items():
        truths = group_truths.get(group, set())
        if len(truths) != 1:
            raise ValueError(f"sampling group must have one truth label: {group}")
        label = "positive" if next(iter(truths)) else "negative"
        exposure[label] += float(weight)
    return exposure


def audit(left_path: Path, right_path: Path, allowed_prefixes: list[str]) -> dict:
    left = yaml.safe_load(left_path.read_text())
    right = yaml.safe_load(right_path.read_text())
    left_sources = features_by_source(left)
    right_sources = features_by_source(right)
    shared = sorted(set(left_sources) & set(right_sources))
    added = sorted(set(right_sources) - set(left_sources))
    removed = sorted(set(left_sources) - set(right_sources))
    changed = {
        source: differences(left_sources[source], right_sources[source])
        for source in shared
        if left_sources[source] != right_sources[source]
    }
    unexpected_added = [
        source
        for source in added
        if not any(source.startswith(prefix) for prefix in allowed_prefixes)
    ]

    ignored = {"features", "sampling_groups", "sampling_plan", "train_dir"}
    frozen_left = {key: value for key, value in left.items() if key not in ignored}
    frozen_right = {key: value for key, value in right.items() if key not in ignored}
    frozen_differences = differences(frozen_left, frozen_right)
    left_exposure = class_exposure(left)
    right_exposure = class_exposure(right)
    exposure_differences = differences(left_exposure, right_exposure)
    passed = not (
        removed
        or changed
        or unexpected_added
        or frozen_differences
        or exposure_differences
    )
    return {
        "left": str(left_path),
        "right": str(right_path),
        "left_sha256": hashlib.sha256(left_path.read_bytes()).hexdigest(),
        "right_sha256": hashlib.sha256(right_path.read_bytes()).hexdigest(),
        "allowed_added_source_prefixes": sorted(allowed_prefixes),
        "shared_sources": len(shared),
        "added_sources": added,
        "removed_sources": removed,
        "changed_shared_sources": changed,
        "unexpected_added_sources": unexpected_added,
        "frozen_config_differences": frozen_differences,
        "left_class_exposure": left_exposure,
        "right_class_exposure": right_exposure,
        "class_exposure_differences": exposure_differences,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--allowed-added-source-prefix", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.left, args.right, args.allowed_added_source_prefix)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
