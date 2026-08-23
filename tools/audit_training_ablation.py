#!/usr/bin/env python3
"""Fail when a controlled training pair differs outside declared fields."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def differences(left: Any, right: Any, path: str = "") -> list[dict]:
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(set(left) | set(right)):
            child_path = f"{path}.{key}" if path else str(key)
            if key not in left:
                result.append({"path": child_path, "left": None, "right": right[key]})
            elif key not in right:
                result.append({"path": child_path, "left": left[key], "right": None})
            else:
                result.extend(differences(left[key], right[key], child_path))
        return result
    if isinstance(left, list) and isinstance(right, list):
        result = []
        for index in range(max(len(left), len(right))):
            child_path = f"{path}[{index}]"
            if index >= len(left):
                result.append({"path": child_path, "left": None, "right": right[index]})
            elif index >= len(right):
                result.append({"path": child_path, "left": left[index], "right": None})
            else:
                result.extend(differences(left[index], right[index], child_path))
        return result
    if left != right:
        return [{"path": path, "left": left, "right": right}]
    return []


def is_allowed(path: str, allowed: list[str]) -> bool:
    return any(path == prefix or path.startswith(f"{prefix}.") for prefix in allowed)


def audit(left_path: Path, right_path: Path, allowed: list[str]) -> dict:
    left = yaml.safe_load(left_path.read_text())
    right = yaml.safe_load(right_path.read_text())
    found = differences(left, right)
    return {
        "left": str(left_path),
        "right": str(right_path),
        "left_sha256": hashlib.sha256(left_path.read_bytes()).hexdigest(),
        "right_sha256": hashlib.sha256(right_path.read_bytes()).hexdigest(),
        "allowed_difference_paths": sorted(allowed),
        "differences": found,
        "unexpected_differences": [
            item for item in found if not is_allowed(item["path"], allowed)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--allowed-difference", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = audit(args.left, args.right, args.allowed_difference)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 1 if report["unexpected_differences"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
