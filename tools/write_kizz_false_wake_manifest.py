#!/usr/bin/env python3
"""Materialize a path-only manifest for one quarantined false-wake split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("training", "held_out"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source = json.loads(args.manifest.read_text())
    cache = json.loads(args.cache_manifest.read_text())
    selected = {
        item["observation_id"]
        for item in cache["splits"][args.split]["observations"]
    }
    examples = [
        {
            "path": str((args.manifest.parent / item["path"]).resolve()),
            "label": 0,
            "source_id": f"device_false_wake_{args.split}",
        }
        for item in source["observations"]
        if item["observation_id"] in selected
    ]
    if not examples:
        raise ValueError(f"false-wake split is empty: {args.split}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema_version": 1, "examples": examples}, indent=2) + "\n")
    print(json.dumps({"split": args.split, "count": len(examples), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
