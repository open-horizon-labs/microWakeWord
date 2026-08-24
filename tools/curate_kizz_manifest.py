#!/usr/bin/env python3
"""Create a deterministic, reversible Kizz training manifest curation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence


def stable_rank(path: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{path}".encode()).hexdigest()


def parse_cap(value: str) -> tuple[str, int]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("source cap must be GROUP=COUNT")
    group, raw_count = value.split("=", 1)
    try:
        count = int(raw_count)
    except ValueError as error:
        raise argparse.ArgumentTypeError("source cap count must be an integer") from error
    if not group or count < 0:
        raise argparse.ArgumentTypeError("source cap must have a non-negative count")
    return group, count


def curate(
    manifest_path: Path,
    output_path: Path,
    *,
    seed: int,
    caps: dict[str, int],
    drops: set[str] | None = None,
) -> dict:
    payload = json.loads(manifest_path.read_text())
    examples = payload.get("examples")
    if payload.get("schema_version") != 2 or not isinstance(examples, list):
        raise ValueError("input must be a schema_version 2 manifest")

    drops = drops or set()
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in examples:
        if item.get("source_group") in drops:
            continue
        grouped[(str(item.get("split")), str(item.get("source_group")))].append(item)

    selected: list[dict] = []
    selection_report = []
    for (split, source_group), values in sorted(grouped.items()):
        ranked = sorted(values, key=lambda item: stable_rank(str(item["path"]), seed))
        limit = caps.get(source_group) if split == "train" else None
        chosen = ranked if limit is None else ranked[:limit]
        selected.extend(chosen)
        selection_report.append(
            {
                "split": split,
                "source_group": source_group,
                "input_count": len(values),
                "selected_count": len(chosen),
                "cap": limit,
            }
        )

    result = {
        "schema_version": 2,
        "examples": sorted(selected, key=lambda item: (item["split"], item["label"], item["path"])),
        "curation": {
            "source_manifest": str(manifest_path.resolve()),
            "seed": seed,
            "training_source_caps": caps,
            "dropped_source_groups": sorted(drops),
            "selection": selection_report,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=24111)
    parser.add_argument("--max-training-source", type=parse_cap, action="append", default=[])
    parser.add_argument("--drop-source", action="append", default=[])
    args = parser.parse_args(argv)
    result = curate(
        args.manifest,
        args.output,
        seed=args.seed,
        caps=dict(args.max_training_source),
        drops=set(args.drop_source),
    )
    print(json.dumps({"output": str(args.output), "example_count": len(result["examples"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
