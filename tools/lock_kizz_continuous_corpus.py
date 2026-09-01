#!/usr/bin/env python3
"""Freeze the raw-audio continuous qualification set before model scoring.

The policy deliberately favors long MUSAN streams so the 100-hour gate does
not turn into tens of thousands of tiny inference calls.  Validation rows are
never selected: the frozen detector threshold was chosen on that partition.
The resulting JSON is also an exclusion list for every later student corpus.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


SCHEMA_VERSION = 1
DEFAULT_MINIMUM_HOURS = 100.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_category(value: str) -> str:
    return "speech" if value == "connected_speech" else value


def lock_rows(rows: list[dict], *, minimum_hours: float = DEFAULT_MINIMUM_HOURS) -> list[dict]:
    """Select a deterministic, validation-disjoint mixed 100-hour corpus."""
    if minimum_hours < 100.0:
        raise ValueError("the continuous qualification lock may not be below 100 hours")

    def base(row: dict) -> bool:
        return (
            row.get("source") == "MUSAN"
            and row.get("split") in {"train", "test"}
        ) or (
            row.get("source") == "ESC-50-derived-backgrounds-v1"
            and row.get("split") == "unassigned"
        )

    selected = [dict(row) for row in rows if base(row)]
    selected_hashes = {row["sha256"] for row in selected}
    exposure = sum(float(row["duration_s"]) for row in selected)
    fillers = sorted(
        (
            row
            for row in rows
            if row.get("source") == "LibriSpeech-train-clean-100"
            and row.get("split") == "train"
            and row["sha256"] not in selected_hashes
        ),
        key=lambda row: (row["sha256"], row["path"]),
    )
    for row in fillers:
        if exposure >= minimum_hours * 3600.0:
            break
        selected.append(dict(row))
        selected_hashes.add(row["sha256"])
        exposure += float(row["duration_s"])
    if exposure < minimum_hours * 3600.0:
        raise ValueError("source inventory cannot satisfy the continuous exposure gate")
    if len(selected_hashes) != len(selected):
        raise ValueError("continuous lock contains duplicate audio hashes")
    if any(row.get("split") == "validation" for row in selected):
        raise ValueError("threshold-selection validation audio entered the lock")
    categories = {_normalized_category(str(row["category"])) for row in selected}
    if categories != {"speech", "music", "noise"}:
        raise ValueError(f"continuous lock lacks required categories: {sorted(categories)}")
    return sorted(selected, key=lambda row: (row["sha256"], row["path"]))


def build_lock(source_csv: Path, output: Path, *, minimum_hours: float) -> dict:
    with source_csv.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    required = {"path", "sha256", "duration_s", "category", "split", "source"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("negative source CSV is empty or missing required columns")
    selected = lock_rows(rows, minimum_hours=minimum_hours)
    examples = []
    for row in selected:
        path = Path(row["path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        examples.append(
            {
                "path": str(path),
                "sha256": row["sha256"],
                "duration_seconds": float(row["duration_s"]),
                "category": _normalized_category(row["category"]),
                "source": row["source"],
                "split": row["split"],
                "license": row.get("license"),
                "provenance": row.get("provenance"),
            }
        )
    seconds = sum(row["duration_seconds"] for row in examples)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "gate_scope": "locked_untouched_continuous_negative_corpus",
        "locked_before_scoring": True,
        "training_eligible": False,
        "source_manifest": str(source_csv.resolve()),
        "source_manifest_sha256": sha256_file(source_csv),
        "selection_policy": {
            "base": "all MUSAN train/test plus ESC-50 unassigned",
            "deterministic_fill": "LibriSpeech train rows sorted by audio SHA",
            "excluded_splits": ["validation", "heldout_deployment"],
            "minimum_hours": minimum_hours,
        },
        "counts": {
            "files": len(examples),
            "exposure_seconds": seconds,
            "exposure_hours": seconds / 3600.0,
            "categories": dict(Counter(row["category"] for row in examples)),
            "sources": dict(Counter(row["source"] for row in examples)),
        },
        "examples": examples,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-hours", type=float, default=DEFAULT_MINIMUM_HOURS)
    args = parser.parse_args()
    report = build_lock(args.source_csv, args.output, minimum_hours=args.minimum_hours)
    print(json.dumps(report["counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
