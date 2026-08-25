#!/usr/bin/env python3
"""Qualify timestamped continuous detector scores with a frozen validation cutoff."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from microwakeword.kizz_continuous_evaluation import (
    qualify_test_streams,
    select_threshold,
    stream_from_mapping,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="JSON with validation and test streams",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refractory-seconds", type=float, default=1.0)
    parser.add_argument("--max-event-duration-seconds", type=float)
    parser.add_argument("--min-recall", type=float, default=0.90)
    parser.add_argument("--max-faph-upper-95", type=float, default=0.10)
    parser.add_argument("--min-negative-exposure-hours", type=float, default=100.0)
    args = parser.parse_args(argv)
    payload = json.loads(args.input.read_text())
    if payload.get("schema_version") != 1:
        parser.error("input schema_version must be 1")
    streams = [stream_from_mapping(item) for item in payload.get("streams", [])]
    validation = [stream for stream in streams if stream.split == "validation"]
    test = [stream for stream in streams if stream.split == "test"]
    try:
        provenance = select_threshold(
            validation,
            min_recall=args.min_recall,
            refractory_seconds=args.refractory_seconds,
            max_event_duration_seconds=args.max_event_duration_seconds,
        )
        result = qualify_test_streams(
            test,
            provenance,
            refractory_seconds=args.refractory_seconds,
            max_event_duration_seconds=args.max_event_duration_seconds,
            min_recall=args.min_recall,
            max_faph_upper_95=args.max_faph_upper_95,
            min_negative_exposure_hours=args.min_negative_exposure_hours,
        )
    except ValueError as error:
        parser.error(str(error))
    report = {
        "schema_version": 1,
        "input": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input),
        "threshold_provenance": asdict(provenance),
        "qualification": asdict(result),
        "config": {
            "refractory_seconds": args.refractory_seconds,
            "max_event_duration_seconds": args.max_event_duration_seconds,
            "min_recall": args.min_recall,
            "max_faph_upper_95": args.max_faph_upper_95,
            "min_negative_exposure_hours": args.min_negative_exposure_hours,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if result.qualified else 2


if __name__ == "__main__":
    raise SystemExit(main())
