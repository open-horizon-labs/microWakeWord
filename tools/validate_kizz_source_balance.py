#!/usr/bin/env python3
"""Reject Kizz manifests that violate the source-balance contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from microwakeword.kizz_data_contract import validate_balance_manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-check-paths", action="store_true")
    args = parser.parse_args(argv)
    report = validate_balance_manifest(
        args.manifest,
        args.contract,
        check_paths=not args.no_check_paths,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
