#!/usr/bin/env python3
"""Validate the bounded Kizz batch-mixture sidecar after training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from microwakeword.kizz_batch_mixture import validate_realized_mixture


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    args = parser.parse_args()
    ledger = json.loads(args.ledger.read_text())
    recipe = yaml.safe_load(args.recipe.read_text())
    guard = recipe.get("mixture_guard")
    if guard is None:
        raise ValueError("recipe does not declare mixture_guard")
    validate_realized_mixture(ledger, guard)
    print("Kizz batch mixture passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
