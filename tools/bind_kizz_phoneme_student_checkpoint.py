#!/usr/bin/env python3
"""Bind one retained Kizz student checkpoint to derived distillation metadata."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

from tools.convert_distilled_student import sha256_file


def _json_safe(value: object) -> object:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    return value


def bind_checkpoint(metadata_path: Path, step: int, output: Path) -> dict:
    if step < 1:
        raise ValueError("checkpoint step must be positive")
    metadata = json.loads(metadata_path.read_text())
    matches = [
        item for item in metadata.get("validation_ledger", [])
        if int(item.get("step", -1)) == step
    ]
    if len(matches) != 1:
        raise ValueError(f"distillation ledger has {len(matches)} entries for step {step}")
    checkpoint = matches[0].get("checkpoint") or {}
    weights = Path(str(checkpoint.get("path", ""))).resolve()
    declared_hash = checkpoint.get("sha256")
    if not weights.is_file() or not declared_hash or sha256_file(weights) != declared_hash:
        raise ValueError("checkpoint path or SHA-256 binding is invalid")
    if output.exists():
        raise FileExistsError(output)

    original_student = dict(metadata.get("student") or {})
    metadata["checkpoint_candidate"] = {
        "base_distillation_metadata": str(metadata_path.resolve()),
        "base_distillation_metadata_sha256": sha256_file(metadata_path),
        "step": step,
        "validation": matches[0],
    }
    metadata["student"] = {
        **original_student,
        "selected_checkpoint": f"step-{step:04d}",
        "weights": str(weights),
        "weights_sha256": declared_hash,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = _json_safe(metadata)
    output.write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distillation-metadata", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    metadata = bind_checkpoint(args.distillation_metadata, args.step, args.output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "step": args.step,
        "weights": metadata["student"]["weights"],
        "weights_sha256": metadata["student"]["weights_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
