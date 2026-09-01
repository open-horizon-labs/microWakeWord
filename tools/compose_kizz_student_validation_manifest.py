#!/usr/bin/env python3
"""Compose clean and StackChan-channel rows into one student validation group."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from microwakeword.kizz_phoneme_teacher import sha256_file


def compose(clean_manifest: Path, device_manifest: Path, output: Path) -> dict:
    clean = json.loads(clean_manifest.read_text())
    device = json.loads(device_manifest.read_text())
    clean_rows = [dict(row) for row in clean.get("examples", [])]
    device_rows = [dict(row) for row in device.get("examples", [])]
    selected_clean = [
        row for row in clean_rows
        if row.get("split") == "validation" and int(row.get("label", -1)) in (0, 1)
    ]
    selected_device = [
        row for row in device_rows
        if row.get("split") == "validation" and int(row.get("label", -1)) == 1
    ]
    if not selected_clean or len(selected_device) != 12:
        raise ValueError("student validation composition is missing required evidence")
    payload = {
        "schema_version": 1,
        "kind": "kizz_student_multichannel_validation",
        "training_eligible": False,
        "sources": {
            "clean": {"path": str(clean_manifest.resolve()), "sha256": sha256_file(clean_manifest)},
            "device": {"path": str(device_manifest.resolve()), "sha256": sha256_file(device_manifest)},
        },
        "counts": {
            "clean": len(selected_clean),
            "device_positive": len(selected_device),
            "total": len(selected_clean) + len(selected_device),
        },
        "examples": selected_clean + selected_device,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--device-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = compose(args.clean_manifest, args.device_manifest, args.output)
    print(json.dumps(result["counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
