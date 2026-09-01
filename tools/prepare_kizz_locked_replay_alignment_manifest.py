#!/usr/bin/env python3
"""Prepare locked test-only replay sources for acoustic phone alignment.

The generic aligner requires ``training_eligible`` inputs.  This sidecar makes
that field true only as an alignment-tool compatibility flag while preserving
an explicit test-only, non-training contract in the enclosing manifest and in
every selected row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(source_manifest: Path, selection: Path) -> dict[str, Any]:
    source = json.loads(source_manifest.read_text())
    selected = json.loads(selection.read_text())
    source_rows = {
        str(row.get("source_id", "")): row for row in source.get("examples", [])
    }
    rows = []
    for locked in selected.get("selected_examples", []):
        row = source_rows.get(str(locked.get("source_id", "")))
        if (
            row is None
            or row.get("split") != "test"
            or int(row.get("label", -1)) != 1
            or row.get("audio_sha256") != locked.get("audio_sha256")
        ):
            raise ValueError("locked replay source differs from source manifest")
        item = dict(row)
        item["training_eligible"] = True
        # The generic aligner groups accepted rows by provenance identity.
        # Fresh qualification manifests use source_id as that immutable identity.
        item.setdefault("provenance_id", str(item["source_id"]))
        item["alignment_only_locked_test"] = True
        item["training_eligible_after_alignment"] = False
        item["locked_deployment_anchor"] = False
        rows.append(item)
    if not rows or len(rows) != selected.get("selected_count"):
        raise ValueError("locked replay selection count differs")
    hashes = [str(row.get("audio_sha256", "")) for row in rows]
    if any(not value for value in hashes) or len(hashes) != len(set(hashes)):
        raise ValueError("locked replay sources have missing or duplicate audio")
    return {
        "schema_version": 1,
        "kind": "kizz_control_locked_replay_alignment_input",
        "gate_scope": "alignment_only_locked_test_positive",
        "training_eligible": False,
        "compatibility_override": {
            "fields": ["examples[].training_eligible", "examples[].provenance_id"],
            "reason": "generic aligner input filter and grouping identity",
            "permitted_use": "acoustic phone alignment only",
        },
        "bindings": {
            "source_manifest": {
                "path": str(source_manifest.resolve()),
                "sha256": sha256_file(source_manifest),
            },
            "selection": {
                "path": str(selection.resolve()),
                "sha256": sha256_file(selection),
            },
        },
        "examples": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare(args.source_manifest.resolve(), args.selection.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "examples": len(result["examples"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
