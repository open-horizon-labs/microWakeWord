#!/usr/bin/env python3
"""Materialize a leakage-free subset of a mined-negative manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

from microwakeword.hard_negative_mining import (
    _write_manifest,
    _write_records,
    sha256_file,
)

FORBIDDEN_COMPONENTS = {"observations", "false-wakes", "evidence"}


def _manifest_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("mining manifest is empty")
    return rows


def _forbidden(path: Path) -> bool:
    return bool(
        {part.casefold() for part in path.resolve().parts} & FORBIDDEN_COMPONENTS
    )


def filter_manifest(
    source_manifest: Path,
    output: Path,
    allowed_archives: Iterable[Path],
    *,
    context_frames: int = 200,
) -> dict:
    """Filter by exact archive path and rebuild the selected feature mmaps."""
    if context_frames <= 0:
        raise ValueError("context_frames must be positive")
    allowed = {str(Path(path).resolve()) for path in allowed_archives}
    if not allowed:
        raise ValueError("at least one allowed archive is required")
    for path_text in allowed:
        path = Path(path_text)
        if _forbidden(path):
            raise ValueError(f"quarantined archive is forbidden: {path}")
        if not path.is_dir() or not (path / "data.ninja").is_file():
            raise ValueError(f"allowed archive is not a RaggedMmap: {path}")

    rows = _manifest_rows(source_manifest)
    selected = []
    verified_hashes: dict[str, str] = {}
    for row in rows:
        source = Path(str(row.get("source_path", ""))).resolve()
        if str(source) not in allowed:
            continue
        if _forbidden(source):
            raise ValueError(f"mined row references quarantined evidence: {source}")
        expected_hash = str(row.get("source_hash", ""))
        actual_hash = verified_hashes.get(str(source))
        if actual_hash is None:
            actual_hash = sha256_file(source / "data.ninja")
            verified_hashes[str(source)] = actual_hash
        if not expected_hash or actual_hash != expected_hash:
            raise ValueError(f"source hash mismatch: {source}")
        selected.append(row)
    if not selected:
        raise ValueError("no mined rows matched the allowed archives")

    high_score = [
        row for row in selected if str(row.get("reason", "")).startswith("high_score:")
    ]
    reserve = [row for row in selected if row.get("reason") == "random_reserve"]
    if not high_score or not reserve:
        raise ValueError(
            "filtered mining data needs hard examples and a random reserve"
        )
    selected_paths = {str(Path(row["source_path"]).resolve()) for row in selected}
    if selected_paths != allowed:
        missing = sorted(allowed - selected_paths)
        raise ValueError(f"allowed archives have no selected rows: {missing}")

    source_digest = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    filter_contract = {
        "schema_version": 1,
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": source_digest,
        "allowed_archives": sorted(allowed),
        "context_frames": context_frames,
    }
    token = hashlib.sha256(
        json.dumps(filter_contract, sort_keys=True).encode()
    ).hexdigest()
    high_count = _write_records(high_score, output, "mined", context_frames, token)
    reserve_count = _write_records(
        reserve, output, "random_reserve", context_frames, token
    )
    _write_manifest(output / "mining-manifest.jsonl", high_score, reserve)
    metadata = {
        **filter_contract,
        "filter_sha256": token,
        "selected": high_count,
        "random_reserve": reserve_count,
        "reserve_fraction": reserve_count / (high_count + reserve_count),
    }
    if not 0.05 <= metadata["reserve_fraction"] <= 0.25:
        raise ValueError("filtered random reserve fraction is outside the safe bound")
    output.mkdir(parents=True, exist_ok=True)
    (output / "mining-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--allowed-archive", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-frames", type=int, default=200)
    args = parser.parse_args()
    result = filter_manifest(
        args.source_manifest,
        args.output,
        args.allowed_archive,
        context_frames=args.context_frames,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
