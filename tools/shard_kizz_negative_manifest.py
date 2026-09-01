#!/usr/bin/env python3
"""Deterministically shard development-negative manifests by source identity."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.mine_kizz_librispeech_hard_negatives import (
    _atomic_json,
    _binding,
    sha256_file,
)


ASSIGNMENT_POLICY = "largest_identity_duration_first_then_least_loaded_shard"
IDENTITY_FIELDS = (
    "ancestry_id",
    "speaker_id",
    "session_id",
    "source_id",
    "audio_sha256",
    "sha256",
    "path",
)


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("examples"), list):
        raise ValueError("source manifest must be an object containing examples")
    return value


def _identity(row: Mapping[str, Any]) -> str:
    for field in IDENTITY_FIELDS:
        value = str(row.get(field, "")).strip()
        if value:
            return f"{field}:{value}"
    raise ValueError("development-negative row has no stable source identity")


def _source_id(row: Mapping[str, Any]) -> str:
    value = str(row.get("source_id", row.get("path", ""))).strip()
    if not value:
        raise ValueError("development-negative row has no source_id or path")
    return value


def _duration(row: Mapping[str, Any]) -> float:
    raw = row.get("duration_seconds")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError("development-negative duration_seconds must be numeric")
    value = float(raw)
    if not 0.0 < value < float("inf"):
        raise ValueError("development-negative duration_seconds must be finite and positive")
    return value


def _eligible(row: Mapping[str, Any]) -> bool:
    split = row.get("split")
    if split not in {"train", "validation"} or row.get("label") != 0:
        return False
    if row.get("locked_holdout") or row.get("locked_deployment_anchor"):
        return False
    if split == "train" and row.get("training_eligible") is False:
        return False
    return True


def _tie_break(seed: str, identity: str) -> str:
    return hashlib.sha256(f"{seed}\0{identity}".encode("utf-8")).hexdigest()


def shard_manifest(
    source_manifest: Path,
    output: Path,
    shard_count: int,
    seed: str = "kizz-negative-manifest-shard-v1",
) -> dict[str, Any]:
    """Publish balanced, identity-disjoint manifest shards atomically."""
    if shard_count < 2:
        raise ValueError("shard_count must be at least two")
    source_manifest = source_manifest.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")

    manifest = _load_manifest(source_manifest)
    eligible = []
    excluded = 0
    for raw in manifest["examples"]:
        if not isinstance(raw, dict):
            raise ValueError("source manifest examples must be objects")
        if _eligible(raw):
            eligible.append(copy.deepcopy(raw))
        else:
            excluded += 1
    if not eligible:
        raise ValueError("source manifest has no eligible development negatives")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        grouped[_identity(row)].append(row)
    if shard_count > len(grouped):
        raise ValueError("shard_count exceeds eligible source identities")

    groups = []
    for identity, rows in grouped.items():
        rows.sort(key=_source_id)
        duration = sum(_duration(row) for row in rows)
        groups.append((identity, duration, rows))
    groups.sort(key=lambda item: (-item[1], _tie_break(seed, item[0]), item[0]))

    shards: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    shard_identities: list[list[str]] = [[] for _ in range(shard_count)]
    shard_seconds = [0.0] * shard_count
    for identity, duration, rows in groups:
        index = min(
            range(shard_count),
            key=lambda candidate: (
                shard_seconds[candidate],
                len(shard_identities[candidate]),
                candidate,
            ),
        )
        shards[index].extend(rows)
        shard_identities[index].append(identity)
        shard_seconds[index] += duration

    parent_binding = _binding(source_manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    files = []
    try:
        for index, rows in enumerate(shards):
            rows.sort(key=_source_id)
            split_counts = {
                split: sum(row["split"] == split for row in rows)
                for split in ("train", "validation")
            }
            payload = {
                "schema_version": 1,
                "kind": "kizz_development_negative_manifest_shard",
                "source_manifest": parent_binding,
                "parent_manifest_kind": manifest.get("kind"),
                "parent_manifest_schema_version": manifest.get("schema_version"),
                "shard": {
                    "index": index,
                    "count": shard_count,
                    "seed": seed,
                    "assignment_policy": ASSIGNMENT_POLICY,
                    "identity_fields_precedence": list(IDENTITY_FIELDS),
                    "identity_count": len(shard_identities[index]),
                    "identity_set_sha256": hashlib.sha256(
                        "\n".join(sorted(shard_identities[index])).encode("utf-8")
                    ).hexdigest(),
                },
                "counts": {
                    "examples": len(rows),
                    "by_split": split_counts,
                    "duration_seconds": shard_seconds[index],
                    "excluded_parent_examples": excluded,
                },
                "examples": rows,
            }
            path = temporary / f"shard-{index:03d}-of-{shard_count:03d}.json"
            _atomic_json(path, payload)
            files.append(
                {
                    "index": index,
                    "path": path.name,
                    "sha256": sha256_file(path),
                    "examples": len(rows),
                    "identities": len(shard_identities[index]),
                    "duration_seconds": shard_seconds[index],
                }
            )
        index_payload = {
            "schema_version": 1,
            "kind": "kizz_development_negative_manifest_shard_index",
            "source_manifest": parent_binding,
            "shard_count": shard_count,
            "seed": seed,
            "assignment_policy": ASSIGNMENT_POLICY,
            "eligible_examples": len(eligible),
            "eligible_identities": len(grouped),
            "excluded_parent_examples": excluded,
            "duration_seconds": sum(shard_seconds),
            "files": files,
        }
        _atomic_json(temporary / "index.json", index_payload)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {output}")
        os.rename(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {
        "output": str(output),
        "index_sha256": sha256_file(output / "index.json"),
        "shards": shard_count,
        "eligible_examples": len(eligible),
        "eligible_identities": len(grouped),
        "duration_seconds": sum(shard_seconds),
        "minimum_shard_seconds": min(shard_seconds),
        "maximum_shard_seconds": max(shard_seconds),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--seed", default="kizz-negative-manifest-shard-v1")
    args = parser.parse_args(argv)
    print(json.dumps(shard_manifest(**vars(args)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
