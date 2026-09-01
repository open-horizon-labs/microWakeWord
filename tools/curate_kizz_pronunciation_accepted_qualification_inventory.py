#!/usr/bin/env python3
"""Compose a balanced, pronunciation-accepted final positive inventory.

Selection uses only source provenance and an independent Allosaurus audit. It
never reads detector or verifier scores, and recursively excludes previously
consumed or reserved source/audio hashes before locking the result.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.prepare_kizz_reserved_multisource_qualification_inventory import (
    _atomic_no_replace_json,
    _binding,
    _relevant_hashes,
    sha256_file,
)


def _load(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{resolved}: expected JSON object")
    return payload


def _rank(row: Mapping[str, Any]) -> tuple[str, str]:
    material = {
        "provider": row.get("provider"),
        "voice": row.get("inventory_voice_id"),
        "source_id": row.get("source_id"),
        "audio_sha256": row.get("audio_sha256"),
    }
    digest = hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return digest, str(row.get("source_id", ""))


def _select_voice_round_robin(rows: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    by_voice: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_voice[str(row["inventory_voice_id"])].append(row)
    for candidates in by_voice.values():
        candidates.sort(key=_rank)
    voices = sorted(by_voice)
    positions = {voice: 0 for voice in voices}
    selected: list[dict[str, Any]] = []
    cursor = 0
    while len(selected) < count:
        progressed = False
        for _ in range(len(voices)):
            voice = voices[cursor % len(voices)]
            cursor += 1
            position = positions[voice]
            if position >= len(by_voice[voice]):
                continue
            selected.append(copy.deepcopy(by_voice[voice][position]))
            positions[voice] += 1
            progressed = True
            break
        if not progressed:
            break
    return selected


def curate(
    source_audits: Sequence[tuple[Path, Path]],
    exclusion_manifests: Sequence[Path],
    provider_counts: Mapping[str, int],
    output: Path,
) -> dict[str, Any]:
    if not source_audits:
        raise ValueError("at least one source/audit pair is required")
    if not provider_counts or any(not provider or count < 1 for provider, count in provider_counts.items()):
        raise ValueError("provider counts must be nonempty positive integers")
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)

    excluded: set[str] = set()
    exclusion_bindings = []
    for path in exclusion_manifests:
        path = path.expanduser().resolve()
        payload = _load(path)
        excluded.update(_relevant_hashes(payload))
        exclusion_bindings.append(_binding(path))

    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    input_pairs = []
    for source_path, audit_path in source_audits:
        source_path = source_path.expanduser().resolve()
        audit_path = audit_path.expanduser().resolve()
        source = _load(source_path)
        audit = _load(audit_path)
        if audit.get("gate_scope") != "independent_source_pronunciation_qc":
            raise ValueError(f"{audit_path}: not an independent pronunciation audit")
        if audit.get("source_manifest_sha256") != sha256_file(source_path):
            raise ValueError(f"{audit_path}: source binding drift")
        results = audit.get("results")
        rows = source.get("examples")
        if not isinstance(results, list) or not isinstance(rows, list):
            raise ValueError("source/audit pair requires examples and results lists")
        accepted = {
            (str(result.get("source_id", "")), str(result.get("audio_sha256", "")))
            for result in results
            if result.get("accepted") is True
        }
        input_pairs.append({"source_manifest": _binding(source_path), "pronunciation_audit": _binding(audit_path)})
        for raw in rows:
            if not isinstance(raw, dict):
                raise ValueError(f"{source_path}: every example must be an object")
            source_id = str(raw.get("source_id", ""))
            audio_hash = str(raw.get("audio_sha256", ""))
            identity = (source_id, audio_hash)
            if not source_id or len(audio_hash) != 64 or identity not in accepted:
                continue
            provider = str(raw.get("provider", ""))
            if provider not in provider_counts or _relevant_hashes(raw) & excluded:
                continue
            path = Path(str(raw.get("path", ""))).expanduser()
            if not path.is_absolute():
                path = source_path.parent / path
            path = path.resolve()
            if not path.is_file() or sha256_file(path) != audio_hash:
                raise ValueError(f"{source_id}: source audio hash drift")
            voice = raw.get("voice_id") or raw.get("provider_voice_id") or raw.get("voice") or raw.get("speaker_id")
            if not isinstance(voice, str) or not voice:
                raise ValueError(f"{source_id}: missing voice identity")
            row = copy.deepcopy(raw)
            row["path"] = str(path)
            row["inventory_voice_id"] = f"{provider}:{voice}"
            previous = candidates.get(identity)
            if previous is not None and previous != row:
                raise ValueError(f"{source_id}: conflicting duplicate source identity")
            candidates[identity] = row

    selected: list[dict[str, Any]] = []
    realized: dict[str, dict[str, int]] = {}
    for provider, count in sorted(provider_counts.items()):
        pool = [row for row in candidates.values() if row.get("provider") == provider]
        chosen = _select_voice_round_robin(pool, count)
        if len(chosen) != count:
            raise ValueError(f"{provider}: only {len(chosen)} accepted unconsumed candidates; need {count}")
        selected.extend(chosen)
        realized[provider] = {
            "count": count,
            "minimum_voices": len({str(row["inventory_voice_id"]) for row in chosen}),
        }
    selected.sort(key=lambda row: (str(row["provider"]), _rank(row)))
    for index, row in enumerate(selected):
        row["candidate_inventory_selection_index"] = index
        row["reserved_evidence_role"] = "target_channel_positive"
        row["evidence_status"] = "reserved"
        row["exclusion_reason"] = "reserved_for_fresh_final_device_qualification"
        row["locked_before_scoring"] = True
        row["training_eligible"] = False

    payload = {
        "schema_version": 1,
        "corpus_id": "kizz-control-pronunciation-accepted-balanced-qualification-inventory-v1",
        "purpose": "fresh_target_channel_positive_candidate_inventory",
        "locked_before_scoring": True,
        "training_eligible": False,
        "inputs": {
            "source_audit_pairs": input_pairs,
            "exclude_manifests": exclusion_bindings,
        },
        "selection_policy": {
            "name": "pronunciation_accepted_provider_then_voice_round_robin_v1",
            "independent_pronunciation_acceptance_required": True,
            "recursive_consumed_hash_exclusion": True,
            "model_scores_read_or_used": False,
        },
        "counts": {
            "selected": len(selected),
            "providers": len(realized),
            "voices": len({str(row["inventory_voice_id"]) for row in selected}),
        },
        "reserved_evidence_contract": {
            "role": "target_channel_positive",
            "status": "reserved",
            "locked_before_scoring": True,
            "training_eligible": False,
            "total_count": len(selected),
            "providers": realized,
        },
        "examples": selected,
    }
    _atomic_no_replace_json(output, payload)
    return payload


def _pair(value: str) -> tuple[Path, Path]:
    parts = value.split("::", 1)
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError("source/audit pair must be SOURCE::AUDIT")
    return Path(parts[0]), Path(parts[1])


def _provider_count(value: str) -> tuple[str, int]:
    parts = value.split("=", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("provider count must be PROVIDER=COUNT")
    try:
        count = int(parts[1])
    except ValueError as error:
        raise argparse.ArgumentTypeError("provider count must be an integer") from error
    return parts[0], count


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audit", action="append", type=_pair, required=True)
    parser.add_argument("--exclude-manifest", action="append", type=Path, default=[])
    parser.add_argument("--provider-count", action="append", type=_provider_count, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    counts = dict(args.provider_count)
    if len(counts) != len(args.provider_count):
        parser.error("provider counts must be unique")
    payload = curate(args.source_audit, args.exclude_manifest, counts, args.output)
    print(json.dumps({"output": str(args.output), "counts": payload["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
