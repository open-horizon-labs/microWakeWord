#!/usr/bin/env python3
"""Reserve unused pronunciation-accepted test renders for fresh device testing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.mine_kizz_librispeech_hard_negatives import _binding, sha256_file


PROVIDERS = ("assemblyai", "deepgram", "elevenlabs", "kokoro")


def _load(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _voice(row: dict[str, Any]) -> str:
    voice = str(row.get("voice_id") or row.get("voice") or "")
    if not voice:
        raise ValueError("qualification source lacks voice identity")
    return voice


def select_fresh_rows(
    source: dict[str, Any],
    audit: dict[str, Any],
    excluded_source_hashes: set[str],
    *,
    per_provider: int,
) -> list[dict[str, Any]]:
    if per_provider < 1:
        raise ValueError("per_provider must be positive")
    accepted = {
        str(row.get("audio_sha256"))
        for row in audit.get("results", [])
        if row.get("accepted") is True
    }
    selected: list[dict[str, Any]] = []
    examples = source.get("examples")
    if not isinstance(examples, list):
        raise ValueError("source manifest examples are missing")
    for provider in PROVIDERS:
        by_voice: dict[str, list[dict[str, Any]]] = {}
        for raw in examples:
            if not isinstance(raw, dict):
                continue
            audio_hash = str(raw.get("audio_sha256", ""))
            if (
                raw.get("provider") != provider
                or raw.get("split") != "test"
                or int(raw.get("label", -1)) != 1
                or raw.get("target_id") != "kizz-control"
                or audio_hash not in accepted
                or audio_hash in excluded_source_hashes
            ):
                continue
            by_voice.setdefault(_voice(raw), []).append(dict(raw))
        for rows in by_voice.values():
            rows.sort(
                key=lambda row: (
                    str(row.get("render_text", "")),
                    str(row.get("audio_sha256", "")),
                )
            )
        voices = sorted(by_voice)
        available = sum(len(rows) for rows in by_voice.values())
        if available < per_provider:
            raise ValueError(
                f"provider {provider} has only {available} unused accepted test renders"
            )
        positions = {voice: 0 for voice in voices}
        provider_rows: list[dict[str, Any]] = []
        while len(provider_rows) < per_provider:
            for voice in voices:
                position = positions[voice]
                if position >= len(by_voice[voice]):
                    continue
                row = dict(by_voice[voice][position])
                positions[voice] += 1
                row["training_eligible"] = False
                row["reserved_evidence_role"] = "target_channel_positive"
                row["fresh_qualification_holdout"] = True
                provider_rows.append(row)
                if len(provider_rows) == per_provider:
                    break
        selected.extend(provider_rows)
    hashes = [str(row.get("audio_sha256", "")) for row in selected]
    if any(not value for value in hashes) or len(hashes) != len(set(hashes)):
        raise ValueError("fresh qualification selection has missing/duplicate audio")
    return selected


def prepare(
    source_manifest: Path,
    pronunciation_audit: Path,
    exclude_evidence: Path,
    *,
    per_provider: int,
) -> dict[str, Any]:
    source_manifest = source_manifest.expanduser().resolve()
    pronunciation_audit = pronunciation_audit.expanduser().resolve()
    exclude_evidence = exclude_evidence.expanduser().resolve()
    source = _load(source_manifest, "source manifest")
    audit = _load(pronunciation_audit, "pronunciation audit")
    excluded = _load(exclude_evidence, "excluded device evidence")
    if (
        audit.get("gate_scope") != "independent_source_pronunciation_qc"
        or audit.get("qualified") is not True
        or audit.get("locked_before_device_capture") is not True
        or audit.get("source_manifest_sha256") != sha256_file(source_manifest)
    ):
        raise ValueError("pronunciation audit is not qualified and source-bound")
    excluded_examples = excluded.get("examples")
    if not isinstance(excluded_examples, list):
        raise ValueError("excluded device evidence examples are missing")
    excluded_hashes = {
        str(row.get("source_audio_sha256", "")) for row in excluded_examples
    }
    rows = select_fresh_rows(
        source, audit, excluded_hashes, per_provider=per_provider
    )
    payload = {
        "schema_version": 1,
        "kind": "kizz_control_fresh_device_qualification_source_inventory",
        "selection_algorithm": "unused_pronunciation_accepted_test_voice_round_robin_v1",
        "locked_before_model_scoring": True,
        "training_eligible": False,
        "per_provider": per_provider,
        "selected_count": len(rows),
        "bindings": {
            "source_manifest": _binding(source_manifest),
            "pronunciation_audit": _binding(pronunciation_audit),
            "excluded_prior_device_evidence": _binding(exclude_evidence),
        },
        "excluded_source_audio_sha256": sorted(excluded_hashes),
        "examples": rows,
    }
    payload["selection_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def derive_pronunciation_audit(
    inventory_path: Path,
    inventory: dict[str, Any],
    parent_audit_path: Path,
) -> dict[str, Any]:
    parent_audit_path = parent_audit_path.expanduser().resolve()
    parent = _load(parent_audit_path, "parent pronunciation audit")
    selected_hashes = {
        str(row["audio_sha256"]) for row in inventory.get("examples", [])
    }
    parent_results = {
        str(row.get("audio_sha256")): dict(row)
        for row in parent.get("results", [])
        if row.get("accepted") is True
    }
    if not selected_hashes or not selected_hashes.issubset(parent_results):
        raise ValueError("fresh inventory is not covered by the parent accepted audit")
    results = [parent_results[digest] for digest in sorted(selected_hashes)]
    return {
        "schema_version": 1,
        "gate_scope": "independent_source_pronunciation_qc",
        "scope": "fresh_device_qualification_sources_only",
        "qualified": True,
        "locked_before_device_capture": True,
        "source_manifest": str(inventory_path.resolve()),
        "source_manifest_sha256": sha256_file(inventory_path),
        "reserved_audio_sha256": sorted(selected_hashes),
        "counts": {
            "audited": len(results),
            "accepted": len(results),
            "reserved": len(results),
            "reserved_rejected": 0,
        },
        "results": results,
        "parent_audit": _binding(parent_audit_path),
        "derivation": "exact accepted-hash subset; no pronunciation rescoring",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--pronunciation-audit", type=Path, required=True)
    parser.add_argument("--exclude-evidence", type=Path, required=True)
    parser.add_argument("--per-provider", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--derived-audit-output", type=Path)
    args = parser.parse_args(argv)
    payload = prepare(
        args.source_manifest,
        args.pronunciation_audit,
        args.exclude_evidence,
        per_provider=args.per_provider,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.derived_audit_output is not None:
        derived = derive_pronunciation_audit(
            args.output,
            payload,
            args.pronunciation_audit,
        )
        args.derived_audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.derived_audit_output.write_text(
            json.dumps(derived, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps({"selected": payload["selected_count"], "output": str(args.output.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
