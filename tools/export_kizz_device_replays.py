#!/usr/bin/env python3
"""Export immutable StackChan replay captures as teacher qualification evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from microwakeword.kizz_evaluation_contract import sha256_file, validate_audio_rows


def export_rows(corpus: Path) -> list[dict]:
    manifest_path = corpus / "device-corpus.json"
    payload = json.loads(manifest_path.read_text())
    rows = []
    for capture in payload.get("captures", []):
        conditions = capture.get("conditions", {})
        if (
            capture.get("truth") != "positive"
            or capture.get("split") != "test"
            or conditions.get("evidence_role")
            != "reserved_target_channel_positive"
        ):
            continue
        audio_path = (corpus / capture["path"]).resolve()
        source_hash = conditions.get("source_audio_sha256")
        source_descriptor = conditions.get("source_descriptor_sha256")
        if not source_hash or not source_descriptor:
            raise ValueError(
                f"capture {capture.get('capture_id')} lacks replay provenance"
            )
        rows.append(
            {
                "source_id": f"device-replay:{capture['capture_id']}",
                "provenance_id": f"device-replay:{capture['capture_id']}",
                "parent_source_id": f"descriptor-sha256:{source_descriptor}",
                "path": str(audio_path),
                "audio_sha256": capture["sha256"],
                "source_audio_sha256": source_hash,
                "label": 1,
                "split": "test",
                "duration_seconds": float(capture["samples"]) / 16_000.0,
                "speaker_id": capture["speaker_id"],
                "session_id": capture["session_id"],
                "phrase": capture["phrase"],
                "device_id": capture["device_id"],
                "device_profile": capture["device_profile"],
                "firmware_sha": capture.get("firmware_sha"),
                "conditions": conditions,
                "training_eligible": False,
                "evidence_role": "natural_positive",
            }
        )
    rows.sort(key=lambda row: row["source_id"])
    validate_audio_rows(rows, group="device_replay_positive")
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--selected-evidence-manifest", type=Path, required=True)
    parser.add_argument("--source-pronunciation-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = export_rows(args.corpus.resolve())
    selection = json.loads(args.selected_evidence_manifest.read_text())
    audit = json.loads(args.source_pronunciation_audit.read_text())
    selected_hashes = set(selection.get("selected_audio_sha256", []))
    captured_hashes = {str(row["source_audio_sha256"]) for row in rows}
    if (
        selection.get("locked_before_teacher_scoring") is not True
        or selection.get("selected_count") != 24
        or selected_hashes != captured_hashes
    ):
        raise ValueError("device captures do not exactly realize the locked 24-source selection")
    if (
        audit.get("gate_scope") != "independent_source_pronunciation_qc"
        or audit.get("qualified") is not True
        or set(audit.get("reserved_audio_sha256", [])) != selected_hashes
        or selection.get("source_pronunciation_audit", {}).get("sha256")
        != sha256_file(args.source_pronunciation_audit)
    ):
        raise ValueError("device captures are not bound to the qualified pronunciation audit")
    provider_counts: dict[str, int] = {}
    voices: dict[str, set[str]] = {}
    for row in rows:
        provider = str(row["conditions"]["source_provider"])
        provider_counts[provider] = provider_counts.get(provider, 0) + 1
        voices.setdefault(provider, set()).add(str(row["conditions"]["source_voice"]))
    if set(provider_counts.values()) != {6} or len(provider_counts) != 4:
        raise ValueError(f"device replay provider balance differs from 6x4: {provider_counts}")
    if any(len(values) < 2 for values in voices.values()):
        raise ValueError("device replay evidence collapsed a provider to one voice")
    payload = {
        "schema_version": 2,
        "kind": "kizz_control_voice_stratified_device_replay_qualification_evidence",
        "source_manifest": str((args.corpus / "device-corpus.json").resolve()),
        "source_manifest_sha256": sha256_file(args.corpus / "device-corpus.json"),
        "selection_manifest": str(args.selected_evidence_manifest.resolve()),
        "selection_manifest_sha256": sha256_file(args.selected_evidence_manifest),
        "source_pronunciation_audit": str(args.source_pronunciation_audit.resolve()),
        "source_pronunciation_audit_sha256": sha256_file(args.source_pronunciation_audit),
        "counts": {
            "total": len(rows),
            "providers": dict(sorted(provider_counts.items())),
            "voices": {key: sorted(value) for key, value in sorted(voices.items())},
        },
        "training_eligible": False,
        "examples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "examples": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
