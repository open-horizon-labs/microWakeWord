#!/usr/bin/env python3
"""Lock pronunciation-qualified fresh sources before detector scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Sequence


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def curate(candidate_manifest: Path, pronunciation_audit: Path, output: Path, count: int, minimum_voices: int) -> dict:
    candidate_manifest = candidate_manifest.expanduser().resolve()
    pronunciation_audit = pronunciation_audit.expanduser().resolve()
    candidates = json.loads(candidate_manifest.read_text(encoding="utf-8"))
    audit = json.loads(pronunciation_audit.read_text(encoding="utf-8"))
    if (
        candidates.get("purpose") != "fresh_target_channel_positive_candidate_inventory"
        or candidates.get("locked_before_scoring") is not True
        or candidates.get("training_eligible") is not False
    ):
        raise ValueError("candidate manifest is not locked fresh evidence")
    if (
        audit.get("gate_scope") != "independent_source_pronunciation_qc"
        or audit.get("source_manifest_sha256") != sha256_file(candidate_manifest)
        or audit.get("scope", {}).get("gate_mode") != "all"
        or audit.get("locked_before_device_capture") is not True
    ):
        raise ValueError("pronunciation audit does not bind the candidate inventory")

    accepted_ids = {
        str(row["source_id"])
        for row in audit.get("results", [])
        if row.get("accepted") is True
    }
    by_voice: dict[str, list[dict]] = defaultdict(list)
    for raw in candidates.get("examples", []):
        row = dict(raw)
        path = Path(row.get("path", "")).resolve()
        if sha256_file(path) != row.get("audio_sha256"):
            raise ValueError(f"candidate source hash drift: {path}")
        if row.get("source_id") in accepted_ids:
            by_voice[str(row["voice"])].append(row)
    for rows in by_voice.values():
        rows.sort(key=lambda row: (str(row.get("render_text")), json.dumps(row.get("settings", {}), sort_keys=True), str(row["source_id"])))
    if len(by_voice) < minimum_voices:
        raise ValueError(f"only {len(by_voice)} pronunciation-qualified voices; need {minimum_voices}")

    selected: list[dict] = []
    positions = {voice: 0 for voice in sorted(by_voice)}
    while len(selected) < count:
        progressed = False
        for voice in sorted(by_voice):
            position = positions[voice]
            if position < len(by_voice[voice]):
                selected.append(dict(by_voice[voice][position]))
                positions[voice] += 1
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    if len(selected) != count:
        raise ValueError(f"only {len(selected)} pronunciation-qualified clips; need {count}")
    selected_voices = sorted({str(row["voice"]) for row in selected})
    if len(selected_voices) < minimum_voices:
        raise ValueError("selected evidence does not realize the minimum voice count")
    for row in selected:
        row["reserved_evidence_role"] = "target_channel_positive"
        row["exclusion_reason"] = "reserved_for_fresh_final_device_qualification"
        row["locked_before_scoring"] = True
        row["training_eligible"] = False

    payload = {
        "schema_version": 1,
        "corpus_id": "kizz-control-fresh-kokoro-final-qualification-v2",
        "purpose": "fresh_target_channel_positive_qualification",
        "locked_before_scoring": True,
        "training_eligible": False,
        "candidate_manifest": {"path": str(candidate_manifest), "sha256": sha256_file(candidate_manifest)},
        "pronunciation_audit": {"path": str(pronunciation_audit), "sha256": sha256_file(pronunciation_audit)},
        "reserved_evidence_contract": {
            "role": "target_channel_positive",
            "locked_before_scoring": True,
            "total_count": count,
            "providers": {"kokoro": {"count": count, "minimum_voices": minimum_voices}},
        },
        "selected_voices": selected_voices,
        "examples": sorted(selected, key=lambda row: str(row["source_id"])),
    }
    _atomic_json(output.expanduser().resolve(), payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--pronunciation-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--minimum-voices", type=int, default=8)
    args = parser.parse_args(argv)
    if args.count < 1 or not 1 <= args.minimum_voices <= args.count:
        parser.error("invalid count or minimum voice requirement")
    payload = curate(args.candidate_manifest, args.pronunciation_audit, args.output, args.count, args.minimum_voices)
    print(json.dumps({"examples": len(payload["examples"]), "voices": len(payload["selected_voices"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
