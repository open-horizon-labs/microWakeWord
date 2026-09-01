#!/usr/bin/env python3
"""Lock fresh sources that pass both pronunciation recognizers before scoring."""

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


def curate(source_manifest: Path, aligned_manifest: Path, output: Path, *, count: int | None, minimum_voices: int, exclude_manifests: Sequence[Path]) -> dict:
    source_manifest = source_manifest.expanduser().resolve()
    aligned_manifest = aligned_manifest.expanduser().resolve()
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    aligned = json.loads(aligned_manifest.read_text(encoding="utf-8"))
    sidecar_path = Path(str(aligned.get("source_manifest", ""))).resolve()
    if (
        not sidecar_path.is_file()
        or aligned.get("source_manifest_sha256") != sha256_file(sidecar_path)
        or aligned.get("target", {}).get("phrase_id") != "kizz-control"
        or aligned.get("aligner", {}).get("backend") != "wav2vec2_ipa_ctc"
    ):
        raise ValueError("aligned manifest is not the pinned Kizz Control acoustic gate")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("bindings", {}).get("source_manifest", {}).get("sha256") != sha256_file(source_manifest):
        raise ValueError("alignment sidecar does not bind the source manifest")

    source_by_id = {str(row["source_id"]): dict(row) for row in source.get("examples", [])}
    excluded_hashes: set[str] = set()
    exclusion_bindings = []
    for path in exclude_manifests:
        path = path.expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        excluded_hashes.update(str(row.get("audio_sha256", "")) for row in payload.get("examples", []))
        exclusion_bindings.append({"path": str(path), "sha256": sha256_file(path)})

    by_voice: dict[str, list[dict]] = defaultdict(list)
    for aligned_row in aligned.get("examples", []):
        row = source_by_id.get(str(aligned_row.get("source_id", "")))
        if row is None or row.get("audio_sha256") != aligned_row.get("audio_sha256"):
            raise ValueError("acoustically aligned row differs from source manifest")
        path = Path(str(row.get("path", ""))).resolve()
        if sha256_file(path) != row.get("audio_sha256"):
            raise ValueError(f"source audio hash drift: {path}")
        if row["audio_sha256"] in excluded_hashes:
            continue
        row["acoustic_alignment_source_id"] = aligned_row["source_id"]
        by_voice[str(row["voice"])].append(row)
    for rows in by_voice.values():
        rows.sort(key=lambda row: (str(row.get("render_text")), json.dumps(row.get("settings", {}), sort_keys=True), str(row["source_id"])))
    available = sum(len(rows) for rows in by_voice.values())
    wanted = available if count is None else count
    if wanted < 1 or wanted > available or len(by_voice) < minimum_voices:
        raise ValueError(f"only {available} aligned clips across {len(by_voice)} voices")

    selected: list[dict] = []
    positions = {voice: 0 for voice in sorted(by_voice)}
    while len(selected) < wanted:
        progressed = False
        for voice in sorted(by_voice):
            position = positions[voice]
            if position < len(by_voice[voice]):
                selected.append(dict(by_voice[voice][position]))
                positions[voice] += 1
                progressed = True
                if len(selected) == wanted:
                    break
        if not progressed:
            break
    selected_voices = sorted({str(row["voice"]) for row in selected})
    if len(selected) != wanted or len(selected_voices) < minimum_voices:
        raise ValueError("aligned selection does not satisfy the requested contract")
    for row in selected:
        row["reserved_evidence_role"] = "target_channel_positive"
        row["exclusion_reason"] = "reserved_for_fresh_final_device_qualification"
        row["locked_before_scoring"] = True
        row["training_eligible"] = False

    payload = {
        "schema_version": 1,
        "corpus_id": "kizz-control-acoustically-aligned-final-qualification-v1",
        "purpose": "fresh_target_channel_positive_qualification",
        "locked_before_scoring": True,
        "training_eligible": False,
        "source_manifest": {"path": str(source_manifest), "sha256": sha256_file(source_manifest)},
        "acoustic_alignment": {"path": str(aligned_manifest), "sha256": sha256_file(aligned_manifest)},
        "excluded_manifests": exclusion_bindings,
        "reserved_evidence_contract": {
            "role": "target_channel_positive",
            "locked_before_scoring": True,
            "total_count": wanted,
            "providers": {"kokoro": {"count": wanted, "minimum_voices": minimum_voices}},
        },
        "selected_voices": selected_voices,
        "examples": sorted(selected, key=lambda row: str(row["source_id"])),
    }
    _atomic_json(output.expanduser().resolve(), payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--aligned-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int)
    parser.add_argument("--minimum-voices", type=int, default=8)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    if args.count is not None and args.count < 1 or args.minimum_voices < 1:
        parser.error("invalid count or minimum voice requirement")
    payload = curate(args.source_manifest, args.aligned_manifest, args.output, count=args.count, minimum_voices=args.minimum_voices, exclude_manifests=args.exclude_manifest)
    print(json.dumps({"examples": len(payload["examples"]), "voices": len(payload["selected_voices"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
