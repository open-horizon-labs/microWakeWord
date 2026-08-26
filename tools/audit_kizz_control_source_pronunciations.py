#!/usr/bin/env python3
"""Independently verify that reserved TTS evidence says /k ɪ z k.../.

This gate uses the pinned English Allosaurus phone recognizer, not the
Wav2Vec2 phoneme teacher being qualified.  It prevents misspelled/nonword TTS
renders such as ``kiss control``, ``kids control``, or ``his control`` from
being mislabeled as positive device evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Sequence


MODEL_NAME = "eng2102"
CANONICAL_PREFIX = ("k", "ɪ", "z", "k")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"Allosaurus model directory is empty: {path}")
    for item in files:
        digest.update(str(item.relative_to(path)).encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def has_canonical_prefix(phones: str) -> bool:
    return tuple(phones.split()[: len(CANONICAL_PREFIX)]) == CANONICAL_PREFIX


def build_report(
    source_manifest: Path,
    output: Path,
    *,
    splits: Sequence[str] = ("test",),
    gate_mode: str = "reserved",
) -> dict:
    from allosaurus import app as allosaurus_app

    requested_splits = tuple(dict.fromkeys(splits))
    if not requested_splits or any(
        split not in {"train", "validation", "test"}
        for split in requested_splits
    ):
        raise ValueError("pronunciation audit splits must be train/validation/test")
    if gate_mode not in {"reserved", "all"}:
        raise ValueError("pronunciation gate mode must be reserved or all")
    payload = json.loads(source_manifest.read_text())
    rows = [
        dict(row)
        for row in payload.get("examples", [])
        if int(row.get("label", -1)) == 1
        and row.get("split") in requested_splits
        and row.get("provider")
        in {"assemblyai", "deepgram", "elevenlabs", "kokoro"}
    ]
    if not rows:
        raise ValueError("source manifest has no runtime-provider test positives")
    recognizer = allosaurus_app.read_recognizer(MODEL_NAME)
    model_path = Path(allosaurus_app.__file__).resolve().parent / "pretrained" / MODEL_NAME
    results = []
    for index, row in enumerate(sorted(rows, key=lambda item: item["source_id"])):
        path = Path(row["path"]).resolve()
        if sha256_file(path) != row.get("audio_sha256"):
            raise ValueError(f"source audio hash drift: {path}")
        phones = recognizer.recognize(str(path), "eng")
        results.append(
            {
                "source_id": row["source_id"],
                "audio_sha256": row["audio_sha256"],
                "provider": row["provider"],
                "voice": row["voice"],
                "render_text": row["render_text"],
                "reserved": row.get("reserved_evidence_role")
                == "target_channel_positive",
                "phones": phones,
                "accepted": has_canonical_prefix(phones),
            }
        )
        if (index + 1) % 24 == 0 or index + 1 == len(rows):
            print(json.dumps({"audited": index + 1, "total": len(rows)}), flush=True)
    reserved = [item for item in results if item["reserved"]]
    gated = reserved if gate_mode == "reserved" else results
    reserved_failures = [item for item in reserved if not item["accepted"]]
    failures = [item for item in gated if not item["accepted"]]
    providers = {
        provider: {
            "count": sum(item["provider"] == provider for item in reserved),
            "voices": sorted(
                {item["voice"] for item in reserved if item["provider"] == provider}
            ),
        }
        for provider in ("assemblyai", "deepgram", "elevenlabs", "kokoro")
    }
    report = {
        "schema_version": 1,
        "gate_scope": "independent_source_pronunciation_qc",
        "qualified": (
            len(reserved) == 24 and not failures
            if gate_mode == "reserved"
            else bool(gated) and not failures
        ),
        "locked_before_device_capture": True,
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": sha256_file(source_manifest),
        "model": {
            "name": MODEL_NAME,
            "package": "allosaurus",
            "package_version": importlib.metadata.version("allosaurus"),
            "model_tree_sha256": sha256_tree(model_path),
        },
        "decision": {
            "required_prefix": list(CANONICAL_PREFIX),
            "rule": "first four recognized English phones equal the canonical prefix",
        },
        "scope": {
            "splits": list(requested_splits),
            "gate_mode": gate_mode,
        },
        "counts": {
            "audited": len(results),
            "accepted": sum(item["accepted"] for item in results),
            "reserved": len(reserved),
            "reserved_rejected": len(reserved_failures),
            "gated": len(gated),
            "gated_rejected": len(failures),
        },
        "reserved_provider_contract": providers,
        "reserved_audio_sha256": [item["audio_sha256"] for item in reserved],
        "reserved_failures": reserved_failures,
        "gated_failures": failures,
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split",
        action="append",
        choices=("train", "validation", "test"),
        help="positive split to audit (repeatable; default: test)",
    )
    parser.add_argument(
        "--gate-mode",
        choices=("reserved", "all"),
        default="reserved",
        help="qualify reserved anchors or every audited positive",
    )
    args = parser.parse_args(argv)
    report = build_report(
        args.source_manifest,
        args.output,
        splits=args.split or ("test",),
        gate_mode=args.gate_mode,
    )
    print(json.dumps({"qualified": report["qualified"], "counts": report["counts"]}, sort_keys=True))
    return 0 if report["qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
