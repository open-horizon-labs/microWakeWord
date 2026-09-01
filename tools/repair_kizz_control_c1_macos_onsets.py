#!/usr/bin/env python3
"""Repair clipped macOS ``say`` onsets with deterministic leading context.

The source renders remain immutable.  Every macOS row is copied to a derived
PCM16 WAV with 200 ms of digital silence prepended, preserving the original
audio bytes after that prefix.  The derived manifest records the parent path,
parent hash, transform, and output hash so pronunciation qualification can bind
to the repaired material without losing provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import wave
from pathlib import Path
from typing import Any, Sequence


SAMPLE_RATE = 16_000
DEFAULT_LEADING_CONTEXT_MS = 200


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repair_wav(source: Path, output: Path, leading_context_ms: int) -> int:
    with wave.open(str(source), "rb") as wav:
        if (
            wav.getframerate() != SAMPLE_RATE
            or wav.getnchannels() != 1
            or wav.getsampwidth() != 2
            or wav.getcomptype() != "NONE"
        ):
            raise ValueError(f"{source}: expected mono 16 kHz PCM16 WAV")
        params = wav.getparams()
        frames = wav.readframes(wav.getnframes())
    prefix_samples = SAMPLE_RATE * leading_context_ms // 1000
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as wav:
        wav.setparams(params._replace(nframes=0))
        wav.writeframes(b"\0\0" * prefix_samples + frames)
    return prefix_samples


def repair(
    source_manifest: Path,
    audio_root: Path,
    output_manifest: Path,
    *,
    leading_context_ms: int = DEFAULT_LEADING_CONTEXT_MS,
) -> dict[str, Any]:
    if leading_context_ms < 1 or SAMPLE_RATE * leading_context_ms % 1000:
        raise ValueError("leading context must map to an integral positive sample count")
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or not isinstance(source.get("examples"), list):
        raise ValueError("source manifest must contain examples")
    rows = [dict(row) for row in source["examples"]]
    repaired = 0
    for row in rows:
        if row.get("provider") != "macos-say":
            continue
        parent = Path(str(row.get("path", ""))).resolve()
        parent_hash = str(row.get("audio_sha256", ""))
        if not parent.is_file() or not parent_hash or sha256_file(parent) != parent_hash:
            raise ValueError(f"macOS source audio drift: {parent}")
        split = str(row.get("split", "unknown"))
        identity = hashlib.sha256(
            (
                f"{row.get('source_id')}\0{parent_hash}\0"
                f"leading-silence-ms={leading_context_ms}"
            ).encode()
        ).hexdigest()
        output = (audio_root / "macos-say" / split / f"{identity[:24]}.wav").resolve()
        prefix_samples = _repair_wav(parent, output, leading_context_ms)
        output_hash = sha256_file(output)
        row["parent_path"] = str(parent)
        row["parent_audio_sha256"] = parent_hash
        row["path"] = str(output)
        row["audio_sha256"] = output_hash
        row["provenance_id"] = f"audio-sha256:{output_hash}"
        row["duration_seconds"] = float(row.get("duration_seconds", 0.0)) + (
            prefix_samples / SAMPLE_RATE
        )
        row["audio_repair"] = {
            "kind": "leading_digital_silence",
            "leading_context_ms": leading_context_ms,
            "leading_context_samples": prefix_samples,
            "sample_rate_hz": SAMPLE_RATE,
            "preserves_parent_pcm_after_prefix": True,
        }
        repaired += 1
    payload = dict(source)
    payload.update(
        {
            "schema_version": 3,
            "recipe": "kizz_control_c1_macos_onset_repair",
            "inputs": {
                "source_manifest": str(source_manifest.resolve()),
                "source_manifest_sha256": sha256_file(source_manifest),
            },
            "repair": {
                "provider": "macos-say",
                "leading_context_ms": leading_context_ms,
                "repaired_examples": repaired,
                "raw_source_mutated": False,
            },
            "examples": rows,
        }
    )
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument(
        "--leading-context-ms", type=int, default=DEFAULT_LEADING_CONTEXT_MS
    )
    args = parser.parse_args(argv)
    payload = repair(
        args.source_manifest,
        args.audio_root,
        args.output_manifest,
        leading_context_ms=args.leading_context_ms,
    )
    print(json.dumps(payload["repair"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
