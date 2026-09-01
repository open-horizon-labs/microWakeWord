#!/usr/bin/env python3
"""Generate a locked, voice-disjoint Kokoro qualification set for Kizz Control."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Sequence

from tools.generate_kizz_control_c1_corpus import (
    KOKORO_VOICES,
    KokoroWorker,
    Task,
    _sha256_file,
    _wav_duration,
)
from tools.generate_kizz_kokoro_phoneme_v3 import ENGLISH_VOICES, model_sha256


MODEL_ID = "hexgrad/Kokoro-82M"
VARIANTS = (
    ("Kiz Control", 0.75),
    ("Kiz Control", 0.80),
    ("Kizz Control", 0.90),
    ("Kizz Control", 1.10),
    ("Kiz Control", 0.85),
    ("Kiz Control", 0.90),
    ("Kiz Control", 0.95),
    ("Kiz Control", 1.00),
    ("Kiz Control", 1.05),
    ("Kiz Control", 1.10),
    ("Kiz Control", 1.15),
    ("Kiz Control", 1.20),
    ("Kiz Control", 1.25),
    ("Kizz Control", 0.80),
    ("Kizz Control", 1.00),
    ("Kizz Control", 1.20),
    ("Kizz, control", 0.90),
    ("Kizz, control", 1.00),
    ("Kizz, control", 1.10),
)
CANONICAL_KIZZ_VOICES = frozenset(voice for voice, _ in KOKORO_VOICES)
FRESH_VOICES = tuple(voice for voice in ENGLISH_VOICES if voice not in CANONICAL_KIZZ_VOICES)


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


def generate(output: Path, kokoro_python: Path, model_path: Path) -> dict:
    output = output.expanduser().resolve()
    audio_root = output / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    model_path = model_path.expanduser().resolve()
    if set(FRESH_VOICES) & CANONICAL_KIZZ_VOICES or len(FRESH_VOICES) != 12:
        raise ValueError("fresh Kokoro voice-disjoint contract drift")

    # Preserve the virtualenv launcher symlink. Resolving it selects the base
    # interpreter and drops the Kokoro environment's site-packages.
    worker = KokoroWorker(kokoro_python.expanduser())
    rows = []
    try:
        for voice in FRESH_VOICES:
            for variant_index, (render_text, speed) in enumerate(VARIANTS):
                task = Task(
                    provider="kokoro",
                    model=MODEL_ID,
                    voice=voice,
                    provider_voice_id=voice,
                    split="test",
                    label=1,
                    text=render_text,
                    variant_index=variant_index,
                    settings={"speed": speed},
                )
                path = audio_root / f"{task.descriptor[:24]}.wav"
                if not path.is_file():
                    worker.render(task, path)
                audio_hash = _sha256_file(path)
                rows.append(
                    {
                        "source_id": f"kizz-control-fresh-kokoro:{task.descriptor}",
                        "descriptor_sha256": task.descriptor,
                        "path": str(path),
                        "audio_sha256": audio_hash,
                        "duration_seconds": _wav_duration(path),
                        "provider": "kokoro",
                        "model": MODEL_ID,
                        "voice": voice,
                        "voice_id": f"kokoro:{voice}",
                        "provider_voice_id": voice,
                        "speaker_id": f"tts:kokoro:{voice}",
                        "session_id": f"tts:kokoro:{voice}:fresh-final-qualification-v1",
                        "ancestry_id": f"tts-ancestry:{task.descriptor}",
                        "source_group": "kokoro_fresh_final_qualification",
                        "split": "test",
                        "label": 1,
                        "semantic_label": "canonical_exact",
                        "render_text": render_text,
                        "settings": {"speed": speed},
                        "training_eligible": False,
                        "exclusion_reason": "candidate_for_fresh_final_device_qualification",
                        "locked_before_scoring": True,
                    }
                )
    finally:
        worker.close()

    hashes = [row["audio_sha256"] for row in rows]
    expected_count = len(FRESH_VOICES) * len(VARIANTS)
    if len(rows) != expected_count or len(set(hashes)) != len(rows):
        raise ValueError("fresh qualification set is incomplete or contains duplicate audio")
    payload = {
        "schema_version": 1,
        "corpus_id": "kizz-control-fresh-kokoro-qualification-candidates-v2",
        "purpose": "fresh_target_channel_positive_candidate_inventory",
        "locked_before_scoring": True,
        "training_eligible": False,
        "model": {
            "id": MODEL_ID,
            "path": str(model_path),
            "sha256": model_sha256(model_path),
        },
        "voice_disjoint_from": {
            "canonical_kizz_control_voices": sorted(CANONICAL_KIZZ_VOICES),
            "fresh_voices": list(FRESH_VOICES),
            "overlap": [],
        },
        "candidate_evidence_contract": {
            "locked_before_detector_scoring": True,
            "total_count": expected_count,
            "providers": {"kokoro": {"count": expected_count, "voices": len(FRESH_VOICES)}},
        },
        "examples": sorted(rows, key=lambda row: row["source_id"]),
    }
    manifest = output / "manifest.json"
    _atomic_json(manifest, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kokoro-python", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = generate(args.output, args.kokoro_python, args.model_path)
    print(json.dumps({"examples": len(payload["examples"]), "voices": len(FRESH_VOICES)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
