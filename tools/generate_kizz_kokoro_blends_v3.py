#!/usr/bin/env python3
"""Generate train-only Kokoro voice blends for canonical Kizz coverage.

The base Kokoro catalog is voice-disjoint across train, validation, and test.
This augmentation only blends pairs of training voices, so it expands vocal
tract/timbre coverage without allowing a held-out voice into training.  Every
waveform is still acoustically qualified by the independent MMS CTC gate
before it can enter a teacher feature cache.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from tools.generate_kizz_kokoro_phoneme_v3 import (
    MODEL_ID,
    RAW_PHONES,
    SAMPLE_RATE,
    TARGET_ID,
    TARGET_PHONES,
    TEXT,
    VOICE_SPLITS,
    _kokoro_synthesizer,
    _wav_bytes,
    model_sha256,
    normalize_pcm16,
)

SCHEMA_VERSION = "canonical-v3-kokoro-train-blend-1"
TRAIN_VOICES = tuple(
    sorted(voice for voice, split in VOICE_SPLITS.items() if split == "train")
)
DEFAULT_SPEEDS = (0.86, 0.94, 1.02, 1.10, 1.18)


def _speed(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("speed must be finite and positive") from error
    if not math.isfinite(result) or result <= 0:
        raise ValueError("speed must be finite and positive")
    return result


def blend_pairs(voices: Iterable[str] = TRAIN_VOICES) -> tuple[tuple[str, str], ...]:
    selected = tuple(sorted(voices))
    if len(selected) < 2 or len(set(selected)) != len(selected):
        raise ValueError("blend voices must contain at least two unique entries")
    unknown = set(selected) - set(TRAIN_VOICES)
    if unknown:
        raise ValueError(
            f"held-out or unknown voices cannot be blended: {sorted(unknown)}"
        )
    return tuple(itertools.combinations(selected, 2))


def _contract(model_hash: str, speeds: tuple[float, ...]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_id": MODEL_ID,
        "model_sha256": model_hash,
        "raw_phonemes": RAW_PHONES,
        "target_id": TARGET_ID,
        "target_phones": list(TARGET_PHONES),
        "base_voice_split": "train",
        "base_voices": list(TRAIN_VOICES),
        "blend_policy": "all unordered pairs, equal tensor mean",
        "speeds": list(speeds),
        "sample_rate": SAMPLE_RATE,
    }


def generate(
    output_dir: Path,
    manifest_path: Path,
    model_path: Path,
    *,
    speeds: Iterable[float] = DEFAULT_SPEEDS,
    synthesizer: Callable[[str, str, float], tuple[Any, int]] | None = None,
) -> dict[str, Any]:
    selected_speeds = tuple(_speed(value) for value in speeds)
    if not selected_speeds or len(set(selected_speeds)) != len(selected_speeds):
        raise ValueError("speeds must be non-empty and unique")
    model_hash = model_sha256(model_path)
    contract = _contract(model_hash, selected_speeds)
    prior = None
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            prior.get("generator") != Path(__file__).stem
            or prior.get("contract") != contract
        ):
            raise ValueError(
                "existing blend manifest has an incompatible generation contract"
            )
    existing = {row["source_id"]: row for row in (prior or {}).get("examples", [])}
    output_dir.mkdir(parents=True, exist_ok=True)
    render = synthesizer or _kokoro_synthesizer(model_path)
    examples = []
    planned = len(blend_pairs()) * len(selected_speeds)
    completed = 0
    for left, right in blend_pairs():
        blend = f"{left},{right}"
        for speed in selected_speeds:
            source_id = f"kokoro-blend-v3:{left}+{right}:{speed:.2f}"
            filename = f"{left}--{right}--speed-{speed:.2f}.wav"
            path = output_dir / filename
            row = existing.get(source_id)
            if row is not None:
                if row.get("path") != str(path.resolve()) or not path.is_file():
                    raise ValueError(
                        f"existing blend row is missing or incompatible: {source_id}"
                    )
                examples.append(row)
                completed += 1
                continue
            if path.exists():
                raise FileExistsError(f"untracked blend output exists: {path}")
            samples, rate = render(RAW_PHONES, blend, speed)
            pcm = normalize_pcm16(samples, rate)
            wav = _wav_bytes(pcm)
            path.write_bytes(wav)
            audio_hash = hashlib.sha256(wav).hexdigest()
            examples.append(
                {
                    "schema_version": 1,
                    "path": str(path.resolve()),
                    "label": 1,
                    "role": "positive",
                    "render_text": TEXT,
                    "raw_phonemes": RAW_PHONES,
                    "provider": "kokoro",
                    "model": MODEL_ID,
                    "model_sha256": model_hash,
                    "voice": blend,
                    "base_voices": [left, right],
                    "blend_weights": [0.5, 0.5],
                    "speed": speed,
                    "split": "train",
                    "speaker_id": f"kokoro-blend:{left}+{right}",
                    "session_id": source_id,
                    "source_id": source_id,
                    "source_group": "kokoro_blended_synthetic",
                    "semantic_label": "canonical_exact",
                    "target_id": TARGET_ID,
                    "target_phones": list(TARGET_PHONES),
                    "sample_rate": SAMPLE_RATE,
                    "channels": 1,
                    "duration_seconds": len(pcm) / 2 / SAMPLE_RATE,
                    "audio_sha256": audio_hash,
                    "output_hash": audio_hash,
                    "provenance_id": f"audio-sha256:{audio_hash}",
                    "ancestry_id": f"kokoro-blend:{left}+{right}",
                    "training_eligible": True,
                }
            )
            completed += 1
            if completed % 25 == 0 or completed == planned:
                print(
                    json.dumps({"generated_or_reused": completed, "planned": planned}),
                    flush=True,
                )
    payload = {
        "schema_version": 1,
        "generator": Path(__file__).stem,
        "contract": contract,
        "examples": examples,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--speed", type=float, action="append", dest="speeds")
    args = parser.parse_args()
    payload = generate(
        args.output_dir,
        args.manifest,
        args.model,
        speeds=args.speeds or DEFAULT_SPEEDS,
    )
    print(
        json.dumps(
            {
                "examples": len(payload["examples"]),
                "train_base_voices": len(TRAIN_VOICES),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
