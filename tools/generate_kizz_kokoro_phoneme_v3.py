#!/usr/bin/env python3
"""Generate the provenance-complete canonical-v3 Kokoro Kizz positives.

The generator deliberately sends Kokoro phonemes, rather than text, to avoid
making grapheme-to-phoneme behavior an unrecorded part of the dataset.  The
manifest is also the resume lock: a different recipe, model, voice catalog,
split map, or speed list cannot reuse an existing manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

SAMPLE_RATE = 16_000
MODEL_ID = "hexgrad/Kokoro-82M"
RAW_PHONES = "hI fI kɪz"
TARGET_ID = "hiphi_kizz"
TARGET_PHONES = ("h", "aɪ", "f", "aɪ", "k", "ɪ", "z")
TEXT = "Hi-Fi Kizz"
PROVIDER = "kokoro"
SCHEMA_VERSION = "canonical-v3-kokoro-phoneme-1"

# This is a closed catalog, not a permissive prefix check.  New Kokoro voices
# must be deliberately classified before they can enter a corpus.
ENGLISH_VOICES = (
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_heart",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",
)
SPLIT_VOICES = {
    "test": ("af_sky", "am_santa", "bf_lily", "bm_lewis", "bm_daniel", "af_nicole"),
    "validation": (
        "af_sarah",
        "am_puck",
        "bf_isabella",
        "bm_george",
        "am_onyx",
        "bf_emma",
    ),
}
VOICE_SPLITS = {
    voice: split for split, voices in SPLIT_VOICES.items() for voice in voices
}
VOICE_SPLITS.update(
    {voice: "train" for voice in ENGLISH_VOICES if voice not in VOICE_SPLITS}
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_sha256(model_path: Path) -> str:
    """Hash a model file or directory deterministically, including file names."""
    path = model_path.resolve()
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise ValueError(f"model path does not exist: {path}")
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"model directory is empty: {path}")
    for item in files:
        digest.update(str(item.relative_to(path)).encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def split_for_voice(voice: str) -> str:
    if voice not in VOICE_SPLITS:
        raise ValueError(f"unknown or undeclared English Kokoro voice: {voice}")
    return VOICE_SPLITS[voice]


def _finite_speed(value: object) -> float:
    try:
        speed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("speed must be finite and positive") from error
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("speed must be finite and positive")
    return speed


def normalize_pcm16(samples: Any, sample_rate: int) -> bytes:
    """Convert Kokoro audio to deterministic 16 kHz mono PCM16."""
    import numpy as np

    values = np.asarray(samples)
    if values.ndim == 2:
        values = values.mean(axis=1)
    if values.ndim != 1 or not len(values) or np.any(~np.isfinite(values)):
        raise ValueError("Kokoro returned invalid audio")
    if int(sample_rate) <= 0:
        raise ValueError("Kokoro output sample rate must be positive")
    if int(sample_rate) != SAMPLE_RATE:
        from scipy.signal import resample_poly

        # Kokoro emits 24 kHz.  Polyphase resampling is deterministic for the
        # pinned generator environment and avoids the aliasing introduced by
        # the former linear interpolation path.
        values = resample_poly(values, SAMPLE_RATE, int(sample_rate))
    values = np.clip(values.astype(np.float64), -1.0, 1.0)
    pcm = np.rint(values * 32767.0).astype("<i2")
    return pcm.tobytes()


def _wav_bytes(pcm: bytes) -> bytes:
    import io
    import wave

    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm)
    return output.getvalue()


def _signature(model_hash: str, speeds: tuple[float, ...]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_id": MODEL_ID,
        "model_sha256": model_hash,
        "raw_phones": RAW_PHONES,
        "target_id": TARGET_ID,
        "target_phones": list(TARGET_PHONES),
        "source_group": "kokoro_phoneme_synthetic",
        "role": "positive",
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "encoding": "PCM_S16LE",
        "speeds": list(speeds),
        "declared_english_voices": list(ENGLISH_VOICES),
        "voice_split_map": {voice: VOICE_SPLITS[voice] for voice in ENGLISH_VOICES},
    }


def validate_existing_manifest(
    path: Path, signature: Mapping[str, Any]
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"existing manifest is unreadable: {path}") from error
    if payload.get("generator") != "generate_kizz_kokoro_phoneme_v3":
        raise ValueError("existing manifest is incompatible with Kokoro canonical-v3")
    if payload.get("contract") != dict(signature):
        raise ValueError("existing manifest has incompatible generation contract")
    if not isinstance(payload.get("examples"), list):
        raise TypeError("existing compatible manifest has no examples list")
    return payload


def _kokoro_synthesizer(
    model_path: Path,
) -> Callable[[str, str, float], tuple[Any, int]]:
    try:
        from kokoro import KModel, KPipeline
    except ImportError as error:  # pragma: no cover - environment-specific
        raise RuntimeError("install kokoro to generate Kokoro audio") from error
    if model_path.is_dir():
        candidates = sorted(model_path.rglob("kokoro-v1_0.pth"))
        if len(candidates) != 1:
            raise ValueError("model directory must contain exactly one kokoro-v1_0.pth")
        model_file = candidates[0]
    else:
        model_file = model_path
    model = KModel(repo_id=MODEL_ID, model=str(model_file))
    pipeline = KPipeline(lang_code="a", repo_id=MODEL_ID, model=model)
    voice_hashes: dict[str, str] = {}

    def render(phones: str, voice: str, speed: float) -> tuple[Any, int]:
        # Kokoro's phoneme mode is selected by the phoneme input marker.  The
        # adapter keeps this call in one place for package API changes.
        pack = pipeline.load_voice(voice)
        voice_hashes.setdefault(
            voice, hashlib.sha256(pack.cpu().numpy().tobytes()).hexdigest()
        )
        result = next(pipeline.generate_from_tokens(phones, voice=voice, speed=speed))
        audio = result.audio
        if hasattr(audio, "cpu"):
            audio = audio.cpu().numpy()
        return audio, 24_000

    render.voice_sha256 = voice_hashes  # type: ignore[attr-defined]
    return render


def generate(
    output_dir: Path,
    manifest_path: Path,
    model_path: Path,
    speeds: Iterable[float] = (0.82, 0.91, 1.0, 1.09, 1.18),
    voices: Iterable[str] = ENGLISH_VOICES,
    synthesizer: Callable[[str, str, float], tuple[Any, int]] | None = None,
) -> dict[str, Any]:
    speeds_tuple = tuple(_finite_speed(value) for value in speeds)
    if not speeds_tuple or len(set(speeds_tuple)) != len(speeds_tuple):
        raise ValueError("speeds must be non-empty and unique")
    selected = tuple(voices)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("voices must be non-empty and unique")
    for voice in selected:
        split_for_voice(voice)
    if set(selected) != set(ENGLISH_VOICES):
        raise ValueError(
            "generation must declare the complete English Kokoro voice catalog"
        )
    model_hash = model_sha256(model_path)
    signature = _signature(model_hash, speeds_tuple)
    existing = validate_existing_manifest(manifest_path, signature)
    existing_rows = {
        row["source_id"]: row for row in (existing or {}).get("examples", [])
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    render = synthesizer or _kokoro_synthesizer(model_path)
    voice_hashes = getattr(render, "voice_sha256", {})
    planned = []
    for voice in ENGLISH_VOICES:
        for speed in speeds_tuple:
            source_id = hashlib.sha256(
                f"{SCHEMA_VERSION}\0{voice}\0{speed:g}".encode()
            ).hexdigest()
            path = output_dir / f"{source_id}.wav"
            if source_id not in existing_rows and path.exists():
                raise FileExistsError(f"output collision: {path}")
            planned.append((voice, speed, source_id, path))
    examples = []
    for voice, speed, source_id, path in planned:
        prior = existing_rows.get(source_id)
        if prior is not None:
            if prior.get("path") != str(path.resolve()) or not path.is_file():
                raise ValueError(
                    f"existing manifest row is incompatible or missing: {source_id}"
                )
            row = prior
        else:
            samples, rate = render(RAW_PHONES, voice, speed)
            pcm = normalize_pcm16(samples, rate)
            wav = _wav_bytes(pcm)
            path.write_bytes(wav)
            audio_hash = hashlib.sha256(wav).hexdigest()
            row = {
                "path": str(path.resolve()),
                "label": 1,
                "text": TEXT,
                "render_text": TEXT,
                "raw_phones": RAW_PHONES,
                "provider": PROVIDER,
                "target_id": TARGET_ID,
                "target_phones": list(TARGET_PHONES),
                "role": "positive",
                "voice": voice,
                "speed": speed,
                "split": split_for_voice(voice),
                "source_group": "kokoro_phoneme_synthetic",
                "semantic_label": "canonical_exact",
                "speaker_id": f"kokoro:{voice}",
                "session_id": f"kokoro:{voice}:{speed:g}",
                "source_id": source_id,
                "provenance_id": f"audio-sha256:{audio_hash}",
                "parent_id": f"model-sha256:{model_hash}",
                "ancestry_id": f"kokoro:{MODEL_ID}:{voice}",
                "audio_sha256": audio_hash,
                "output_hash": audio_hash,
                "voice_sha256": voice_hashes.get(
                    voice, hashlib.sha256(voice.encode("utf-8")).hexdigest()
                ),
                "model_id": MODEL_ID,
                "model_sha256": model_hash,
                "sample_rate": SAMPLE_RATE,
                "channels": 1,
                "duration": len(pcm) / 2 / SAMPLE_RATE,
                "duration_seconds": len(pcm) / 2 / SAMPLE_RATE,
                "training_eligible": True,
            }
        examples.append(row)
    payload = {
        "generator": "generate_kizz_kokoro_phoneme_v3",
        "contract": signature,
        "examples": examples,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="local Kokoro model file or snapshot directory",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--speed", type=float, action="append", dest="speeds")
    args = parser.parse_args()
    generate(
        args.output_dir,
        args.manifest,
        args.model,
        args.speeds or (0.82, 0.91, 1.0, 1.09, 1.18),
    )


if __name__ == "__main__":
    main()
