#!/usr/bin/env python3
"""Generate train-only, multi-provider hard negatives for Kizz Control.

This corpus is deliberately separate from validation and final qualification.
It expands the exact phonetic confusions emitted by the frozen detector while
keeping provider voices assigned to the original training split.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.generate_kizz_control_c1_corpus import (
    ASSEMBLYAI_VOICES,
    DEEPGRAM_VOICES,
    ELEVENLABS_VOICES,
    KOKORO_VOICES,
    KokoroWorker,
    Task,
    _assemblyai,
    _deepgram,
    _elevenlabs,
    _load_env,
    _sha256_file,
    _wav_duration,
    causal_negative_decision,
)


SAMPLE_RATE = 16_000
DEFAULT_ENV_FILE = Path.home() / ".config" / "open-horizon-labs" / "voice.env"
DEFAULT_KOKORO_PYTHON = Path(os.environ.get("KIZZ_KOKORO_PYTHON", sys.executable))
CRITICAL_TEXTS = (
    "Kiss Control",
    "Kids Control",
    "Quiz Control",
    "Kizz patrol",
)
CONTEXT_TEXTS = (
    "Kiss, control",
    "Kids, control",
    "Quiz, control",
    "Kiss control?",
    "Kids control?",
    "Quiz control?",
    "Please use Kiss Control",
    "Please use Kids Control",
    "Please use Quiz Control",
    "I said Kiss Control",
    "I said Kids Control",
    "I said Quiz Control",
    "This control",
    "His control",
    "Kids can troll",
    "High five, kids",
    "The kids control the television",
    "The quiz controls the score",
    "The kitchen controls are broken",
    "This controller is missing",
)
PROVIDERS = ("assemblyai", "deepgram", "elevenlabs", "kokoro")


def _provider_voices(provider: str) -> list[tuple[str, str, str]]:
    if provider == "elevenlabs":
        return [(name, voice_id, split) for name, voice_id, split in ELEVENLABS_VOICES]
    if provider == "deepgram":
        return [(voice, voice, split) for voice, split in DEEPGRAM_VOICES]
    if provider == "assemblyai":
        return [(voice, voice, split) for voice, split in ASSEMBLYAI_VOICES]
    if provider == "kokoro":
        return [(voice, voice, split) for voice, split in KOKORO_VOICES]
    raise ValueError(f"unsupported provider: {provider}")


def planned_tasks(provider: str, contexts_per_voice: int = 8) -> list[Task]:
    if contexts_per_voice < 0 or contexts_per_voice > len(CONTEXT_TEXTS):
        raise ValueError("contexts_per_voice is outside the available phrase inventory")
    tasks: list[Task] = []
    train_voices = [voice for voice in _provider_voices(provider) if voice[2] == "train"]
    for voice_index, (voice, voice_id, split) in enumerate(train_voices):
        selected = list(CRITICAL_TEXTS)
        selected.extend(
            CONTEXT_TEXTS[(voice_index * contexts_per_voice + offset) % len(CONTEXT_TEXTS)]
            for offset in range(contexts_per_voice)
        )
        for phrase_index, text in enumerate(selected):
            decision = causal_negative_decision(text)
            if not decision["qualified"]:
                raise ValueError(f"causally unlearnable negative phrase: {text}")
            if provider == "elevenlabs":
                model = "eleven_multilingual_v2"
                settings: dict[str, Any] = {
                    "stability": (0.38, 0.52, 0.68, 0.80)[phrase_index % 4],
                    "similarity_boost": 0.82,
                    "style": 0.0,
                    "use_speaker_boost": True,
                }
                seed = 731_000 + voice_index * 100 + phrase_index
            elif provider == "deepgram":
                model = voice_id
                settings = {
                    "encoding": "linear16",
                    "sample_rate": SAMPLE_RATE,
                    "speed": (0.84, 0.92, 1.0, 1.08, 1.16)[phrase_index % 5],
                }
                seed = None
            elif provider == "assemblyai":
                model = "voice-agent-api"
                settings = {"source_rate_hz": 24_000, "converted_rate_hz": SAMPLE_RATE}
                seed = None
            else:
                model = "hexgrad/Kokoro-82M"
                settings = {"speed": (0.84, 0.92, 1.0, 1.08, 1.16)[phrase_index % 5]}
                seed = None
            tasks.append(
                Task(
                    provider=provider,
                    model=model,
                    voice=voice,
                    provider_voice_id=voice_id,
                    split=split,
                    label=0,
                    text=text,
                    variant_index=phrase_index,
                    settings=settings,
                    seed=seed,
                )
            )
    return tasks


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = None
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        descriptor = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(descriptor, path)
    finally:
        descriptor.unlink(missing_ok=True)


def _samples(path: Path) -> int:
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2 or source.getframerate() != SAMPLE_RATE:
            raise ValueError(f"{path}: expected mono PCM16 at 16 kHz")
        return source.getnframes()


def _capture(task: Task, output: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    digest = _sha256_file(output)
    capture_id = f"kizz-collision-hardening-{task.provider}-{task.descriptor[:20]}"
    speaker_id = f"tts-{task.provider}-{task.voice}".replace("_", "-")
    return {
        "capture_id": capture_id,
        "conditions": {
            "render_text": task.text,
            "descriptor_sha256": task.descriptor,
            "provider_metadata": metadata,
            "settings": task.settings,
        },
        "detected": False,
        "device_id": "source-synthesis",
        "device_profile": "source_tts_pcm16_16khz",
        "path": str(output),
        "phrase": task.text,
        "pronunciation": "phonetic_collision",
        "samples": _samples(output),
        "session_id": f"tts:{task.provider}:{task.voice}:collision-hardening-v1",
        "sha256": digest,
        "source": f"tts_collision_hardening:{task.provider}",
        "speaker_id": speaker_id,
        "split": "train",
        "truth": "hard_negative",
    }


def generate(
    *,
    provider: str,
    output: Path,
    env_file: Path,
    kokoro_python: Path,
    contexts_per_voice: int,
    max_paid_audio_seconds: float,
) -> dict[str, Any]:
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "device-corpus.json"
    existing: dict[str, dict[str, Any]] = {}
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing = {str(row["capture_id"]): row for row in previous.get("captures", [])}
    env = _load_env(env_file)
    keys = {
        "assemblyai": env.get("ASSEMBLYAI_API_KEY"),
        "deepgram": env.get("DEEPGRAM_API_KEY"),
        "elevenlabs": env.get("ELEVEN_LABS_API_KEY") or env.get("ELEVENLABS_API_KEY"),
    }
    if provider in keys and not keys[provider]:
        raise RuntimeError(f"{provider} API key is not configured")
    tasks = planned_tasks(provider, contexts_per_voice)
    captures: list[dict[str, Any]] = []
    paid_seconds = 0.0
    worker = KokoroWorker(kokoro_python) if provider == "kokoro" else None
    try:
        for ordinal, task in enumerate(tasks, 1):
            capture_id = f"kizz-collision-hardening-{task.provider}-{task.descriptor[:20]}"
            previous = existing.get(capture_id)
            if previous is not None:
                previous_path = output / str(previous["path"])
                if previous_path.is_file() and _sha256_file(previous_path) == previous.get("sha256"):
                    captures.append(previous)
                    paid_seconds += _wav_duration(previous_path) if provider != "kokoro" else 0.0
                    continue
            if provider != "kokoro" and paid_seconds >= max_paid_audio_seconds:
                raise RuntimeError("paid-audio duration guard reached")
            audio_dir = output / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            audio_path = audio_dir / f"{task.descriptor[:20]}.wav"
            metadata: dict[str, Any] = {}
            if provider == "elevenlabs":
                _elevenlabs(task, str(keys[provider]), audio_path)
            elif provider == "deepgram":
                metadata = _deepgram(task, str(keys[provider]), audio_path)
            elif provider == "assemblyai":
                metadata = asyncio.run(_assemblyai(task, str(keys[provider]), audio_path))
            elif provider == "kokoro":
                assert worker is not None
                worker.render(task, audio_path)
            duration = _wav_duration(audio_path)
            if not 0.2 <= duration <= 12.0:
                raise ValueError(f"generated duration outside bounds: {duration:.3f}s")
            row = _capture(task, audio_path, metadata)
            row["path"] = str(audio_path.relative_to(output))
            captures.append(row)
            if provider != "kokoro":
                paid_seconds += duration
            payload = _manifest(provider, captures)
            _atomic_json(manifest_path, payload)
            print(json.dumps({"attempt": ordinal, "provider": provider, "voice": task.voice, "text": task.text, "duration_seconds": duration}, sort_keys=True), flush=True)
    finally:
        if worker is not None:
            worker.close()
    payload = _manifest(provider, captures)
    _atomic_json(manifest_path, payload)
    return {
        "provider": provider,
        "planned": len(tasks),
        "captures": len(captures),
        "voices": len({row["speaker_id"] for row in captures}),
        "duration_seconds": sum(int(row["samples"]) / SAMPLE_RATE for row in captures),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
    }


def _manifest(provider: str, captures: Sequence[dict[str, Any]]) -> dict[str, Any]:
    speakers = {
        row["speaker_id"]: {
            "age_group": "unknown",
            "kind": "synthetic",
            "provider": provider,
            "split": "train",
            "voice": row["speaker_id"].removeprefix(f"tts-{provider}-"),
            "voice_id": row["speaker_id"],
        }
        for row in captures
    }
    return {
        "schema_version": 2,
        "corpus_id": f"kizz-control-collision-hardening-{provider}-v1",
        "device_profiles": {
            "source_tts_pcm16_16khz": {
                "audio": {"channels": 1, "sample_format": "s16le", "sample_rate": SAMPLE_RATE}
            }
        },
        "speakers": speakers,
        "captures": list(captures),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=PROVIDERS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--kokoro-python", type=Path, default=DEFAULT_KOKORO_PYTHON)
    parser.add_argument("--contexts-per-voice", type=int, default=8)
    parser.add_argument("--max-paid-audio-seconds", type=float, default=1800.0)
    args = parser.parse_args(argv)
    report = generate(
        provider=args.provider,
        output=args.output,
        env_file=args.env_file,
        kokoro_python=args.kokoro_python,
        contexts_per_voice=args.contexts_per_voice,
        max_paid_audio_seconds=args.max_paid_audio_seconds,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
