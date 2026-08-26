#!/usr/bin/env python3
"""Generate the bounded, voice-disjoint Kizz Control C1 speech corpus.

The corpus is intentionally a mixture of independent TTS families.  A task is
identified by a stable descriptor and is resumable; completed audio is never
regenerated.  Paid synthesis is bounded by ``--max-paid-audio-seconds`` and
the output manifest records provider, voice, split, text, settings, and hashes.

This is source audio only.  Acoustic pronunciation qualification, overlays,
and feature extraction are separate fail-closed stages.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import re
import struct
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from microwakeword.wake_phrase import KIZZ_CONTROL


SAMPLE_RATE = 16_000
DEFAULT_ENV_FILE = Path("/Users/muness1/.config/open-horizon-labs/voice.env")
PAID_PROVIDERS = frozenset(("assemblyai", "deepgram", "elevenlabs"))
EXPECTED_PROVIDERS = frozenset(
    ("assemblyai", "deepgram", "elevenlabs", "kokoro", "macos-say")
)
# The target-channel qualification holdout must realize the provider *and* voice
# diversity that exists in the source inventory.  The original recipe reserved
# every punctuation/speed variant from one voice per provider.  That looked
# provider-balanced in aggregate while collapsing each provider to one acoustic
# identity.  Reserve six deterministic clips per runtime provider, distributing
# them across every test voice before assigning a second/third clip to a voice.
# macOS ``say`` remains source-audit material but is not an approved runtime
# positive provider for Kizz Control C1.
RESERVED_REPLAY_VARIANTS = {
    "assemblyai": frozenset(
        {
            ("tyler", 2),
            ("victor", 0),
            ("winter", 0),
            ("eleanor", 0),
            ("tyler", 1),
            ("victor", 1),
        }
    ),
    "deepgram": frozenset(
        {
            ("aura-2-arcas-en", 1),
            ("aura-2-pandora-en", 1),
            ("aura-2-helena-en", 1),
            ("aura-2-arcas-en", 4),
            ("aura-2-pandora-en", 2),
            ("aura-2-helena-en", 5),
        }
    ),
    "elevenlabs": frozenset(
        {
            ("kizz-adult-test", 0),
            ("kizz-adult-test", 1),
            ("kizz-child-test", 1),
            ("kizz-child-test", 2),
            ("kizz-adult-test", 3),
            ("kizz-child-test", 4),
        }
    ),
    "kokoro": frozenset(
        {
            ("af_river", 6),
            ("am_onyx", 1),
            ("am_onyx", 5),
            ("bf_isabella", 5),
            ("bf_isabella", 6),
            ("bm_lewis", 1),
        }
    ),
}
RUNTIME_POSITIVE_PROVIDERS = frozenset(RESERVED_REPLAY_VARIANTS)
RESERVED_REPLAYS_PER_PROVIDER = 6

POSITIVE_TEXTS = (
    "Kizz Control",
    "Kizz, control",
    "Kizz control.",
    "Kiz Control",
    "Kizz... control",
    "Kizz Control!",
    "kizz control",
    "Kizz control?",
)

COLLISION_TEXTS = (
    "Kids Control",
    "Kiss Control",
    "Quiz Control",
    "This control",
    "His control",
    "Kizz controller",
    "Kizz controlled",
    "Kizz patrol",
    "The kids control the television",
    "Kids can troll",
    "The kitchen controls are broken",
    "This controller is missing",
)


def _normalized_letters(text: str) -> str:
    """Return the pronunciation-label comparison form for a render prompt."""
    return "".join(character for character in text.casefold() if character.isalpha())


def causal_negative_decision(text: str) -> dict[str, Any]:
    """Reject negative labels that contain the complete wake as a prefix.

    A streaming detector must fire as soon as the canonical phrase completes.
    It cannot use a later suffix in ``controller`` or ``controlled`` to revoke
    that decision.  Keeping such rows as negatives creates contradictory frame
    supervision, so they remain in the source audit but cannot enter training.
    """
    canonical = _normalized_letters(KIZZ_CONTROL.text)
    candidate = _normalized_letters(text)
    impossible = candidate.startswith(canonical)
    return {
        "qualified": not impossible,
        "normalized_text": candidate,
        "canonical_prefix": canonical,
        "reason": "causally_unlearnable_suffix_extension" if impossible else None,
    }

ELEVENLABS_VOICES = (
    ("kizz-adult-train-a", "TNNkX6cWpXTdjh8kooeb", "train"),
    ("kizz-adult-train-b", "JKVgBZo4Dd7KuCbnqzsK", "train"),
    ("kizz-child-train-a", "wq0ARB4PPBM4uXWw05pP", "train"),
    ("kizz-child-train-b", "QU94KFzaF0bX44id0iF6", "train"),
    ("kizz-adult-validation", "YvSEGi6fC0oS5DaeSb2N", "validation"),
    ("kizz-child-validation", "Zp2KrKuSxprqa7jA0NCl", "validation"),
    ("kizz-adult-test", "2KTKfRJMZKxX12PesDI5", "test"),
    ("kizz-child-test", "IKT2xkGjbwJhZO8ptpOG", "test"),
)

DEEPGRAM_VOICES = (
    ("aura-2-thalia-en", "train"),
    ("aura-2-apollo-en", "train"),
    ("aura-2-amalthea-en", "train"),
    ("aura-2-hyperion-en", "train"),
    ("aura-2-andromeda-en", "validation"),
    ("aura-2-draco-en", "validation"),
    ("aura-2-aries-en", "validation"),
    ("aura-2-arcas-en", "test"),
    ("aura-2-pandora-en", "test"),
    ("aura-2-helena-en", "test"),
)

ASSEMBLYAI_VOICES = (
    ("alba", "train"),
    ("anna", "train"),
    ("bella", "train"),
    ("charles", "train"),
    ("david", "train"),
    ("emma", "train"),
    ("estelle", "train"),
    ("eve", "train"),
    ("george", "train"),
    ("helen", "train"),
    ("ivy", "train"),
    ("james", "validation"),
    ("kyle", "validation"),
    ("martha", "validation"),
    ("river", "validation"),
    ("tyler", "test"),
    ("victor", "test"),
    ("winter", "test"),
    ("eleanor", "test"),
)

KOKORO_VOICES = (
    ("af_heart", "train"),
    ("af_bella", "train"),
    ("af_nova", "train"),
    ("am_adam", "train"),
    ("am_eric", "train"),
    ("am_liam", "train"),
    ("bf_emma", "train"),
    ("bm_george", "train"),
    ("af_sky", "validation"),
    ("am_michael", "validation"),
    ("bf_alice", "validation"),
    ("bm_daniel", "validation"),
    ("af_river", "test"),
    ("am_onyx", "test"),
    ("bf_isabella", "test"),
    ("bm_lewis", "test"),
)

MACOS_VOICES = (
    ("Samantha", "train"),
    ("Daniel", "train"),
    ("Karen", "train"),
    ("Moira", "train"),
    ("Tessa", "train"),
    ("Rishi", "train"),
    ("Reed (English (US))", "validation"),
    ("Flo (English (US))", "validation"),
    ("Grandma (English (US))", "validation"),
    ("Sandy (English (US))", "test"),
    ("Shelley (English (UK))", "test"),
    ("Eddy (English (US))", "test"),
)


@dataclass(frozen=True)
class Task:
    provider: str
    model: str
    voice: str
    provider_voice_id: str
    split: str
    label: int
    text: str
    variant_index: int
    settings: dict[str, Any]
    seed: int | None = None

    @property
    def semantic_label(self) -> str:
        return "canonical_exact" if self.label else "phonetic_collision"

    @property
    def descriptor(self) -> str:
        payload = {
            "provider": self.provider,
            "model": self.model,
            "voice": self.voice,
            "provider_voice_id": self.provider_voice_id,
            "split": self.split,
            "label": self.label,
            "text": self.text,
            "variant_index": self.variant_index,
            "settings": self.settings,
            "seed": self.seed,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def is_reserved_replay_task(task: Task) -> bool:
    """Return whether a source clip belongs to the immutable replay holdout."""
    return (
        task.label == 1
        and task.split == "test"
        and (task.voice, task.variant_index)
        in RESERVED_REPLAY_VARIANTS.get(task.provider, ())
    )


def _load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.is_file():
        for raw in path.read_text(encoding="utf-8").splitlines():
            if "=" not in raw or raw.lstrip().startswith("#"):
                continue
            key, value = raw.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    values.update(
        {key: value for key, value in os.environ.items() if key.endswith("API_KEY")}
    )
    return values


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as source:
        if (
            source.getnchannels() != 1
            or source.getsampwidth() != 2
            or source.getframerate() != SAMPLE_RATE
        ):
            raise ValueError(f"{path}: expected mono PCM16 at {SAMPLE_RATE} Hz")
        return source.getnframes() / source.getframerate()


def _write_pcm(path: Path, pcm: bytes, sample_rate: int = SAMPLE_RATE) -> None:
    if not pcm or len(pcm) % 2:
        raise ValueError("provider returned invalid PCM16")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm)


def _normalize_with_ffmpeg(source: Path, output: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output),
        ],
        check=True,
        timeout=120,
    )


def _request_bytes(
    url: str, body: bytes, headers: dict[str, str], timeout: float = 90.0
) -> bytes:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _request_bytes_and_headers(
    url: str, body: bytes, headers: dict[str, str], timeout: float = 90.0
) -> tuple[bytes, dict[str, str]]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read(), {key.lower(): value for key, value in response.headers.items()}


def _elevenlabs(task: Task, key: str, output: Path) -> None:
    url = (
        "https://api.elevenlabs.io/v1/text-to-speech/"
        + urllib.parse.quote(task.provider_voice_id, safe="")
        + "?output_format=pcm_16000"
    )
    body = json.dumps(
        {
            "text": task.text,
            "model_id": task.model,
            "language_code": "en",
            "seed": task.seed,
            "voice_settings": task.settings,
        }
    ).encode()
    pcm = _request_bytes(
        url,
        body,
        {
            "xi-api-key": key,
            "Content-Type": "application/json",
            "Accept": "audio/pcm",
        },
    )
    _write_pcm(output, pcm)


def _deepgram_payload_text(task: Task) -> tuple[str, int]:
    text = task.text
    expected_pronunciations = 0
    if task.label == 1:
        match = re.search(r"\bKiz{1,2}\b", text, flags=re.IGNORECASE)
        if not match:
            raise ValueError("Deepgram canonical positive has no Kizz token")
        override = (
            r'\{"word": "'
            + match.group(0)
            + r'", "pronounce": "kɪz"\}'
        )
        text = text[: match.start()] + override + text[match.end() :]
        expected_pronunciations = 1
    return text, expected_pronunciations


def _deepgram(task: Task, key: str, output: Path) -> dict[str, Any]:
    speed = float(task.settings.get("speed", 1.0))
    url = (
        "https://api.deepgram.com/v1/speak?model="
        + urllib.parse.quote(task.model)
        + "&encoding=linear16&sample_rate=16000&container=wav&speed="
        + urllib.parse.quote(str(speed))
    )
    text, expected_pronunciations = _deepgram_payload_text(task)
    data, response_headers = _request_bytes_and_headers(
        url,
        json.dumps({"text": text}).encode(),
        {"Authorization": "Token " + key, "Content-Type": "application/json"},
    )
    applied = int(response_headers.get("dg-pronunciations-applied", "0"))
    if applied != expected_pronunciations:
        raise RuntimeError(
            "Deepgram pronunciation contract failed: "
            f"expected {expected_pronunciations}, applied {applied}"
        )
    # Deepgram's streamed RIFF response may retain an open-ended data length.
    # Close the container through ffmpeg before hashing or measuring it.
    with tempfile.TemporaryDirectory() as directory:
        raw = Path(directory) / "deepgram-stream.wav"
        raw.write_bytes(data)
        _normalize_with_ffmpeg(raw, output)
    return {
        "request_id": response_headers.get("dg-request-id"),
        "model_name": response_headers.get("dg-model-name"),
        "pronunciations_applied": applied,
        "pronunciation_warnings": response_headers.get("dg-pronunciation-warnings"),
        "speed_used": response_headers.get("dg-speed-used"),
    }


async def _assemblyai(task: Task, key: str, output: Path) -> dict[str, Any]:
    try:
        import websockets
    except ImportError as error:
        raise RuntimeError("install websockets to use AssemblyAI Voice Agents") from error
    chunks: list[bytes] = []
    events: list[str | None] = []
    session_id = None
    session = {
        "system_prompt": (
            "Speak the supplied greeting exactly once with no additional words. "
            "Do not answer it and do not ask a question."
        ),
        "greeting": task.text,
        "input": {
            "format": {"encoding": "audio/pcm"},
            "turn_detection": {"interrupt_response": False},
        },
        "output": {
            "voice": task.provider_voice_id,
            "format": {"encoding": "audio/pcm"},
        },
    }
    async with websockets.connect(
        "wss://agents.assemblyai.com/v1/ws",
        additional_headers={"Authorization": "Bearer " + key},
        open_timeout=20,
        close_timeout=5,
        max_size=10_000_000,
    ) as socket:
        await socket.send(json.dumps({"type": "session.update", "session": session}))
        deadline = time.monotonic() + 30.0
        async for raw in socket:
            message = json.loads(raw)
            event = message.get("type")
            events.append(event)
            if event == "session.ready":
                session_id = message.get("session_id")
            elif event == "reply.audio":
                chunks.append(base64.b64decode(message.get("data", "")))
            elif event == "session.error":
                raise RuntimeError(json.dumps(message, sort_keys=True))
            elif event == "reply.done":
                break
            if time.monotonic() > deadline:
                raise TimeoutError("AssemblyAI fixed greeting timed out")
    if not chunks:
        raise RuntimeError("AssemblyAI returned no audio")
    with tempfile.TemporaryDirectory() as directory:
        raw = Path(directory) / "reply.raw"
        raw.write_bytes(b"".join(chunks))
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-f",
                "s16le",
                "-ar",
                "24000",
                "-ac",
                "1",
                "-i",
                str(raw),
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(output),
            ],
            check=True,
            timeout=120,
        )
    return {"session_id": session_id, "events": events, "session_update": session}


class KokoroWorker:
    """Keep Kokoro's 82M model loaded across the complete local cohort."""

    _CODE = r"""
from kokoro import KPipeline
import json
import numpy as np
import sys
import wave
pipeline = KPipeline(lang_code='a')
for line in sys.stdin:
    task = json.loads(line)
    try:
        with wave.open(task['path'], 'wb') as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(24000)
            for result in pipeline(task['text'], voice=task['voice'], speed=float(task['speed']), split_pattern=r'\n+'):
                if result.audio is not None:
                    output.writeframes((result.audio.numpy() * 32767).astype(np.int16).tobytes())
        print('KOKORO_DONE ' + json.dumps({'ok': True}), flush=True)
    except Exception as error:
        print('KOKORO_DONE ' + json.dumps({'ok': False, 'error': repr(error)}), flush=True)
"""

    def __init__(self, python: Path) -> None:
        self.process = subprocess.Popen(
            [str(python), "-u", "-c", self._CODE],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def render(self, task: Task, output: Path) -> None:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("Kokoro worker pipes are unavailable")
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "kokoro-24k.wav"
            self.process.stdin.write(
                json.dumps(
                    {
                        "voice": task.provider_voice_id,
                        "text": task.text,
                        "speed": task.settings["speed"],
                        "path": str(raw),
                    }
                )
                + "\n"
            )
            self.process.stdin.flush()
            while True:
                line = self.process.stdout.readline()
                if not line:
                    raise RuntimeError("Kokoro worker exited before rendering")
                if line.startswith("KOKORO_DONE "):
                    result = json.loads(line.removeprefix("KOKORO_DONE "))
                    break
            if not result.get("ok"):
                raise RuntimeError(str(result.get("error")))
            _normalize_with_ffmpeg(raw, output)

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=5)


def _macos(task: Task, output: Path) -> None:
    with tempfile.TemporaryDirectory() as directory:
        raw = Path(directory) / "say.aiff"
        subprocess.run(
            [
                "say",
                "-v",
                task.provider_voice_id,
                "-r",
                str(task.settings["rate_wpm"]),
                "-o",
                str(raw),
                task.text,
            ],
            check=True,
            timeout=60,
        )
        _normalize_with_ffmpeg(raw, output)


def _tasks() -> list[Task]:
    tasks: list[Task] = []

    def texts_for(voice_index: int) -> list[tuple[int, str, int]]:
        positive = [(1, text, index) for index, text in enumerate(POSITIVE_TEXTS)]
        collision = [
            (0, COLLISION_TEXTS[(voice_index * 4 + index) % len(COLLISION_TEXTS)], index)
            for index in range(4)
        ]
        return positive + collision

    for voice_index, (name, voice_id, split) in enumerate(ELEVENLABS_VOICES):
        for label, text, index in texts_for(voice_index):
            settings = {
                "stability": 0.52 if index % 2 == 0 else 0.72,
                "similarity_boost": 0.82,
                "style": 0.0,
                "use_speaker_boost": True,
            }
            tasks.append(
                Task(
                    "elevenlabs",
                    "eleven_multilingual_v2",
                    name,
                    voice_id,
                    split,
                    label,
                    text,
                    index,
                    settings,
                    231_000 + voice_index * 100 + label * 20 + index,
                )
            )
    for voice_index, (voice, split) in enumerate(DEEPGRAM_VOICES):
        for label, text, index in texts_for(voice_index):
            speeds = (0.82, 0.90, 0.96, 1.0, 1.04, 1.10, 1.18, 1.26)
            tasks.append(
                Task(
                    "deepgram",
                    voice,
                    voice,
                    voice,
                    split,
                    label,
                    text,
                    index,
                    {
                        "encoding": "linear16",
                        "sample_rate": SAMPLE_RATE,
                        "speed": speeds[index % len(speeds)],
                        "positive_pronunciation_override": "kɪz",
                    },
                )
            )
    for voice_index, (voice, split) in enumerate(ASSEMBLYAI_VOICES):
        # Voice Agent synthesis is real-time and variable. Six positives plus
        # four collisions per voice keeps the cohort near the requested 200.
        for label, text, index in texts_for(voice_index):
            if label == 1 and index >= 6:
                continue
            tasks.append(
                Task(
                    "assemblyai",
                    "voice-agent-api",
                    voice,
                    voice,
                    split,
                    label,
                    text,
                    index,
                    {"source_rate_hz": 24_000, "converted_rate_hz": SAMPLE_RATE},
                )
            )
    speeds = (0.84, 0.90, 0.96, 1.0, 1.06, 1.12, 1.20, 1.28)
    for voice_index, (voice, split) in enumerate(KOKORO_VOICES):
        for label, text, index in texts_for(voice_index):
            tasks.append(
                Task(
                    "kokoro",
                    "hexgrad/Kokoro-82M",
                    voice,
                    voice,
                    split,
                    label,
                    text,
                    index,
                    {"speed": speeds[index % len(speeds)]},
                )
            )
    rates = (132, 145, 158, 172, 186, 200, 216, 232)
    for voice_index, (voice, split) in enumerate(MACOS_VOICES):
        for label, text, index in texts_for(voice_index):
            tasks.append(
                Task(
                    "macos-say",
                    "macOS-system-tts",
                    voice,
                    voice,
                    split,
                    label,
                    text,
                    index,
                    {"rate_wpm": rates[index % len(rates)]},
                )
            )
    return tasks


def _row(task: Task, path: Path, provider_metadata: dict[str, Any]) -> dict[str, Any]:
    audio_sha = _sha256_file(path)
    descriptor = task.descriptor
    source_id = f"kizz-control-c1:{descriptor}"
    source_group = (
        f"{task.provider.replace('-', '_')}_synthetic"
        if task.label
        else "kizz_control_phonetic_collision"
    )
    reserved_replay = is_reserved_replay_task(task)
    causal_decision = causal_negative_decision(task.text) if not task.label else None
    training_eligible = not reserved_replay and (
        task.label == 1 or bool(causal_decision and causal_decision["qualified"])
    )
    row = {
        "path": str(path.resolve()),
        "label": task.label,
        "training_eligible": training_eligible,
        "semantic_label": task.semantic_label,
        "source_id": source_id,
        "source_group": source_group,
        "provider": task.provider,
        "model": task.model,
        "voice": task.voice,
        "voice_id": f"tts:{task.provider}:{task.voice}",
        "speaker_id": f"tts:{task.provider}:{task.voice}",
        "session_id": f"tts:{task.provider}:{task.voice}:{task.split}",
        "split": task.split,
        "render_text": task.text,
        "target_id": KIZZ_CONTROL.phrase_id if task.label else None,
        "target_phones": list(KIZZ_CONTROL.phones) if task.label else [],
        "settings": task.settings,
        "seed": task.seed,
        "descriptor_sha256": descriptor,
        "audio_sha256": audio_sha,
        "provenance_id": f"audio-sha256:{audio_sha}",
        "parent_id": f"synthesis-source:{descriptor}",
        "ancestry_id": f"tts-ancestry:{descriptor}",
        "duration_seconds": _wav_duration(path),
        "device_rendered": False,
        "provider_metadata": provider_metadata,
    }
    if causal_decision is not None:
        row["causal_negative_contract"] = causal_decision
    if reserved_replay:
        row["exclusion_reason"] = "reserved_for_device_replay"
        row["reserved_evidence_role"] = "target_channel_positive"
    elif causal_decision is not None and not causal_decision["qualified"]:
        row["exclusion_reason"] = str(causal_decision["reason"])
    return row


def _write_manifest(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        eligibility = "eligible" if row.get("training_eligible") is True else "excluded"
        key = f"{row['split']}:{row['provider']}:{row['label']}:{eligibility}"
        counts[key] = counts.get(key, 0) + 1
    payload = {
        "schema_version": 2,
        "wake_phrase": {
            "phrase_id": KIZZ_CONTROL.phrase_id,
            "text": KIZZ_CONTROL.text,
            "ctc_transcript": KIZZ_CONTROL.ctc_transcript,
            "phones": list(KIZZ_CONTROL.phones),
        },
        "counts": dict(sorted(counts.items())),
        "examples": sorted(
            rows,
            key=lambda row: (
                row["split"],
                -int(row["label"]),
                row["provider"],
                row["voice"],
                row["source_id"],
            ),
        ),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def retain_active_rows(
    rows: Sequence[dict[str, Any]], tasks: Sequence[Task]
) -> tuple[list[dict[str, Any]], int]:
    """Remove stale recipe descriptors before a resumable generation pass."""
    active_descriptors = {task.descriptor for task in tasks}
    retained = [
        row
        for row in rows
        if row.get("descriptor_sha256") in active_descriptors
    ]
    return retained, len(rows) - len(retained)


def corpus_mix_report(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Verify provider use and voice disjointness over realized parent audio."""
    reasons: list[dict[str, Any]] = []
    splits: dict[str, Any] = {}
    voice_splits: dict[tuple[str, str], set[str]] = {}
    descriptors: set[str] = set()
    audio_hashes: set[str] = set()
    eligible = [row for row in rows if row.get("training_eligible") is True]
    causal_exclusions = []
    reserved = [
        row
        for row in rows
        if row.get("reserved_evidence_role") == "target_channel_positive"
    ]
    reserved_hashes = [str(row.get("audio_sha256", "")) for row in reserved]
    if len(reserved_hashes) != len(set(reserved_hashes)) or any(
        not value for value in reserved_hashes
    ):
        reasons.append({"reason": "reserved_replay_audio_not_unique"})
    reserved_contract: dict[str, Any] = {}
    for provider in sorted(RUNTIME_POSITIVE_PROVIDERS):
        provider_rows = [row for row in reserved if row.get("provider") == provider]
        voices = sorted({str(row.get("voice", "")) for row in provider_rows})
        reserved_contract[provider] = {
            "count": len(provider_rows),
            "voices": voices,
        }
        if len(provider_rows) != RESERVED_REPLAYS_PER_PROVIDER:
            reasons.append(
                {
                    "reason": "reserved_replay_provider_count",
                    "provider": provider,
                    "expected": RESERVED_REPLAYS_PER_PROVIDER,
                    "actual": len(provider_rows),
                }
            )
        if len(voices) < 2:
            reasons.append(
                {
                    "reason": "reserved_replay_voice_collapse",
                    "provider": provider,
                    "voices": voices,
                }
            )
        if any(
            row.get("split") != "test" or row.get("training_eligible") is not False
            for row in provider_rows
        ):
            reasons.append(
                {"reason": "reserved_replay_not_held_out", "provider": provider}
            )
    unexpected_reserved = sorted(
        {str(row.get("provider")) for row in reserved} - RUNTIME_POSITIVE_PROVIDERS
    )
    if unexpected_reserved:
        reasons.append(
            {
                "reason": "reserved_replay_unapproved_provider",
                "providers": unexpected_reserved,
            }
        )
    for row in rows:
        if int(row.get("label", -1)) != 0:
            continue
        decision = causal_negative_decision(str(row.get("render_text", "")))
        if not decision["qualified"]:
            causal_exclusions.append(
                {
                    "source_id": row.get("source_id"),
                    "render_text": row.get("render_text"),
                    "reason": decision["reason"],
                }
            )
            if row.get("training_eligible") is True:
                reasons.append(
                    {
                        "reason": "causally_unlearnable_negative_is_eligible",
                        "source_id": row.get("source_id"),
                    }
                )
    for index, row in enumerate(eligible):
        descriptor = str(row.get("descriptor_sha256", ""))
        audio_hash = str(row.get("audio_sha256", ""))
        if not descriptor or descriptor in descriptors:
            reasons.append({"reason": "missing_or_duplicate_descriptor", "index": index})
        descriptors.add(descriptor)
        if not audio_hash or audio_hash in audio_hashes:
            reasons.append({"reason": "missing_or_duplicate_audio", "index": index})
        audio_hashes.add(audio_hash)
        voice_splits.setdefault(
            (str(row.get("provider", "")), str(row.get("voice", ""))), set()
        ).add(str(row.get("split", "")))
    for (provider, voice), assigned in sorted(voice_splits.items()):
        if len(assigned) != 1:
            reasons.append(
                {
                    "reason": "voice_crosses_splits",
                    "provider": provider,
                    "voice": voice,
                    "splits": sorted(assigned),
                }
            )
    for split in ("train", "validation", "test"):
        positives = [
            row
            for row in eligible
            if row.get("split") == split and int(row.get("label", -1)) == 1
        ]
        negatives = [
            row
            for row in eligible
            if row.get("split") == split and int(row.get("label", -1)) == 0
        ]
        counts = {
            provider: sum(row.get("provider") == provider for row in positives)
            for provider in sorted(EXPECTED_PROVIDERS)
        }
        total = len(positives)
        shares = {
            provider: count / total if total else 0.0
            for provider, count in counts.items()
        }
        missing = sorted(provider for provider, count in counts.items() if not count)
        if missing:
            reasons.append(
                {"reason": "positive_provider_missing", "split": split, "providers": missing}
            )
        if shares and max(shares.values()) > 0.35 + 1e-12:
            provider = max(shares, key=lambda item: (shares[item], item))
            reasons.append(
                {
                    "reason": "positive_provider_overrepresented",
                    "split": split,
                    "provider": provider,
                    "share": shares[provider],
                    "maximum": 0.35,
                }
            )
        if not negatives:
            reasons.append({"reason": "collision_split_empty", "split": split})
        splits[split] = {
            "positive_count": total,
            "collision_count": len(negatives),
            "positive_provider_counts": counts,
            "positive_provider_shares": shares,
        }
    return {
        "schema_version": 1,
        "qualified": not reasons,
        "expected_positive_providers": sorted(EXPECTED_PROVIDERS),
        "maximum_positive_provider_share": 0.35,
        "reserved_replay_contract": reserved_contract,
        "splits": splits,
        "causal_negative_exclusions": causal_exclusions,
        "violations": reasons,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument(
        "--provider",
        action="append",
        choices=("assemblyai", "deepgram", "elevenlabs", "kokoro", "macos-say"),
        help="Generate only selected providers; repeat for more than one.",
    )
    parser.add_argument("--max-paid-audio-seconds", type=float, default=1200.0)
    parser.add_argument(
        "--max-new-tasks",
        type=int,
        help="Bound newly attempted tasks for a resumable provider pilot.",
    )
    parser.add_argument(
        "--kokoro-python",
        type=Path,
        default=Path("/private/tmp/kokoro-venv/bin/python"),
    )
    args = parser.parse_args(argv)
    if args.max_paid_audio_seconds <= 0:
        parser.error("--max-paid-audio-seconds must be positive")
    providers = set(args.provider or ()) or set(EXPECTED_PROVIDERS)
    root = args.output.resolve()
    audio_root = root / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    errors_path = root / "errors.jsonl"
    planned_tasks = _tasks()
    planned_by_descriptor = {task.descriptor: task for task in planned_tasks}
    rows: list[dict[str, Any]] = []
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows, pruned_obsolete_rows = retain_active_rows(
            list(payload.get("examples", [])), planned_tasks
        )
    else:
        pruned_obsolete_rows = 0
    # Normalize manifests written by older versions of this tool. Exact audio
    # duplicates remain auditable but cannot masquerade as distinct parents.
    first_by_audio: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: str(item.get("source_id", ""))):
        task = planned_by_descriptor.get(str(row.get("descriptor_sha256", "")))
        if task is not None and is_reserved_replay_task(task):
            row["training_eligible"] = False
            row["exclusion_reason"] = "reserved_for_device_replay"
            row["reserved_evidence_role"] = "target_channel_positive"
        elif row.get("reserved_evidence_role") == "target_channel_positive":
            row.pop("reserved_evidence_role", None)
            if row.get("exclusion_reason") == "reserved_for_device_replay":
                row.pop("exclusion_reason", None)
            if int(row.get("label", -1)) == 1:
                row["training_eligible"] = True
        if int(row.get("label", -1)) == 0:
            decision = causal_negative_decision(str(row.get("render_text", "")))
            row["causal_negative_contract"] = decision
            if not decision["qualified"]:
                row["training_eligible"] = False
                row["exclusion_reason"] = str(decision["reason"])
        audio_hash = str(row.get("audio_sha256", ""))
        if not audio_hash:
            continue
        first = first_by_audio.get(audio_hash)
        if first is None:
            first_by_audio[audio_hash] = row
        elif row.get("training_eligible") is True:
            row["training_eligible"] = False
            row["exclusion_reason"] = "duplicate_audio"
            row["duplicate_of_source_id"] = first.get("source_id")
    by_descriptor = {row.get("descriptor_sha256"): row for row in rows}
    paid_seconds = sum(
        float(row.get("duration_seconds", 0.0))
        for row in rows
        if row.get("provider") in PAID_PROVIDERS
    )
    env = _load_env(args.env_file)
    keys = {
        "assemblyai": env.get("ASSEMBLYAI_API_KEY"),
        "deepgram": env.get("DEEPGRAM_API_KEY"),
        "elevenlabs": env.get("ELEVEN_LABS_API_KEY")
        or env.get("ELEVENLABS_API_KEY"),
    }
    attempted = succeeded = skipped = 0
    kokoro_worker = (
        KokoroWorker(args.kokoro_python) if "kokoro" in providers else None
    )
    for task in planned_tasks:
        if task.provider not in providers:
            continue
        if args.max_new_tasks is not None and attempted >= args.max_new_tasks:
            break
        attempted += 1
        previous = by_descriptor.get(task.descriptor)
        if previous is not None:
            previous_path = Path(previous["path"])
            if (
                previous_path.is_file()
                and _sha256_file(previous_path) == previous.get("audio_sha256")
            ):
                skipped += 1
                continue
            rows.remove(previous)
            by_descriptor.pop(task.descriptor, None)
        if task.provider in PAID_PROVIDERS and paid_seconds >= args.max_paid_audio_seconds:
            raise RuntimeError("paid-audio duration guard reached")
        if task.provider in keys and not keys[task.provider]:
            raise RuntimeError(f"{task.provider} API key is not configured")
        provider_root = audio_root / task.provider / task.split
        provider_root.mkdir(parents=True, exist_ok=True)
        output = provider_root / f"{task.descriptor[:20]}.wav"
        metadata: dict[str, Any] = {}
        try:
            if task.provider == "elevenlabs":
                _elevenlabs(task, str(keys[task.provider]), output)
            elif task.provider == "deepgram":
                metadata = _deepgram(task, str(keys[task.provider]), output)
            elif task.provider == "assemblyai":
                metadata = asyncio.run(
                    _assemblyai(task, str(keys[task.provider]), output)
                )
            elif task.provider == "kokoro":
                if kokoro_worker is None:  # pragma: no cover - provider set controls this.
                    raise AssertionError("Kokoro worker was not initialized")
                kokoro_worker.render(task, output)
            elif task.provider == "macos-say":
                _macos(task, output)
            else:  # pragma: no cover - argparse/task table closes this branch.
                raise AssertionError(task.provider)
            duration = _wav_duration(output)
            if not 0.20 <= duration <= 12.0:
                raise ValueError(f"generated duration outside bounds: {duration:.3f}s")
            row = _row(task, output, metadata)
            duplicate = first_by_audio.get(row["audio_sha256"])
            if duplicate is not None:
                row["training_eligible"] = False
                row["exclusion_reason"] = "duplicate_audio"
                row["duplicate_of_source_id"] = duplicate.get("source_id")
            else:
                first_by_audio[row["audio_sha256"]] = row
            rows.append(row)
            by_descriptor[task.descriptor] = row
            succeeded += 1
            if task.provider in PAID_PROVIDERS:
                paid_seconds += duration
            _write_manifest(manifest_path, rows)
            print(
                json.dumps(
                    {
                        "attempt": attempted,
                        "provider": task.provider,
                        "voice": task.voice,
                        "split": task.split,
                        "label": task.label,
                        "duration_seconds": duration,
                        "paid_audio_seconds": paid_seconds,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        except Exception as error:
            output.unlink(missing_ok=True)
            with errors_path.open("a", encoding="utf-8") as errors:
                errors.write(
                    json.dumps(
                        {
                            "descriptor_sha256": task.descriptor,
                            "provider": task.provider,
                            "voice": task.voice,
                            "split": task.split,
                            "label": task.label,
                            "text": task.text,
                            "error": repr(error),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            print(
                json.dumps(
                    {
                        "attempt": attempted,
                        "provider": task.provider,
                        "voice": task.voice,
                        "error": repr(error),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if kokoro_worker is not None:
        kokoro_worker.close()
    _write_manifest(manifest_path, rows)
    mix_report = corpus_mix_report(rows)
    summary = {
        "attempted": attempted,
        "succeeded_this_run": succeeded,
        "resumed": skipped,
        "total_examples": len(rows),
        "paid_audio_seconds": paid_seconds,
        "pruned_obsolete_manifest_rows": pruned_obsolete_rows,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "complete_mix_requested": providers == set(EXPECTED_PROVIDERS),
        "mix_contract": mix_report,
    }
    (root / "generation-report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not summary["complete_mix_requested"] or mix_report["qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
