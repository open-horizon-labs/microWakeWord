"""Validation helpers for real-device wake-word corpora."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import wave

MANIFEST_NAME = "device-corpus.json"
TRUTHS = {"positive", "hard_negative", "ambient_negative"}
SPLITS = {"train", "validation", "test"}
CAPTURE_SOURCES = {"human", "synthetic_playback", "ambient", "simulated"}
SPEAKER_KINDS = {"human", "synthetic", "ambient"}
AGE_GROUPS = {"child", "adult", "unknown", "not_applicable"}
REQUIRED_AUDIO = {
    "sample_rate": 16000,
    "channels": 1,
    "sample_format": "s16le",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required(item: dict, key: str, kind: type):
    value = item.get(key)
    if not isinstance(value, kind) or (kind is str and not value):
        raise ValueError(f"capture requires non-empty {key}")
    return value


def _validate_phrase_span(item: dict, capture_id: str, duration_ms: float) -> None:
    span = item.get("phrase_span")
    if span is None:
        return
    if not isinstance(span, dict):
        raise ValueError(f"capture {capture_id} phrase_span must be an object")
    start_ms = span.get("start_ms")
    end_ms = span.get("end_ms")
    if (
        not isinstance(start_ms, (int, float))
        or isinstance(start_ms, bool)
        or not isinstance(end_ms, (int, float))
        or isinstance(end_ms, bool)
    ):
        raise ValueError(
            f"capture {capture_id} phrase_span requires numeric start_ms and end_ms"
        )
    if start_ms < 0 or end_ms <= start_ms or end_ms > duration_ms:
        raise ValueError(
            f"capture {capture_id} phrase_span must satisfy "
            f"0 <= start_ms < end_ms <= {duration_ms:g}"
        )


def validate_device_corpus(root: Path) -> dict:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(f"missing device corpus manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 2:
        raise ValueError("device corpus schema_version must be 2")
    _required(manifest, "corpus_id", str)
    profiles = manifest.get("device_profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("device corpus requires device_profiles")
    for profile_name, profile in profiles.items():
        if (
            not isinstance(profile_name, str)
            or not profile_name
            or not isinstance(profile, dict)
        ):
            raise ValueError("each device profile requires a non-empty name and object")
        audio = profile.get("audio")
        if not isinstance(audio, dict):
            raise ValueError(f"device profile {profile_name} requires audio")
        for key, expected in REQUIRED_AUDIO.items():
            if audio.get(key) != expected:
                raise ValueError(
                    f"device profile {profile_name} audio {key} must be {expected}"
                )
        for key in ("frontend", "gain_profile"):
            if not isinstance(audio.get(key), str) or not audio[key]:
                raise ValueError(f"device profile {profile_name} requires audio {key}")
        if not isinstance(audio.get("preprocessing", {}), dict):
            raise ValueError(
                f"device profile {profile_name} preprocessing must be an object"
            )
    speakers = manifest.get("speakers")
    if not isinstance(speakers, dict) or not speakers:
        raise ValueError("device corpus requires registered speakers")
    for speaker_id, speaker in speakers.items():
        if (
            not isinstance(speaker_id, str)
            or not speaker_id
            or not isinstance(speaker, dict)
        ):
            raise ValueError(
                "each registered speaker requires a non-empty ID and object"
            )
        kind = speaker.get("kind")
        age_group = speaker.get("age_group")
        split = speaker.get("split")
        if kind not in SPEAKER_KINDS:
            raise ValueError(f"registered speaker {speaker_id} has invalid kind")
        if age_group not in AGE_GROUPS:
            raise ValueError(f"registered speaker {speaker_id} has invalid age_group")
        if split not in SPLITS:
            raise ValueError(f"registered speaker {speaker_id} has invalid split")
        if kind == "human":
            if age_group not in {"child", "adult"}:
                raise ValueError(
                    f"human speaker {speaker_id} requires child or adult age_group"
                )
            if speaker.get("identity_verified") is not True:
                raise ValueError(
                    f"human speaker {speaker_id} requires identity_verified=true"
                )
        elif kind == "ambient" and age_group != "not_applicable":
            raise ValueError(
                f"ambient speaker {speaker_id} requires age_group=not_applicable"
            )
    captures = manifest.get("captures")
    if not isinstance(captures, list):
        raise ValueError("device corpus captures must be a list")

    capture_ids: set[str] = set()
    speaker_splits: dict[str, str] = {}
    session_splits: dict[str, str] = {}
    device_profiles: dict[str, str] = {}
    for item in captures:
        if not isinstance(item, dict):
            raise ValueError("each device capture must be an object")
        capture_id = _required(item, "capture_id", str)
        if capture_id in capture_ids:
            raise ValueError(f"duplicate capture_id: {capture_id}")
        capture_ids.add(capture_id)
        truth = _required(item, "truth", str)
        source = _required(item, "source", str)
        split = _required(item, "split", str)
        speaker = _required(item, "speaker_id", str)
        session = _required(item, "session_id", str)
        device_id = _required(item, "device_id", str)
        device_profile = _required(item, "device_profile", str)
        if device_profile not in profiles:
            raise ValueError(f"capture {capture_id} references unknown device_profile")
        if (
            device_id in device_profiles
            and device_profiles[device_id] != device_profile
        ):
            raise ValueError(f"device {device_id} crosses device profiles")
        device_profiles[device_id] = device_profile
        _required(item, "phrase", str)
        if truth not in TRUTHS:
            raise ValueError(f"unsupported truth for {capture_id}: {truth}")
        if source not in CAPTURE_SOURCES:
            raise ValueError(f"unsupported source for {capture_id}: {source}")
        if truth == "ambient_negative" and source not in {"ambient", "simulated"}:
            raise ValueError(
                f"ambient capture {capture_id} requires ambient or simulated source"
            )
        if split not in SPLITS:
            raise ValueError(f"unsupported split for {capture_id}: {split}")
        if speaker not in speakers:
            raise ValueError(f"capture {capture_id} references unregistered speaker")
        speaker_profile = speakers[speaker]
        if speaker_profile["split"] != split:
            raise ValueError(
                f"capture {capture_id} split differs from registered speaker {speaker}"
            )
        expected_kind = {
            "human": "human",
            "synthetic_playback": "synthetic",
            "ambient": "ambient",
        }.get(source)
        if expected_kind and speaker_profile["kind"] != expected_kind:
            raise ValueError(
                f"capture {capture_id} source does not match registered speaker kind"
            )
        if not isinstance(item.get("detected"), bool):
            raise ValueError(f"capture {capture_id} requires boolean detected")
        if speaker in speaker_splits and speaker_splits[speaker] != split:
            raise ValueError(f"speaker {speaker} crosses corpus splits")
        if session in session_splits and session_splits[session] != split:
            raise ValueError(f"session {session} crosses corpus splits")
        speaker_splits[speaker] = split
        session_splits[session] = split

        relative = Path(_required(item, "path", str))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"capture {capture_id} path must remain inside corpus")
        wav_path = root / relative
        if not wav_path.is_file():
            raise ValueError(f"capture audio is missing: {wav_path}")
        with wave.open(str(wav_path), "rb") as wav:
            if (
                wav.getnchannels() != 1
                or wav.getsampwidth() != 2
                or wav.getframerate() != 16000
                or wav.getcomptype() != "NONE"
            ):
                raise ValueError(
                    f"capture {capture_id} must be mono 16 kHz signed-16 PCM WAV"
                )
            if wav.getnframes() != item.get("samples"):
                raise ValueError(
                    f"capture {capture_id} sample count does not match WAV"
                )
            duration_ms = wav.getnframes() * 1000 / wav.getframerate()
            _validate_phrase_span(item, capture_id, duration_ms)
        if sha256(wav_path) != item.get("sha256"):
            raise ValueError(f"capture {capture_id} SHA-256 does not match WAV")
    return manifest


def captures_for(
    root: Path, manifest: dict, truth: str, split: str | None = None
) -> list[tuple[dict, Path]]:
    return [
        (item, root / item["path"])
        for item in manifest["captures"]
        if item["truth"] == truth and (split is None or item["split"] == split)
    ]
