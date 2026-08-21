"""Validation helpers for real-device wake-word corpora."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import wave

MANIFEST_NAME = "device-corpus.json"
TRUTHS = {"positive", "hard_negative", "ambient_negative"}
SPLITS = {"train", "validation", "test"}
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


def validate_device_corpus(root: Path) -> dict:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(f"missing device corpus manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("device corpus schema_version must be 1")
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
        if split not in SPLITS:
            raise ValueError(f"unsupported split for {capture_id}: {split}")
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
