#!/usr/bin/env python3
"""Build the first Kizz manifest after the legacy Piper purge.

This intentionally accepts only explicit, non-Piper roots.  It records an
audio-content hash for every example and refuses conflicting duplicate labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split_for(value: str, *, train: int = 8, validation: int = 1) -> str:
    bucket = int(hashlib.sha256(value.encode()).hexdigest()[:8], 16) % 10
    if bucket < train:
        return "train"
    if bucket < train + validation:
        return "validation"
    return "test"


def audio_files(root: Path, suffixes: tuple[str, ...] = (".wav", ".flac")) -> list[Path]:
    return sorted(
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )


def add(
    rows: list[dict],
    path: Path,
    *,
    label: int,
    source_group: str,
    split: str,
    speaker_id: str,
    session_id: str,
) -> None:
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"missing audio path: {path}")
    rows.append(
        {
            "path": str(path),
            "label": int(label),
            "source_group": source_group,
            "split": split,
            "speaker_id": speaker_id,
            "session_id": session_id,
            "audio_sha256": digest(path),
        }
    )


def deduplicate(rows: Iterable[dict]) -> list[dict]:
    seen: dict[str, tuple[int, str]] = {}
    result = []
    for row in rows:
        key = row["audio_sha256"]
        identity = (int(row["label"]), row["source_group"])
        previous = seen.get(key)
        if previous is not None:
            if previous[0] != identity[0]:
                raise ValueError(f"duplicate audio has conflicting labels: {key}")
            continue
        seen[key] = identity
        result.append(row)
    return result


def build(args: argparse.Namespace) -> dict:
    rows: list[dict] = []

    add(
        rows,
        args.human_positive,
        label=1,
        source_group="human_device",
        split="train",
        speaker_id="human:muness",
        session_id="human:muness:kizz-train-20260821",
    )

    # Use one room-scale export only; v18/v19/v26 were byte-identical.
    for path in audio_files(args.device_positive_root):
        stem = path.stem
        speaker = stem.split("el-device-train-kizz-")[-1].rsplit("-", 1)[0]
        add(
            rows,
            path,
            label=1,
            source_group="device_replay",
            split=split_for(str(path), train=8, validation=1),
            speaker_id=f"device-replay:{speaker}",
            session_id=f"device-replay:roomscale-v18:{path.name}",
        )

    for path in audio_files(args.kokoro_root):
        if path.parent != args.kokoro_root.resolve():
            continue
        add(
            rows,
            path,
            label=1,
            source_group="kokoro_synthetic",
            split="train",
            speaker_id=f"kokoro:{path.stem}",
            session_id=f"kokoro:{path.stem}:pilot",
        )

    for path in audio_files(args.labeled_negative_root):
        relative = path.relative_to(args.labeled_negative_root.resolve())
        split = next((part for part in relative.parts if part in {"train", "validation", "test"}), None)
        if split is None:
            continue
        speaker_parts = [part for part in relative.parts if part.startswith("labeled-kizz_")]
        speaker = speaker_parts[-1] if speaker_parts else relative.parts[-2]
        add(
            rows,
            path,
            label=0,
            source_group="labeled_tts_collision",
            split=split,
            speaker_id=f"elevenlabs:{speaker}",
            session_id=f"elevenlabs:{speaker}:{relative.parts[0]}:{split}",
        )

    for path in audio_files(args.kokoro_root / "collisions"):
        add(
            rows,
            path,
            label=0,
            source_group="kokoro_collision",
            split=split_for(path.stem, train=7, validation=2),
            speaker_id=f"kokoro:{path.stem.split('--', 1)[0]}",
            session_id=f"kokoro-collision:{path.stem}",
        )

    for path in audio_files(args.public_speech_root):
        relative = path.relative_to(args.public_speech_root.resolve())
        speaker = relative.parts[0]
        chapter = relative.parts[1] if len(relative.parts) > 1 else "unknown"
        add(
            rows,
            path,
            label=0,
            source_group="public_speech",
            split=split_for(speaker, train=8, validation=1),
            speaker_id=f"librispeech:{speaker}",
            session_id=f"librispeech:{speaker}:{chapter}",
        )

    for path in audio_files(args.background_root):
        source = str(path.relative_to(args.background_root.resolve()).parent)
        add(
            rows,
            path,
            label=0,
            source_group="background",
            split=split_for(str(path), train=8, validation=1),
            speaker_id=f"background:{source}",
            session_id=f"background:{source}:{split_for(str(path), train=8, validation=1)}",
        )

    false_wake = json.loads(args.false_wake_manifest.read_text())
    observations = false_wake.get("observations", [])
    if len(observations) < 12:
        raise ValueError("need at least 12 reviewed false-wake observations")
    ordered = sorted(observations, key=lambda row: row["observation_id"])
    for index, item in enumerate(ordered):
        path = (args.false_wake_manifest.parent / item["path"]).resolve()
        split = "train" if index < len(ordered) - 12 else "test"
        add(
            rows,
            path,
            label=0,
            source_group="device_false_wake",
            split=split,
            speaker_id="device:kizz-1",
            session_id=f"false-wake:{item['observation_id']}",
        )

    rows = deduplicate(rows)
    train_caps = {
        "labeled_tts_collision": 7,
        "public_speech": 7,
        "background": 7,
        "device_false_wake": 50,
        "kokoro_collision": 7,
    }
    capped: list[dict] = []
    for source_group in sorted({row["source_group"] for row in rows}):
        group = [
            row
            for row in rows
            if row["source_group"] == source_group and row["split"] == "train"
        ]
        cap = train_caps.get(source_group)
        if cap is not None and len(group) > cap:
            group = sorted(group, key=lambda row: row["audio_sha256"])[:cap]
        capped.extend(group)
    capped.extend(row for row in rows if row["split"] != "train")
    rows = capped
    payload = {
        "schema_version": 2,
        "sample_rate": 16_000,
        "context_samples": 32_000,
        "examples": sorted(rows, key=lambda row: (row["split"], row["label"], row["path"])),
        "clean_slate": {
            "piper_policy": "excluded",
            "device_positive_root": str(args.device_positive_root.resolve()),
            "kokoro_root": str(args.kokoro_root.resolve()),
            "labeled_negative_root": str(args.labeled_negative_root.resolve()),
            "public_speech_root": str(args.public_speech_root.resolve()),
            "background_root": str(args.background_root.resolve()),
            "false_wake_manifest": str(args.false_wake_manifest.resolve()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-positive", type=Path, required=True)
    parser.add_argument("--device-positive-root", type=Path, required=True)
    parser.add_argument("--kokoro-root", type=Path, required=True)
    parser.add_argument("--labeled-negative-root", type=Path, required=True)
    parser.add_argument("--public-speech-root", type=Path, required=True)
    parser.add_argument("--background-root", type=Path, required=True)
    parser.add_argument("--false-wake-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = build(args)
    print(json.dumps({"output": str(args.output), "examples": len(payload["examples"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
