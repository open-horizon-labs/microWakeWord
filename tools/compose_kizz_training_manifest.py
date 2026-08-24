#!/usr/bin/env python3
"""Compose a gated Kizz manifest from explicit acoustic source groups."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence


def rank(path: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{path}".encode()).hexdigest()


def select_training(values: list[dict], limit: int, seed: int) -> list[dict]:
    if len(values) <= limit:
        return values
    return sorted(values, key=lambda item: rank(item["path"], seed))[:limit]


def device_replay_examples(corpus_path: Path) -> list[dict]:
    corpus = json.loads(corpus_path.read_text())
    examples = []
    for capture in corpus.get("captures", []):
        if capture.get("truth") != "positive":
            continue
        path = (corpus_path.parent / capture["path"]).resolve()
        examples.append(
            {
                "path": str(path),
                "label": 1,
                "source_group": "device_replay",
                "split": capture.get("split", "train"),
                "speaker_id": f"device:{capture['speaker_id']}",
                "session_id": f"device:{capture['session_id']}",
            }
        )
    return examples


def device_positive_examples(root: Path) -> list[dict]:
    examples = []
    for path in sorted(root.rglob("*.wav")):
        speaker = "muness" if path.stem.startswith("muness-") else path.stem.split("-kizz-")[-1].rsplit("-", 1)[0]
        examples.append(
            {
                "path": str(path.resolve()),
                "label": 1,
                "source_group": "device_replay",
                "split": "train",
                "speaker_id": f"device:{speaker}",
                "session_id": f"device:{root.parent.name}",
            }
        )
    return examples


def public_speech_examples(root: Path) -> list[dict]:
    examples = []
    for path in sorted(root.rglob("*.flac")):
        relative = path.relative_to(root)
        parts = relative.parts
        if len(parts) < 3:
            continue
        speaker, chapter = parts[0], parts[1]
        examples.append(
            {
                "path": str(path.resolve()),
                "label": 0,
                "source_group": "public_speech",
                "split": "train",
                "speaker_id": f"librispeech:{speaker}",
                "session_id": f"librispeech:{speaker}:{chapter}",
            }
        )
    return examples


def background_examples(root: Path) -> list[dict]:
    examples = []
    for path in sorted(root.rglob("*.wav")):
        source = str(path.relative_to(root).parent)
        examples.append(
            {
                "path": str(path.resolve()),
                "label": 0,
                "source_group": "background",
                "split": "train",
                "speaker_id": f"background:{source}",
                "session_id": f"background:{source}",
            }
        )
    return examples


def compose(
    base_manifest: Path,
    device_corpus: Path,
    device_positive_root: Path,
    public_speech_root: Path,
    background_root: Path,
    output: Path,
    *,
    seed: int,
    public_speech_limit: int,
    background_limit: int,
) -> dict:
    base = json.loads(base_manifest.read_text())
    if base.get("schema_version") != 2:
        raise ValueError("base manifest must use schema_version 2")
    examples = []
    for item in base["examples"]:
        if "piper" in item.get("source_group", "").lower():
            continue
        examples.append(item)

    device_examples = device_replay_examples(device_corpus) + device_positive_examples(device_positive_root)
    seen_audio: set[str] = set()
    unique_device_examples = []
    for item in device_examples:
        digest = hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()
        if digest in seen_audio:
            continue
        seen_audio.add(digest)
        unique_device_examples.append(item)
    public_examples = select_training(public_speech_examples(public_speech_root), public_speech_limit, seed)
    ambient_examples = select_training(background_examples(background_root), background_limit, seed)
    examples.extend(unique_device_examples)
    examples.extend(public_examples)
    examples.extend(ambient_examples)

    paths = [item["path"] for item in examples]
    if len(paths) != len(set(paths)):
        raise ValueError("composed manifest contains duplicate paths")
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise ValueError(f"composed manifest has missing paths: {missing[:3]}")

    result = {
        "schema_version": 2,
        "sample_rate": 16_000,
        "context_samples": 32_000,
        "examples": sorted(examples, key=lambda item: (item["split"], item["label"], item["path"])),
        "composition": {
            "base_manifest": str(base_manifest.resolve()),
            "device_corpus": str(device_corpus.resolve()),
            "device_positive_root": str(device_positive_root.resolve()),
            "public_speech_root": str(public_speech_root.resolve()),
            "background_root": str(background_root.resolve()),
            "seed": seed,
            "training_caps": {
                "public_speech": public_speech_limit,
                "background": background_limit,
            },
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--device-corpus", type=Path, required=True)
    parser.add_argument("--device-positive-root", type=Path, required=True)
    parser.add_argument("--public-speech-root", type=Path, required=True)
    parser.add_argument("--background-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=24117)
    parser.add_argument("--public-speech-limit", type=int, default=40)
    parser.add_argument("--background-limit", type=int, default=20)
    args = parser.parse_args(argv)
    limits = (args.public_speech_limit, args.background_limit)
    if any(value < 0 for value in limits):
        parser.error("source limits must be non-negative")
    result = compose(
        args.base_manifest,
        args.device_corpus,
        args.device_positive_root,
        args.public_speech_root,
        args.background_root,
        args.output,
        seed=args.seed,
        public_speech_limit=args.public_speech_limit,
        background_limit=args.background_limit,
    )
    print(json.dumps({"output": str(args.output), "example_count": len(result["examples"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
