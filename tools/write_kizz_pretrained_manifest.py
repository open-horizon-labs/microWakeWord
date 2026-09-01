#!/usr/bin/env python3
"""Write a provenance-bound raw-waveform manifest for the D teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from microwakeword.kizz_pretrained_teacher import list_audio_files


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_METADATA_CACHE: dict[Path, dict[Path, dict]] = {}


def metadata_for(path: Path) -> dict:
    """Recover speaker/session provenance from the generator sidecar when present."""
    for parent in (path.parent, *path.parents):
        metadata_path = parent / "synthesis-metadata.jsonl"
        if not metadata_path.is_file():
            continue
        rows = _METADATA_CACHE.get(metadata_path)
        if rows is None:
            rows = {}
            for line in metadata_path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    rows[(metadata_path.parent / row["file"]).resolve()] = row
            _METADATA_CACHE[metadata_path] = rows
        row = rows.get(path.resolve(), {})
        if row.get("speaker_id"):
            speaker_id = f"{row.get('provider', 'tts')}:{row['speaker_id']}"
        elif "speaker_1" in row or "speaker_2" in row:
            speaker_id = f"piper:{row.get('speaker_1')}:{row.get('speaker_2')}"
        else:
            speaker_id = None
        return {
            "speaker_id": speaker_id,
            "session_id": f"synthesis:{metadata_path.parent}",
        }
    return {}


def directory_examples(
    root: Path,
    label: int,
    source_group: str,
    split: str = "train",
) -> list[dict]:
    return [
        {
            "path": str(path.resolve()),
            "label": int(label),
            "source_group": source_group,
            "split": split,
            **metadata_for(path),
        }
        for path in list_audio_files(root)
    ]


def false_wake_examples(manifest_path: Path, root: Path, split: str) -> list[dict]:
    cache_path = root / "false-wake-feature-cache-v1" / "manifest.json"
    cache = json.loads(cache_path.read_text())
    selected = {
        item["observation_id"]
        for item in cache["splits"][split]["observations"]
    }
    manifest = json.loads(manifest_path.read_text())
    examples = []
    for item in manifest["observations"]:
        if item["observation_id"] not in selected:
            continue
        path = (manifest_path.parent / item["path"]).resolve()
        metadata_path = path.with_suffix(".json")
        observation = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        examples.append(
            {
                "path": str(path),
                "label": 0,
                "source_group": "device_false_wake",
                "split": "train",
                "speaker_id": f"device:{observation.get('device_id', 'unknown')}",
                "session_id": f"observation:{item['observation_id']}",
            }
        )
    if not examples:
        raise ValueError(f"false-wake split is empty: {split}")
    return examples


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-root", type=Path, required=True)
    parser.add_argument("--negative-root", type=Path, required=True)
    parser.add_argument("--false-wake-manifest", type=Path, required=True)
    parser.add_argument("--false-wake-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    positive_examples = []
    for path in list_audio_files(args.positive_root):
        relative_parts = path.relative_to(args.positive_root).parts
        source_group = (
            "labeled_tts"
            if any(part.startswith("labeled-kizz_") for part in relative_parts)
            else "piper_synthetic"
        )
        positive_examples.append(
            {
                "path": str(path.resolve()),
                "label": 1,
                "source_group": source_group,
                "split": next(
                    (part for part in relative_parts if part in {"train", "validation", "test"}),
                    "train",
                ),
                **metadata_for(path),
            }
        )
    examples = positive_examples
    negative_root = args.negative_root / "hard_negative"
    if not negative_root.is_dir():
        raise ValueError(f"negative root has no hard_negative directory: {negative_root}")
    for split_dir in sorted(negative_root.rglob("*")):
        if split_dir.is_dir() and split_dir.name in {"train", "validation", "test"}:
            examples.extend(
                directory_examples(split_dir, 0, "piper_hard_negative", split_dir.name)
            )
    examples.extend(false_wake_examples(args.false_wake_manifest, args.false_wake_root, "training"))
    payload = {
        "schema_version": 2,
        "sample_rate": 16_000,
        "context_samples": 32_000,
        "examples": examples,
        "source_manifest": str(args.false_wake_manifest.resolve()),
        "source_manifest_sha256": sha256_file(args.false_wake_manifest),
        "false_wake_split_manifest": str((args.false_wake_root / "false-wake-feature-cache-v1" / "manifest.json").resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "example_count": len(examples)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
