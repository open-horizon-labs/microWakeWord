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


def directory_examples(root: Path, label: int, source_id: str) -> list[dict]:
    return [
        {"path": str(path.resolve()), "label": int(label), "source_id": source_id}
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
        examples.append(
            {"path": str(path), "label": 0, "source_id": f"device_false_wake_{split}"}
        )
    if {item["source_id"] for item in examples} != {f"device_false_wake_{split}"}:
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

    examples = directory_examples(args.positive_root, 1, "canonical_positive")
    for directory in sorted(args.negative_root.iterdir()):
        if not directory.is_dir():
            continue
        for split in ("train",):
            split_dir = directory / split
            if split_dir.is_dir():
                examples.extend(directory_examples(split_dir, 0, f"hard_negative:{directory.name}"))
    examples.extend(false_wake_examples(args.false_wake_manifest, args.false_wake_root, "training"))
    payload = {
        "schema_version": 1,
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
