#!/usr/bin/env python3
"""Reuse provenance-bound labeled TTS without calling the provider again."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import yaml

if __package__:
    from tools.add_labeled_voice_samples import slug
else:
    from add_labeled_voice_samples import slug


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(root: Path) -> tuple[Path, dict]:
    path = root / "generation-manifest.json"
    if not path.is_file():
        raise ValueError(f"missing generation manifest: {path}")
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 2:
        raise ValueError(f"unsupported generation manifest: {path}")
    return path, manifest


def _link_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256(destination) != _sha256(source):
            raise ValueError(f"existing reused sample differs: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def reuse_labeled_samples(
    recipe_path: Path, generated: Path, source_generated: Path
) -> dict:
    """Import matching labeled voices, allowing an intentional class relabel."""
    recipe = yaml.safe_load(recipe_path.read_text())
    target_path, target = _manifest(generated)
    _, source = _manifest(source_generated)
    recipe_hash = _sha256(recipe_path)
    if target.get("recipe_sha256") != recipe_hash:
        raise ValueError("target generated corpus does not match the recipe")

    labels: dict[str, str] = {}
    for class_name, key in (
        ("positive", "positive_phrases"),
        ("hard_negative", "hard_negative_phrases"),
    ):
        for phrase in recipe[key]:
            text = phrase["text"]
            if text in labels:
                raise ValueError(f"recipe labels the same text twice: {text}")
            labels[text] = class_name

    source_items: dict[str, list[dict]] = {}
    for item in source.get("plan", []):
        if item.get("speaker_id") and item.get("text") in labels:
            source_items.setdefault(item["text"], []).append(item)
    missing = sorted(set(labels) - set(source_items))
    if missing:
        raise ValueError(f"source lacks labeled voice samples for: {missing}")

    reused = []
    for text, class_name in labels.items():
        for source_item in source_items[text]:
            source_dir = Path(source_item["output"])
            metadata = source_dir / "synthesis-metadata.jsonl"
            wavs = sorted(source_dir.glob("*.wav"))
            if len(wavs) != int(source_item["samples"]) or not metadata.is_file():
                raise ValueError(f"incomplete labeled source: {source_dir}")
            records = [
                json.loads(line)
                for line in metadata.read_text().splitlines()
                if line.strip()
            ]
            if len(records) != len(wavs) or {
                record.get("file") for record in records
            } != {wav.name for wav in wavs}:
                raise ValueError(f"invalid labeled provenance: {metadata}")

            destination = (
                generated
                / class_name
                / slug(text)
                / source_item["split"]
                / f"labeled-{slug(source_item['speaker_name'])}"
            )
            for wav in wavs:
                _link_file(wav, destination / wav.name)
            _link_file(metadata, destination / metadata.name)
            reused.append(
                {
                    **source_item,
                    "class": class_name,
                    "output": str(destination),
                    "reused_from": str(source_dir),
                    "source_generation_manifest_sha256": _sha256(
                        source_generated / "generation-manifest.json"
                    ),
                }
            )

    target["plan"] = [
        item for item in target.get("plan", []) if not item.get("speaker_id")
    ] + reused
    target["labeled_voice_catalog"] = source.get("labeled_voice_catalog")
    target["labeled_voice_catalog_sha256"] = source.get("labeled_voice_catalog_sha256")
    target["reused_labeled_from"] = str(source_generated)
    with tempfile.NamedTemporaryFile(
        "w", dir=generated, prefix="generation-manifest.", delete=False
    ) as temporary:
        json.dump(target, temporary, indent=2)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(target_path)
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--reuse-generated", type=Path, required=True)
    args = parser.parse_args()
    manifest = reuse_labeled_samples(args.recipe, args.generated, args.reuse_generated)
    print(
        json.dumps(
            {
                "labeled_plan_items": sum(
                    bool(item.get("speaker_id")) for item in manifest["plan"]
                ),
                "reused_from": manifest["reused_labeled_from"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
