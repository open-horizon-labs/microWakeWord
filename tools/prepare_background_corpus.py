#!/usr/bin/env python3
"""Build provenance-bound indoor/outdoor augmentation and stress cohorts."""

from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from microwakeword.device_corpus import validate_device_corpus

ESC50_INDOOR = {
    "breathing",
    "brushing_teeth",
    "can_opening",
    "clapping",
    "clock_alarm",
    "clock_tick",
    "coughing",
    "crying_baby",
    "door_wood_creaks",
    "door_wood_knock",
    "drinking_sipping",
    "glass_breaking",
    "keyboard_typing",
    "laughing",
    "mouse_click",
    "sneezing",
    "snoring",
    "toilet_flush",
    "vacuum_cleaner",
    "washing_machine",
}

ESC50_OUTDOOR = {
    "airplane",
    "car_horn",
    "chainsaw",
    "chirping_birds",
    "church_bells",
    "crackling_fire",
    "crickets",
    "engine",
    "fireworks",
    "footsteps",
    "helicopter",
    "insects",
    "pouring_water",
    "rain",
    "sea_waves",
    "siren",
    "thunderstorm",
    "train",
    "water_drops",
    "wind",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(source) != sha256(destination):
            raise ValueError(f"existing background differs: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        shutil.copy2(source, destination)


def add_esc50(source: Path, output: Path) -> tuple[list[dict], dict]:
    metadata = source / "meta/esc50.csv"
    if not metadata.exists():
        raise ValueError(f"missing ESC-50 metadata: {metadata}")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    files = []
    with metadata.open(newline="") as rows:
        for row in csv.DictReader(rows):
            category = row["category"]
            if category in ESC50_INDOOR:
                environment = "indoor"
            elif category in ESC50_OUTDOOR:
                environment = "outdoor"
            else:
                continue
            evidence_split = "stress" if int(row["fold"]) == 5 else "train"
            audio = source / "audio" / row["filename"]
            destination = output / environment / evidence_split / row["filename"]
            hardlink(audio, destination)
            files.append(
                {
                    "path": destination.relative_to(output).as_posix(),
                    "source": "esc50",
                    "source_path": audio.relative_to(source).as_posix(),
                    "source_split": int(row["fold"]),
                    "environment": environment,
                    "evidence_split": evidence_split,
                    "category": category,
                    "sha256": sha256(audio),
                }
            )
    return files, {
        "name": "ESC-50",
        "url": "https://github.com/karolpiczak/ESC-50",
        "commit": commit,
        "license": "Creative Commons Attribution-NonCommercial 3.0",
    }


def add_device_ambient(corpus: Path, output: Path) -> tuple[list[dict], dict]:
    manifest = validate_device_corpus(corpus)
    files = []
    for capture in manifest["captures"]:
        if capture["truth"] != "ambient_negative" or capture["split"] != "train":
            continue
        audio = corpus / capture["path"]
        destination = output / "indoor/train" / f"device-{audio.name}"
        hardlink(audio, destination)
        files.append(
            {
                "path": destination.relative_to(output).as_posix(),
                "source": "device_corpus",
                "source_path": capture["path"],
                "capture_id": capture["capture_id"],
                "device_profile": capture["device_profile"],
                "environment": "indoor",
                "evidence_split": "train",
                "category": capture.get("pronunciation") or "room-tone",
                "sha256": capture["sha256"],
            }
        )
    return files, {
        "name": manifest["corpus_id"],
        "manifest_sha256": sha256(corpus / "device-corpus.json"),
    }


def prepare(
    output: Path,
    esc50: Path | None = None,
    device_corpus: Path | None = None,
) -> dict:
    if not esc50 and not device_corpus:
        raise ValueError("at least one background source is required")
    files = []
    sources = []
    if esc50:
        source_files, provenance = add_esc50(esc50, output)
        files.extend(source_files)
        sources.append(provenance)
    if device_corpus:
        source_files, provenance = add_device_ambient(device_corpus, output)
        files.extend(source_files)
        sources.append(provenance)
    manifest = {
        "schema_version": 1,
        "sources": sources,
        "counts": {
            environment: {
                split: sum(
                    item["environment"] == environment
                    and item["evidence_split"] == split
                    for item in files
                )
                for split in ("train", "stress")
            }
            for environment in ("indoor", "outdoor")
        },
        "files": sorted(files, key=lambda item: item["path"]),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "background-corpus.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--esc50", type=Path)
    parser.add_argument("--device-corpus", type=Path)
    args = parser.parse_args()
    manifest = prepare(args.output, args.esc50, args.device_corpus)
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
