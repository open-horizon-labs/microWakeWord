#!/usr/bin/env python3
"""Promote only canonical, pre-existing device positives into Kizz v32."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from microwakeword.device_corpus import validate_device_corpus

CANONICAL_PRONUNCIATIONS = {
    "hi_fi",
    "hi_fi_kizz",
    "hi_fi_repeated",
    "natural-close",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_promotion_manifest(corpus: Path, device_manifest: dict) -> dict:
    entries = []
    for item in device_manifest["captures"]:
        if (
            item.get("truth") != "positive"
            or item.get("split") not in {"train", "test"}
            or item.get("pronunciation") not in CANONICAL_PRONUNCIATIONS
            or item.get("source") not in {"human", "synthetic_playback"}
            or not isinstance(item.get("phrase_span"), dict)
        ):
            continue
        # Keep phrase-span coordinates bound to the original device capture.
        # Feature generation performs its own deterministic aligned crop.
        audio = corpus / item["path"]
        if not audio.is_file():
            raise ValueError(f"selected device audio is missing: {audio}")
        entries.append(
            {
                "id": item["capture_id"],
                "wav_path": str(audio.resolve()),
                "sha256": sha256(audio),
                "truth": "positive",
                "split": item["split"],
                "text": "Hi-Fi Kizz",
                "phrase_span": item["phrase_span"],
                "provenance": (
                    f"device-corpus:{device_manifest['corpus_id']}:"
                    f"{item['capture_id']}:{item['sha256']}"
                ),
                "human_reviewed": True,
                "training_eligible": True,
                "source_capture": item,
                "selection_basis": (
                    "validated pre-existing device corpus; canonical "
                    "pronunciation allowlist; original phrase-span coordinates"
                ),
            }
        )
    if not entries:
        raise ValueError("no canonical device positives matched the promotion policy")
    speakers_by_split: dict[str, set[str]] = {}
    for entry in entries:
        speaker = entry["source_capture"].get("speaker_id")
        if not isinstance(speaker, str) or not speaker:
            raise ValueError(f"selected capture lacks speaker identity: {entry['id']}")
        speakers_by_split.setdefault(entry["split"], set()).add(speaker)
    train_test_overlap = speakers_by_split.get("train", set()) & speakers_by_split.get(
        "test", set()
    )
    if train_test_overlap:
        raise ValueError(
            "canonical device speakers overlap between train and test: "
            f"{sorted(train_test_overlap)}"
        )
    return {
        "schema_version": 1,
        "purpose": "Canonical device-path positives for the Kizz v32 recipe",
        "source_corpus": str(corpus.resolve()),
        "source_corpus_manifest_sha256": sha256(corpus / "device-corpus.json"),
        "canonical_pronunciations": sorted(CANONICAL_PRONUNCIATIONS),
        "speaker_ids_by_split": {
            split: sorted(speakers)
            for split, speakers in sorted(speakers_by_split.items())
        },
        "entries": sorted(
            entries,
            key=lambda entry: (
                {"train": 0, "validation": 1, "test": 2}[entry["split"]],
                entry["id"],
            ),
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device_manifest = validate_device_corpus(args.corpus)
    promotion = build_promotion_manifest(args.corpus, device_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(promotion, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "entries": len(promotion["entries"]),
                "train": sum(
                    entry["split"] == "train" for entry in promotion["entries"]
                ),
                "test": sum(entry["split"] == "test" for entry in promotion["entries"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
