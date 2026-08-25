#!/usr/bin/env python3
"""Compose the fail-closed canonical Kizz corpus and locked anchor manifests.

Rendering text is not ground truth.  Every artifact receives an explicit
semantic label, target-phone contract, ancestry, duration, and immutable source
identity.  Near phrases are useful, but only as negatives.  Household wake
captures remain locked qualification anchors and never enter the training
manifest.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


CANONICAL_TARGET_ID = "hiphi_kizz"
CANONICAL_PHONES = ("h", "aɪ", "f", "aɪ", "k", "ɪ", "z")
CANONICAL_RENDER_TEXTS = frozenset(
    {"Hi-Fi Kizz", "Hi Phi Kizz", "HiPhi Kizz", "Hi-Phi Kizz"}
)
CANONICAL_DEVICE_PRONUNCIATIONS = frozenset(
    {"hi_fi", "hi_fi_kizz", "hi_phi_kizz", "natural-close"}
)
REPEATED_DEVICE_PRONUNCIATIONS = frozenset({"hi_fi_repeated"})
MIN_TRAINING_AUDIO_SECONDS = 0.20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_id(*values: object) -> str:
    return hashlib.sha256(
        "\0".join(str(value) for value in values).encode("utf-8")
    ).hexdigest()


def collision_label(text: str) -> str:
    """Return a deterministic negative family for a non-canonical rendering."""

    normalized = " ".join(text.casefold().replace("-", " ").split())
    if normalized.endswith(" kids"):
        return "kids_collision"
    if "high five" in normalized:
        return "high_five_collision"
    if normalized.startswith("hiffy"):
        return "hiffy_collision"
    if normalized.startswith("hippy"):
        return "hippy_collision"
    return "vowel_collision"


def _positive_semantics(text: str) -> tuple[int, str]:
    if text in CANONICAL_RENDER_TEXTS:
        return 1, "canonical_exact"
    return 0, collision_label(text)


def _require_file(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"missing corpus artifact: {resolved}")
    return resolved


def _base_common(
    *,
    path: Path,
    label: int,
    semantic_label: str,
    source_group: str,
    split: str,
    speaker_id: str,
    session_id: str,
    source_id: str,
    provenance_id: str,
    parent_id: str,
    ancestry_id: str,
    duration_seconds: float,
    audio_sha256: str,
) -> dict[str, Any]:
    if duration_seconds <= 0:
        raise ValueError(f"non-positive duration for {path}")
    return {
        "path": str(_require_file(path)),
        "label": int(label),
        "semantic_label": semantic_label,
        "source_group": source_group,
        "split": split,
        "speaker_id": speaker_id,
        "session_id": session_id,
        "source_id": source_id,
        "provenance_id": provenance_id,
        "parent_id": parent_id,
        "ancestry_id": ancestry_id,
        "duration_seconds": float(duration_seconds),
        "audio_sha256": audio_sha256,
    }


def synthesis_examples(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict]]:
    payload = json.loads(path.read_text())
    rows: list[dict[str, Any]] = []
    by_path: dict[str, dict] = {}
    for item in payload.get("examples", []):
        text = str(item["text"])
        label, semantic_label = _positive_semantics(text)
        provider = str(item["provider"])
        voice = str(item.get("voice", "unknown"))
        split = str(item["split"])
        output_hash = str(item["output_hash"])
        source_hash = str(item.get("source_hash") or output_hash)
        ancestry_id = f"tts-ancestry:{_stable_id(provider, voice, text)}"
        row = _base_common(
            path=Path(item["path"]),
            label=label,
            semantic_label=semantic_label,
            source_group=(
                {
                    "assemblyai": "assemblyai_synthetic",
                    "deepgram": "deepgram_synthetic",
                    "elevenlabs": "elevenlabs_synthetic",
                    "kokoro": "kokoro_synthetic",
                    "household-device-anchor": "device_tts_anchor",
                }.get(provider, f"synthetic_{provider}")
                if label
                else "phonetic_collision"
            ),
            split=split,
            speaker_id=f"tts:{provider}:{voice}",
            session_id=f"tts:{provider}:{voice}:{split}",
            source_id=f"synthesis:{output_hash}",
            provenance_id=f"audio-sha256:{output_hash}",
            parent_id=f"synthesis-source:{source_hash}",
            ancestry_id=ancestry_id,
            duration_seconds=float(item["duration"]),
            audio_sha256=output_hash,
        )
        row.update(
            {
                "render_text": text,
                "provider": provider,
                "voice": voice,
                "target_id": CANONICAL_TARGET_ID if label else None,
                "target_phones": list(CANONICAL_PHONES) if label else [],
                "training_eligible": True,
            }
        )
        rows.append(row)
        by_path[str(Path(item["path"]).resolve())] = {
            "source": item,
            "row": row,
        }
    return rows, by_path


def overlay_examples(
    path: Path, positive_by_path: dict[str, dict]
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    rows = []
    for item in payload.get("examples", []):
        derived_path = str(Path(item["derived_from_positive"]).resolve())
        base_record = positive_by_path.get(derived_path)
        if base_record is None:
            raise ValueError(
                f"overlay parent is absent from synthesis manifest: {derived_path}"
            )
        base = base_record["row"]
        output_hash = str(item["output_hash"])
        row = _base_common(
            path=Path(item["path"]),
            label=int(base["label"]),
            semantic_label=str(base["semantic_label"]),
            source_group="noisy_overlay" if base["label"] else "collision_overlay",
            split=str(item["split"]),
            speaker_id=str(base["speaker_id"]),
            session_id=f"overlay:{base['session_id']}",
            source_id=f"overlay:{output_hash}",
            provenance_id=f"audio-sha256:{output_hash}",
            parent_id=f"audio-sha256:{item['positive_hash']}",
            ancestry_id=str(base["ancestry_id"]),
            duration_seconds=float(item["duration_s"]),
            audio_sha256=output_hash,
        )
        row.update(
            {
                "render_text": base["render_text"],
                "provider": base["provider"],
                "voice": base["voice"],
                "target_id": base["target_id"],
                "target_phones": base["target_phones"],
                "background_source_id": f"audio-sha256:{item['background_hash']}",
                "background_category": item["background_category"],
                "snr_db": float(item["snr_db"]),
                "training_eligible": True,
            }
        )
        rows.append(row)
    return rows


def _device_is_canonical(capture: dict[str, Any]) -> bool:
    return (
        str(capture.get("phrase", "")) in CANONICAL_RENDER_TEXTS
        and str(capture.get("pronunciation", "")) in CANONICAL_DEVICE_PRONUNCIATIONS
    )


def device_examples(
    path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    payload = json.loads(path.read_text())
    training_rows = []
    locked_human_positives = []
    excluded: Counter[str] = Counter()
    for capture in payload.get("captures", []):
        if capture.get("truth") != "positive":
            continue
        source = str(capture.get("source", "unknown"))
        canonical = _device_is_canonical(capture)
        pronunciation = str(capture.get("pronunciation", ""))
        if source == "human":
            if pronunciation in REPEATED_DEVICE_PRONUNCIATIONS:
                excluded["human_repeated_opportunity_geometry_unreviewed"] += 1
                continue
            if not canonical:
                excluded["human_noncanonical"] += 1
                continue
        label = 1 if canonical else 0
        semantic_label = (
            "canonical_exact"
            if canonical
            else collision_label(str(capture.get("phrase", "")))
        )
        capture_hash = str(capture["sha256"])
        source_wav_hash = str(
            capture.get("conditions", {}).get("source_wav_sha256") or capture_hash
        )
        row = _base_common(
            path=path.parent / capture["path"],
            label=label,
            semantic_label=semantic_label,
            source_group=(
                "device_human_positive"
                if source == "human"
                else "device_replay" if canonical else "device_collision"
            ),
            split="test" if source == "human" else str(capture["split"]),
            speaker_id=f"device:{capture['speaker_id']}",
            session_id=f"device:{capture['session_id']}",
            source_id=f"device-capture:{capture['capture_id']}",
            provenance_id=f"audio-sha256:{capture_hash}",
            parent_id=f"source-audio-sha256:{source_wav_hash}",
            ancestry_id=f"device-ancestry:{source_wav_hash}",
            duration_seconds=float(capture["samples"]) / 16_000.0,
            audio_sha256=capture_hash,
        )
        row.update(
            {
                "render_text": capture.get("phrase"),
                "pronunciation": pronunciation,
                "target_id": CANONICAL_TARGET_ID if canonical else None,
                "target_phones": list(CANONICAL_PHONES) if canonical else [],
                "device_profile": capture.get("device_profile"),
                "firmware_sha": capture.get("firmware_sha"),
                "phrase_span": capture.get("phrase_span"),
            }
        )
        if source == "human":
            row.update(
                {
                    "role": "positive",
                    "locked_deployment_anchor": True,
                    "training_eligible": False,
                    "source_split": capture.get("split"),
                }
            )
            locked_human_positives.append(row)
        else:
            row["training_eligible"] = True
            training_rows.append(row)
    return training_rows, locked_human_positives, excluded


def public_negative_examples(
    path: Path,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows = []
    excluded: Counter[str] = Counter()
    with path.open(newline="") as source:
        for item in csv.DictReader(source):
            if item.get("eligible_for_target") != "True":
                continue
            original_split = str(item.get("split", ""))
            split = original_split
            if split not in {"train", "validation", "test"}:
                continue
            duration_seconds = float(item["duration_s"])
            if duration_seconds < MIN_TRAINING_AUDIO_SECONDS:
                excluded["shorter_than_200ms"] += 1
                continue
            category = str(item["category"])
            source_group = {
                "connected_speech": "public_speech",
                "domestic_far_field_speech": "domestic_speech",
                "music": "music",
                "noise": "background_noise",
                "silence": "silence",
            }.get(category)
            if source_group is None or category == "household_false_wake":
                continue
            audio_hash = str(item["sha256"])
            identity = str(item.get("speaker_or_session") or audio_hash)
            source_collection = str(item["source"])
            # MUSAN's speaker_or_session column is a broad subset label such as
            # ``librivox`` or ``fma``, not a person/session identity.  Randomly
            # distributing those files across splits would therefore claim
            # disjointness that the metadata cannot prove.  Keep MUSAN as
            # training augmentation; locked qualification uses independent
            # speaker/session-aware continuous streams.
            if source_collection == "MUSAN":
                split = "train"
            is_speech = category in {"connected_speech", "domestic_far_field_speech"}
            rows.append(
                {
                    **_base_common(
                        path=Path(item["path"]),
                        label=0,
                        semantic_label=source_group,
                        source_group=source_group,
                        split=split,
                        speaker_id=(
                            f"public-speaker:{source_collection}:{identity}"
                            if is_speech
                            else f"nonspeech-item:{audio_hash}"
                        ),
                        session_id=(
                            f"public-session:{source_collection}:{identity}"
                            if is_speech
                            else f"nonspeech-session:{audio_hash}"
                        ),
                        source_id=f"public-audio:{audio_hash}",
                        provenance_id=f"audio-sha256:{audio_hash}",
                        parent_id=f"public-source:{source_collection}:{identity}",
                        ancestry_id=f"public-ancestry:{audio_hash}",
                        duration_seconds=duration_seconds,
                        audio_sha256=audio_hash,
                    ),
                    "license": item.get("license"),
                    "provenance": item.get("provenance"),
                    "source_collection": source_collection,
                    "source_split": original_split,
                    "training_eligible": True,
                }
            )
    return rows, excluded


def false_wake_anchor_examples(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if payload.get("training_eligible") is not False:
        raise ValueError("false-wake anchor source must be quarantined")
    rows = []
    for item in payload.get("observations", []):
        audio_path = _require_file(path.parent / item["audio_path"])
        metadata_path = _require_file(path.parent / item["metadata_path"])
        metadata = json.loads(metadata_path.read_text())
        audio_hash = str(item["audio_sha256"])
        rows.append(
            {
                **_base_common(
                    path=audio_path,
                    label=0,
                    semantic_label="device_false_wake",
                    source_group="device_false_wake",
                    split="test",
                    speaker_id=f"device:{metadata.get('device_id', 'kizz-1')}",
                    session_id=f"false-wake-session:{metadata.get('firmware_sha', 'unknown')}",
                    source_id=f"false-wake:{item['observation_id']}",
                    provenance_id=f"audio-sha256:{audio_hash}",
                    parent_id=f"observation:{item['observation_id']}",
                    ancestry_id=f"false-wake-ancestry:{audio_hash}",
                    duration_seconds=float(metadata["samples"]) / 16_000.0,
                    audio_sha256=audio_hash,
                ),
                "role": "anchor",
                "locked_deployment_anchor": True,
                "training_eligible": False,
                "human_review_basis": item.get("human_review_basis"),
                "review": item.get("review"),
                "metadata_path": str(metadata_path),
            }
        )
    return rows


def _dedupe(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_hash: dict[str, dict[str, Any]] = {}
    for row in rows:
        audio_hash = str(row["audio_sha256"])
        previous = by_hash.get(audio_hash)
        if previous and int(previous["label"]) != int(row["label"]):
            raise ValueError(f"audio has conflicting labels: {audio_hash}")
        if previous is None:
            by_hash[audio_hash] = row
    return sorted(
        by_hash.values(),
        key=lambda item: (item["split"], int(item["label"]), item["source_id"]),
    )


def _summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[tuple[str, int, str]] = Counter()
    durations: defaultdict[tuple[str, int, str], float] = defaultdict(float)
    semantic: Counter[str] = Counter()
    for row in rows:
        key = (str(row["split"]), int(row["label"]), str(row["source_group"]))
        counts[key] += 1
        durations[key] += float(row["duration_seconds"])
        semantic[str(row["semantic_label"])] += 1
    return {
        "examples": sum(counts.values()),
        "by_split_label_source": [
            {
                "split": split,
                "label": label,
                "source_group": source,
                "count": count,
                "duration_seconds": round(durations[(split, label, source)], 6),
            }
            for (split, label, source), count in sorted(counts.items())
        ],
        "semantic_labels": dict(sorted(semantic.items())),
    }


def compose(
    *,
    synthesis_manifest: Path,
    overlay_manifest: Path,
    device_corpus: Path,
    public_negative_manifest: Path,
    false_wake_manifest: Path,
    output: Path,
) -> dict[str, Any]:
    synthesis, by_path = synthesis_examples(synthesis_manifest)
    overlays = overlay_examples(overlay_manifest, by_path)
    device, positive_anchors, device_excluded = device_examples(device_corpus)
    public_negatives, public_negative_excluded = public_negative_examples(
        public_negative_manifest
    )
    false_wake_anchors = false_wake_anchor_examples(false_wake_manifest)
    examples = _dedupe([*synthesis, *overlays, *device, *public_negatives])
    if any(row.get("training_eligible") is not True for row in examples):
        raise ValueError("training manifest contains an ineligible example")
    if any(
        int(row["label"]) == 1 and row["semantic_label"] != "canonical_exact"
        for row in examples
    ):
        raise ValueError("non-canonical example reached the positive class")

    output.mkdir(parents=True, exist_ok=True)
    source_manifests = {
        "synthesis": str(synthesis_manifest.resolve()),
        "overlays": str(overlay_manifest.resolve()),
        "device_corpus": str(device_corpus.resolve()),
        "public_negatives": str(public_negative_manifest.resolve()),
        "false_wake_anchors": str(false_wake_manifest.resolve()),
    }
    source_hashes = {
        key: sha256_file(Path(value)) for key, value in source_manifests.items()
    }
    manifest = {
        "schema_version": 2,
        "sample_rate": 16_000,
        "target": {
            "id": CANONICAL_TARGET_ID,
            "phones": list(CANONICAL_PHONES),
            "canonical_render_texts": sorted(CANONICAL_RENDER_TEXTS),
            "policy": "rendering variants outside the declared phone-equivalent set are negatives",
        },
        "examples": examples,
        "source_manifests": source_manifests,
        "source_manifest_sha256": source_hashes,
    }
    positive_anchor_manifest = {
        "schema_version": 2,
        "training_eligible": False,
        "anchor_type": "natural_household_positive",
        "examples": _dedupe(positive_anchors),
    }
    false_wake_anchor_manifest = {
        "schema_version": 2,
        "training_eligible": False,
        "anchor_type": "reviewed_household_false_wake",
        "examples": _dedupe(false_wake_anchors),
    }
    report = {
        "schema_version": 1,
        "manifest": "manifest.json",
        "summary": _summary(examples),
        "locked_positive_anchors": _summary(positive_anchors),
        "locked_false_wake_anchors": _summary(false_wake_anchors),
        "device_exclusions": dict(sorted(device_excluded.items())),
        "public_negative_exclusions": dict(sorted(public_negative_excluded.items())),
        "source_manifest_sha256": source_hashes,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (output / "locked-positive-anchors.json").write_text(
        json.dumps(positive_anchor_manifest, indent=2, sort_keys=True) + "\n"
    )
    (output / "locked-false-wake-anchors.json").write_text(
        json.dumps(false_wake_anchor_manifest, indent=2, sort_keys=True) + "\n"
    )
    (output / "composition-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthesis-manifest", type=Path, required=True)
    parser.add_argument("--overlay-manifest", type=Path, required=True)
    parser.add_argument("--device-corpus", type=Path, required=True)
    parser.add_argument("--public-negative-manifest", type=Path, required=True)
    parser.add_argument("--false-wake-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = compose(
        synthesis_manifest=args.synthesis_manifest,
        overlay_manifest=args.overlay_manifest,
        device_corpus=args.device_corpus,
        public_negative_manifest=args.public_negative_manifest,
        false_wake_manifest=args.false_wake_manifest,
        output=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
