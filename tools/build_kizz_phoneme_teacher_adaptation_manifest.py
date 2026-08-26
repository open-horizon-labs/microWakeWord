#!/usr/bin/env python3
"""Build an immutable, leakage-checked adaptation manifest for the Kizz teacher.

This sidecar references existing audio only.  It never copies, rewrites, or
promotes audio into a corpus.  The adaptation partition is deliberately small
and is constructed from the distillation corpus' train rows plus separately
captured target-device positives.  Train and validation device captures are
qualified independently; qualification voices and audio parents are excluded
before any row is emitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from microwakeword.kizz_phoneme_teacher import MODEL_ID, MODEL_REVISION
from microwakeword.wake_phrase import KIZZ_CONTROL


APPROVED_PROVIDERS = ("assemblyai", "deepgram", "elevenlabs", "kokoro")
REQUIRED_NEGATIVE_GROUPS = ("public_speech", "phonetic_collision", "device_collision")
PHONETIC_GROUPS = {"phonetic_collision", "kizz_control_phonetic_collision"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload(path: Path) -> object:
    return json.loads(path.read_text())


def rows_from(path: Path, *, required: bool = True) -> list[dict]:
    payload = _payload(path)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = next(
            (payload[key] for key in ("examples", "records", "captures", "items", "anchors")
             if isinstance(payload.get(key), list)),
            [],
        )
    else:
        rows = []
    if required and not rows:
        raise ValueError(f"manifest contains no rows: {path}")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"manifest rows must be objects: {path}")
    return [dict(row) for row in rows]


def _audio_hash(row: dict) -> str | None:
    for key in ("audio_sha256", "sha256", "source_audio_sha256"):
        value = row.get(key)
        if value:
            return str(value).lower()
    return None


def _path(row: dict) -> str | None:
    value = row.get("path")
    return str(Path(value).resolve()) if value else None


def _partition_id(row: dict) -> str | None:
    for key in ("partition_id", "partition_identity", "source_id", "provenance_id", "parent_source_id"):
        value = row.get(key)
        if value:
            return str(value)
    return None


def _source_audio_hash(row: dict) -> str | None:
    value = row.get("source_audio_sha256")
    if value is None:
        value = row.get("conditions", {}).get("source_audio_sha256")
    return str(value).lower() if value else None


def _provider_voice(row: dict) -> tuple[str, str] | None:
    conditions = row.get("conditions", {})
    provider = row.get("provider") or conditions.get("source_provider")
    voice = row.get("voice") or conditions.get("source_voice")
    if provider is None or voice is None:
        return None
    return str(provider).lower(), str(voice).lower()


def _identities(row: dict) -> set[tuple[str, str]]:
    identities: set[tuple[str, str]] = set()
    for value in (_audio_hash(row), _path(row), _partition_id(row)):
        if value:
            identities.add(("hash" if value == _audio_hash(row) else "value", value))
    # Keep source IDs distinct from paths/hashes even when a fixture uses the
    # same-looking value in two fields.
    for key in ("source_id", "provenance_id", "parent_source_id", "partition_id", "partition_identity"):
        if row.get(key):
            identities.add((key, str(row[key])))
    if value := _source_audio_hash(row):
        identities.add(("source_audio_sha256", value))
    return identities


def _provenance(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _normalize_provider(row: dict) -> str | None:
    provider = row.get("provider")
    if provider is None:
        provider = row.get("conditions", {}).get("source_provider")
    return str(provider).lower() if provider is not None else None


def _adaptation_row(row: dict, *, source_group: str, label: int, source_split: str) -> dict:
    output = dict(row)
    output["label"] = label
    output["split"] = source_split if source_group == "device_channel_positive" else "train"
    output["source_split"] = source_split
    output["source_group"] = source_group
    output["training_eligible"] = True
    return output


def _device_rows(corpus: Path, *, expected_split: str, evidence_role: str) -> list[dict]:
    payload = _payload(corpus)
    if not isinstance(payload, dict) or not isinstance(payload.get("captures"), list):
        raise ValueError("device training corpus must contain captures")
    rows = []
    for capture in payload["captures"]:
        conditions = dict(capture.get("conditions", {}))
        if (capture.get("truth") != "positive"
                or capture.get("split") != expected_split
                or conditions.get("evidence_role") != evidence_role):
            continue
        provider = str(conditions.get("source_provider", "")).lower()
        voice = str(conditions.get("source_voice", "")).lower()
        if provider not in APPROVED_PROVIDERS or not voice:
            raise ValueError("device training capture lacks approved provider/voice")
        relative = Path(str(capture["path"]))
        audio_path = relative if relative.is_absolute() else corpus.parent / relative
        if (
            not audio_path.is_file()
            or not capture.get("sha256")
            or sha256_file(audio_path) != str(capture["sha256"]).lower()
        ):
            raise ValueError(f"device adaptation capture hash drift: {audio_path}")
        rows.append(
            {
                "source_id": f"device-adaptation:{capture['capture_id']}",
                "provenance_id": f"device-adaptation:{capture['capture_id']}",
                "parent_source_id": (
                    "source-audio-sha256:"
                    + str(conditions.get("source_audio_sha256", ""))
                ),
                "path": str(audio_path.resolve()),
                "audio_sha256": str(capture.get("sha256", "")).lower(),
                "source_audio_sha256": str(
                    conditions.get("source_audio_sha256", "")
                ).lower(),
                "label": 1,
                "split": expected_split,
                "source_group": "device_channel_positive",
                "provider": provider,
                "voice": voice,
                "speaker_id": capture.get("speaker_id"),
                "session_id": capture.get("session_id"),
                "device_id": capture.get("device_id"),
                "device_profile": capture.get("device_profile"),
                "conditions": conditions,
            }
        )
    if not rows:
        raise ValueError(f"device {expected_split} corpus has no eligible adaptation captures")
    return rows


def _device_training_rows(corpus: Path) -> list[dict]:
    return _device_rows(
        corpus,
        expected_split="train",
        evidence_role="teacher_adaptation_target_channel_positive",
    )


def _device_validation_rows(corpus: Path) -> list[dict]:
    return _device_rows(
        corpus,
        expected_split="validation",
        evidence_role="teacher_adaptation_target_channel_validation_positive",
    )


def _device_identity_keys(row: dict) -> set[tuple[str, str]]:
    keys = {
        ("audio", value) for value in (_audio_hash(row), _path(row)) if value
    }
    if value := _source_audio_hash(row):
        keys.add(("source_parent", value))
    if value := _provider_voice(row):
        keys.add(("provider_voice", "\0".join(value)))
    for field in ("speaker_id", "session_id"):
        if row.get(field):
            keys.add((field, str(row[field])))
    return keys


def _report_counts(quality: dict, *, corpus: Path, evidence: Path, rows: list[dict], role: str) -> None:
    if (
        not isinstance(quality, dict)
        or quality.get("kind") != "kizz_control_teacher_adaptation_device_replay_quality"
        or quality.get("qualified") is not True
        or quality.get("inputs", {}).get("corpus_sha256") != sha256_file(corpus)
        or quality.get("inputs", {}).get("qualification_evidence_sha256") != sha256_file(evidence)
    ):
        raise ValueError(f"device {role} quality report is absent, stale, or unqualified")
    counts = quality.get("counts", {})
    actual_providers = Counter(_normalize_provider(row) for row in rows)
    actual_voices = {
        provider: sorted(
            str(row.get("voice", "")).lower()
            for row in rows
            if _normalize_provider(row) == provider
        )
        for provider in APPROVED_PROVIDERS
    }
    reported_providers = counts.get("providers")
    reported_voices = counts.get("voices")
    if not isinstance(reported_providers, dict) or not isinstance(reported_voices, dict):
        raise ValueError(f"device {role} quality report lacks provider/voice counts")
    if dict(sorted(actual_providers.items())) != {
        str(provider).lower(): int(count) for provider, count in sorted(reported_providers.items())
    }:
        raise ValueError(f"device {role} captures do not exactly realize quality provider counts")
    normalized_reported_voices = {
        str(provider).lower(): sorted(str(voice).lower() for voice in voices)
        for provider, voices in reported_voices.items()
    }
    if actual_voices != {
        provider: normalized_reported_voices.get(provider, [])
        for provider in APPROVED_PROVIDERS
    }:
        raise ValueError(f"device {role} captures do not exactly realize quality provider voices")
    if any(actual_providers.get(provider, 0) == 0 for provider in APPROVED_PROVIDERS):
        raise ValueError(f"device {role} captures must include every approved provider")


def _stable_validation_speakers(rows: list[dict], *, fraction: float = 0.15) -> set[str]:
    """Select speaker-disjoint adaptation-dev identities from train-only data."""
    grouped: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        if row["source_group"] == "device_channel_positive":
            continue
        family = (
            f"positive:{row['provider']}"
            if int(row["label"]) == 1
            else f"negative:{row['source_group']}"
        )
        speaker = str(row.get("speaker_id") or row.get("source_id"))
        grouped.setdefault((family, speaker), set()).add(str(row.get("source_id")))
    selected: set[str] = set()
    families = sorted({family for family, _ in grouped})
    for family in families:
        speakers = sorted(
            (speaker for candidate, speaker in grouped if candidate == family),
            key=lambda speaker: hashlib.sha256(
                f"kizz-control-adaptation-dev-v1\0{family}\0{speaker}".encode()
            ).hexdigest(),
        )
        if len(speakers) < 2:
            raise ValueError(f"adaptation family cannot be speaker-disjoint: {family}")
        count = max(1, round(len(speakers) * fraction))
        count = min(count, len(speakers) - 1)
        selected.update(speakers[:count])
    return selected


def _assign_adaptation_splits(rows: list[dict]) -> None:
    validation_speakers = _stable_validation_speakers(rows)
    for row in rows:
        speaker = str(row.get("speaker_id") or row.get("source_id"))
        if row["source_group"] != "device_channel_positive":
            row["split"] = "validation" if speaker in validation_speakers else "train"
        row["training_eligible"] = row["split"] == "train"
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    for provider in APPROVED_PROVIDERS:
        for split_rows, split in ((train, "train"), (validation, "validation")):
            if not any(
                int(row["label"]) == 1
                and row["source_group"] != "device_channel_positive"
                and _normalize_provider(row) == provider
                for row in split_rows
            ):
                raise ValueError(
                    f"speaker-disjoint adaptation {split} lacks provider {provider}"
                )
    for group in ("public_speech", "kizz_control_phonetic_collision", "device_collision"):
        for split_rows, split in ((train, "train"), (validation, "validation")):
            if not any(row["source_group"] == group for row in split_rows):
                raise ValueError(
                    f"speaker-disjoint adaptation {split} lacks negative group {group}"
                )


def build_manifest(
    distillation_corpus: Path,
    distillation_teacher_manifest: Path,
    device_training_corpus: Path,
    device_training_quality: Path,
    device_validation_corpus: Path,
    device_validation_quality: Path,
    current_device_evidence: Path,
    current_device_qualification: Path,
    continuous_negative_lock: Path,
    false_wake_anchors: Path,
) -> dict:
    corpus_rows = rows_from(distillation_corpus)
    teacher_rows = rows_from(distillation_teacher_manifest)
    if {json.dumps(row, sort_keys=True) for row in corpus_rows} != {json.dumps(row, sort_keys=True) for row in teacher_rows}:
        raise ValueError("corpus.json and teacher-manifest.json rows differ")

    exclusions: dict[tuple[str, str], str] = {}

    def add_exclusions(rows: Iterable[dict], name: str) -> None:
        for row in rows:
            for identity in _identities(row):
                previous = exclusions.setdefault(identity, name)
                if previous != name:
                    # Cross-evidence duplication is harmless for exclusion but
                    # must remain visible in the provenance contract.
                    exclusions[identity] = f"{previous}+{name}"

    add_exclusions(rows_from(current_device_evidence), "current_device_evidence")
    qualification_payload = _payload(current_device_qualification)
    qualification_rows = (
        qualification_payload.get("results", {}).get("natural_positive", [])
        if isinstance(qualification_payload, dict)
        else []
    )
    add_exclusions(qualification_rows, "current_device_qualification")
    add_exclusions(rows_from(continuous_negative_lock), "continuous_negative_lock")
    add_exclusions(rows_from(false_wake_anchors), "false_wake_anchors")

    _report_counts(
        _payload(device_training_quality),
        corpus=device_training_corpus,
        evidence=current_device_evidence,
        rows=(device_train := _device_training_rows(device_training_corpus)),
        role="training",
    )
    _report_counts(
        _payload(device_validation_quality),
        corpus=device_validation_corpus,
        evidence=current_device_evidence,
        rows=(device_validation := _device_validation_rows(device_validation_corpus)),
        role="validation",
    )

    candidates: list[dict] = []
    materialized_device_rows_excluded = 0
    for row in corpus_rows:
        if row.get("split") != "train":
            continue
        if int(row.get("label", -1)) == 1:
            if row.get("source_group") == "device_channel_positive":
                # Device positives have a separately qualified source contract
                # below.  A materialized corpus copy is the same evidence, not
                # an additional positive observation.
                materialized_device_rows_excluded += 1
                continue
            provider = _normalize_provider(row)
            if provider not in APPROVED_PROVIDERS:
                raise ValueError(f"unapproved positive provider: {provider!r}")
            candidate = _adaptation_row(row, source_group=str(row.get("source_group", "positive")), label=1, source_split="train")
        elif int(row.get("label", -1)) == 0:
            group = str(row.get("source_group", ""))
            if group not in {"public_speech", "device_collision", *PHONETIC_GROUPS}:
                continue
            candidate = _adaptation_row(row, source_group=group, label=0, source_split="train")
        else:
            continue
        candidates.append(candidate)

    heldout_voices = {
        value
        for row in rows_from(current_device_evidence)
        if (value := _provider_voice(row)) is not None
    }
    device_candidates = device_train
    device_provider_counts = Counter(_normalize_provider(row) for row in device_candidates)
    device_parent_hashes = {value for row in device_candidates if (value := _source_audio_hash(row))}
    # A device replay replaces its clean rendering as the adaptation view of
    # that parent. Keeping both would let one source recording count twice and
    # would make parent-disjoint validation claims false.
    candidates = [
        row for row in candidates if _source_audio_hash(row) not in device_parent_hashes
    ]
    for row in device_candidates:
        provider_voice = _provider_voice(row)
        if provider_voice in heldout_voices:
            raise ValueError(
                f"device adaptation voice overlaps current qualification: {provider_voice}"
            )
        candidates.append(
            _adaptation_row(
                row,
                source_group="device_channel_positive",
                label=1,
                source_split="train",
            )
        )

    device_all = device_candidates + device_validation
    locked_device_keys = {
        key
        for evidence_rows in (
            rows_from(current_device_evidence),
            qualification_rows,
        )
        for evidence_row in evidence_rows
        for key in _device_identity_keys(evidence_row)
    }
    seen_device_keys: dict[tuple[str, str], str] = {}
    for row in device_all:
        for key in _device_identity_keys(row):
            if key in locked_device_keys:
                raise ValueError(f"device capture overlaps locked target-device evidence: {key[1]}")
            if key in seen_device_keys:
                raise ValueError(f"device train/validation overlap: {key[1]}")
            seen_device_keys[key] = str(row.get("source_id", ""))
    validation_provider_counts = Counter(_normalize_provider(row) for row in device_validation)
    if any(device_provider_counts.get(provider, 0) == 0 for provider in APPROVED_PROVIDERS):
        raise ValueError("device training data must include every approved provider")
    if any(validation_provider_counts.get(provider, 0) == 0 for provider in APPROVED_PROVIDERS):
        raise ValueError("device validation data must include every approved provider")
    for row in device_validation:
        candidates.append(
            _adaptation_row(
                row,
                source_group="device_channel_positive",
                label=1,
                source_split="validation",
            )
        )

    seen: dict[tuple[str, str], str] = {}
    for row in candidates:
        identities = _identities(row)
        if not identities:
            raise ValueError(f"training row has no hash/path/partition identity: {row.get('source_id')}")
        for identity in identities:
            if identity in exclusions:
                raise ValueError(f"training row overlaps excluded {exclusions[identity]}: {identity[1]}")
            if identity in seen:
                raise ValueError(f"duplicate training partition identity: {identity[1]}")
            seen[identity] = str(row.get("source_id", row.get("path", "")))

    _assign_adaptation_splits(candidates)
    candidates.sort(key=lambda row: (str(row.get("split", "")), str(row.get("source_id", "")), _audio_hash(row) or "", _path(row) or ""))
    providers = Counter(_normalize_provider(row) for row in candidates if row["label"] == 1 and _normalize_provider(row))
    missing = [provider for provider in APPROVED_PROVIDERS if not providers.get(provider)]
    if missing:
        raise ValueError(f"positive provider coverage is incomplete: {missing}")
    clean_providers = Counter(
        _normalize_provider(row)
        for row in candidates
        if row["label"] == 1
        and row["source_group"] != "device_channel_positive"
        and _normalize_provider(row)
    )
    missing = [provider for provider in APPROVED_PROVIDERS if not clean_providers.get(provider)]
    if missing:
        raise ValueError(f"clean positive provider coverage is incomplete: {missing}")
    negative_groups = Counter("phonetic_collision" if row["source_group"] in PHONETIC_GROUPS else row["source_group"] for row in candidates if row["label"] == 0)
    missing = [group for group in REQUIRED_NEGATIVE_GROUPS if not negative_groups.get(group)]
    if missing:
        raise ValueError(f"negative group coverage is incomplete: {missing}")

    return {
        "schema_version": 1,
        "kind": "kizz_phoneme_teacher_adaptation_manifest",
        "immutable": True,
        "base_teacher": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "adaptation_role": "pinned_ipa_ctc_teacher",
        },
        "wake_phrase": {
            "phrase_id": KIZZ_CONTROL.phrase_id,
            "text": KIZZ_CONTROL.text,
            "phones": list(KIZZ_CONTROL.phones),
            "collision_paths": {
                name: list(phones)
                for name, phones in zip(
                    KIZZ_CONTROL.collision_transcripts,
                    KIZZ_CONTROL.collision_phones,
                    strict=True,
                )
            },
        },
        "contract": {
            "training_source_split": "train",
            "approved_positive_providers": list(APPROVED_PROVIDERS),
            "required_negative_groups": list(REQUIRED_NEGATIVE_GROUPS),
            "excluded_roles": ["current_device_v3", "false_wake_anchor", "qualification_validation_positive", "qualification_test_positive", "frozen_continuous_negative"],
            "audio_policy": "reference_only_no_audio_copy",
            "adaptation_validation_policy": "speaker_disjoint_train_inventory_plus_physical_device_validation_v1",
            "device_positive_policy": "qualified_capture_source_only_no_materialized_corpus_duplicates",
            "materialized_device_rows_excluded": materialized_device_rows_excluded,
        },
        "counts": {
            "total": len(candidates),
            "positive": sum(row["label"] == 1 for row in candidates),
            "negative": sum(row["label"] == 0 for row in candidates),
            "providers": dict(sorted(providers.items())),
            "clean_providers": dict(sorted(clean_providers.items())),
            "source_groups": dict(sorted(Counter(row["source_group"] for row in candidates).items())),
            "negative_groups": dict(sorted(negative_groups.items())),
            "splits": dict(sorted(Counter(row["split"] for row in candidates).items())),
            "device_train": {
                "total": len(device_train),
                "providers": dict(sorted(Counter(_normalize_provider(row) for row in device_train).items())),
            },
            "device_validation": {
                "total": len(device_validation),
                "providers": dict(sorted(Counter(_normalize_provider(row) for row in device_validation).items())),
            },
        },
        "input_provenance": {
            name: _provenance(path) for name, path in (
                ("distillation_corpus", distillation_corpus),
                ("distillation_teacher_manifest", distillation_teacher_manifest),
                ("device_training_corpus", device_training_corpus),
                ("device_training_quality", device_training_quality),
                ("device_validation_corpus", device_validation_corpus),
                ("device_validation_quality", device_validation_quality),
                ("current_device_evidence", current_device_evidence),
                ("current_device_qualification", current_device_qualification),
                ("continuous_negative_lock", continuous_negative_lock),
                ("false_wake_anchors", false_wake_anchors),
            )
        },
        "examples": candidates,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("distillation-corpus", "distillation-teacher-manifest", "device-training-corpus", "device-training-quality", "device-validation-corpus", "device-validation-quality", "current-device-evidence", "current-device-qualification", "continuous-negative-lock", "false-wake-anchors"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_manifest(
        args.distillation_corpus, args.distillation_teacher_manifest,
        args.device_training_corpus,
        args.device_training_quality,
        args.device_validation_corpus,
        args.device_validation_quality,
        args.current_device_evidence, args.current_device_qualification,
        args.continuous_negative_lock, args.false_wake_anchors,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "examples": result["counts"]["total"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
