#!/usr/bin/env python3
"""Freeze disjoint C1 training negatives and a 100-hour MUSAN lock."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import soundfile as sf


FAMILIES = ("speech", "music", "noise")
SOURCE_GROUPS = {
    "speech": "public_speech",
    "music": "music",
    "noise": "background_noise",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _music_groups(root: Path) -> dict[str, str]:
    groups: dict[str, str] = {}
    for annotations in sorted((root / "music").glob("*/ANNOTATIONS")):
        collection = annotations.parent.name
        for raw in annotations.read_text(encoding="utf-8").splitlines():
            parts = raw.split()
            if len(parts) < 2 or not parts[0].startswith("music-"):
                continue
            groups[parts[0]] = f"musan:music:{collection}:{parts[3] if len(parts) >= 4 else parts[0]}"
    return groups


def inventory_musan(root: Path) -> list[dict[str, Any]]:
    root = root.resolve()
    music_groups = _music_groups(root)
    rows = []
    for family in FAMILIES:
        for path in sorted((root / family).rglob("*.wav")):
            resolved = path.resolve()
            if root not in resolved.parents or resolved.is_symlink():
                raise ValueError(f"unsafe MUSAN path: {path}")
            info = sf.info(resolved)
            if info.samplerate <= 0 or info.frames <= 0 or info.channels < 1:
                raise ValueError(f"invalid MUSAN audio: {path}")
            relative = resolved.relative_to(root).as_posix()
            stem = resolved.stem
            source_collection = resolved.parent.name
            if family == "music":
                group_id = music_groups.get(
                    stem, f"musan:music:{source_collection}:{stem}"
                )
            else:
                group_id = f"musan:{family}:{source_collection}:{stem}"
            audio_hash = sha256_file(resolved)
            rows.append(
                {
                    "path": str(resolved),
                    "audio_sha256": audio_hash,
                    "sha256": audio_hash,
                    "duration_seconds": info.frames / info.samplerate,
                    "sample_rate_hz": info.samplerate,
                    "channels": info.channels,
                    "family": family,
                    "source_group": SOURCE_GROUPS[family],
                    "source_collection": source_collection,
                    "source_file_id": f"musan:{relative}",
                    "source_group_id": group_id,
                    "source_id": f"musan:{relative}",
                    "provenance_id": f"audio-sha256:{audio_hash}",
                    "parent_id": group_id,
                    "ancestry_id": group_id,
                    "speaker_id": group_id,
                    "session_id": group_id,
                    "semantic_label": f"{family}_negative",
                    "label": 0,
                }
            )
    if not rows or {row["family"] for row in rows} != set(FAMILIES):
        raise ValueError("MUSAN inventory lacks speech, music, or noise")
    hashes = [row["audio_sha256"] for row in rows]
    if len(hashes) != len(set(hashes)):
        raise ValueError("MUSAN inventory contains duplicate audio hashes")
    return rows


def partition_rows(
    rows: Sequence[dict[str, Any]],
    *,
    minimum_holdout_hours: float = 100.0,
    train_target_hours_per_family: float = 3.0,
    seed: int = 231,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if minimum_holdout_hours < 100.0:
        raise ValueError("continuous holdout may not be below 100 hours")
    if train_target_hours_per_family <= 0:
        raise ValueError("training target hours must be positive")
    total_seconds = sum(float(row["duration_seconds"]) for row in rows)
    if total_seconds < minimum_holdout_hours * 3600:
        raise ValueError("MUSAN cannot satisfy the continuous holdout")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["family"]), str(row["source_group_id"]))].append(dict(row))

    selected_groups: set[tuple[str, str]] = set()
    for family in FAMILIES:
        target_seconds = train_target_hours_per_family * 3600
        family_groups = []
        for key, values in groups.items():
            if key[0] != family:
                continue
            duration = sum(float(row["duration_seconds"]) for row in values)
            rank = hashlib.sha256(f"{seed}\0{family}\0{key[1]}".encode()).digest()
            family_groups.append((rank, duration, key))
        selected_seconds = 0.0
        for _, duration, key in sorted(family_groups):
            if selected_seconds + duration <= target_seconds:
                selected_groups.add(key)
                selected_seconds += duration
        if selected_seconds < min(1800.0, target_seconds / 2):
            remaining = sorted(
                (item for item in family_groups if item[2] not in selected_groups),
                key=lambda item: (item[1], item[0]),
            )
            if not remaining:
                raise ValueError(f"MUSAN {family} has no trainable group")
            _, duration, key = remaining[0]
            selected_groups.add(key)
            selected_seconds += duration

    train = []
    holdout = []
    for row in rows:
        item = dict(row)
        key = (str(item["family"]), str(item["source_group_id"]))
        if key in selected_groups:
            item.update(
                {
                    "split": "train",
                    "training_eligible": True,
                    "locked_deployment_anchor": False,
                }
            )
            train.append(item)
        else:
            item.update(
                {
                    "split": "test",
                    "training_eligible": False,
                    "locked_deployment_anchor": True,
                }
            )
            holdout.append(item)

    train_groups = {row["source_group_id"] for row in train}
    holdout_groups = {row["source_group_id"] for row in holdout}
    if train_groups & holdout_groups:
        raise ValueError("MUSAN source group crosses train and continuous holdout")
    train_hashes = {row["audio_sha256"] for row in train}
    holdout_hashes = {row["audio_sha256"] for row in holdout}
    if train_hashes & holdout_hashes:
        raise ValueError("MUSAN audio crosses train and continuous holdout")
    holdout_seconds = sum(float(row["duration_seconds"]) for row in holdout)
    if holdout_seconds < minimum_holdout_hours * 3600:
        raise ValueError("training partition consumed the 100-hour holdout margin")
    for name, values in (("train", train), ("holdout", holdout)):
        if {row["family"] for row in values} != set(FAMILIES):
            raise ValueError(f"MUSAN {name} partition lacks a required family")
    return sorted(train, key=lambda row: row["source_id"]), sorted(
        holdout, key=lambda row: row["source_id"]
    )


def _load_background_examples(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    examples = payload.get("examples")
    if not isinstance(examples, list):
        raise ValueError("background manifest must contain canonical examples")
    result = []
    for row in examples:
        item = dict(row)
        audio = Path(str(item.get("path", ""))).resolve()
        expected = str(item.get("audio_sha256", item.get("sha256", "")))
        if not audio.is_file() or not expected or sha256_file(audio) != expected:
            raise ValueError(f"background audio hash drift: {audio}")
        item["path"] = str(audio)
        item["audio_sha256"] = expected
        result.append(item)
    return result


def freeze(
    musan_root: Path,
    musan_archive: Path,
    assets_output: Path,
    continuous_output: Path,
    report_output: Path,
    *,
    background_manifest: Path | None = None,
    minimum_holdout_hours: float = 100.0,
    train_target_hours_per_family: float = 3.0,
    seed: int = 231,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    inventory = inventory_musan(musan_root)
    train, holdout = partition_rows(
        inventory,
        minimum_holdout_hours=minimum_holdout_hours,
        train_target_hours_per_family=train_target_hours_per_family,
        seed=seed,
    )
    backgrounds = _load_background_examples(background_manifest)
    locked_hashes = {row["audio_sha256"] for row in holdout}
    asset_rows = train + backgrounds
    asset_hashes = [str(row["audio_sha256"]) for row in asset_rows]
    if len(asset_hashes) != len(set(asset_hashes)) or locked_hashes & set(asset_hashes):
        raise ValueError("negative assets duplicate or overlap the continuous lock")
    inputs = {
        "musan_root": str(musan_root.resolve()),
        "musan_archive": str(musan_archive.resolve()),
        "musan_archive_sha256": sha256_file(musan_archive),
        "background_manifest": str(background_manifest.resolve())
        if background_manifest
        else None,
        "background_manifest_sha256": sha256_file(background_manifest)
        if background_manifest
        else None,
    }
    assets = {
        "schema_version": 1,
        "kind": "kizz_control_c1_frozen_negative_assets",
        "frozen": True,
        "seed": seed,
        "inputs": inputs,
        "counts": {
            "examples": len(asset_rows),
            "musan_train_hours": sum(row["duration_seconds"] for row in train) / 3600,
            "by_split": dict(Counter(str(row.get("split")) for row in asset_rows)),
            "by_source_group": dict(
                Counter(str(row.get("source_group")) for row in asset_rows)
            ),
        },
        "examples": sorted(
            asset_rows,
            key=lambda row: (str(row.get("split")), str(row.get("source_id"))),
        ),
    }
    continuous_examples = [
        {
            "path": row["path"],
            "sha256": row["audio_sha256"],
            "audio_sha256": row["audio_sha256"],
            "duration_seconds": row["duration_seconds"],
            "category": row["family"],
            "source": "MUSAN",
            "split": "test",
            "source_id": row["source_id"],
            "source_group_id": row["source_group_id"],
            "license": "MUSAN per-subcollection licenses",
        }
        for row in holdout
    ]
    exposure_seconds = sum(row["duration_seconds"] for row in continuous_examples)
    continuous = {
        "schema_version": 2,
        "gate_scope": "locked_untouched_continuous_negative_corpus",
        "locked_before_scoring": True,
        "training_eligible": False,
        "inputs": inputs,
        "selection_policy": {
            "seed": seed,
            "minimum_hours": minimum_holdout_hours,
            "group_disjoint": True,
            "train_target_hours_per_family": train_target_hours_per_family,
        },
        "counts": {
            "files": len(continuous_examples),
            "exposure_seconds": exposure_seconds,
            "exposure_hours": exposure_seconds / 3600,
            "categories": dict(Counter(row["category"] for row in continuous_examples)),
        },
        "examples": continuous_examples,
    }
    report = {
        "schema_version": 1,
        "kind": "kizz_control_c1_negative_asset_freeze",
        "qualified": continuous["counts"]["exposure_hours"] >= minimum_holdout_hours,
        "inputs": inputs,
        "partition": {
            "inventory_files": len(inventory),
            "inventory_hours": sum(row["duration_seconds"] for row in inventory) / 3600,
            "train_files": len(train),
            "train_hours": sum(row["duration_seconds"] for row in train) / 3600,
            "continuous_files": len(holdout),
            "continuous_hours": continuous["counts"]["exposure_hours"],
            "train_categories": dict(Counter(row["family"] for row in train)),
            "continuous_categories": dict(Counter(row["family"] for row in holdout)),
            "hash_overlap": 0,
            "source_group_overlap": 0,
        },
    }
    for path, payload in (
        (assets_output, assets),
        (continuous_output, continuous),
        (report_output, report),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return assets, continuous, report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--musan-root", type=Path, required=True)
    parser.add_argument("--musan-archive", type=Path, required=True)
    parser.add_argument("--background-manifest", type=Path)
    parser.add_argument("--assets-output", type=Path, required=True)
    parser.add_argument("--continuous-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--minimum-holdout-hours", type=float, default=100.0)
    parser.add_argument("--train-target-hours-per-family", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=231)
    args = parser.parse_args(argv)
    _, _, report = freeze(
        args.musan_root,
        args.musan_archive,
        args.assets_output,
        args.continuous_output,
        args.report_output,
        background_manifest=args.background_manifest,
        minimum_holdout_hours=args.minimum_holdout_hours,
        train_target_hours_per_family=args.train_target_hours_per_family,
        seed=args.seed,
    )
    print(json.dumps(report["partition"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
