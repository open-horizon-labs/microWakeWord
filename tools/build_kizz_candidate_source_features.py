#!/usr/bin/env python3
"""Materialize fixed detector inputs for Kizz candidate-verifier training.

Positive training rows receive deterministic random placement, room response,
gain, and background overlays. Validation/test positives remain clean. Public
negative audio is deterministically windowed without reading locked anchors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf

from tools.build_kizz_aligned_teacher_features_v3 import (
    CONTEXT_SAMPLES,
    SAMPLE_RATE,
    apply_gain_db,
    apply_room_impulse_response,
    frontend,
    load_audio,
    mix_at_snr,
    sha256_file,
)


SPLITS = ("train", "validation", "test")
BACKGROUND_GROUPS = frozenset(("public_speech", "music", "background_noise"))


def _load(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("examples") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: expected an examples manifest")
    return [dict(row) for row in rows]


def _seed(seed: int, *parts: object) -> int:
    digest = hashlib.sha256(
        "\0".join((str(seed), *(str(part) for part in parts))).encode()
    ).digest()
    return int.from_bytes(digest[:8], "little")


def _verified_audio(row: Mapping[str, Any]) -> Path:
    path = Path(str(row.get("path", ""))).expanduser().resolve()
    expected = str(row.get("audio_sha256", row.get("sha256", "")))
    if not path.is_file() or not expected or sha256_file(path) != expected:
        raise ValueError(f"audio binding drift: {path}")
    return path


def _active_trim(samples: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float32)
    peak = float(np.max(np.abs(values)))
    if peak <= 1e-8:
        raise ValueError("source audio is silent")
    active = np.flatnonzero(np.abs(values) >= max(peak * 0.025, 1e-4))
    margin = int(0.12 * SAMPLE_RATE)
    start = max(0, int(active[0]) - margin)
    stop = min(len(values), int(active[-1]) + margin + 1)
    return values[start:stop]


def _place(
    samples: np.ndarray, rng: np.random.Generator, *, random_position: bool
) -> tuple[np.ndarray, tuple[float, float]]:
    values = _active_trim(samples)
    if len(values) > CONTEXT_SAMPLES:
        excess = len(values) - CONTEXT_SAMPLES
        start = excess // 2
        values = values[start : start + CONTEXT_SAMPLES]
    available = CONTEXT_SAMPLES - len(values)
    left = int(rng.integers(0, available + 1)) if random_position else available // 2
    output = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)
    output[left : left + len(values)] = values
    return output, (left / SAMPLE_RATE, (left + len(values)) / SAMPLE_RATE)


def _background_context(
    samples: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, int, bool]:
    values = np.asarray(samples, dtype=np.float32)
    if len(values) >= CONTEXT_SAMPLES:
        start = int(rng.integers(0, len(values) - CONTEXT_SAMPLES + 1))
        return values[start : start + CONTEXT_SAMPLES], start, False
    repeats = math.ceil(CONTEXT_SAMPLES / len(values))
    return np.tile(values, repeats)[:CONTEXT_SAMPLES], 0, repeats > 1


def _write_waveform(root: Path, source_id: str, values: np.ndarray) -> tuple[Path, str]:
    name = hashlib.sha256(source_id.encode()).hexdigest()[:28] + ".wav"
    path = root / name
    sf.write(path, values, SAMPLE_RATE, subtype="PCM_16")
    return path.resolve(), sha256_file(path)


def _lineage(row: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for key in (
        "provider",
        "speaker_id",
        "session_id",
        "ancestry_id",
        "ancestry_ids",
        "voice_id",
        "voice",
        "source_group",
    ):
        if row.get(key) not in (None, "", []):
            result[key] = row[key]
    return result


def build(
    source_manifest: Path,
    negative_manifests: Sequence[Path],
    rir_manifest: Path,
    output: Path,
    *,
    overlay_snr_db: Sequence[float] = (-5.0, 0.0, 5.0, 10.0, 15.0),
    negative_windows_per_source: int = 2,
    maximum_negative_windows_per_group_split: int = 2000,
    seed: int = 231,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output}")
    if negative_windows_per_source < 1 or maximum_negative_windows_per_group_split < 1:
        raise ValueError("negative window limits must be positive")
    snrs = tuple(float(value) for value in overlay_snr_db)
    if not snrs or any(not math.isfinite(value) for value in snrs):
        raise ValueError("overlay SNR values must be finite and nonempty")

    source_rows = _load(source_manifest)
    negative_rows = [row for path in negative_manifests for row in _load(path)]
    rir_rows = [
        row
        for row in _load(rir_manifest)
        if row.get("split") == "train" and row.get("training_eligible") is True
    ]
    backgrounds = [
        row
        for row in negative_rows
        if row.get("split") == "train"
        and row.get("training_eligible") is True
        and row.get("source_group") in BACKGROUND_GROUPS
        and row.get("locked_deployment_anchor") is not True
    ]
    if not backgrounds or not rir_rows:
        raise ValueError("training overlays require eligible backgrounds and RIRs")
    for row in (*backgrounds, *rir_rows):
        _verified_audio(row)

    output = output.resolve()
    wave_root = output / "waveforms"
    wave_root.mkdir(parents=True)
    features: list[np.ndarray] = []
    ledger: list[dict[str, Any]] = []
    skipped_silent_negative_windows = 0

    def append(
        source_id: str,
        row: Mapping[str, Any],
        waveform: np.ndarray,
        *,
        label: int,
        parent_source_id: str,
        augmentation: Mapping[str, Any] | None,
    ) -> None:
        audio_path, audio_hash = _write_waveform(wave_root, source_id, waveform)
        feature = np.asarray(frontend(waveform), dtype=np.float32)
        feature_hash = hashlib.sha256(np.ascontiguousarray(feature).tobytes()).hexdigest()
        index = len(features)
        features.append(feature)
        ledger.append(
            {
                "source_id": source_id,
                "parent_source_id": parent_source_id,
                "split": str(row["split"]),
                "label": int(label),
                "feature_index": index,
                "feature_sha256": feature_hash,
                "path": str(audio_path),
                "audio_sha256": audio_hash,
                "source_audio_sha256": str(
                    row.get("audio_sha256", row.get("sha256", ""))
                ),
                "duration_seconds": CONTEXT_SAMPLES / SAMPLE_RATE,
                "augmentation": dict(augmentation) if augmentation else None,
                **_lineage(row),
            }
        )

    active_sources = [
        row
        for row in source_rows
        if row.get("training_eligible") is True
        and row.get("split") in SPLITS
        and int(row.get("label", -1)) in (0, 1)
    ]
    for row in sorted(active_sources, key=lambda item: str(item["source_id"])):
        source_id = str(row["source_id"])
        waveform = load_audio(_verified_audio(row))
        clean_rng = np.random.default_rng(_seed(seed, source_id, "clean"))
        clean, active_span = _place(waveform, clean_rng, random_position=False)
        append(
            f"{source_id}::clean",
            row,
            clean,
            label=int(row["label"]),
            parent_source_id=source_id,
            augmentation={"variant": "clean", "active_span_seconds": active_span},
        )
        if row["split"] != "train":
            continue
        variant_snrs = snrs if int(row["label"]) == 1 else (snrs[len(snrs) // 2],)
        for variant_index, snr in enumerate(variant_snrs):
            variant_seed = _seed(seed, source_id, "overlay", variant_index)
            rng = np.random.default_rng(variant_seed)
            foreground, active_span = _place(waveform, rng, random_position=True)
            background_row = backgrounds[int(rng.integers(0, len(backgrounds)))]
            background, crop, tiled = _background_context(
                load_audio(_verified_audio(background_row)), rng
            )
            rir_row = rir_rows[int(rng.integers(0, len(rir_rows)))]
            foreground, arrival = apply_room_impulse_response(
                foreground, load_audio(_verified_audio(rir_row))
            )
            gain_db = float(rng.uniform(-6.0, 3.0))
            foreground = apply_gain_db(foreground, gain_db)
            mixed = mix_at_snr(foreground, background, active_span, snr)
            append(
                f"{source_id}::overlay::{variant_index}",
                row,
                mixed,
                label=int(row["label"]),
                parent_source_id=source_id,
                augmentation={
                    "variant": "randomized_background_rir_overlay",
                    "seed": variant_seed,
                    "snr_db": snr,
                    "foreground_gain_db": gain_db,
                    "active_span_seconds": active_span,
                    "background_source_id": background_row["source_id"],
                    "background_audio_sha256": background_row.get("audio_sha256"),
                    "background_crop_start_sample": crop,
                    "background_tiled": tiled,
                    "rir_source_id": rir_row["source_id"],
                    "rir_audio_sha256": rir_row.get("audio_sha256", rir_row.get("sha256")),
                    "rir_arrival_trim_samples": arrival,
                },
            )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in negative_rows:
        split = str(row.get("split", ""))
        group = str(row.get("source_group", ""))
        if (
            split in SPLITS
            and group in BACKGROUND_GROUPS
            and int(row.get("label", -1)) == 0
            and row.get("locked_deployment_anchor") is not True
            and (
                (split == "train" and row.get("training_eligible") is True)
                or (split != "train" and row.get("training_eligible") is False)
            )
        ):
            grouped[(split, group)].append(row)

    for (split, group), rows in sorted(grouped.items()):
        ranked = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"{seed}\0{split}\0{group}\0{row.get('source_id')}".encode()
            ).digest(),
        )
        emitted = 0
        for row in ranked:
            waveform = load_audio(_verified_audio(row))
            for window_index in range(negative_windows_per_source):
                if emitted >= maximum_negative_windows_per_group_split:
                    break
                rng = np.random.default_rng(
                    _seed(seed, row.get("source_id"), "negative", window_index)
                )
                if len(waveform) >= CONTEXT_SAMPLES:
                    start = int(rng.integers(0, len(waveform) - CONTEXT_SAMPLES + 1))
                    context = waveform[start : start + CONTEXT_SAMPLES]
                    tiled = False
                else:
                    repeats = math.ceil(CONTEXT_SAMPLES / len(waveform))
                    context = np.tile(waveform, repeats)[:CONTEXT_SAMPLES]
                    start = 0
                    tiled = repeats > 1
                # Some environmental recordings contain long digital-silence
                # tails.  Keeping those creates the exact same all-zero WAV in
                # multiple splits, which is both uninformative and a strict
                # content-hash leak across train/validation/test.
                if float(np.max(np.abs(context))) <= 1e-5:
                    skipped_silent_negative_windows += 1
                    continue
                parent = str(row["source_id"])
                append(
                    f"{parent}::window::{window_index}",
                    {**row, "split": split, "source_group": group},
                    context,
                    label=0,
                    parent_source_id=parent,
                    augmentation={
                        "variant": "deterministic_negative_window",
                        "seed": _seed(seed, parent, "negative", window_index),
                        "crop_start_sample": start,
                        "tiled": tiled,
                    },
                )
                emitted += 1
            if emitted >= maximum_negative_windows_per_group_split:
                break

    counts = Counter((row["split"], row["label"], row["source_group"]) for row in ledger)
    for split in SPLITS:
        if not any(row["split"] == split and row["label"] == 1 for row in ledger):
            raise ValueError(f"{split} has no positives")
        if not any(row["split"] == split and row["label"] == 0 for row in ledger):
            raise ValueError(f"{split} has no negatives")

    feature_array = np.stack(features).astype(np.float32, copy=False)
    feature_path = output / "source-features.npy"
    np.save(feature_path, feature_array, allow_pickle=False)
    payload = {
        "schema_version": 1,
        "recipe": "kizz_control_candidate_source_features_v1",
        "array_sha256": {feature_path.name: sha256_file(feature_path)},
        "input_bindings": {
            "source_manifest": {
                "path": str(source_manifest.resolve()),
                "sha256": sha256_file(source_manifest),
            },
            "negative_manifests": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path in negative_manifests
            ],
            "rir_manifest": {
                "path": str(rir_manifest.resolve()),
                "sha256": sha256_file(rir_manifest),
            },
        },
        "augmentation_contract": {
            "training_only": True,
            "overlay_snr_db": list(snrs),
            "randomized_background": True,
            "randomized_rir": True,
            "seed": seed,
            "skipped_silent_negative_windows": skipped_silent_negative_windows,
        },
        "counts": [
            {"split": split, "label": label, "source_group": group, "count": count}
            for (split, label, group), count in sorted(counts.items())
        ],
        "examples": ledger,
    }
    (output / "source-manifest.json").write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--negative-manifest", type=Path, action="append", required=True)
    parser.add_argument("--rir-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlay-snr-db", type=float, action="append")
    parser.add_argument("--negative-windows-per-source", type=int, default=2)
    parser.add_argument("--maximum-negative-windows-per-group-split", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=231)
    args = parser.parse_args(argv)
    report = build(
        args.source_manifest,
        args.negative_manifest,
        args.rir_manifest,
        args.output,
        overlay_snr_db=args.overlay_snr_db or (-5.0, 0.0, 5.0, 10.0, 15.0),
        negative_windows_per_source=args.negative_windows_per_source,
        maximum_negative_windows_per_group_split=args.maximum_negative_windows_per_group_split,
        seed=args.seed,
    )
    print(json.dumps({"examples": len(report["examples"]), "counts": report["counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
