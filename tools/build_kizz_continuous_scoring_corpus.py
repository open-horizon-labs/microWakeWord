#!/usr/bin/env python3
"""Compose fixed continuous-context Kizz detector scoring features.

Each existing immutable 260-frame source receives a real ambient feature prefix.
Background choice, crop, and level are deterministic and split-local. The
product C microfrontend runs on quantized prefix PCM, while the source tensor is
preserved byte-for-byte so historical crop/pad decisions cannot drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from microwakeword.audio.audio_utils import MicroFrontend
from tools.trace_kizz_ordered_state_detector import feature_sha256, sha256_file


SAMPLE_RATE = 16_000
FEATURE_BINS = 40
# Firmware supplies one 10 ms hop per call. The frontend retains its own
# 30 ms analysis window and therefore emits two fewer frames than calls.
SAMPLES_PER_CALL = 160
SPLITS = ("train", "validation", "test")
BACKGROUND_GAIN_DB = (-6.0, -3.0, 0.0, 3.0, 6.0)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        with open(temporary, "wb") as output:
            np.save(output, values, allow_pickle=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _load_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _read_mono(path: Path) -> np.ndarray:
    values, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate <= 0 or values.shape[1] < 1 or not len(values):
        raise ValueError(f"invalid audio: {path}")
    mono = np.mean(np.asarray(values, dtype=np.float32), axis=1)
    if rate != SAMPLE_RATE:
        divisor = math.gcd(int(rate), SAMPLE_RATE)
        mono = np.asarray(
            resample_poly(mono, SAMPLE_RATE // divisor, int(rate) // divisor),
            dtype=np.float32,
        )
    if not np.all(np.isfinite(mono)):
        raise ValueError(f"non-finite audio: {path}")
    return mono


def _stable_digest(*values: object) -> bytes:
    return hashlib.sha256("\0".join(map(str, values)).encode("utf-8")).digest()


def _select_background(
    pools: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    split: str,
    source_id: str,
    seed: int,
) -> Mapping[str, Any]:
    pool = pools.get(split)
    if not pool:
        raise ValueError(f"background corpus has no {split} files")
    index = int.from_bytes(_stable_digest(seed, split, source_id)[:8], "big") % len(pool)
    return pool[index]


def _frontend(pcm: np.ndarray, expected_frames: int) -> np.ndarray:
    processor = MicroFrontend()
    process = getattr(processor, "process_samples", None) or processor.ProcessSamples
    raw = np.asarray(pcm, dtype="<i2").tobytes()
    offset = 0
    rows: list[np.ndarray] = []
    while offset + SAMPLES_PER_CALL * 2 <= len(raw):
        result = process(raw[offset : offset + SAMPLES_PER_CALL * 2])
        emitted = np.asarray(result.features, dtype=np.float32)
        if emitted.size:
            if emitted.size % FEATURE_BINS:
                raise ValueError("microfrontend feature width drift")
            rows.extend(emitted.reshape(-1, FEATURE_BINS))
        used = int(getattr(result, "samples_read", SAMPLES_PER_CALL))
        if not 0 < used <= SAMPLES_PER_CALL:
            raise ValueError("microfrontend made invalid progress")
        offset += used * 2
    features = np.asarray(rows, dtype=np.float32)
    if features.shape != (expected_frames, FEATURE_BINS) or not np.all(
        np.isfinite(features)
    ):
        raise ValueError(
            f"microfrontend emitted {features.shape}, expected {(expected_frames, FEATURE_BINS)}"
        )
    return features


def build(
    source_manifest: Path,
    source_features: Path,
    background_corpus: Path,
    output_directory: Path,
    *,
    prefix_seconds: float = 2.0,
    seed: int = 2027,
) -> dict[str, Any]:
    source_manifest = source_manifest.resolve()
    source_features = source_features.resolve()
    background_corpus = background_corpus.resolve()
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_directory}")
    if not math.isfinite(prefix_seconds) or prefix_seconds <= 0:
        raise ValueError("prefix_seconds must be finite and positive")
    source_payload = _load_object(source_manifest, "source manifest")
    background_payload = _load_object(background_corpus, "background corpus")
    raw_sources = source_payload.get("examples")
    raw_backgrounds = background_payload.get("files")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("source manifest has no examples")
    if not isinstance(raw_backgrounds, list) or not raw_backgrounds:
        raise ValueError("background corpus has no files")
    original_features = np.load(source_features, mmap_mode="r", allow_pickle=False)
    if (
        original_features.ndim != 3
        or tuple(original_features.shape[1:]) != (260, FEATURE_BINS)
        or len(original_features) != len(raw_sources)
    ):
        raise ValueError("source features must be [N,260,40]")
    declared_source_features = source_payload.get("array_sha256", {}).get(
        source_features.name
    )
    if declared_source_features != sha256_file(source_features):
        raise ValueError("source feature-array hash drift")

    prefix_frames = int(round(prefix_seconds / 0.01))
    prefix_audio_samples = (prefix_frames + 2) * SAMPLES_PER_CALL

    background_root = background_corpus.parent
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    excluded_short_backgrounds = 0
    for raw in raw_backgrounds:
        if not isinstance(raw, dict) or raw.get("evidence_split") not in SPLITS:
            raise ValueError("background rows require train/validation/test evidence_split")
        path = (background_root / str(raw.get("path", ""))).resolve()
        if not path.is_file() or sha256_file(path) != raw.get("sha256"):
            raise ValueError(f"background hash drift: {path}")
        info = sf.info(path)
        if info.samplerate <= 0 or info.frames / info.samplerate + 1e-9 < (
            prefix_audio_samples / SAMPLE_RATE
        ):
            excluded_short_backgrounds += 1
            continue
        pools[str(raw["evidence_split"])].append({**raw, "_path": path})
    for split in SPLITS:
        pools[split].sort(key=lambda row: (str(row.get("sha256")), str(row["_path"])))

    # The detector consumes a fixed feature context even when the source audio
    # clips have different recorded durations.  Derive that context from the
    # bound tensor geometry instead of treating per-clip audio duration as model
    # geometry.  The frontend's 30 ms analysis window means N feature rows span
    # (N + 2) 10 ms calls.
    derived_source_duration = (int(original_features.shape[1]) + 2) * 0.01
    declared_source_duration = source_payload.get("context_duration_seconds")
    if declared_source_duration is not None:
        if (
            not isinstance(declared_source_duration, (int, float))
            or not math.isfinite(float(declared_source_duration))
            or abs(float(declared_source_duration) - derived_source_duration) > 1e-9
        ):
            raise ValueError("source context duration differs from feature geometry")
    source_duration = derived_source_duration
    expected_frames = prefix_frames + int(original_features.shape[1])

    features: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    seen_source_ids: set[str] = set()
    for index, raw in enumerate(raw_sources):
        if not isinstance(raw, dict):
            raise ValueError("source examples must be objects")
        split = str(raw.get("split"))
        source_id = str(raw.get("source_id", ""))
        if split not in SPLITS or not source_id or source_id in seen_source_ids:
            raise ValueError("source identity/split drift")
        seen_source_ids.add(source_id)
        if raw.get("capture_path"):
            source_path = Path(str(raw["capture_path"])).resolve()
            source_hash = raw.get("capture_audio_sha256")
        else:
            source_path = Path(str(raw.get("path", ""))).resolve()
            source_hash = raw.get("audio_sha256", raw.get("sha256"))
        if not source_path.is_file() or sha256_file(source_path) != source_hash:
            raise ValueError(f"source audio hash drift: {source_path}")
        source_index = raw.get("feature_index", index)
        if source_index != index:
            raise ValueError("source feature ordering drift")
        source_tensor = np.asarray(original_features[index], dtype=np.float32)
        if feature_sha256(source_tensor) != raw.get("feature_sha256"):
            raise ValueError(f"source feature hash drift: {source_id}")

        background_row = _select_background(
            pools, split=split, source_id=source_id, seed=seed
        )
        background_path = Path(background_row["_path"])
        background = _read_mono(background_path)
        if len(background) < prefix_audio_samples:
            raise ValueError(f"background is too short after resampling: {background_path}")
        digest = _stable_digest(seed, source_id, background_row["sha256"])
        crop_start = int.from_bytes(digest[8:16], "big") % (
            len(background) - prefix_audio_samples + 1
        )
        gain_db = BACKGROUND_GAIN_DB[
            int.from_bytes(digest[16:24], "big") % len(BACKGROUND_GAIN_DB)
        ]
        prefix = background[crop_start : crop_start + prefix_audio_samples]
        prefix = np.clip(prefix * np.float32(10.0 ** (gain_db / 20.0)), -1.0, 1.0)
        prefix_pcm = np.rint(prefix * 32767.0).astype("<i2")
        prefix_features = _frontend(prefix_pcm, prefix_frames)
        item_features = np.concatenate((prefix_features, source_tensor), axis=0)
        if item_features.shape != (expected_frames, FEATURE_BINS):
            raise ValueError("continuous feature concatenation shape drift")
        composed_source_id = f"{source_id}::continuous-prefix-v1"
        feature_hash = feature_sha256(item_features)
        lineage = {
            key: raw[key]
            for key in (
                "provider",
                "source_group",
                "speaker_id",
                "voice_id",
                "session_id",
                "ancestry_id",
                "parent_source_audio_sha256",
            )
            if raw.get(key) not in (None, "")
        }
        rows.append(
            {
                "source_id": composed_source_id,
                "parent_source_id": source_id,
                "feature_index": index,
                "feature_sha256": feature_hash,
                "source_audio_sha256": str(source_hash),
                "split": split,
                "label": int(raw["label"]),
                "duration_seconds": float(source_duration) + prefix_frames * 0.01,
                **(
                    {
                        "foreground_duration_seconds": float(
                            raw.get("duration_seconds", raw.get("source_duration_seconds"))
                        )
                    }
                    if isinstance(
                        raw.get("duration_seconds", raw.get("source_duration_seconds")),
                        (int, float),
                    )
                    and math.isfinite(
                        float(raw.get("duration_seconds", raw.get("source_duration_seconds")))
                    )
                    and float(
                        raw.get("duration_seconds", raw.get("source_duration_seconds"))
                    )
                    > 0
                    else {}
                ),
                **lineage,
                "composition": {
                    "recipe": "continuous_ambient_feature_prefix_v1",
                    "seed": seed,
                    "prefix_seconds": prefix_frames * 0.01,
                    "background_gain_db": gain_db,
                    "foreground": {
                        "path": str(source_path),
                        "sha256": str(source_hash),
                        "feature_sha256": str(raw["feature_sha256"]),
                    },
                    "background": {
                        "path": str(background_path),
                        "sha256": str(background_row["sha256"]),
                        "source": background_row.get("source"),
                        "category": background_row.get("category"),
                        "environment": background_row.get("environment"),
                        "evidence_split": split,
                        "crop_start_sample": crop_start,
                    },
                },
            }
        )
        features.append(item_features)

    feature_array = np.stack(features).astype(np.float32, copy=False)
    output_directory.mkdir(parents=True)
    features_path = output_directory / "source-features.npy"
    _atomic_npy(features_path, feature_array)
    manifest = {
        "schema_version": 1,
        "recipe": "kizz_control_continuous_feature_prefix_scoring_corpus_v1",
        "deployment_qualification": False,
        "context_duration_seconds": float(source_duration) + prefix_frames * 0.01,
        "input_shape": [expected_frames, FEATURE_BINS],
        "composition": {
            "seed": seed,
            "prefix_seconds": prefix_frames * 0.01,
            "background_gain_db": list(BACKGROUND_GAIN_DB),
            "background_split_policy": "same_split_only",
            "source_feature_policy": "preserved_byte_for_byte",
            "prefix_frontend": "product_c_microfrontend",
            "excluded_short_backgrounds": excluded_short_backgrounds,
        },
        "inputs": {
            "source_manifest": {
                "path": str(source_manifest),
                "sha256": sha256_file(source_manifest),
            },
            "source_features": {
                "path": str(source_features),
                "sha256": sha256_file(source_features),
            },
            "background_corpus": {
                "path": str(background_corpus),
                "sha256": sha256_file(background_corpus),
            },
        },
        "array_sha256": {features_path.name: sha256_file(features_path)},
        "counts": {
            "examples": len(rows),
            "positive": sum(row["label"] == 1 for row in rows),
            "negative": sum(row["label"] == 0 for row in rows),
            "by_split": {
                split: sum(row["split"] == split for row in rows) for split in SPLITS
            },
        },
        "examples": rows,
    }
    manifest_path = output_directory / "source-manifest.json"
    _atomic_bytes(manifest_path, _canonical_bytes(manifest))
    return {
        "output": str(output_directory),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_features": str(features_path),
        "source_features_sha256": sha256_file(features_path),
        "shape": list(feature_array.shape),
        "counts": manifest["counts"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-features", type=Path, required=True)
    parser.add_argument("--background-corpus", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--prefix-seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args(argv)
    print(json.dumps(build(**vars(args)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
