#!/usr/bin/env python3
"""Materialize every eligible training-speech negative for compact Kizz."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.build_kizz_aligned_teacher_features_v3 import (
    CONTEXT_SAMPLES,
    SAMPLE_RATE,
    frontend,
    load_audio,
)
from tools.build_kizz_phoneme_distillation_corpus import (
    _locked_hashes,
    _rows,
    sha256_file,
)


def deterministic_context(
    samples: np.ndarray, audio_sha256: str
) -> tuple[np.ndarray, int]:
    values = np.asarray(samples, dtype=np.float32)
    if len(values) >= CONTEXT_SAMPLES:
        extent = len(values) - CONTEXT_SAMPLES + 1
        start = int(audio_sha256[:16], 16) % extent
        return values[start : start + CONTEXT_SAMPLES], start
    left = (CONTEXT_SAMPLES - len(values)) // 2
    return np.pad(values, (left, CONTEXT_SAMPLES - len(values) - left)), -left


def select_rows(source_manifest: Path, continuous_lock: Path) -> list[dict]:
    locked = _locked_hashes(continuous_lock)
    selected = []
    for row in _rows(source_manifest):
        identity = str(row.get("audio_sha256", ""))
        if (
            int(row.get("label", -1)) == 0
            and row.get("split") == "train"
            and row.get("source_group") == "public_speech"
            and identity
            and identity not in locked
            and row.get("training_eligible") is not False
            and not row.get("locked_deployment_anchor")
        ):
            selected.append(dict(row))
    selected.sort(key=lambda row: (row["audio_sha256"], row["source_id"]))
    if not selected:
        raise ValueError("source manifest has no eligible training speech")
    if len({row["audio_sha256"] for row in selected}) != len(selected):
        raise ValueError("expanded public negatives contain duplicate audio")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--continuous-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = select_rows(args.source_manifest, args.continuous_lock)
    args.output.mkdir(parents=True, exist_ok=True)
    features_path = args.output / "features.npy"
    features = np.lib.format.open_memmap(
        features_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(rows), 260, 40),
    )
    ledger = []
    for index, row in enumerate(rows):
        path = Path(row["path"]).resolve()
        if not path.is_file() or sha256_file(path) != row["audio_sha256"]:
            raise ValueError(f"source audio hash drift: {path}")
        context, start = deterministic_context(load_audio(path), row["audio_sha256"])
        pcm = np.rint(np.clip(context, -1.0, 1.0) * 32767.0).astype("<i2")
        features[index] = frontend(pcm.astype(np.float32) / 32767.0)
        ledger.append(
            {
                "source_id": row["source_id"],
                "path": str(path),
                "audio_sha256": row["audio_sha256"],
                "speaker_id": row.get("speaker_id"),
                "context_start_samples": start,
                "context_samples": CONTEXT_SAMPLES,
                "sample_rate": SAMPLE_RATE,
            }
        )
        if (index + 1) % 250 == 0 or index + 1 == len(rows):
            features.flush()
            print(json.dumps({"materialized": index + 1, "total": len(rows)}), flush=True)
    del features
    metadata = {
        "schema_version": 1,
        "representation": "expanded_public_speech_fixed_context_features",
        "selection": {
            "split": "train",
            "source_group": "public_speech",
            "algorithm": "all_eligible_sorted_by_audio_sha256_source_id",
            "context_algorithm": "sha256_offset_fixed_2.62_seconds",
        },
        "source_manifest": {
            "path": str(args.source_manifest.resolve()),
            "sha256": sha256_file(args.source_manifest),
        },
        "continuous_lock": {
            "path": str(args.continuous_lock.resolve()),
            "sha256": sha256_file(args.continuous_lock),
        },
        "features": {
            "path": str(features_path.resolve()),
            "sha256": sha256_file(features_path),
            "shape": [len(rows), 260, 40],
            "dtype": "float16",
        },
        "count": len(rows),
        "ledger_sha256": hashlib.sha256(
            json.dumps(ledger, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "examples": ledger,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
