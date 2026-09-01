#!/usr/bin/env python3
"""Compose untouched target-device replay features for deployed INT8 tracing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


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


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def prepare(provenance_paths: Sequence[Path], output: Path) -> dict:
    if not provenance_paths:
        raise ValueError("at least one device feature provenance is required")
    arrays = []
    rows = []
    bindings = []
    capture_ids: set[str] = set()
    capture_hashes: set[str] = set()
    source_hashes: set[str] = set()
    feature_index = 0
    for provenance_path in provenance_paths:
        provenance_path = provenance_path.expanduser().resolve()
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if (
            provenance.get("kind") != "kizz_control_ordered_state_target_device_replay_features"
            or provenance.get("gate_scope") != "locked_test_only_target_channel_positive_features"
            or provenance.get("training_eligible") is not False
        ):
            raise ValueError(f"{provenance_path}: not locked device replay features")
        binding = provenance.get("outputs", {}).get("features", {})
        feature_path = Path(str(binding.get("path", ""))).resolve()
        if not feature_path.is_file() or sha256_file(feature_path) != binding.get("sha256"):
            raise ValueError(f"{provenance_path}: feature binding drift")
        values = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        results = provenance.get("results", [])
        if values.ndim != 3 or tuple(values.shape[1:]) != (260, 40) or len(values) != len(results):
            raise ValueError(f"{provenance_path}: feature/result shape drift")
        for index, (sample, result) in enumerate(zip(values, results)):
            capture_id = str(result.get("capture_id", ""))
            capture_hash = str(result.get("audio_sha256", ""))
            source_hash = str(result.get("source_audio_sha256", ""))
            capture_path = Path(str(result.get("path", ""))).resolve()
            if (
                not capture_id
                or capture_id in capture_ids
                or not capture_hash
                or capture_hash in capture_hashes
                or not source_hash
                or source_hash in source_hashes
                or not capture_path.is_file()
                or sha256_file(capture_path) != capture_hash
            ):
                raise ValueError("device replay identity, ancestry, or audio hash drift")
            rows.append(
                {
                    "source_id": f"device-qualification:{capture_id}",
                    "feature_index": feature_index,
                    "feature_sha256": feature_sha256(sample),
                    "split": "test",
                    "label": 1,
                    "provider": result.get("provider"),
                    "voice": result.get("voice"),
                    "capture_audio_sha256": capture_hash,
                    "source_audio_sha256": source_hash,
                    "envelope_correlation": result.get("envelope_correlation"),
                    "playback_lag_seconds": result.get("playback_lag_seconds"),
                }
            )
            feature_index += 1
            capture_ids.add(capture_id)
            capture_hashes.add(capture_hash)
            source_hashes.add(source_hash)
        arrays.append(np.asarray(values))
        bindings.append({"path": str(provenance_path), "sha256": sha256_file(provenance_path), "count": len(values)})

    output = output.expanduser().resolve()
    feature_path = output / "device-replay-features.npy"
    manifest_path = output / "device-replay-manifest.json"
    combined = np.concatenate(arrays, axis=0)
    _atomic_npy(feature_path, combined)
    payload = {
        "schema_version": 1,
        "recipe": "kizz_control_fresh_target_device_trace_bundle_v1",
        "deployment_qualification": False,
        "locked_before_detector_scoring": True,
        "training_eligible": False,
        "feature_provenance": bindings,
        "array_sha256": {feature_path.name: sha256_file(feature_path)},
        "counts": {"examples": len(rows), "voices": len({str(row["voice"]) for row in rows})},
        "examples": rows,
    }
    _atomic_json(manifest_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-provenance", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = prepare(args.feature_provenance, args.output)
    print(json.dumps(payload["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
