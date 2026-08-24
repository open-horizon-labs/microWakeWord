#!/usr/bin/env python3
"""Materialize quarantined false-wake audio as fixed teacher hard-negative windows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from tools.score_kizz_teacher import windows_for_item
from tools.score_ordered_state_streams import frontend_features, read_wav


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--held-out-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=24107)
    args = parser.parse_args(argv)
    if not 0 < args.held_out_fraction < 1:
        parser.error("held-out-fraction must be between zero and one")

    manifest = json.loads(args.manifest.read_text())
    base = args.manifest.parent
    records = sorted(manifest["observations"], key=lambda item: item["observation_id"])
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(records))
    held_out_count = max(1, round(len(records) * args.held_out_fraction))
    held_out_indexes = set(int(value) for value in order[:held_out_count])
    windows_by_split = {"training": [], "held_out": []}
    records_by_split = {"training": [], "held_out": []}
    for index, record in enumerate(records):
        split = "held_out" if index in held_out_indexes else "training"
        path = (base / record["audio_path"]).resolve()
        samples, _ = read_wav(path)
        features = np.asarray(list(frontend_features(samples)), dtype=np.float32)
        windows = windows_for_item(features, all_windows=True)
        windows_by_split[split].extend(np.asarray(windows, dtype=np.float16))
        records_by_split[split].append(
            {
                "observation_id": record["observation_id"],
                "audio_sha256": record["audio_sha256"],
                "window_count": len(windows),
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "seed": args.seed,
        "held_out_fraction": args.held_out_fraction,
        "splits": {},
    }
    for split, values in windows_by_split.items():
        output = args.output / f"{split}.npy"
        array = np.asarray(values, dtype=np.float16).reshape((-1, 260, 40))
        np.save(output, array)
        metadata["splits"][split] = {
            "path": str(output.resolve()),
            "sha256": sha256_file(output),
            "window_count": len(array),
            "observations": records_by_split[split],
        }
    (args.output / "manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
