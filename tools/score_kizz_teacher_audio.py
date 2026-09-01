#!/usr/bin/env python3
"""Score teacher windows over quarantined WAV evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from microwakeword.kizz_teacher import build_teacher
from microwakeword.ordered_state import ordered_state_sequence_score_numpy
from tools.score_ordered_state_streams import frontend_features, read_wav
from tools.score_kizz_teacher import windows_for_item


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text())
    base = args.manifest.parent
    model = build_teacher()
    model.load_weights(args.model)
    reports = []
    for record in manifest["observations"]:
        path = (base / record["audio_path"]).resolve()
        samples, duration = read_wav(path)
        features = np.asarray(list(frontend_features(samples)), dtype=np.float32)
        windows = windows_for_item(features, all_windows=True)
        scores = []
        for offset in range(0, len(windows), args.batch_size):
            logits = model.predict(
                np.asarray(windows[offset : offset + args.batch_size]), verbose=0
            )
            scores.extend(float(x) for x in ordered_state_sequence_score_numpy(logits))
        reports.append(
            {
                "observation_id": record["observation_id"],
                "path": str(path),
                "duration_seconds": duration,
                "feature_frames": int(len(features)),
                "window_count": len(scores),
                "maximum_score": float(max(scores)) if scores else None,
                "mean_score": float(np.mean(scores)) if scores else None,
                "scores": scores,
            }
        )
    result = {
        "schema_version": 1,
        "model": str(args.model.resolve()),
        "manifest": str(args.manifest.resolve()),
        "observation_count": len(reports),
        "reports": sorted(reports, key=lambda x: x["maximum_score"], reverse=True),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "observation_count": len(reports),
        "top_false_wakes": [
            {"id": x["observation_id"], "maximum_score": x["maximum_score"]}
            for x in result["reports"][:10]
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
