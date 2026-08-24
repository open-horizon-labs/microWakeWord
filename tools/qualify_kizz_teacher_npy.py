#!/usr/bin/env python3
"""Window-level qualification report for fixed-feature C teacher arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from microwakeword.kizz_teacher import build_teacher
from tools.qualify_kizz_teacher import fast_sequence_scores


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scores(model, path: Path, batch_size: int) -> np.ndarray:
    values = np.load(path, mmap_mode="r")
    result = []
    for offset in range(0, len(values), batch_size):
        logits = model.predict(
            np.asarray(values[offset : offset + batch_size], dtype=np.float32),
            verbose=0,
        )
        result.extend(fast_sequence_scores(logits))
    return np.asarray(result, dtype=np.float64)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--positive", type=Path, action="append", required=True)
    p.add_argument("--negative", type=Path, action="append", required=True)
    p.add_argument("--heldout", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--min-recall", type=float, default=0.90)
    p.add_argument("--max-faph", type=float, default=0.10)
    args = p.parse_args()

    model = build_teacher()
    model.load_weights(args.model)
    positive = np.concatenate(
        [scores(model, path, args.batch_size) for path in args.positive]
    )
    negative = np.concatenate(
        [scores(model, path, args.batch_size) for path in args.negative]
    )
    heldout = scores(model, args.heldout, args.batch_size)
    thresholds = np.unique(np.concatenate([positive, negative]))
    exposure_seconds = len(negative) * 2.6
    candidates = []
    for threshold in thresholds:
        recall = float(np.mean(positive >= threshold))
        false_accepts = int(np.sum(negative >= threshold))
        faph = false_accepts / max(exposure_seconds / 3600.0, 1e-12)
        if recall >= args.min_recall and faph <= args.max_faph:
            candidates.append((recall, -faph, float(threshold), false_accepts))
    if candidates:
        recall, neg_faph, threshold, false_accepts = max(candidates)
        qualified = True
    else:
        recall = 0.0
        neg_faph = 0.0
        threshold = None
        false_accepts = None
        qualified = False
    heldout_accepts = None if threshold is None else int(np.sum(heldout >= threshold))
    result = {
        "schema_version": 1,
        "model": str(args.model.resolve()),
        "model_sha256": sha256(args.model),
        "positive_count": len(positive),
        "negative_count": len(negative),
        "negative_exposure_seconds_assumed": exposure_seconds,
        "heldout_false_wake_count": len(heldout),
        "heldout_false_wake_accepts": heldout_accepts,
        "limits": {
            "min_recall": args.min_recall,
            "max_faph": args.max_faph,
            "max_heldout_false_wake_accepts": 0,
        },
        "operating_point": {
            "qualified_window_gate": qualified,
            "positive_recall": recall,
            "faph_assuming_2_6s_windows": -neg_faph,
            "threshold": threshold,
            "false_accepts": false_accepts,
        },
        "score_summary": {
            "positive": [
                float(np.min(positive)),
                float(np.median(positive)),
                float(np.max(positive)),
            ],
            "negative": [
                float(np.min(negative)),
                float(np.median(negative)),
                float(np.max(negative)),
            ],
            "heldout": [
                float(np.min(heldout)),
                float(np.median(heldout)),
                float(np.max(heldout)),
            ],
        },
        "qualified": bool(qualified and heldout_accepts == 0),
        "note": "C arrays are fixed windows; exposure is an explicit 2.6-second approximation, not an ambient-hours qualification.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["qualified"] else 2)


if __name__ == "__main__":
    main()
