#!/usr/bin/env python3
"""Score an offline Kizz teacher on continuous feature sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from mmap_ninja.ragged import RaggedMmap

from microwakeword.kizz_teacher import FEATURE_BINS, INPUT_FRAMES
from microwakeword.ordered_state import ordered_state_sequence_score_numpy
from microwakeword.kizz_teacher import build_teacher


def parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("source must be ID=PATH")
    source_id, raw_path = value.split("=", 1)
    path = Path(raw_path).resolve()
    if not source_id or not path.is_dir():
        raise argparse.ArgumentTypeError("source must be ID=existing directory")
    return source_id, path


def windows_for_item(item: np.ndarray, *, all_windows: bool) -> list[np.ndarray]:
    item = np.asarray(item, dtype=np.float32)
    if item.ndim != 2 or item.shape[1] != FEATURE_BINS:
        raise ValueError("feature items must have shape [frames, 40]")
    if len(item) <= INPUT_FRAMES:
        result = np.zeros((INPUT_FRAMES, FEATURE_BINS), dtype=np.float32)
        result[: len(item)] = item
        return [result]
    starts = range(0, len(item) - INPUT_FRAMES + 1, 3) if all_windows else [0]
    return [item[start : start + INPUT_FRAMES] for start in starts]


def summarize(values: Sequence[float], threshold: float) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "threshold": float(threshold),
        "accepted": int(np.sum(values >= threshold)),
    }


def score_source(model, source_id: str, path: Path, args: argparse.Namespace) -> dict:
    mmap = RaggedMmap(path)
    values = []
    item_count = min(len(mmap), args.max_items) if args.max_items else len(mmap)
    batch = []

    def flush():
        if not batch:
            return
        logits = model.predict(np.asarray(batch, dtype=np.float32), verbose=0)
        values.extend(
            float(score)
            for score in ordered_state_sequence_score_numpy(logits)
        )
        batch.clear()

    for item_index in range(item_count):
        for window in windows_for_item(mmap[item_index], all_windows=args.all_windows):
            batch.append(window)
            if len(batch) >= args.batch_size:
                flush()
    flush()
    return {
        "id": source_id,
        "path": str(path),
        "item_count": item_count,
        "window_count": len(values),
        "summary": summarize(values, args.threshold),
        "scores": values if args.include_scores else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--positive-source", type=parse_source, required=True)
    parser.add_argument("--negative-source", type=parse_source, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--all-windows", action="store_true")
    parser.add_argument("--include-scores", action="store_true")
    args = parser.parse_args(argv)
    model = build_teacher()
    model.load_weights(args.model)
    reports = [score_source(model, *args.positive_source, args)]
    reports += [score_source(model, *source, args) for source in args.negative_source]
    result = {
        "schema_version": 1,
        "model": str(args.model.resolve()),
        "threshold": args.threshold,
        "positive": reports[0],
        "negative": reports[1:],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
