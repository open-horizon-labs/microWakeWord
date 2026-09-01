#!/usr/bin/env python3
"""Hard-gate an offline Kizz teacher before any student distillation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from mmap_ninja.ragged import RaggedMmap

from microwakeword.kizz_teacher import FEATURE_BINS, INPUT_FRAMES, build_teacher
from tools.score_kizz_teacher import windows_for_item


FEATURE_STEP_SECONDS = 0.01


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("source must be ID=PATH")
    source_id, raw_path = value.split("=", 1)
    path = Path(raw_path).resolve()
    if not source_id or not path.is_dir():
        raise argparse.ArgumentTypeError("source must be ID=existing directory")
    return source_id, path


def fast_sequence_scores(logits: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of the deployed 21-state Viterbi recurrence."""
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values, axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    rejection = np.logaddexp(log_probs[:, :, 0], log_probs[:, :, 1])
    emissions = log_probs[:, :, 2:] - rejection[:, :, None]
    alpha = np.full((len(values), 21), -np.inf, dtype=np.float64)
    completed = np.full(len(values), -np.inf, dtype=np.float64)
    log_self = np.log(0.6)
    log_next = np.log(0.4)
    for frame in range(emissions.shape[1]):
        advance = np.concatenate(
            [
                np.full((len(values), 1), -np.inf),
                alpha[:, :-1] + log_next,
            ],
            axis=1,
        )
        restart = np.concatenate(
            [
                np.zeros((len(values), 1)),
                np.full((len(values), 20), -np.inf),
            ],
            axis=1,
        )
        alpha = np.maximum.reduce([alpha + log_self, advance, restart])
        alpha += emissions[:, frame, :]
        completed = np.maximum(completed, alpha[:, -1])
    return completed


def score_windows(model, windows: Sequence[np.ndarray], batch_size: int) -> np.ndarray:
    values = []
    for start in range(0, len(windows), batch_size):
        batch = np.asarray(windows[start : start + batch_size], dtype=np.float32)
        logits = model.predict(batch, verbose=0)
        values.extend(fast_sequence_scores(logits))
    return np.asarray(values, dtype=np.float64)


def score_ragged_items(model, path: Path, batch_size: int) -> tuple[np.ndarray, float]:
    mmap = RaggedMmap(path)
    item_scores = np.full(len(mmap), -np.inf, dtype=np.float64)
    exposure_seconds = 0.0
    windows = []
    item_indexes = []

    def flush() -> None:
        if not windows:
            return
        scores = score_windows(model, windows, batch_size)
        for item_index, score in zip(item_indexes, scores):
            item_scores[item_index] = max(item_scores[item_index], float(score))
        windows.clear()
        item_indexes.clear()

    for index in range(len(mmap)):
        item = np.asarray(mmap[index], dtype=np.float32)
        exposure_seconds += len(item) * FEATURE_STEP_SECONDS
        for window in windows_for_item(item, all_windows=True):
            windows.append(window)
            item_indexes.append(index)
            if len(windows) >= batch_size:
                flush()
    flush()
    return item_scores, exposure_seconds


def choose_operating_point(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
    negative_exposure_seconds: float,
    *,
    min_recall: float,
    max_faph: float,
) -> dict:
    if not len(positive_scores) or not len(negative_scores):
        raise ValueError("positive and negative score sets must not be empty")
    thresholds = np.unique(np.concatenate([positive_scores, negative_scores]))
    candidates = []
    for threshold in thresholds:
        recall = float(np.mean(positive_scores >= threshold))
        false_accepts = int(np.sum(negative_scores >= threshold))
        faph = false_accepts / (negative_exposure_seconds / 3600.0)
        if recall >= min_recall and faph <= max_faph:
            candidates.append((recall, -faph, float(threshold), false_accepts))
    if not candidates:
        return {
            "qualified": False,
            "threshold": None,
            "positive_recall": float(np.max([np.mean(positive_scores >= x) for x in thresholds])),
            "minimum_faph_at_recall": None,
            "false_accepts": None,
        }
    recall, negative_faph, threshold, false_accepts = max(candidates)
    return {
        "qualified": True,
        "threshold": threshold,
        "positive_recall": recall,
        "faph": -negative_faph,
        "false_accepts": false_accepts,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--positive-source", type=parse_source, required=True)
    parser.add_argument("--negative-source", type=parse_source, action="append", required=True)
    parser.add_argument("--heldout-false-wake-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--min-recall", type=float, default=0.90)
    parser.add_argument("--max-faph", type=float, default=0.10)
    parser.add_argument("--max-heldout-false-wake-accepts", type=int, default=0)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or not 0 < args.min_recall <= 1 or args.max_faph < 0:
        parser.error("invalid qualification limits")

    heldout = np.load(args.heldout_false_wake_features, mmap_mode="r")
    if heldout.ndim != 3 or tuple(heldout.shape[1:]) != (INPUT_FRAMES, FEATURE_BINS):
        parser.error("heldout false-wake features must have shape [N, 260, 40]")

    model = build_teacher()
    model.load_weights(args.model)
    positive_scores, positive_exposure = score_ragged_items(
        model, args.positive_source[1], args.batch_size
    )
    negative_reports = []
    negative_scores = []
    negative_exposure = 0.0
    for source_id, path in args.negative_source:
        scores, exposure = score_ragged_items(model, path, args.batch_size)
        negative_scores.extend(scores)
        negative_exposure += exposure
        negative_reports.append(
            {
                "id": source_id,
                "path": str(path),
                "item_count": int(len(scores)),
                "exposure_seconds": exposure,
                "false_accepts": None,
            }
        )
    negative_scores = np.asarray(negative_scores, dtype=np.float64)
    heldout_scores = score_windows(
        model,
        [np.asarray(value, dtype=np.float32) for value in heldout],
        args.batch_size,
    )
    operating_point = choose_operating_point(
        positive_scores,
        negative_scores,
        negative_exposure,
        min_recall=args.min_recall,
        max_faph=args.max_faph,
    )
    threshold = operating_point["threshold"]
    heldout_accepts = (
        None if threshold is None else int(np.sum(heldout_scores >= threshold))
    )
    qualified = bool(
        operating_point["qualified"]
        and heldout_accepts is not None
        and heldout_accepts <= args.max_heldout_false_wake_accepts
    )
    result = {
        "schema_version": 1,
        "model": str(args.model.resolve()),
        "model_sha256": sha256_file(args.model),
        "positive_source": {
            "id": args.positive_source[0],
            "path": str(args.positive_source[1]),
            "item_count": int(len(positive_scores)),
            "exposure_seconds": positive_exposure,
        },
        "negative_sources": negative_reports,
        "negative_item_count": int(len(negative_scores)),
        "negative_exposure_seconds": negative_exposure,
        "heldout_false_wake_features": str(args.heldout_false_wake_features.resolve()),
        "heldout_false_wake_feature_count": int(len(heldout_scores)),
        "heldout_false_wake_accepts": heldout_accepts,
        "limits": {
            "min_recall": args.min_recall,
            "max_faph": args.max_faph,
            "max_heldout_false_wake_accepts": args.max_heldout_false_wake_accepts,
        },
        "operating_point": operating_point,
        "qualified": qualified,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if qualified else 2


if __name__ == "__main__":
    raise SystemExit(main())
