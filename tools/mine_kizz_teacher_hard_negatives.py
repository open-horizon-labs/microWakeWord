#!/usr/bin/env python3
"""Mine diverse high-scoring training-only windows for the Kizz teacher.

The miner never reads validation, test, quarantine, or deployment-anchor data.
It takes bounded deterministic samples from each explicitly named training
archive, ranks them with one frozen 9-state teacher checkpoint, and writes a
separate cache per source.  Fine-tuning retains the original random source and
splits that source's sampling pressure with its mined cache; hard mining cannot
silently erase broad random-negative coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from microwakeword.kizz_feature_archive import (
    decode_frontend_features,
    open_feature_archive,
)
from microwakeword.kizz_teacher import FEATURE_BINS, INPUT_FRAMES, build_teacher
from microwakeword.ordered_state import KIZZ_SINGLE_STATE_TOPOLOGY

FORBIDDEN_PARTS = frozenset(
    (
        "validation",
        "testing",
        "test",
        "quarantine",
        "observations",
        "evidence",
        "false-wakes",
    )
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("source must be ID=PATH")
    source_id, raw = value.split("=", 1)
    path = Path(raw).resolve()
    if not source_id or not path.is_dir():
        raise argparse.ArgumentTypeError("source must name an existing archive")
    parts = {part.casefold() for part in path.parts}
    if parts & FORBIDDEN_PARTS:
        raise argparse.ArgumentTypeError(f"non-training source is forbidden: {path}")
    return source_id, path


def sequence_scores_from_logits(logits: np.ndarray) -> np.ndarray:
    """Vectorized 9-state Viterbi completion score for offline mining."""
    values = np.asarray(logits, dtype=np.float64)
    topology = KIZZ_SINGLE_STATE_TOPOLOGY
    if values.ndim != 3 or values.shape[-1] != topology.state_count:
        raise ValueError("logits must have shape [batch, time, 9]")
    shifted = values - np.max(values, axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    rejection = np.logaddexp(log_probs[:, :, 0], log_probs[:, :, 1])
    emissions = log_probs[:, :, 2:] - rejection[:, :, None]
    alpha = np.full((len(values), 7), -np.inf, dtype=np.float64)
    best_completion = np.full(len(values), -np.inf, dtype=np.float64)
    log_self = math.log(0.6)
    log_next = math.log(0.4)
    for time in range(values.shape[1]):
        frame = emissions[:, time, :]
        restart = np.full_like(alpha, -np.inf)
        restart[:, 0] = frame[:, 0]
        loop = alpha + log_self + frame
        advance = np.full_like(alpha, -np.inf)
        advance[:, 1:] = alpha[:, :-1] + log_next + frame[:, 1:]
        alpha = np.maximum(np.maximum(restart, loop), advance)
        best_completion = np.maximum(best_completion, alpha[:, -1])
    return best_completion


def candidate_windows(
    archive: Any,
    *,
    max_items: int,
    windows_per_item: int,
    seed: int,
) -> list[tuple[int, int]]:
    if max_items < 1 or windows_per_item < 1:
        raise ValueError("candidate limits must be positive")
    rng = np.random.default_rng(seed)
    count = min(len(archive), max_items)
    indices = np.sort(rng.choice(len(archive), size=count, replace=False))
    candidates = []
    for item_index in indices:
        item = archive[int(item_index)]
        if np.asarray(item).ndim != 2 or np.asarray(item).shape[1] != FEATURE_BINS:
            raise ValueError("training archive contains an invalid feature item")
        length = len(item)
        if length <= INPUT_FRAMES:
            starts = [0]
        else:
            maximum = length - INPUT_FRAMES
            if windows_per_item == 1:
                starts = [int(rng.integers(0, maximum + 1))]
            else:
                anchors = np.linspace(0, maximum, windows_per_item)
                jitter = max(1, maximum // (2 * windows_per_item))
                starts = [
                    min(
                        max(
                            round(anchor) + int(rng.integers(-jitter, jitter + 1)),
                            0,
                        ),
                        maximum,
                    )
                    for anchor in anchors
                ]
        candidates.extend(
            (int(item_index), int(start)) for start in sorted(set(starts))
        )
    return candidates


def window(archive: Any, item_index: int, start: int) -> np.ndarray:
    item = decode_frontend_features(archive[item_index])
    result = np.zeros((INPUT_FRAMES, FEATURE_BINS), dtype=np.float32)
    available = item[start : start + INPUT_FRAMES]
    result[: len(available)] = available
    return result


def mine(
    sources: Sequence[tuple[str, Path]],
    weights: Path,
    output: Path,
    *,
    hidden_size: int = 128,
    recurrent_layers: int = 7,
    output_frames: int = 87,
    max_items_per_source: int = 4096,
    windows_per_item: int = 2,
    top_per_source: int = 256,
    batch_size: int = 64,
    seed: int = 24103,
) -> dict[str, Any]:
    if not sources:
        raise ValueError("at least one training source is required")
    if top_per_source < 1 or batch_size < 1:
        raise ValueError("mining limits must be positive")
    if output.exists():
        raise ValueError(f"hard-negative output already exists: {output}")
    model = build_teacher(
        hidden_size=hidden_size,
        recurrent_layers=recurrent_layers,
        output_frames=output_frames,
        topology=KIZZ_SINGLE_STATE_TOPOLOGY,
    )
    model.load_weights(weights)
    output.mkdir(parents=True)
    source_reports = []
    for source_number, (source_id, path) in enumerate(sources):
        archive = open_feature_archive(path)
        candidates = candidate_windows(
            archive,
            max_items=max_items_per_source,
            windows_per_item=windows_per_item,
            seed=seed + source_number * 1009,
        )
        scores = np.empty(len(candidates), dtype=np.float64)
        for first in range(0, len(candidates), batch_size):
            batch_candidates = candidates[first : first + batch_size]
            features = np.stack(
                [window(archive, item, start) for item, start in batch_candidates]
            )
            logits = np.asarray(model(features, training=False))
            scores[first : first + len(features)] = sequence_scores_from_logits(logits)
        order = sorted(
            range(len(candidates)),
            key=lambda index: (
                -float(scores[index]),
                candidates[index][0],
                candidates[index][1],
            ),
        )[: min(top_per_source, len(candidates))]
        selected_features = np.stack(
            [window(archive, *candidates[index]) for index in order]
        ).astype(np.float32, copy=False)
        destination = output / f"hard-{source_id}.npy"
        np.save(destination, selected_features)
        selected = []
        for rank, index in enumerate(order):
            item_index, start = candidates[index]
            feature_hash = hashlib.sha256(selected_features[rank].tobytes()).hexdigest()
            selected.append(
                {
                    "rank": rank,
                    "item_index": item_index,
                    "start_feature_frame": start,
                    "score": float(scores[index]),
                    "feature_sha256": feature_hash,
                }
            )
        source_reports.append(
            {
                "source_id": source_id,
                "path": str(path),
                "archive_items": len(archive),
                "candidate_count": len(candidates),
                "selected_count": len(order),
                "selected_score_min": float(scores[order[-1]]),
                "selected_score_max": float(scores[order[0]]),
                "output": str(destination),
                "output_sha256": sha256_file(destination),
                "selected": selected,
            }
        )
    report = {
        "schema_version": 1,
        "recipe": "kizz_teacher_training_only_hard_mining_v1",
        "weights": str(weights.resolve()),
        "weights_sha256": sha256_file(weights),
        "topology_state_count": 9,
        "model": {
            "hidden_size": hidden_size,
            "recurrent_layers": recurrent_layers,
            "output_frames": output_frames,
        },
        "limits": {
            "max_items_per_source": max_items_per_source,
            "windows_per_item": windows_per_item,
            "top_per_source": top_per_source,
            "batch_size": batch_size,
        },
        "seed": seed,
        "sources": source_reports,
    }
    (output / "hard-negative-mining.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=parse_source, action="append", required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--recurrent-layers", type=int, default=7)
    parser.add_argument("--output-frames", type=int, default=87)
    parser.add_argument("--max-items-per-source", type=int, default=4096)
    parser.add_argument("--windows-per-item", type=int, default=2)
    parser.add_argument("--top-per-source", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=24103)
    args = parser.parse_args(argv)
    report = mine(
        args.source,
        args.weights,
        args.output,
        hidden_size=args.hidden_size,
        recurrent_layers=args.recurrent_layers,
        output_frames=args.output_frames,
        max_items_per_source=args.max_items_per_source,
        windows_per_item=args.windows_per_item,
        top_per_source=args.top_per_source,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                row["source_id"]: {
                    "candidates": row["candidate_count"],
                    "selected": row["selected_count"],
                    "score_min": row["selected_score_min"],
                    "score_max": row["selected_score_max"],
                }
                for row in report["sources"]
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
