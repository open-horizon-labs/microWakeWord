#!/usr/bin/env python3
"""Cache original-resolution clip decisions from the qualified Kizz teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import tensorflow as tf

from microwakeword.kizz_phoneme_teacher import choose_validation_threshold
from microwakeword.phoneme_student import compact_phone_contract
from tools.cache_kizz_phoneme_teacher_posteriors import load_cache


SCHEMA_VERSION = 1
WINDOW_LENGTHS_FRAMES = (28, 34, 40, 48, 58, 70, 80)
HOP_FRAMES = 3


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _manifest_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    for key in ("examples", "records", "items"):
        if isinstance(payload.get(key), list):
            return [dict(row) for row in payload[key]]
    raise ValueError("manifest has no examples/records/items list")


def _path_fits(windows: np.ndarray, path: list[int], blank_id: int) -> np.ndarray:
    count, length, _ = windows.shape
    targets = tf.tile(tf.constant([path], dtype=tf.int32), [count, 1])
    losses = tf.nn.ctc_loss(
        targets,
        tf.transpose(tf.convert_to_tensor(windows), [1, 0, 2]),
        tf.fill([count], len(path)),
        tf.fill([count], length),
        logits_time_major=True,
        blank_index=int(blank_id),
    )
    return -np.asarray(losses, dtype=np.float64) / len(path)


def forward_sum_sliding_scores(
    log_probs: np.ndarray,
    contract: dict,
    *,
    window_lengths: tuple[int, ...] = WINDOW_LENGTHS_FRAMES,
    hop: int = HOP_FRAMES,
    beta: float = 0.0,
    batch_size: int = 128,
    suffix_only: bool = False,
) -> dict[str, np.ndarray]:
    """Vectorize the teacher's qualified sliding-window CTC rule."""
    values = np.asarray(log_probs, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != len(contract["tokens"]):
        raise ValueError("teacher posteriors must be [clip, frame, compact token]")
    if not np.isfinite(values).all():
        raise ValueError("teacher posteriors must be finite")
    lengths = tuple(int(value) for value in window_lengths)
    if (
        not lengths
        or any(value <= 0 for value in lengths)
        or int(hop) <= 0
        or not math.isfinite(float(beta))
        or int(batch_size) < 1
    ):
        raise ValueError("invalid forward-sum scoring geometry")
    paths = [
        list(contract["canonical_path"]),
        *(list(path) for path in contract["collision_paths"].values()),
    ]
    blank_id = int(contract["blank_id"])
    result = {
        "raw_canonical_fit": np.full(len(values), -math.inf, dtype=np.float64),
        "raw_collision_margin": np.full(len(values), -math.inf, dtype=np.float64),
        "deployment_canonical_fit": np.full(len(values), -math.inf, dtype=np.float64),
        "deployment_collision_margin": np.full(
            len(values), -math.inf, dtype=np.float64
        ),
        "raw_start_frame": np.full(len(values), -1, dtype=np.int32),
        "raw_end_frame": np.full(len(values), -1, dtype=np.int32),
        "deployment_start_frame": np.full(len(values), -1, dtype=np.int32),
        "deployment_end_frame": np.full(len(values), -1, dtype=np.int32),
    }
    for batch_start in range(0, len(values), int(batch_size)):
        batch = values[batch_start : batch_start + int(batch_size)]
        raw_fit = np.full(len(batch), -math.inf, dtype=np.float64)
        raw_margin = np.full(len(batch), -math.inf, dtype=np.float64)
        deployed_fit = np.full(len(batch), -math.inf, dtype=np.float64)
        deployed_margin = np.full(len(batch), -math.inf, dtype=np.float64)
        raw_start_frame = np.full(len(batch), -1, dtype=np.int32)
        raw_end_frame = np.full(len(batch), -1, dtype=np.int32)
        deployed_start_frame = np.full(len(batch), -1, dtype=np.int32)
        deployed_end_frame = np.full(len(batch), -1, dtype=np.int32)
        frame_count = batch.shape[1]
        for requested in lengths:
            length = min(requested, frame_count)
            tail = frame_count - length
            if suffix_only:
                starts = [tail]
            else:
                starts = list(range(0, frame_count - length + 1, int(hop)))
                if not starts or starts[-1] != tail:
                    starts.append(tail)
            windows = np.stack(
                [batch[:, start : start + length] for start in starts], axis=1
            ).reshape(-1, length, batch.shape[2])
            fits = np.stack(
                [_path_fits(windows, path, blank_id) for path in paths], axis=1
            ).reshape(len(batch), len(starts), len(paths))
            canonical = fits[:, :, 0]
            margin = canonical - np.max(fits[:, :, 1:], axis=2)

            raw_index = np.argmax(canonical, axis=1)
            candidate_raw = canonical[np.arange(len(batch)), raw_index]
            candidate_raw_margin = margin[np.arange(len(batch)), raw_index]
            improve_raw = (candidate_raw > raw_fit) | (
                (candidate_raw == raw_fit) & (candidate_raw_margin > raw_margin)
            )
            raw_fit[improve_raw] = candidate_raw[improve_raw]
            raw_margin[improve_raw] = candidate_raw_margin[improve_raw]
            start_values = np.asarray(starts, dtype=np.int32)[raw_index]
            raw_start_frame[improve_raw] = start_values[improve_raw]
            raw_end_frame[improve_raw] = start_values[improve_raw] + length

            eligible = margin >= float(beta)
            eligible_fit = np.where(eligible, canonical, -math.inf)
            deployed_index = np.argmax(eligible_fit, axis=1)
            candidate_fit = eligible_fit[np.arange(len(batch)), deployed_index]
            candidate_margin = margin[np.arange(len(batch)), deployed_index]
            improve = (candidate_fit > deployed_fit) | (
                (candidate_fit == deployed_fit) & (candidate_margin > deployed_margin)
            )
            deployed_fit[improve] = candidate_fit[improve]
            deployed_margin[improve] = candidate_margin[improve]
            deployed_start_values = np.asarray(starts, dtype=np.int32)[deployed_index]
            deployed_start_frame[improve] = deployed_start_values[improve]
            deployed_end_frame[improve] = deployed_start_values[improve] + length
        destination = slice(batch_start, batch_start + len(batch))
        result["raw_canonical_fit"][destination] = raw_fit
        result["raw_collision_margin"][destination] = raw_margin
        result["deployment_canonical_fit"][destination] = deployed_fit
        result["deployment_collision_margin"][destination] = deployed_margin
        result["raw_start_frame"][destination] = raw_start_frame
        result["raw_end_frame"][destination] = raw_end_frame
        result["deployment_start_frame"][destination] = deployed_start_frame
        result["deployment_end_frame"][destination] = deployed_end_frame
    result["eligible"] = np.isfinite(result["deployment_canonical_fit"])
    result["decision_score"] = result["raw_canonical_fit"] + np.minimum(
        result["raw_collision_margin"], 0.0
    )
    if any(
        np.any(~np.isfinite(result[key]))
        for key in (
            "raw_canonical_fit",
            "raw_collision_margin",
            "decision_score",
        )
    ):
        raise ValueError("raw teacher sequence targets must be finite")
    return result


def _split_report(rows: list[dict], scores: dict[str, np.ndarray], split: str) -> dict:
    indexes = [index for index, row in enumerate(rows) if row["split"] == split]
    positive = np.asarray(
        [
            scores["deployment_canonical_fit"][i]
            for i in indexes
            if rows[i]["label"] == 1
        ],
        dtype=np.float64,
    )
    negative_indexes = [i for i in indexes if rows[i]["label"] == 0]
    negative = np.asarray(
        [scores["deployment_canonical_fit"][i] for i in negative_indexes],
        dtype=np.float64,
    )
    exposure = sum(float(rows[i]["duration_seconds"]) for i in negative_indexes)
    point = choose_validation_threshold(
        positive,
        negative,
        negative_exposure_seconds=exposure,
        min_recall=0.90,
        max_faph=0.10,
    )
    finite_negative = negative[np.isfinite(negative)]
    ceiling = float(np.max(finite_negative)) if len(finite_negative) else -math.inf
    return {
        "positive_count": len(positive),
        "negative_count": len(negative),
        "negative_exposure_seconds": exposure,
        "eligible_positive_recall": float(np.mean(np.isfinite(positive))),
        "eligible_negative_count": int(np.isfinite(negative).sum()),
        "zero_false_accept_recall": float(np.mean(positive > ceiling)),
        "operating_point": point,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--corpus", type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--posterior-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    corpus_path = args.corpus / "corpus.json" if args.corpus else None
    corpus = json.loads(corpus_path.read_text()) if corpus_path else None
    manifest_path = corpus_path if corpus_path else args.manifest
    rows = corpus["examples"] if corpus else _manifest_rows(manifest_path)
    prefix = args.posterior_cache.with_suffix("")
    raw_cache = json.loads(prefix.with_suffix(".json").read_text())
    cache, arrays = load_cache(
        prefix,
        expected_model_revision=raw_cache["model"]["revision"],
        expected_weights_sha256=raw_cache["model"]["weights_sha256"],
    )
    expected_manifest_sha256 = (
        corpus["manifests"]["teacher"]["sha256"]
        if corpus
        else _sha256_file(manifest_path)
    )
    if cache.get("manifest_sha256") != expected_manifest_sha256:
        raise ValueError("posterior cache is not bound to the active corpus")
    offsets = arrays["offsets"]
    lengths = np.diff(offsets)
    if len(lengths) != len(rows) or not len(lengths) or np.any(lengths != lengths[0]):
        raise ValueError(
            "sequence-score cache requires equal fixed-context teacher clips"
        )
    values = np.stack(
        [
            arrays["log_posteriors"][offsets[i] : offsets[i + 1]]
            for i in range(len(rows))
        ]
    )
    contract = compact_phone_contract()
    if cache.get("vocabulary", {}).get("tokens") != contract["tokens"]:
        raise ValueError("teacher cache compact vocabulary differs")
    scores = forward_sum_sliding_scores(values, contract, batch_size=args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), **scores)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "representation": "qualified_teacher_original_resolution_clip_decisions",
        "corpus": (
            {
                "path": str(corpus_path.resolve()),
                "sha256": _sha256_file(corpus_path),
                "teacher_manifest_sha256": expected_manifest_sha256,
            }
            if corpus
            else None
        ),
        "manifest": (
            {
                "path": str(manifest_path.resolve()),
                "sha256": expected_manifest_sha256,
            }
            if not corpus
            else None
        ),
        "posterior_cache": {
            "prefix": str(prefix.resolve()),
            "json_sha256": _sha256_file(prefix.with_suffix(".json")),
            "npz_sha256": _sha256_file(prefix.with_suffix(".npz")),
            "cache_sha256": cache["cache_sha256"],
        },
        "teacher_model": cache["model"],
        "compact_phone_contract_sha256": _canonical_hash(contract),
        "scorer": {
            "algorithm": "forward_sum_ctc",
            "window_lengths_frames": list(WINDOW_LENGTHS_FRAMES),
            "hop_frames": HOP_FRAMES,
            "frame_stride_seconds": cache["timing"]["frame_stride_seconds"],
            "beta": 0.0,
            "window_selection": "filter_margin_then_max_canonical_then_margin",
            "decision_score": "raw_canonical_fit + min(raw_collision_margin, 0)",
        },
        "counts": {"examples": len(rows), "teacher_frames": int(lengths[0])},
        "split_reports": (
            {
                split: _split_report(rows, scores, split)
                for split in ("train", "validation", "test")
            }
            if corpus
            else None
        ),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
