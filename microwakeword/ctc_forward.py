"""Portable forward-sum CTC scoring for clip-level teacher decisions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ForwardScore:
    start_frame: int
    end_frame: int
    canonical_fit: float
    collision_fit: float
    collision_margin: float

    @property
    def eligible(self) -> bool:
        return math.isfinite(self.canonical_fit)


def _contract_paths(
    contract: Mapping,
) -> tuple[int, tuple[int, ...], tuple[tuple[int, ...], ...], int]:
    tokens = tuple(contract.get("tokens", ()))
    blank = int(contract.get("blank_id", -1))
    canonical = tuple(int(value) for value in contract.get("canonical_path", ()))
    collisions = tuple(
        tuple(int(value) for value in path)
        for path in contract.get("collision_paths", {}).values()
    )
    paths = (canonical, *collisions)
    if (
        not tokens
        or not 0 <= blank < len(tokens)
        or not canonical
        or not collisions
        or any(
            not path
            or any(token < 0 or token >= len(tokens) for token in path)
            for path in paths
        )
    ):
        raise ValueError("malformed compact phone contract")
    return len(tokens), canonical, collisions, blank


def ctc_forward_log_probability(
    log_probs: Sequence[Sequence[float]], path: Sequence[int], blank_id: int
) -> float:
    """Sum the probabilities of every valid CTC alignment in log space."""
    frames = np.asarray(log_probs, dtype=np.float64)
    tokens = tuple(int(value) for value in path)
    if frames.ndim != 2 or not len(frames) or not tokens:
        raise ValueError("CTC inputs must contain frames and path tokens")
    if not 0 <= int(blank_id) < frames.shape[1]:
        raise ValueError("blank ID is outside the vocabulary")
    if any(value < 0 or value >= frames.shape[1] for value in tokens):
        raise ValueError("path token is outside the vocabulary")
    expanded = [int(blank_id)]
    for token in tokens:
        expanded.extend((token, int(blank_id)))
    scores = np.full(len(expanded), -math.inf, dtype=np.float64)
    scores[0] = frames[0, expanded[0]]
    scores[1] = frames[0, expanded[1]]
    for frame in frames[1:]:
        next_scores = np.full_like(scores, -math.inf)
        for state, token in enumerate(expanded):
            candidates = [scores[state]]
            if state > 0:
                candidates.append(scores[state - 1])
            if (
                state > 1
                and token != blank_id
                and token != expanded[state - 2]
            ):
                candidates.append(scores[state - 2])
            next_scores[state] = np.logaddexp.reduce(candidates) + frame[token]
        scores = next_scores
    return float(np.logaddexp(scores[-1], scores[-2]))


def _log_softmax_rows(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("logits must be a finite, non-empty [time, vocabulary] matrix")
    maximum = np.max(values, axis=1, keepdims=True)
    return values - maximum - np.log(
        np.sum(np.exp(values - maximum), axis=1, keepdims=True)
    )


def exhaustive_suffix_forward_score(
    logits: Sequence[Sequence[float]],
    contract: Mapping,
    *,
    window_lengths: Sequence[int],
    beta: float,
) -> ForwardScore:
    """Score configured suffixes with the firmware's forward-sum CTC rule."""
    output_count, canonical, collisions, blank_id = _contract_paths(contract)
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or not len(values) or values.shape[1] != output_count:
        raise ValueError("logits differ from the compact vocabulary")
    normalized = _log_softmax_rows(values)
    lengths = tuple(int(value) for value in window_lengths)
    if not lengths or any(value <= 0 for value in lengths):
        raise ValueError("window lengths must be positive")
    if not math.isfinite(float(beta)):
        raise ValueError("beta must be finite")
    best: ForwardScore | None = None
    for requested in sorted(set(lengths)):
        length = min(requested, len(normalized))
        window = normalized[-length:]
        canonical_fit = (
            ctc_forward_log_probability(window, canonical, blank_id)
            / len(canonical)
        )
        collision_fit = max(
            ctc_forward_log_probability(window, path, blank_id) / len(path)
            for path in collisions
        )
        margin = canonical_fit - collision_fit
        if margin < beta:
            continue
        candidate = ForwardScore(
            len(normalized) - length,
            len(normalized),
            canonical_fit,
            collision_fit,
            margin,
        )
        if best is None or (
            candidate.canonical_fit,
            candidate.collision_margin,
        ) > (best.canonical_fit, best.collision_margin):
            best = candidate
    return best or ForwardScore(0, len(normalized), -math.inf, -math.inf, -math.inf)


def exhaustive_sliding_forward_score(
    log_probs: Sequence[Sequence[float]],
    contract: Mapping,
    *,
    window_lengths: Sequence[int],
    hop: int,
    beta: float,
) -> ForwardScore:
    """Apply the qualified teacher's sliding-window forward-sum decision rule."""
    output_count, canonical, collisions, blank_id = _contract_paths(contract)
    values = np.asarray(log_probs, dtype=np.float64)
    if values.ndim != 2 or not len(values) or values.shape[1] != output_count:
        raise ValueError("log probabilities differ from the compact vocabulary")
    if not np.isfinite(values).all():
        raise ValueError("log probabilities must be finite")
    lengths = tuple(int(value) for value in window_lengths)
    if not lengths or any(value <= 0 for value in lengths) or int(hop) <= 0:
        raise ValueError("window lengths and hop must be positive")
    if not math.isfinite(float(beta)):
        raise ValueError("beta must be finite")
    best: ForwardScore | None = None
    for requested in lengths:
        length = min(requested, len(values))
        starts = list(range(0, len(values) - length + 1, int(hop)))
        tail = len(values) - length
        if not starts or starts[-1] != tail:
            starts.append(tail)
        for start in starts:
            window = values[start : start + length]
            canonical_fit = (
                ctc_forward_log_probability(window, canonical, blank_id)
                / len(canonical)
            )
            collision_fit = max(
                ctc_forward_log_probability(window, path, blank_id) / len(path)
                for path in collisions
            )
            margin = canonical_fit - collision_fit
            if margin < beta:
                continue
            candidate = ForwardScore(
                start,
                start + length,
                canonical_fit,
                collision_fit,
                margin,
            )
            if best is None or (
                candidate.canonical_fit,
                candidate.collision_margin,
            ) > (best.canonical_fit, best.collision_margin):
                best = candidate
    return best or ForwardScore(0, 0, -math.inf, math.inf, -math.inf)


__all__ = [
    "ForwardScore",
    "ctc_forward_log_probability",
    "exhaustive_sliding_forward_score",
    "exhaustive_suffix_forward_score",
]
