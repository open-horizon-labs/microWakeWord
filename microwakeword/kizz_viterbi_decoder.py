"""Portable bounded-memory Viterbi CTC decoder for Kizz Control.

There is one neural stream: compact phoneme logits.  This module supplies the
deterministic temporal decision rule around it.  ``exhaustive_suffix_score``
is the reference implementation; ``StreamingViterbiCTCDecoder`` retains only
the largest configured suffix and applies the same recurrence at each frame.
The loops intentionally use ordinary scalar operations and fixed-size logical
state, making the algorithm straightforward to port to C++ on ESP32.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


NEGATIVE_INFINITY = -math.inf


@dataclass(frozen=True)
class ViterbiScore:
    """Best canonical/collision suffix score, normalized by token count."""

    start_frame: int
    end_frame: int
    canonical_fit: float
    collision_fit: float
    collision_margin: float

    @property
    def eligible(self) -> bool:
        return math.isfinite(self.canonical_fit)


@dataclass(frozen=True)
class ViterbiDetection:
    start_frame: int
    end_frame: int
    canonical_fit: float
    collision_margin: float


def _validate_contract(contract: Mapping) -> tuple[int, tuple[int, ...], tuple[tuple[int, ...], ...], int]:
    try:
        tokens = tuple(contract["tokens"])
        blank = int(contract["blank_id"])
        canonical = tuple(int(value) for value in contract["canonical_path"])
        collisions = tuple(
            tuple(int(value) for value in path)
            for path in contract["collision_paths"].values()
        )
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise ValueError("malformed compact phone contract") from error
    if not tokens or len(set(tokens)) != len(tokens):
        raise ValueError("compact phone contract tokens must be non-empty and unique")
    if not 0 <= blank < len(tokens) or not canonical or not collisions:
        raise ValueError("compact phone contract has invalid blank/path declarations")
    paths = (canonical,) + collisions
    if any(not path or any(value < 0 or value >= len(tokens) for value in path) for path in paths):
        raise ValueError("compact phone contract path contains an invalid token")
    if any(path == canonical for path in collisions):
        raise ValueError("canonical path must not also be a collision")
    return len(tokens), canonical, collisions, blank


def _log_softmax(frame: Sequence[float], output_count: int) -> np.ndarray:
    values = np.asarray(frame, dtype=np.float64)
    if values.shape != (output_count,):
        raise ValueError(f"student frame must have shape ({output_count},), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("student logits must be finite")
    maximum = float(np.max(values))
    total = float(np.sum(np.exp(values - maximum)))
    if not math.isfinite(total) or total <= 0:
        raise ValueError("student logits cannot be normalized")
    return values - maximum - math.log(total)


def _viterbi_ctc_log_probability(log_probs: np.ndarray, path: Sequence[int], blank_id: int) -> float:
    """Maximum CTC path log probability for one token path."""
    frames = np.asarray(log_probs, dtype=np.float64)
    if frames.ndim != 2 or not len(frames):
        raise ValueError("log_probs must be a non-empty [time, vocabulary] matrix")
    if not path:
        raise ValueError("CTC path must be non-empty")
    expanded = [blank_id]
    for token in path:
        expanded.extend((int(token), blank_id))
    state_count = len(expanded)
    scores = np.full(state_count, NEGATIVE_INFINITY, dtype=np.float64)
    scores[0] = float(frames[0, blank_id])
    scores[1] = float(frames[0, expanded[1]])
    for frame in frames[1:]:
        next_scores = np.full(state_count, NEGATIVE_INFINITY, dtype=np.float64)
        for state, token in enumerate(expanded):
            best = scores[state]
            if state > 0:
                best = max(best, scores[state - 1])
            if state > 1 and token != blank_id and token != expanded[state - 2]:
                best = max(best, scores[state - 2])
            next_scores[state] = best + frame[token]
        scores = next_scores
    return float(max(scores[-1], scores[-2]))


def _score_normalized_window(log_probs: np.ndarray, *, canonical: tuple[int, ...], collisions: tuple[tuple[int, ...], ...], blank_id: int, start_frame: int) -> ViterbiScore:
    canonical_fit = _viterbi_ctc_log_probability(log_probs, canonical, blank_id) / len(canonical)
    collision_fit = max(
        _viterbi_ctc_log_probability(log_probs, path, blank_id) / len(path)
        for path in collisions
    )
    if math.isfinite(canonical_fit) and math.isfinite(collision_fit):
        margin = canonical_fit - collision_fit
    elif math.isfinite(canonical_fit):
        margin = math.inf
    else:
        margin = NEGATIVE_INFINITY
    return ViterbiScore(
        start_frame=start_frame,
        end_frame=start_frame + len(log_probs),
        canonical_fit=canonical_fit,
        collision_fit=collision_fit,
        collision_margin=margin,
    )


def exhaustive_suffix_score(
    logits: Sequence[Sequence[float]],
    contract: Mapping,
    *,
    window_lengths: Sequence[int],
    beta: float,
) -> ViterbiScore:
    """Exhaustively score every configured suffix with max/add CTC Viterbi."""
    output_count, canonical, collisions, blank_id = _validate_contract(contract)
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or not len(values):
        raise ValueError("logits must be a non-empty [time, vocabulary] matrix")
    if values.shape[1] != output_count:
        raise ValueError("logits vocabulary differs from compact phone contract")
    lengths = tuple(int(value) for value in window_lengths)
    if not lengths or any(value <= 0 for value in lengths):
        raise ValueError("window lengths must be positive")
    if not math.isfinite(float(beta)):
        raise ValueError("collision margin beta must be finite")
    normalized = np.vstack([_log_softmax(frame, output_count) for frame in values])
    best: ViterbiScore | None = None
    for requested in sorted(set(lengths)):
        length = min(requested, len(normalized))
        candidate = _score_normalized_window(
            normalized[-length:], canonical=canonical, collisions=collisions,
            blank_id=blank_id, start_frame=len(normalized) - length,
        )
        if candidate.collision_margin < beta:
            continue
        if best is None or (candidate.canonical_fit, candidate.collision_margin) > (best.canonical_fit, best.collision_margin):
            best = candidate
    return best or ViterbiScore(0, len(normalized), NEGATIVE_INFINITY, NEGATIVE_INFINITY, NEGATIVE_INFINITY)


class StreamingViterbiCTCDecoder:
    """Bounded suffix-buffer decoder with no learned verifier."""

    def __init__(self, contract: Mapping, *, window_lengths: Sequence[int], threshold: float, beta: float, cooldown_frames: int) -> None:
        output_count, _, _, _ = _validate_contract(contract)
        lengths = tuple(int(value) for value in window_lengths)
        if not lengths or any(value <= 0 for value in lengths):
            raise ValueError("window lengths must be positive")
        if not math.isfinite(float(threshold)) or not math.isfinite(float(beta)):
            raise ValueError("threshold and collision margin beta must be finite")
        if int(cooldown_frames) < 0:
            raise ValueError("cooldown_frames must be non-negative")
        self.contract = dict(contract)
        self.output_count = output_count
        self.window_lengths = lengths
        self.maximum_buffer_frames = max(lengths)
        self.threshold = float(threshold)
        self.beta = float(beta)
        self.cooldown_frames = int(cooldown_frames)
        self.reset()

    @property
    def buffered_frames(self) -> int:
        return len(self._frames)

    @property
    def frame_index(self) -> int:
        return self._index

    def reset(self) -> None:
        self._frames: list[np.ndarray] = []
        self._index = 0
        self._cooldown = 0

    def score(self) -> ViterbiScore:
        if not self._frames:
            return ViterbiScore(0, 0, NEGATIVE_INFINITY, NEGATIVE_INFINITY, NEGATIVE_INFINITY)
        local = exhaustive_suffix_score(np.vstack(self._frames), self.contract, window_lengths=self.window_lengths, beta=self.beta)
        buffer_start = self._index - len(self._frames)
        return ViterbiScore(local.start_frame + buffer_start, local.end_frame + buffer_start, local.canonical_fit, local.collision_fit, local.collision_margin)

    def step(self, logits: Sequence[float]) -> ViterbiDetection | None:
        self._frames.append(_log_softmax(logits, self.output_count))
        if len(self._frames) > self.maximum_buffer_frames:
            del self._frames[0]
        current = self._index
        self._index += 1
        if self._cooldown:
            self._cooldown -= 1
            return None
        scored = self.score()
        if scored.canonical_fit < self.threshold:
            return None
        self._cooldown = self.cooldown_frames
        return ViterbiDetection(
            start_frame=scored.start_frame,
            end_frame=current,
            canonical_fit=scored.canonical_fit,
            collision_margin=scored.collision_margin,
        )


__all__ = ["ViterbiDetection", "ViterbiScore", "StreamingViterbiCTCDecoder", "exhaustive_suffix_score"]
