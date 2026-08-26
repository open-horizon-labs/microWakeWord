"""Optional Numba accelerator for the portable suffix forward-sum decoder."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np


_KERNEL = None


def _compiled_kernel():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    try:
        from numba import njit, prange
    except ImportError as error:  # pragma: no cover - deployment tooling installs Numba
        raise RuntimeError("accelerated CTC scoring requires numba") from error

    @njit(inline="always")
    def logaddexp(left, right):
        if left == -np.inf:
            return right
        if right == -np.inf:
            return left
        if left > right:
            return left + math.log1p(math.exp(right - left))
        return right + math.log1p(math.exp(left - right))

    @njit
    def path_fit(log_probs, start, length, path, path_length, blank_id):
        state_count = 2 * path_length + 1
        previous = np.full(state_count, -np.inf, dtype=np.float64)
        current = np.empty(state_count, dtype=np.float64)
        previous[0] = log_probs[start, blank_id]
        previous[1] = log_probs[start, path[0]]
        for frame_index in range(1, length):
            for state in range(state_count):
                token = blank_id if state % 2 == 0 else path[state // 2]
                value = previous[state]
                if state > 0:
                    value = logaddexp(value, previous[state - 1])
                if state > 1 and token != blank_id:
                    previous_token = (
                        blank_id
                        if (state - 2) % 2 == 0
                        else path[(state - 2) // 2]
                    )
                    if token != previous_token:
                        value = logaddexp(value, previous[state - 2])
                current[state] = value + log_probs[start + frame_index, token]
            swap = previous
            previous = current
            current = swap
        return logaddexp(previous[state_count - 1], previous[state_count - 2]) / path_length

    @njit(parallel=True)
    def score(logits, paths, path_lengths, window_lengths, blank_id, beta):
        sequence_count, frame_count, output_count = logits.shape
        result = np.full(sequence_count, -np.inf, dtype=np.float64)
        for sequence_index in prange(sequence_count):
            log_probs = np.empty((frame_count, output_count), dtype=np.float64)
            for frame_index in range(frame_count):
                maximum = float(logits[sequence_index, frame_index, 0])
                for token in range(1, output_count):
                    maximum = max(
                        maximum, float(logits[sequence_index, frame_index, token])
                    )
                total = 0.0
                for token in range(output_count):
                    total += math.exp(
                        float(logits[sequence_index, frame_index, token]) - maximum
                    )
                normalizer = maximum + math.log(total)
                for token in range(output_count):
                    log_probs[frame_index, token] = (
                        float(logits[sequence_index, frame_index, token]) - normalizer
                    )
            best_fit = -np.inf
            best_margin = -np.inf
            for length_index in range(len(window_lengths)):
                length = min(int(window_lengths[length_index]), frame_count)
                start = frame_count - length
                canonical_fit = path_fit(
                    log_probs,
                    start,
                    length,
                    paths[0],
                    int(path_lengths[0]),
                    blank_id,
                )
                collision_fit = -np.inf
                for path_index in range(1, len(path_lengths)):
                    candidate = path_fit(
                        log_probs,
                        start,
                        length,
                        paths[path_index],
                        int(path_lengths[path_index]),
                        blank_id,
                    )
                    collision_fit = max(collision_fit, candidate)
                margin = canonical_fit - collision_fit
                if margin >= beta and (
                    canonical_fit > best_fit
                    or (canonical_fit == best_fit and margin > best_margin)
                ):
                    best_fit = canonical_fit
                    best_margin = margin
            result[sequence_index] = best_fit
        return result

    _KERNEL = score
    return _KERNEL


def suffix_forward_sum_scores(
    logits: Sequence[Sequence[Sequence[float]]],
    contract: Mapping,
    *,
    window_lengths: Sequence[int],
    beta: float,
) -> np.ndarray:
    """Score independent sequences with firmware-equivalent suffix CTC semantics."""
    values = np.ascontiguousarray(np.asarray(logits, dtype=np.float32))
    tokens = tuple(contract.get("tokens", ()))
    blank_id = int(contract.get("blank_id", -1))
    canonical = tuple(int(value) for value in contract.get("canonical_path", ()))
    collisions = tuple(
        tuple(int(value) for value in path)
        for path in contract.get("collision_paths", {}).values()
    )
    paths = (canonical, *collisions)
    lengths = np.asarray(tuple(int(value) for value in window_lengths), dtype=np.int32)
    if (
        values.ndim != 3
        or not len(values)
        or values.shape[2] != len(tokens)
        or not np.isfinite(values).all()
        or not tokens
        or not 0 <= blank_id < len(tokens)
        or not canonical
        or not collisions
        or not len(lengths)
        or np.any(lengths <= 0)
        or not math.isfinite(float(beta))
    ):
        raise ValueError("invalid accelerated suffix forward-sum inputs")
    max_path = max(len(path) for path in paths)
    padded = np.zeros((len(paths), max_path), dtype=np.int32)
    path_lengths = np.asarray([len(path) for path in paths], dtype=np.int32)
    for index, path in enumerate(paths):
        if not path or any(token < 0 or token >= len(tokens) for token in path):
            raise ValueError("path token is outside the compact vocabulary")
        padded[index, : len(path)] = path
    return np.asarray(
        _compiled_kernel()(
            values,
            padded,
            path_lengths,
            lengths,
            blank_id,
            float(beta),
        ),
        dtype=np.float64,
    )


__all__ = ["suffix_forward_sum_scores"]
