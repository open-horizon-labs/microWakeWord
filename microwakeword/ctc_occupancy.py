"""Sequence-conditioned CTC occupation targets for knowledge distillation.

Unlike raw frame posterior matching, these targets condition the teacher on a
declared token sequence and marginalize every valid CTC alignment.  The result
at each frame is a probability distribution over output tokens that sums to
one and can be used as a soft target for a smaller CTC student.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def _logsumexp(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return -math.inf
    maximum = max(finite)
    return maximum + math.log(sum(math.exp(value - maximum) for value in finite))


def _expanded_path(path: Sequence[int], blank_id: int) -> np.ndarray:
    tokens = tuple(int(value) for value in path)
    if not tokens:
        raise ValueError("CTC path must not be empty")
    expanded = [int(blank_id)]
    for token in tokens:
        expanded.extend((token, int(blank_id)))
    return np.asarray(expanded, dtype=np.int32)


def _can_skip(expanded: np.ndarray, state: int, blank_id: int) -> bool:
    return (
        state >= 2
        and int(expanded[state]) != int(blank_id)
        and int(expanded[state]) != int(expanded[state - 2])
    )


def ctc_state_occupation_log_probs(
    log_probs: np.ndarray,
    path: Sequence[int],
    blank_id: int,
) -> np.ndarray:
    """Return ``log p(token_t | input, path)`` for every input frame.

    ``log_probs`` must already be normalized over the vocabulary.  The
    forward/backward recursion uses the standard CTC transition topology,
    including the repeated-token skip restriction.
    """

    values = np.asarray(log_probs, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not values.shape[1]:
        raise ValueError("log_probs must be a non-empty [time, vocabulary] matrix")
    if not np.isfinite(values).all():
        raise ValueError("log_probs must be finite")
    if not 0 <= int(blank_id) < values.shape[1]:
        raise ValueError("blank ID is outside the vocabulary")
    tokens = tuple(int(value) for value in path)
    if any(value < 0 or value >= values.shape[1] for value in tokens):
        raise ValueError("path token is outside the vocabulary")

    expanded = _expanded_path(tokens, int(blank_id))
    frame_count = len(values)
    state_count = len(expanded)
    if frame_count < len(tokens):
        raise ValueError("not enough frames for the CTC path")

    alpha = np.full((frame_count, state_count), -math.inf, dtype=np.float64)
    alpha[0, 0] = values[0, expanded[0]]
    if state_count > 1:
        alpha[0, 1] = values[0, expanded[1]]
    for frame in range(1, frame_count):
        for state, token in enumerate(expanded):
            predecessors = [alpha[frame - 1, state]]
            if state > 0:
                predecessors.append(alpha[frame - 1, state - 1])
            if _can_skip(expanded, state, int(blank_id)):
                predecessors.append(alpha[frame - 1, state - 2])
            alpha[frame, state] = _logsumexp(predecessors) + values[frame, token]

    log_probability = _logsumexp((alpha[-1, -1], alpha[-1, -2]))
    if not math.isfinite(log_probability):
        raise ValueError("CTC path has zero probability")

    beta = np.full_like(alpha, -math.inf)
    beta[-1, -1] = 0.0
    beta[-1, -2] = 0.0
    for frame in range(frame_count - 2, -1, -1):
        for state in range(state_count):
            successors = [values[frame + 1, expanded[state]] + beta[frame + 1, state]]
            if state + 1 < state_count:
                successors.append(
                    values[frame + 1, expanded[state + 1]] + beta[frame + 1, state + 1]
                )
            if state + 2 < state_count and _can_skip(
                expanded, state + 2, int(blank_id)
            ):
                successors.append(
                    values[frame + 1, expanded[state + 2]] + beta[frame + 1, state + 2]
                )
            beta[frame, state] = _logsumexp(successors)

    state_log_posterior = alpha + beta - log_probability
    token_log_posterior = np.full_like(values, -math.inf)
    for token in range(values.shape[1]):
        states = np.flatnonzero(expanded == token)
        if len(states):
            for frame in range(frame_count):
                token_log_posterior[frame, token] = _logsumexp(
                    state_log_posterior[frame, states]
                )
    row_norm = np.asarray(
        [_logsumexp(row) for row in token_log_posterior], dtype=np.float64
    )
    token_log_posterior -= row_norm[:, None]
    return token_log_posterior.astype(np.float32)


__all__ = ["ctc_state_occupation_log_probs"]
