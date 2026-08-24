"""Ordered-state wake-word decoding and training helpers.

The detector is deliberately a single acoustic model.  Its output is a
sequence of ordered phone states, followed by silence/background outputs; the
small decoder enforces the order without requiring a second neural network.
NumPy is kept usable without TensorFlow so the streaming decoder can be used
by tests and embedded-model evaluation tools independently of training.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Optional, Sequence

import numpy as np

try:  # TensorFlow is optional for NumPy-only evaluation and tests.
    import tensorflow as tf
except ImportError:  # pragma: no cover - exercised when TensorFlow is absent.
    tf = None


KIZZ_PHONES = ("HH", "AY", "F", "AY", "K", "IH", "Z")


@dataclass(frozen=True)
class OrderedStateTopology:
    """A left-to-right state topology for one canonical wake phrase."""

    phones: tuple[str, ...]
    states_per_phone: int = 3
    silence_name: str = "silence"
    background_name: str = "background"

    def __post_init__(self) -> None:
        if not self.phones:
            raise ValueError("topology must contain at least one phone")
        if self.states_per_phone < 1:
            raise ValueError("states_per_phone must be positive")

    @property
    def ordered_state_count(self) -> int:
        return len(self.phones) * self.states_per_phone

    @property
    def silence_index(self) -> int:
        return self.ordered_state_count

    @property
    def background_index(self) -> int:
        return self.ordered_state_count + 1

    @property
    def state_count(self) -> int:
        return self.ordered_state_count + 2

    @property
    def state_names(self) -> tuple[str, ...]:
        ordered = tuple(
            f"{phone}:{state + 1}"
            for phone in self.phones
            for state in range(self.states_per_phone)
        )
        return ordered + (self.silence_name, self.background_name)

    def phone_state_index(self, phone_index: int, state_index: int) -> int:
        if not 0 <= phone_index < len(self.phones):
            raise IndexError("phone index out of range")
        if not 0 <= state_index < self.states_per_phone:
            raise IndexError("state index out of range")
        return phone_index * self.states_per_phone + state_index


KIZZ_TOPOLOGY = OrderedStateTopology(KIZZ_PHONES)


@dataclass(frozen=True)
class OrderedStateEvent:
    """A completed ordered-state detection in stream coordinates."""

    start_frame: int
    end_frame: int
    score: float
    background_score: float


def _logsumexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -np.inf
    maximum = float(np.max(finite))
    return maximum + math.log(float(np.sum(np.exp(finite - maximum))))


def _log_probabilities(frame: Sequence[float], from_logits: bool) -> np.ndarray:
    values = np.asarray(frame, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("each frame must be one-dimensional")
    if from_logits:
        values = values - np.max(values)
        probabilities = np.exp(values)
        probabilities /= np.sum(probabilities)
    else:
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("probability frames must be finite and non-negative")
        total = float(np.sum(values))
        if total <= 0.0:
            raise ValueError("probability frame must contain positive mass")
        probabilities = values / total
    return np.log(np.maximum(probabilities, np.finfo(np.float64).tiny))


def ordered_state_sequence_score_numpy(
    state_scores: Sequence[Sequence[Sequence[float]]],
    topology: OrderedStateTopology = KIZZ_TOPOLOGY,
    *,
    from_logits: bool = True,
) -> np.ndarray:
    """Return the NumPy equivalent of :func:`ordered_state_sequence_score`."""
    values = np.asarray(state_scores)
    if values.ndim != 3:
        raise ValueError("state_scores must have shape [batch, time, state]")
    if values.shape[-1] != topology.state_count:
        raise ValueError("state_scores has the wrong state dimension")
    ordered = topology.ordered_state_count
    results = []
    for sequence in values:
        alpha = np.full(ordered, -np.inf, dtype=np.float64)
        completions = []
        background = []
        for frame in sequence:
            log_probs = _log_probabilities(frame, from_logits)
            columns = []
            for state in range(ordered):
                candidates = [log_probs[state], alpha[state] + math.log(0.6)]
                if state > 0:
                    candidates.append(alpha[state - 1] + math.log(0.4))
                columns.append(_logsumexp(np.asarray(candidates)))
            alpha = np.asarray(columns)
            completions.append(alpha[-1])
            background.append(log_probs[topology.background_index])
        results.append(_logsumexp(np.asarray(completions)) - _logsumexp(np.asarray(background)))
    return np.asarray(results)


class OrderedStateDecoder:
    """Numerically stable streaming decoder for an ordered-state topology.

    The decoder uses a log-domain forward recurrence.  A state may remain in
    place or advance exactly one state per frame; it cannot skip or reorder a
    state.  The first state may begin a new path on every frame.  A detection
    competes against the current background probability and starts an explicit
    cooldown after emission.
    """

    def __init__(
        self,
        topology: OrderedStateTopology = KIZZ_TOPOLOGY,
        *,
        completion_margin: float = 0.0,
        cooldown_frames: int = 0,
        self_loop_probability: float = 0.6,
        next_state_probability: float = 0.4,
        state_evidence_floor: float = 0.0,
        from_logits: bool = False,
    ) -> None:
        if cooldown_frames < 0:
            raise ValueError("cooldown_frames must be non-negative")
        if not 0.0 < self_loop_probability <= 1.0:
            raise ValueError("self_loop_probability must be in (0, 1]")
        if not 0.0 < next_state_probability <= 1.0:
            raise ValueError("next_state_probability must be in (0, 1]")
        self.topology = topology
        self.completion_margin = float(completion_margin)
        self.cooldown_frames = int(cooldown_frames)
        self._log_self = math.log(self_loop_probability)
        self._log_next = math.log(next_state_probability)
        self.state_evidence_floor = float(state_evidence_floor)
        self.from_logits = from_logits
        self.reset()

    def reset(self, frame_index: int = 0) -> None:
        """Discard partial progress and set the next frame coordinate."""
        self._scores = np.full(self.topology.ordered_state_count, -np.inf)
        self._starts = np.full(self.topology.ordered_state_count, -1, dtype=np.int64)
        self._frame_index = int(frame_index)
        self._cooldown_remaining = 0

    @property
    def cooldown_remaining(self) -> int:
        return self._cooldown_remaining

    @property
    def scores(self) -> np.ndarray:
        return self._scores.copy()

    def step(
        self,
        frame: Sequence[float],
        frame_index: Optional[int] = None,
    ) -> Optional[OrderedStateEvent]:
        """Consume one state-probability frame and possibly emit an event."""
        if frame_index is not None:
            if frame_index < self._frame_index:
                raise ValueError("frame_index must not move backwards")
            self._frame_index = int(frame_index)
        current_frame = self._frame_index
        log_probs = _log_probabilities(frame, self.from_logits)
        if log_probs.size != self.topology.state_count:
            raise ValueError(
                f"expected {self.topology.state_count} state scores, got {log_probs.size}"
            )

        previous = self._scores
        previous_starts = self._starts
        next_scores = np.full_like(previous, -np.inf)
        next_starts = np.full_like(previous_starts, -1)
        background_score = float(log_probs[self.topology.background_index])
        for state in range(self.topology.ordered_state_count):
            candidates = []
            starts = []
            emission_score = float(log_probs[state] - background_score)
            if emission_score <= self.state_evidence_floor:
                continue
            if state == 0:
                candidates.append(emission_score)  # only state zero may start
                starts.append(current_frame)
            if np.isfinite(previous[state]):
                candidates.append(previous[state] + self._log_self + emission_score)
                starts.append(int(previous_starts[state]))
            if state > 0 and np.isfinite(previous[state - 1]):
                candidates.append(
                    previous[state - 1] + self._log_next + emission_score
                )
                starts.append(int(previous_starts[state - 1]))
            if not candidates:
                continue
            best = int(np.argmax(candidates))
            next_scores[state] = _logsumexp(np.asarray(candidates))
            next_starts[state] = starts[best]

        self._scores = next_scores
        self._starts = next_starts
        completed_score = float(next_scores[-1])
        event = None
        if self._cooldown_remaining:
            self._cooldown_remaining -= 1
        elif completed_score >= self.completion_margin:
            event = OrderedStateEvent(
                start_frame=int(next_starts[-1]),
                end_frame=current_frame,
                score=completed_score,
                background_score=background_score,
            )
            self._cooldown_remaining = self.cooldown_frames
            self._scores.fill(-np.inf)
            self._starts.fill(-1)

        self._frame_index = current_frame + 1
        return event

    def decode(
        self,
        frames: Iterable[Sequence[float]],
        *,
        start_frame: int = 0,
    ) -> list[OrderedStateEvent]:
        """Decode a batch using precisely the same recurrence as ``step``."""
        self.reset(start_frame)
        events = []
        for frame in frames:
            event = self.step(frame)
            if event is not None:
                events.append(event)
        return events


def _require_tensorflow():
    if tf is None:  # pragma: no cover - depends on the environment.
        raise ImportError("TensorFlow is required for ordered-state training helpers")
    return tf


def ordered_state_sequence_score(
    state_scores,
    topology: OrderedStateTopology = KIZZ_TOPOLOGY,
    *,
    from_logits: bool = True,
):
    """Return a differentiable completed-phrase-vs-background logit.

    ``state_scores`` has shape ``[batch, time, topology.state_count]``.  The
    recurrence is the TensorFlow equivalent of the NumPy forward decoder: it
    permits starts at every frame, self-loops, and one-state advances.  A
    log-sum-exp over possible completion times makes the score useful for
    training phrases at arbitrary positions in a stream.
    """
    tensorflow = _require_tensorflow()
    scores = tensorflow.convert_to_tensor(state_scores)
    if scores.shape.rank != 3:
        raise ValueError("state_scores must have shape [batch, time, state]")
    if scores.shape[-1] is not None and scores.shape[-1] != topology.state_count:
        raise ValueError("state_scores has the wrong state dimension")
    if from_logits:
        log_probs = tensorflow.nn.log_softmax(scores, axis=-1)
    else:
        probabilities = scores / tensorflow.reduce_sum(scores, axis=-1, keepdims=True)
        log_probs = tensorflow.math.log(
            tensorflow.clip_by_value(probabilities, tensorflow.keras.backend.epsilon(), 1.0)
        )
    ordered = topology.ordered_state_count
    log_self = tensorflow.math.log(tensorflow.constant(0.6, dtype=log_probs.dtype))
    log_next = tensorflow.math.log(tensorflow.constant(0.4, dtype=log_probs.dtype))
    alpha = tensorflow.fill(
        [tensorflow.shape(log_probs)[0], ordered],
        tensorflow.constant(-np.inf, dtype=log_probs.dtype),
    )
    completions = []
    for time in range(log_probs.shape[1] or 0):
        emission = log_probs[:, time, :ordered]
        columns = []
        for state in range(ordered):
            candidates = [emission[:, state]]
            if state >= 0:
                candidates.append(alpha[:, state] + log_self)
            if state > 0:
                candidates.append(alpha[:, state - 1] + log_next)
            columns.append(tensorflow.reduce_logsumexp(tensorflow.stack(candidates, axis=-1), axis=-1))
        alpha = tensorflow.stack(columns, axis=1)
        completions.append(alpha[:, -1])
    if not completions:
        raise ValueError("state_scores must contain at least one time frame")
    completion = tensorflow.reduce_logsumexp(tensorflow.stack(completions, axis=1), axis=1)
    background = tensorflow.reduce_logsumexp(log_probs[:, :, topology.background_index], axis=1)
    return completion - background


def ordered_state_sequence_loss(
    state_scores,
    labels,
    topology: OrderedStateTopology = KIZZ_TOPOLOGY,
    *,
    frame_state_targets=None,
    sequence_weight: float = 1.0,
    frame_weight: float = 0.0,
    from_logits: bool = True,
):
    """Combine end-metric classification with optional aligned state loss."""
    tensorflow = _require_tensorflow()
    sequence_logits = ordered_state_sequence_score(
        state_scores, topology, from_logits=from_logits
    )
    labels = tensorflow.cast(tensorflow.reshape(labels, [-1]), sequence_logits.dtype)
    sequence_loss = tensorflow.reduce_mean(
        tensorflow.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=sequence_logits)
    )
    total = tensorflow.cast(sequence_weight, sequence_loss.dtype) * sequence_loss
    if frame_state_targets is not None and frame_weight:
        targets = tensorflow.cast(frame_state_targets, tensorflow.int32)
        frame_loss = tensorflow.reduce_mean(
            tensorflow.keras.losses.sparse_categorical_crossentropy(
                targets, state_scores, from_logits=from_logits
            )
        )
        total += tensorflow.cast(frame_weight, total.dtype) * frame_loss
    return total


__all__ = [
    "KIZZ_PHONES",
    "KIZZ_TOPOLOGY",
    "OrderedStateDecoder",
    "OrderedStateEvent",
    "OrderedStateTopology",
    "ordered_state_sequence_score_numpy",
    "ordered_state_sequence_loss",
    "ordered_state_sequence_score",
]
