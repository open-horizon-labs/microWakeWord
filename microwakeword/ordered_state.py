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


KIZZ_PHONES = ("h", "aɪ", "f", "aɪ", "k", "ɪ", "z")


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
    def background_index(self) -> int:
        return 0

    @property
    def silence_index(self) -> int:
        return 1

    @property
    def first_ordered_state_index(self) -> int:
        return 2

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
        return (self.background_name, self.silence_name) + ordered

    def phone_state_index(self, phone_index: int, state_index: int) -> int:
        if not 0 <= phone_index < len(self.phones):
            raise IndexError("phone index out of range")
        if not 0 <= state_index < self.states_per_phone:
            raise IndexError("state index out of range")
        return (
            self.first_ordered_state_index
            + phone_index * self.states_per_phone
            + state_index
        )

    def ordered_state_index(self, state_index: int) -> int:
        if not 0 <= state_index < self.ordered_state_count:
            raise IndexError("ordered state index out of range")
        return self.first_ordered_state_index + state_index


KIZZ_TOPOLOGY = OrderedStateTopology(KIZZ_PHONES)


@dataclass(frozen=True)
class OrderedStateEvent:
    """A completed ordered-state detection in stream coordinates."""

    start_frame: int
    end_frame: int
    score: float
    rejection_score: float


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
    state_evidence_floor: float | None = None,
    self_loop_probability: float = 0.6,
    next_state_probability: float = 0.4,
) -> np.ndarray:
    """Return the NumPy equivalent of :func:`ordered_state_sequence_score`."""
    values = np.asarray(state_scores)
    if values.ndim != 3:
        raise ValueError("state_scores must have shape [batch, time, state]")
    if values.shape[-1] != topology.state_count:
        raise ValueError("state_scores has the wrong state dimension")
    if not 0.0 < self_loop_probability <= 1.0:
        raise ValueError("self_loop_probability must be in (0, 1]")
    if not 0.0 < next_state_probability <= 1.0:
        raise ValueError("next_state_probability must be in (0, 1]")
    log_self = math.log(self_loop_probability)
    log_next = math.log(next_state_probability)
    ordered = topology.ordered_state_count
    results = []
    for sequence in values:
        alpha = np.full(ordered, -np.inf, dtype=np.float64)
        completions = []
        for frame in sequence:
            log_probs = _log_probabilities(frame, from_logits)
            rejection = np.logaddexp(
                log_probs[topology.background_index],
                log_probs[topology.silence_index],
            )
            columns = []
            for state in range(ordered):
                emission = log_probs[topology.ordered_state_index(state)] - rejection
                if (
                    state_evidence_floor is not None
                    and emission <= state_evidence_floor
                ):
                    emission = -np.inf
                candidates = []
                if state == 0:
                    candidates.append(emission)
                if np.isfinite(alpha[state]):
                    candidates.append(alpha[state] + log_self + emission)
                if state > 0:
                    candidates.append(alpha[state - 1] + log_next + emission)
                columns.append(max(candidates, default=-np.inf))
            alpha = np.asarray(columns)
            completions.append(alpha[-1])
        results.append(max(completions, default=-np.inf))
    return np.asarray(results)


class OrderedStateDecoder:
    """Numerically stable streaming decoder for an ordered-state topology.

    The decoder uses a log-domain Viterbi recurrence.  A state may remain in
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
        state_evidence_floor: float | None = None,
        from_logits: bool = True,
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
        self.state_evidence_floor = (
            None if state_evidence_floor is None else float(state_evidence_floor)
        )
        self.from_logits = from_logits
        self.reset()

    def reset(self, frame_index: int = 0) -> None:
        """Discard partial progress and set the next frame coordinate."""
        self._scores = np.full(self.topology.ordered_state_count, -np.inf)
        self._starts = np.full(self.topology.ordered_state_count, -1, dtype=np.int64)
        self._frame_index = int(frame_index)
        self._cooldown_remaining = 0

    def rearm(self) -> None:
        """Discard partial progress while preserving the stream coordinate."""
        frame_index = self._frame_index
        self.reset(frame_index)

    @property
    def cooldown_remaining(self) -> int:
        return self._cooldown_remaining

    @property
    def scores(self) -> np.ndarray:
        return self._scores.copy()

    @property
    def current_completion_score(self) -> float:
        return float(self._scores[-1])

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
        rejection_score = float(
            np.logaddexp(
                log_probs[self.topology.background_index],
                log_probs[self.topology.silence_index],
            )
        )
        if self._cooldown_remaining:
            self._cooldown_remaining -= 1
            self._scores.fill(-np.inf)
            self._starts.fill(-1)
            self._frame_index = current_frame + 1
            return None
        for state in range(self.topology.ordered_state_count):
            candidates = []
            starts = []
            emission_score = float(
                log_probs[self.topology.ordered_state_index(state)] - rejection_score
            )
            if (
                self.state_evidence_floor is not None
                and emission_score <= self.state_evidence_floor
            ):
                continue
            if state == 0:
                candidates.append(emission_score)  # only state zero may start
                starts.append(current_frame)
            if np.isfinite(previous[state]):
                candidates.append(previous[state] + self._log_self + emission_score)
                starts.append(int(previous_starts[state]))
            if state > 0 and np.isfinite(previous[state - 1]):
                candidates.append(previous[state - 1] + self._log_next + emission_score)
                starts.append(int(previous_starts[state - 1]))
            if not candidates:
                continue
            best = int(np.argmax(candidates))
            next_scores[state] = candidates[best]
            next_starts[state] = starts[best]

        self._scores = next_scores
        self._starts = next_starts
        completed_score = float(next_scores[-1])
        event = None
        if completed_score >= self.completion_margin:
            event = OrderedStateEvent(
                start_frame=int(next_starts[-1]),
                end_frame=current_frame,
                score=completed_score,
                rejection_score=rejection_score,
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


class OrderedStateMarginSweepDecoder:
    """Evaluate several completion margins in one ordered-state recurrence.

    This is an evaluation-only companion to :class:`OrderedStateDecoder`.
    The acoustic frame is normalized once, then the Viterbi score and start
    arrays are updated for every margin in parallel.  Each margin remains an
    independent decoder: it has its own cooldown, path reset, and events.
    """

    def __init__(
        self,
        topology: OrderedStateTopology = KIZZ_TOPOLOGY,
        *,
        completion_margins: Sequence[float],
        cooldown_frames: int = 0,
        self_loop_probability: float = 0.6,
        next_state_probability: float = 0.4,
        state_evidence_floor: float | None = None,
        from_logits: bool = True,
    ) -> None:
        try:
            margins = tuple(completion_margins)
        except TypeError as error:
            raise ValueError(
                "completion_margins must be a non-empty sequence"
            ) from error
        if not margins:
            raise ValueError("completion_margins must be a non-empty sequence")
        try:
            margins_array = np.asarray(margins, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "completion_margins must contain finite numbers"
            ) from error
        if margins_array.ndim != 1 or np.any(~np.isfinite(margins_array)):
            raise ValueError("completion_margins must contain finite numbers")
        if cooldown_frames < 0:
            raise ValueError("cooldown_frames must be non-negative")
        if not 0.0 < self_loop_probability <= 1.0:
            raise ValueError("self_loop_probability must be in (0, 1]")
        if not 0.0 < next_state_probability <= 1.0:
            raise ValueError("next_state_probability must be in (0, 1]")
        self.completion_margins = margins_array.copy()
        self.topology = topology
        self.cooldown_frames = int(cooldown_frames)
        self._log_self = math.log(self_loop_probability)
        self._log_next = math.log(next_state_probability)
        self.state_evidence_floor = (
            None if state_evidence_floor is None else float(state_evidence_floor)
        )
        self.from_logits = from_logits
        self.reset()

    def reset(self, frame_index: int = 0) -> None:
        """Discard all partial paths and set the next frame coordinate."""
        shape = (self.completion_margins.size, self.topology.ordered_state_count)
        self._scores = np.full(shape, -np.inf)
        self._starts = np.full(shape, -1, dtype=np.int64)
        self._frame_index = int(frame_index)
        self._cooldown_remaining = np.zeros(
            self.completion_margins.size, dtype=np.int64
        )

    def rearm(self) -> None:
        """Discard all partial paths while preserving the stream coordinate."""
        frame_index = self._frame_index
        self.reset(frame_index)

    @property
    def cooldown_remaining(self) -> np.ndarray:
        return self._cooldown_remaining.copy()

    @property
    def scores(self) -> np.ndarray:
        return self._scores.copy()

    @property
    def current_completion_scores(self) -> np.ndarray:
        return self._scores[:, -1].copy()

    def step(
        self,
        frame: Sequence[float],
        frame_index: Optional[int] = None,
    ) -> list[Optional[OrderedStateEvent]]:
        """Consume one frame and return one event slot for each margin."""
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
        rejection_score = float(
            np.logaddexp(
                log_probs[self.topology.background_index],
                log_probs[self.topology.silence_index],
            )
        )
        events: list[Optional[OrderedStateEvent]] = [
            None
        ] * self.completion_margins.size

        cooling = self._cooldown_remaining > 0
        if np.any(cooling):
            self._cooldown_remaining[cooling] -= 1
            self._scores[cooling] = -np.inf
            self._starts[cooling] = -1

        active = ~cooling
        if np.any(active):
            emissions = np.asarray(
                [
                    log_probs[self.topology.ordered_state_index(state)]
                    - rejection_score
                    for state in range(self.topology.ordered_state_count)
                ]
            )
            if self.state_evidence_floor is not None:
                emissions[emissions <= self.state_evidence_floor] = -np.inf

            previous = self._scores[active]
            previous_starts = self._starts[active]
            ordered = self.topology.ordered_state_count
            restart = np.full_like(previous, -np.inf)
            restart[:, 0] = emissions[0]
            restart_starts = np.full_like(previous_starts, -1)
            restart_starts[:, 0] = current_frame
            loop = previous + self._log_self + emissions[None, :]
            loop_starts = previous_starts
            advance = np.full_like(previous, -np.inf)
            advance[:, 1:] = previous[:, :-1] + self._log_next
            advance += emissions[None, :]
            advance_starts = np.full_like(previous_starts, -1)
            advance_starts[:, 1:] = previous_starts[:, :-1]
            candidates = np.stack([restart, loop, advance], axis=-1)
            starts = np.stack([restart_starts, loop_starts, advance_starts], axis=-1)
            best = np.argmax(candidates, axis=-1)
            rows = np.arange(previous.shape[0])[:, None]
            states = np.arange(ordered)[None, :]
            next_scores = candidates[rows, states, best]
            next_starts = starts[rows, states, best]
            self._scores[active] = next_scores
            self._starts[active] = next_starts

            completed = active & (self._scores[:, -1] >= self.completion_margins)
            for margin_index in np.flatnonzero(completed):
                events[margin_index] = OrderedStateEvent(
                    start_frame=int(self._starts[margin_index, -1]),
                    end_frame=current_frame,
                    score=float(self._scores[margin_index, -1]),
                    rejection_score=rejection_score,
                )
            self._cooldown_remaining[completed] = self.cooldown_frames
            self._scores[completed] = -np.inf
            self._starts[completed] = -1

        self._frame_index = current_frame + 1
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
    state_evidence_floor: float | None = None,
    self_loop_probability: float = 0.6,
    next_state_probability: float = 0.4,
):
    """Return a differentiable completed-phrase-vs-background logit.

    ``state_scores`` has shape ``[batch, time, topology.state_count]``.  The
    recurrence is the TensorFlow equivalent of the NumPy Viterbi decoder: it
    permits starts at every frame, self-loops, and one-state advances.  A
    maximum over possible completion times matches the streaming decision while
    allowing phrases at arbitrary positions in a stream.
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
            tensorflow.clip_by_value(
                probabilities, tensorflow.keras.backend.epsilon(), 1.0
            )
        )
    ordered = topology.ordered_state_count
    if not 0.0 < self_loop_probability <= 1.0:
        raise ValueError("self_loop_probability must be in (0, 1]")
    if not 0.0 < next_state_probability <= 1.0:
        raise ValueError("next_state_probability must be in (0, 1]")
    log_self = tensorflow.math.log(
        tensorflow.constant(self_loop_probability, dtype=log_probs.dtype)
    )
    log_next = tensorflow.math.log(
        tensorflow.constant(next_state_probability, dtype=log_probs.dtype)
    )
    alpha = tensorflow.fill(
        [tensorflow.shape(log_probs)[0], ordered],
        tensorflow.constant(-np.inf, dtype=log_probs.dtype),
    )
    completions = []
    time_steps = log_probs.shape[1]
    if time_steps is None:
        raise ValueError("state_scores must have a statically known time dimension")
    for time in range(time_steps):
        rejection = tensorflow.reduce_logsumexp(
            tensorflow.gather(
                log_probs[:, time, :],
                [topology.background_index, topology.silence_index],
                axis=1,
            ),
            axis=1,
        )
        emission = (
            log_probs[
                :,
                time,
                topology.first_ordered_state_index : topology.state_count,
            ]
            - rejection[:, None]
        )
        if state_evidence_floor is not None:
            emission = tensorflow.where(
                emission > state_evidence_floor,
                emission,
                tensorflow.constant(-np.inf, dtype=emission.dtype),
            )
        negative_infinity = tensorflow.constant(-np.inf, dtype=emission.dtype)
        advance = tensorflow.concat(
            [
                tensorflow.fill([tensorflow.shape(alpha)[0], 1], negative_infinity),
                alpha[:, :-1] + log_next,
            ],
            axis=1,
        )
        restart = tensorflow.concat(
            [
                tensorflow.zeros([tensorflow.shape(alpha)[0], 1], emission.dtype),
                tensorflow.fill(
                    [tensorflow.shape(alpha)[0], ordered - 1], negative_infinity
                ),
            ],
            axis=1,
        )
        predecessor = tensorflow.reduce_max(
            tensorflow.stack([alpha + log_self, advance, restart], axis=-1),
            axis=-1,
        )
        alpha = predecessor + emission
        completions.append(alpha[:, -1])
    if not completions:
        raise ValueError("state_scores must contain at least one time frame")
    completion = tensorflow.reduce_max(tensorflow.stack(completions, axis=1), axis=1)
    return completion


def ordered_state_sequence_loss(
    state_scores,
    labels,
    topology: OrderedStateTopology = KIZZ_TOPOLOGY,
    *,
    frame_state_targets=None,
    sequence_weight: float = 1.0,
    frame_weight: float = 0.0,
    from_logits: bool = True,
    state_evidence_floor: float | None = None,
    self_loop_probability: float = 0.6,
    next_state_probability: float = 0.4,
):
    """Combine end-metric classification with optional aligned state loss."""
    tensorflow = _require_tensorflow()
    sequence_logits = ordered_state_sequence_score(
        state_scores,
        topology,
        from_logits=from_logits,
        state_evidence_floor=state_evidence_floor,
        self_loop_probability=self_loop_probability,
        next_state_probability=next_state_probability,
    )
    labels = tensorflow.cast(tensorflow.reshape(labels, [-1]), sequence_logits.dtype)
    sequence_loss = tensorflow.reduce_mean(
        tensorflow.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=sequence_logits
        )
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
    "OrderedStateMarginSweepDecoder",
    "OrderedStateEvent",
    "OrderedStateTopology",
    "ordered_state_sequence_score_numpy",
    "ordered_state_sequence_loss",
    "ordered_state_sequence_score",
]
