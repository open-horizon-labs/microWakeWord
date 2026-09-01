"""Compact phoneme-student contracts shared by training and firmware scoring.

The student emits one CTC distribution, not a second wake verifier.  A tiny
deterministic decoder compares the canonical phone path with declared collision
paths.  Repeated phones remain repeated path entries while sharing one acoustic
output, which is the normal CTC representation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from microwakeword.kizz_phoneme_teacher import WindowScore, score_window
from microwakeword.ordered_state_model import spectrogram_slices_dropped
from microwakeword.wake_phrase import KIZZ_CONTROL, WakePhraseSpec

BLANK = "<blank>"
OTHER = "OTHER"
FRONTEND_STEP_SECONDS = 0.010
FRONTEND_WINDOW_SECONDS = 0.030


def compact_phone_contract(phrase: WakePhraseSpec = KIZZ_CONTROL) -> dict:
    phones = tuple(
        dict.fromkeys(
            phrase.phones
            + tuple(phone for path in phrase.collision_phones for phone in path)
        )
    )
    tokens = (BLANK,) + phones + (OTHER,)
    ids = {token: index for index, token in enumerate(tokens)}
    return {
        "schema_version": 1,
        "phrase_id": phrase.phrase_id,
        "phrase_text": phrase.text,
        "tokens": list(tokens),
        "blank_id": ids[BLANK],
        "other_id": ids[OTHER],
        "canonical_path": [ids[phone] for phone in phrase.phones],
        "collision_paths": {
            transcript: [ids[phone] for phone in path]
            for transcript, path in zip(
                phrase.collision_transcripts, phrase.collision_phones, strict=True
            )
        },
    }


def student_output_times_seconds(flags, output_frames: int) -> np.ndarray:
    """Return the latest acoustic evidence center for each causal output.

    This is derived from the valid-convolution receptive field, frontend frame
    geometry, and model stride.  It replaces the old hand-selected distillation
    offset.
    """
    if output_frames < 1:
        raise ValueError("output_frames must be positive")
    dropped = spectrogram_slices_dropped(flags)
    first = FRONTEND_WINDOW_SECONDS / 2 + dropped * FRONTEND_STEP_SECONDS
    step = int(flags.stride) * FRONTEND_STEP_SECONDS
    return first + step * np.arange(output_frames, dtype=np.float64)


def student_stream_phase_offset_frames(flags) -> int:
    """Return the observed prefix used to prime the fixed-stride stream.

    The non-streaming model's first valid output ends at the receptive-field
    boundary.  Internal-state inference emits after each complete stride-sized
    input.  A zero-prefixed primer ending with this many observed frames makes
    the first subsequent chunk begin at the phase where both timelines
    represent identical feature windows.
    """
    stride = int(flags.stride)
    if stride < 1:
        raise ValueError("student stride must be positive")
    if getattr(flags, "causal_memory", False):
        return 0
    receptive_field_frames = spectrogram_slices_dropped(flags) + 1
    return receptive_field_frames % stride


def resample_log_posteriors(
    teacher_log_posteriors: np.ndarray,
    *,
    teacher_frame_center_seconds: float,
    teacher_frame_stride_seconds: float,
    student_times_seconds: Sequence[float],
) -> np.ndarray:
    """Linearly resample probability mass onto the causal student timeline."""
    values = np.asarray(teacher_log_posteriors, dtype=np.float64)
    targets = np.asarray(student_times_seconds, dtype=np.float64)
    if values.ndim != 2 or not len(values):
        raise ValueError("teacher posteriors must have shape [time, class]")
    if teacher_frame_stride_seconds <= 0 or np.any(~np.isfinite(targets)):
        raise ValueError("invalid teacher/student timing")
    probabilities = np.exp(values)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    source_times = (
        teacher_frame_center_seconds
        + teacher_frame_stride_seconds * np.arange(len(values), dtype=np.float64)
    )
    result = np.column_stack(
        [
            np.interp(
                targets,
                source_times,
                probabilities[:, column],
                left=probabilities[0, column],
                right=probabilities[-1, column],
            )
            for column in range(probabilities.shape[1])
        ]
    )
    result /= result.sum(axis=1, keepdims=True)
    return np.log(np.maximum(result, np.finfo(np.float32).tiny)).astype(np.float32)


def suffix_window_score(
    log_probs: np.ndarray,
    *,
    canonical_path: Sequence[int],
    collision_paths: Sequence[Sequence[int]],
    blank_id: int,
    window_lengths: Sequence[int],
    beta: float,
) -> WindowScore:
    """Return the best eligible CTC score among windows ending now."""
    values = np.asarray(log_probs, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not window_lengths:
        raise ValueError("log_probs and window_lengths must be non-empty")
    best: WindowScore | None = None
    for requested in sorted(set(int(value) for value in window_lengths)):
        if requested <= 0:
            raise ValueError("window lengths must be positive")
        length = min(requested, len(values))
        candidate = score_window(
            values[-length:],
            canonical_tokens=canonical_path,
            collision_tokens=collision_paths,
            blank_id=blank_id,
            start_frame=len(values) - length,
        )
        if candidate.collision_margin < beta:
            continue
        if best is None or (candidate.canonical_fit, candidate.collision_margin) > (
            best.canonical_fit,
            best.collision_margin,
        ):
            best = candidate
    return best or WindowScore(0, 0, -math.inf, math.inf, -math.inf)


@dataclass(frozen=True)
class PhonemeDetection:
    start_frame: int
    end_frame: int
    canonical_fit: float
    collision_margin: float


class StreamingPhonemeDecoder:
    """Bounded-memory suffix CTC detector matching the firmware contract."""

    def __init__(
        self,
        contract: Mapping,
        *,
        window_lengths: Sequence[int],
        threshold: float,
        beta: float,
        cooldown_frames: int,
    ) -> None:
        if cooldown_frames < 0:
            raise ValueError("cooldown_frames must be non-negative")
        self.contract = dict(contract)
        self.window_lengths = tuple(int(value) for value in window_lengths)
        if not self.window_lengths or min(self.window_lengths) <= 0:
            raise ValueError("window lengths must be positive")
        self.threshold = float(threshold)
        self.beta = float(beta)
        self.cooldown_frames = int(cooldown_frames)
        self.reset()

    def reset(self) -> None:
        self._frames: list[np.ndarray] = []
        self._index = 0
        self._cooldown = 0

    def step(
        self, frame: Sequence[float], *, from_logits: bool = True
    ) -> PhonemeDetection | None:
        values = np.asarray(frame, dtype=np.float64)
        if values.shape != (len(self.contract["tokens"]),):
            raise ValueError("student frame has the wrong compact vocabulary size")
        if from_logits:
            values = values - np.max(values)
            values = values - np.log(np.exp(values).sum())
        else:
            if np.any(values < 0) or not np.isfinite(values).all() or values.sum() <= 0:
                raise ValueError("probability frame is invalid")
            values = np.log(
                np.maximum(values / values.sum(), np.finfo(np.float64).tiny)
            )
        self._frames.append(values)
        maximum = max(self.window_lengths)
        if len(self._frames) > maximum:
            del self._frames[: len(self._frames) - maximum]
        current = self._index
        self._index += 1
        if self._cooldown:
            self._cooldown -= 1
            return None
        scored = suffix_window_score(
            np.stack(self._frames),
            canonical_path=self.contract["canonical_path"],
            collision_paths=tuple(self.contract["collision_paths"].values()),
            blank_id=int(self.contract["blank_id"]),
            window_lengths=self.window_lengths,
            beta=self.beta,
        )
        if scored.canonical_fit < self.threshold:
            return None
        self._cooldown = self.cooldown_frames
        buffered_start = current - len(self._frames) + 1
        return PhonemeDetection(
            start_frame=buffered_start + scored.start_frame,
            end_frame=current,
            canonical_fit=scored.canonical_fit,
            collision_margin=scored.collision_margin,
        )


__all__ = [
    "BLANK",
    "OTHER",
    "PhonemeDetection",
    "StreamingPhonemeDecoder",
    "compact_phone_contract",
    "resample_log_posteriors",
    "student_output_times_seconds",
    "suffix_window_score",
]
