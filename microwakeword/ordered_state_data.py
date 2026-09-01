# coding=utf-8
# Copyright 2026 Open Horizon Labs.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Validated alignment contract for canonical ordered-state examples.

Text is provenance, not a source of positive aliases. A positive is valid only
when its declared phone sequence is the canonical Kizz sequence. Phrase-only
spans can train the sequence/end objective; phone spans additionally enable the
auxiliary frame-state objective.
"""

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np

from microwakeword.ordered_state import KIZZ_PHONES, KIZZ_TOPOLOGY

CANONICAL_KIZZ_PHONES = KIZZ_PHONES


@dataclass(frozen=True)
class TimeSpan:
    start_s: float
    end_s: float

    def __post_init__(self):
        if (
            not math.isfinite(self.start_s)
            or not math.isfinite(self.end_s)
            or self.start_s < 0
            or self.end_s <= self.start_s
        ):
            raise ValueError("span must have non-negative start and positive duration")


@dataclass(frozen=True)
class PhoneSpan(TimeSpan):
    phone: str


@dataclass(frozen=True)
class OrderedStateExample:
    source_id: str
    truth: bool
    duration_s: float
    text: str | None = None
    phrase_span: TimeSpan | None = None
    phone_spans: tuple[PhoneSpan, ...] = ()
    expected_phones: tuple[str, ...] = CANONICAL_KIZZ_PHONES

    def __post_init__(self):
        if not self.source_id:
            raise ValueError("source_id is required")
        if not math.isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if self.phrase_span and self.phrase_span.end_s > self.duration_s:
            raise ValueError("phrase span exceeds example duration")
        if self.truth and not self.phrase_span:
            raise ValueError("positive example requires a phrase span")
        if not self.truth and (self.phrase_span or self.phone_spans):
            raise ValueError(
                "negative example cannot declare canonical phrase alignment"
            )
        if self.phone_spans:
            phones = tuple(span.phone for span in self.phone_spans)
            if phones != self.expected_phones:
                raise ValueError(
                    "positive phone sequence must exactly match canonical selected phrase"
                )
            previous_end = -1.0
            for span in self.phone_spans:
                if span.start_s + 1e-6 < previous_end:
                    raise ValueError("phone spans must be ordered and non-overlapping")
                if not self.phrase_span or (
                    span.start_s + 1e-6 < self.phrase_span.start_s
                    or span.end_s > self.phrase_span.end_s + 1e-6
                ):
                    raise ValueError("phone spans must remain inside the phrase span")
                previous_end = span.end_s


def example_from_mapping(
    record: Mapping,
    *,
    expected_phones: Sequence[str] = CANONICAL_KIZZ_PHONES,
) -> OrderedStateExample:
    """Parse and validate a JSON-compatible example record."""

    def time_span(value):
        if value is None:
            return None
        return TimeSpan(float(value["start_s"]), float(value["end_s"]))

    phone_spans = tuple(
        PhoneSpan(
            start_s=float(span["start_s"]),
            end_s=float(span["end_s"]),
            phone=str(span["phone"]),
        )
        for span in record.get("phone_spans", [])
    )
    return OrderedStateExample(
        source_id=str(record["source_id"]),
        truth=bool(record["truth"]),
        duration_s=float(record["duration_s"]),
        text=record.get("text"),
        phrase_span=time_span(record.get("phrase_span")),
        phone_spans=phone_spans,
        expected_phones=tuple(expected_phones),
    )


def frame_state_targets(
    example: OrderedStateExample,
    frame_times_s: Sequence[float],
    *,
    background_index: int = KIZZ_TOPOLOGY.background_index,
    silence_index: int = KIZZ_TOPOLOGY.silence_index,
    first_keyword_index: int = KIZZ_TOPOLOGY.first_ordered_state_index,
    states_per_phone: int = KIZZ_TOPOLOGY.states_per_phone,
) -> np.ndarray | None:
    """Create aligned frame targets, or ``None`` for weak span-only positives.

    Each aligned phone span is divided into equal beginning/middle/end states.
    Forced-alignment boundaries remain the source of truth; this function does
    not infer phones from text or invent alignments from a phrase-only span.
    """
    frame_times = np.asarray(frame_times_s, dtype=np.float64)
    if states_per_phone < 1:
        raise ValueError("states_per_phone must be positive")
    if frame_times.ndim != 1:
        raise ValueError("frame_times_s must be one-dimensional")
    if frame_times.size and (
        np.any(~np.isfinite(frame_times))
        or np.any(frame_times < 0)
        or np.any(frame_times > example.duration_s)
    ):
        raise ValueError("frame times must remain inside the example")
    rejection_index = silence_index if example.truth else background_index
    targets = np.full(frame_times.shape, rejection_index, dtype=np.int32)
    if not example.truth:
        return targets
    if not example.phone_spans:
        return None

    for phone_index, span in enumerate(example.phone_spans):
        duration = span.end_s - span.start_s
        relative = (frame_times - span.start_s) / duration
        in_phone = (relative >= 0.0) & (relative < 1.0)
        thirds = np.minimum(
            (relative * states_per_phone).astype(np.int32), states_per_phone - 1
        )
        targets[in_phone] = (
            first_keyword_index + phone_index * states_per_phone + thirds[in_phone]
        )
    return targets


def validate_examples(
    records: Iterable[Mapping],
    require_phone_alignment: bool = False,
    *,
    expected_phones: Sequence[str] = CANONICAL_KIZZ_PHONES,
) -> list[OrderedStateExample]:
    """Validate a collection and require both product error directions."""
    examples = [
        example_from_mapping(record, expected_phones=expected_phones)
        for record in records
    ]
    if not examples:
        raise ValueError("ordered-state dataset is empty")
    truths = {example.truth for example in examples}
    if truths != {False, True}:
        raise ValueError(
            "ordered-state dataset must include positive and negative data"
        )
    if require_phone_alignment and any(
        example.truth and not example.phone_spans for example in examples
    ):
        raise ValueError("frame-state supervision requires aligned positive phones")
    return examples
