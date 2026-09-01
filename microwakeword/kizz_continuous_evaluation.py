"""Continuous-stream wake qualification primitives.

This module deliberately does not depend on the Kizz data contract.  It evaluates
the detector that would run in a stream: threshold crossings become events,
events are separated by a refractory period, and negative exposure is measured
from the supplied stream durations rather than from a window-count assumption.

The JSON/CLI representation uses ``streams`` containing ordered ``timestamps``
and ``scores``.  Positive streams may provide explicit ``opportunities``; when
they do not, the complete stream is one positive opportunity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence


MIN_NEGATIVE_EXPOSURE_HOURS = 100.0
MIN_RECALL = 0.90
MAX_FAPH_UPPER_BOUND = 0.10


@dataclass(frozen=True)
class PositiveOpportunity:
    """An interval in which one wake phrase was intentionally spoken."""

    opportunity_id: str
    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        if not self.opportunity_id.strip():
            raise ValueError("positive opportunity ID must not be empty")
        if not math.isfinite(self.start_seconds) or not math.isfinite(self.end_seconds):
            raise ValueError("positive opportunity bounds must be finite")
        if self.start_seconds < 0:
            raise ValueError("positive opportunity starts before the stream")
        if self.end_seconds < self.start_seconds:
            raise ValueError("positive opportunity ends before it starts")


@dataclass(frozen=True)
class ScoreStream:
    """One ordered stream of detector scores and its measured duration."""

    stream_id: str
    split: str
    role: str
    timestamps_seconds: tuple[float, ...]
    scores: tuple[float, ...]
    duration_seconds: float
    opportunities: tuple[PositiveOpportunity, ...] = ()
    locked_deployment_anchor: bool = False

    def __post_init__(self) -> None:
        if not self.stream_id.strip():
            raise ValueError("stream ID must not be empty")
        if not self.split.strip():
            raise ValueError(f"{self.stream_id}: split must not be empty")
        if len(self.timestamps_seconds) != len(self.scores):
            raise ValueError(
                f"{self.stream_id}: timestamps and scores differ in length"
            )
        if not math.isfinite(self.duration_seconds) or self.duration_seconds <= 0:
            raise ValueError(f"{self.stream_id}: duration must be positive")
        if any(
            not math.isfinite(value)
            for value in (*self.timestamps_seconds, *self.scores)
        ):
            raise ValueError(f"{self.stream_id}: timestamps and scores must be finite")
        if any(
            timestamp < 0 or timestamp > self.duration_seconds
            for timestamp in self.timestamps_seconds
        ):
            raise ValueError(
                f"{self.stream_id}: timestamp falls outside stream duration"
            )
        if any(
            right < left
            for left, right in zip(self.timestamps_seconds, self.timestamps_seconds[1:])
        ):
            raise ValueError(f"{self.stream_id}: timestamps must be ordered")
        if self.role not in {"positive", "negative", "anchor"}:
            raise ValueError(f"{self.stream_id}: unsupported stream role {self.role!r}")
        if self.role == "positive" and not self.opportunities:
            object.__setattr__(
                self,
                "opportunities",
                (PositiveOpportunity(self.stream_id, 0.0, self.duration_seconds),),
            )
        opportunity_ids = [item.opportunity_id for item in self.opportunities]
        if len(opportunity_ids) != len(set(opportunity_ids)):
            raise ValueError(f"{self.stream_id}: duplicate positive opportunity ID")
        if any(item.end_seconds > self.duration_seconds for item in self.opportunities):
            raise ValueError(
                f"{self.stream_id}: positive opportunity exceeds stream duration"
            )
        if self.role != "positive" and self.opportunities:
            raise ValueError(
                f"{self.stream_id}: only positive streams may declare opportunities"
            )
        if self.locked_deployment_anchor and self.role != "anchor":
            raise ValueError(
                f"{self.stream_id}: locked deployment anchors must use the anchor role"
            )


@dataclass(frozen=True)
class DetectionEvent:
    start_seconds: float
    end_seconds: float
    peak_score: float
    peak_timestamp_seconds: float


@dataclass(frozen=True)
class ThresholdProvenance:
    """Evidence that a threshold was fitted on validation, not test."""

    selection_split: str
    positive_stream_ids: tuple[str, ...]
    negative_stream_ids: tuple[str, ...]
    threshold: float
    validation_recall: float | None = None
    validation_false_accepts: int | None = None
    validation_negative_exposure_seconds: float | None = None
    validation_faph_upper_95: float | None = None

    def validate_for_test(self) -> None:
        if self.selection_split != "validation":
            raise ValueError(
                "threshold fitting on test is forbidden; selection_split must be validation"
            )
        if not self.positive_stream_ids or not self.negative_stream_ids:
            raise ValueError("threshold provenance requires validation stream IDs")
        if not math.isfinite(self.threshold):
            raise ValueError("threshold provenance must contain a finite threshold")


@dataclass(frozen=True)
class QualificationResult:
    qualified: bool
    threshold: float
    recall: float
    detected_opportunities: int
    positive_opportunities: int
    false_accepts: int
    negative_exposure_seconds: float
    false_accepts_per_hour: float
    false_accepts_per_hour_upper_95: float
    locked_anchor_false_accepts: int
    reasons: tuple[str, ...]


def detect_events(
    timestamps_seconds: Sequence[float],
    scores: Sequence[float],
    threshold: float,
    *,
    refractory_seconds: float = 1.0,
    max_event_duration_seconds: float | None = None,
) -> list[DetectionEvent]:
    """Convert ordered scores into event-level detections.

    A detection is emitted on the first threshold crossing eligible after the
    previous event's refractory period.  A continuously high run is one event;
    a later crossing after the refractory period is another event even when it
    occurs in the same recording.  ``max_event_duration_seconds`` caps only the
    reported high-run end, leaving event counting deterministic. It is not a
    substitute for a decoder-level maximum keyword-path duration.
    """

    if len(timestamps_seconds) != len(scores):
        raise ValueError("timestamps and scores must have equal length")
    if refractory_seconds < 0:
        raise ValueError("refractory_seconds must be non-negative")
    if max_event_duration_seconds is not None and max_event_duration_seconds <= 0:
        raise ValueError("max_event_duration_seconds must be positive when supplied")
    events: list[DetectionEvent] = []
    active_start: float | None = None
    active_peak = float("-inf")
    active_peak_time = 0.0
    last_event_start = float("-inf")

    def finish(end_time: float) -> None:
        nonlocal active_start, active_peak, active_peak_time
        if active_start is None:
            return
        end = max(active_start, end_time)
        if max_event_duration_seconds is not None:
            end = min(end, active_start + max_event_duration_seconds)
        events.append(DetectionEvent(active_start, end, active_peak, active_peak_time))
        active_start = None
        active_peak = float("-inf")

    for timestamp, score in zip(timestamps_seconds, scores):
        timestamp = float(timestamp)
        score = float(score)
        if active_start is not None and score < threshold:
            finish(timestamp)
        if score < threshold:
            continue
        if active_start is not None:
            if score > active_peak:
                active_peak, active_peak_time = score, timestamp
            continue
        if timestamp < last_event_start + refractory_seconds:
            continue
        active_start = timestamp
        active_peak = score
        active_peak_time = timestamp
        last_event_start = timestamp
    if timestamps_seconds:
        finish(float(timestamps_seconds[-1]))
    return events


def _opportunity_hits(
    events: Sequence[DetectionEvent], opportunities: Sequence[PositiveOpportunity]
) -> int:
    return sum(
        any(
            opportunity.start_seconds <= event.start_seconds <= opportunity.end_seconds
            for event in events
        )
        for opportunity in opportunities
    )


def poisson_upper_95(false_accepts: int, exposure_hours: float) -> float:
    """One-sided 95% Poisson upper bound for the event rate per hour."""

    if false_accepts < 0 or exposure_hours <= 0:
        raise ValueError("false_accepts must be non-negative and exposure positive")
    target = 0.05
    if false_accepts == 0:
        return -math.log(target) / exposure_hours

    # Solve P(N <= k | lambda) = .05 by bisection. Evaluate the CDF in log
    # space so high-false-accept diagnostics do not silently underflow.
    def log_cdf(lam: float) -> float:
        if lam == 0:
            return 0.0
        log_terms = [
            -lam + value * math.log(lam) - math.lgamma(value + 1)
            for value in range(false_accepts + 1)
        ]
        maximum = max(log_terms)
        return maximum + math.log(
            math.fsum(math.exp(item - maximum) for item in log_terms)
        )

    low, high = 0.0, max(1.0, false_accepts + 1.0)
    log_target = math.log(target)
    while log_cdf(high) > log_target:
        high *= 2.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if log_cdf(middle) > log_target:
            low = middle
        else:
            high = middle
    return high / exposure_hours


def _validate_streams(
    streams: Iterable[ScoreStream], expected_split: str
) -> list[ScoreStream]:
    values = list(streams)
    if not values:
        raise ValueError("at least one stream is required")
    wrong = [stream.stream_id for stream in values if stream.split != expected_split]
    if wrong:
        raise ValueError(f"streams have unexpected split {expected_split!r}: {wrong}")
    stream_ids = [stream.stream_id for stream in values]
    if len(stream_ids) != len(set(stream_ids)):
        raise ValueError("stream IDs must be unique within a qualification split")
    return values


def select_threshold(
    validation_streams: Iterable[ScoreStream],
    *,
    min_recall: float = MIN_RECALL,
    refractory_seconds: float = 1.0,
    max_event_duration_seconds: float | None = None,
) -> ThresholdProvenance:
    """Select a threshold using validation only and return frozen provenance."""

    streams = _validate_streams(validation_streams, "validation")
    positives = [stream for stream in streams if stream.role == "positive"]
    negatives = [stream for stream in streams if stream.role == "negative"]
    if not positives or not negatives:
        raise ValueError("validation needs both positive and negative streams")
    candidate_scores = sorted(
        {float(score) for stream in streams for score in stream.scores},
        reverse=True,
    )
    candidates = []
    total_opportunities = sum(len(stream.opportunities) for stream in positives)
    for threshold in candidate_scores:
        detected = sum(
            _opportunity_hits(
                detect_events(
                    stream.timestamps_seconds,
                    stream.scores,
                    threshold,
                    refractory_seconds=refractory_seconds,
                    max_event_duration_seconds=max_event_duration_seconds,
                ),
                stream.opportunities,
            )
            for stream in positives
        )
        recall = detected / total_opportunities if total_opportunities else 0.0
        if recall >= min_recall:
            false_accepts = sum(
                len(
                    detect_events(
                        stream.timestamps_seconds,
                        stream.scores,
                        threshold,
                        refractory_seconds=refractory_seconds,
                        max_event_duration_seconds=max_event_duration_seconds,
                    )
                )
                for stream in negatives
            )
            exposure_hours = (
                sum(stream.duration_seconds for stream in negatives) / 3600.0
            )
            upper = poisson_upper_95(false_accepts, exposure_hours)
            candidates.append(
                (
                    upper,
                    -recall,
                    -threshold,
                    threshold,
                    recall,
                    false_accepts,
                    exposure_hours * 3600.0,
                )
            )
    if not candidates:
        raise ValueError("no validation threshold reaches the requested recall")
    selected = min(candidates)
    threshold = selected[3]
    return ThresholdProvenance(
        selection_split="validation",
        positive_stream_ids=tuple(stream.stream_id for stream in positives),
        negative_stream_ids=tuple(stream.stream_id for stream in negatives),
        threshold=threshold,
        validation_recall=selected[4],
        validation_false_accepts=selected[5],
        validation_negative_exposure_seconds=selected[6],
        validation_faph_upper_95=selected[0],
    )


def qualify_test_streams(
    test_streams: Iterable[ScoreStream],
    provenance: ThresholdProvenance,
    *,
    refractory_seconds: float = 1.0,
    max_event_duration_seconds: float | None = None,
    min_recall: float = MIN_RECALL,
    max_faph_upper_95: float = MAX_FAPH_UPPER_BOUND,
    min_negative_exposure_hours: float = MIN_NEGATIVE_EXPOSURE_HOURS,
) -> QualificationResult:
    """Evaluate a frozen validation threshold on the held-out test streams."""

    provenance.validate_for_test()
    streams = _validate_streams(test_streams, "test")
    validation_ids = set(provenance.positive_stream_ids) | set(
        provenance.negative_stream_ids
    )
    test_ids = {stream.stream_id for stream in streams}
    overlap = validation_ids & test_ids
    if overlap:
        raise ValueError(
            f"validation threshold streams overlap test streams: {sorted(overlap)}"
        )
    positives = [stream for stream in streams if stream.role == "positive"]
    negatives = [stream for stream in streams if stream.role == "negative"]
    anchors = [stream for stream in streams if stream.role == "anchor"]
    if not positives or not negatives:
        raise ValueError("test needs both positive and negative streams")
    events_by_stream = {
        stream.stream_id: detect_events(
            stream.timestamps_seconds,
            stream.scores,
            provenance.threshold,
            refractory_seconds=refractory_seconds,
            max_event_duration_seconds=max_event_duration_seconds,
        )
        for stream in streams
    }
    total_opportunities = sum(len(stream.opportunities) for stream in positives)
    detected_opportunities = sum(
        _opportunity_hits(events_by_stream[stream.stream_id], stream.opportunities)
        for stream in positives
    )
    recall = (
        detected_opportunities / total_opportunities if total_opportunities else 0.0
    )
    false_accepts = sum(len(events_by_stream[stream.stream_id]) for stream in negatives)
    exposure_seconds = sum(stream.duration_seconds for stream in negatives)
    exposure_hours = exposure_seconds / 3600.0
    faph = false_accepts / exposure_hours if exposure_hours else float("inf")
    upper = poisson_upper_95(false_accepts, exposure_hours)
    anchor_false_accepts = sum(
        len(events_by_stream[stream.stream_id]) for stream in anchors
    )
    reasons = []
    if recall < min_recall:
        reasons.append(f"recall {recall:.4f} is below {min_recall:.4f}")
    if upper > max_faph_upper_95:
        reasons.append(f"FAPH upper bound {upper:.4f} exceeds {max_faph_upper_95:.4f}")
    if exposure_hours < min_negative_exposure_hours:
        reasons.append(
            f"negative exposure {exposure_hours:.4f}h is below {min_negative_exposure_hours:.4f}h"
        )
    if anchor_false_accepts:
        reasons.append(f"{anchor_false_accepts} locked deployment-anchor false accepts")
    return QualificationResult(
        qualified=not reasons,
        threshold=provenance.threshold,
        recall=recall,
        detected_opportunities=detected_opportunities,
        positive_opportunities=total_opportunities,
        false_accepts=false_accepts,
        negative_exposure_seconds=exposure_seconds,
        false_accepts_per_hour=faph,
        false_accepts_per_hour_upper_95=upper,
        locked_anchor_false_accepts=anchor_false_accepts,
        reasons=tuple(reasons),
    )


def stream_from_mapping(value: Mapping[str, object]) -> ScoreStream:
    """Parse the small JSON representation consumed by the CLI."""

    opportunities = tuple(
        PositiveOpportunity(
            str(item["id"]), float(item["start_seconds"]), float(item["end_seconds"])
        )
        for item in value.get("opportunities", [])  # type: ignore[union-attr]
    )
    return ScoreStream(
        stream_id=str(value["id"]),
        split=str(value["split"]),
        role=str(value["role"]),
        timestamps_seconds=tuple(float(item) for item in value["timestamps_seconds"]),  # type: ignore[index]
        scores=tuple(float(item) for item in value["scores"]),  # type: ignore[index]
        duration_seconds=float(value["duration_seconds"]),
        opportunities=opportunities,
        locked_deployment_anchor=bool(value.get("locked_deployment_anchor", False)),
    )
