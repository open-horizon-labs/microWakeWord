#!/usr/bin/env python3
"""Evaluate an ordered-state wake detector on continuous score streams.

The decoder is deliberately supplied by the caller.  It must expose
``step(frame, frame_index=...)`` or ``update(score, timestamp)`` and may expose
``rearm()``.  The call returns either a false-y value or a truthy value; a
mapping may provide event fields such as ``timestamp`` and ``score``.  The
evaluator resets a decoder only at source boundaries and calls ``rearm`` after
cooldown, preserving streaming state within each source.

Input JSON is either a list of records, ``{"records": [...]}``, or
``{"sources": [{"id": ..., "records": [...], "label": ...}, ...]}``.
JSON Lines is also accepted.  A record has ``timestamp`` and ``score`` or
``logit``.  Timestamps are seconds from the beginning of their source.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from microwakeword.ordered_state import KIZZ_TOPOLOGY


@dataclass(frozen=True)
class ScoreRecord:
    timestamp: float
    score: Any
    logit: Any = None
    index: int = 0


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _record(raw: Mapping[str, Any], index: int) -> ScoreRecord:
    if "timestamp" not in raw:
        raise ValueError(f"score record {index} is missing timestamp")
    if "score" in raw:
        score = _numeric_payload(raw["score"])
        logit = _numeric_payload(raw["logit"]) if "logit" in raw else None
    elif "logit" in raw:
        logit = _numeric_payload(raw["logit"])
        score = _sigmoid(logit) if isinstance(logit, float) else logit
    else:
        raise ValueError(f"score record {index} needs score or logit")
    timestamp = float(raw["timestamp"])
    if not math.isfinite(timestamp) or not _finite_payload(score):
        raise ValueError(f"score record {index} contains a non-finite value")
    return ScoreRecord(timestamp, score, logit, index)


def _numeric_payload(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_numeric_payload(item) for item in value]
    return float(value)


def _finite_payload(value: Any) -> bool:
    if isinstance(value, list):
        return all(_finite_payload(item) for item in value)
    return math.isfinite(value)


def _scalar_values(value: Any) -> list[float]:
    if isinstance(value, list):
        result: list[float] = []
        for item in value:
            result.extend(_scalar_values(item))
        return result
    return [float(value)]


def load_sources(path: Path) -> list[dict[str, Any]]:
    """Load deterministic score sources from JSON or JSON Lines."""
    text = path.read_text()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]

    if isinstance(payload, list):
        return [{"id": path.stem, "records": payload}]
    if not isinstance(payload, Mapping):
        raise ValueError("input must be a record list or source object")
    if "timestamp" in payload and ("score" in payload or "logit" in payload):
        return [{"id": path.stem, "records": [payload]}]
    if "records" in payload:
        return [dict(payload)]
    sources = payload.get("sources")
    if not isinstance(sources, list):
        raise ValueError("input object needs records or sources")
    return [dict(source) for source in sources]


def _reset(decoder: Any) -> None:
    reset = getattr(decoder, "reset", None)
    if reset is not None:
        reset()


def _update(decoder: Any, record: ScoreRecord) -> Any:
    step = getattr(decoder, "step", None)
    if step is not None:
        return step(record.score, frame_index=record.index)
    update = getattr(decoder, "update", None)
    if update is None:
        raise TypeError("decoder must provide update(score, timestamp)")
    return update(record.score, record.timestamp)


def _event_from_result(result: Any, record: ScoreRecord) -> dict[str, Any] | None:
    if not result:
        return None
    event: dict[str, Any] = {}
    if isinstance(result, Mapping):
        event.update(result)
    elif hasattr(result, "__dict__"):
        event.update(vars(result))
    else:
        event["result"] = result
    event = {key: _json_value(value) for key, value in event.items()}
    event.setdefault("timestamp", record.timestamp)
    event.setdefault("score", record.score)
    event.setdefault("index", record.index)
    return event


def _add_frame_coordinates(
    event: dict[str, Any], records: Sequence[ScoreRecord]
) -> dict[str, Any]:
    for frame_key, timestamp_key in (
        ("start_frame", "start_timestamp"),
        ("end_frame", "end_timestamp"),
    ):
        frame = event.get(frame_key)
        if isinstance(frame, int) and 0 <= frame < len(records):
            event.setdefault(timestamp_key, records[frame].timestamp)
    return event


def _json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        return _json_value(value.item())
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "median": None,
            "p95": None,
        }
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)]
    return {
        "count": len(values),
        "minimum": min(values),
        "maximum": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": p95,
    }


def poisson_upper_bound_95(events: int, exposure_seconds: float) -> float | None:
    """Return the one-sided 95% Poisson upper rate bound per hour."""
    if exposure_seconds < 0:
        raise ValueError("exposure_seconds must be non-negative")
    if exposure_seconds == 0:
        return None
    if events < 0:
        raise ValueError("events must be non-negative")
    hours = exposure_seconds / 3600.0
    if events == 0:
        return -math.log(0.05) / hours
    try:
        from scipy.stats import chi2
    except ImportError as error:  # pragma: no cover - project normally has scipy
        raise RuntimeError("scipy is required for non-zero Poisson bounds") from error
    return float(0.5 * chi2.ppf(0.95, 2 * (events + 1)) / hours)


def _frame_step(source: Mapping[str, Any], default: float | None) -> float | None:
    value = source.get("frame_step_seconds", default)
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("frame_step_seconds must be finite and positive")
    return value


def _exposure(
    source: Mapping[str, Any],
    records: Sequence[ScoreRecord],
    frame_step_seconds: float | None,
) -> float:
    for key in ("exposure_seconds", "duration_seconds"):
        if key in source:
            value = float(source[key])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and non-negative")
            return value
    if not records:
        return 0.0
    if len(records) < 2:
        return frame_step_seconds or 0.0
    span = max(0.0, records[-1].timestamp - records[0].timestamp)
    return span + (frame_step_seconds or 0.0)


def _has_declared_exposure(source: Mapping[str, Any]) -> bool:
    return "exposure_seconds" in source or "duration_seconds" in source


def evaluate_sources(
    sources: Iterable[Mapping[str, Any]],
    decoder_factory: Callable[[], Any],
    cooldown_seconds: float = 0.0,
    require_declared_exposure: bool = False,
    frame_step_seconds: float | None = None,
) -> dict[str, Any]:
    """Evaluate sources with one fresh decoder per continuous source."""
    if cooldown_seconds < 0:
        raise ValueError("cooldown_seconds must be non-negative")
    source_reports: list[dict[str, Any]] = []
    all_scores: list[float] = []
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    total_exposure = 0.0
    total_events = 0
    false_accepts = 0
    negative_exposure = 0.0
    positive_sources = 0
    detected_positive_sources = 0

    for source_number, source in enumerate(sources):
        if require_declared_exposure and not _has_declared_exposure(source):
            raise ValueError(
                f"source {source_number} needs exposure_seconds or duration_seconds"
            )
        raw_records = source.get("records", [])
        records = [_record(raw, index) for index, raw in enumerate(raw_records)]
        if not records and (
            require_declared_exposure
            or float(source.get("exposure_seconds", source.get("duration_seconds", 0)))
            > 0
        ):
            raise ValueError(
                f"source {source_number} has exposure but no score records"
            )
        if any(b.timestamp < a.timestamp for a, b in zip(records, records[1:])):
            raise ValueError(
                f"source {source_number} records are not timestamp ordered"
            )
        source_frame_step = _frame_step(source, frame_step_seconds)
        if source_frame_step is not None:
            for previous, current in zip(records, records[1:]):
                actual = current.timestamp - previous.timestamp
                if not math.isclose(
                    actual,
                    source_frame_step,
                    rel_tol=0.0,
                    abs_tol=max(1e-6, source_frame_step * 1e-4),
                ):
                    raise ValueError(
                        f"source {source_number} frame timestamps do not match "
                        "frame_step_seconds"
                    )
        decoder = decoder_factory()
        if cooldown_seconds and getattr(decoder, "cooldown_frames", 0):
            raise ValueError("configure cooldown in the evaluator or decoder, not both")
        _reset(decoder)
        events: list[dict[str, Any]] = []
        cooldown_until: float | None = None
        rearmed = True
        for item in records:
            if cooldown_until is not None and item.timestamp >= cooldown_until:
                rearm = getattr(decoder, "rearm", None)
                if rearm is not None and not rearmed:
                    rearm()
                cooldown_until = None
                rearmed = True
            if cooldown_until is not None:
                continue
            event = _event_from_result(_update(decoder, item), item)
            if event is not None:
                event = _add_frame_coordinates(event, records)
                events.append(event)
                total_events += 1
                if cooldown_seconds:
                    cooldown_until = item.timestamp + cooldown_seconds
                    rearmed = False
                else:
                    rearm = getattr(decoder, "rearm", None)
                    if rearm is not None:
                        rearm()

        exposure = _exposure(source, records, source_frame_step)
        total_exposure += exposure
        label = str(source.get("label", source.get("class", "negative"))).lower()
        positive = label in {"positive", "wake", "target", "true", "1"}
        detected = bool(events)
        if positive:
            positive_sources += 1
            positive_scores.extend(
                value for item in records for value in _scalar_values(item.score)
            )
            detected_positive_sources += detected
        else:
            negative_scores.extend(
                value for item in records for value in _scalar_values(item.score)
            )
            false_accepts += len(events)
            negative_exposure += exposure
        all_scores.extend(
            value for item in records for value in _scalar_values(item.score)
        )
        source_reports.append(
            {
                "source_id": source.get("id", source.get("source_id", source_number)),
                "session_id": source.get("session_id"),
                "label": label,
                "positive": positive,
                "exposure_seconds": exposure,
                "exposure_hours": exposure / 3600.0,
                "frame_step_seconds": source_frame_step,
                "events": events,
                "event_count": len(events),
                "detected": detected,
                "positive_recall": (1.0 if detected else 0.0) if positive else None,
                "false_rejection_rate": (
                    (0.0 if detected else 1.0) if positive else None
                ),
                "scores": _summary(
                    [value for item in records for value in _scalar_values(item.score)]
                ),
            }
        )

    recall = detected_positive_sources / positive_sources if positive_sources else None
    sessions: dict[str, list[dict[str, Any]]] = {}
    for source in source_reports:
        session_id = source["session_id"]
        if session_id is not None:
            sessions.setdefault(str(session_id), []).append(source)
    session_reports = []
    for session_id, session_sources in sorted(sessions.items()):
        positives = [source for source in session_sources if source["positive"]]
        detected_positives = sum(source["detected"] for source in positives)
        session_recall = detected_positives / len(positives) if positives else None
        session_reports.append(
            {
                "session_id": session_id,
                "source_count": len(session_sources),
                "positive_sources": len(positives),
                "detected_positive_sources": detected_positives,
                "positive_recall": session_recall,
                "false_rejection_rate": (
                    1.0 - session_recall if session_recall is not None else None
                ),
            }
        )
    return {
        "sources": source_reports,
        "sessions": session_reports,
        "exposure_seconds": total_exposure,
        "exposure_hours": total_exposure / 3600.0,
        "negative_exposure_seconds": negative_exposure,
        "negative_exposure_hours": negative_exposure / 3600.0,
        "total_events": total_events,
        "false_accepts": false_accepts,
        "false_accepts_per_hour": (
            false_accepts / (negative_exposure / 3600.0) if negative_exposure else None
        ),
        "poisson_upper_bound_95_per_hour": poisson_upper_bound_95(
            false_accepts, negative_exposure
        ),
        "positive_sources": positive_sources,
        "detected_positive_sources": detected_positive_sources,
        "positive_recall": recall,
        "false_rejection_rate": (1.0 - recall if recall is not None else None),
        "score_distributions": {
            "all": _summary(all_scores),
            "positive": _summary(positive_scores),
            "negative": _summary(negative_scores),
        },
    }


def _decoder_factory(spec: str, decoder_args: dict[str, Any]) -> Callable[[], Any]:
    module_name, separator, class_name = spec.partition(":")
    if not separator:
        raise ValueError("--decoder must be module:Class")
    cls = getattr(importlib.import_module(module_name), class_name)
    return lambda: cls(**decoder_args)


def apply_decoder_contract(
    decoder_args: Mapping[str, Any], contract: Mapping[str, Any]
) -> tuple[dict[str, Any], float]:
    """Merge artifact-bound score settings and reject CLI divergence."""
    if contract.get("schema_version") != 1:
        raise ValueError("unsupported ordered-state decoder contract")
    if int(contract.get("state_count", -1)) != KIZZ_TOPOLOGY.state_count:
        raise ValueError("decoder contract has the wrong state count")
    contract_args = contract.get("decoder_args")
    if not isinstance(contract_args, Mapping):
        raise ValueError("decoder contract needs decoder_args")
    merged = dict(contract_args)
    for key, value in decoder_args.items():
        if key in merged and merged[key] != value:
            raise ValueError(f"decoder argument {key} conflicts with artifact contract")
        merged[key] = value
    frame_step = float(contract["frame_step_seconds"])
    if not math.isfinite(frame_step) or frame_step <= 0:
        raise ValueError("decoder contract frame_step_seconds must be positive")
    return merged, frame_step


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--decoder", required=True, help="module:StreamingStateDecoder")
    parser.add_argument("--decoder-args", default="{}", help="JSON constructor object")
    parser.add_argument(
        "--decoder-contract",
        type=Path,
        help="Artifact-bound JSON score settings; required for qualification",
    )
    parser.add_argument("--cooldown-seconds", type=float, default=0.0)
    parser.add_argument(
        "--frame-step-seconds",
        type=float,
        help="Expected timestamp cadence; includes the final frame in inferred exposure",
    )
    parser.add_argument(
        "--require-declared-exposure",
        action="store_true",
        help="Reject inferred durations; required for qualification FAPH reports",
    )
    parser.add_argument("--output", type=Path, default=Path("-"))
    args = parser.parse_args(argv)
    if args.require_declared_exposure and not args.decoder_contract:
        parser.error("qualification requires --decoder-contract")
    decoder_args = json.loads(args.decoder_args)
    frame_step_seconds = args.frame_step_seconds
    if args.decoder_contract:
        decoder_args, contract_frame_step = apply_decoder_contract(
            decoder_args, json.loads(args.decoder_contract.read_text())
        )
        if frame_step_seconds is not None and not math.isclose(
            frame_step_seconds, contract_frame_step
        ):
            parser.error("--frame-step-seconds conflicts with decoder contract")
        frame_step_seconds = contract_frame_step
    report = evaluate_sources(
        load_sources(args.input),
        _decoder_factory(args.decoder, decoder_args),
        cooldown_seconds=args.cooldown_seconds,
        require_declared_exposure=args.require_declared_exposure,
        frame_step_seconds=frame_step_seconds,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if str(args.output) == "-":
        print(rendered)
    else:
        args.output.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
