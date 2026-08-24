#!/usr/bin/env python3
"""Score continuous RaggedMmap feature streams with the ordered-state artifact.

The manifest is deliberately explicit.  It contains ``sources``; each source
has ``id``, ``path``, ``split`` (``train``, ``validation``, or ``test``), and
``exposure_seconds``.  The declared exposure must equal the stored feature
frames multiplied by ``feature_step_seconds``.  Optional ``occurrences`` are
per-item positive intervals in seconds and are used only for recall reporting.

RaggedMmap items are independent continuous streams: the model and decoder
are reset between items, never between stride groups within an item.  Stored
features are consumed in their original order.  A trailing partial stride is
reported but not scored; it is not silently included in scored exposure.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from mmap_ninja.ragged import RaggedMmap

from microwakeword.ordered_state import (
    KIZZ_TOPOLOGY,
    OrderedStateMarginSweepDecoder,
)
from tools.evaluate_ordered_state import apply_decoder_contract, poisson_upper_bound_95
from tools.score_ordered_state_streams import (
    QuantizedStreamingModel,
    event_matches_occurrence,
)

FEATURE_STEP_SECONDS = 0.01
MODEL_STEP_SECONDS = 0.03
MAX_EVENT_RECORDS_PER_SOURCE_MARGIN = 10_000


def sha256_path(path: Path) -> str:
    """Hash a file or directory deterministically."""
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    paths = sorted(p for p in path.rglob("*") if p.is_file())
    for child in paths:
        relative = child.relative_to(path)
        encoded = str(relative).encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        with child.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, Mapping) or not isinstance(payload.get("sources"), list):
        raise ValueError("manifest must contain a sources list")
    sources = []
    for index, raw in enumerate(payload["sources"]):
        if not isinstance(raw, Mapping):
            raise ValueError(f"sources[{index}] must be an object")
        item = dict(raw)
        if not item.get("path") or item.get("split") not in {
            "train",
            "validation",
            "test",
        }:
            raise ValueError(f"sources[{index}] needs path and declared split")
        label = str(item.get("label", "")).lower()
        if label not in {"positive", "negative"}:
            raise ValueError(
                f"sources[{index}] needs an explicit positive or negative label"
            )
        item["label"] = label
        exposure = item.get("exposure_seconds")
        if label == "negative":
            if (
                not isinstance(exposure, (int, float))
                or not math.isfinite(float(exposure))
                or float(exposure) < 0
            ):
                raise ValueError(
                    f"sources[{index}] negative source needs finite non-negative exposure_seconds"
                )
        elif exposure is not None and (
            not isinstance(exposure, (int, float))
            or not math.isfinite(float(exposure))
            or float(exposure) < 0
        ):
            raise ValueError(
                f"sources[{index}] positive exposure_seconds must be finite and non-negative"
            )
        if label == "positive" and not isinstance(item.get("occurrences"), list):
            raise ValueError(f"sources[{index}] positive source needs occurrences")
        if label == "positive" and not item["occurrences"]:
            raise ValueError(
                f"sources[{index}] positive source needs at least one occurrence"
            )
        item["id"] = str(item.get("id", Path(item["path"]).stem))
        feature_step = item.get(
            "feature_step_seconds",
            payload.get("feature_step_seconds", FEATURE_STEP_SECONDS),
        )
        if not math.isclose(
            float(feature_step), FEATURE_STEP_SECONDS, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(f"sources[{index}] has a non-10 ms feature cadence")
        item["feature_step_seconds"] = FEATURE_STEP_SECONDS
        item["path"] = (
            str((path.parent / item["path"]).resolve())
            if not Path(item["path"]).is_absolute()
            else item["path"]
        )
        sources.append(item)
    result = dict(payload)
    result["sources"] = sources
    return result


@dataclass
class _OnlineScoreStats:
    count: int = 0
    finite_count: int = 0
    finite_minimum: float = math.inf
    finite_maximum: float = -math.inf
    finite_sum: float = 0.0
    negative_infinity_count: int = 0
    positive_infinity_count: int = 0

    def update(self, value: float) -> None:
        self.count += 1
        if math.isfinite(value):
            self.finite_count += 1
            self.finite_minimum = min(self.finite_minimum, value)
            self.finite_maximum = max(self.finite_maximum, value)
            self.finite_sum += value
        elif value == -math.inf:
            self.negative_infinity_count += 1
        elif value == math.inf:
            self.positive_infinity_count += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "finite_count": self.finite_count,
            "finite_minimum": (self.finite_minimum if self.finite_count else None),
            "finite_maximum": (self.finite_maximum if self.finite_count else None),
            "finite_mean": (
                self.finite_sum / self.finite_count if self.finite_count else None
            ),
            "negative_infinity_count": self.negative_infinity_count,
            "positive_infinity_count": self.positive_infinity_count,
        }


def _event_value(event: Any, name: str, default: Any = None) -> Any:
    return getattr(
        event, name, event.get(name, default) if isinstance(event, Mapping) else default
    )


def _decoded_feature(feature: np.ndarray) -> np.ndarray:
    if np.issubdtype(feature.dtype, np.uint16):
        return feature.astype(np.float32) * 0.0390625
    return np.asarray(feature, dtype=np.float32)


def _validate_contract(model: Any, frame_step: float) -> int:
    shape = tuple(int(value) for value in model.input["shape"])
    if len(shape) != 3 or shape[0] != 1 or shape[-1] != 40 or shape[1] != 3:
        raise ValueError(
            f"feature stream requires quantized [1, 3, 40] input, got {shape}"
        )
    if int(np.prod(model.output["shape"])) != KIZZ_TOPOLOGY.state_count:
        raise ValueError("artifact output must contain exactly 23 state logits")
    if not math.isclose(frame_step, MODEL_STEP_SECONDS, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "ordered-state feature scorer requires the 30 ms decoder cadence"
        )
    return shape[1]


def _occurrences_for_item(
    source: Mapping[str, Any], item_index: int
) -> list[Mapping[str, Any]]:
    occurrences = source.get("occurrences", [])
    if not isinstance(occurrences, list):
        raise ValueError(f"{source['id']}: occurrences must be a list")
    return [
        item for item in occurrences if int(item.get("item_index", -1)) == item_index
    ]


def score_feature_source(
    source: Mapping[str, Any],
    model: Any,
    decoder_args: Mapping[str, Any],
    frame_step: float,
    model_hash: str,
    decoder_hash: str,
    completion_margin: float,
    cooldown_frames: int,
) -> dict[str, Any]:
    return _score_feature_source_margins(
        source,
        model,
        decoder_args,
        frame_step,
        model_hash,
        decoder_hash,
        [completion_margin],
        cooldown_frames,
    )[0]


def _score_feature_source_margins(
    source: Mapping[str, Any],
    model: Any,
    decoder_args: Mapping[str, Any],
    frame_step: float,
    model_hash: str,
    decoder_hash: str,
    completion_margins: Sequence[float],
    cooldown_frames: int,
) -> list[dict[str, Any]]:
    """Score one source once, fanning each model output out to its decoders."""
    if not completion_margins:
        raise ValueError("at least one completion margin is required")
    path = Path(source["path"])
    if not path.is_dir():
        raise ValueError(f"{source['id']}: RaggedMmap directory does not exist: {path}")
    mmap = RaggedMmap(path)
    stride = _validate_contract(model, frame_step)
    stored_frames = scored_frames = 0
    states = [
        {
            "events": 0,
            "completion_stats": _OnlineScoreStats(),
            "event_records": [],
            "event_records_truncated": 0,
            "detected_occurrences": set(),
        }
        for _ in completion_margins
    ]
    declared_occurrences = 0
    for item_index in range(len(mmap)):
        features = np.asarray(mmap[item_index])
        if features.ndim != 2 or features.shape[1] != 40:
            raise ValueError(
                f"{source['id']} item {item_index}: expected [frames, 40] features"
            )
        stored_frames += int(features.shape[0])
        model.reset()
        decoder = OrderedStateMarginSweepDecoder(
            **decoder_args,
            completion_margins=completion_margins,
            cooldown_frames=cooldown_frames,
        )
        decoder.reset()
        item_events = [[] for _ in completion_margins]
        for offset in range(0, len(features) - stride + 1, stride):
            logits = model.step(_decoded_feature(features[offset : offset + stride]))
            frame_index = offset // stride
            scored_frames += stride
            events = decoder.step(logits, frame_index=frame_index)
            completion_scores = decoder.current_completion_scores
            for margin_index, event in enumerate(events):
                state = states[margin_index]
                state["completion_stats"].update(
                    float(_event_value(event, "score", math.nan))
                    if event is not None
                    else float(completion_scores[margin_index])
                )
                if event is not None:
                    state["events"] += 1
                    item_event = {
                        "item_index": item_index,
                        "start_timestamp": float(_event_value(event, "start_frame", 0))
                        * frame_step,
                        "end_timestamp": (
                            float(_event_value(event, "end_frame", 0)) + 1.0
                        )
                        * frame_step,
                        "score": float(_event_value(event, "score", math.nan)),
                    }
                    item_event["duration_seconds"] = (
                        item_event["end_timestamp"] - item_event["start_timestamp"]
                    )
                    item_events[margin_index].append(item_event)
                    if (
                        len(state["event_records"])
                        < MAX_EVENT_RECORDS_PER_SOURCE_MARGIN
                    ):
                        state["event_records"].append(item_event)
                    else:
                        state["event_records_truncated"] += 1
        occurrences = _occurrences_for_item(source, item_index)
        declared_occurrences += len(occurrences)
        for occurrence_index, occurrence in enumerate(occurrences):
            start = float(occurrence["start_seconds"])
            end = float(occurrence["end_seconds"])
            if not (0 <= start < end <= features.shape[0] * FEATURE_STEP_SECONDS):
                raise ValueError(
                    f"{source['id']} item {item_index}: invalid occurrence"
                )
            occurrence_id = str(
                occurrence.get("id", f"{item_index}:{occurrence_index}")
            )
            for margin_index, events_for_margin in enumerate(item_events):
                if any(
                    event_matches_occurrence(event, start, end)
                    for event in events_for_margin
                ):
                    states[margin_index]["detected_occurrences"].add(occurrence_id)
    label = str(source["label"])
    measured_exposure = stored_frames * FEATURE_STEP_SECONDS
    declared_exposure = (
        float(source["exposure_seconds"])
        if source.get("exposure_seconds") is not None
        else None
    )
    if label == "negative" and not math.isclose(
        declared_exposure, measured_exposure, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError(
            f"{source['id']}: declared exposure {declared_exposure} != stored exposure {measured_exposure}"
        )
    scored_exposure = scored_frames * FEATURE_STEP_SECONDS
    negative_scored_exposure = scored_exposure if label == "negative" else 0.0
    source_hash = sha256_path(path)
    expected_hash = source.get("expected_path_sha256")
    if expected_hash is not None and source_hash != expected_hash:
        raise ValueError(
            f"{source['id']}: source hash {source_hash} != expected {expected_hash}"
        )
    reports = []
    for margin, state in zip(completion_margins, states):
        events = int(state["events"])
        negative_events = events if label == "negative" else 0
        detected_occurrences = state["detected_occurrences"]
        reports.append(
            {
                "source_id": str(source["id"]),
                "split": str(source["split"]),
                "label": label,
                "category": source.get("category"),
                "channel": source.get("channel"),
                "source_family": source.get("source_family"),
                "session_id": source.get("session_id"),
                "speaker_id": source.get("speaker_id"),
                "path": str(path.resolve()),
                "source_sha256": source_hash,
                "declared_exposure_seconds": declared_exposure,
                "measured_exposure_seconds": measured_exposure,
                "stored_feature_frames": stored_frames,
                "scored_feature_frames": scored_frames,
                "trailing_feature_frames": stored_frames - scored_frames,
                "scored_exposure_seconds": scored_exposure,
                "event_count": events,
                "negative_event_count": negative_events,
                "negative_scored_exposure_seconds": negative_scored_exposure,
                "faph": (
                    negative_events / (negative_scored_exposure / 3600.0)
                    if negative_scored_exposure
                    else None
                ),
                "poisson_upper_bound_95_per_hour": (
                    poisson_upper_bound_95(negative_events, negative_scored_exposure)
                    if negative_scored_exposure
                    else None
                ),
                "completion_score_stats": state["completion_stats"].as_dict(),
                "events": state["event_records"],
                "event_records_truncated": state["event_records_truncated"],
                "positive_occurrence_count": declared_occurrences,
                "detected_positive_occurrence_count": len(detected_occurrences),
                "positive_occurrence_recall": (
                    len(detected_occurrences) / declared_occurrences
                    if declared_occurrences
                    else None
                ),
                "model_sha256": model_hash,
                "decoder_contract_sha256": decoder_hash,
                "completion_margin": margin,
                "cooldown_frames": cooldown_frames,
            }
        )
    return reports


def _merge_stats(reports: list[dict[str, Any]]) -> dict[str, Any]:
    stats = [item["completion_score_stats"] for item in reports]
    finite_count = sum(int(item["finite_count"]) for item in stats)
    finite_sum = sum(
        float(item["finite_mean"]) * int(item["finite_count"])
        for item in stats
        if item["finite_mean"] is not None
    )
    minima = [
        float(item["finite_minimum"])
        for item in stats
        if item["finite_minimum"] is not None
    ]
    maxima = [
        float(item["finite_maximum"])
        for item in stats
        if item["finite_maximum"] is not None
    ]
    return {
        "count": sum(int(item["count"]) for item in stats),
        "finite_count": finite_count,
        "finite_minimum": min(minima) if minima else None,
        "finite_maximum": max(maxima) if maxima else None,
        "finite_mean": finite_sum / finite_count if finite_count else None,
        "negative_infinity_count": sum(
            int(item["negative_infinity_count"]) for item in stats
        ),
        "positive_infinity_count": sum(
            int(item["positive_infinity_count"]) for item in stats
        ),
    }


def _aggregate(
    reports: list[dict[str, Any]], margin: float, cooldown_frames: int
) -> dict[str, Any]:
    negative_exposure = sum(
        float(item["negative_scored_exposure_seconds"]) for item in reports
    )
    negative_events = sum(int(item["negative_event_count"]) for item in reports)
    total_events = sum(int(item["event_count"]) for item in reports)
    occurrences = sum(int(item["positive_occurrence_count"]) for item in reports)
    detected = sum(int(item["detected_positive_occurrence_count"]) for item in reports)
    return {
        "completion_margin": margin,
        "cooldown_frames": cooldown_frames,
        "source_count": len(reports),
        "exposure_seconds": negative_exposure,
        "negative_scored_exposure_seconds": negative_exposure,
        "event_count": total_events,
        "negative_event_count": negative_events,
        "faph": (
            negative_events / (negative_exposure / 3600.0)
            if negative_exposure
            else None
        ),
        "poisson_upper_bound_95_per_hour": (
            poisson_upper_bound_95(negative_events, negative_exposure)
            if negative_exposure
            else None
        ),
        "positive_occurrence_count": occurrences,
        "detected_positive_occurrence_count": detected,
        "positive_occurrence_recall": detected / occurrences if occurrences else None,
        "completion_score_stats": _merge_stats(reports),
    }


def _negative_category_aggregates(
    reports: list[dict[str, Any]], margin: float, cooldown_frames: int
) -> dict[str, dict[str, Any]]:
    categories: dict[str, list[dict[str, Any]]] = {}
    for report in reports:
        if report["label"] != "negative":
            continue
        category = str(report.get("category") or "unclassified")
        categories.setdefault(category, []).append(report)
    return {
        category: _aggregate(category_reports, margin, cooldown_frames)
        for category, category_reports in sorted(categories.items())
    }


def run_manifest(
    manifest_path: Path,
    model_path: Path,
    contract_path: Path,
    output_path: Path,
    margins: Sequence[float],
    cooldown_frames: int,
    max_faph: float,
    min_recall: float,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    contract = json.loads(contract_path.read_text())
    decoder_args, frame_step = apply_decoder_contract({}, contract)
    model_hash = sha256_path(model_path)
    decoder_hash = sha256_path(contract_path)
    model = QuantizedStreamingModel(model_path)
    _validate_contract(model, frame_step)
    if cooldown_frames < 0:
        raise ValueError("cooldown_frames must be non-negative")
    if len(margins) > 1:
        if manifest.get("positive_occurrence_geometry") != "exact_phrase_span":
            raise ValueError(
                "completion-margin sweep requires exact positive phrase spans"
            )
        if any(source["split"] != "validation" for source in manifest["sources"]):
            raise ValueError(
                "completion-margin sweep may consume validation sources only"
            )
        if not any(source["label"] == "negative" for source in manifest["sources"]):
            raise ValueError("completion-margin sweep needs negative exposure")
        if not any(source["label"] == "positive" for source in manifest["sources"]):
            raise ValueError("completion-margin sweep needs a positive occurrence")
    all_reports: list[dict[str, Any]] = []
    sweep: list[dict[str, Any]] = []
    if any(not math.isfinite(margin) for margin in margins):
        raise ValueError("completion margins must be finite")
    reports_by_margin = [[] for _ in margins]
    for source in manifest["sources"]:
        source_reports = _score_feature_source_margins(
            source,
            model,
            decoder_args,
            frame_step,
            model_hash,
            decoder_hash,
            margins,
            cooldown_frames,
        )
        for reports, source_report in zip(reports_by_margin, source_reports):
            reports.append(source_report)
    for margin, reports in zip(margins, reports_by_margin):
        aggregate = _aggregate(reports, margin, cooldown_frames)
        aggregate["negative_categories"] = _negative_category_aggregates(
            reports, margin, cooldown_frames
        )
        sweep.append(aggregate)
        if len(margins) == 1:
            all_reports = reports
    selected = None
    if len(margins) > 1:
        eligible = [
            item
            for item in sweep
            if item["faph"] is not None
            and item["faph"] <= max_faph
            and item["positive_occurrence_recall"] is not None
            and item["positive_occurrence_recall"] >= min_recall
        ]
        selected = (
            max(
                eligible,
                key=lambda item: (
                    float(item["positive_occurrence_recall"]),
                    -float(item["faph"]),
                    float(item["completion_margin"]),
                ),
            )
            if eligible
            else None
        )
    result = {
        "schema_version": 1,
        "frame_step_seconds": frame_step,
        "feature_step_seconds": FEATURE_STEP_SECONDS,
        "stride": 3,
        "model_sha256": model_hash,
        "decoder_contract_sha256": decoder_hash,
        "manifest_sha256": sha256_path(manifest_path),
        "declared_split_counts": {
            split: sum(source["split"] == split for source in manifest["sources"])
            for split in ("train", "validation", "test")
        },
        "completion_margin_sweep": sweep,
        "selected_operating_point": selected,
        "reports": all_reports,
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--decoder-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--completion-margin", type=float, default=0.0)
    parser.add_argument(
        "--sweep-margins", help="comma-separated validation-only margins"
    )
    parser.add_argument("--cooldown-frames", type=int, default=0)
    parser.add_argument("--max-faph", type=float, default=0.1)
    parser.add_argument("--min-recall", type=float, default=0.9)
    args = parser.parse_args(argv)
    margins = (
        [float(item) for item in args.sweep_margins.split(",")]
        if args.sweep_margins
        else [args.completion_margin]
    )
    if (
        not 0 <= args.min_recall <= 1
        or not math.isfinite(args.max_faph)
        or args.max_faph < 0
    ):
        parser.error("invalid sweep operating-point limits")
    run_manifest(
        args.manifest,
        args.model,
        args.decoder_contract,
        args.output,
        margins,
        args.cooldown_frames,
        args.max_faph,
        args.min_recall,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
