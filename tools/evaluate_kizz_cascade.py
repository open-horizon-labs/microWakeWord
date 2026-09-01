#!/usr/bin/env python3
"""Evaluate a frozen Kizz detector -> verifier cascade from score traces only.

The evaluator never runs either model.  It verifies two immutable trace files,
pairs verifier scores to detector candidate IDs, removes temporally overlapping
detector candidates, selects both thresholds on validation, and applies those
frozen thresholds to test exactly once.

Trace schema (abridged)::

    {
      "schema_version": 1,
      "trace_kind": "detector",  # or "verifier"
      "artifact": {"sha256": "...", "path": "optional/model.tflite"},
      "sources": [{
        "source_id": "...", "split": "validation", "truth": "positive",
        "duration_seconds": 1.5, "audio_sha256": "...",
        "speaker_id": "optional-disjoint-identity",
        "opportunities": [{"opportunity_id": "...", "start_seconds": 0,
                            "end_seconds": 1.5}],
        "events": [{"candidate_id": "...", "start_seconds": 0.2,
                    "end_seconds": 1.1, "timestamp_seconds": 0.8,
                    "score": 0.91, "window_sha256": "optional"}]
      }]
    }

Positive clips may omit ``opportunities``; the complete clip is then one wake
opportunity.  Negative sources contribute their measured duration to FAPH.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from microwakeword.kizz_continuous_evaluation import poisson_upper_95
from microwakeword.kizz_evaluation_contract import require_disjoint_groups


SCHEMA_VERSION = 1
DEFAULT_DETECTOR_RECALL_TARGET = 0.98
DEFAULT_CASCADE_RECALL_TARGET = 0.95
DEFAULT_MAX_FAPH = 0.10
DEFAULT_MIN_NEGATIVE_EXPOSURE_HOURS = 100.0
CONFIDENCE_LEVEL = 0.95


@dataclass(frozen=True)
class Opportunity:
    opportunity_id: str
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class Candidate:
    source_id: str
    candidate_id: str
    split: str
    truth: str
    start_seconds: float
    end_seconds: float
    timestamp_seconds: float
    detector_score: float
    verifier_score: float
    opportunity_id: str | None
    window_sha256: str | None


@dataclass(frozen=True)
class Source:
    source_id: str
    split: str
    truth: str
    duration_seconds: float
    audio_sha256: str
    identity_row: Mapping[str, Any]
    opportunities: tuple[Opportunity, ...]
    candidates: tuple[Candidate, ...]
    raw_candidate_count: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, label: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return text


def _finite(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _truth(value: object, label: str) -> str:
    if value in (1, True, "positive"):
        return "positive"
    if value in (0, False, "negative"):
        return "negative"
    raise ValueError(f"{label} truth must be positive/negative or 1/0")


def _artifact_provenance(trace: Mapping[str, Any], trace_path: Path) -> dict[str, Any]:
    artifact = trace.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ValueError(f"{trace_path}: trace must declare artifact provenance")
    digest = _require_sha256(artifact.get("sha256"), f"{trace_path} artifact")
    result: dict[str, Any] = {"sha256": digest}
    if artifact.get("path"):
        artifact_path = Path(str(artifact["path"]))
        if not artifact_path.is_absolute():
            artifact_path = trace_path.parent / artifact_path
        artifact_path = artifact_path.resolve()
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        actual = sha256_file(artifact_path)
        if actual != digest:
            raise ValueError(
                f"{trace_path}: artifact hash drift: declared {digest}, got {actual}"
            )
        result["path"] = str(artifact_path)
        result["bytes"] = artifact_path.stat().st_size
    if artifact.get("artifact_id"):
        result["artifact_id"] = str(artifact["artifact_id"])
    return result


def _load_trace(
    path: Path, *, expected_sha256: str, expected_kind: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path.resolve()
    expected = _require_sha256(expected_sha256, f"{expected_kind} trace expected hash")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{expected_kind} trace hash drift: expected {expected}, got {actual}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: trace root must be an object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported trace schema")
    if payload.get("trace_kind") != expected_kind:
        raise ValueError(f"{path}: expected {expected_kind} trace")
    if not isinstance(payload.get("sources"), list) or not payload["sources"]:
        raise ValueError(f"{path}: trace needs non-empty sources")
    provenance = {
        "path": str(path),
        "sha256": actual,
        "bytes": path.stat().st_size,
        "artifact": _artifact_provenance(payload, path),
    }
    return payload, provenance


def _source_metadata(row: Mapping[str, Any], *, trace_label: str) -> dict[str, Any]:
    source_id = str(row.get("source_id") or "").strip()
    if not source_id:
        raise ValueError(f"{trace_label}: source has no source_id")
    split = str(row.get("split") or "")
    if split not in {"validation", "test"}:
        raise ValueError(f"{trace_label} source {source_id}: unsupported split {split!r}")
    truth = _truth(row.get("truth", row.get("label")), f"{source_id}")
    duration = _finite(row.get("duration_seconds"), f"{source_id} duration_seconds")
    if duration <= 0:
        raise ValueError(f"{source_id}: duration_seconds must be positive")
    audio_sha256 = _require_sha256(row.get("audio_sha256"), f"{source_id} audio")
    return {
        "source_id": source_id,
        "split": split,
        "truth": truth,
        "duration_seconds": duration,
        "audio_sha256": audio_sha256,
    }


def _opportunities(
    row: Mapping[str, Any], metadata: Mapping[str, Any]
) -> tuple[Opportunity, ...]:
    source_id = str(metadata["source_id"])
    duration = float(metadata["duration_seconds"])
    if metadata["truth"] == "negative":
        if row.get("opportunities"):
            raise ValueError(f"{source_id}: negative source cannot have opportunities")
        return ()
    raw = row.get("opportunities")
    if raw is None:
        return (Opportunity(source_id, 0.0, duration),)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{source_id}: positive opportunities must be non-empty")
    values: list[Opportunity] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"{source_id}: opportunity {index} must be an object")
        opportunity_id = str(item.get("opportunity_id") or "").strip()
        if not opportunity_id or opportunity_id in seen:
            raise ValueError(f"{source_id}: opportunity IDs must be unique and non-empty")
        seen.add(opportunity_id)
        start = _finite(item.get("start_seconds"), f"{opportunity_id} start")
        end = _finite(item.get("end_seconds"), f"{opportunity_id} end")
        if start < 0 or end < start or end > duration:
            raise ValueError(f"{source_id}: opportunity is outside source duration")
        values.append(Opportunity(opportunity_id, start, end))
    values.sort(key=lambda item: (item.start_seconds, item.end_seconds, item.opportunity_id))
    for left, right in zip(values, values[1:]):
        if right.start_seconds <= left.end_seconds:
            raise ValueError(f"{source_id}: positive opportunities must not overlap")
    return tuple(values)


def _events(row: Mapping[str, Any], *, role: str, duration: float) -> dict[str, dict[str, Any]]:
    raw = row.get("events")
    if not isinstance(raw, list):
        raise ValueError(f"{role} source {row.get('source_id')}: events must be a list")
    result: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"{role} event {index} must be an object")
        candidate_id = str(item.get("candidate_id") or "").strip()
        if not candidate_id or candidate_id in result:
            raise ValueError(f"{role} candidate IDs must be unique and non-empty per source")
        start = _finite(item.get("start_seconds"), f"{role} {candidate_id} start")
        end = _finite(item.get("end_seconds"), f"{role} {candidate_id} end")
        timestamp = _finite(
            item.get("timestamp_seconds"), f"{role} {candidate_id} timestamp"
        )
        score = _finite(item.get("score"), f"{role} {candidate_id} score")
        if start < 0 or end < start or end > duration or not start <= timestamp <= end:
            raise ValueError(f"{role} candidate {candidate_id} is outside source duration")
        window_hash = item.get("window_sha256")
        if window_hash is not None:
            window_hash = _require_sha256(window_hash, f"{role} {candidate_id} window")
        result[candidate_id] = {
            "candidate_id": candidate_id,
            "start_seconds": start,
            "end_seconds": end,
            "timestamp_seconds": timestamp,
            "score": score,
            "window_sha256": window_hash,
        }
    return result


def _opportunity_key(source_id: str, opportunity_id: str) -> str:
    return f"{source_id}\x1f{opportunity_id}"


def _opportunity_at(
    source_id: str, timestamp: float, opportunities: Sequence[Opportunity]
) -> str | None:
    matches = [
        item.opportunity_id
        for item in opportunities
        if item.start_seconds <= timestamp <= item.end_seconds
    ]
    if len(matches) > 1:  # Defensive: opportunity validation already forbids this.
        raise ValueError("candidate timestamp matches overlapping opportunities")
    return _opportunity_key(source_id, matches[0]) if matches else None


def _deduplicate_candidates(candidates: Sequence[Candidate]) -> tuple[Candidate, ...]:
    """Collapse each transitive interval-overlap cluster before thresholding.

    The representative is chosen using detector score only, so verifier values
    cannot influence which candidate reaches the verifier threshold sweep.
    """
    ordered = sorted(
        candidates,
        key=lambda item: (item.start_seconds, item.end_seconds, item.candidate_id),
    )
    if not ordered:
        return ()
    clusters: list[list[Candidate]] = []
    cluster = [ordered[0]]
    cluster_end = ordered[0].end_seconds
    for item in ordered[1:]:
        if item.start_seconds <= cluster_end:
            cluster.append(item)
            cluster_end = max(cluster_end, item.end_seconds)
        else:
            clusters.append(cluster)
            cluster = [item]
            cluster_end = item.end_seconds
    clusters.append(cluster)
    return tuple(
        sorted(values, key=lambda item: (-item.detector_score, item.candidate_id))[0]
        for values in clusters
    )


def _pair_sources(
    detector: Mapping[str, Any], verifier: Mapping[str, Any]
) -> tuple[Source, ...]:
    def indexed(payload: Mapping[str, Any], label: str) -> dict[str, Mapping[str, Any]]:
        result: dict[str, Mapping[str, Any]] = {}
        for raw in payload["sources"]:
            if not isinstance(raw, Mapping):
                raise ValueError(f"{label}: source rows must be objects")
            metadata = _source_metadata(raw, trace_label=label)
            source_id = metadata["source_id"]
            if source_id in result:
                raise ValueError(f"{label}: duplicate source_id {source_id}")
            result[source_id] = raw
        return result

    detector_rows = indexed(detector, "detector")
    verifier_rows = indexed(verifier, "verifier")
    if set(detector_rows) != set(verifier_rows):
        missing_verifier = sorted(set(detector_rows) - set(verifier_rows))
        missing_detector = sorted(set(verifier_rows) - set(detector_rows))
        raise ValueError(
            "detector/verifier source sets drift: "
            f"missing verifier={missing_verifier}, missing detector={missing_detector}"
        )

    sources: list[Source] = []
    for source_id in sorted(detector_rows):
        detector_row = detector_rows[source_id]
        verifier_row = verifier_rows[source_id]
        detector_meta = _source_metadata(detector_row, trace_label="detector")
        verifier_meta = _source_metadata(verifier_row, trace_label="verifier")
        if detector_meta != verifier_meta:
            raise ValueError(f"{source_id}: detector/verifier source metadata drift")
        identity_keys = (
            "source_audio_sha256",
            "provenance_id",
            "ancestry_id",
            "parent_id",
            "parent_source_id",
            "speaker_id",
            "voice_id",
            "session_id",
        )
        if any(detector_row.get(key) != verifier_row.get(key) for key in identity_keys):
            raise ValueError(f"{source_id}: detector/verifier identity metadata drift")
        opportunities = _opportunities(detector_row, detector_meta)
        verifier_opportunities = _opportunities(verifier_row, verifier_meta)
        if opportunities != verifier_opportunities:
            raise ValueError(f"{source_id}: detector/verifier opportunity drift")
        detector_events = _events(
            detector_row, role="detector", duration=detector_meta["duration_seconds"]
        )
        verifier_events = _events(
            verifier_row, role="verifier", duration=detector_meta["duration_seconds"]
        )
        if set(detector_events) != set(verifier_events):
            raise ValueError(f"{source_id}: detector/verifier candidate sets drift")
        paired: list[Candidate] = []
        for candidate_id in sorted(detector_events):
            detector_event = detector_events[candidate_id]
            verifier_event = verifier_events[candidate_id]
            for key in (
                "start_seconds",
                "end_seconds",
                "timestamp_seconds",
                "window_sha256",
            ):
                if detector_event[key] != verifier_event[key]:
                    raise ValueError(
                        f"{source_id}/{candidate_id}: detector/verifier event metadata drift"
                    )
            paired.append(
                Candidate(
                    source_id=source_id,
                    candidate_id=candidate_id,
                    split=detector_meta["split"],
                    truth=detector_meta["truth"],
                    start_seconds=detector_event["start_seconds"],
                    end_seconds=detector_event["end_seconds"],
                    timestamp_seconds=detector_event["timestamp_seconds"],
                    detector_score=detector_event["score"],
                    verifier_score=verifier_event["score"],
                    opportunity_id=_opportunity_at(
                        source_id, detector_event["timestamp_seconds"], opportunities
                    ),
                    window_sha256=detector_event["window_sha256"],
                )
            )
        deduplicated = _deduplicate_candidates(paired)
        identity_row = {
            key: detector_row[key]
            for key in (
                "source_id",
                "audio_sha256",
                "source_audio_sha256",
                "provenance_id",
                "ancestry_id",
                "parent_id",
                "parent_source_id",
                "speaker_id",
                "voice_id",
                "session_id",
            )
            if detector_row.get(key)
        }
        sources.append(
            Source(
                source_id=source_id,
                split=detector_meta["split"],
                truth=detector_meta["truth"],
                duration_seconds=detector_meta["duration_seconds"],
                audio_sha256=detector_meta["audio_sha256"],
                identity_row=identity_row,
                opportunities=opportunities,
                candidates=deduplicated,
                raw_candidate_count=len(paired),
            )
        )
    return tuple(sources)


def _validate_partitions(sources: Sequence[Source]) -> dict[str, Any]:
    groups = {
        split: [source.identity_row for source in sources if source.split == split]
        for split in ("validation", "test")
    }
    if not groups["validation"] or not groups["test"]:
        raise ValueError("both validation and test sources are required")
    require_disjoint_groups(groups, include_partition_identity=True)
    for split, rows in groups.items():
        hashes = [str(row["audio_sha256"]) for row in rows]
        if len(hashes) != len(set(hashes)):
            raise ValueError(f"{split} contains duplicate audio hashes")
    return {
        "validation_sources": len(groups["validation"]),
        "test_sources": len(groups["test"]),
        "source_id_hash_and_partition_identity_overlap": 0,
        "duplicate_audio_hashes_within_split": 0,
    }


def _log_binomial_tail(n: int, k: int, probability: float) -> float:
    if probability <= 0:
        return float("-inf")
    if probability >= 1:
        return 0.0
    terms = [
        math.log(math.comb(n, value))
        + value * math.log(probability)
        + (n - value) * math.log1p(-probability)
        for value in range(k, n + 1)
    ]
    maximum = max(terms)
    return maximum + math.log(math.fsum(math.exp(item - maximum) for item in terms))


def binomial_lower_95(successes: int, trials: int) -> float | None:
    """One-sided exact Clopper-Pearson lower confidence bound."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("binomial counts are invalid")
    if trials == 0:
        return None
    if successes == 0:
        return 0.0
    target = math.log(1.0 - CONFIDENCE_LEVEL)
    low, high = 0.0, successes / trials
    for _ in range(80):
        middle = (low + high) / 2.0
        if _log_binomial_tail(trials, successes, middle) < target:
            low = middle
        else:
            high = middle
    return high


def _count_at_or_above(sorted_scores: Sequence[float], threshold: float) -> int:
    return len(sorted_scores) - bisect.bisect_left(sorted_scores, threshold)


def _split_evidence(sources: Sequence[Source], split: str) -> dict[str, Any]:
    selected = [source for source in sources if source.split == split]
    positives = [source for source in selected if source.truth == "positive"]
    negatives = [source for source in selected if source.truth == "negative"]
    opportunities = [item for source in positives for item in source.opportunities]
    if not positives or not opportunities:
        raise ValueError(f"{split} has no positive wake opportunities")
    if not negatives:
        raise ValueError(f"{split} has no continuous negative sources")
    exposure_hours = math.fsum(source.duration_seconds for source in negatives) / 3600.0
    total_duration = math.fsum(source.duration_seconds for source in selected)
    return {
        "sources": selected,
        "positives": positives,
        "negatives": negatives,
        "opportunity_count": len(opportunities),
        "negative_exposure_hours": exposure_hours,
        "total_duration_seconds": total_duration,
    }


def _detector_profile(
    evidence: Mapping[str, Any], detector_threshold: float
) -> dict[str, Any]:
    sources: Sequence[Source] = evidence["sources"]
    invoked = [
        candidate
        for source in sources
        for candidate in source.candidates
        if candidate.detector_score >= detector_threshold
    ]
    opportunity_ids = {
        _opportunity_key(source.source_id, item.opportunity_id)
        for source in evidence["positives"]
        for item in source.opportunities
    }
    detector_hits = {
        candidate.opportunity_id
        for candidate in invoked
        if candidate.opportunity_id in opportunity_ids
    }
    best_verifier_by_opportunity: dict[str, float] = {}
    for candidate in invoked:
        if candidate.opportunity_id not in opportunity_ids:
            continue
        best_verifier_by_opportunity[candidate.opportunity_id] = max(
            best_verifier_by_opportunity.get(candidate.opportunity_id, float("-inf")),
            candidate.verifier_score,
        )
    return {
        "detector_threshold": detector_threshold,
        "invoked_count": len(invoked),
        "negative_invocations": sum(candidate.truth == "negative" for candidate in invoked),
        "detector_hits": detector_hits,
        "best_verifier_by_opportunity": best_verifier_by_opportunity,
        "verifier_scores": sorted(candidate.verifier_score for candidate in invoked),
        "negative_verifier_scores": sorted(
            candidate.verifier_score for candidate in invoked if candidate.truth == "negative"
        ),
    }


def _metrics_from_profile(
    evidence: Mapping[str, Any], profile: Mapping[str, Any], verifier_threshold: float,
    *, include_confidence: bool
) -> dict[str, Any]:
    detector_hits: set[str] = profile["detector_hits"]
    joint_hits = {
        opportunity_id
        for opportunity_id, score in profile["best_verifier_by_opportunity"].items()
        if score >= verifier_threshold
    }
    accepted_count = _count_at_or_above(profile["verifier_scores"], verifier_threshold)
    true_accepts = len(joint_hits)
    false_accepts = accepted_count - true_accepts
    negative_false_accepts = _count_at_or_above(
        profile["negative_verifier_scores"], verifier_threshold
    )
    opportunities = int(evidence["opportunity_count"])
    detector_recall = len(detector_hits) / opportunities
    conditional_recall = len(joint_hits) / len(detector_hits) if detector_hits else 0.0
    joint_recall = len(joint_hits) / opportunities
    precision = true_accepts / accepted_count if accepted_count else 0.0
    exposure_hours = float(evidence["negative_exposure_hours"])
    faph = negative_false_accepts / exposure_hours
    faph_upper = poisson_upper_95(negative_false_accepts, exposure_hours)
    total_seconds = float(evidence["total_duration_seconds"])
    result = {
        "detector_threshold": profile["detector_threshold"],
        "verifier_threshold": verifier_threshold,
        "positive_opportunities": opportunities,
        "detector_detected_opportunities": len(detector_hits),
        "joint_detected_opportunities": len(joint_hits),
        "detector_recall": detector_recall,
        "conditional_verifier_recall": conditional_recall,
        "joint_recall": joint_recall,
        "accepted_events": accepted_count,
        "true_accepts": true_accepts,
        "false_accepts": false_accepts,
        "negative_false_accepts": negative_false_accepts,
        "precision": precision,
        "negative_exposure_hours": exposure_hours,
        "false_accepts_per_hour": faph,
        "false_accepts_per_hour_upper_95": faph_upper,
        "detector_candidates": profile["invoked_count"],
        "detector_candidate_rate_per_second": profile["invoked_count"] / total_seconds,
        "detector_candidate_rate_per_hour": profile["invoked_count"] * 3600.0 / total_seconds,
        "negative_candidate_rate_per_hour": profile["negative_invocations"] / exposure_hours,
        "verifier_invocations": profile["invoked_count"],
        "verifier_invocations_per_second": profile["invoked_count"] / total_seconds,
    }
    if include_confidence:
        result["confidence"] = {
            "level": CONFIDENCE_LEVEL,
            "binomial_method": "one_sided_exact_clopper_pearson_lower",
            "poisson_method": "one_sided_exact_poisson_upper",
            "detector_recall_lower_95": binomial_lower_95(
                len(detector_hits), opportunities
            ),
            "conditional_verifier_recall_lower_95": binomial_lower_95(
                len(joint_hits), len(detector_hits)
            ),
            "joint_recall_lower_95": binomial_lower_95(len(joint_hits), opportunities),
            "precision_lower_95": binomial_lower_95(true_accepts, accepted_count),
            "false_accepts_per_hour_upper_95": faph_upper,
        }
    return result


def _metrics(
    evidence: Mapping[str, Any], detector_threshold: float, verifier_threshold: float
) -> dict[str, Any]:
    return _metrics_from_profile(
        evidence,
        _detector_profile(evidence, detector_threshold),
        verifier_threshold,
        include_confidence=True,
    )


def _detector_thresholds(evidence: Mapping[str, Any]) -> list[float]:
    maxima: list[float] = []
    for source in evidence["positives"]:
        for opportunity in source.opportunities:
            scores = [
                candidate.detector_score
                for candidate in source.candidates
                if candidate.opportunity_id
                == _opportunity_key(source.source_id, opportunity.opportunity_id)
            ]
            if scores:
                maxima.append(max(scores))
    return sorted(set(maxima), reverse=True)


def select_validation_operating_point(
    evidence: Mapping[str, Any], *, detector_recall_target: float,
    cascade_recall_target: float, max_faph: float
) -> dict[str, Any]:
    evaluated = 0
    detector_points = 0
    qualifying_count = 0
    selected: dict[str, Any] | None = None
    opportunities = int(evidence["opportunity_count"])
    def rank(point: Mapping[str, Any]) -> tuple[float, ...]:
        return (
            float(point["precision"]),
            float(point["joint_recall"]),
            float(point["conditional_verifier_recall"]),
            -float(point["false_accepts_per_hour_upper_95"]),
            -float(point["verifier_invocations_per_second"]),
            float(point["verifier_threshold"]),
            float(point["detector_threshold"]),
        )

    for detector_threshold in _detector_thresholds(evidence):
        profile = _detector_profile(evidence, detector_threshold)
        if len(profile["detector_hits"]) / opportunities < detector_recall_target:
            continue
        detector_points += 1
        verifier_thresholds = sorted(
            set(profile["best_verifier_by_opportunity"].values()), reverse=True
        )
        for verifier_threshold in verifier_thresholds:
            evaluated += 1
            point = _metrics_from_profile(
                evidence, profile, verifier_threshold, include_confidence=False
            )
            if (
                point["joint_recall"] >= cascade_recall_target
                and point["false_accepts_per_hour_upper_95"] <= max_faph
            ):
                qualifying_count += 1
                if selected is None or rank(point) > rank(selected):
                    selected = point
    if selected is None:
        return {
            "qualified": False,
            "selection_split": "validation",
            "detector_thresholds_meeting_recall_target": detector_points,
            "joint_operating_points_evaluated": evaluated,
            "qualifying_operating_points": 0,
            "thresholds": None,
            "metrics": None,
        }

    selected = _metrics(
        evidence, selected["detector_threshold"], selected["verifier_threshold"]
    )
    return {
        "qualified": True,
        "selection_split": "validation",
        "detector_thresholds_meeting_recall_target": detector_points,
        "joint_operating_points_evaluated": evaluated,
        "qualifying_operating_points": qualifying_count,
        "thresholds": {
            "detector": selected["detector_threshold"],
            "verifier": selected["verifier_threshold"],
        },
        "metrics": selected,
        "ranking": [
            "precision descending",
            "joint recall descending",
            "conditional verifier recall descending",
            "FAPH upper 95 ascending",
            "verifier invocation rate ascending",
            "verifier threshold descending",
            "detector threshold descending",
        ],
    }


def _check_target(value: float, label: str) -> float:
    value = _finite(value, label)
    if not 0 < value <= 1:
        raise ValueError(f"{label} must be in (0, 1]")
    return value


def evaluate_cascade(
    detector_trace: Path,
    verifier_trace: Path,
    *,
    detector_trace_sha256: str,
    verifier_trace_sha256: str,
    detector_recall_target: float = DEFAULT_DETECTOR_RECALL_TARGET,
    cascade_recall_target: float = DEFAULT_CASCADE_RECALL_TARGET,
    max_faph: float = DEFAULT_MAX_FAPH,
    min_negative_exposure_hours: float = DEFAULT_MIN_NEGATIVE_EXPOSURE_HOURS,
) -> dict[str, Any]:
    detector_recall_target = _check_target(
        detector_recall_target, "detector_recall_target"
    )
    cascade_recall_target = _check_target(
        cascade_recall_target, "cascade_recall_target"
    )
    max_faph = _finite(max_faph, "max_faph")
    min_negative_exposure_hours = _finite(
        min_negative_exposure_hours, "min_negative_exposure_hours"
    )
    if max_faph <= 0 or min_negative_exposure_hours <= 0:
        raise ValueError("FAPH and exposure limits must be positive")

    detector, detector_provenance = _load_trace(
        detector_trace,
        expected_sha256=detector_trace_sha256,
        expected_kind="detector",
    )
    verifier, verifier_provenance = _load_trace(
        verifier_trace,
        expected_sha256=verifier_trace_sha256,
        expected_kind="verifier",
    )
    sources = _pair_sources(detector, verifier)
    leakage = _validate_partitions(sources)
    validation = _split_evidence(sources, "validation")
    test = _split_evidence(sources, "test")
    for split, evidence in (("validation", validation), ("test", test)):
        if evidence["negative_exposure_hours"] < min_negative_exposure_hours:
            raise ValueError(
                f"{split} negative exposure {evidence['negative_exposure_hours']:.6f}h "
                f"is below required {min_negative_exposure_hours:.6f}h"
            )

    selection = select_validation_operating_point(
        validation,
        detector_recall_target=detector_recall_target,
        cascade_recall_target=cascade_recall_target,
        max_faph=max_faph,
    )
    deduplication = {
        split: {
            "raw_candidates": sum(
                source.raw_candidate_count for source in sources if source.split == split
            ),
            "deduplicated_candidates": sum(
                len(source.candidates) for source in sources if source.split == split
            ),
        }
        for split in ("validation", "test")
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "evaluation": "kizz_control_c1_joint_detector_verifier_cascade",
        "qualified": False,
        "failure_reasons": [],
        "configuration": {
            "detector_recall_target": detector_recall_target,
            "cascade_recall_target": cascade_recall_target,
            "max_false_accepts_per_hour_upper_95": max_faph,
            "min_negative_exposure_hours_per_split": min_negative_exposure_hours,
            "candidate_deduplication": "transitive_closed_interval_overlap; highest_detector_score_representative",
        },
        "provenance": {
            "detector_trace": detector_provenance,
            "verifier_trace": verifier_provenance,
        },
        "leakage_checks": leakage,
        "deduplication": deduplication,
        "threshold_selection": selection,
        "validation": selection["metrics"],
        "test": None,
        "protocol": {
            "threshold_fit_split": "validation",
            "test_used_for_threshold_selection": False,
            "test_evaluations": 0,
            "test_scored_once_at_frozen_thresholds": False,
        },
    }
    if not selection["qualified"]:
        report["failure_reasons"].append("validation_operating_point_not_found")
        return report

    thresholds = selection["thresholds"]
    # This is intentionally the only test metric evaluation in the code path.
    test_metrics = _metrics(test, thresholds["detector"], thresholds["verifier"])
    report["test"] = test_metrics
    report["protocol"].update(
        test_evaluations=1, test_scored_once_at_frozen_thresholds=True
    )
    if test_metrics["detector_recall"] < detector_recall_target:
        report["failure_reasons"].append("test_detector_recall_below_target")
    if test_metrics["joint_recall"] < cascade_recall_target:
        report["failure_reasons"].append("test_cascade_recall_below_target")
    if test_metrics["false_accepts_per_hour_upper_95"] > max_faph:
        report["failure_reasons"].append("test_faph_upper_95_above_limit")
    report["qualified"] = not report["failure_reasons"]
    return report


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-trace", type=Path, required=True)
    parser.add_argument("--detector-trace-sha256", required=True)
    parser.add_argument("--verifier-trace", type=Path, required=True)
    parser.add_argument("--verifier-trace-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--detector-recall-target", type=float, default=DEFAULT_DETECTOR_RECALL_TARGET
    )
    parser.add_argument(
        "--cascade-recall-target", type=float, default=DEFAULT_CASCADE_RECALL_TARGET
    )
    parser.add_argument("--max-faph", type=float, default=DEFAULT_MAX_FAPH)
    parser.add_argument(
        "--min-negative-exposure-hours",
        type=float,
        default=DEFAULT_MIN_NEGATIVE_EXPOSURE_HOURS,
    )
    args = parser.parse_args(argv)
    report = evaluate_cascade(
        args.detector_trace,
        args.verifier_trace,
        detector_trace_sha256=args.detector_trace_sha256,
        verifier_trace_sha256=args.verifier_trace_sha256,
        detector_recall_target=args.detector_recall_target,
        cascade_recall_target=args.cascade_recall_target,
        max_faph=args.max_faph,
        min_negative_exposure_hours=args.min_negative_exposure_hours,
    )
    _write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "qualified": report["qualified"],
                "failure_reasons": report["failure_reasons"],
                "thresholds": report["threshold_selection"]["thresholds"],
                "test": report["test"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0 if report["qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
