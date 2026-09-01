#!/usr/bin/env python3
"""Build a detector-conditioned Kizz Control verifier dataset.

The verifier never sees arbitrary full clips.  Every training/evaluation row is
cut around an event emitted by one frozen detector score trace.  Positive clips
without an event remain in ``detector_misses`` and never become verifier rows.

The output deliberately follows the reusable part of the ranked-decision
trainer contract: ``corpus.json`` and fixed ``features.npy`` with per-example
``split`` and ``label`` fields.  Additional fixed arrays make detector-score and
trigger geometry available to a candidate-specific trainer without reparsing
JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SPLITS = ("train", "validation", "test")
HASH_FIELDS = (
    "audio_sha256",
    "sha256",
    "source_audio_sha256",
    "parent_source_audio_sha256",
    "feature_sha256",
)
SOURCE_HASH_FIELDS = HASH_FIELDS[:-1]
IDENTITY_FIELDS = (
    "source_id",
    "parent_source_id",
    "speaker_id",
    "session_id",
    "ancestry_id",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return payload


def _examples(payload: Mapping[str, Any], path: Path) -> list[dict[str, Any]]:
    rows = payload.get("examples", payload.get("records"))
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"{path}: expected examples or records list")
    return [dict(row) for row in rows]


def _binding_path(binding: Mapping[str, Any], name: str) -> Path:
    raw = binding.get("path")
    expected = binding.get("sha256")
    if not isinstance(raw, str) or not raw or not isinstance(expected, str):
        raise ValueError(f"detector {name} binding requires path and sha256")
    path = Path(raw).expanduser().resolve()
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"detector {name} hash drift: {path}")
    return path


def _verified_detector(trace_payload: Mapping[str, Any]) -> dict[str, Any]:
    detector = trace_payload.get("detector")
    if not isinstance(detector, dict):
        raise ValueError("detector trace manifest requires detector provenance")
    verified: dict[str, Any] = {}
    for name in ("artifact", "config", "threshold"):
        binding = detector.get(name)
        if not isinstance(binding, dict):
            raise ValueError(f"detector provenance requires {name} binding")
        path = _binding_path(binding, name)
        verified[name] = {
            "path": str(path),
            "sha256": str(binding["sha256"]),
        }
    threshold = detector["threshold"].get("value")
    if not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)):
        raise ValueError("detector threshold binding requires finite value")
    verified["threshold"]["value"] = float(threshold)
    policy = detector.get("event_policy", "recorded_events")
    if policy not in {"recorded_events", "threshold_regions"}:
        raise ValueError("detector event_policy must be recorded_events or threshold_regions")
    geometry = detector.get("score_geometry")
    if not isinstance(geometry, dict):
        raise ValueError("detector score_geometry is required")
    stride = geometry.get("feature_stride_frames")
    offset = geometry.get("feature_offset_frames")
    hop_ms = geometry.get("feature_hop_ms")
    if not isinstance(stride, int) or stride < 1:
        raise ValueError("feature_stride_frames must be a positive integer")
    if not isinstance(offset, int) or offset < 0:
        raise ValueError("feature_offset_frames must be a nonnegative integer")
    if not isinstance(hop_ms, (int, float)) or float(hop_ms) <= 0:
        raise ValueError("feature_hop_ms must be positive")
    verified["event_policy"] = policy
    verified["score_geometry"] = {
        "feature_stride_frames": stride,
        "feature_offset_frames": offset,
        "feature_hop_ms": float(hop_ms),
    }
    return verified


def _require_binding(
    payload: Mapping[str, Any], key: str, path: Path, *, observed_sha256: str | None = None
) -> dict[str, str]:
    binding = payload.get(key)
    if not isinstance(binding, dict):
        raise ValueError(f"detector trace manifest requires {key} binding")
    expected = binding.get("sha256")
    actual = observed_sha256 or sha256_file(path)
    if expected != actual:
        raise ValueError(f"detector trace {key} binding differs from {path}")
    bound_path = binding.get("path")
    if bound_path and Path(str(bound_path)).expanduser().resolve() != path.resolve():
        raise ValueError(f"detector trace {key} path differs from {path}")
    return {"path": str(path.resolve()), "sha256": actual}


def _identity_values(row: Mapping[str, Any]) -> set[str]:
    values: set[str] = set()
    for key in IDENTITY_FIELDS:
        value = row.get(key)
        if value not in (None, ""):
            values.add(str(value))
    for key in ("ancestry_ids", "parent_source_ids"):
        raw = row.get(key, [])
        if isinstance(raw, list):
            values.update(str(value) for value in raw if value not in (None, ""))
    return values


def _hash_values(
    row: Mapping[str, Any], *, include_feature: bool = True
) -> set[str]:
    values: set[str] = set()
    for key in HASH_FIELDS if include_feature else SOURCE_HASH_FIELDS:
        value = row.get(key)
        if isinstance(value, str) and value:
            values.add(value)
    raw = row.get("ancestry_sha256", [])
    if isinstance(raw, list):
        values.update(str(value) for value in raw if value)
    return values


def _verify_source_rows(rows: Sequence[dict[str, Any]], features: np.ndarray) -> None:
    if not rows:
        raise ValueError("source manifest is empty")
    feature_count = len(features)
    if len(rows) != feature_count:
        raise ValueError("source manifest and feature array example counts differ")
    seen_ids: set[str] = set()
    seen_indexes: set[int] = set()
    for ordinal, row in enumerate(rows):
        source_id = row.get("source_id")
        split = row.get("split")
        label = row.get("label")
        if not isinstance(source_id, str) or not source_id or source_id in seen_ids:
            raise ValueError("source_id values must be unique nonempty strings")
        if split not in SPLITS or label not in (0, 1, False, True):
            raise ValueError(f"{source_id}: invalid split or binary label")
        index = row.get("feature_index", ordinal)
        if not isinstance(index, int) or not 0 <= index < feature_count:
            raise ValueError(f"{source_id}: invalid feature_index")
        if index in seen_indexes:
            raise ValueError("feature_index values must be unique")
        row["feature_index"] = index
        declared_feature_hash = row.get("feature_sha256")
        observed_feature_hash = _feature_sha256(np.asarray(features[index]))
        if declared_feature_hash != observed_feature_hash:
            raise ValueError(f"{source_id}: source feature hash drift")
        row["_observed_feature_sha256"] = observed_feature_hash
        seen_ids.add(source_id)
        seen_indexes.add(index)
        hashes = _hash_values(row)
        if not hashes:
            raise ValueError(f"{source_id}: exact source/feature hash is required")
        audio_path = row.get("path")
        expected = row.get("audio_sha256", row.get("sha256"))
        if audio_path and expected:
            path = Path(str(audio_path)).expanduser().resolve()
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"{source_id}: source audio hash drift")


def _check_split_leakage(rows: Sequence[Mapping[str, Any]]) -> None:
    for left_index, left in enumerate(SPLITS):
        left_rows = [row for row in rows if row["split"] == left]
        left_ids = set().union(*(_identity_values(row) for row in left_rows)) if left_rows else set()
        left_hashes = (
            set().union(*(_hash_values(row, include_feature=False) for row in left_rows))
            if left_rows
            else set()
        )
        for right in SPLITS[left_index + 1 :]:
            right_rows = [row for row in rows if row["split"] == right]
            right_ids = set().union(*(_identity_values(row) for row in right_rows)) if right_rows else set()
            right_hashes = (
                set().union(
                    *(_hash_values(row, include_feature=False) for row in right_rows)
                )
                if right_rows
                else set()
            )
            identity_overlap = left_ids & right_ids
            hash_overlap = left_hashes & right_hashes
            if identity_overlap:
                raise ValueError(
                    f"{left}/{right} identity overlap: {sorted(identity_overlap)[:3]}"
                )
            if hash_overlap:
                raise ValueError(
                    f"{left}/{right} hash overlap: {sorted(hash_overlap)[:3]}"
                )


def _locked_evidence(path: Path | None) -> tuple[set[str], set[str], dict[str, str] | None]:
    if path is None:
        return set(), set(), None
    payload = _load_json(path)
    if (
        payload.get("gate_scope") != "locked_untouched_continuous_negative_corpus"
        or payload.get("locked_before_scoring") is not True
    ):
        raise ValueError("holdout manifest is not the locked untouched continuous corpus")
    rows = _examples(payload, path)
    hashes = (
        set().union(*(_hash_values(row, include_feature=False) for row in rows))
        if rows
        else set()
    )
    identities = set().union(*(_identity_values(row) for row in rows)) if rows else set()
    return hashes, identities, {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _check_locked_holdout(
    rows: Sequence[Mapping[str, Any]], locked_hashes: set[str], locked_ids: set[str]
) -> None:
    for row in rows:
        if row["split"] != "train":
            continue
        if row.get("locked_deployment_anchor") or row.get("locked_holdout"):
            raise ValueError(f"{row['source_id']}: locked holdout may not enter train")
        if (
            _hash_values(row, include_feature=False) & locked_hashes
            or _identity_values(row) & locked_ids
        ):
            raise ValueError(f"{row['source_id']}: locked holdout may not enter train")


def threshold_region_events(scores: Sequence[float], threshold: float) -> list[dict[str, Any]]:
    """Return one deterministic peak from each contiguous above-threshold region."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("detector scores must be one-dimensional and finite")
    events: list[dict[str, Any]] = []
    start = 0
    while start < len(values):
        if values[start] < threshold:
            start += 1
            continue
        stop = start + 1
        while stop < len(values) and values[stop] >= threshold:
            stop += 1
        peak = start + int(np.argmax(values[start:stop]))
        events.append({"score_frame_index": peak, "score": float(values[peak])})
        start = stop
    return events


def _recorded_events(
    trace: Mapping[str, Any], scores: np.ndarray, threshold: float
) -> list[dict[str, Any]]:
    raw_events = trace.get("events")
    if not isinstance(raw_events, list):
        raise ValueError(f"{trace.get('source_id')}: recorded_events requires events list")
    events: list[dict[str, Any]] = []
    seen: set[int] = set()
    for raw in raw_events:
        if not isinstance(raw, dict):
            raise ValueError("detector event must be an object")
        index = raw.get("score_frame_index", raw.get("frame_index"))
        if not isinstance(index, int) or not 0 <= index < len(scores) or index in seen:
            raise ValueError(f"{trace.get('source_id')}: invalid/duplicate detector event")
        observed = float(scores[index])
        declared = raw.get("score", observed)
        if not isinstance(declared, (int, float)) or not math.isclose(
            float(declared), observed, rel_tol=1e-6, abs_tol=1e-7
        ):
            raise ValueError(f"{trace.get('source_id')}: detector event score drift")
        if observed < threshold:
            raise ValueError(f"{trace.get('source_id')}: detector event is below threshold")
        event = dict(raw)
        event["score_frame_index"] = index
        event["score"] = observed
        events.append(event)
        seen.add(index)
    return sorted(events, key=lambda item: (item["score_frame_index"], -item["score"]))


def _trace_events(
    trace: Mapping[str, Any], detector: Mapping[str, Any]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    scores = np.asarray(trace.get("scores"), dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.all(np.isfinite(scores)):
        raise ValueError(f"{trace.get('source_id')}: scores must be a finite nonempty vector")
    threshold = float(detector["threshold"]["value"])
    if detector["event_policy"] == "recorded_events":
        events = _recorded_events(trace, scores, threshold)
    else:
        events = threshold_region_events(scores, threshold)
    return scores, events


def _feature_trigger_frame(
    trace: Mapping[str, Any], score_index: int, detector: Mapping[str, Any]
) -> int:
    indexes = trace.get("feature_frame_indexes")
    if indexes is not None:
        if not isinstance(indexes, list) or score_index >= len(indexes):
            raise ValueError(f"{trace.get('source_id')}: invalid feature_frame_indexes")
        value = indexes[score_index]
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{trace.get('source_id')}: invalid feature frame index")
        return value
    geometry = detector["score_geometry"]
    return int(geometry["feature_offset_frames"]) + score_index * int(
        geometry["feature_stride_frames"]
    )


def candidate_window(
    features: np.ndarray,
    trigger_frame: int,
    pre_context_frames: int,
    post_context_frames: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Extract a fixed detector-triggered window with deterministic zero padding."""
    if features.ndim != 2:
        raise ValueError("one source feature tensor must be [frames,bins]")
    if trigger_frame < 0 or pre_context_frames < 0 or post_context_frames < 0:
        raise ValueError("trigger and context frame counts must be nonnegative")
    if trigger_frame >= len(features):
        raise ValueError("detector trigger maps beyond source feature frames")
    length = pre_context_frames + 1 + post_context_frames
    start = trigger_frame - pre_context_frames
    stop = trigger_frame + post_context_frames + 1
    source_start = max(0, start)
    source_stop = min(len(features), stop)
    left = source_start - start
    right = stop - source_stop
    output = np.zeros((length, features.shape[1]), dtype=features.dtype)
    if source_stop > source_start:
        output[left : length - right if right else length] = features[source_start:source_stop]
    return output, {
        "requested_start_frame": start,
        "requested_stop_frame_exclusive": stop,
        "source_start_frame": source_start,
        "source_stop_frame_exclusive": source_stop,
        "left_padding_frames": left,
        "right_padding_frames": right,
    }


def _copy_lineage(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "source_id",
        "parent_source_id",
        "speaker_id",
        "voice_id",
        "provider",
        "session_id",
        "ancestry_id",
        "ancestry_ids",
        "source_group",
        "audio_sha256",
        "sha256",
        "source_audio_sha256",
        "parent_source_audio_sha256",
        "feature_sha256",
    )
    return {key: row[key] for key in keys if key in row}


def _hard_negative_key(row: Mapping[str, Any]) -> tuple[float, str]:
    return (-float(row["detector_score"]), str(row["candidate_id"]))


def _select_hard_negatives(
    rows: Sequence[dict[str, Any]], top_k: int, group_by: str
) -> list[dict[str, Any]]:
    if top_k < 1:
        raise ValueError("hard_negative_top_k must be positive")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if group_by == "session":
            value = row.get("session_id")
            if value in (None, ""):
                raise ValueError("session hard-negative grouping requires session_id")
            key = str(value)
        elif group_by == "source":
            key = str(row["parent_source_id"])
        else:
            raise ValueError("hard_negative_group_by must be source or session")
        grouped[key].append(row)
    selected: list[dict[str, Any]] = []
    for key in sorted(grouped):
        selected.extend(sorted(grouped[key], key=_hard_negative_key)[:top_k])
    return selected


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        with open(temporary, "wb") as handle:
            np.save(handle, values, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def build_candidate_verifier_dataset(
    source_manifest: Path,
    source_features: Path,
    detector_traces: Path,
    output: Path,
    *,
    locked_holdout_manifest: Path | None = None,
    pre_context_frames: int = 220,
    post_context_frames: int = 39,
    hard_negative_top_k: int = 4,
    hard_negative_group_by: str = "source",
) -> dict[str, Any]:
    """Build and write the deterministic candidate-conditioned dataset."""
    if pre_context_frames < 0 or post_context_frames < 0:
        raise ValueError("context frame counts must be nonnegative")
    if locked_holdout_manifest is None:
        raise ValueError("locked_holdout_manifest is required")
    source_payload = _load_json(source_manifest)
    trace_payload = _load_json(detector_traces)
    features = np.load(source_features, mmap_mode="r", allow_pickle=False)
    if features.ndim != 3 or features.shape[0] < 1 or features.shape[2] < 1:
        raise ValueError("source features must be [examples,frames,bins]")
    feature_sha = sha256_file(source_features)
    source_binding = _require_binding(trace_payload, "source_manifest", source_manifest)
    feature_binding = _require_binding(
        trace_payload, "source_features", source_features, observed_sha256=feature_sha
    )
    declared_array_hash = source_payload.get("array_sha256", {}).get(source_features.name)
    if declared_array_hash is not None and declared_array_hash != feature_sha:
        raise ValueError("source manifest feature-array hash differs")
    detector = _verified_detector(trace_payload)
    rows = _examples(source_payload, source_manifest)
    _verify_source_rows(rows, features)
    _check_split_leakage(rows)
    locked_hashes, locked_ids, locked_binding = _locked_evidence(locked_holdout_manifest)
    _check_locked_holdout(rows, locked_hashes, locked_ids)

    traces = _examples(trace_payload, detector_traces)
    trace_by_id: dict[str, dict[str, Any]] = {}
    for trace in traces:
        source_id = trace.get("source_id")
        if not isinstance(source_id, str) or source_id in trace_by_id:
            raise ValueError("detector traces require unique source_id values")
        trace_by_id[source_id] = trace
    source_ids = {str(row["source_id"]) for row in rows}
    if set(trace_by_id) != source_ids:
        missing = sorted(source_ids - set(trace_by_id))
        extra = sorted(set(trace_by_id) - source_ids)
        raise ValueError(f"detector trace/source mismatch; missing={missing[:3]} extra={extra[:3]}")

    candidate_rows: list[dict[str, Any]] = []
    candidate_features: dict[str, np.ndarray] = {}
    misses: list[dict[str, Any]] = []
    exposure_seconds: Counter[str] = Counter()
    exposure_seconds_by_label: Counter[tuple[str, int]] = Counter()
    raw_counts: Counter[tuple[str, int]] = Counter()
    for row in sorted(rows, key=lambda item: str(item["source_id"])):
        source_id = str(row["source_id"])
        trace = trace_by_id[source_id]
        for key, expected in (
            ("feature_index", row["feature_index"]),
            ("split", row["split"]),
            ("label", int(row["label"])),
            ("source_feature_sha256", row["_observed_feature_sha256"]),
        ):
            if trace.get(key) != expected:
                raise ValueError(f"{source_id}: detector trace {key} drift")
        scores, events = _trace_events(trace, detector)
        source_feature = np.asarray(features[int(row["feature_index"])])
        if not np.issubdtype(source_feature.dtype, np.number) or not np.all(
            np.isfinite(source_feature)
        ):
            raise ValueError(f"{source_id}: source features must be finite numeric values")
        duration = row.get("duration_seconds")
        if duration is None:
            duration = len(source_feature) * detector["score_geometry"]["feature_hop_ms"] / 1000.0
        if not isinstance(duration, (int, float)) or float(duration) <= 0:
            raise ValueError(f"{source_id}: duration_seconds must be positive")
        exposure_seconds[str(row["split"])] += float(duration)
        exposure_seconds_by_label[(str(row["split"]), int(row["label"]))] += float(
            duration
        )
        raw_counts[(str(row["split"]), int(row["label"]))] += len(events)
        if int(row["label"]) == 1 and not events:
            misses.append(
                {
                    **_copy_lineage(row),
                    "split": row["split"],
                    "label": 1,
                    "detector_miss": True,
                    "maximum_detector_score": float(np.max(scores)),
                    "detector_threshold": detector["threshold"]["value"],
                    "score_frame_count": len(scores),
                }
            )
            continue
        for event_ordinal, event in enumerate(events):
            score_frame = int(event["score_frame_index"])
            feature_frame = _feature_trigger_frame(trace, score_frame, detector)
            window, geometry = candidate_window(
                source_feature, feature_frame, pre_context_frames, post_context_frames
            )
            event_fingerprint = hashlib.sha256(
                f"{source_id}\0{score_frame}\0{float(event['score']):.17g}".encode()
            ).hexdigest()
            candidate_id = f"{source_id}::detector-candidate::{event_fingerprint[:20]}"
            candidate = {
                "candidate_id": candidate_id,
                "source_id": candidate_id,
                "parent_source_id": source_id,
                "split": row["split"],
                "label": int(row["label"]),
                "detector_conditioned": True,
                "detector_event_ordinal": event_ordinal,
                "detector_score": float(event["score"]),
                "detector_score_frame_index": score_frame,
                "detector_feature_frame_index": feature_frame,
                "window": geometry,
                "duration_seconds": (
                    (pre_context_frames + 1 + post_context_frames)
                    * detector["score_geometry"]["feature_hop_ms"]
                    / 1000.0
                ),
                **{
                    key: value
                    for key, value in _copy_lineage(row).items()
                    if key not in {"source_id", "parent_source_id"}
                },
            }
            if row.get("parent_source_id") not in (None, ""):
                candidate["source_parent_source_id"] = str(row["parent_source_id"])
            preserved_event = dict(event)
            preserved_event["score_frame_index"] = score_frame
            preserved_event["score"] = float(event["score"])
            candidate["detector_event"] = preserved_event
            candidate_rows.append(candidate)
            candidate_features[candidate_id] = window

    positives = [row for row in candidate_rows if row["label"] == 1]
    raw_negatives = [row for row in candidate_rows if row["label"] == 0]
    raw_train_negatives = [row for row in raw_negatives if row["split"] == "train"]
    heldout_negatives = [row for row in raw_negatives if row["split"] != "train"]
    train_negatives = _select_hard_negatives(
        raw_train_negatives, hard_negative_top_k, hard_negative_group_by
    )
    negatives = train_negatives + heldout_negatives
    selected = sorted(
        positives + negatives,
        key=lambda row: (
            SPLITS.index(str(row["split"])),
            -int(row["label"]),
            str(row["candidate_id"]),
        ),
    )
    if not selected:
        raise ValueError("frozen detector emitted no verifier candidates")
    for index, row in enumerate(selected):
        row["feature_index"] = index

    fixed_features = np.stack(
        [candidate_features[str(row["candidate_id"])] for row in selected]
    ).astype(np.float16, copy=False)
    for index, row in enumerate(selected):
        row["candidate_feature_sha256"] = hashlib.sha256(
            np.ascontiguousarray(fixed_features[index]).tobytes()
        ).hexdigest()
    labels = np.asarray([row["label"] for row in selected], dtype=np.int8)
    detector_scores = np.asarray(
        [row["detector_score"] for row in selected], dtype=np.float32
    )
    score_frames = np.asarray(
        [row["detector_score_frame_index"] for row in selected], dtype=np.int32
    )
    feature_frames = np.asarray(
        [row["detector_feature_frame_index"] for row in selected], dtype=np.int32
    )

    output.mkdir(parents=True, exist_ok=True)
    arrays = {
        "features.npy": fixed_features,
        "labels.npy": labels,
        "detector_scores.npy": detector_scores,
        "detector_score_frames.npy": score_frames,
        "detector_feature_frames.npy": feature_frames,
    }
    for name, values in arrays.items():
        _atomic_npy(output / name, values)
    array_hashes = {name: sha256_file(output / name) for name in arrays}

    selected_counts = Counter((str(row["split"]), int(row["label"])) for row in selected)
    miss_counts = Counter(str(row["split"]) for row in misses)
    counts_by_split: dict[str, Any] = {}
    for split in SPLITS:
        seconds = float(exposure_seconds[split])
        negative_seconds = float(exposure_seconds_by_label[(split, 0)])
        positive_sources = sum(
            1 for row in rows if row["split"] == split and int(row["label"]) == 1
        )
        raw_total = raw_counts[(split, 0)] + raw_counts[(split, 1)]
        counts_by_split[split] = {
            "source_examples": sum(1 for row in rows if row["split"] == split),
            "exposure_seconds": seconds,
            "raw_detector_candidates": raw_total,
            "raw_positive_candidates": raw_counts[(split, 1)],
            "raw_negative_candidates": raw_counts[(split, 0)],
            "selected_positive_candidates": selected_counts[(split, 1)],
            "selected_negative_candidates": selected_counts[(split, 0)],
            "detector_missed_positives": miss_counts[split],
            "detector_positive_source_recall": (
                (positive_sources - miss_counts[split]) / positive_sources
                if positive_sources
                else None
            ),
            "raw_candidate_rate_per_second": raw_total / seconds if seconds else 0.0,
            "raw_candidate_rate_per_hour": raw_total * 3600.0 / seconds if seconds else 0.0,
            "negative_exposure_seconds": negative_seconds,
            "raw_negative_candidate_rate_per_hour": (
                raw_counts[(split, 0)] * 3600.0 / negative_seconds
                if negative_seconds
                else 0.0
            ),
        }
    report: dict[str, Any] = {
        "schema_version": 1,
        "recipe": "kizz_control_candidate_conditioned_verifier_v1",
        "candidate_condition": "frozen_detector_trigger_only",
        "trainer_compatibility": {
            "core_format": "distill_kizz_ranked_decision_student corpus.json/features.npy",
            "input_shape_matches_existing_ranked_decision_student": list(fixed_features.shape[1:])
            == [260, 40],
            "teacher_specific_side_inputs_still_required_by_existing_trainer": True,
        },
        "input_shape": list(fixed_features.shape[1:]),
        "window_contract": {
            "pre_context_frames": pre_context_frames,
            "trigger_frames": 1,
            "post_context_frames": post_context_frames,
            "padding": "zero",
        },
        "hard_negative_selection": {
            "ranking": "detector_score_descending_then_candidate_id",
            "top_k": hard_negative_top_k,
            "group_by": hard_negative_group_by,
            "scope": "train_only",
            "raw_training_count": len(raw_train_negatives),
            "selected_training_count": len(train_negatives),
            "heldout_candidates_unfiltered": len(heldout_negatives),
        },
        "detector": detector,
        "bindings": {
            "source_manifest": source_binding,
            "source_features": feature_binding,
            "detector_traces": {
                "path": str(detector_traces.resolve()),
                "sha256": sha256_file(detector_traces),
            },
            "locked_holdout": locked_binding,
        },
        "manifests": {"source": source_binding},
        "counts": {
            "selected_candidates": len(selected),
            "selected_positives": int(np.sum(labels == 1)),
            "selected_negatives": int(np.sum(labels == 0)),
            "detector_missed_positives": len(misses),
            "by_split": counts_by_split,
        },
        "detector_misses": sorted(
            misses, key=lambda row: (SPLITS.index(str(row["split"])), str(row["source_id"]))
        ),
        "examples": selected,
        "array_sha256": array_hashes,
    }
    _atomic_bytes(output / "corpus.json", _canonical_bytes(report))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-features", type=Path, required=True)
    parser.add_argument("--detector-traces", type=Path, required=True)
    parser.add_argument("--locked-holdout-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pre-context-frames", type=int, default=220)
    parser.add_argument("--post-context-frames", type=int, default=39)
    parser.add_argument("--hard-negative-top-k", type=int, default=4)
    parser.add_argument(
        "--hard-negative-group-by", choices=("source", "session"), default="source"
    )
    args = parser.parse_args(argv)
    report = build_candidate_verifier_dataset(
        args.source_manifest,
        args.source_features,
        args.detector_traces,
        args.output,
        locked_holdout_manifest=args.locked_holdout_manifest,
        pre_context_frames=args.pre_context_frames,
        post_context_frames=args.post_context_frames,
        hard_negative_top_k=args.hard_negative_top_k,
        hard_negative_group_by=args.hard_negative_group_by,
    )
    print(json.dumps(report["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
