#!/usr/bin/env python3
"""Append detector-conditioned candidates from the development device corpus.

This extension deliberately does not call ``validate_device_corpus``: that
validator opens every capture, including test captures.  Here, only train
captures are eligible for audio access.  Validation and test rows are recorded
as quarantined metadata, and their paths are never opened or hashed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import tempfile
import wave
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.evaluate_kizz_int8_continuous_cascade import TFLiteRuntime
from tools.mine_kizz_librispeech_hard_negatives import (
    FEATURE_BINS,
    WINDOW_FRAMES,
    _binding,
    _feature_hash,
    _mine_file,
    sha256_file,
)
from tools.simulate_kizz_int8_cascade import load_firmware_artifact
from tools.trace_kizz_ordered_state_detector import (
    _threshold_from_report,
    _validate_artifact,
)
from tools.train_kizz_candidate_verifier import (
    SPLITS,
    _atomic_bytes,
    _atomic_npy,
    _canonical_bytes,
    _verify_split_disjointness,
    load_verified_dataset,
)


TRUTHS = {"positive", "hard_negative", "ambient_negative"}
NONTRAIN_SPLITS = {"validation", "test"}
ARRAY_NAMES = (
    "features.npy",
    "labels.npy",
    "detector_scores.npy",
    "detector_feature_frames.npy",
    "detector_score_frames.npy",
)
SHA256_LENGTH = 64


def _require_sha256(value: object, label: str) -> str:
    digest = str(value or "")
    if len(digest) != SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return digest


def _absolute_bindings(value: Any, relative_to: Path) -> Any:
    if isinstance(value, Mapping):
        copied = {
            key: _absolute_bindings(child, relative_to)
            for key, child in value.items()
        }
        if "path" in copied and "sha256" in copied:
            path = Path(str(copied["path"])).expanduser()
            if not path.is_absolute():
                path = relative_to / path
            copied["path"] = str(path.resolve())
        return copied
    if isinstance(value, list):
        return [_absolute_bindings(child, relative_to) for child in value]
    return copy.deepcopy(value)


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"device corpus is not readable JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("device corpus must be a JSON object")
    captures = payload.get("captures")
    if not isinstance(captures, list):
        raise ValueError("device corpus captures must be a list")
    return payload


def _qualified_capture_ids(
    report_path: Path, device_corpus: Path, capture_ids: set[str]
) -> set[str]:
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"device quality report is not readable JSON: {report_path}") from error
    if not isinstance(report, dict):
        raise ValueError("device quality report must be a JSON object")
    if (
        report.get("kind")
        != "kizz_control_teacher_adaptation_device_replay_quality"
        or report.get("gate_scope") != "train_only_target_channel_positive_quality"
    ):
        raise ValueError("device quality report has the wrong contract")
    inputs = report.get("inputs")
    if (
        not isinstance(inputs, Mapping)
        or inputs.get("corpus_sha256") != sha256_file(device_corpus)
    ):
        raise ValueError("device quality report is not bound to the device corpus")
    results = report.get("results")
    if not isinstance(results, list):
        raise ValueError("device quality report results are missing")
    decisions: dict[str, bool] = {}
    for index, result in enumerate(results):
        if not isinstance(result, Mapping):
            raise ValueError(f"device quality result[{index}] is malformed")
        capture_id = result.get("capture_id")
        qualified = result.get("qualified")
        if (
            not isinstance(capture_id, str)
            or capture_id in decisions
            or not isinstance(qualified, bool)
        ):
            raise ValueError("device quality report has malformed or duplicate decisions")
        decisions[capture_id] = qualified
    if set(decisions) != capture_ids:
        raise ValueError("device quality report does not cover the device corpus")
    qualified_ids = {capture_id for capture_id, accepted in decisions.items() if accepted}
    if not qualified_ids:
        raise ValueError("device quality report accepted no captures")
    return qualified_ids


def _capture_lock_reason(capture: Mapping[str, Any]) -> str | None:
    if capture.get("locked_holdout") is True:
        return "locked_holdout"
    if capture.get("locked_deployment_anchor") is True:
        return "locked_deployment_anchor"
    if capture.get("deployment_anchor") is True:
        return "deployment_anchor"
    return None


def _validate_capture_metadata(
    capture: Mapping[str, Any], *, index: int, manifest_root: Path
) -> tuple[str, str, Path, str, int, float]:
    prefix = f"capture[{index}]"
    capture_id = capture.get("capture_id")
    if not isinstance(capture_id, str) or not capture_id:
        raise ValueError(f"{prefix} requires a nonempty capture_id")
    truth = capture.get("truth")
    if truth not in TRUTHS:
        raise ValueError(f"{capture_id}: unsupported truth {truth!r}")
    split = capture.get("split")
    if split not in SPLITS:
        raise ValueError(f"{capture_id}: unsupported split {split!r}")
    for field in ("speaker_id", "session_id", "source", "phrase"):
        if not isinstance(capture.get(field), str) or not capture[field]:
            raise ValueError(f"{capture_id}: requires nonempty {field}")
    if not isinstance(capture.get("detected"), bool):
        raise ValueError(f"{capture_id}: detected must be boolean")

    raw_path = capture.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{capture_id}: path must be a nonempty string")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{capture_id}: path must remain inside device corpus")

    declared_hash = capture.get("sha256")
    if declared_hash is None:
        declared_hash = capture.get("audio_sha256")
    declared_hash = _require_sha256(declared_hash, f"{capture_id} audio hash")
    if capture.get("sha256") is not None and capture.get("audio_sha256") is not None:
        if capture["sha256"] != capture["audio_sha256"]:
            raise ValueError(f"{capture_id}: sha256 and audio_sha256 disagree")

    samples = capture.get("samples")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise ValueError(f"{capture_id}: samples must be a positive integer")
    declared_duration = capture.get("duration_seconds")
    if declared_duration is not None:
        if (
            isinstance(declared_duration, bool)
            or not isinstance(declared_duration, (int, float))
            or not math.isfinite(float(declared_duration))
            or float(declared_duration) <= 0
        ):
            raise ValueError(f"{capture_id}: duration_seconds must be positive and finite")
        duration_hint = float(declared_duration)
    else:
        duration_hint = samples / 16_000.0
    return (
        capture_id,
        truth,
        manifest_root / relative,
        declared_hash,
        samples,
        duration_hint,
    )


def _select_train_captures(
    manifest: Mapping[str, Any], manifest_path: Path, *, target_phrase: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    captures = manifest["captures"]
    seen: set[str] = set()
    train: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    for index, raw in enumerate(captures):
        if not isinstance(raw, dict):
            raise ValueError(f"capture[{index}] must be a JSON object")
        capture_id, truth, path, declared_hash, samples, duration_hint = (
            _validate_capture_metadata(
                raw, index=index, manifest_root=manifest_path.parent.resolve()
            )
        )
        if capture_id in seen:
            raise ValueError(f"duplicate capture_id: {capture_id}")
        seen.add(capture_id)
        split = str(raw["split"])
        lock_reason = _capture_lock_reason(raw)
        if split == "train" and lock_reason is not None:
            raise ValueError(f"{capture_id}: {lock_reason} may not be consumed")
        normalized = dict(raw)
        normalized.update(
            {
                "capture_id": capture_id,
                "truth": truth,
                "path": str(path),
                "declared_sha256": declared_hash,
                "declared_samples": samples,
                "declared_duration_seconds": duration_hint,
            }
        )
        phrase_matches = " ".join(str(raw["phrase"]).casefold().split()) == " ".join(
            target_phrase.casefold().split()
        )
        if split == "train" and truth == "positive" and not phrase_matches:
            quarantine.append(
                {
                    "capture_id": capture_id,
                    "split": split,
                    "truth": truth,
                    "reason": "positive_phrase_mismatch",
                    "declared_phrase": raw["phrase"],
                    "target_phrase": target_phrase,
                    "path": str(path),
                }
            )
        elif split == "train":
            train.append(normalized)
        else:
            quarantine.append(
                {
                    "capture_id": capture_id,
                    "split": split,
                    "truth": truth,
                    "reason": lock_reason or "non_train_split",
                    "path": str(path),
                }
            )
    return train, quarantine


def _validate_train_audio(capture: Mapping[str, Any]) -> tuple[str, float]:
    path = Path(str(capture["path"]))
    if not path.is_file():
        raise ValueError(f"{capture['capture_id']}: audio is missing: {path}")
    observed_hash = sha256_file(path)
    if observed_hash != capture["declared_sha256"]:
        raise ValueError(f"{capture['capture_id']}: audio SHA-256 does not match")
    try:
        with wave.open(str(path), "rb") as audio:
            if (
                audio.getnchannels() != 1
                or audio.getsampwidth() != 2
                or audio.getframerate() != 16_000
                or audio.getcomptype() != "NONE"
            ):
                raise ValueError(
                    f"{capture['capture_id']}: audio must be mono 16 kHz signed-16 PCM WAV"
                )
            observed_samples = audio.getnframes()
            duration = observed_samples / audio.getframerate()
    except (OSError, wave.Error) as error:
        raise ValueError(f"{capture['capture_id']}: invalid WAV audio") from error
    if observed_samples != capture["declared_samples"]:
        raise ValueError(f"{capture['capture_id']}: sample count/duration drift")
    if not math.isclose(
        duration,
        float(capture["declared_duration_seconds"]),
        rel_tol=0.0,
        abs_tol=0.001,
    ):
        raise ValueError(f"{capture['capture_id']}: duration_seconds drift")
    return observed_hash, duration


def _candidate_id(capture_id: str, ordinal: int, score: float, feature_hash: str) -> str:
    material = f"{capture_id}\0{ordinal}\0{score:.17g}\0{feature_hash}".encode()
    fingerprint = hashlib.sha256(material).hexdigest()[:20]
    return f"device-corpus:{capture_id}::detector-candidate::{fingerprint}"


def _candidate_row(
    capture: Mapping[str, Any],
    *,
    audio_hash: str,
    duration: float,
    label: int,
    ordinal: int,
    score: float,
    trigger: int,
    feature_hash: str,
) -> dict[str, Any]:
    capture_id = str(capture["capture_id"])
    source_id = f"device-corpus:{capture_id}"
    candidate_id = _candidate_id(capture_id, ordinal, score, feature_hash)
    source = str(capture["source"])
    semantic = "wake_word" if label == 1 else str(capture["truth"])
    return {
        "source_id": candidate_id,
        "candidate_id": candidate_id,
        "parent_source_id": source_id,
        "source_parent_source_id": source_id,
        "speaker_id": str(capture["speaker_id"]),
        "session_id": str(capture["session_id"]),
        "ancestry_id": source_id,
        "audio_sha256": audio_hash,
        "source_audio_sha256": audio_hash,
        "parent_source_audio_sha256": audio_hash,
        "source_group": f"device_corpus_{source}",
        "semantic_label": semantic,
        "provider": f"device_corpus:{source}",
        "split": "train",
        "source_manifest_split": "train",
        "capture_id": capture_id,
        "label": label,
        "duration_seconds": duration,
        "detector_conditioned": True,
        "detector_score": float(score),
        "detector_feature_frame_index": int(trigger),
        "detector_score_frame_index": int(trigger),
        "detector_event_ordinal": int(ordinal),
        "detector_event": {
            "feature_frame_index": int(trigger),
            "feature_time_seconds": float(trigger) * 0.01,
            "score": float(score),
            "score_frame_index": int(trigger),
        },
        "window": {
            "requested_start_frame": int(trigger) - 259,
            "requested_stop_frame_exclusive": int(trigger) + 1,
            "source_start_frame": max(0, int(trigger) - 259),
            "source_stop_frame_exclusive": int(trigger) + 1,
            "left_padding_frames": max(0, 259 - int(trigger)),
            "right_padding_frames": 0,
        },
        "candidate_feature_sha256": feature_hash,
        "feature_sha256": feature_hash,
    }


def _load_frame_array(base: Any, name: str) -> np.ndarray:
    path = base.root / name
    if not path.is_file():
        raise ValueError(f"base dataset lacks required array: {path}")
    values = np.load(path, allow_pickle=False)
    if values.shape != (len(base.rows),):
        raise ValueError(f"base dataset {name} shape differs from corpus rows")
    return np.asarray(values)


def _stable_binding(staged_path: Path, final_path: Path) -> dict[str, Any]:
    """Bind staged bytes to their final path before the atomic directory rename."""
    return {
        "path": str(final_path.resolve()),
        "sha256": sha256_file(staged_path),
        "bytes": staged_path.stat().st_size,
    }


def _update_counts(
    corpus: dict[str, Any],
    *,
    source_count: int,
    exposure_seconds: float,
    negative_exposure_seconds: float,
    raw_candidates: int,
    raw_positive_candidates: int,
    raw_negative_candidates: int,
    appended_positive: int,
    appended_negative: int,
    detector_misses: int,
    top_k_per_file: int,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    counts = corpus.get("counts")
    if not isinstance(counts, dict) or not isinstance(counts.get("by_split"), dict):
        raise ValueError("base corpus lacks counts.by_split")
    train = counts["by_split"].get("train")
    if not isinstance(train, dict):
        raise ValueError("base corpus lacks train counts")
    for field, value in (
        ("source_examples", source_count),
        ("exposure_seconds", exposure_seconds),
        ("negative_exposure_seconds", negative_exposure_seconds),
        ("raw_detector_candidates", raw_candidates),
        ("raw_positive_candidates", raw_positive_candidates),
        ("raw_negative_candidates", raw_negative_candidates),
        ("selected_positive_candidates", appended_positive),
        ("selected_negative_candidates", appended_negative),
        ("detector_missed_positives", detector_misses),
    ):
        old = train.get(field, 0)
        if not isinstance(old, (int, float)) or isinstance(old, bool):
            raise ValueError(f"base corpus train count {field} is malformed")
        train[field] = old + value
    train["detector_positive_source_recall"] = None
    train["raw_candidate_rate_per_second"] = (
        train["raw_detector_candidates"] / train["exposure_seconds"]
        if train["exposure_seconds"]
        else 0.0
    )
    train["raw_candidate_rate_per_hour"] = train["raw_candidate_rate_per_second"] * 3600.0
    train["raw_negative_candidate_rate_per_hour"] = (
        train["raw_negative_candidates"] * 3600.0 / train["negative_exposure_seconds"]
        if train["negative_exposure_seconds"]
        else 0.0
    )

    policy = corpus.get("hard_negative_selection")
    if not isinstance(policy, dict):
        raise ValueError("base corpus lacks hard_negative_selection")
    raw_training = policy.get("raw_training_count")
    if not isinstance(raw_training, int) or raw_training < 0:
        raise ValueError("base corpus raw_training_count is malformed")
    policy["raw_training_count"] = raw_training + raw_negative_candidates
    policy["selected_training_count"] = sum(
        int(row["split"] == "train" and int(row["label"]) == 0) for row in rows
    )
    policy["top_k"] = max(int(policy.get("top_k", 1)), top_k_per_file)
    corpus["hard_negative_selection"] = policy
    counts["selected_candidates"] = len(rows)
    counts["selected_positives"] = sum(int(int(row["label"]) == 1) for row in rows)
    counts["selected_negatives"] = sum(int(int(row["label"]) == 0) for row in rows)
    counts["detector_missed_positives"] = int(counts.get("detector_missed_positives", 0)) + detector_misses


def extend(
    base_dataset: Path,
    base_corpus_sha256: str,
    device_corpus: Path,
    detector_metadata: Path,
    detector_model: Path,
    detector_threshold_report: Path,
    output: Path,
    top_k_per_file: int = 4,
    target_phrase: str = "Kizz Control",
    device_quality_report: Path | None = None,
) -> dict[str, Any]:
    if top_k_per_file < 1:
        raise ValueError("top_k_per_file must be positive")
    target_phrase = " ".join(str(target_phrase).split())
    if not target_phrase:
        raise ValueError("target_phrase must be nonempty")
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")

    base = load_verified_dataset(
        base_dataset, expected_corpus_sha256=base_corpus_sha256
    )
    device_corpus = device_corpus.expanduser().resolve()
    manifest = _load_manifest(device_corpus)
    train_captures, quarantine = _select_train_captures(
        manifest, device_corpus, target_phrase=target_phrase
    )
    quality_report_binding = None
    if device_quality_report is not None:
        device_quality_report = device_quality_report.expanduser().resolve()
        qualified_ids = _qualified_capture_ids(
            device_quality_report,
            device_corpus,
            {str(capture["capture_id"]) for capture in manifest["captures"]},
        )
        rejected = [
            capture for capture in train_captures
            if str(capture["capture_id"]) not in qualified_ids
        ]
        train_captures = [
            capture for capture in train_captures
            if str(capture["capture_id"]) in qualified_ids
        ]
        quarantine.extend(
            {
                "capture_id": capture["capture_id"],
                "split": capture["split"],
                "truth": capture["truth"],
                "reason": "device_quality_gate_rejected",
                "path": capture["path"],
            }
            for capture in rejected
        )
        quality_report_binding = _binding(device_quality_report)

    detector_metadata = detector_metadata.expanduser().resolve()
    detector_model = detector_model.expanduser().resolve()
    threshold_report = detector_threshold_report.expanduser().resolve()
    _, topology, detector_contract = _validate_artifact(
        detector_metadata, detector_model
    )
    threshold, threshold_provenance = _threshold_from_report(
        threshold_report, topology
    )
    detector = load_firmware_artifact(detector_metadata, "detector")
    runtime = TFLiteRuntime(detector_model, detector)

    rows = [copy.deepcopy(row) for row in base.rows]
    existing_hashes = {
        str(row["candidate_feature_sha256"])
        for row in rows
        if row.get("candidate_feature_sha256")
    }
    new_features: list[np.ndarray] = []
    new_scores: list[float] = []
    new_feature_frames: list[int] = []
    new_score_frames: list[int] = []
    source_ledger: list[dict[str, Any]] = []
    detector_miss_rows: list[dict[str, Any]] = []
    total_seconds = 0.0
    negative_seconds = 0.0
    raw_candidates = 0
    raw_positive_candidates = 0
    raw_negative_candidates = 0
    appended_positive = 0
    appended_negative = 0
    detector_misses = 0
    duplicates = 0

    for capture in train_captures:
        audio_hash, duration = _validate_train_audio(capture)
        total_seconds += duration
        label = 1 if capture["truth"] == "positive" else 0
        if label == 0:
            negative_seconds += duration
        requested_top_k = 1 if label == 1 else top_k_per_file
        candidates, frame_count, hop_count = _mine_file(
            Path(capture["path"]),
            Path(capture["path"]).parent,
            runtime,
            topology,
            detector_contract,
            threshold,
            top_k=requested_top_k,
        )
        selected_candidates = list(candidates[:requested_top_k])
        raw_count = len(selected_candidates)
        raw_candidates += raw_count
        if label == 1:
            raw_positive_candidates += raw_count
        else:
            raw_negative_candidates += raw_count
        if label == 1 and not selected_candidates:
            detector_misses += 1
            miss = {
                "source_id": f"device-corpus:{capture['capture_id']}",
                "capture_id": capture["capture_id"],
                "split": "train",
                "label": 1,
                "truth": "positive",
                "detector_miss": True,
                "audio_sha256": audio_hash,
                "duration_seconds": duration,
            }
            detector_miss_rows.append(miss)

        appended = 0
        for ordinal, candidate in enumerate(selected_candidates):
            try:
                score, trigger, feature = candidate
                score = float(score)
                trigger = int(trigger)
                feature = np.asarray(feature, dtype=np.float32)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{capture['capture_id']}: detector candidate is malformed"
                ) from error
            if not math.isfinite(score) or trigger < 0:
                raise ValueError(
                    f"{capture['capture_id']}: detector candidate score/frame is malformed"
                )
            if feature.shape != (WINDOW_FRAMES, FEATURE_BINS) or not np.all(
                np.isfinite(feature)
            ):
                raise ValueError(f"{capture['capture_id']}: detector feature contract drift")
            feature_values = feature.astype(base.features.dtype, copy=False)
            feature_hash = _feature_hash(feature_values)
            if feature_hash in existing_hashes:
                duplicates += 1
                continue
            existing_hashes.add(feature_hash)
            row = _candidate_row(
                capture,
                audio_hash=audio_hash,
                duration=duration,
                label=label,
                ordinal=ordinal,
                score=score,
                trigger=trigger,
                feature_hash=feature_hash,
            )
            rows.append(row)
            new_features.append(feature_values)
            new_scores.append(score)
            new_feature_frames.append(trigger)
            new_score_frames.append(trigger)
            appended += 1
            if label == 1:
                appended_positive += 1
            else:
                appended_negative += 1

        source_ledger.append(
            {
                "capture_id": capture["capture_id"],
                "path": capture["path"],
                "audio_sha256": audio_hash,
                "duration_seconds": duration,
                "samples": capture["declared_samples"],
                "truth": capture["truth"],
                "label": label,
                "split": "train",
                "candidate_count": raw_count,
                "appended_candidate_count": appended,
                "duplicates_skipped": raw_count - appended,
                "detector_miss": bool(label == 1 and not selected_candidates),
                "frontend_feature_frames": int(frame_count),
                "detector_hops": int(hop_count),
            }
        )

    for index, row in enumerate(rows):
        if row.get("feature_index") != index:
            if index < len(base.rows):
                raise ValueError("base candidate feature order changed")
            row["feature_index"] = index
    _verify_split_disjointness(rows)

    base_features = np.asarray(base.features)
    base_labels = np.asarray(base.labels)
    base_scores = np.asarray(base.detector_scores)
    base_feature_frames = _load_frame_array(base, "detector_feature_frames.npy")
    base_score_frames = _load_frame_array(base, "detector_score_frames.npy")
    empty_features = np.empty(
        (0, *base_features.shape[1:]), dtype=base_features.dtype
    )
    arrays = {
        "features.npy": np.concatenate(
            [base_features, np.asarray(new_features, dtype=base_features.dtype).reshape(
                (-1, *base_features.shape[1:])
            ) if new_features else empty_features]
        ),
        "labels.npy": np.concatenate(
            [base_labels, np.asarray([row["label"] for row in rows[len(base.rows):]], dtype=base_labels.dtype)]
        ),
        "detector_scores.npy": np.concatenate(
            [base_scores, np.asarray(new_scores, dtype=base_scores.dtype)]
        ),
        "detector_feature_frames.npy": np.concatenate(
            [base_feature_frames, np.asarray(new_feature_frames, dtype=base_feature_frames.dtype)]
        ),
        "detector_score_frames.npy": np.concatenate(
            [base_score_frames, np.asarray(new_score_frames, dtype=base_score_frames.dtype)]
        ),
    }
    if len(arrays["features.npy"]) != len(rows):
        raise ValueError("extension arrays and corpus rows differ")

    corpus = _absolute_bindings(base.corpus, base.root)
    corpus["examples"] = rows
    corpus.setdefault("bindings", {})["base_candidate_corpus"] = _binding(
        base.corpus_path
    )
    corpus["bindings"]["device_corpus"] = _binding(device_corpus)
    if quality_report_binding is not None:
        corpus["bindings"]["device_quality_report"] = quality_report_binding
    corpus["detector"] = {
        "artifact": _binding(detector_model),
        "config": _binding(detector_metadata),
        "metadata": _binding(detector_metadata),
        "threshold_report": _binding(threshold_report),
        "threshold": threshold,
        "threshold_provenance": threshold_provenance,
    }
    _update_counts(
        corpus,
        source_count=len(train_captures),
        exposure_seconds=total_seconds,
        negative_exposure_seconds=negative_seconds,
        raw_candidates=raw_candidates,
        raw_positive_candidates=raw_positive_candidates,
        raw_negative_candidates=raw_negative_candidates,
        appended_positive=appended_positive,
        appended_negative=appended_negative,
        detector_misses=detector_misses,
        top_k_per_file=top_k_per_file,
        rows=rows,
    )
    corpus["detector_misses"] = list(corpus.get("detector_misses", [])) + detector_miss_rows
    corpus["bindings"]["device_corpus_extension_ledger"] = {
        "path": "device-corpus-candidate-extension-ledger.json",
        "sha256": "pending",
    }
    corpus["bindings"]["device_corpus_extension_provenance"] = {
        "path": "provenance.json",
        "sha256": "pending",
    }
    corpus["device_corpus_extension"] = {
        "schema_version": 1,
        "base_rows": len(base.rows),
        "appended_rows": len(rows) - len(base.rows),
        "top_k_negative_per_file": top_k_per_file,
        "positive_top_k_per_file": 1,
        "train_only_audio_access": True,
        "quarantined_non_train_captures": len(quarantine),
        "detector_misses": detector_misses,
        "duplicates_skipped": duplicates,
        "locked_holdout_used": False,
        "locked_deployment_anchor_used": False,
        "quality_gate_required": device_quality_report is not None,
        "quality_gate_rejected": sum(
            row.get("reason") == "device_quality_gate_rejected" for row in quarantine
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for name, values in arrays.items():
            _atomic_npy(temporary / name, values)
        ledger = {
            "schema_version": 1,
            "kind": "kizz_device_corpus_candidate_extension_ledger",
            "device_corpus": _binding(device_corpus),
            "device_quality_report": quality_report_binding,
            "detector": {
                "metadata": _binding(detector_metadata),
                "model": _binding(detector_model),
                "threshold_report": _binding(threshold_report),
                "threshold": threshold,
                "threshold_provenance": threshold_provenance,
            },
            "selection_policy": {
                "train_only": True,
                "target_phrase": target_phrase,
                "positive_phrase_match": "casefolded_whitespace_normalized_exact_match",
                "positive_top_k": 1,
                "negative_top_k": top_k_per_file,
                "truth_mapping": {
                    "positive": 1,
                    "hard_negative": 0,
                    "ambient_negative": 0,
                },
                "test_audio_opened": False,
                "test_audio_hashed": False,
                "locked_holdout_read": False,
                "deployment_anchor_read": False,
            },
            "files": source_ledger,
            "quarantined_captures": quarantine,
            "detector_misses": detector_miss_rows,
            "counts": {
                "train_captures": len(train_captures),
                "quarantined_captures": len(quarantine),
                "raw_candidates": raw_candidates,
                "raw_positive_candidates": raw_positive_candidates,
                "raw_negative_candidates": raw_negative_candidates,
                "appended_positive_candidates": appended_positive,
                "appended_negative_candidates": appended_negative,
                "duplicates_skipped": duplicates,
                "detector_misses": detector_misses,
                "exposure_seconds": total_seconds,
                "negative_exposure_seconds": negative_seconds,
            },
        }
        _atomic_bytes(
            temporary / "device-corpus-candidate-extension-ledger.json",
            _canonical_bytes(ledger),
        )
        provenance = {
            "schema_version": 1,
            "kind": "kizz_device_corpus_candidate_extension_provenance",
            "immutable_base_candidate_corpus": _binding(base.corpus_path),
            "expected_base_corpus_sha256": base_corpus_sha256,
            "device_corpus": _binding(device_corpus),
            "detector": ledger["detector"],
            "output_contract": {
                "recipe": corpus["recipe"],
                "candidate_condition": corpus["candidate_condition"],
                "input_shape": list(arrays["features.npy"].shape[1:]),
                "base_rows_preserved_as_prefix": True,
                "candidate_feature_deduplication": "sha256(float-array-bytes)",
            },
            "selection": corpus["device_corpus_extension"],
        }
        _atomic_bytes(temporary / "provenance.json", _canonical_bytes(provenance))

        ledger_binding = _stable_binding(
            temporary / "device-corpus-candidate-extension-ledger.json",
            output / "device-corpus-candidate-extension-ledger.json",
        )
        provenance_binding = _stable_binding(
            temporary / "provenance.json", output / "provenance.json"
        )
        corpus["bindings"]["device_corpus_extension_ledger"] = ledger_binding
        corpus["bindings"]["device_corpus_extension_provenance"] = provenance_binding
        corpus["array_sha256"] = {
            name: sha256_file(temporary / name) for name in ARRAY_NAMES
        }
        _atomic_bytes(temporary / "corpus.json", _canonical_bytes(corpus))
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {output}")
        os.rename(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    result_hash = sha256_file(output / "corpus.json")
    load_verified_dataset(output, expected_corpus_sha256=result_hash)
    return {
        "output": str(output),
        "corpus_sha256": result_hash,
        "base_rows": len(base.rows),
        "rows": len(rows),
        "appended_rows": len(rows) - len(base.rows),
        "appended_positive_candidates": appended_positive,
        "appended_negative_candidates": appended_negative,
        "detector_misses": detector_misses,
        "duplicates_skipped": duplicates,
        "quarantined_captures": len(quarantine),
    }


# Match the naming convention of the existing candidate-mining tools.
mine = extend


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--base-corpus-sha256", required=True)
    parser.add_argument("--device-corpus", type=Path, required=True)
    parser.add_argument("--device-quality-report", type=Path)
    parser.add_argument("--detector-metadata", type=Path, required=True)
    parser.add_argument("--detector-model", type=Path, required=True)
    parser.add_argument("--detector-threshold-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k-per-file", type=int, default=4)
    parser.add_argument("--target-phrase", default="Kizz Control")
    args = parser.parse_args(argv)
    print(json.dumps(extend(**vars(args)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
