#!/usr/bin/env python3
"""Fail-closed qualification of the deployed Kizz Control INT8 student.

The validation partition alone selects the threshold.  Held-out evidence is
hash-checked and immutable.  Continuous negatives use one stateful TFLite
interpreter per file and count refractory-separated decoder events.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import soundfile as sf
import tensorflow as tf

from microwakeword.audio.audio_utils import MicroFrontend
from microwakeword.ctc_forward import exhaustive_suffix_forward_score
from microwakeword.kizz_continuous_evaluation import detect_events
from microwakeword.kizz_evaluation_contract import require_disjoint_groups, validate_audio_rows
from microwakeword.kizz_viterbi_decoder import exhaustive_suffix_score
from microwakeword.phoneme_student import (
    compact_phone_contract,
    student_stream_phase_offset_frames,
)
from tools.convert_distilled_student import load_distillation_contract, sha256_file
from tools.distill_kizz_student import student_flags

TARGET_RATE = 16_000
FEATURE_BINS = 40
WINDOW_FRAMES = 260
STRIDE_FRAMES = 3
WINDOW_LENGTHS = (19, 23, 27, 32, 39, 47, 54)
DEFAULT_BETA = 0.0
DEFAULT_REFRACTORY_SECONDS = 1.0
FORWARD_SCORE_BATCH = 256
VITERBI_DECODER_MODULE = Path(__file__).resolve().parents[1] / "microwakeword/kizz_viterbi_decoder.py"
FORWARD_SUM_DECODER_MODULE = Path(__file__).resolve().parents[1] / "microwakeword/ctc_forward.py"


def _payload(path: Path) -> object:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _rows(path: Path) -> list[dict]:
    payload = _payload(path)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = next((payload[k] for k in ("examples", "records", "anchors", "observations", "files", "items") if isinstance(payload.get(k), list)), None)
    else:
        rows = None
    if rows is None or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"manifest has no supported row list: {path}")
    return [dict(row) for row in rows]


def _audio_hash(row: dict) -> str | None:
    return row.get("audio_sha256") or row.get("source_audio_sha256") or row.get("sha256")


def _select_evidence_rows(name: str, rows: list[dict]) -> list[dict]:
    """Select the immutable partition represented by each CLI manifest."""
    if name == "validation":
        rows = [row for row in rows if row.get("split") == "validation"]
        if not rows:
            raise ValueError("validation manifest has no validation split")
        if any("label" not in row or int(row["label"]) not in (0, 1) for row in rows):
            raise ValueError("validation evidence requires binary labels")
    elif name == "test":
        rows = [row for row in rows if row.get("split") == "test"]
        if not rows:
            raise ValueError("test manifest has no test split")
        if any(row.get("label") != 1 for row in rows):
            raise ValueError("test evidence must contain only label=1")
    elif name == "target":
        rows = [row for row in rows if row.get("label") == 1]
        if not rows:
            raise ValueError("target-channel evidence has no label=1 rows")
        if any("label" not in row for row in rows):
            raise ValueError("target-channel evidence must contain only label=1")
    elif name == "false_wakes":
        rows = [row for row in rows if row.get("label") == 0]
        if not rows:
            raise ValueError("false-wake evidence has no label=0 rows")
        if any("label" not in row for row in rows):
            raise ValueError("false-wake evidence must contain only label=0")
    return rows


def _validate_evidence(paths: dict[str, Path]) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    groups = {name: _rows(path) for name, path in paths.items()}
    groups = {name: _select_evidence_rows(name, rows) for name, rows in groups.items()}
    # The continuous lock predates the general evidence schema and uses
    # ``sha256`` plus an absolute path as its stable identity.  Normalize that
    # representation in memory; the on-disk lock remains byte-for-byte bound
    # by the manifest hash reported below.
    for row in groups.get("continuous", []):
        row.setdefault("audio_sha256", row.get("sha256"))
        row.setdefault("source_id", "continuous:" + str(row.get("sha256") or row.get("path")))
    contracts = {}
    for name, rows in groups.items():
        if not rows:
            raise ValueError(f"{name} evidence manifest is empty")
        contracts[name] = validate_audio_rows(rows, group=name, require_locked_anchor=name == "false_wakes")
        if any(row.get("training_eligible") is True for row in rows):
            raise ValueError(f"{name} contains training-eligible evidence")
    validation = groups.get("validation", [])
    heldout = [row for name, rows in groups.items() if name != "validation" for row in rows]
    require_disjoint_groups({"validation": validation, "heldout": heldout})
    require_disjoint_groups({name: rows for name, rows in groups.items() if name != "validation"})
    if "continuous" in groups:
        locked = _payload(paths["continuous"])
        if not isinstance(locked, dict) or locked.get("locked_before_scoring") is not True or locked.get("training_eligible") is not False:
            raise ValueError("continuous manifest is not an immutable, training-ineligible lock")
    return groups, contracts


def _resample(values: np.ndarray, rate: int) -> np.ndarray:
    if rate == TARGET_RATE:
        return values.astype(np.float32, copy=False)
    from scipy.signal import resample_poly
    return np.asarray(resample_poly(values, TARGET_RATE, rate), dtype=np.float32)


def load_pcm(path: Path) -> np.ndarray:
    values, rate = sf.read(path, always_2d=False, dtype="float32")
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 2:
        values = values.mean(axis=1)
    if values.ndim != 1 or not len(values):
        raise ValueError(f"audio is not non-empty mono: {path}")
    return _resample(values, int(rate))


def frontend_features(samples: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float32)
    # The C frontend needs one 30-ms analysis window before it can emit a
    # frame.  Very short locked noise files are valid zero-context evidence;
    # right-pad them exactly as the downstream 2.60-s student window is padded.
    if len(values) < 480:
        values = np.pad(values, (0, 480 - len(values)))
    frontend = MicroFrontend()
    process = getattr(frontend, "process_samples", None) or frontend.ProcessSamples
    pcm = np.clip(values * 32768.0, -32768, 32767).astype("<i2").tobytes()
    frames: list[np.ndarray] = []
    offset = 0
    while offset + 320 <= len(pcm):
        result = process(pcm[offset:offset + 320])
        used = int(getattr(result, "samples_read", 160))
        if used <= 0:
            raise ValueError("C MicroFrontend made no progress")
        if result.features:
            frame_values = np.asarray(result.features, dtype=np.float32)
            if frame_values.size % FEATURE_BINS:
                raise ValueError(
                    f"C MicroFrontend feature count is not divisible by {FEATURE_BINS}"
                )
            frames.append(frame_values.reshape(-1, FEATURE_BINS))
        offset += used * 2
    if not frames:
        raise ValueError("C MicroFrontend produced no frames")
    values = np.concatenate(frames, axis=0)
    if values.ndim != 2 or values.shape[1] != FEATURE_BINS:
        raise ValueError(f"C MicroFrontend emitted {values.shape}")
    return values


def feature_windows(features: np.ndarray) -> Iterable[np.ndarray]:
    if len(features) <= WINDOW_FRAMES:
        padded = np.zeros((WINDOW_FRAMES, FEATURE_BINS), dtype=np.float32)
        padded[:len(features)] = features
        yield padded
        return
    for start in range(0, len(features) - WINDOW_FRAMES + 1, STRIDE_FRAMES):
        yield np.asarray(features[start:start + WINDOW_FRAMES], dtype=np.float32)


def _quantize(values: np.ndarray, detail: dict) -> np.ndarray:
    dtype = np.dtype(detail["dtype"])
    if dtype == np.dtype(np.float32):
        return np.asarray(values, dtype=np.float32)
    if not np.issubdtype(dtype, np.integer):
        raise ValueError(f"unsupported TFLite input dtype: {dtype}")
    scale, zero = detail["quantization"]
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("TFLite input has invalid quantization")
    info = np.iinfo(dtype)
    return np.clip(np.rint(np.asarray(values) / scale + zero), info.min, info.max).astype(dtype)


def _dequantize(values: np.ndarray, detail: dict) -> np.ndarray:
    dtype = np.dtype(detail["dtype"])
    if dtype == np.dtype(np.float32):
        return np.asarray(values, dtype=np.float32)
    if not np.issubdtype(dtype, np.integer):
        raise ValueError(f"unsupported TFLite output dtype: {dtype}")
    scale, zero = detail["quantization"]
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("TFLite output has invalid quantization")
    return (np.asarray(values, dtype=np.float32) - zero) * scale


def _as_sequence(value: Any, output_count: int) -> np.ndarray:
    values = np.asarray(value)
    if values.ndim == 3:
        values = values[0]
    if values.ndim != 2 or values.shape[-1] != output_count:
        raise ValueError(f"streaming model output must be [time, {output_count}], got {values.shape}")
    return values


class DeployedStudent:
    """The deployed internal-state model, with one interpreter per stream."""

    def __init__(
        self,
        artifact: Path,
        *,
        output_frames: int = 66,
        stream_phase_offset_frames: int = 0,
    ) -> None:
        self.artifact = artifact
        self.output_frames = int(output_frames)
        self.stream_phase_offset_frames = int(stream_phase_offset_frames)
        if self.stream_phase_offset_frames < 0:
            raise ValueError("stream phase offset must be non-negative")

    def stream_logits(self, features: np.ndarray, contract: dict) -> np.ndarray:
        flags = student_flags(len(contract["tokens"]))
        interpreter = tf.lite.Interpreter(model_path=str(self.artifact))
        interpreter.allocate_tensors()
        inputs, outputs = interpreter.get_input_details(), interpreter.get_output_details()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("streaming artifact must have exactly one input and output")
        input_detail, output_detail = inputs[0], outputs[0]
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != FEATURE_BINS:
            raise ValueError(f"features must be [frames, {FEATURE_BINS}], got {values.shape}")
        emitted = []
        chunks = []
        if self.stream_phase_offset_frames:
            if self.stream_phase_offset_frames >= flags.stride:
                raise ValueError("stream phase offset must be smaller than stride")
            primer = np.zeros((flags.stride, FEATURE_BINS), dtype=np.float32)
            primer[-self.stream_phase_offset_frames:] = values[:self.stream_phase_offset_frames]
            chunks.append(primer)
        chunks.extend(
            values[offset:offset + flags.stride]
            for offset in range(
                self.stream_phase_offset_frames,
                len(values) - flags.stride + 1,
                flags.stride,
            )
        )
        for chunk in chunks:
            interpreter.set_tensor(input_detail["index"], _quantize(chunk, input_detail)[None, ...])
            interpreter.invoke()
            emitted.append(_as_sequence(_dequantize(interpreter.get_tensor(output_detail["index"]), output_detail), len(contract["tokens"])))
        if not emitted:
            raise ValueError("streaming artifact emitted no output")
        return np.concatenate(emitted, axis=0).astype(np.float32, copy=False)


def _decoder_score(
    logits: np.ndarray,
    contract: dict,
    *,
    beta: float,
    decoder_algorithm: str = "forward_sum_ctc",
) -> float:
    scorer = (
        exhaustive_suffix_forward_score
        if decoder_algorithm == "forward_sum_ctc"
        else exhaustive_suffix_score
        if decoder_algorithm == "max_add_ctc_viterbi"
        else None
    )
    if scorer is None:
        raise ValueError("unsupported student decoder algorithm")
    scored = scorer(
        logits, contract, window_lengths=WINDOW_LENGTHS, beta=beta
    )
    return float(scored.canonical_fit) if scored.eligible else -math.inf


def _forward_sum_batch_scores(
    sequences: Sequence[np.ndarray], contract: dict, *, beta: float
) -> list[float]:
    if len(sequences) == 0:
        return []
    from microwakeword.ctc_forward_accelerated import suffix_forward_sum_scores

    values = np.stack(sequences).astype(np.float32, copy=False)
    scored = suffix_forward_sum_scores(
        values,
        contract,
        window_lengths=WINDOW_LENGTHS,
        beta=beta,
    )
    return [float(value) for value in scored]


def _stream_window_scores(model: Any, features: np.ndarray, contract: dict, *, beta: float, decoder_algorithm: str = "forward_sum_ctc") -> tuple[list[float], list[float]]:
    values = np.asarray(features, dtype=np.float32)
    if len(values) < WINDOW_FRAMES:
        padded = np.zeros((WINDOW_FRAMES, FEATURE_BINS), dtype=np.float32)
        padded[:len(values)] = values
        values = padded
    if hasattr(model, "stream_logits"):
        logits = model.stream_logits(values, contract)
        flags = student_flags(len(contract["tokens"]))
        phase_offset = int(getattr(model, "stream_phase_offset_frames", 0))
        warmup = int(phase_offset > 0) + (WINDOW_FRAMES - phase_offset) // flags.stride - model.output_frames
        if warmup < 0:
            raise ValueError("invalid streaming warmup geometry")
        starts = list(range(0, len(values) - WINDOW_FRAMES + 1, STRIDE_FRAMES)) or [0]
        scores, timestamps, pending = [], [], []
        for start in starts:
            first = start // flags.stride + warmup
            sequence = logits[first:first + model.output_frames]
            if len(sequence) != model.output_frames:
                raise ValueError("streaming artifact did not provide a complete causal window")
            if decoder_algorithm == "forward_sum_ctc":
                pending.append(sequence)
                if len(pending) == FORWARD_SCORE_BATCH:
                    scores.extend(
                        _forward_sum_batch_scores(pending, contract, beta=beta)
                    )
                    pending.clear()
            else:
                scores.append(
                    _decoder_score(
                        sequence,
                        contract,
                        beta=beta,
                        decoder_algorithm=decoder_algorithm,
                    )
                )
            timestamps.append(start * 0.010)
        if pending:
            scores.extend(_forward_sum_batch_scores(pending, contract, beta=beta))
        return scores, timestamps
    scores, timestamps = [], []
    for index, window in enumerate(feature_windows(values)):
        logits = model.score_window(window, contract) if hasattr(model, "score_window") else np.asarray(model(window[None, ...], training=False))[0]
        scores.append(
            _decoder_score(
                logits,
                contract,
                beta=beta,
                decoder_algorithm=decoder_algorithm,
            )
        )
        timestamps.append(index * STRIDE_FRAMES * 0.010)
    return scores, timestamps


def score_features(model: Any, features: np.ndarray, contract: dict, beta: float = DEFAULT_BETA, decoder_algorithm: str = "forward_sum_ctc") -> float:
    scores, _ = _stream_window_scores(
        model,
        features,
        contract,
        beta=beta,
        decoder_algorithm=decoder_algorithm,
    )
    return max(scores, default=-math.inf)


def _false_wake_context(row: dict, samples: np.ndarray, *, context_seconds: float) -> tuple[np.ndarray, dict]:
    metadata_path = Path(str(row.get("metadata_path", ""))).resolve()
    metadata = _payload(metadata_path)
    if not isinstance(metadata, dict):
        raise ValueError("false-wake metadata is not an object")
    if metadata.get("sha256") != _audio_hash(row):
        raise ValueError("false-wake metadata audio hash differs from manifest")
    expected_id = str(row.get("source_id", "")).removeprefix("false-wake:")
    if metadata.get("observation_id") != expected_id:
        raise ValueError("false-wake metadata observation ID differs from manifest")
    trigger = float(metadata.get("pre_wake_ms")) / 1000.0
    if not 0 < trigger <= len(samples) / TARGET_RATE:
        raise ValueError("false-wake trigger offset is outside the recording")
    start = max(0.0, trigger - context_seconds)
    selected = samples[round(start * TARGET_RATE):round(trigger * TARGET_RATE)]
    if not len(selected):
        raise ValueError("false-wake pre-trigger context is empty")
    return selected, {"metadata_path": str(metadata_path), "metadata_sha256": sha256_file(metadata_path), "wake_trigger_seconds": trigger, "context_seconds": context_seconds, "context_start_seconds": start, "context_end_seconds": trigger}


def _estimated_file_memory_bytes(path: Path, *, output_count: int) -> int:
    info = sf.info(str(path))
    if info.channels < 1 or info.samplerate <= 0:
        raise ValueError(f"invalid audio header: {path}")
    samples = int(math.ceil(info.frames * TARGET_RATE / info.samplerate))
    feature_frames = max(1, samples // 160)
    return samples * 4 + feature_frames * FEATURE_BINS * 4 + max(1, feature_frames // STRIDE_FRAMES) * output_count * 4


def score_rows(
    rows: Sequence[dict],
    model: Any,
    contract: dict,
    *,
    beta: float,
    decoder_algorithm: str = "forward_sum_ctc",
    false_wake_context_seconds: float | None = None,
    progress_label: str | None = None,
    progress_interval: int = 100,
    max_file_memory_bytes: int | None = None,
) -> tuple[list[dict], float, float]:
    scored, successful_exposure, attempted_exposure = [], 0.0, 0.0
    for row in rows:
        item = dict(row)
        path = Path(str(row.get("path", ""))).resolve()
        item["path"] = str(path)
        declared = row.get("duration_seconds")
        if declared is not None:
            attempted_exposure += float(declared)
        try:
            if max_file_memory_bytes is not None:
                estimated = _estimated_file_memory_bytes(path, output_count=len(contract["tokens"]))
                if estimated > max_file_memory_bytes:
                    raise MemoryError(f"estimated qualification memory {estimated} exceeds limit {max_file_memory_bytes}: {path}")
            samples = load_pcm(path)
            duration = len(samples) / TARGET_RATE
            if declared is not None and abs(float(declared) - duration) > 1e-3:
                raise ValueError(f"duration differs from manifest: {declared} != {duration}")
            context = None
            if false_wake_context_seconds is not None:
                samples, context = _false_wake_context(row, samples, context_seconds=false_wake_context_seconds)
            scores, timestamps = _stream_window_scores(
                model,
                frontend_features(samples),
                contract,
                beta=beta,
                decoder_algorithm=decoder_algorithm,
            )
            item.update(score=max(scores, default=-math.inf), score_values=scores, score_timestamps_seconds=timestamps, duration_seconds=duration, failure_reasons=[])
            if context is not None:
                # The selected waveform ends exactly at the trigger.  The
                # fixed student input may be zero-padded after that boundary;
                # padding is not post-trigger evidence and must not invalidate
                # the pre-trigger proof.
                context["best_window_is_pre_wake"] = context["context_end_seconds"] <= context["wake_trigger_seconds"] + 1e-6
                item["wake_context"] = context
            successful_exposure += duration
        except Exception as error:  # noqa: BLE001
            item.update(score=None, accepted=False, duration_seconds=declared, failure_reasons=[f"scoring_error:{type(error).__name__}:{error}"])
        scored.append(item)
        if progress_label and (len(scored) % max(1, progress_interval) == 0 or len(scored) == len(rows)):
            print(json.dumps({"group": progress_label, "scored": len(scored), "total": len(rows)}, sort_keys=True), flush=True)
    return scored, successful_exposure, attempted_exposure


def choose_validation_threshold(rows: Sequence[dict], *, min_recall: float, max_faph: float) -> dict:
    failures = [row for row in rows if row.get("score") is None or row.get("failure_reasons")]
    valid = [row for row in rows if row.get("score") is not None and not row.get("failure_reasons")]
    positives = np.asarray([row["score"] for row in valid if int(row.get("label", 0)) == 1], dtype=float)
    negatives = np.asarray([row["score"] for row in valid if int(row.get("label", 0)) == 0], dtype=float)
    exposure = sum(float(row.get("duration_seconds", 0.0)) for row in valid if int(row.get("label", 0)) == 0)
    if not len(positives) or not len(negatives) or exposure <= 0:
        raise ValueError("validation needs positive, negative, and negative exposure rows")
    candidates = sorted(set(float(value) for value in np.concatenate((positives, negatives)))) + [math.inf]
    negative_ceiling = float(np.max(negatives))
    zero_fp_threshold = float(np.nextafter(negative_ceiling, math.inf))
    zero_fp_recall = float(np.mean(positives >= zero_fp_threshold))
    required_true_positives = int(math.ceil(min_recall * len(positives)))
    recall_floor_threshold = float(np.sort(positives)[::-1][required_true_positives - 1])
    false_accepts_at_recall_floor = int(np.sum(negatives >= recall_floor_threshold))
    faph_at_recall_floor = false_accepts_at_recall_floor / (exposure / 3600.0)
    choices = []
    for threshold in candidates:
        tp = int(np.sum(positives >= threshold)); fp = int(np.sum(negatives >= threshold))
        recall = tp / len(positives); faph = fp / (exposure / 3600.0)
        if recall >= min_recall and faph <= max_faph:
            choices.append((threshold, recall, faph, tp, fp))
    if choices:
        threshold, recall, faph, tp, fp = max(choices, key=lambda item: (item[0], item[1]))
        point = {"qualified": True, "threshold": threshold, "recall": recall, "faph": faph, "true_positives": tp, "false_accepts": fp, "negative_exposure_seconds": exposure, "selection": "validation_only"}
    else:
        point = {
            "qualified": False,
            "reason": "no_validation_threshold_meets_recall_and_faph",
            "recall": float(max(np.mean(positives >= value) for value in candidates)),
            "faph": None,
        }
    point.update({
        "zero_false_accept_recall": zero_fp_recall,
        "zero_false_accept_threshold": _json_score(zero_fp_threshold),
        "negative_ceiling": _json_score(negative_ceiling),
        "recall_floor": min_recall,
        "threshold_at_recall_floor": _json_score(recall_floor_threshold),
        "false_accepts_at_recall_floor": false_accepts_at_recall_floor,
        "faph_at_recall_floor": faph_at_recall_floor,
    })
    if failures:
        point.update(qualified=False, reason="validation_scoring_failure", scoring_failures=len(failures))
    return point


def poisson_upper_95(false_accepts: int, exposure_hours: float) -> float:
    if false_accepts < 0 or exposure_hours <= 0:
        raise ValueError("false_accepts must be non-negative and exposure must be positive")
    if false_accepts == 0:
        return -math.log(0.05) / exposure_hours
    lo, hi = 0.0, max(10.0, false_accepts * 4.0)
    for _ in range(100):
        mid = (lo + hi) / 2
        cdf = sum(math.exp(-mid) * mid**k / math.factorial(k) for k in range(false_accepts + 1))
        if cdf > 0.05:
            lo = mid
        else:
            hi = mid
    return hi / exposure_hours


def _apply_threshold(rows: list[dict], threshold: float) -> dict:
    accepted = [row for row in rows if row.get("score") is not None and not row.get("failure_reasons") and float(row["score"]) >= threshold]
    positives = [row for row in rows if int(row.get("label", 0)) == 1]
    return {"count": len(rows), "scoring_failures": sum(bool(row.get("failure_reasons")) for row in rows), "accepted": len(accepted), "positive_count": len(positives), "positive_accepted": sum(row in accepted for row in positives), "recall": (sum(row in accepted for row in positives) / len(positives)) if positives else None, "accepted_ids": [row.get("source_id") or row.get("id") for row in accepted]}


def _json_score(value: object) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _validate_artifact(args: argparse.Namespace, contract: dict, metadata: dict, metadata_hash: str, weights_hash: str) -> tuple[dict, Path, dict]:
    artifact = _payload(args.artifact_metadata)
    if not isinstance(artifact, dict) or int(artifact.get("schema_version", 0)) < 2:
        raise ValueError("INT8 artifact metadata schema is too old")
    if artifact.get("compact_phone_contract") != contract:
        raise ValueError("artifact phone contract drift")
    source = artifact.get("source") or {}
    if source.get("distillation_metadata_sha256") != metadata_hash or source.get("weights_sha256") != weights_hash:
        raise ValueError("artifact is not bound to exact distillation metadata and weights")
    representative = Path(str(source.get("representative_features", ""))).resolve()
    if not representative.is_file() or source.get("representative_features_sha256") != sha256_file(representative):
        raise ValueError("artifact representative-feature provenance drifted")
    decoder = artifact.get("decoder") or {}
    expected_decoder = (metadata.get("decoder") or {}).get("contract") or {}
    decoder_algorithm = expected_decoder.get("algorithm")
    decoder_module = (
        FORWARD_SUM_DECODER_MODULE
        if decoder_algorithm == "forward_sum_ctc"
        else VITERBI_DECODER_MODULE
        if decoder_algorithm == "max_add_ctc_viterbi"
        else None
    )
    expected_type = (
        "deterministic_suffix_forward_sum_ctc"
        if decoder_algorithm == "forward_sum_ctc"
        else "deterministic_suffix_viterbi_ctc"
    )
    if (
        decoder_module is None
        or decoder.get("type") != expected_type
        or decoder.get("algorithm") != decoder_algorithm
        or decoder.get("contract_sha256")
        != (metadata.get("decoder") or {}).get("contract_sha256")
        or decoder.get("distillation_decoder_contract") != expected_decoder
    ):
        raise ValueError("artifact decoder contract is missing or drifted")
    if decoder.get("reference_module_sha256") != sha256_file(decoder_module):
        raise ValueError("artifact decoder reference hash is missing or drifted")
    info = artifact.get("artifact") or {}; path = (args.artifact_metadata.parent / str(info.get("filename", ""))).resolve()
    if path.parent != args.artifact_metadata.parent.resolve() or not path.is_file() or sha256_file(path) != info.get("sha256") or path.stat().st_size != info.get("bytes"):
        raise ValueError("INT8 artifact bytes or hash drifted")
    if (artifact.get("input") or {}).get("dtype") != "int8" or (artifact.get("output") or {}).get("dtype") != "uint8":
        raise ValueError("deployed artifact must have int8 input and uint8 output")
    if int(np.prod((artifact.get("output") or {}).get("shape", []))) != len(contract["tokens"]):
        raise ValueError("INT8 artifact output does not match compact phone contract")
    equivalence = artifact.get("equivalence") or {}
    if not equivalence.get("paths"):
        raise ValueError("artifact is missing equivalence evidence")
    if equivalence.get("limits") != {
        "max_abs": 2.0,
        "max_mean_abs": 0.15,
        "max_decision_mismatch": 0.10,
    }:
        raise ValueError("artifact equivalence limits are missing or drifted")
    timeline = artifact.get("timeline") or {}
    expected_phase = student_stream_phase_offset_frames(
        student_flags(len(contract["tokens"]))
    )
    if timeline.get("stream_phase_offset_frames") != expected_phase:
        raise ValueError("artifact streaming phase offset is missing or drifted")
    if timeline.get("stream_phase_priming") != "zero_prefix_then_observed_prefix":
        raise ValueError("artifact streaming phase priming is missing or drifted")
    return artifact, path, {"path": str(args.artifact_metadata.resolve()), "sha256": sha256_file(args.artifact_metadata)}


def qualify(args: argparse.Namespace) -> dict:
    metadata, contract, metadata_hash, weights_hash = load_distillation_contract(args.distillation_metadata, args.weights)
    artifact, artifact_path, artifact_provenance = _validate_artifact(args, contract, metadata, metadata_hash, weights_hash)
    decoder_contract = metadata["decoder"]["contract"]
    decoder_algorithm = decoder_contract["algorithm"]
    decoder_module = (
        FORWARD_SUM_DECODER_MODULE
        if decoder_algorithm == "forward_sum_ctc"
        else VITERBI_DECODER_MODULE
    )
    manifest_paths = {"validation": args.validation_manifest, "test": args.test_manifest, "target": args.target_channel_manifest, "false_wakes": args.false_wake_manifest}
    if args.continuous_manifest:
        manifest_paths["continuous"] = args.continuous_manifest
    groups, evidence_contracts = _validate_evidence(manifest_paths)
    timeline = artifact.get("timeline") or {}
    model = DeployedStudent(
        artifact_path,
        output_frames=int(timeline.get("output_frames", 66)),
        stream_phase_offset_frames=int(timeline["stream_phase_offset_frames"]),
    )
    validation, valid_exposure, attempted_exposure = score_rows(groups["validation"], model, contract, beta=args.beta, decoder_algorithm=decoder_algorithm)
    point = choose_validation_threshold(validation, min_recall=args.min_recall, max_faph=args.max_faph)
    threshold = float(point.get("threshold", math.inf))
    test, _, _ = score_rows(groups["test"], model, contract, beta=args.beta, decoder_algorithm=decoder_algorithm)
    target, _, _ = score_rows(groups["target"], model, contract, beta=args.beta, decoder_algorithm=decoder_algorithm)
    false_wakes, _, _ = score_rows(groups["false_wakes"], model, contract, beta=args.beta, decoder_algorithm=decoder_algorithm, false_wake_context_seconds=args.false_wake_context_seconds)
    test_result, target_result, false_result = _apply_threshold(test, threshold), _apply_threshold(target, threshold), _apply_threshold(false_wakes, threshold)
    continuous = None
    if args.continuous_manifest:
        rows, _, _ = score_rows(
            groups["continuous"],
            model,
            contract,
            beta=args.beta,
            decoder_algorithm=decoder_algorithm,
            progress_label="continuous",
            progress_interval=args.progress_interval,
            max_file_memory_bytes=args.max_file_memory_mb * 1024 * 1024,
        )
        results, event_count, attempted, scored = [], 0, 0.0, 0.0
        for row in rows:
            attempted += float(row.get("duration_seconds") or 0.0)
            if row.get("failure_reasons"):
                results.append({"source_id": row.get("source_id"), "path": row.get("path"), "duration_seconds": row.get("duration_seconds"), "failure_reasons": row["failure_reasons"], "events": []})
                continue
            duration = float(row["duration_seconds"]); scored += duration
            events = detect_events(row["score_timestamps_seconds"], row["score_values"], threshold, refractory_seconds=args.refractory_seconds)
            event_count += len(events)
            results.append({"source_id": row.get("source_id"), "path": row.get("path"), "duration_seconds": duration, "events": [{"start_seconds": e.start_seconds, "end_seconds": e.end_seconds, "peak_score": e.peak_score, "peak_timestamp_seconds": e.peak_timestamp_seconds} for e in events]})
        hours = scored / 3600.0; upper = poisson_upper_95(event_count, hours) if hours else math.inf
        failures = sum(bool(row.get("failure_reasons")) for row in rows)
        continuous = {"rows": len(rows), "scoring_failures": failures, "attempted_exposure_seconds": attempted, "scored_exposure_seconds": scored, "exposure_hours": hours, "false_accepts": event_count, "faph": event_count / hours if hours else None, "faph_upper_95": upper if hours else None, "refractory_seconds": args.refractory_seconds, "untouched_manifest": True, "results": results, "qualified": not failures and attempted == scored and hours >= args.min_continuous_hours and upper <= args.max_continuous_faph}
    reasons = []
    if not point.get("qualified"): reasons.append("validation_operating_point_not_qualified")
    if any(row.get("failure_reasons") for group in (validation, test, target, false_wakes) for row in group): reasons.append("qualification_audio_scoring_failure")
    if test_result["recall"] is None or test_result["recall"] < args.min_recall: reasons.append("aligned_test_recall_below_minimum")
    if len(target) != args.expected_target_positives: reasons.append("target_channel_count_not_exact")
    if target_result["recall"] is None or target_result["recall"] < args.min_recall: reasons.append("target_channel_recall_below_minimum")
    if len(false_wakes) != args.expected_false_wakes: reasons.append("false_wake_anchor_count_not_exact")
    if false_result["accepted"]: reasons.append("locked_false_wake_accepted")
    if any(not row.get("wake_context", {}).get("best_window_is_pre_wake", False) for row in false_wakes): reasons.append("false_wake_trigger_context_not_proven")
    if continuous is not None and not continuous["qualified"]: reasons.append("continuous_negative_gate_failed")
    report = {"schema_version": 2, "gate_scope": "student_deployment_qualification", "qualified": not reasons, "failure_reasons": reasons, "model": {"weights": str(args.weights.resolve()), "weights_sha256": weights_hash, "distillation_metadata": str(args.distillation_metadata.resolve()), "distillation_metadata_sha256": metadata_hash}, "artifact_metadata": {**artifact_provenance, "artifact_sha256": artifact["artifact"]["sha256"], "artifact_bytes": artifact["artifact"]["bytes"]}, "decoder": {"type": ("deterministic_suffix_forward_sum_ctc" if decoder_algorithm == "forward_sum_ctc" else "deterministic_suffix_viterbi_ctc"), "algorithm": decoder_algorithm, "reference_module": str(decoder_module), "reference_module_sha256": sha256_file(decoder_module), "contract_sha256": metadata["decoder"]["contract_sha256"], "contract": decoder_contract, "beta": args.beta, "window_lengths": list(WINDOW_LENGTHS), "threshold_selection": "validation_only", "streaming": "one_interpreter_per_file", "refractory_seconds": args.refractory_seconds}, "evidence": {name: {"path": str(path.resolve()), "sha256": sha256_file(path), **evidence_contracts[name]} for name, path in manifest_paths.items()}, "compact_phone_contract": contract, "frontend": {"implementation": "repository_pymicro-features", "class": "MicroFrontend", "sample_rate": TARGET_RATE}, "resource_guard": {"mode": "preflight_per_file", "max_file_memory_mb": args.max_file_memory_mb, "continuous_progress_interval": args.progress_interval}, "threshold": point, "exposure": {"validation_negative_attempted_seconds": attempted_exposure, "validation_negative_scored_seconds": valid_exposure}, "counts": {"aligned_test": len(test), "aligned_test_accepted": test_result["accepted"], "target_channel_positives": len(target), "target_channel_accepted": target_result["accepted"], "false_wake_anchors": len(false_wakes), "false_wake_accepted": false_result["accepted"]}, "results": {"validation": _apply_threshold(validation, threshold), "aligned_test": test_result, "target_channel": target_result, "false_wakes": false_result}, "score_summary": {name: [{"source_id": row.get("source_id") or row.get("id"), "label": row.get("label"), "score": _json_score(row.get("score")), "failure_reasons": row.get("failure_reasons", [])} for row in rows] for name, rows in (("validation", validation), ("aligned_test", test), ("target_channel", target), ("false_wakes", false_wakes))}, "continuous_negative": continuous}
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("validation_manifest", "test_manifest", "target_channel_manifest", "false_wake_manifest"):
        parser.add_argument("--" + name.replace("_", "-"), dest=name, type=Path, required=True)
    parser.add_argument("--continuous-manifest", type=Path); parser.add_argument("--weights", type=Path, required=True); parser.add_argument("--distillation-metadata", type=Path, required=True); parser.add_argument("--artifact-metadata", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA); parser.add_argument("--false-wake-context-seconds", type=float, default=2.0); parser.add_argument("--refractory-seconds", type=float, default=DEFAULT_REFRACTORY_SECONDS); parser.add_argument("--min-recall", type=float, default=0.90); parser.add_argument("--max-faph", type=float, default=0.10); parser.add_argument("--expected-target-positives", type=int, default=24); parser.add_argument("--expected-false-wakes", type=int, default=62); parser.add_argument("--min-continuous-hours", type=float, default=100.0); parser.add_argument("--max-continuous-faph", type=float, default=0.10); parser.add_argument("--progress-interval", type=int, default=100); parser.add_argument("--max-file-memory-mb", type=int, default=256)
    args = parser.parse_args(argv)
    if args.false_wake_context_seconds <= 0 or args.refractory_seconds < 0 or args.progress_interval < 1 or args.max_file_memory_mb < 1: parser.error("invalid context/refractory/progress/memory setting")
    args.output.parent.mkdir(parents=True, exist_ok=True); report = qualify(args)
    print(json.dumps({"qualified": report["qualified"], "threshold": report["threshold"], "counts": report["counts"], "continuous_negative": report["continuous_negative"]}, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 2


if __name__ == "__main__": raise SystemExit(main())
