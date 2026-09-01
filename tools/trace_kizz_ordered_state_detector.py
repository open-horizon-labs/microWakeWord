#!/usr/bin/env python3
"""Trace the deployed Kizz Control ordered-state INT8 detector.

The tracer is a provenance boundary between conversion and candidate-verifier
training.  It accepts only the schema-v2 stateful streaming artifact contract,
resets detector state between every source example, and records finite generic
ordered-state prefix scores plus their exact source-feature frame coordinates.
It does not qualify the detector or cascade for deployment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from microwakeword.ordered_state import (
    OrderedStateTopology,
    ordered_state_sequence_score_numpy,
)
from microwakeword.wake_phrase import KIZZ_CONTROL


ARTIFACT_KIND = "kizz_control_ordered_state_detector_streaming_int8"
MODEL_FILENAME = "kizz_control_detector_ordered_state_streaming_int8.tflite"
VERIFIER_ARTIFACT_KIND = (
    "kizz_control_ordered_state_candidate_verifier_streaming_int8"
)
VERIFIER_MODEL_FILENAME = (
    "kizz_control_ordered_state_candidate_verifier_streaming_int8.tflite"
)
TRACE_RECIPE = "kizz_control_ordered_state_deployed_int8_trace_v1"
SPLITS = ("train", "validation", "test")
EXPECTED_INPUT_SHAPE = (1, 3, 40)
EXPECTED_OUTPUT_SHAPE = (1, 1, 12)
EXPECTED_DECODER_ARGUMENTS = {
    "from_logits": True,
    "state_evidence_floor": None,
    "self_loop_probability": 0.6,
    "next_state_probability": 0.4,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def feature_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not readable JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _resolve(raw: object, anchor: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} path is missing")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = anchor / path
    return path.resolve()


def _canonical_sha(raw: object, label: str) -> str:
    if (
        not isinstance(raw, str)
        or len(raw) != 64
        or any(character not in "0123456789abcdef" for character in raw)
    ):
        raise ValueError(f"{label} SHA-256 is invalid")
    return raw


def _verify_binding(
    value: Mapping[str, Any], anchor: Path, label: str
) -> dict[str, str]:
    path = _resolve(value.get("path"), anchor, label)
    expected = _canonical_sha(value.get("sha256"), label)
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} hash drift: expected {expected}, got {actual}")
    return {"path": str(path), "sha256": actual}


def _verify_nested_bindings(value: object, anchor: Path, label: str) -> None:
    """Verify every conventional path/SHA binding in a metadata subtree."""
    if isinstance(value, Mapping):
        has_path = "path" in value
        has_sha = "sha256" in value
        if has_path != has_sha:
            raise ValueError(f"{label} has an incomplete path/hash binding")
        if has_path:
            _verify_binding(value, anchor, label)
        for key, child in value.items():
            if key not in {"path", "sha256"}:
                _verify_nested_bindings(child, anchor, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _verify_nested_bindings(child, anchor, f"{label}[{index}]")


def _finite_number(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _quantization_contract(
    value: object, dtype: str, label: str
) -> tuple[float, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} quantization contract is invalid")
    scale = _finite_number(value[0], f"{label} quantization scale")
    zero = value[1]
    if scale <= 0 or not isinstance(zero, int):
        raise ValueError(f"{label} quantization contract is invalid")
    limits = np.iinfo(np.dtype(dtype))
    if not limits.min <= zero <= limits.max:
        raise ValueError(f"{label} zero point is outside {dtype}")
    return scale, zero


def _topology(metadata: Mapping[str, Any]) -> OrderedStateTopology:
    value = metadata.get("topology")
    if not isinstance(value, Mapping):
        raise ValueError("artifact topology is missing")
    topology = OrderedStateTopology(
        tuple(str(phone) for phone in value.get("phones", ())),
        int(value.get("states_per_phone", 0)),
    )
    expected_names = list(topology.state_names)
    if (
        topology.phones != tuple(KIZZ_CONTROL.phones)
        or topology.states_per_phone != 1
        or topology.state_count != 12
        or value.get("phrase_id") != KIZZ_CONTROL.phrase_id
        or value.get("state_count") != topology.state_count
        or value.get("state_names") != expected_names
        or value.get("background_index") != topology.background_index
        or value.get("silence_index") != topology.silence_index
        or value.get("first_ordered_state_index")
        != topology.first_ordered_state_index
    ):
        raise ValueError("artifact ordered-state topology contract drift")
    return topology


def _validate_artifact(
    metadata_path: Path, model_path: Path
) -> tuple[dict[str, Any], OrderedStateTopology, dict[str, Any]]:
    metadata = _load_json(metadata_path, "firmware artifact")
    detector_role = metadata.get("kind") == ARTIFACT_KIND
    verifier_role = metadata.get("kind") == VERIFIER_ARTIFACT_KIND
    expected_role = (
        "permissive_detector_candidate_generator"
        if detector_role
        else "detector_conditioned_ordered_state_candidate_verifier"
    )
    if (
        metadata.get("schema_version") != 2
        or not (detector_role or verifier_role)
        or metadata.get("student_role") != expected_role
        or metadata.get("deployment_qualification") is not False
    ):
        raise ValueError("firmware artifact is not a supported non-deployment ordered-state model")

    artifact = metadata.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ValueError("firmware artifact model binding is missing")
    expected_filename = artifact.get("filename")
    role_filename = MODEL_FILENAME if detector_role else VERIFIER_MODEL_FILENAME
    if expected_filename != role_filename or model_path.name != expected_filename:
        raise ValueError("firmware artifact filename contract drift")
    expected_hash = _canonical_sha(artifact.get("sha256"), "TFLite artifact")
    if not model_path.is_file() or sha256_file(model_path) != expected_hash:
        raise ValueError("TFLite artifact hash drift")
    if artifact.get("bytes") != model_path.stat().st_size:
        raise ValueError("TFLite artifact byte count drift")

    source = metadata.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("firmware artifact source provenance is missing")
    _verify_nested_bindings(source, metadata_path.parent, "artifact.source")

    topology = _topology(metadata)
    decoder = metadata.get("decoder")
    if not isinstance(decoder, Mapping):
        raise ValueError("artifact decoder contract is missing")
    arguments = decoder.get("arguments")
    if (
        decoder.get("algorithm") != "ordered_state_sequence_score_numpy"
        or decoder.get("contract_version") != 1
        or decoder.get("score_semantics")
        != "maximum_complete_left_to_right_log_odds_path"
        or arguments != EXPECTED_DECODER_ARGUMENTS
    ):
        raise ValueError("artifact decoder semantics drift")
    expected_decoder_hash = sha256_json(
        {
            "topology": dict(metadata["topology"]),
            "algorithm": decoder["algorithm"],
            "arguments": dict(arguments),
        }
    )
    if decoder.get("contract_sha256") != expected_decoder_hash:
        raise ValueError("artifact decoder contract hash drift")
    reference = _resolve(
        decoder.get("reference_module"), metadata_path.parent,
        "decoder reference module",
    )
    expected_reference_hash = _canonical_sha(
        decoder.get("reference_module_sha256"), "decoder reference module"
    )
    if not reference.is_file() or sha256_file(reference) != expected_reference_hash:
        raise ValueError("decoder reference module hash drift")

    timeline = metadata.get("timeline")
    if not isinstance(timeline, Mapping):
        raise ValueError("artifact timeline contract is missing")
    stride = timeline.get("stream_input_frames_per_call")
    phase = timeline.get("stream_phase_offset_frames")
    calls = timeline.get("streaming_calls_per_260_frame_example")
    warmup = timeline.get("streaming_warmup_outputs_discarded")
    output_times = timeline.get("offline_output_times_seconds")
    if (
        timeline.get("frontend_feature_step_seconds") != 0.01
        or timeline.get("frontend_window_seconds") != 0.03
        or timeline.get("offline_input_frames") != 260
        or timeline.get("offline_output_frames") != 66
        or stride != 3
        or not isinstance(phase, int)
        or not 0 <= phase < stride
        or timeline.get("stream_hop_seconds") != 0.03
        or timeline.get("stream_phase_priming")
        != "zero_prefix_then_observed_prefix"
        or timeline.get("causal_tail_alignment")
        != "derived_from_calls_minus_offline_output_frames"
        or not isinstance(calls, int)
        or not isinstance(warmup, int)
        or calls - warmup != 66
        or not isinstance(output_times, list)
        or len(output_times) != 66
        or any(
            not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for value in output_times
        )
        or any(float(right) <= float(left) for left, right in zip(output_times, output_times[1:]))
    ):
        raise ValueError("artifact streaming timeline contract drift")

    tensors = metadata.get("tensor_contracts")
    if not isinstance(tensors, Mapping):
        raise ValueError("artifact tensor contracts are missing")
    input_spec = tensors.get("input")
    output_spec = tensors.get("output")
    if not isinstance(input_spec, Mapping) or not isinstance(output_spec, Mapping):
        raise ValueError("artifact input/output tensor contracts are missing")
    if (
        tuple(input_spec.get("shape", ())) != EXPECTED_INPUT_SHAPE
        or input_spec.get("dtype") != "int8"
        or tuple(output_spec.get("shape", ())) != EXPECTED_OUTPUT_SHAPE
        or output_spec.get("dtype") != "uint8"
        or tensors.get("output_semantics") != "unnormalized_ordered_state_logits"
    ):
        raise ValueError("artifact tensor topology contract drift")
    input_quantization = _quantization_contract(
        input_spec.get("quantization"), "int8", "input"
    )
    output_quantization = _quantization_contract(
        output_spec.get("quantization"), "uint8", "output"
    )

    memory = metadata.get("static_memory_contract")
    audit = memory.get("tensor_audit") if isinstance(memory, Mapping) else None
    if (
        not isinstance(memory, Mapping)
        or memory.get("batch_size") != 1
        or memory.get("fixed_input_shape") is not True
        or memory.get("fixed_output_shape") is not True
        or memory.get("dynamic_tensor_shapes_forbidden") is not True
        or memory.get("external_state_tensor_count") != 0
        or memory.get("persistent_state") != "internal_tflite_variables"
        or not isinstance(audit, Mapping)
        or audit.get("input_count") != 1
        or audit.get("output_count") != 1
        or audit.get("dynamic_shape_tensor_count") != 0
    ):
        raise ValueError("artifact state/static-memory contract drift")

    equivalence = metadata.get("equivalence")
    if not isinstance(equivalence, Mapping):
        raise ValueError("artifact equivalence evidence is missing")
    evidence = dict(equivalence)
    declared_evidence_hash = evidence.pop("evidence_sha256", None)
    if declared_evidence_hash != sha256_json(evidence):
        raise ValueError("artifact equivalence evidence hash drift")
    if (
        equivalence.get("algorithm")
        != "generic_ordered_state_sequence_score_numpy_v1"
        or equivalence.get("from_logits") is not True
        or equivalence.get("streaming_calls_per_example") != calls
        or equivalence.get("streaming_warmup_outputs_discarded") != warmup
        or equivalence.get("offline_shape") != [66, topology.state_count]
    ):
        raise ValueError("artifact equivalence/timeline contract drift")

    runtime_contract = {
        "input_quantization": input_quantization,
        "output_quantization": output_quantization,
        "stride": stride,
        "phase_offset": phase,
        "calls": calls,
        "warmup": warmup,
        "feature_step_seconds": 0.01,
        "decoder_arguments": dict(arguments),
    }
    return metadata, topology, runtime_contract


def _threshold_from_report(
    path: Path, topology: OrderedStateTopology
) -> tuple[float, dict[str, Any]]:
    report = _load_json(path, "threshold report")
    _verify_nested_bindings(report, path.parent, "threshold_report")
    selection = report.get("threshold_selection", report.get("selection"))
    if not isinstance(selection, Mapping):
        raise ValueError("threshold report has no selection record")
    if (
        selection.get("fit_split") != "validation"
        or selection.get("test_used_for_selection") is not False
    ):
        raise ValueError("threshold must be selected on validation only")
    if report.get("deployment_qualification") is not False:
        raise ValueError("threshold report must preserve non-deployment qualification")
    threshold = _finite_number(selection.get("threshold"), "threshold")
    topology_value = report.get("topology")
    if isinstance(topology_value, Mapping) and (
        topology_value.get("state_count") != topology.state_count
        or tuple(topology_value.get("phones", ())) != topology.phones
        or topology_value.get("states_per_phone") != topology.states_per_phone
    ):
        raise ValueError("threshold report topology differs from artifact")
    return threshold, {
        "mode": "validation_only_report",
        "path": str(path),
        "sha256": sha256_file(path),
        "fit_split": "validation",
        "test_used_for_selection": False,
    }


def _select_validation_threshold(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
    *,
    minimum_recall: float,
    maximum_false_candidate_fraction: float,
) -> dict[str, Any]:
    if (
        positive_scores.ndim != 1
        or negative_scores.ndim != 1
        or not len(positive_scores)
        or not len(negative_scores)
        or not np.all(np.isfinite(positive_scores))
        or not np.all(np.isfinite(negative_scores))
    ):
        raise ValueError("threshold fitting requires finite validation positives and negatives")
    if not 0 < minimum_recall <= 1 or not 0 <= maximum_false_candidate_fraction <= 1:
        raise ValueError("invalid threshold recall/false-candidate constraints")
    thresholds = np.unique(np.concatenate([positive_scores, negative_scores]))
    points: list[dict[str, Any]] = []
    for raw_threshold in thresholds:
        threshold = float(raw_threshold)
        detected = int(np.sum(positive_scores >= threshold))
        false_candidates = int(np.sum(negative_scores >= threshold))
        recall = detected / len(positive_scores)
        false_fraction = false_candidates / len(negative_scores)
        points.append(
            {
                "threshold": threshold,
                "positive_opportunities": len(positive_scores),
                "detected_opportunities": detected,
                "opportunity_recall": recall,
                "negative_windows": len(negative_scores),
                "false_candidates": false_candidates,
                "false_candidate_fraction": false_fraction,
                "meets_minimum_recall": recall >= minimum_recall,
                "meets_maximum_false_candidate_fraction": (
                    false_fraction <= maximum_false_candidate_fraction
                ),
            }
        )
    selected = max(
        points,
        key=lambda point: (
            point["meets_minimum_recall"]
            and point["meets_maximum_false_candidate_fraction"],
            point["meets_minimum_recall"],
            -point["false_candidates"],
            point["opportunity_recall"],
            point["threshold"],
        ),
    )
    return {
        **selected,
        "fit_split": "validation",
        "test_used_for_selection": False,
        "minimum_recall": minimum_recall,
        "maximum_false_candidate_fraction": maximum_false_candidate_fraction,
        "qualified": (
            selected["meets_minimum_recall"]
            and selected["meets_maximum_false_candidate_fraction"]
        ),
        "selection_order": (
            "joint_constraints_then_recall_constraint_then_fewest_false_candidates_"
            "then_recall_then_highest_threshold"
        ),
    }


def _split_metrics(
    scored: Sequence[tuple[Mapping[str, Any], np.ndarray]], threshold: float
) -> dict[str, Any]:
    positives = np.asarray(
        [float(np.max(scores)) for row, scores in scored if int(row["label"]) == 1],
        dtype=np.float64,
    )
    negatives = np.asarray(
        [float(np.max(scores)) for row, scores in scored if int(row["label"]) == 0],
        dtype=np.float64,
    )
    detected = int(np.sum(positives >= threshold))
    false_candidates = int(np.sum(negatives >= threshold))
    return {
        "positive_opportunities": len(positives),
        "detected_opportunities": detected,
        "opportunity_recall": detected / len(positives) if len(positives) else None,
        "negative_windows": len(negatives),
        "false_candidates": false_candidates,
        "false_candidate_fraction": (
            false_candidates / len(negatives) if len(negatives) else None
        ),
    }


def _source_rows(
    manifest_path: Path,
    features_path: Path,
    *,
    feature_hash_field: str = "feature_sha256",
    expected_input_frames: int | None = 260,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    manifest = _load_json(manifest_path, "source manifest")
    rows_value = manifest.get("examples", manifest.get("records"))
    if not isinstance(rows_value, list) or not rows_value:
        raise ValueError("source manifest examples are missing")
    features = np.load(features_path, mmap_mode="r", allow_pickle=False)
    if (
        features.ndim != 3
        or features.shape[2] != 40
        or (expected_input_frames is not None and features.shape[1] != expected_input_frames)
        or (expected_input_frames is None and features.shape[1] < 260)
        or not np.issubdtype(features.dtype, np.number)
    ):
        expected = "[N,>=260,40]" if expected_input_frames is None else "[N,260,40]"
        raise ValueError(f"source features must be numeric {expected}")
    if len(rows_value) != len(features):
        raise ValueError("source manifest and feature array counts differ")
    declared = manifest.get("array_sha256", {}).get(features_path.name)
    observed_array_hash = sha256_file(features_path)
    if declared != observed_array_hash:
        raise ValueError("source feature-array hash drift")

    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    indexes: set[int] = set()
    for ordinal, raw in enumerate(rows_value):
        if not isinstance(raw, Mapping):
            raise ValueError("source manifest row is not an object")
        row = dict(raw)
        source_id = row.get("source_id")
        index = row.get("feature_index", ordinal)
        split = row.get("split")
        label = row.get("label")
        if not isinstance(source_id, str) or not source_id or source_id in ids:
            raise ValueError("source IDs must be unique nonempty strings")
        if not isinstance(index, int) or not 0 <= index < len(features) or index in indexes:
            raise ValueError(f"{source_id}: feature index identity drift")
        if split not in SPLITS or label not in (0, 1, False, True):
            raise ValueError(f"{source_id}: split/label identity is invalid")
        sample = np.asarray(features[index])
        if not np.all(np.isfinite(sample)):
            raise ValueError(f"{source_id}: source features are non-finite")
        observed = feature_sha256(sample)
        if row.get(feature_hash_field) != observed:
            raise ValueError(f"{source_id}: row feature SHA drift")
        row["feature_index"] = index
        row["label"] = int(label)
        rows.append(row)
        ids.add(source_id)
        indexes.add(index)
    if indexes != set(range(len(features))):
        raise ValueError("source feature indexes are not a complete identity mapping")
    return manifest, rows, features


def _default_interpreter_factory(model_path: Path) -> Any:
    try:
        import tensorflow as tf

        return tf.lite.Interpreter(model_path=str(model_path))
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError as error:
            raise RuntimeError(
                "TensorFlow or tflite-runtime is required to execute the detector"
            ) from error
        return Interpreter(model_path=str(model_path))


def _detail_contract(
    detail: Mapping[str, Any], expected_shape: tuple[int, ...], expected_dtype: str,
    expected_quantization: tuple[float, int], label: str,
) -> None:
    shape = tuple(int(value) for value in detail.get("shape", ()))
    dtype = np.dtype(detail.get("dtype")).name
    scale, zero = detail.get("quantization", (0.0, 0))
    if (
        shape != expected_shape
        or dtype != expected_dtype
        or not math.isclose(float(scale), expected_quantization[0], rel_tol=0, abs_tol=1e-12)
        or int(zero) != expected_quantization[1]
    ):
        raise ValueError(f"runtime {label} tensor differs from firmware artifact contract")


def _stream_chunks(
    sample: np.ndarray, stride: int, phase_offset: int
) -> list[tuple[np.ndarray, int]]:
    chunks: list[tuple[np.ndarray, int]] = []
    if phase_offset:
        primer = np.zeros((stride, sample.shape[1]), dtype=np.float32)
        primer[-phase_offset:] = sample[:phase_offset]
        chunks.append((primer, phase_offset - 1))
    for offset in range(phase_offset, len(sample) - stride + 1, stride):
        chunks.append((np.asarray(sample[offset : offset + stride], dtype=np.float32), offset + stride - 1))
    return chunks


def _run_one(
    model_path: Path,
    sample: np.ndarray,
    topology: OrderedStateTopology,
    contract: Mapping[str, Any],
    interpreter_factory: Callable[[Path], Any],
    *,
    fixed_timeline: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    # A fresh interpreter is the strongest portable reset for internal TFLite
    # variables.  No example can inherit recurrent state from another example.
    interpreter = interpreter_factory(model_path)
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("runtime must expose exactly one input and one output")
    input_detail = inputs[0]
    output_detail = outputs[0]
    _detail_contract(
        input_detail, EXPECTED_INPUT_SHAPE, "int8",
        tuple(contract["input_quantization"]), "input",
    )
    _detail_contract(
        output_detail, EXPECTED_OUTPUT_SHAPE, "uint8",
        tuple(contract["output_quantization"]), "output",
    )
    input_scale, input_zero = contract["input_quantization"]
    output_scale, output_zero = contract["output_quantization"]
    input_limits = np.iinfo(np.int8)
    emitted: list[np.ndarray] = []
    feature_ends: list[int] = []
    chunks = _stream_chunks(sample, contract["stride"], contract["phase_offset"])
    if fixed_timeline and len(chunks) != contract["calls"]:
        raise ValueError("source feature timeline differs from firmware artifact calls")
    for chunk, feature_end in chunks:
        quantized = np.clip(
            np.rint(chunk / input_scale + input_zero),
            input_limits.min,
            input_limits.max,
        ).astype(np.int8)
        interpreter.set_tensor(int(input_detail["index"]), quantized[None, ...])
        interpreter.invoke()
        raw = np.asarray(interpreter.get_tensor(int(output_detail["index"])))
        if raw.shape != EXPECTED_OUTPUT_SHAPE or raw.dtype != np.uint8:
            raise ValueError("runtime output violates deployed uint8 tensor contract")
        logits = (raw.astype(np.float32) - output_zero) * output_scale
        if not np.all(np.isfinite(logits)):
            raise ValueError("dequantized detector logits are non-finite")
        emitted.append(logits[0, 0])
        feature_ends.append(feature_end)
    warmup = int(contract["warmup"])
    aligned = np.asarray(emitted[warmup:], dtype=np.float32)
    aligned_feature_ends = feature_ends[warmup:]
    if fixed_timeline:
        if aligned.shape != (66, topology.state_count):
            raise ValueError("deployed stream does not produce the aligned 66-frame timeline")
    elif aligned.ndim != 2 or aligned.shape[1] != topology.state_count:
        raise ValueError("continuous deployed stream output topology drift")

    # Prefix scoring uses the exact artifact-bound generic decoder.  Prefixes
    # shorter than the ordered path have -inf completion scores and are omitted,
    # leaving a finite timeline accepted by the strict candidate builder.
    first_complete = topology.ordered_state_count - 1
    scores = np.asarray(
        [
            ordered_state_sequence_score_numpy(
                aligned[None, : stop + 1], topology,
                **contract["decoder_arguments"],
            )[0]
            for stop in range(first_complete, len(aligned))
        ],
        dtype=np.float64,
    )
    if not len(scores) or not np.all(np.isfinite(scores)):
        raise ValueError("ordered-state score timeline is empty or non-finite")
    score_feature_ends = np.asarray(aligned_feature_ends[first_complete:], dtype=np.int32)
    stream_calls = list(range(warmup + first_complete, len(chunks)))
    return aligned, scores, score_feature_ends, stream_calls


def _threshold_regions(scores: np.ndarray, threshold: float) -> list[int]:
    events: list[int] = []
    start = 0
    while start < len(scores):
        if scores[start] < threshold:
            start += 1
            continue
        stop = start + 1
        while stop < len(scores) and scores[stop] >= threshold:
            stop += 1
        events.append(start + int(np.argmax(scores[start:stop])))
        start = stop
    return events


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as target:
            target.write(content)
            target.flush()
            os.fsync(target.fileno())
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
        with open(temporary, "wb") as target:
            np.save(target, values, allow_pickle=False)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def trace_detector(
    firmware_artifact: Path,
    model: Path,
    source_manifest: Path,
    source_features: Path,
    output: Path,
    *,
    threshold: float | None = None,
    threshold_report: Path | None = None,
    threshold_output: Path | None = None,
    minimum_recall: float = 0.95,
    maximum_false_candidate_fraction: float = 0.20,
    allow_continuous_context: bool = False,
    evaluation_only: bool = False,
    interpreter_factory: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    firmware_artifact = firmware_artifact.expanduser().resolve()
    model = model.expanduser().resolve()
    source_manifest = source_manifest.expanduser().resolve()
    source_features = source_features.expanduser().resolve()
    output = output.expanduser().resolve()
    modes = sum(
        value is not None for value in (threshold, threshold_report, threshold_output)
    )
    if modes != 1:
        raise ValueError(
            "provide exactly one explicit threshold, validation-only report, or threshold output"
        )
    metadata, topology, contract = _validate_artifact(firmware_artifact, model)
    candidate_verifier = metadata.get("candidate_conditioned") is True
    _, rows, features = _source_rows(
        source_manifest,
        source_features,
        feature_hash_field=(
            "candidate_feature_sha256"
            if candidate_verifier and not allow_continuous_context
            else "feature_sha256"
        ),
        expected_input_frames=None if allow_continuous_context else 260,
    )
    if evaluation_only:
        rows = [row for row in rows if row["split"] in {"validation", "test"}]
        if not rows:
            raise ValueError("evaluation-only tracing requires validation or test rows")

    output.mkdir(parents=True, exist_ok=True)
    factory = interpreter_factory or _default_interpreter_factory
    scored_by_id: dict[
        str, tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, list[int]]
    ] = {}

    def score_rows(selected_rows: Sequence[dict[str, Any]]) -> None:
        for row in sorted(selected_rows, key=lambda value: str(value["source_id"])):
            index = int(row["feature_index"])
            sample = np.asarray(features[index], dtype=np.float32)
            logits, scores, feature_frames, stream_calls = _run_one(
                model,
                sample,
                topology,
                contract,
                factory,
                fixed_timeline=not allow_continuous_context,
            )
            scored_by_id[str(row["source_id"])] = (
                row, logits, scores, feature_frames, stream_calls
            )

    if threshold_report is not None:
        threshold_report = threshold_report.expanduser().resolve()
        threshold_value, threshold_provenance = _threshold_from_report(
            threshold_report, topology
        )
        threshold_binding = {
            "path": str(threshold_report),
            "sha256": sha256_file(threshold_report),
            "value": threshold_value,
        }
        score_rows(rows)
    elif threshold_output is not None:
        threshold_output = threshold_output.expanduser().resolve()
        if threshold_output == output / "detector-traces.json":
            raise ValueError("threshold output must be distinct from trace manifest")
        validation_rows = [row for row in rows if row["split"] == "validation"]
        score_rows(validation_rows)
        validation_scored = [
            (row, scored_by_id[str(row["source_id"])][2])
            for row in validation_rows
        ]
        validation_positives = np.asarray(
            [
                float(np.max(scores))
                for row, scores in validation_scored
                if int(row["label"]) == 1
            ],
            dtype=np.float64,
        )
        validation_negatives = np.asarray(
            [
                float(np.max(scores))
                for row, scores in validation_scored
                if int(row["label"]) == 0
            ],
            dtype=np.float64,
        )
        selection = _select_validation_threshold(
            validation_positives,
            validation_negatives,
            minimum_recall=minimum_recall,
            maximum_false_candidate_fraction=maximum_false_candidate_fraction,
        )
        threshold_value = float(selection["threshold"])
        threshold_document = {
            "schema_version": 1,
            "kind": (
                "kizz_control_ordered_state_candidate_verifier_int8_validation_threshold"
                if candidate_verifier
                else "kizz_control_deployed_int8_validation_threshold"
            ),
            "deployment_qualification": False,
            "qualification_scope": (
                "candidate_verifier_threshold_only"
                if candidate_verifier
                else "candidate_detector_threshold_only"
            ),
            "selection": selection,
            "threshold": threshold_value,
            "bindings": {
                "artifact": {"path": str(model), "sha256": sha256_file(model)},
                "config": {
                    "path": str(firmware_artifact),
                    "sha256": sha256_file(firmware_artifact),
                },
                "source_manifest": {
                    "path": str(source_manifest),
                    "sha256": sha256_file(source_manifest),
                },
                "source_features": {
                    "path": str(source_features),
                    "sha256": sha256_file(source_features),
                },
            },
            "topology": dict(metadata["topology"]),
            "decoder_contract_sha256": metadata["decoder"]["contract_sha256"],
            "test_metrics": None,
            "test_scored_after_threshold_frozen": True,
        }
        _atomic_bytes(
            threshold_output,
            (
                json.dumps(
                    threshold_document,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8"),
        )
        # The file and its hash are frozen before a test example is invoked.
        threshold_hash = sha256_file(threshold_output)
        threshold_binding = {
            "path": str(threshold_output),
            "sha256": threshold_hash,
            "value": threshold_value,
        }
        threshold_provenance = {
            "mode": "fitted_deployed_int8_validation_only",
            "path": str(threshold_output),
            "sha256": threshold_hash,
            "fit_split": "validation",
            "test_used_for_selection": False,
            "qualified": bool(selection["qualified"]),
        }
        score_rows([row for row in rows if row["split"] != "validation"])
    else:
        threshold_value = _finite_number(threshold, "explicit threshold")
        threshold_path = output / "detector-threshold.json"
        threshold_document = {
            "schema_version": 1,
            "kind": "explicit_deployed_int8_detector_threshold",
            "threshold": threshold_value,
            "selection_claim": None,
            "test_used_for_selection": None,
            "deployment_qualification": False,
        }
        _atomic_bytes(
            threshold_path,
            (json.dumps(threshold_document, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"),
        )
        threshold_provenance = {
            "mode": "explicit_cli_no_selection_claim",
            "path": str(threshold_path),
            "sha256": sha256_file(threshold_path),
            "fit_split": None,
            "test_used_for_selection": None,
        }
        threshold_binding = {**threshold_provenance, "value": threshold_value}
        threshold_binding.pop("mode")
        threshold_binding.pop("fit_split")
        threshold_binding.pop("test_used_for_selection")
        score_rows(rows)

    traces: list[dict[str, Any]] = []
    logits_rows: list[np.ndarray] = []
    score_array_rows: list[np.ndarray] = []
    frame_rows: list[np.ndarray] = []
    for row in sorted(rows, key=lambda value: str(value["source_id"])):
        index = int(row["feature_index"])
        _, logits, scores, feature_frames, stream_calls = scored_by_id[
            str(row["source_id"])
        ]
        event_indexes = _threshold_regions(scores, threshold_value)
        events = [
            {
                "score_frame_index": int(score_index),
                "frame_index": int(score_index),
                "score": float(scores[score_index]),
                "feature_frame_index": int(feature_frames[score_index]),
                "feature_time_seconds": float(
                    feature_frames[score_index] * contract["feature_step_seconds"]
                ),
                "stream_call_index": int(stream_calls[score_index]),
                "aligned_output_frame_index": int(
                    stream_calls[score_index] - contract["warmup"]
                ),
            }
            for score_index in event_indexes
        ]
        traces.append(
            {
                "source_id": str(row["source_id"]),
                "feature_index": index,
                "split": str(row["split"]),
                "label": int(row["label"]),
                "source_feature_sha256": str(row["feature_sha256"]),
                "scores": [float(value) for value in scores],
                "feature_frame_indexes": [int(value) for value in feature_frames],
                "stream_call_indexes": stream_calls,
                "events": events,
            }
        )
        logits_rows.append(logits)
        score_array_rows.append(scores)
        frame_rows.append(feature_frames)

    metrics = {
        split: _split_metrics(
            [
                (row, scored_by_id[str(row["source_id"])][2])
                for row in rows
                if row["split"] == split
            ],
            threshold_value,
        )
        for split in SPLITS
    }

    arrays = {
        "state-logits.npy": np.stack(logits_rows).astype(np.float32, copy=False),
        "scores.npy": np.stack(score_array_rows).astype(np.float64, copy=False),
        "feature-frame-indexes.npy": np.stack(frame_rows).astype(np.int32, copy=False),
    }
    for name, values in arrays.items():
        _atomic_npy(output / name, values)
    array_bindings = {
        name: {
            "path": str(output / name),
            "sha256": sha256_file(output / name),
            "shape": [int(value) for value in values.shape],
            "dtype": str(values.dtype),
        }
        for name, values in arrays.items()
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "recipe": TRACE_RECIPE,
        "deployment_qualification": False,
        "qualification_scope": "deployed_int8_trace_generation_only",
        "source_manifest": {
            "path": str(source_manifest), "sha256": sha256_file(source_manifest)
        },
        "source_features": {
            "path": str(source_features), "sha256": sha256_file(source_features)
        },
        "detector": {
            "artifact": {"path": str(model), "sha256": sha256_file(model)},
            "config": {
                "path": str(firmware_artifact),
                "sha256": sha256_file(firmware_artifact),
            },
            "threshold": threshold_binding,
            "threshold_provenance": threshold_provenance,
            "event_policy": "recorded_events",
            "score_geometry": {
                "feature_stride_frames": contract["stride"],
                "feature_offset_frames": int(frame_rows[0][0]),
                "feature_hop_ms": contract["feature_step_seconds"] * 1000.0,
            },
            "topology": dict(metadata["topology"]),
            "decoder": {
                "algorithm": metadata["decoder"]["algorithm"],
                "contract_sha256": metadata["decoder"]["contract_sha256"],
                "arguments": dict(contract["decoder_arguments"]),
            },
            "timeline": dict(metadata["timeline"]),
            "source_context": {
                "mode": (
                    "continuous_fixed_length"
                    if allow_continuous_context
                    else "artifact_equivalence_260_frames"
                ),
                "input_frames": int(features.shape[1]),
                "evaluation_only": evaluation_only,
            },
            "state_reset": "fresh_interpreter_and_allocate_tensors_per_example",
        },
        "arrays": array_bindings,
        "counts": {
            "examples": len(traces),
            "positive_examples": sum(row["label"] == 1 for row in traces),
            "negative_examples": sum(row["label"] == 0 for row in traces),
            "candidate_events": sum(len(row["events"]) for row in traces),
        },
        "evaluation": {
            "threshold_frozen_before_test_scoring": True,
            "test_used_for_selection": False,
            "by_split": metrics,
        },
        "examples": traces,
    }
    manifest_path = output / "detector-traces.json"
    _atomic_bytes(
        manifest_path,
        (json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8"),
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--firmware-artifact", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    threshold = parser.add_mutually_exclusive_group(required=True)
    threshold.add_argument("--threshold", type=float)
    threshold.add_argument("--threshold-report", type=Path)
    threshold.add_argument("--threshold-output", type=Path)
    parser.add_argument("--minimum-recall", type=float, default=0.95)
    parser.add_argument(
        "--maximum-false-candidate-fraction", type=float, default=0.20
    )
    parser.add_argument(
        "--allow-continuous-context",
        action="store_true",
        help="score fixed-length contexts longer than the 260-frame equivalence fixture",
    )
    parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help="score validation and test rows only; train examples remain unread",
    )
    args = parser.parse_args(argv)
    report = trace_detector(
        args.firmware_artifact,
        args.model,
        args.source_manifest,
        args.source_features,
        args.output,
        threshold=args.threshold,
        threshold_report=args.threshold_report,
        threshold_output=args.threshold_output,
        minimum_recall=args.minimum_recall,
        maximum_false_candidate_fraction=args.maximum_false_candidate_fraction,
        allow_continuous_context=args.allow_continuous_context,
        evaluation_only=args.evaluation_only,
    )
    print(json.dumps(report["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
