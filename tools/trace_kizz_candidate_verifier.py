#!/usr/bin/env python3
"""Score every frozen detector candidate with the deployed INT8 verifier.

The resulting detector/verifier trace pair is accepted by
``evaluate_kizz_cascade.py``.  This tool performs no threshold selection and
preserves validation/test separation; it only records dequantized deployed
model scores with provenance-bound candidate geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_sha(value: object, label: str) -> str:
    digest = str(value or "")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return digest


def _resolve(raw: object, anchor: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} path is required")
    path = Path(raw).expanduser()
    return (anchor / path).resolve() if not path.is_absolute() else path.resolve()


def _binding(value: object, anchor: Path, label: str) -> Path:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} binding is required")
    path = _resolve(value.get("path"), anchor, label)
    expected = _require_sha(value.get("sha256"), f"{label} hash")
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"{label} hash drift")
    if value.get("bytes") is not None and value["bytes"] != path.stat().st_size:
        raise ValueError(f"{label} byte-size drift")
    return path


def _quantization(value: object, label: str) -> tuple[float, int]:
    if isinstance(value, Mapping):
        scale, zero = value.get("scale"), value.get("zero_point")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        scale, zero = value
    else:
        raise ValueError(f"{label} quantization is invalid")
    scale = float(scale)
    zero = int(zero)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"{label} quantization scale is invalid")
    return scale, zero


class Int8Verifier:
    def __init__(self, model_path: Path):
        try:
            import tensorflow as tf
            interpreter_type = tf.lite.Interpreter
        except ImportError:
            from tflite_runtime.interpreter import Interpreter as interpreter_type
        self.runner = interpreter_type(model_path=str(model_path), num_threads=1)
        self.runner.allocate_tensors()
        self.input = self.runner.get_input_details()[0]
        self.output = self.runner.get_output_details()[0]

    def __call__(self, feature: np.ndarray) -> float:
        in_scale, in_zero = self.input["quantization"]
        out_scale, out_zero = self.output["quantization"]
        target_shape = tuple(int(v) for v in self.input["shape"])
        values = np.asarray(feature, dtype=np.float32)
        if values.shape == (260, 40) and target_shape == (1, 260, 40, 1):
            values = values[None, ..., None]
        elif values.shape == (260, 40) and target_shape == (1, 260, 40):
            values = values[None, ...]
        if values.shape != target_shape:
            raise ValueError(f"verifier input shape drift: {values.shape} != {target_shape}")
        info = np.iinfo(self.input["dtype"])
        quantized = np.clip(np.rint(values / in_scale + in_zero), info.min, info.max)
        self.runner.set_tensor(self.input["index"], quantized.astype(self.input["dtype"]))
        self.runner.invoke()
        raw = float(np.asarray(self.runner.get_tensor(self.output["index"])).reshape(-1)[0])
        return (raw - out_zero) * out_scale


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _source_base(row: Mapping[str, Any]) -> dict[str, Any]:
    audio_sha256 = row.get("audio_sha256", row.get("source_audio_sha256"))
    base = {
        "source_id": str(row["source_id"]),
        "split": str(row["split"]),
        "truth": "positive" if int(row["label"]) == 1 else "negative",
        "duration_seconds": float(row["duration_seconds"]),
        "audio_sha256": _require_sha(audio_sha256, "source audio"),
        "events": [],
    }
    for key in (
        "source_audio_sha256", "provenance_id", "ancestry_id", "parent_id",
        "parent_source_id", "speaker_id", "voice_id", "session_id",
    ):
        if row.get(key):
            base[key] = row[key]
    return base


def _corpus_array_paths(
    corpus_path: Path, corpus: Mapping[str, Any]
) -> dict[str, Path]:
    declared = corpus.get("array_sha256")
    if not isinstance(declared, Mapping) or "features.npy" not in declared:
        raise ValueError("candidate corpus array hashes are required")
    paths: dict[str, Path] = {}
    for name, expected in declared.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("candidate array filename is invalid")
        path = corpus_path.parent / name
        if not path.is_file() or sha256_file(path) != _require_sha(
            expected, f"candidate array {name} hash"
        ):
            raise ValueError(f"candidate array {name} hash drift")
        paths[name] = path
    return paths


def _evaluation_source_rows(
    sources: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    rows = {
        str(row["source_id"]): dict(row)
        for row in sources
        if row.get("split") in {"validation", "test"}
    }
    reconstructed: set[str] = set()
    for candidate in candidates:
        if candidate.get("split") not in {"validation", "test"}:
            continue
        source_id = str(candidate.get("parent_source_id", ""))
        if not source_id:
            raise ValueError("candidate parent source identity is missing")
        if source_id not in rows:
            row = dict(candidate)
            row["source_id"] = source_id
            row["audio_sha256"] = candidate.get(
                "source_audio_sha256", candidate.get("audio_sha256")
            )
            rows[source_id] = row
            reconstructed.add(source_id)
        elif source_id in reconstructed:
            existing = _source_base(rows[source_id])
            observed = dict(candidate)
            observed["source_id"] = source_id
            observed["audio_sha256"] = candidate.get(
                "source_audio_sha256", candidate.get("audio_sha256")
            )
            if _source_base(observed) != existing:
                raise ValueError(f"candidate source identity conflict: {source_id}")
    return rows


def trace(
    metadata_path: Path,
    output_dir: Path,
    *,
    candidate_corpus: Path | None = None,
    evaluation_only: bool = False,
    scorer_factory: Callable[[Path], Callable[[np.ndarray], float]] = Int8Verifier,
) -> dict[str, Any]:
    metadata_path = metadata_path.resolve()
    metadata = _load_object(metadata_path, "verifier metadata")
    if (
        metadata.get("schema_version") != 1
        or metadata.get("kind") != "kizz_control_candidate_verifier_fixed_window_int8"
        or metadata.get("candidate_conditioned") is not True
        or metadata.get("deployment_qualification") is not False
    ):
        raise ValueError("unsupported or incorrectly qualified verifier metadata")
    artifact = metadata.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ValueError("verifier artifact binding is required")
    model_path = metadata_path.parent / str(artifact.get("filename"))
    if sha256_file(model_path) != _require_sha(artifact.get("sha256"), "artifact hash"):
        raise ValueError("verifier artifact hash drift")

    inputs = metadata.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("verifier input bindings are required")
    training_corpus_path = _binding(
        inputs.get("candidate_corpus"), metadata_path.parent, "candidate corpus"
    )
    array_bindings = inputs.get("candidate_arrays")
    if not isinstance(array_bindings, Mapping):
        raise ValueError("candidate array bindings are required")
    training_feature_path = _binding(
        array_bindings.get("features.npy"), metadata_path.parent, "candidate features"
    )
    for name, value in array_bindings.items():
        _binding(value, metadata_path.parent, f"candidate array {name}")
    if candidate_corpus is None:
        corpus_path = training_corpus_path
        feature_path = training_feature_path
        corpus = _load_object(corpus_path, "candidate corpus")
    else:
        corpus_path = candidate_corpus.expanduser().resolve()
        corpus = _load_object(corpus_path, "candidate corpus")
        feature_path = _corpus_array_paths(corpus_path, corpus)["features.npy"]
    source_binding = corpus.get("manifests", {}).get("source")
    source_path = _binding(source_binding, corpus_path.parent, "source manifest")
    source_manifest = _load_object(source_path, "source manifest")
    detector_trace_binding = corpus.get("bindings", {}).get("detector_traces")
    detector_trace_path = _binding(detector_trace_binding, corpus_path.parent, "detector trace")
    detector_trace = _load_object(detector_trace_path, "detector trace")
    detector_artifact = detector_trace.get("detector", {}).get("artifact")
    detector_model_path = _binding(detector_artifact, detector_trace_path.parent, "detector artifact")

    examples = corpus.get("examples")
    sources = source_manifest.get("examples")
    if not isinstance(examples, list) or not isinstance(sources, list):
        raise ValueError("candidate/source examples are required")
    features = np.load(feature_path, mmap_mode="r")
    if features.shape != (len(examples), 260, 40):
        raise ValueError("candidate feature array shape drift")
    source_rows = _evaluation_source_rows(sources, examples)
    detector_sources = {key: _source_base(row) for key, row in source_rows.items()}
    verifier_sources = {key: _source_base(row) for key, row in source_rows.items()}
    scorer = scorer_factory(model_path)
    split_scores: dict[str, list[float]] = {"train": [], "validation": [], "test": []}
    scored = 0
    for index, candidate in enumerate(examples):
        split = str(candidate.get("split"))
        if evaluation_only and split not in {"validation", "test"}:
            continue
        score = float(scorer(features[index]))
        if not math.isfinite(score):
            raise ValueError("verifier emitted a non-finite score")
        split_scores.setdefault(split, []).append(score)
        scored += 1
        if split not in {"validation", "test"}:
            continue
        source_id = str(candidate.get("parent_source_id"))
        if source_id not in source_rows:
            raise ValueError(f"candidate source binding drift: {source_id}")
        event_info = candidate.get("detector_event")
        window = candidate.get("window")
        if not isinstance(event_info, Mapping) or not isinstance(window, Mapping):
            raise ValueError("candidate event/window geometry is missing")
        duration = float(source_rows[source_id]["duration_seconds"])
        timestamp = min(duration, max(0.0, float(event_info["feature_time_seconds"])))
        start = max(0.0, float(window["requested_start_frame"]) * 0.01)
        end = min(duration, float(window["requested_stop_frame_exclusive"]) * 0.01)
        if end < timestamp:
            end = timestamp
        common = {
            "candidate_id": str(candidate["candidate_id"]),
            "start_seconds": start,
            "end_seconds": end,
            "timestamp_seconds": timestamp,
            "window_sha256": _require_sha(candidate.get("candidate_feature_sha256"), "candidate feature"),
        }
        detector_sources[source_id]["events"].append({**common, "score": float(candidate["detector_score"])})
        verifier_sources[source_id]["events"].append({**common, "score": score})

    artifact_row = {"path": str(model_path), "sha256": sha256_file(model_path)}
    detector_artifact_row = {"path": str(detector_model_path), "sha256": sha256_file(detector_model_path)}
    common_provenance = {
        "candidate_corpus": {"path": str(corpus_path), "sha256": sha256_file(corpus_path)},
        "source_manifest": {"path": str(source_path), "sha256": sha256_file(source_path)},
        "detector_trace": {"path": str(detector_trace_path), "sha256": sha256_file(detector_trace_path)},
        "verifier_metadata": {"path": str(metadata_path), "sha256": sha256_file(metadata_path)},
        "verifier_training_corpus": {
            "path": str(training_corpus_path),
            "sha256": sha256_file(training_corpus_path),
        },
        "evaluation_only": evaluation_only,
        "threshold_selection_performed": False,
        "test_used_for_selection": False,
    }
    detector_payload = {
        "schema_version": 1, "trace_kind": "detector", "artifact": detector_artifact_row,
        "provenance": common_provenance, "sources": [detector_sources[key] for key in sorted(detector_sources)],
    }
    verifier_payload = {
        "schema_version": 1, "trace_kind": "verifier", "artifact": artifact_row,
        "provenance": common_provenance, "sources": [verifier_sources[key] for key in sorted(verifier_sources)],
    }
    output_dir = output_dir.resolve()
    detector_output = output_dir / "detector-trace.json"
    verifier_output = output_dir / "verifier-int8-trace.json"
    _atomic_json(detector_output, detector_payload)
    _atomic_json(verifier_output, verifier_payload)
    summary = {
        "schema_version": 1,
        "trace_kind": "deployed_int8_candidate_verifier_summary",
        "candidate_count": scored,
        "scored_by_split": {key: len(value) for key, value in split_scores.items()},
        "score_range_by_split": {
            key: {"minimum": min(value), "maximum": max(value)} if value else None
            for key, value in split_scores.items()
        },
        "detector_trace": {"path": str(detector_output), "sha256": sha256_file(detector_output)},
        "verifier_trace": {"path": str(verifier_output), "sha256": sha256_file(verifier_output)},
        "provenance": common_provenance,
        "deployment_qualification": False,
    }
    _atomic_json(output_dir / "trace-summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verifier-metadata", type=Path, required=True)
    parser.add_argument(
        "--candidate-corpus",
        type=Path,
        help="score a provenance-bound evaluation corpus instead of the training corpus",
    )
    parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help="score only validation and test candidates; training rows remain unread",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            trace(
                args.verifier_metadata,
                args.output_dir,
                candidate_corpus=args.candidate_corpus,
                evaluation_only=args.evaluation_only,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
