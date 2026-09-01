#!/usr/bin/env python3
"""Mine deployed-detector hard negatives from training-only Mini LibriSpeech.

The locked 100-hour MUSAN corpus is never read.  This tool appends at most four
detector candidates per utterance to an existing verifier corpus while leaving
validation and test rows byte-for-byte equivalent in content and order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.evaluate_kizz_int8_continuous_cascade import (
    OrderedStateCausalScorer,
    TFLiteRuntime,
    _dequantize,
    stream_repository_frontend,
)
from tools.trace_kizz_ordered_state_detector import (
    _threshold_from_report,
    _validate_artifact,
)
from tools.simulate_kizz_int8_cascade import load_firmware_artifact
from tools.train_kizz_candidate_verifier import load_verified_dataset


TOP_K_PER_UTTERANCE = 4
WINDOW_FRAMES = 260
FEATURE_BINS = 40
TARGET_SAMPLE_RATE = 16_000
FEATURE_SAMPLES = 480


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _feature_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


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


def _binding(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _source_identity(root: Path, path: Path) -> tuple[str, str, str]:
    relative = path.relative_to(root)
    if len(relative.parts) < 3:
        raise ValueError(f"unexpected LibriSpeech path: {relative}")
    speaker, chapter = relative.parts[0], relative.parts[1]
    stem = path.stem
    return (
        f"librispeech-mini:{stem}",
        f"librispeech-mini:{speaker}:{chapter}",
        f"librispeech-mini:{speaker}",
    )


def _stream_training_frontend(path: Path):
    """Yield exact frontend frames, resampling train-only source assets when needed."""
    with sf.SoundFile(path) as audio:
        if audio.samplerate == TARGET_SAMPLE_RATE:
            yield from stream_repository_frontend(path)
            return
        rate = int(audio.samplerate)
        values = audio.read(dtype="float32", always_2d=True)
    if rate <= 0 or values.shape[1] < 1:
        raise ValueError(f"invalid training audio contract: {path}")
    from scipy.signal import resample_poly

    divisor = math.gcd(rate, TARGET_SAMPLE_RATE)
    mono = np.mean(np.asarray(values, dtype=np.float32), axis=1)
    mono = np.asarray(
        resample_poly(mono, TARGET_SAMPLE_RATE // divisor, rate // divisor),
        dtype=np.float32,
    )

    from microwakeword.audio.audio_utils import MicroFrontend

    frontend = MicroFrontend()
    process = getattr(frontend, "process_samples", None) or frontend.ProcessSamples
    pcm = np.clip(mono * 32768.0, -32768, 32767).astype("<i2")
    pending = bytearray(pcm.tobytes())
    while len(pending) >= FEATURE_SAMPLES * 2:
        result = process(bytes(pending[: FEATURE_SAMPLES * 2]))
        used = int(getattr(result, "samples_read", FEATURE_SAMPLES))
        if used <= 0 or used > FEATURE_SAMPLES:
            raise ValueError("C MicroFrontend made invalid progress")
        del pending[: used * 2]
        if result.features:
            emitted = np.asarray(result.features, dtype=np.float32)
            if emitted.size % FEATURE_BINS:
                raise ValueError("C MicroFrontend feature width drift")
            for frame in emitted.reshape(-1, FEATURE_BINS):
                if not np.all(np.isfinite(frame)):
                    raise ValueError("C MicroFrontend emitted non-finite features")
                yield np.asarray(frame, dtype=np.float32)


def _mine_file(
    path,
    root,
    runtime,
    topology,
    detector_contract,
    threshold,
    *,
    top_k=TOP_K_PER_UTTERANCE,
):
    if top_k < 1:
        raise ValueError("top_k must be positive")
    runtime.reset()
    scorer = OrderedStateCausalScorer(topology, detector_contract)
    stride = int(detector_contract["stride"])
    phase = int(detector_contract["phase_offset"])
    warmup = int(detector_contract["warmup"])
    ring = deque(maxlen=WINDOW_FRAMES)
    group = []
    active = None
    candidates = []
    hops = 0
    frames = 0

    def invoke(chunk, trigger):
        nonlocal hops, active
        raw = runtime.invoke(chunk)
        hops += 1
        if hops <= warmup:
            return
        logits = _dequantize(raw, runtime._contract, "detector").reshape(-1)
        score = float(scorer.step(logits))
        above = math.isfinite(score) and score >= threshold
        if above:
            if active is None or score > active[0]:
                values = np.zeros((WINDOW_FRAMES, FEATURE_BINS), dtype=np.float32)
                observed = list(ring)
                values[-len(observed):] = np.stack(observed)
                active = (score, trigger, values)
        elif active is not None:
            candidates.append(active)
            active = None

    for raw in _stream_training_frontend(path):
        frame = np.asarray(raw, dtype=np.float32)
        ring.append(frame.copy())
        frame_index = frames
        frames += 1
        if phase and frame_index + 1 == phase:
            primer = np.zeros((stride, FEATURE_BINS), dtype=np.float32)
            primer[-phase:] = np.stack(list(ring)[-phase:])
            invoke(primer, frame_index)
        if frame_index >= phase:
            group.append(frame)
            if len(group) == stride:
                invoke(np.stack(group), frame_index)
                group.clear()
    if active is not None:
        candidates.append(active)
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[:top_k], frames, hops


def mine(
    base_dataset: Path,
    base_corpus_sha256: str,
    librispeech_root: Path,
    detector_metadata: Path,
    detector_model: Path,
    detector_threshold_report: Path,
    output: Path,
) -> dict[str, Any]:
    base = load_verified_dataset(base_dataset, expected_corpus_sha256=base_corpus_sha256)
    detector_metadata = detector_metadata.resolve()
    detector_model = detector_model.resolve()
    detector_meta, topology, detector_contract = _validate_artifact(detector_metadata, detector_model)
    threshold, threshold_provenance = _threshold_from_report(detector_threshold_report, topology)
    detector = load_firmware_artifact(detector_metadata, "detector")
    runtime = TFLiteRuntime(detector_model, detector)
    librispeech_root = librispeech_root.resolve()
    files = sorted(librispeech_root.rglob("*.flac"))
    if not files:
        raise ValueError("Mini LibriSpeech contains no FLAC files")

    rows = [dict(row) for row in base.rows]
    new_features = []
    new_scores = []
    new_feature_frames = []
    new_score_frames = []
    source_ledger = []
    total_seconds = 0.0
    total_hops = 0
    for path in files:
        audio_hash = sha256_file(path)
        with sf.SoundFile(path) as audio:
            if audio.samplerate != 16000 or audio.channels != 1:
                raise ValueError(f"LibriSpeech audio contract drift: {path}")
            duration = len(audio) / audio.samplerate
        total_seconds += duration
        source_id, session_id, speaker_id = _source_identity(librispeech_root, path)
        values, frame_count, hop_count = _mine_file(
            path, librispeech_root, runtime, topology, detector_contract, threshold
        )
        total_hops += hop_count
        source_ledger.append({
            "source_id": source_id, "path": str(path), "audio_sha256": audio_hash,
            "duration_seconds": duration, "speaker_id": speaker_id,
            "session_id": session_id, "candidate_count": len(values),
            "frontend_feature_frames": frame_count, "detector_hops": hop_count,
        })
        for ordinal, (score, trigger, feature) in enumerate(values):
            feature16 = feature.astype(np.float16)
            feature_hash = _feature_hash(feature16)
            material = f"{source_id}\0{trigger}\0{score:.17g}\0{feature_hash}".encode()
            suffix = hashlib.sha256(material).hexdigest()[:20]
            candidate_id = f"{source_id}::detector-candidate::{suffix}"
            rows.append({
                "source_id": candidate_id,
                "candidate_id": candidate_id,
                "parent_source_id": source_id,
                "source_parent_source_id": source_id,
                "speaker_id": speaker_id,
                "session_id": session_id,
                "ancestry_id": source_id,
                "audio_sha256": audio_hash,
                "source_audio_sha256": audio_hash,
                "parent_source_audio_sha256": audio_hash,
                "source_group": "librispeech_public_speech",
                "semantic_label": "non_wake_public_speech",
                "provider": "openslr_librispeech",
                "split": "train",
                "label": 0,
                "duration_seconds": duration,
                "detector_conditioned": True,
                "detector_score": score,
                "detector_feature_frame_index": trigger,
                "detector_score_frame_index": trigger,
                "detector_event_ordinal": ordinal,
                "detector_event": {
                    "feature_frame_index": trigger,
                    "feature_time_seconds": trigger * 0.01,
                    "score": score,
                    "score_frame_index": trigger,
                },
                "window": {
                    "requested_start_frame": trigger - 259,
                    "requested_stop_frame_exclusive": trigger + 1,
                    "source_start_frame": max(0, trigger - 259),
                    "source_stop_frame_exclusive": trigger + 1,
                    "left_padding_frames": max(0, 259 - trigger),
                    "right_padding_frames": 0,
                },
                "candidate_feature_sha256": feature_hash,
                "feature_sha256": feature_hash,
            })
            new_features.append(feature16)
            new_scores.append(score)
            new_feature_frames.append(trigger)
            new_score_frames.append(trigger)

    base_count = len(base.rows)
    for index, row in enumerate(rows):
        row["feature_index"] = index
    features = np.concatenate([
        np.asarray(base.features, dtype=np.float16),
        np.asarray(new_features, dtype=np.float16).reshape(-1, WINDOW_FRAMES, FEATURE_BINS),
    ])
    labels = np.concatenate([np.asarray(base.labels, dtype=np.int8), np.zeros(len(new_features), dtype=np.int8)])
    scores = np.concatenate([np.asarray(base.detector_scores, dtype=np.float32), np.asarray(new_scores, dtype=np.float32)])
    old_feature_frames = np.load(base.root / "detector_feature_frames.npy")
    old_score_frames = np.load(base.root / "detector_score_frames.npy")
    feature_frames = np.concatenate([old_feature_frames.astype(np.int32), np.asarray(new_feature_frames, dtype=np.int32)])
    score_frames = np.concatenate([old_score_frames.astype(np.int32), np.asarray(new_score_frames, dtype=np.int32)])

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    arrays = {
        "features.npy": features,
        "labels.npy": labels,
        "detector_scores.npy": scores,
        "detector_feature_frames.npy": feature_frames,
        "detector_score_frames.npy": score_frames,
    }
    for name, values in arrays.items():
        np.save(output / name, values, allow_pickle=False)
    source_manifest = {
        "schema_version": 1,
        "kind": "kizz_training_only_librispeech_hard_negative_sources",
        "training_eligible": True,
        "locked_musan_read": False,
        "source": "OpenSLR SLR31 Mini LibriSpeech train-clean-5",
        "license": "CC BY 4.0",
        "files": source_ledger,
        "counts": {"files": len(files), "candidates": len(new_features), "exposure_seconds": total_seconds},
        "detector": {"artifact": _binding(detector_model), "metadata": _binding(detector_metadata), "threshold_report": _binding(detector_threshold_report)},
    }
    source_manifest_path = output / "librispeech-hard-negative-sources.json"
    _atomic_json(source_manifest_path, source_manifest)

    corpus = json.loads(json.dumps(base.corpus))
    corpus["examples"] = rows
    corpus["array_sha256"] = {name: sha256_file(output / name) for name in arrays}
    corpus.setdefault("bindings", {})["base_candidate_corpus"] = _binding(base.corpus_path)
    corpus["bindings"]["training_hard_negative_sources"] = _binding(source_manifest_path)
    corpus["hard_negative_selection"].update(
        raw_training_count=int(corpus["hard_negative_selection"]["raw_training_count"]) + len(new_features),
        selected_training_count=int(corpus["hard_negative_selection"]["selected_training_count"]) + len(new_features),
        top_k=TOP_K_PER_UTTERANCE,
    )
    corpus["counts"]["selected_candidates"] = len(rows)
    corpus["counts"]["selected_negatives"] = int(np.sum(labels == 0))
    corpus["counts"]["selected_positives"] = int(np.sum(labels == 1))
    train_counts = corpus["counts"]["by_split"]["train"]
    train_counts["raw_detector_candidates"] += len(new_features)
    train_counts["raw_negative_candidates"] += len(new_features)
    train_counts["selected_negative_candidates"] += len(new_features)
    train_counts["source_examples"] += len(files)
    train_counts["exposure_seconds"] += total_seconds
    train_counts["negative_exposure_seconds"] += total_seconds
    train_counts["raw_candidate_rate_per_second"] = train_counts["raw_detector_candidates"] / train_counts["exposure_seconds"]
    train_counts["raw_candidate_rate_per_hour"] = train_counts["raw_candidate_rate_per_second"] * 3600
    train_counts["raw_negative_candidate_rate_per_hour"] = train_counts["raw_negative_candidates"] * 3600 / train_counts["negative_exposure_seconds"]
    corpus["training_hard_negative_extension"] = {
        "source_manifest": _binding(source_manifest_path),
        "base_rows": base_count,
        "appended_training_negatives": len(new_features),
        "locked_musan_used_for_training": False,
        "threshold": threshold,
        "threshold_provenance": threshold_provenance,
    }
    _atomic_json(output / "corpus.json", corpus)
    return {
        "output": str(output), "corpus_sha256": sha256_file(output / "corpus.json"),
        "source_files": len(files), "source_hours": total_seconds / 3600,
        "appended_training_negatives": len(new_features), "rows": len(rows),
        "detector_hops": total_hops,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--base-corpus-sha256", required=True)
    parser.add_argument("--librispeech-root", type=Path, required=True)
    parser.add_argument("--detector-metadata", type=Path, required=True)
    parser.add_argument("--detector-model", type=Path, required=True)
    parser.add_argument("--detector-threshold-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(mine(**vars(args)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
