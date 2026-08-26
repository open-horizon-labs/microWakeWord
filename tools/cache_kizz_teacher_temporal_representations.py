#!/usr/bin/env python3
"""Cache frame-aligned hidden representations from a qualified Kizz teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from microwakeword.kizz_phoneme_teacher import resolve_hf_weights_path
from microwakeword.phoneme_student import (
    compact_phone_contract,
    student_output_times_seconds,
)
from tools.cache_kizz_phoneme_teacher_posteriors import (
    TARGET_SAMPLE_RATE,
    _load_audio,
    _rows,
    sha256_file,
)
from tools.distill_kizz_phoneme_student import (
    OUTPUT_FRAMES,
    student_flags_for_architecture,
)


def load_temporal_representation_cache(prefix: Path) -> tuple[dict, np.ndarray]:
    metadata = json.loads(prefix.with_suffix(".json").read_text())
    matrix = np.load(prefix.with_suffix(".npy"), mmap_mode="r")
    if (
        metadata.get("schema_version") != 1
        or metadata.get("representation")
        != "qualified_teacher_last_hidden_frame_aligned_train_pca"
        or list(matrix.shape) != metadata.get("shape")
        or matrix.ndim != 3
        or str(matrix.dtype) != metadata.get("dtype")
    ):
        raise ValueError("teacher temporal representation cache contract differs")
    unsigned = {key: value for key, value in metadata.items() if key != "cache_sha256"}
    digest = hashlib.sha256()
    digest.update(np.asarray(matrix).tobytes(order="C"))
    digest.update(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode())
    if digest.hexdigest() != metadata.get("cache_sha256"):
        raise ValueError("teacher temporal representation cache is stale or corrupt")
    return metadata, matrix


def _resample(
    values: np.ndarray, source_times: np.ndarray, target_times: np.ndarray
) -> np.ndarray:
    right = np.searchsorted(source_times, target_times, side="left")
    right = np.clip(right, 1, len(source_times) - 1)
    left = right - 1
    span = source_times[right] - source_times[left]
    weight = ((target_times - source_times[left]) / span).astype(np.float32)
    return values[left] * (1.0 - weight[:, None]) + values[right] * weight[:, None]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--teacher-qualification", type=Path, required=True)
    parser.add_argument("--posterior-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--student-architecture", default="dilated_temporal_memory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dimension", type=int, default=96)
    parser.add_argument("--pca-frame-stride", type=int, default=8)
    args = parser.parse_args()

    qualification = json.loads(args.teacher_qualification.read_text())
    if qualification.get("qualified") is not True:
        parser.error("teacher must be qualified")
    identity = qualification.get("model", {})
    weights = resolve_hf_weights_path(
        identity["id"], revision=identity["revision"], local_files_only=True
    )
    if sha256_file(weights) != identity.get("weights_sha256"):
        parser.error("qualified teacher weights changed")
    posterior = json.loads(args.posterior_cache.with_suffix(".json").read_text())
    timing = posterior["timing"]
    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

    device = torch.device(args.device)
    common = {"revision": identity["revision"], "local_files_only": True}
    processor = Wav2Vec2FeatureExtractor.from_pretrained(identity["id"], **common)
    model = Wav2Vec2ForCTC.from_pretrained(identity["id"], **common).to(device).eval()
    rows = _rows(args.manifest)
    train_indexes = [i for i, row in enumerate(rows) if row.get("split") == "train"]
    if not train_indexes:
        parser.error("manifest has no training rows")
    raw: dict[int, np.ndarray] = {}
    for completed, index in enumerate(train_indexes, 1):
        path = Path(rows[index]["path"])
        waveform, sample_rate = _load_audio(path)
        if sample_rate != TARGET_SAMPLE_RATE or sha256_file(path) != rows[index].get(
            "audio_sha256"
        ):
            raise ValueError(f"temporal representation audio drifted: {path}")
        inputs = processor(
            waveform, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt"
        )
        kwargs = {
            key: value.to(device)
            for key, value in inputs.items()
            if not key.startswith("_")
        }
        with torch.inference_mode():
            output = model(**kwargs, output_hidden_states=True)
        hidden = output.hidden_states[-1][0].detach().float().cpu().numpy()
        if hidden.ndim != 2 or not np.isfinite(hidden).all():
            raise ValueError("teacher temporal hidden state is invalid")
        raw[index] = hidden.astype(np.float16)
        if completed % 25 == 0 or completed == len(train_indexes):
            print(
                json.dumps({"cached": completed, "total": len(train_indexes)}),
                flush=True,
            )

    sampled = np.concatenate(
        [
            np.asarray(raw[i][:: args.pca_frame_stride], dtype=np.float32)
            for i in train_indexes
        ],
        axis=0,
    )
    center = sampled.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(sampled - center, full_matrices=False)
    basis = right[: args.dimension]
    pivots = np.argmax(np.abs(basis), axis=1)
    signs = np.sign(basis[np.arange(len(basis)), pivots])
    signs[signs == 0] = 1
    basis *= signs[:, None]

    contract = compact_phone_contract()
    target_times = student_output_times_seconds(
        student_flags_for_architecture(
            args.student_architecture, len(contract["tokens"])
        ),
        OUTPUT_FRAMES,
    )
    matrix = np.zeros((len(rows), OUTPUT_FRAMES, args.dimension), dtype=np.float16)
    for index in train_indexes:
        values = np.asarray(raw[index], dtype=np.float32)
        source_times = float(timing["frame_center_seconds"]) + np.arange(
            len(values)
        ) * float(timing["frame_stride_seconds"])
        projected = (values - center) @ basis.T
        matrix[index] = _resample(projected, source_times, target_times).astype(
            np.float16
        )

    metadata = {
        "schema_version": 1,
        "representation": "qualified_teacher_last_hidden_frame_aligned_train_pca",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "teacher_qualification": {
            "path": str(args.teacher_qualification.resolve()),
            "sha256": sha256_file(args.teacher_qualification),
        },
        "posterior_cache": {
            "prefix": str(args.posterior_cache.resolve()),
            "json_sha256": sha256_file(args.posterior_cache.with_suffix(".json")),
        },
        "model": identity,
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "train_indexes": train_indexes,
        "student_architecture": args.student_architecture,
        "student_output_times_seconds": target_times.tolist(),
        "projection": {
            "algorithm": "train_frame_stride_centered_svd_with_deterministic_sign",
            "dimension": args.dimension,
            "frame_stride": args.pca_frame_stride,
            "basis_sha256": hashlib.sha256(basis.tobytes(order="C")).hexdigest(),
            "center_sha256": hashlib.sha256(center.tobytes(order="C")).hexdigest(),
        },
    }
    digest = hashlib.sha256()
    digest.update(matrix.tobytes(order="C"))
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    metadata["cache_sha256"] = digest.hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=args.output.name, suffix=".npy", dir=args.output.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        np.save(temporary_path, matrix)
        os.replace(temporary_path, args.output.with_suffix(".npy"))
    finally:
        temporary_path.unlink(missing_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
