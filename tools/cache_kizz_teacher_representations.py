#!/usr/bin/env python3
"""Cache utterance-level hidden representations from a qualified teacher."""

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

from microwakeword.kizz_phoneme_teacher import load_hf_teacher, resolve_hf_weights_path
from tools.cache_kizz_phoneme_teacher_posteriors import (
    TARGET_SAMPLE_RATE,
    _load_audio,
    _rows,
    sha256_file,
)


def _hidden_mean(model, processor, waveform: np.ndarray, device) -> np.ndarray:
    inputs = processor(waveform, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")
    kwargs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
        if not key.startswith("_")
    }
    import torch

    with torch.inference_mode():
        output = model(**kwargs, output_hidden_states=True)
    hidden = output.hidden_states[-1][0].detach().float().cpu().numpy()
    if hidden.ndim != 2 or not len(hidden) or not np.isfinite(hidden).all():
        raise ValueError("teacher hidden representation is invalid")
    return hidden.mean(axis=0, dtype=np.float64).astype(np.float32)


def load_representation_cache(prefix: Path) -> tuple[dict, np.ndarray]:
    metadata = json.loads(prefix.with_suffix(".json").read_text())
    matrix = np.load(prefix.with_suffix(".npy"), mmap_mode="r")
    if metadata.get("schema_version") != 1 or list(matrix.shape) != metadata.get(
        "shape"
    ):
        raise ValueError("teacher representation cache contract differs")
    if str(matrix.dtype) != metadata.get("dtype") or matrix.ndim != 2:
        raise ValueError("teacher representation cache dtype or rank differs")
    unsigned = {key: value for key, value in metadata.items() if key != "cache_sha256"}
    digest = hashlib.sha256()
    digest.update(np.asarray(matrix).tobytes(order="C"))
    digest.update(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode())
    if digest.hexdigest() != metadata.get("cache_sha256"):
        raise ValueError("teacher representation cache is stale or corrupt")
    return metadata, matrix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--teacher-qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dimension", type=int, default=96)
    args = parser.parse_args()

    qualification = json.loads(args.teacher_qualification.read_text())
    if qualification.get("qualified") is not True:
        parser.error("teacher must be qualified")
    identity = qualification.get("model", {})
    if not all(identity.get(key) for key in ("id", "revision", "weights_sha256")):
        parser.error("teacher qualification lacks exact model identity")
    weights = resolve_hf_weights_path(
        identity["id"], revision=identity["revision"], local_files_only=True
    )
    if sha256_file(weights) != identity["weights_sha256"]:
        parser.error("qualified teacher weights changed")
    model, processor, _, device = load_hf_teacher(
        identity["id"],
        revision=identity["revision"],
        device=args.device,
        local_files_only=True,
    )

    rows = _rows(args.manifest)
    if not rows:
        parser.error("teacher manifest is empty")
    vectors = []
    source_hashes = []
    for index, row in enumerate(rows):
        path = Path(row["path"])
        waveform, sample_rate = _load_audio(path)
        if sample_rate != TARGET_SAMPLE_RATE:
            raise ValueError(f"audio must be {TARGET_SAMPLE_RATE} Hz: {path}")
        audio_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        declared_sha256 = row.get("audio_sha256")
        if declared_sha256 and declared_sha256 != audio_sha256:
            raise ValueError(f"manifest audio hash differs: {path}")
        vectors.append(_hidden_mean(model, processor, waveform, device))
        source_hashes.append(audio_sha256)
        if (index + 1) % 25 == 0 or index + 1 == len(rows):
            print(json.dumps({"cached": index + 1, "total": len(rows)}), flush=True)
    raw = np.asarray(vectors, dtype=np.float32)
    train_indexes = [
        index for index, row in enumerate(rows) if row.get("split") == "train"
    ]
    if not train_indexes:
        parser.error("teacher manifest has no training split")
    if args.dimension < 2 or args.dimension > min(len(train_indexes), raw.shape[1]):
        parser.error("--dimension is outside the train-rank bound")
    center = raw[train_indexes].mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(raw[train_indexes] - center, full_matrices=False)
    basis = right[: args.dimension]
    # Fix the arbitrary SVD sign so identical inputs produce identical bytes.
    pivots = np.argmax(np.abs(basis), axis=1)
    signs = np.sign(basis[np.arange(len(basis)), pivots])
    signs[signs == 0] = 1
    basis *= signs[:, None]
    matrix = ((raw - center) @ basis.T).astype(np.float16)
    metadata = {
        "schema_version": 1,
        "representation": "qualified_teacher_last_hidden_time_mean_train_pca",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "teacher_qualification": {
            "path": str(args.teacher_qualification.resolve()),
            "sha256": sha256_file(args.teacher_qualification),
        },
        "model": identity,
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "projection": {
            "algorithm": "train_split_centered_svd_with_deterministic_sign",
            "dimension": args.dimension,
            "train_examples": len(train_indexes),
            "basis_sha256": hashlib.sha256(basis.tobytes(order="C")).hexdigest(),
            "center_sha256": hashlib.sha256(center.tobytes(order="C")).hexdigest(),
        },
        "source_audio_sha256": source_hashes,
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
