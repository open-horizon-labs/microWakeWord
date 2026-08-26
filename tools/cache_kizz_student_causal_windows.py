#!/usr/bin/env python3
"""Cache a frozen student's decision curve at every causal endpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from microwakeword.ordered_state_model import model as build_student
from microwakeword.phoneme_student import compact_phone_contract
from tools.cache_kizz_teacher_causal_windows import causal_suffix_score_grid
from tools.cache_kizz_teacher_sequence_scores import _canonical_hash, _sha256_file
from tools.distill_kizz_phoneme_student import (
    INPUT_SHAPE,
    OUTPUT_FRAMES,
    WINDOW_LENGTHS_FRAMES,
    student_architecture_contract,
    student_flags_for_architecture,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--distillation", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    corpus_path = args.corpus / "corpus.json"
    corpus = json.loads(corpus_path.read_text())
    rows = corpus["examples"]
    distillation = json.loads(args.distillation.read_text())
    contract = compact_phone_contract()
    architecture = distillation.get("architecture", {})
    architecture_id = architecture.get("architecture_id", "control_mixconv")
    if distillation.get(
        "compact_phone_contract"
    ) != contract or architecture != student_architecture_contract(
        contract, architecture_id
    ):
        parser.error("source student contract differs")
    checkpoint_sha = _sha256_file(args.checkpoint)
    known = {
        item.get("checkpoint", {}).get("sha256")
        for item in distillation.get("validation_ledger", [])
    }
    if checkpoint_sha not in known:
        parser.error("checkpoint is absent from the source validation ledger")
    features = np.load(args.corpus / "features.npy", mmap_mode="r")
    if len(features) != len(rows):
        parser.error("corpus feature count differs")
    model = build_student(
        student_flags_for_architecture(architecture_id, len(contract["tokens"])),
        INPUT_SHAPE,
        None,
    )
    model.load_weights(args.checkpoint)
    logits = []
    for start in range(0, len(features), args.batch_size):
        logits.append(
            np.asarray(
                model(
                    np.asarray(
                        features[start : start + args.batch_size], dtype=np.float32
                    ),
                    training=False,
                ),
                dtype=np.float32,
            )
        )
    values = np.concatenate(logits, axis=0)
    if values.shape[1:] != (OUTPUT_FRAMES, len(contract["tokens"])):
        parser.error("source student output geometry differs")
    endpoints = np.arange(min(WINDOW_LENGTHS_FRAMES), OUTPUT_FRAMES + 1, dtype=np.int32)
    compact = causal_suffix_score_grid(
        values,
        contract,
        end_frames=endpoints,
        window_lengths=WINDOW_LENGTHS_FRAMES,
        beta=0.0,
    )
    # Expand to direct output-frame indexing. Prefixes shorter than the minimum
    # deployment window are invalid and deliberately unavailable for sampling.
    scores = {}
    for key, matrix in compact.items():
        fill = False if matrix.dtype == bool else np.nan
        expanded = np.full((len(rows), OUTPUT_FRAMES), fill, dtype=matrix.dtype)
        expanded[:, endpoints - 1] = matrix
        scores[key] = expanded
    scores["student_end_frame"] = endpoints
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), **scores)
    metadata = {
        "schema_version": 1,
        "representation": "frozen_student_causal_endpoint_decisions",
        "corpus": {
            "path": str(corpus_path.resolve()),
            "sha256": _sha256_file(corpus_path),
            "features_sha256": _sha256_file(args.corpus / "features.npy"),
        },
        "source_student": {
            "distillation": {
                "path": str(args.distillation.resolve()),
                "sha256": _sha256_file(args.distillation),
            },
            "checkpoint": {
                "path": str(args.checkpoint.resolve()),
                "sha256": checkpoint_sha,
            },
            "recipe": distillation["recipe"],
            "architecture": architecture,
        },
        "compact_phone_contract_sha256": _canonical_hash(contract),
        "scorer": {
            "algorithm": "forward_sum_ctc",
            "window_lengths_frames": list(WINDOW_LENGTHS_FRAMES),
            "beta": 0.0,
            "window_selection": "suffix_only_at_each_causal_student_endpoint",
        },
        "counts": {
            "examples": len(rows),
            "student_frames": OUTPUT_FRAMES,
            "valid_endpoints": len(endpoints),
        },
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(metadata["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
