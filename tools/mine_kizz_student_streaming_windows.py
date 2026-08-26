#!/usr/bin/env python3
"""Mine each corpus clip's hardest streaming window with a frozen student."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from microwakeword.ordered_state_model import model as build_student
from microwakeword.phoneme_student import compact_phone_contract
from tools.cache_kizz_teacher_sequence_scores import (
    _canonical_hash,
    _sha256_file,
    _split_report,
    forward_sum_sliding_scores,
)
from tools.distill_kizz_phoneme_student import (
    INPUT_SHAPE,
    WINDOW_LENGTHS_FRAMES,
    student_architecture_contract,
)
from tools.distill_kizz_student import student_flags


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
    if distillation.get("compact_phone_contract") != contract:
        raise ValueError("source student compact vocabulary differs")
    if distillation.get("architecture") != student_architecture_contract(contract):
        raise ValueError("source student architecture differs")
    known_checkpoints = {
        item.get("checkpoint", {}).get("sha256")
        for item in distillation.get("validation_ledger", [])
    }
    checkpoint_sha = _sha256_file(args.checkpoint)
    if checkpoint_sha not in known_checkpoints:
        raise ValueError("mining checkpoint is not in the source run ledger")
    features = np.load(args.corpus / "features.npy", mmap_mode="r")
    if len(features) != len(rows):
        raise ValueError("corpus features and metadata length differ")
    model = build_student(
        student_flags(len(contract["tokens"])), INPUT_SHAPE, None
    )
    model.load_weights(args.checkpoint)
    logits = []
    for start in range(0, len(features), args.batch_size):
        logits.append(
            np.asarray(
                model(
                    np.asarray(
                        features[start : start + args.batch_size],
                        dtype=np.float32,
                    ),
                    training=False,
                ),
                dtype=np.float32,
            )
        )
    values = np.concatenate(logits, axis=0)
    scores = forward_sum_sliding_scores(
        values,
        contract,
        window_lengths=WINDOW_LENGTHS_FRAMES,
        hop=1,
        beta=0.0,
        batch_size=args.batch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), **scores)
    metadata = {
        "schema_version": 1,
        "representation": "student_streaming_window_hard_mining",
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
        },
        "compact_phone_contract_sha256": _canonical_hash(contract),
        "scorer": {
            "algorithm": "forward_sum_ctc",
            "window_lengths_frames": list(WINDOW_LENGTHS_FRAMES),
            "hop_frames": 1,
            "beta": 0.0,
            "window_selection": "filter_margin_then_max_canonical_then_margin",
        },
        "counts": {"examples": len(rows), "student_frames": int(values.shape[1])},
        "split_reports": {
            split: _split_report(rows, scores, split)
            for split in ("train", "validation", "test")
        },
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
