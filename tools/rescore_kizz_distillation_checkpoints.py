#!/usr/bin/env python3
"""Re-select saved float checkpoints with deployment-equivalent CTC scoring."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from microwakeword.ordered_state_model import model as build_student
from microwakeword.phoneme_student import compact_phone_contract
from tools.distill_kizz_phoneme_student import (
    INPUT_SHAPE,
    _student_scores,
    checkpoint_selection_key,
    sha256_file,
    student_architecture_contract,
    student_flags_for_architecture,
)
from tools.qualify_kizz_phoneme_student import choose_validation_threshold


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distillation-metadata", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    metadata = json.loads(args.distillation_metadata.read_text())
    contract = compact_phone_contract()
    if metadata.get("compact_phone_contract") != contract:
        parser.error("distillation compact-phone contract differs")
    architecture_id = metadata.get("architecture", {}).get(
        "architecture_id", "control_mixconv"
    )
    if metadata.get("architecture") != student_architecture_contract(
        contract, architecture_id
    ):
        parser.error("distillation architecture contract differs")
    decoder = metadata.get("decoder", {}).get("contract", {})
    algorithm = decoder.get("algorithm")
    if algorithm not in ("forward_sum_ctc", "max_add_ctc_viterbi"):
        parser.error("distillation decoder is invalid")
    corpus = json.loads((args.corpus / "corpus.json").read_text())
    rows = corpus["examples"]
    features = np.load(args.corpus / "features.npy", mmap_mode="r")
    indexes = [
        index for index, row in enumerate(rows) if row.get("split") == "validation"
    ]
    selected_rows = [rows[index] for index in indexes]
    selected_features = np.asarray(features[indexes], dtype=np.float32)
    if not any(int(row["label"]) == 1 for row in selected_rows) or not any(
        int(row["label"]) == 0 for row in selected_rows
    ):
        parser.error("validation split needs both labels")

    model = build_student(
        student_flags_for_architecture(architecture_id, len(contract["tokens"])),
        INPUT_SHAPE,
        None,
    )
    ledger = []
    best_key = None
    best = None
    for original in metadata.get("validation_ledger", []):
        checkpoint = Path(original["checkpoint"]["path"])
        digest = sha256_file(checkpoint)
        if digest != original["checkpoint"]["sha256"]:
            raise ValueError(f"checkpoint hash differs: {checkpoint}")
        model.load_weights(checkpoint)
        scores = _student_scores(
            model,
            selected_features,
            contract,
            args.batch_size,
            decoder_algorithm=algorithm,
        )
        scored_rows = [
            {
                "label": int(row["label"]),
                "duration_seconds": float(row.get("duration_seconds", 0.0)),
                "score": float(score),
                "failure_reasons": [],
            }
            for row, score in zip(selected_rows, scores)
        ]
        point = choose_validation_threshold(scored_rows, min_recall=0.90, max_faph=0.10)
        positives = scores[
            np.asarray([int(row["label"]) == 1 for row in selected_rows])
        ]
        negatives = scores[
            np.asarray([int(row["label"]) == 0 for row in selected_rows])
        ]
        separation = (
            float(np.min(positives) - np.max(negatives))
            if np.isfinite(positives).all() and np.isfinite(negatives).all()
            else None
        )
        item = {
            "step": int(original["step"]),
            "checkpoint": {"path": str(checkpoint.resolve()), "sha256": digest},
            "operating_point": point,
            "separation": separation,
        }
        ledger.append(item)
        key = (
            *checkpoint_selection_key(
                point, float(point["zero_false_accept_recall"]), separation
            ),
            -int(original["step"]),
        )
        if best_key is None or key > best_key:
            best_key, best = key, item
        print(json.dumps(item), flush=True)
    if best is None:
        raise ValueError("distillation metadata has no validation checkpoints")
    report = {
        "schema_version": 1,
        "kind": "kizz_deployment_equivalent_checkpoint_selection",
        "source": {
            "distillation_metadata": str(args.distillation_metadata.resolve()),
            "distillation_metadata_sha256": sha256_file(args.distillation_metadata),
            "corpus": str((args.corpus / "corpus.json").resolve()),
            "corpus_sha256": sha256_file(args.corpus / "corpus.json"),
        },
        "scoring_fix": "log_softmax_then_deployment_suffix_forward_sum_ctc",
        "selection_key": [
            "qualified",
            "zero_false_accept_recall",
            "negative_false_accepts_at_recall_floor",
            "recall",
            "separation",
            "earliest_step",
        ],
        "selected": best,
        "ledger": ledger,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
