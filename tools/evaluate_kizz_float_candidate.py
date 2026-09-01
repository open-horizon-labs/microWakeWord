#!/usr/bin/env python3
"""Evaluate a float Kizz student on fixed clip evidence before quantization."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from microwakeword.ordered_state_model import model as build_student
from microwakeword.phoneme_student import compact_phone_contract
from tools.distill_kizz_phoneme_student import (
    INPUT_SHAPE,
    sha256_file,
    student_architecture_contract,
    student_flags_for_architecture,
)
from tools.qualify_kizz_phoneme_student import (
    _apply_threshold,
    _validate_evidence,
    choose_validation_threshold,
    score_rows,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distillation-metadata", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--checkpoint-selection", type=Path)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--target-channel-manifest", type=Path, required=True)
    parser.add_argument("--false-wake-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-validation-recall", type=float, default=0.90)
    parser.add_argument("--max-validation-faph", type=float, default=0.10)
    parser.add_argument("--min-aligned-accepted", type=int, default=23)
    parser.add_argument("--min-target-accepted", type=int, default=20)
    parser.add_argument("--expected-aligned", type=int, default=26)
    parser.add_argument("--expected-target", type=int, default=24)
    parser.add_argument("--expected-false-wakes", type=int, default=62)
    parser.add_argument("--false-wake-context-seconds", type=float, default=2.0)
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
    weights_hash = sha256_file(args.weights)
    if args.checkpoint_selection:
        selection = json.loads(args.checkpoint_selection.read_text())
        if (
            selection.get("source", {}).get("distillation_metadata_sha256")
            != sha256_file(args.distillation_metadata)
            or selection.get("selected", {}).get("checkpoint", {}).get("sha256")
            != weights_hash
        ):
            parser.error("checkpoint selection is not bound to metadata and weights")
    elif metadata.get("student", {}).get("weights_sha256") != weights_hash:
        parser.error(
            "weights are not the selected student bound by distillation metadata"
        )
    decoder_algorithm = metadata.get("decoder", {}).get("contract", {}).get("algorithm")
    if decoder_algorithm not in ("forward_sum_ctc", "max_add_ctc_viterbi"):
        parser.error("distillation decoder contract is invalid")

    paths = {
        "validation": args.validation_manifest,
        "test": args.test_manifest,
        "target": args.target_channel_manifest,
        "false_wakes": args.false_wake_manifest,
    }
    groups, evidence_contracts = _validate_evidence(paths)
    model = build_student(
        student_flags_for_architecture(architecture_id, len(contract["tokens"])),
        INPUT_SHAPE,
        None,
    )
    model.load_weights(args.weights)
    validation, _, _ = score_rows(
        groups["validation"],
        model,
        contract,
        beta=0.0,
        decoder_algorithm=decoder_algorithm,
    )
    point = choose_validation_threshold(
        validation,
        min_recall=args.min_validation_recall,
        max_faph=args.max_validation_faph,
    )
    # A failed validation gate cannot qualify, but its deterministic zero-FP
    # threshold still gives a comparable held-out recall diagnostic.
    threshold = float(
        point.get("threshold", point.get("zero_false_accept_threshold", math.inf))
    )
    aligned, _, _ = score_rows(
        groups["test"],
        model,
        contract,
        beta=0.0,
        decoder_algorithm=decoder_algorithm,
    )
    target, _, _ = score_rows(
        groups["target"],
        model,
        contract,
        beta=0.0,
        decoder_algorithm=decoder_algorithm,
    )
    false_wakes, _, _ = score_rows(
        groups["false_wakes"],
        model,
        contract,
        beta=0.0,
        decoder_algorithm=decoder_algorithm,
        false_wake_context_seconds=args.false_wake_context_seconds,
    )
    results = {
        "aligned": _apply_threshold(aligned, threshold),
        "target": _apply_threshold(target, threshold),
        "false_wakes": _apply_threshold(false_wakes, threshold),
    }
    reasons = []
    if not point.get("qualified"):
        reasons.append("validation_operating_point_not_qualified")
    if len(aligned) != args.expected_aligned:
        reasons.append("aligned_count_not_exact")
    if results["aligned"]["accepted"] < args.min_aligned_accepted:
        reasons.append("aligned_recall_below_float_gate")
    if len(target) != args.expected_target:
        reasons.append("target_count_not_exact")
    if results["target"]["accepted"] < args.min_target_accepted:
        reasons.append("target_recall_below_float_gate")
    if len(false_wakes) != args.expected_false_wakes:
        reasons.append("false_wake_count_not_exact")
    if results["false_wakes"]["accepted"]:
        reasons.append("locked_false_wake_accepted")
    if any(
        row.get("failure_reasons")
        for collection in (validation, aligned, target, false_wakes)
        for row in collection
    ):
        reasons.append("evidence_scoring_failure")
    report = {
        "schema_version": 1,
        "kind": "kizz_float_candidate_clip_qualification",
        "qualified_for_continuous_evaluation": not reasons,
        "failure_reasons": reasons,
        "model": {
            "recipe": metadata.get("recipe"),
            "distillation_metadata": str(args.distillation_metadata.resolve()),
            "distillation_metadata_sha256": sha256_file(args.distillation_metadata),
            "weights": str(args.weights.resolve()),
            "weights_sha256": weights_hash,
            "checkpoint_selection": (
                {
                    "path": str(args.checkpoint_selection.resolve()),
                    "sha256": sha256_file(args.checkpoint_selection),
                }
                if args.checkpoint_selection
                else None
            ),
        },
        "selection": {
            "source": "validation_only",
            "threshold": threshold,
            "used_fallback_zero_fp_threshold": not point.get("qualified"),
            "validation_operating_point": point,
        },
        "gates": {
            "aligned": {
                "minimum_accepted": args.min_aligned_accepted,
                "expected": args.expected_aligned,
            },
            "target": {
                "minimum_accepted": args.min_target_accepted,
                "expected": args.expected_target,
            },
            "false_wakes": {
                "maximum_accepted": 0,
                "expected": args.expected_false_wakes,
            },
        },
        "evidence": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                **evidence_contracts[name],
            }
            for name, path in paths.items()
        },
        "results": results,
        "scores": {
            name: [
                {
                    "source_id": row.get("source_id"),
                    "label": row.get("label"),
                    "score": row.get("score"),
                    "failure_reasons": row.get("failure_reasons", []),
                }
                for row in rows
            ]
            for name, rows in (
                ("validation", validation),
                ("aligned", aligned),
                ("target", target),
                ("false_wakes", false_wakes),
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "qualified_for_continuous_evaluation": not reasons,
                "failure_reasons": reasons,
                "selection": report["selection"],
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not reasons else 2


if __name__ == "__main__":
    raise SystemExit(main())
