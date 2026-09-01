#!/usr/bin/env python3
"""Package a qualified Kizz student or an explicit hardware-evaluation candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Sequence


MODEL_FILENAME = "hiphi_kizz_ordered.tflite"
HEADER_FILENAME = "kizz_control_model_contract.h"
PROVENANCE_FILENAME = "kizz_control_model.provenance.json"
EXPECTED_INPUT_SHAPE = [1, 3, 40]
EXPECTED_OUTPUT_SHAPE = [1, 1, 20]
EXPECTED_WINDOWS = [19, 23, 27, 32, 39, 47, 54]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _cpp_float(value: float) -> str:
    literal = format(value, ".9g")
    if "." not in literal and "e" not in literal.lower():
        literal += ".0"
    return literal + "f"


def validate_package_inputs(
    qualification_path: Path,
    *,
    experimental_hardware_evaluation: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, Any]]:
    qualification_path = qualification_path.resolve()
    qualification = _load(qualification_path)
    if qualification.get("schema_version") != 2:
        raise ValueError("student qualification schema must be version 2")
    if qualification.get("gate_scope") != "student_deployment_qualification":
        raise ValueError("report is not a student deployment qualification")
    deployment_qualified = (
        qualification.get("qualified") is True
        and not qualification.get("failure_reasons")
    )
    if not experimental_hardware_evaluation and not deployment_qualified:
        raise ValueError("student deployment qualification did not pass")

    point = qualification.get("threshold") or {}
    if experimental_hardware_evaluation and not deployment_qualified:
        threshold = _finite(
            point.get("zero_false_accept_threshold"),
            "experimental zero-false-accept raw threshold",
        )
        if float(point.get("zero_false_accept_recall") or 0.0) < 0.70:
            raise ValueError(
                "experimental student validation recall is below 70 percent at zero false accepts"
            )
    else:
        threshold = _finite(point.get("threshold"), "qualified raw threshold")
        if point.get("qualified") is not True or point.get("selection") != "validation_only":
            raise ValueError("student threshold was not selected from validation only")
    decoder = qualification.get("decoder") or {}
    beta = _finite(decoder.get("beta"), "collision beta")
    if (
        decoder.get("type") != "deterministic_suffix_forward_sum_ctc"
        or decoder.get("algorithm") != "forward_sum_ctc"
    ):
        raise ValueError("student decoder type differs from firmware")
    if decoder.get("window_lengths") != EXPECTED_WINDOWS:
        raise ValueError("student decoder windows differ from firmware")
    if decoder.get("threshold_selection") != "validation_only":
        raise ValueError("student decoder threshold provenance drifted")

    continuous = qualification.get("continuous_negative")
    if not experimental_hardware_evaluation:
        if not isinstance(continuous, dict) or continuous.get("qualified") is not True:
            raise ValueError("student continuous-negative qualification did not pass")
        if _finite(continuous.get("exposure_hours"), "continuous exposure") < 100.0:
            raise ValueError("student continuous-negative exposure is below 100 hours")
        if _finite(continuous.get("faph_upper_95"), "continuous FAPH upper bound") > 0.1:
            raise ValueError("student continuous-negative confidence bound exceeds 0.1 FAPH")
    # Experimental hardware evaluation is allowed to carry known continuous-
    # negative failures.  They remain explicit in the generated provenance and
    # the contract header remains fail-closed as not deployment-qualified.
    # Requiring zero events here made it impossible to flash the exact candidate
    # whose device behavior we need to measure, and encouraged packaging an
    # older report that omitted the continuous corpus instead.

    counts = qualification.get("counts") or {}
    if int(counts.get("target_channel_positives", 0)) != 24:
        raise ValueError("student qualification does not contain exactly 24 target-channel positives")
    if int(counts.get("false_wake_anchors", 0)) != 62:
        raise ValueError("student qualification does not contain exactly 62 false-wake anchors")
    if int(counts.get("false_wake_accepted", -1)) != 0:
        raise ValueError("student accepted a locked false wake")
    results = qualification.get("results") or {}
    if not experimental_hardware_evaluation:
        if float((results.get("aligned_test") or {}).get("recall") or 0.0) < 0.9:
            raise ValueError("student aligned-test recall is below 90 percent")
        if float((results.get("target_channel") or {}).get("recall") or 0.0) < 0.9:
            raise ValueError("student target-channel recall is below 90 percent")
    else:
        score_summary = qualification.get("score_summary") or {}
        false_wake_accepts = sum(
            1
            for row in score_summary.get("false_wakes", [])
            if row.get("score") is not None and float(row["score"]) >= threshold
        )
        validation_negative_accepts = sum(
            1
            for row in score_summary.get("validation", [])
            if int(row.get("label", -1)) == 0
            and row.get("score") is not None
            and float(row["score"]) >= threshold
        )
        if false_wake_accepts or validation_negative_accepts:
            raise ValueError(
                "experimental zero-false-accept threshold accepted negative evidence"
            )

    artifact_ref = qualification.get("artifact_metadata") or {}
    artifact_metadata_path = Path(str(artifact_ref.get("path", ""))).resolve()
    if not artifact_metadata_path.is_file():
        raise ValueError("qualified artifact metadata does not exist")
    if sha256_file(artifact_metadata_path) != artifact_ref.get("sha256"):
        raise ValueError("qualified artifact metadata hash drifted")
    artifact = _load(artifact_metadata_path)
    if artifact.get("schema_version") != 2:
        raise ValueError("firmware artifact schema must be version 2")
    if (artifact.get("input") or {}).get("dtype") != "int8" or (
        artifact.get("input") or {}
    ).get("shape") != EXPECTED_INPUT_SHAPE:
        raise ValueError("firmware model input contract is not int8[1,3,40]")
    if (artifact.get("output") or {}).get("dtype") != "uint8" or (
        artifact.get("output") or {}
    ).get("shape") != EXPECTED_OUTPUT_SHAPE:
        raise ValueError("firmware model output contract is not uint8[1,1,20]")

    artifact_info = artifact.get("artifact") or {}
    artifact_path = (
        artifact_metadata_path.parent / str(artifact_info.get("filename", ""))
    ).resolve()
    if artifact_path.parent != artifact_metadata_path.parent.resolve() or not artifact_path.is_file():
        raise ValueError("qualified INT8 artifact path is invalid")
    artifact_hash = sha256_file(artifact_path)
    if (
        artifact_hash != artifact_info.get("sha256")
        or artifact_hash != artifact_ref.get("artifact_sha256")
        or artifact_path.stat().st_size != int(artifact_info.get("bytes", -1))
        or artifact_path.stat().st_size != int(artifact_ref.get("artifact_bytes", -1))
    ):
        raise ValueError("qualified INT8 artifact bytes or hash drifted")

    contract = qualification.get("compact_phone_contract")
    if not isinstance(contract, dict) or len(contract.get("tokens", [])) != 20:
        raise ValueError("student compact-phone contract must contain 20 outputs")
    if artifact.get("compact_phone_contract") != contract:
        raise ValueError("qualification and artifact compact-phone contracts differ")
    if decoder.get("contract_sha256") != (artifact.get("decoder") or {}).get(
        "contract_sha256"
    ):
        raise ValueError("qualification and artifact decoder contracts differ")
    contract_hash = str(decoder.get("contract_sha256", ""))
    if len(contract_hash) != 64:
        raise ValueError("qualification lacks an exact decoder contract hash")
    distillation_hash = str(
        (qualification.get("model") or {}).get("distillation_metadata_sha256", "")
    )
    if len(distillation_hash) != 64:
        raise ValueError("qualification lacks an exact distillation metadata hash")

    return qualification, artifact, artifact_path, {
        "threshold": threshold,
        "beta": beta,
        "deployment_qualified": deployment_qualified,
        "hardware_evaluation_only": not deployment_qualified,
        "contract_sha256": contract_hash,
        "distillation_metadata_sha256": distillation_hash,
    }


def package(
    qualification_path: Path,
    output: Path,
    *,
    experimental_hardware_evaluation: bool = False,
) -> dict[str, Any]:
    qualification, artifact, artifact_path, deployment = validate_package_inputs(
        qualification_path,
        experimental_hardware_evaluation=experimental_hardware_evaluation,
    )
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_output = output / MODEL_FILENAME
    shutil.copyfile(artifact_path, model_output)
    model_hash = sha256_file(model_output)
    qualification_hash = sha256_file(qualification_path.resolve())
    artifact_metadata_path = Path(qualification["artifact_metadata"]["path"]).resolve()
    artifact_metadata_hash = sha256_file(artifact_metadata_path)
    distillation_hash = deployment["distillation_metadata_sha256"]

    header = f"""#pragma once

#include <cstddef>

// Generated by tools/package_kizz_phoneme_student_firmware.py. Do not edit.
namespace kizz_control_deployment {{
inline constexpr float kRawScoreThreshold = {_cpp_float(deployment['threshold'])};
inline constexpr float kCollisionBeta = {_cpp_float(deployment['beta'])};
inline constexpr float kDisplayProbabilityCutoff = 0.70f;
inline constexpr bool kDeploymentQualified = {'true' if deployment['deployment_qualified'] else 'false'};
inline constexpr bool kHardwareEvaluationOnly = {'true' if deployment['hardware_evaluation_only'] else 'false'};
inline constexpr std::size_t kOutputCount = 20;
inline constexpr char kModelSha256[] = "{model_hash}";
inline constexpr char kQualificationSha256[] = "{qualification_hash}";
inline constexpr char kArtifactMetadataSha256[] = "{artifact_metadata_hash}";
inline constexpr char kDistillationMetadataSha256[] = "{distillation_hash}";
inline constexpr char kDecoderContractSha256[] = "{deployment['contract_sha256']}";
inline constexpr char kDecoderAlgorithm[] = "forward_sum_ctc";
}}  // namespace kizz_control_deployment
"""
    (output / HEADER_FILENAME).write_text(header)
    provenance = {
        "schema_version": 1,
        "deployment_status": (
            "qualified"
            if deployment["deployment_qualified"]
            else "experimental_hardware_evaluation"
        ),
        "model": {"filename": MODEL_FILENAME, "sha256": model_hash, "bytes": model_output.stat().st_size},
        "contract_header": {"filename": HEADER_FILENAME, "sha256": sha256_file(output / HEADER_FILENAME)},
        "student_qualification": {"path": str(qualification_path.resolve()), "sha256": qualification_hash},
        "artifact_metadata": {"path": str(artifact_metadata_path), "sha256": artifact_metadata_hash},
        "distillation_metadata_sha256": distillation_hash,
        "decoder": {"algorithm": "forward_sum_ctc", "raw_score_threshold": deployment["threshold"], "beta": deployment["beta"], "contract_sha256": deployment["contract_sha256"], "window_lengths": EXPECTED_WINDOWS},
        "tensor_contract": {"input": artifact["input"], "output": artifact["output"]},
        "qualification_summary": {
            "qualified": qualification.get("qualified") is True,
            "failure_reasons": list(qualification.get("failure_reasons") or []),
            "counts": qualification["counts"],
            "threshold": qualification["threshold"],
            "continuous_negative": (
                {
                    key: qualification["continuous_negative"].get(key)
                    for key in (
                        "exposure_hours",
                        "false_accepts",
                        "faph",
                        "faph_upper_95",
                        "qualified",
                    )
                }
                if isinstance(qualification.get("continuous_negative"), dict)
                else None
            ),
        },
    }
    (output / PROVENANCE_FILENAME).write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    return provenance


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--experimental-hardware-evaluation",
        action="store_true",
        help=(
            "Package a non-production candidate only when its validation-only "
            "zero-false-accept point has at least 70% recall and rejects every "
            "locked validation and false-wake negative."
        ),
    )
    args = parser.parse_args(argv)
    print(
        json.dumps(
            package(
                args.qualification,
                args.output,
                experimental_hardware_evaluation=args.experimental_hardware_evaluation,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
