#!/usr/bin/env python3
"""Recover a self-contained runtime contract from a deployed Kizz detector.

This does not recreate lost training or float-model equivalence evidence.  It
binds the exact deployed TFLite bytes, their live tensor contract, the firmware
provenance that named the artifact, and the generic ordered-state decoder so a
new corpus can be traced without pretending the original conversion directory
survived.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from microwakeword.ordered_state import OrderedStateTopology
from microwakeword.phoneme_student import student_output_times_seconds
from microwakeword.wake_phrase import KIZZ_CONTROL
from tools.distill_kizz_student import student_flags


MODEL_FILENAME = "kizz_control_detector_ordered_state_streaming_int8.tflite"
METADATA_FILENAME = "firmware-artifact.json"
PROVENANCE_FILENAME = "deployed-cascade-provenance.json"
REFERENCE_FILENAME = "ordered_state_reference.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _tensor_spec(detail: dict[str, Any]) -> dict[str, Any]:
    scale, zero = detail["quantization"]
    return {
        "shape": [int(value) for value in detail["shape"]],
        "dtype": np.dtype(detail["dtype"]).name,
        "quantization": [float(scale), int(zero)],
    }


def recover(model: Path, cascade_provenance: Path, output: Path) -> dict[str, Any]:
    import tensorflow as tf

    model = model.expanduser().resolve()
    cascade_provenance = cascade_provenance.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output}")
    if not model.is_file() or not cascade_provenance.is_file():
        raise FileNotFoundError("deployed model and cascade provenance are required")

    provenance = json.loads(cascade_provenance.read_text(encoding="utf-8"))
    detector = provenance.get("models", {}).get("detector", {})
    model_hash = sha256_file(model)
    if (
        detector.get("sha256") != model_hash
        or detector.get("bytes") != model.stat().st_size
        or detector.get("tensor_contract") != "int8[1,3,40] -> uint8[1,1,12]"
    ):
        raise ValueError("deployed detector differs from cascade provenance")

    interpreter = tf.lite.Interpreter(model_path=str(model))
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    tensors = interpreter.get_tensor_details()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("deployed detector must expose one input and one output")
    input_spec = _tensor_spec(inputs[0])
    output_spec = _tensor_spec(outputs[0])
    if input_spec["shape"] != [1, 3, 40] or input_spec["dtype"] != "int8":
        raise ValueError("deployed detector input contract drift")
    if output_spec["shape"] != [1, 1, 12] or output_spec["dtype"] != "uint8":
        raise ValueError("deployed detector output contract drift")
    dynamic = sum(
        any(int(value) < 0 for value in detail["shape_signature"])
        for detail in tensors
    )
    if dynamic:
        raise ValueError("deployed detector contains dynamic tensor shapes")

    output.mkdir(parents=True)
    artifact_path = output / MODEL_FILENAME
    provenance_path = output / PROVENANCE_FILENAME
    reference_path = output / REFERENCE_FILENAME
    shutil.copyfile(model, artifact_path)
    shutil.copyfile(cascade_provenance, provenance_path)
    reference_source = Path(__file__).resolve().parents[1] / "microwakeword" / "ordered_state.py"
    shutil.copyfile(reference_source, reference_path)

    topology = OrderedStateTopology(tuple(KIZZ_CONTROL.phones), 1)
    topology_payload = {
        "phrase_id": KIZZ_CONTROL.phrase_id,
        "phones": list(topology.phones),
        "states_per_phone": topology.states_per_phone,
        "state_count": topology.state_count,
        "state_names": list(topology.state_names),
        "background_index": topology.background_index,
        "silence_index": topology.silence_index,
        "first_ordered_state_index": topology.first_ordered_state_index,
    }
    decoder_arguments = {
        "from_logits": True,
        "state_evidence_floor": None,
        "self_loop_probability": 0.6,
        "next_state_probability": 0.4,
    }
    decoder_contract = {
        "topology": topology_payload,
        "algorithm": "ordered_state_sequence_score_numpy",
        "arguments": decoder_arguments,
    }

    flags = student_flags(topology.state_count)
    phase = 2
    stride = 3
    calls = 87
    warmup = calls - 66
    output_times = [
        float(value) for value in student_output_times_seconds(flags, 66)
    ]
    equivalence = {
        "algorithm": "generic_ordered_state_sequence_score_numpy_v1",
        "from_logits": True,
        "offline_shape": [66, topology.state_count],
        "streaming_calls_per_example": calls,
        "streaming_warmup_outputs_discarded": warmup,
        "evidence_scope": "recovered_deployed_int8_runtime_contract_only",
        "float_model_equivalence_recovered": False,
        "training_provenance_recovered": False,
    }
    equivalence["evidence_sha256"] = sha256_json(equivalence)

    metadata = {
        "schema_version": 2,
        "kind": "kizz_control_ordered_state_detector_streaming_int8",
        "student_role": "permissive_detector_candidate_generator",
        "deployment_qualification": False,
        "recovery_scope": {
            "runtime_contract_only": True,
            "lost_training_directory_reconstructed": False,
            "new_qualification_required": True,
        },
        "artifact": {
            "filename": artifact_path.name,
            "bytes": artifact_path.stat().st_size,
            "sha256": sha256_file(artifact_path),
        },
        "source": {
            "deployed_cascade_provenance": {
                "path": str(provenance_path),
                "sha256": sha256_file(provenance_path),
            }
        },
        "topology": topology_payload,
        "decoder": {
            "algorithm": "ordered_state_sequence_score_numpy",
            "contract_version": 1,
            "arguments": decoder_arguments,
            "score_semantics": "maximum_complete_left_to_right_log_odds_path",
            "contract_sha256": sha256_json(decoder_contract),
            "reference_module": str(reference_path),
            "reference_module_sha256": sha256_file(reference_path),
        },
        "timeline": {
            "frontend_feature_step_seconds": 0.01,
            "frontend_window_seconds": 0.03,
            "offline_input_frames": 260,
            "offline_output_frames": 66,
            "stream_input_frames_per_call": stride,
            "stream_hop_seconds": 0.03,
            "stream_phase_offset_frames": phase,
            "stream_phase_priming": "zero_prefix_then_observed_prefix",
            "streaming_calls_per_260_frame_example": calls,
            "streaming_warmup_outputs_discarded": warmup,
            "offline_output_times_seconds": output_times,
            "causal_tail_alignment": "derived_from_calls_minus_offline_output_frames",
        },
        "tensor_contracts": {
            "input": input_spec,
            "output": output_spec,
            "output_semantics": "unnormalized_ordered_state_logits",
        },
        "static_memory_contract": {
            "batch_size": 1,
            "fixed_input_shape": True,
            "fixed_output_shape": True,
            "dynamic_tensor_shapes_forbidden": True,
            "external_state_tensor_count": 0,
            "persistent_state": "internal_tflite_variables",
            "tensor_audit": {
                "input_count": len(inputs),
                "output_count": len(outputs),
                "dynamic_shape_tensor_count": dynamic,
                "tensor_count": len(tensors),
            },
        },
        "equivalence": equivalence,
        "limitations": [
            "runtime recovery does not reproduce lost float-model equivalence evidence",
            "runtime recovery does not qualify the detector or cascade for deployment",
        ],
    }
    (output / METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cascade-provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    metadata = recover(args.model, args.cascade_provenance, args.output)
    print(json.dumps({"artifact": metadata["artifact"], "recovery_scope": metadata["recovery_scope"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
