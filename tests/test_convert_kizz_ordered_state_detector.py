import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from microwakeword.ordered_state import OrderedStateTopology
from microwakeword.wake_phrase import KIZZ_CONTROL
from tools.convert_kizz_ordered_state_detector import (
    ARTIFACT_FILENAME,
    METADATA_FILENAME,
    ConversionRuntime,
    TensorSpec,
    convert,
    sha256_file,
    sha256_json,
    validate_inputs,
)


class FakeBackend:
    def __init__(
        self,
        *,
        input_shape=(1, 3, 40),
        output_shape=None,
        int8_delta=0.01,
    ):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.int8_delta = float(int8_delta)

    def convert(
        self,
        *,
        flags,
        topology,
        weights,
        representative_features,
        calibration_indices,
        phase_offset,
        work_dir,
    ):
        del weights, representative_features, calibration_indices, work_dir
        states = topology.state_count
        calls = 1 + (260 - phase_offset) // flags.stride
        warmup = calls - 66

        def offline(sample):
            base = np.linspace(-2.0, 2.0, 66 * states, dtype=np.float32)
            return (base.reshape(1, 66, states) + float(sample[0, 0]) * 1e-4)

        def streaming(sample):
            output = offline(sample)[0]
            return np.concatenate(
                [np.zeros((warmup, states), dtype=np.float32), output], axis=0
            )

        def int8(sample):
            output = streaming(sample)
            output[warmup:] += self.int8_delta
            return output

        output_shape = self.output_shape or (1, 1, states)
        return ConversionRuntime(
            artifact=b"TFL3-fake-ordered-state-detector",
            input_spec=TensorSpec(self.input_shape, "int8", (0.25, -3)),
            output_spec=TensorSpec(output_shape, "uint8", (0.05, 127)),
            run_float=offline,
            run_streaming_float=streaming,
            run_streaming_int8=int8,
            tensor_audit={
                "tensor_count": 17,
                "declared_tensor_bytes_sum": 4096,
                "dynamic_shape_tensor_count": 0,
                "variable_tensor_count": 4,
                "input_count": 1,
                "output_count": 1,
            },
            model_parameters=54321,
            framework={"fake": "1"},
        )


class ConvertKizzOrderedStateDetectorTests(unittest.TestCase):
    def fixture(self, root: Path):
        topology = OrderedStateTopology(tuple(KIZZ_CONTROL.phones), 1)
        teacher_weights = root / "teacher.weights.h5"
        teacher_weights.write_bytes(b"teacher")
        teacher_training = root / "teacher-training.json"
        teacher_training.write_text("{}", encoding="utf-8")
        feature_provenance = root / "feature-provenance.json"
        feature_provenance.write_text("{}", encoding="utf-8")
        gate = root / "detector-gate.json"
        gate_payload = {
            "schema_version": 1,
            "gate_scope": "teacher_detector_synthetic_bootstrap_prequalification",
            "qualified": True,
            "eligible_for_detector_distillation": True,
            "deployment_qualification": False,
            "eligible_for_final_deployment": False,
            "selected_checkpoint": {
                "best_weights_path": str(teacher_weights),
                "best_weights_sha256": sha256_file(teacher_weights),
            },
            "selection": {
                "split": "validation",
                "minimum_recall": 0.95,
                "opportunity_recall": 0.97,
                "threshold": -4.0,
            },
            "training_report": {
                "path": str(teacher_training),
                "sha256": sha256_file(teacher_training),
            },
            "bindings": {
                "feature_provenance": {
                    "path": str(feature_provenance),
                    "sha256": sha256_file(feature_provenance),
                }
            },
            "topology": {
                "phones": list(topology.phones),
                "states_per_phone": 1,
                "state_names": list(topology.state_names),
            },
        }
        gate.write_text(json.dumps(gate_payload), encoding="utf-8")

        cache_prefix = root / "cache"
        outputs = {}
        cache_hashes = {}
        for filename in (
            "features.npy",
            "targets.npy",
            "labels.npy",
            "teacher_logits.npy",
        ):
            path = root / filename
            path.write_bytes(filename.encode("utf-8"))
            digest = sha256_file(path)
            outputs[filename.removesuffix(".npy")] = {
                "path": str(path),
                "sha256": digest,
            }
            cache_hashes[filename] = digest
        cache = cache_prefix.with_suffix(".json")
        cache_payload = {
            "schema_version": 2,
            "cache_role": "detector_student_distillation",
            "deployment_qualification": False,
            "topology": {
                "phones": list(topology.phones),
                "states_per_phone": 1,
                "state_count": topology.state_count,
            },
            "selected_teacher": {
                "best_weights": {
                    "path": str(teacher_weights),
                    "sha256": sha256_file(teacher_weights),
                }
            },
            "teacher_training": {
                "path": str(teacher_training),
                "sha256": sha256_file(teacher_training),
            },
            "feature_provenance": {
                "path": str(feature_provenance),
                "sha256": sha256_file(feature_provenance),
            },
            "outputs": outputs,
        }
        cache_payload["cache_sha256"] = sha256_json(outputs)
        cache.write_text(json.dumps(cache_payload), encoding="utf-8")

        weights = root / "best.weights.h5"
        weights.write_bytes(b"student")
        representative = root / "representative.npy"
        np.save(
            representative,
            np.random.default_rng(231).normal(size=(4, 260, 40)).astype(np.float32),
        )
        training = root / "distillation-training.json"
        metadata = {
            "schema_version": 1,
            "model": "ordered_state_causal_student_distilled",
            "student_role": "permissive_detector_candidate_generator",
            "deployment_qualification": False,
            "input_shape": [260, 40],
            "output_shape": [66, topology.state_count],
            "topology": {
                "phones": list(topology.phones),
                "states_per_phone": 1,
                "state_count": topology.state_count,
            },
            "cache_prefix": str(cache_prefix),
            "cache_files_sha256": cache_hashes,
            "detector_teacher_gate": str(gate),
            "detector_teacher_gate_sha256": sha256_file(gate),
            "teacher_qualification": None,
            "teacher_qualification_sha256": None,
            "continuous_qualification": None,
            "continuous_qualification_sha256": None,
            "student": {
                "selected_checkpoint": "best",
                "weights": str(weights),
                "weights_sha256": sha256_file(weights),
            },
        }
        training.write_text(json.dumps(metadata), encoding="utf-8")
        float_qualification = root / "float-detector-qualification.json"
        float_qualification.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "evaluation": "kizz_control_float_ordered_state_detector",
                    "qualified_for_detector_conversion": True,
                    "deployment_qualification": False,
                    "failure_reasons": [],
                    "model": {
                        "distillation_metadata": str(training),
                        "distillation_metadata_sha256": sha256_file(training),
                        "weights": str(weights),
                        "weights_sha256": sha256_file(weights),
                    },
                    "topology": metadata["topology"],
                    "threshold_selection": {
                        "fit_split": "validation",
                        "test_used_for_selection": False,
                        "minimum_recall": 0.95,
                        "maximum_false_candidate_fraction": 0.20,
                        "threshold": -7.25,
                        "opportunity_recall": 0.97,
                        "false_candidate_fraction": 0.10,
                    },
                    "test": {
                        "opportunity_recall": 0.95,
                        "false_candidate_fraction": 0.12,
                    },
                }
            ),
            encoding="utf-8",
        )
        return argparse.Namespace(
            distillation_training=training,
            weights=weights,
            representative_features=representative,
            float_qualification=float_qualification,
            output=root / "output",
            calibration_examples=3,
            equivalence_examples=3,
        )

    def test_fake_conversion_emits_bound_schema_v2_detector_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            report = convert(args, backend=FakeBackend())
            artifact = args.output / ARTIFACT_FILENAME
            metadata = args.output / METADATA_FILENAME
            self.assertEqual(report["schema_version"], 2)
            self.assertEqual(
                report["student_role"], "permissive_detector_candidate_generator"
            )
            self.assertFalse(report["deployment_qualification"])
            self.assertEqual(report["tensor_contracts"]["input"]["shape"], [1, 3, 40])
            self.assertEqual(
                report["tensor_contracts"]["output"]["shape"], [1, 1, 12]
            )
            self.assertEqual(
                report["artifact"]["sha256"], hashlib.sha256(artifact.read_bytes()).hexdigest()
            )
            self.assertEqual(
                report["source"]["representative_features"]["sha256"],
                sha256_file(args.representative_features),
            )
            self.assertEqual(
                report["equivalence"]["algorithm"],
                "generic_ordered_state_sequence_score_numpy_v1",
            )
            self.assertEqual(
                report["decoder"]["provisional_float_threshold"]["threshold"],
                -7.25,
            )
            self.assertEqual(
                report["decoder"]["provisional_float_threshold"][
                    "qualification_report_sha256"
                ],
                sha256_file(args.float_qualification),
            )
            self.assertTrue(
                report["static_memory_contract"]["hardware_high_water_measurement_required"]
            )
            self.assertEqual(json.loads(metadata.read_text()), report)

    def test_fake_conversion_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root)
            first = convert(args, backend=FakeBackend())
            args.output = root / "second-output"
            second = convert(args, backend=FakeBackend())
            self.assertEqual(first, second)
            self.assertEqual(
                (root / "output" / METADATA_FILENAME).read_bytes(),
                (root / "second-output" / METADATA_FILENAME).read_bytes(),
            )

    def test_rejects_role_and_deployment_drift_before_output(self):
        for field, value, message in (
            ("student_role", "single_stage_wake_word", "student_role"),
            ("deployment_qualification", True, "deployment_qualification"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                args = self.fixture(Path(directory))
                payload = json.loads(args.distillation_training.read_text())
                payload[field] = value
                args.distillation_training.write_text(json.dumps(payload))
                with self.assertRaisesRegex(ValueError, message):
                    convert(args, backend=FakeBackend())
                self.assertFalse(args.output.exists())

    def test_rejects_selected_weight_hash_or_path_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            args.weights.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, r"selected.weights"):
                validate_inputs(
                    args.distillation_training,
                    args.weights,
                    args.representative_features,
                    args.float_qualification,
                )

    def test_rejects_cache_hash_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            (Path(directory) / "teacher_logits.npy").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash drift"):
                validate_inputs(
                    args.distillation_training,
                    args.weights,
                    args.representative_features,
                    args.float_qualification,
                )

    def test_rejects_gate_hash_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            gate = Path(directory) / "detector-gate.json"
            gate.write_text(gate.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "gate.*hash drift"):
                validate_inputs(
                    args.distillation_training,
                    args.weights,
                    args.representative_features,
                    args.float_qualification,
                )

    def test_rejects_topology_drift_across_training_cache_or_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            payload = json.loads(args.distillation_training.read_text())
            payload["topology"]["phones"][-1] = "wrong"
            args.distillation_training.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "not Kizz Control"):
                validate_inputs(
                    args.distillation_training,
                    args.weights,
                    args.representative_features,
                    args.float_qualification,
                )

    def test_requires_qualified_validation_only_float_report(self):
        mutations = (
            ("qualified_for_detector_conversion", False, "not qualified"),
            ("deployment_qualification", True, "non-deployment"),
        )
        for field, value, message in mutations:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                args = self.fixture(Path(directory))
                report = json.loads(args.float_qualification.read_text())
                report[field] = value
                args.float_qualification.write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, message):
                    validate_inputs(
                        args.distillation_training,
                        args.weights,
                        args.representative_features,
                        args.float_qualification,
                    )

    def test_rejects_float_report_model_or_threshold_contract_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            report = json.loads(args.float_qualification.read_text())
            report["model"]["weights_sha256"] = "0" * 64
            args.float_qualification.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "different selected weights"):
                validate_inputs(
                    args.distillation_training,
                    args.weights,
                    args.representative_features,
                    args.float_qualification,
                )
        for field, value in (
            ("fit_split", "test"),
            ("test_used_for_selection", True),
            ("minimum_recall", 0.94),
            ("maximum_false_candidate_fraction", 0.21),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                args = self.fixture(Path(directory))
                report = json.loads(args.float_qualification.read_text())
                report["threshold_selection"][field] = value
                args.float_qualification.write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, "threshold-selection"):
                    validate_inputs(
                        args.distillation_training,
                        args.weights,
                        args.representative_features,
                        args.float_qualification,
                    )

    def test_rejects_representative_and_tflite_shape_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            np.save(args.representative_features, np.zeros((2, 259, 40), np.float32))
            with self.assertRaisesRegex(ValueError, r"\[N, 260, 40\]"):
                convert(args, backend=FakeBackend())
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, r"input shape.*\[1, 3, 40\]"):
                convert(args, backend=FakeBackend(input_shape=(1, 4, 40)))
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "output shape"):
                convert(args, backend=FakeBackend(output_shape=(1, 1, 11)))

    def test_rejects_int8_score_equivalence_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "int8_streaming failed"):
                convert(args, backend=FakeBackend(int8_delta=20.0))


if __name__ == "__main__":
    unittest.main()
