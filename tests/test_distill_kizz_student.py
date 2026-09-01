import json
import tempfile
import unittest
from pathlib import Path

from tools.distill_kizz_student import (
    detector_cache_teacher,
    require_detector_teacher_gate,
    require_teacher_qualification,
    sha256_file,
    student_flags,
)


class DistillKizzStudentGateTests(unittest.TestCase):
    def test_small_detector_architecture_preserves_geometry(self):
        reference = student_flags(12)
        small = student_flags(12, "control_mixconv_small")
        self.assertEqual(small.mixconv_kernel_sizes, reference.mixconv_kernel_sizes)
        self.assertEqual(small.stride, reference.stride)
        self.assertEqual(small.num_states, reference.num_states)
        self.assertEqual(small.pointwise_filters, "48,48,48,48")
        self.assertEqual(small.first_conv_filters, 24)

    def reports(self, root: Path) -> tuple[Path, Path, Path]:
        weights = root / "teacher.weights.h5"
        weights.write_bytes(b"teacher")
        model_sha = sha256_file(weights)
        teacher = root / "teacher.json"
        teacher.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "gate_scope": "teacher_clip_and_anchor_prequalification",
                    "qualified": True,
                    "reasons": [],
                    "model_sha256": model_sha,
                    "validation": {"operating_point": {"threshold": 1.25}},
                }
            )
        )
        continuous = root / "continuous.json"
        continuous.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "gate_scope": "untouched_continuous_qualification",
                    "qualified": True,
                    "model_sha256": model_sha,
                    "test_is_untouched": True,
                    "config": {
                        "min_negative_exposure_hours": 100.0,
                        "max_faph_upper_95": 0.1,
                    },
                    "qualification": {
                        "qualified": True,
                        "threshold": 1.0,
                        "recall": 0.9,
                        "negative_exposure_seconds": 360000.0,
                        "false_accepts_per_hour_upper_95": 0.05,
                        "locked_anchor_false_accepts": 0,
                    },
                }
            )
        )
        return weights, teacher, continuous

    def test_requires_both_teacher_and_continuous_hard_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            weights, teacher, continuous = self.reports(Path(directory))
            teacher_report, continuous_report = require_teacher_qualification(
                teacher, weights, continuous
            )
            self.assertTrue(teacher_report["qualified"])
            self.assertTrue(continuous_report["qualified"])

    def test_rejects_short_exposure_and_wrong_model(self):
        with tempfile.TemporaryDirectory() as directory:
            weights, teacher, continuous = self.reports(Path(directory))
            report = json.loads(continuous.read_text())
            report["qualification"]["negative_exposure_seconds"] = 359999.0
            continuous.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "less than 100 hours"):
                require_teacher_qualification(teacher, weights, continuous)
            report["qualification"]["negative_exposure_seconds"] = 360000.0
            report["model_sha256"] = "0" * 64
            continuous.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "different weights"):
                require_teacher_qualification(teacher, weights, continuous)

    def test_rejects_anchor_accept_and_permissive_confidence_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            weights, teacher, continuous = self.reports(Path(directory))
            report = json.loads(continuous.read_text())
            report["qualification"]["locked_anchor_false_accepts"] = 1
            continuous.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "locked anchor"):
                require_teacher_qualification(teacher, weights, continuous)
            report["qualification"]["locked_anchor_false_accepts"] = 0
            report["qualification"]["false_accepts_per_hour_upper_95"] = 0.11
            continuous.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "FAPH upper bound"):
                require_teacher_qualification(teacher, weights, continuous)

    def test_detector_gate_permits_distillation_but_not_deployment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "best.weights.h5"
            weights.write_bytes(b"detector-teacher")
            training = root / "teacher-training.json"
            training.write_text("{}", encoding="utf-8")
            feature = root / "feature-provenance.json"
            feature.write_text("{}", encoding="utf-8")
            gate = root / "detector-gate.json"
            gate.write_text(
                json.dumps(
                    {
                        "gate_scope": "teacher_detector_synthetic_bootstrap_prequalification",
                        "qualified": True,
                        "eligible_for_detector_distillation": True,
                        "deployment_qualification": False,
                        "eligible_for_final_deployment": False,
                        "selected_checkpoint": {
                            "best_weights_path": str(weights.resolve()),
                            "best_weights_sha256": sha256_file(weights),
                        },
                        "selection": {
                            "split": "validation",
                            "minimum_recall": 0.95,
                            "opportunity_recall": 0.97,
                            "threshold": -1.0,
                        },
                        "training_report": {
                            "path": str(training.resolve()),
                            "sha256": sha256_file(training),
                        },
                        "bindings": {
                            "feature_provenance": {
                                "path": str(feature.resolve()),
                                "sha256": sha256_file(feature),
                            }
                        },
                        "topology": {
                            "phones": ["k"],
                            "states_per_phone": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            report = require_detector_teacher_gate(gate, weights)
            self.assertTrue(report["eligible_for_detector_distillation"])
            self.assertFalse(report["deployment_qualification"])

    def test_detector_gate_rejects_recall_or_provenance_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "best.weights.h5"
            weights.write_bytes(b"detector-teacher")
            training = root / "teacher-training.json"
            training.write_text("{}", encoding="utf-8")
            payload = {
                "gate_scope": "teacher_detector_synthetic_bootstrap_prequalification",
                "qualified": True,
                "eligible_for_detector_distillation": True,
                "deployment_qualification": False,
                "eligible_for_final_deployment": False,
                "selected_checkpoint": {
                    "best_weights_path": str(weights.resolve()),
                    "best_weights_sha256": sha256_file(weights),
                },
                "selection": {
                    "split": "validation",
                    "minimum_recall": 0.95,
                    "opportunity_recall": 0.94,
                    "threshold": -1.0,
                },
                "training_report": {
                    "path": str(training.resolve()),
                    "sha256": sha256_file(training),
                },
                "bindings": {},
                "topology": {"phones": ["k"], "states_per_phone": 1},
            }
            gate = root / "detector-gate.json"
            gate.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "recall contract"):
                require_detector_teacher_gate(gate, weights)
            payload["selection"]["opportunity_recall"] = 0.97
            gate.write_text(json.dumps(payload), encoding="utf-8")
            training.write_text('{"drift": true}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "provenance hash drift"):
                require_detector_teacher_gate(gate, weights)

    def test_detector_cache_binds_teacher_and_all_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "best.weights.h5"
            weights.write_bytes(b"teacher")
            outputs = {}
            for name in ("features", "targets", "labels", "teacher_logits"):
                path = root / f"{name}.npy"
                path.write_bytes(name.encode())
                outputs[name] = {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                }
            metadata = {
                "schema_version": 2,
                "cache_role": "detector_student_distillation",
                "deployment_qualification": False,
                "selected_teacher": {
                    "best_weights": {
                        "path": str(weights.resolve()),
                        "sha256": sha256_file(weights),
                    }
                },
                "outputs": outputs,
            }
            selected, binding = detector_cache_teacher(metadata)
            self.assertEqual(selected, weights.resolve())
            self.assertEqual(binding["sha256"], sha256_file(weights))
            (root / "teacher_logits.npy").write_bytes(b"drift")
            with self.assertRaisesRegex(ValueError, "teacher_logits hash drift"):
                detector_cache_teacher(metadata)


if __name__ == "__main__":
    unittest.main()
