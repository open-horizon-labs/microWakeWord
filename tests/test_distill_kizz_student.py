import json
import tempfile
import unittest
from pathlib import Path

from tools.distill_kizz_student import require_teacher_qualification, sha256_file


class DistillKizzStudentGateTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
