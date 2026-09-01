import json
import tempfile
import unittest
from pathlib import Path

from tools.bind_kizz_detector_teacher import bind_detector_teacher, sha256_file


class BindKizzDetectorTeacherTests(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        bindings = {}
        for name in (
            "feature_provenance",
            "balance_manifest",
            "balance_report",
            "batch_mixture_recipe",
            "batch_mixture_ledger",
            "positive_source_balance_report",
            "positive_features",
            "positive_targets",
        ):
            path = root / name
            path.write_text(name, encoding="utf-8")
            bindings[name] = str(path)
            bindings[f"{name}_sha256"] = sha256_file(path)
        checkpoint = root / "checkpoint-000200.weights.h5"
        checkpoint.write_bytes(b"selected")
        (root / "best.weights.h5").write_bytes(b"selected")
        selected = {
            "opportunity_recall": 0.97,
            "false_accepts": 12,
            "separation": -2.0,
            "validation_loss": 0.2,
            "threshold": 1.5,
            "zero_false_accepts": False,
        }
        report = {
            "schema_version": 1,
            "checkpoint_selection": "validation_min_false_accepts_subject_to_recall_floor",
            "selection_min_recall": 0.95,
            "checkpoint_selection_ledger": [
                {
                    "step": 100,
                    "positive_count": 100,
                    "negative_count": 100,
                    "selected": {**selected, "false_accepts": 30},
                },
                {
                    "step": 200,
                    "positive_count": 100,
                    "negative_count": 100,
                    "selected": selected,
                },
            ],
            "best_validation": selected,
            "topology": {"states_per_phone": 1},
            **bindings,
        }
        path = root / "teacher-training.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        return path

    def test_binds_exact_selected_checkpoint_for_distillation_only(self):
        with tempfile.TemporaryDirectory() as directory:
            report = bind_detector_teacher(self._fixture(Path(directory)))
            self.assertTrue(report["qualified"])
            self.assertTrue(report["eligible_for_detector_distillation"])
            self.assertFalse(report["deployment_qualification"])
            self.assertEqual(report["selected_checkpoint"]["step"], 200)
            self.assertEqual(report["selection"]["opportunity_recall"], 0.97)
            self.assertEqual(report["selection"]["false_candidate_fraction"], 0.12)

    def test_rejects_best_weights_or_reference_hash_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._fixture(root)
            (root / "best.weights.h5").write_bytes(b"drift")
            with self.assertRaisesRegex(ValueError, "best teacher weights differ"):
                bind_detector_teacher(path)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._fixture(root)
            (root / "feature_provenance").write_text("drift", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "feature_provenance hash drift"):
                bind_detector_teacher(path)

    def test_fails_closed_when_candidate_pressure_exceeds_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            report = bind_detector_teacher(
                self._fixture(Path(directory)),
                maximum_false_candidate_fraction=0.10,
            )
            self.assertFalse(report["qualified"])
            self.assertIn(
                "validation_false_candidate_fraction_above_limit",
                report["failure_reasons"],
            )


if __name__ == "__main__":
    unittest.main()
