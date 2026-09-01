import json
import tempfile
import unittest
from pathlib import Path

from tools.package_kizz_phoneme_student_firmware import (
    HEADER_FILENAME,
    MODEL_FILENAME,
    PROVENANCE_FILENAME,
    package,
    sha256_file,
)


class PackageKizzPhonemeStudentFirmwareTests(unittest.TestCase):
    def qualified_fixture(self, root: Path) -> Path:
        artifact_dir = root / "artifact"
        artifact_dir.mkdir()
        model = artifact_dir / "student.tflite"
        model.write_bytes(b"qualified-int8-model")
        contract = {
            "tokens": [f"token-{index}" for index in range(20)],
            "blank_id": 0,
            "canonical_path": [1, 2, 3],
            "collision_paths": {"collision": [1, 3, 2]},
        }
        contract_hash = "c" * 64
        artifact_metadata = artifact_dir / "firmware-artifact.json"
        artifact_metadata.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "artifact": {
                        "filename": model.name,
                        "sha256": sha256_file(model),
                        "bytes": model.stat().st_size,
                    },
                    "compact_phone_contract": contract,
                    "input": {"dtype": "int8", "shape": [1, 3, 40]},
                    "output": {"dtype": "uint8", "shape": [1, 1, 20]},
                    "decoder": {"contract_sha256": contract_hash},
                }
            )
        )
        qualification = root / "qualification.json"
        qualification.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "gate_scope": "student_deployment_qualification",
                    "qualified": True,
                    "failure_reasons": [],
                    "model": {"distillation_metadata_sha256": "d" * 64},
                    "artifact_metadata": {
                        "path": str(artifact_metadata),
                        "sha256": sha256_file(artifact_metadata),
                        "artifact_sha256": sha256_file(model),
                        "artifact_bytes": model.stat().st_size,
                    },
                    "decoder": {
                        "type": "deterministic_suffix_forward_sum_ctc",
                        "algorithm": "forward_sum_ctc",
                        "contract_sha256": contract_hash,
                        "beta": 0.0,
                        "window_lengths": [19, 23, 27, 32, 39, 47, 54],
                        "threshold_selection": "validation_only",
                    },
                    "compact_phone_contract": contract,
                    "threshold": {
                        "qualified": True,
                        "threshold": -1.25,
                        "selection": "validation_only",
                    },
                    "counts": {
                        "target_channel_positives": 24,
                        "false_wake_anchors": 62,
                        "false_wake_accepted": 0,
                    },
                    "results": {
                        "aligned_test": {"recall": 0.92},
                        "target_channel": {"recall": 0.92},
                    },
                    "continuous_negative": {
                        "qualified": True,
                        "exposure_hours": 100.0,
                        "false_accepts": 0,
                        "faph": 0.0,
                        "faph_upper_95": 0.03,
                    },
                }
            )
        )
        return qualification

    def test_packages_exact_qualified_artifact_and_generated_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qualification = self.qualified_fixture(root)
            output = root / "firmware"
            provenance = package(qualification, output)
            self.assertEqual(
                sha256_file(output / MODEL_FILENAME),
                provenance["model"]["sha256"],
            )
            header = (output / HEADER_FILENAME).read_text()
            self.assertIn("kRawScoreThreshold = -1.25f", header)
            self.assertIn("kCollisionBeta = 0.0f", header)
            self.assertIn("kDeploymentQualified = true", header)
            self.assertIn("kHardwareEvaluationOnly = false", header)
            self.assertIn('kDecoderAlgorithm[] = "forward_sum_ctc"', header)
            self.assertTrue((output / PROVENANCE_FILENAME).is_file())

    def test_explicit_hardware_evaluation_path_preserves_unqualified_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qualification = self.qualified_fixture(root)
            payload = json.loads(qualification.read_text())
            payload["qualified"] = False
            payload["failure_reasons"] = ["held_out_recall_below_minimum"]
            payload["threshold"] = {
                "qualified": False,
                "zero_false_accept_recall": 0.72,
                "zero_false_accept_threshold": -2.125,
            }
            payload["results"] = {
                "aligned_test": {"recall": 0.5},
                "target_channel": {"recall": 0.6},
            }
            payload["score_summary"] = {
                "validation": [
                    {"label": 1, "score": -2.0},
                    {"label": 0, "score": -2.2},
                ],
                "false_wakes": [{"label": 0, "score": -3.0}],
            }
            payload["continuous_negative"] = None
            qualification.write_text(json.dumps(payload))
            output = root / "firmware"
            provenance = package(
                qualification,
                output,
                experimental_hardware_evaluation=True,
            )
            self.assertEqual(
                provenance["deployment_status"],
                "experimental_hardware_evaluation",
            )
            header = (output / HEADER_FILENAME).read_text()
            self.assertIn("kRawScoreThreshold = -2.125f", header)
            self.assertIn("kDeploymentQualified = false", header)
            self.assertIn("kHardwareEvaluationOnly = true", header)

    def test_experimental_package_preserves_known_continuous_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qualification = self.qualified_fixture(root)
            payload = json.loads(qualification.read_text())
            payload["qualified"] = False
            payload["failure_reasons"] = ["continuous_negative_gate_failed"]
            payload["threshold"] = {
                "qualified": True,
                "zero_false_accept_recall": 0.72,
                "zero_false_accept_threshold": -2.125,
            }
            payload["score_summary"] = {
                "validation": [
                    {"label": 1, "score": -2.0},
                    {"label": 0, "score": -2.2},
                ],
                "false_wakes": [{"label": 0, "score": -3.0}],
            }
            payload["continuous_negative"] = {
                "qualified": False,
                "exposure_hours": 100.0,
                "false_accepts": 6,
                "faph": 0.06,
                "faph_upper_95": 0.1184,
            }
            qualification.write_text(json.dumps(payload))

            provenance = package(
                qualification,
                root / "firmware",
                experimental_hardware_evaluation=True,
            )

            self.assertEqual(
                provenance["qualification_summary"]["failure_reasons"],
                ["continuous_negative_gate_failed"],
            )
            self.assertEqual(
                provenance["qualification_summary"]["continuous_negative"][
                    "false_accepts"
                ],
                6,
            )
            self.assertFalse(provenance["qualification_summary"]["qualified"])

    def test_rejects_report_that_did_not_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qualification = self.qualified_fixture(root)
            payload = json.loads(qualification.read_text())
            payload["qualified"] = False
            payload["failure_reasons"] = ["continuous_negative_gate_failed"]
            qualification.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "did not pass"):
                package(qualification, root / "firmware")
            self.assertFalse((root / "firmware").exists())

    def test_rejects_tampered_model_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qualification = self.qualified_fixture(root)
            metadata = json.loads(qualification.read_text())["artifact_metadata"]
            artifact_metadata = Path(metadata["path"])
            artifact = json.loads(artifact_metadata.read_text())
            (artifact_metadata.parent / artifact["artifact"]["filename"]).write_bytes(
                b"tampered"
            )
            with self.assertRaisesRegex(ValueError, "bytes or hash drifted"):
                package(qualification, root / "firmware")
            self.assertFalse((root / "firmware").exists())


if __name__ == "__main__":
    unittest.main()
