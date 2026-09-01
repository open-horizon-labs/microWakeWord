import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.package_kizz_control_cascade import package


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PackageKizzCascadeTests(unittest.TestCase):
    def fixture(self, root: Path, *, recall=1.0, faph=0.388):
        models = {}
        metadata = {}
        thresholds = {}
        for role in ("detector", "compact", "ordered"):
            models[role] = root / f"{role}.tflite"
            models[role].write_bytes(f"model:{role}".encode())
            metadata[role] = root / f"{role}.metadata.json"
            metadata[role].write_text(json.dumps({"role": role}), encoding="utf-8")
            thresholds[role] = root / f"{role}.threshold.json"
            thresholds[role].write_text(json.dumps({"role": role}), encoding="utf-8")
        hashes = {role: digest(path) for role, path in models.items()}
        positive = root / "positive.json"
        positive.write_text(
            json.dumps(
                {
                    "test_scored_after_threshold_frozen": True,
                    "locked_audio_used_for_tuning": False,
                    "detector_threshold": {"value": -18.2},
                    "test": {
                        "threshold_frozen_before_audio_access": True,
                        "recall": recall,
                        "accepted_candidates": 12,
                        "detector_candidates": 12,
                    },
                    "bindings": {
                        "detector": {"sha256": hashes["detector"]},
                        "compact": {"sha256": hashes["compact"]},
                    },
                }
            ),
            encoding="utf-8",
        )
        continuous = root / "continuous.json"
        continuous.write_text(
            json.dumps(
                {
                    "kind": "kizz_control_int8_continuous_negative_cascade_v1",
                    "bindings": {
                        role: {"sha256": value} for role, value in hashes.items()
                    },
                    "policy": {
                        "detector_threshold": -18.2,
                        "verifier_logit_threshold": 0.0,
                        "ordered_verifier_score_threshold": -19.3,
                    },
                    "metrics": {
                        "exposure_hours": 100.47,
                        "detector_candidates": 19105,
                        "compact_verifier_accepts": 991,
                        "compact_verifier_acceptance_fraction": 0.0519,
                        "accepted_false_wakes": 39,
                        "accepted_false_wakes_per_hour": faph,
                        "accepted_false_wake_rate_confidence": {
                            "one_sided_upper_95_per_hour": 0.507
                        },
                    },
                    "physical_hardware_proof": {
                        "present": False,
                        "remaining": ["exact-artifact physical soak"],
                    },
                }
            ),
            encoding="utf-8",
        )
        return argparse.Namespace(
            detector_model=models["detector"],
            compact_model=models["compact"],
            ordered_model=models["ordered"],
            detector_metadata=metadata["detector"],
            compact_metadata=metadata["compact"],
            ordered_metadata=metadata["ordered"],
            detector_threshold=thresholds["detector"],
            compact_threshold=thresholds["compact"],
            ordered_threshold=thresholds["ordered"],
            positive_report=positive,
            continuous_report=continuous,
            output=root / "package",
            minimum_test_recall=0.95,
            minimum_negative_hours=100.0,
            formal_faph=0.1,
            accepted_practical_faph=0.4,
            accept_observed_operating_point=True,
        )

    def test_packages_bound_models_without_claiming_hardware_qualification(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            manifest = package(args)
            self.assertTrue((args.output / "cascade.json").is_file())
            self.assertTrue(manifest["qualification"]["host_cascade_evidence_passed"])
            self.assertFalse(manifest["qualification"]["formal_gate_passed"])
            self.assertFalse(manifest["qualification"]["physical_hardware_qualified"])
            self.assertTrue(
                manifest["qualification"][
                    "practical_operating_point_explicitly_accepted"
                ]
            )

    def test_rejects_test_recall_below_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary), recall=0.5)
            with self.assertRaisesRegex(ValueError, "below"):
                package(args)
            self.assertFalse(args.output.exists())

    def test_requires_explicit_acceptance_above_formal_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            args.accept_observed_operating_point = False
            with self.assertRaisesRegex(ValueError, "accept-observed"):
                package(args)

    def test_formal_pass_does_not_require_exception_acceptance(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary), faph=0.05)
            args.accept_observed_operating_point = False
            manifest = package(args)
            self.assertTrue(manifest["qualification"]["formal_gate_passed"])
            self.assertFalse(
                manifest["qualification"][
                    "practical_operating_point_explicitly_accepted"
                ]
            )


if __name__ == "__main__":
    unittest.main()
