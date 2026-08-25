import json
import tempfile
import unittest
from pathlib import Path

import yaml

from microwakeword.data import FeatureHandler
from microwakeword.kizz_batch_mixture import (
    validate_declared_mixture,
    validate_mixture_guard,
    validate_realized_mixture,
)
from tools.write_stratified_training_config import stratified_config


GUARD = {
    "schema_version": 1,
    "require_all_active_groups": True,
    "minimum_realized_samples": 10,
    "tolerances": {"sample_share": 0.02, "weighted_pressure_share": 0.03},
    "expected": {
        "classes": {
            "positive": {"sample_share": 0.5, "weighted_pressure_share": 0.5},
            "negative": {"sample_share": 0.5, "weighted_pressure_share": 0.5},
        },
        "groups": {
            "canonical": {"sample_share": 0.5, "weighted_pressure_share": 0.5},
            "public_speech": {"sample_share": 0.5, "weighted_pressure_share": 0.5},
        },
    },
}


class KizzBatchMixtureTest(unittest.TestCase):
    def test_canonical_v3_recipe_declares_a_closed_six_group_mixture(self):
        recipe = yaml.safe_load(
            (
                Path(__file__).resolve().parents[1]
                / "recipes/kizz/batch-mixture-canonical-v3.yaml"
            ).read_text()
        )
        validate_mixture_guard(recipe["mixture_guard"])
        groups = recipe["mixture_guard"]["expected"]["groups"]
        self.assertEqual(
            set(groups),
            {
                "canonical_positive",
                "device_positive",
                "phonetic_collision",
                "public_speech",
                "music",
                "background_noise",
            },
        )

    def test_declared_plan_rejects_public_speech_hidden_in_a_90_percent_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base.yaml"
            base.write_text("features: []\n")
            plan = root / "plan.yaml"
            plan.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "sampling_groups": {"canonical": 0.1, "public_speech": 0.9},
                        "mixture_guard": GUARD,
                        "sources": [
                            {
                                "features_dir": "/canonical",
                                "source_name": "canonical",
                                "truth": True,
                                "group": "canonical",
                            },
                            {
                                "features_dir": "/public",
                                "source_name": "public",
                                "truth": False,
                                "group": "public_speech",
                            },
                        ],
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "sampling share"):
                stratified_config(base, plan)

    def test_declared_guard_rejects_undeclared_active_group(self):
        summary = {
            "classes": {
                "positive": {"sampling_share": 0.5, "weighted_pressure_share": 0.5},
                "negative": {"sampling_share": 0.5, "weighted_pressure_share": 0.5},
            },
            "groups": {
                "canonical": {"sampling_share": 0.5, "weighted_pressure_share": 0.5},
                "public_speech": {
                    "sampling_share": 0.4,
                    "weighted_pressure_share": 0.4,
                },
                "music": {"sampling_share": 0.1, "weighted_pressure_share": 0.1},
            },
        }
        with self.assertRaisesRegex(ValueError, "undeclared"):
            validate_declared_mixture(summary, GUARD)

    def test_guard_rejects_weighted_pressure_targets_that_do_not_sum_to_one(self):
        invalid = json.loads(json.dumps(GUARD))
        invalid["expected"]["groups"]["canonical"]["weighted_pressure_share"] = 0.4
        with self.assertRaisesRegex(ValueError, "weighted pressure shares"):
            validate_mixture_guard(invalid)

    def test_realized_count_and_pressure_drift_fail_closed(self):
        ledger = {
            "total_samples": 100,
            "mixture_guard": GUARD,
            "realized_classes": {
                "positive": {"share": 0.5, "weighted_pressure_share": 0.5},
                "negative": {"share": 0.5, "weighted_pressure_share": 0.5},
            },
            "realized_groups": {
                "canonical": {"share": 0.1, "weighted_pressure_share": 0.5},
                "public_speech": {"share": 0.9, "weighted_pressure_share": 0.5},
            },
        }
        with self.assertRaisesRegex(ValueError, "realized group sampling share"):
            validate_realized_mixture(ledger, GUARD)

        ledger["realized_groups"]["canonical"]["share"] = 0.5
        ledger["realized_groups"]["public_speech"]["share"] = 0.5
        ledger["realized_groups"]["public_speech"]["weighted_pressure_share"] = 0.9
        with self.assertRaisesRegex(ValueError, "weighted pressure share"):
            validate_realized_mixture(ledger, GUARD)

    def test_realized_ledger_cannot_be_checked_against_a_different_recipe(self):
        ledger = {
            "total_samples": 100,
            "mixture_guard": {**GUARD, "minimum_realized_samples": 999},
            "realized_classes": {},
            "realized_groups": {},
        }
        with self.assertRaisesRegex(ValueError, "does not exactly match"):
            validate_realized_mixture(ledger, GUARD)

    def test_feature_handler_ledger_contains_realized_classes_and_guard(self):
        handler = FeatureHandler.__new__(FeatureHandler)
        handler.sampling_group_weights = {"canonical": 0.5, "public_speech": 0.5}
        handler.training_sampling_counts = {
            ("canonical", "wake"): 5,
            ("public_speech", "speech"): 5,
        }
        handler.training_weighted_pressure = {
            ("canonical", "wake"): 5.0,
            ("public_speech", "speech"): 5.0,
        }
        handler.sampling_source_config = {
            "wake": {"truth": True, "penalty_weight": 1.0},
            "speech": {"truth": False, "penalty_weight": 1.0},
        }
        handler.mixture_guard = GUARD
        ledger = handler.sampling_ledger()
        self.assertEqual(ledger["realized_classes"]["positive"]["samples"], 5)
        self.assertEqual(ledger["mixture_guard"], GUARD)


if __name__ == "__main__":
    unittest.main()
