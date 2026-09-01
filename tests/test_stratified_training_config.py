import json
import tempfile
import unittest
from pathlib import Path

import yaml

from tools.write_stratified_training_config import expand_source, stratified_config


class StratifiedTrainingConfigTest(unittest.TestCase):
    def test_expands_phrase_sources_and_preserves_exact_group_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base.yaml"
            base.write_text("features: []\n")
            features = root / "features.json"
            features.write_text(
                json.dumps(
                    {
                        "feature_sources": [
                            {
                                "class": "positive",
                                "text": "Wake One",
                                "feature_split": "training",
                                "features_dir": "/features/one",
                            },
                            {
                                "class": "positive",
                                "text": "Wake Two",
                                "feature_split": "training",
                                "features_dir": "/features/two",
                            },
                        ]
                    }
                )
            )
            plan = root / "plan.yaml"
            plan.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "sampling_groups": {"designed": 0.75, "device": 0.25},
                        "config_overrides": {"batch_size": 64},
                        "sources": [
                            {
                                "feature_build_manifest": str(features),
                                "class": "positive",
                                "source_prefix": "voice",
                                "truth": True,
                                "group": "designed",
                            },
                            {
                                "features_dir": "/device",
                                "source_name": "device",
                                "truth": True,
                                "group": "device",
                            },
                        ],
                    }
                )
            )

            config = stratified_config(base, plan)

            self.assertEqual(config["sampling_groups"]["designed"], 0.75)
            self.assertEqual(config["batch_size"], 64)
            self.assertEqual(len(config["features"]), 3)
            self.assertEqual(
                [item["sampling_source"] for item in config["features"][:2]],
                ["voice:Wake One", "voice:Wake Two"],
            )
            balance = config["sampling_plan"]["planned_balance"]
            self.assertEqual(balance["positive_sampling_share"], 1.0)
            self.assertEqual(balance["negative_sampling_share"], 0.0)

    def test_reports_and_guards_weighted_pressure_not_only_sample_share(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base.yaml"
            base.write_text(
                yaml.safe_dump(
                    {
                        "features": [],
                        "positive_class_weight": [1, 1],
                        "negative_class_weight": [2, 3],
                    }
                )
            )
            plan = root / "plan.yaml"
            plan.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "sampling_groups": {"positive": 0.6, "negative": 0.4},
                        "balance_guard": {
                            "maximum_negative_sampling_share": 0.4,
                            "maximum_negative_weighted_pressure_share": 0.7,
                        },
                        "sources": [
                            {
                                "features_dir": "/positive",
                                "source_name": "positive",
                                "truth": True,
                                "group": "positive",
                                "penalty_weight": 1,
                            },
                            {
                                "features_dir": "/negative",
                                "source_name": "negative",
                                "truth": False,
                                "group": "negative",
                                "penalty_weight": 2,
                            },
                        ],
                    }
                )
            )

            with self.assertRaisesRegex(
                ValueError, "negative weighted pressure share exceeds balance guard"
            ):
                stratified_config(base, plan)

            loaded = yaml.safe_load(plan.read_text())
            loaded["balance_guard"]["maximum_negative_weighted_pressure_share"] = 0.9
            plan.write_text(yaml.safe_dump(loaded))
            config = stratified_config(base, plan)
            balance = config["sampling_plan"]["planned_balance"]
            self.assertAlmostEqual(balance["negative_sampling_share"], 0.4)
            self.assertAlmostEqual(
                balance["weighted_pressure_stages"][0]["negative_share"], 8 / 11
            )
            self.assertAlmostEqual(
                balance["weighted_pressure_stages"][1]["negative_share"], 0.8
            )

    def test_rejects_groups_that_mix_truth_classes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base.yaml"
            base.write_text("features: []\n")
            plan = root / "plan.yaml"
            plan.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "sampling_groups": {"mixed": 1},
                        "sources": [
                            {
                                "features_dir": "/positive",
                                "source_name": "positive",
                                "truth": True,
                                "group": "mixed",
                            },
                            {
                                "features_dir": "/negative",
                                "source_name": "negative",
                                "truth": False,
                                "group": "mixed",
                            },
                        ],
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "must contain one truth class"):
                stratified_config(base, plan)

    def test_phrase_filters_support_relabeling_a_mislabeled_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "features.json"
            manifest.write_text(
                json.dumps(
                    {
                        "feature_sources": [
                            {
                                "class": "positive",
                                "text": "Wake One",
                                "feature_split": "training",
                                "features_dir": "/features/one",
                            },
                            {
                                "class": "positive",
                                "text": "High Five Kizz",
                                "feature_split": "training",
                                "features_dir": "/features/collision",
                            },
                        ]
                    }
                )
            )

            designed = expand_source(
                {
                    "feature_build_manifest": str(manifest),
                    "class": "positive",
                    "exclude_phrases": ["High Five Kizz"],
                    "source_prefix": "designed",
                    "truth": True,
                    "group": "positive",
                }
            )
            collision = expand_source(
                {
                    "feature_build_manifest": str(manifest),
                    "class": "positive",
                    "include_phrases": ["High Five Kizz"],
                    "source_prefix": "collision",
                    "truth": False,
                    "group": "negative",
                }
            )

            self.assertEqual([item["sampling_source"] for item in designed], ["designed:Wake One"])
            self.assertFalse(collision[0]["truth"])
            self.assertEqual(collision[0]["sampling_source"], "collision:High Five Kizz")

    def test_phrase_filters_must_be_explicit_and_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "features.json"
            manifest.write_text(json.dumps({"feature_sources": []}))
            base = {
                "feature_build_manifest": str(manifest),
                "source_prefix": "source",
                "truth": True,
                "group": "positive",
            }
            with self.assertRaisesRegex(ValueError, "cannot combine"):
                expand_source(
                    {
                        **base,
                        "include_phrases": ["Wake"],
                        "exclude_phrases": ["Collision"],
                    }
                )


if __name__ == "__main__":
    unittest.main()
