import json
import tempfile
import unittest
from pathlib import Path

import yaml

from tools.write_stratified_training_config import stratified_config


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


if __name__ == "__main__":
    unittest.main()
