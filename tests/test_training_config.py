import unittest
from pathlib import Path

from tools.write_recipe_training_config import training_config


class DeviceTrainingConfigTest(unittest.TestCase):
    def test_device_only_pass_is_bounded_and_uses_only_device_sources(self):
        config = training_config(
            Path("workspace"),
            Path("trained"),
            device_features_dir=Path("device-features"),
            device_only=True,
        )

        self.assertEqual(config["training_steps"], [3000, 2000])
        self.assertEqual(config["eval_step_interval"], 500)
        self.assertEqual(config["batch_size"], 32)
        self.assertEqual(
            [source["features_dir"] for source in config["features"]],
            [
                "device-features/positive",
                "device-features/hard_negative",
                "device-features/ambient_negative",
            ],
        )

    def test_device_only_pass_requires_device_features(self):
        with self.assertRaisesRegex(ValueError, "device training"):
            training_config(Path("workspace"), Path("trained"), device_only=True)

    def test_adaptation_trains_on_general_sources_but_evaluates_on_device(self):
        config = training_config(
            Path("workspace"),
            Path("trained"),
            device_features_dir=Path("device-features"),
            device_adaptation=True,
        )

        self.assertEqual(config["training_steps"], [6000, 4000])
        self.assertEqual(config["batch_size"], 64)
        self.assertEqual(config["minimization_metric"], "validation_false_positives")
        self.assertEqual(config["target_minimization"], 0)
        self.assertTrue(
            all(
                source.get("evaluation_enabled") is False
                for source in config["features"][:-3]
            )
        )
        self.assertTrue(
            all(
                source.get("evaluation_enabled", True)
                for source in config["features"][-3:]
            )
        )
        self.assertEqual(
            [source["truncation_strategy"] for source in config["features"][-3:]],
            ["truncate_end", "truncate_end", "split"],
        )

    def test_train_only_device_subset_does_not_claim_device_validation(self):
        config = training_config(
            Path("workspace"),
            Path("trained"),
            device_features_dir=Path("device-features"),
            device_train_only=True,
        )

        device_sources = config["features"][-2:]
        self.assertEqual(
            [source["features_dir"] for source in device_sources],
            ["device-features/positive", "device-features/hard_negative"],
        )
        self.assertTrue(
            all(source["evaluation_enabled"] is False for source in device_sources)
        )
        self.assertTrue(
            all(
                source.get("evaluation_enabled", True)
                for source in config["features"][:-2]
            )
        )

    def test_initialized_device_fine_tune_is_bounded_and_low_rate(self):
        config = training_config(
            Path("workspace"),
            Path("trained"),
            features_dir=Path("collision-features"),
            device_features_dir=Path("device-features"),
            device_train_only=True,
            initial_weights=Path("base/best_weights.weights.h5"),
            device_positive_sampling_weight=0.25,
            device_hard_negative_sampling_weight=0.25,
        )

        self.assertEqual(config["training_steps"], [1000, 500])
        self.assertEqual(config["learning_rates"], [0.00005, 0.00001])
        self.assertEqual(config["negative_class_weight"], [24, 32])
        self.assertEqual(config["initial_weights"], "base/best_weights.weights.h5")
        self.assertTrue(config["freeze_batch_normalization"])
        self.assertEqual(
            config["features"][1]["features_dir"],
            "collision-features/hard_negative",
        )
        self.assertEqual(config["features"][-2]["sampling_weight"], 0.25)
        self.assertEqual(config["features"][-1]["sampling_weight"], 0.25)

    def test_positive_only_device_fine_tune_keeps_general_collision_gate(self):
        config = training_config(
            Path("workspace"),
            Path("trained"),
            device_features_dir=Path("device-features"),
            device_train_only=True,
            device_positive_only=True,
            initial_weights=Path("base/best_weights.weights.h5"),
        )

        device_sources = [
            source
            for source in config["features"]
            if source["features_dir"].startswith("device-features/")
        ]
        self.assertEqual(
            [source["features_dir"] for source in device_sources],
            ["device-features/positive"],
        )
        self.assertFalse(device_sources[0]["evaluation_enabled"])
        self.assertIn(
            "workspace/features/hard_negative",
            [source["features_dir"] for source in config["features"]],
        )


if __name__ == "__main__":
    unittest.main()
