import importlib.util
import sys
import unittest
from pathlib import Path

import yaml
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "generate_recipe_samples.py"
SPEC = importlib.util.spec_from_file_location("generate_recipe_samples", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

CONFIG_SCRIPT = ROOT / "tools" / "write_recipe_training_config.py"
CONFIG_SPEC = importlib.util.spec_from_file_location("write_recipe_training_config", CONFIG_SCRIPT)
CONFIG_MODULE = importlib.util.module_from_spec(CONFIG_SPEC)
sys.modules[CONFIG_SPEC.name] = CONFIG_MODULE
CONFIG_SPEC.loader.exec_module(CONFIG_MODULE)


class KizzRecipeTest(unittest.TestCase):
    def setUp(self):
        self.recipe = yaml.safe_load((ROOT / "recipes/kizz/corpus.yaml").read_text())

    def test_one_class_contains_short_and_brand_forms(self):
        phrases = {entry["text"] for entry in self.recipe["positive_phrases"]}
        self.assertIn("Kizz", phrases)
        self.assertIn("Hi-Fi Kizz", phrases)
        self.assertIn("Hi Phi Kizz", phrases)
        self.assertIn("Hee Fee Kizz", phrases)
        self.assertIn("Hippy Kizz", phrases)
        counts = {
            entry["text"]: entry["samples"]
            for entry in self.recipe["positive_phrases"]
        }
        self.assertGreaterEqual(counts["Hee Fee Kizz"], 2000)
        self.assertGreaterEqual(counts["Hippy Kizz"], 2000)

    def test_near_sounding_words_are_hard_negatives(self):
        phrases = {entry["text"] for entry in self.recipe["hard_negative_phrases"]}
        self.assertTrue({"kids", "kiss", "quiz", "Hi-Fi"}.issubset(phrases))

    def test_generator_command_preserves_variation_grid(self):
        phrase = self.recipe["positive_phrases"][0]
        command = MODULE.generator_command(
            phrase,
            self.recipe["generation"],
            Path("model.pt"),
            Path("out"),
            8,
        )
        self.assertIn("--length-scales", command)
        self.assertIn("--noise-scales", command)
        self.assertIn("--noise-scale-ws", command)
        self.assertIn("--slerp-weights", command)

    def test_training_selects_for_ambient_false_accepts_first(self):
        config = CONFIG_MODULE.training_config(Path("work"), Path("trained"))
        self.assertEqual(config["minimization_metric"], "ambient_false_positives_per_hour")
        hard_negative = next(
            item for item in config["features"] if item["features_dir"].endswith("hard_negative")
        )
        self.assertFalse(hard_negative["truth"])
        self.assertGreater(hard_negative["penalty_weight"], 1.0)

    def test_microfrontend_accepts_current_pybind_api(self):
        from microwakeword.audio.audio_utils import generate_features_for_clip

        features = generate_features_for_clip(np.zeros(1600, dtype=np.int16))
        self.assertEqual(features.shape[1], 40)

    def test_quantization_ignores_only_partial_trailing_stride(self):
        from microwakeword.utils import streaming_calibration_slices

        spectrogram = np.arange(8 * 40, dtype=np.float32).reshape(8, 40)
        chunks = list(streaming_calibration_slices(spectrogram, 3))
        self.assertEqual(len(chunks), 2)
        np.testing.assert_array_equal(chunks[0], spectrogram[0:3])
        np.testing.assert_array_equal(chunks[1], spectrogram[3:6])


if __name__ == "__main__":
    unittest.main()
