import importlib.util
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml
import numpy as np
from scipy.io import wavfile


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

EVALUATOR_SCRIPT = ROOT / "tools" / "evaluate_recipe_model.py"
EVALUATOR_SPEC = importlib.util.spec_from_file_location(
    "evaluate_recipe_model", EVALUATOR_SCRIPT
)
EVALUATOR_MODULE = importlib.util.module_from_spec(EVALUATOR_SPEC)
sys.modules[EVALUATOR_SPEC.name] = EVALUATOR_MODULE
EVALUATOR_SPEC.loader.exec_module(EVALUATOR_MODULE)

FEATURE_SCRIPT = ROOT / "tools" / "build_recipe_features.py"
FEATURE_SPEC = importlib.util.spec_from_file_location(
    "build_recipe_features", FEATURE_SCRIPT
)
FEATURE_MODULE = importlib.util.module_from_spec(FEATURE_SPEC)
sys.modules[FEATURE_SPEC.name] = FEATURE_MODULE
FEATURE_SPEC.loader.exec_module(FEATURE_MODULE)


class KizzRecipeTest(unittest.TestCase):
    def setUp(self):
        self.recipe = yaml.safe_load((ROOT / "recipes/kizz/corpus.yaml").read_text())

    def test_one_class_contains_brand_pronunciations(self):
        phrases = {entry["text"] for entry in self.recipe["positive_phrases"]}
        self.assertNotIn("Kizz", phrases)
        self.assertIn("Hi-Fi Kizz", phrases)
        self.assertIn("Hi Phi Kizz", phrases)
        self.assertIn("Hee Fee Kizz", phrases)
        self.assertIn("Hippy Kizz", phrases)
        self.assertIn("High Fee Kizz", phrases)
        self.assertIn("Hee Fye Kizz", phrases)
        self.assertIn("High Fye Kizz", phrases)
        self.assertIn("Hiffy Kizz", phrases)
        counts = {
            entry["text"]: entry["samples"]
            for entry in self.recipe["positive_phrases"]
        }
        self.assertGreaterEqual(counts["Hee Fee Kizz"], 2000)
        self.assertGreaterEqual(counts["Hippy Kizz"], 2000)
        self.assertTrue(all(count >= 3000 for count in counts.values()))

    def test_near_sounding_words_are_hard_negatives(self):
        phrases = {entry["text"] for entry in self.recipe["hard_negative_phrases"]}
        self.assertTrue({"Kizz", "kids", "kiss", "quiz", "Hi-Fi"}.issubset(phrases))
        self.assertTrue(
            {"Hi-Fi Kiss", "Hippy Kiss", "Wi-Fi Kizz", "Happy Kizz"}.issubset(
                phrases
            )
        )

    def test_pronunciation_probes_are_unseen_during_training(self):
        probes = yaml.safe_load(
            (ROOT / "recipes/kizz/probes.yaml").read_text()
        )
        training = {
            phrase["text"].casefold()
            for phrase in self.recipe["positive_phrases"]
        }
        probe_phrases = {
            phrase["text"].casefold()
            for phrase in probes["positive_phrases"]
        }
        self.assertIn("high phi kizz", probe_phrases)
        self.assertTrue(training.isdisjoint(probe_phrases))

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

    def test_feature_build_rejects_stale_corpus_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = root / "recipe.yaml"
            recipe.write_text("name: test\n")
            expected = root / "generated" / "positive" / "expected"
            expected.mkdir(parents=True)
            (root / "generated" / "hard_negative").mkdir()
            manifest = {
                "recipe_sha256": hashlib.sha256(recipe.read_bytes()).hexdigest(),
                "plan": [
                    {
                        "class": "positive",
                        "output": str(expected),
                        "samples": 0,
                    }
                ],
            }
            (root / "generated" / "generation-manifest.json").write_text(
                json.dumps(manifest)
            )
            (root / "generated" / "positive" / "stale").mkdir()
            with self.assertRaisesRegex(ValueError, "extra="):
                FEATURE_MODULE.validate_generated_corpus(
                    recipe, root / "generated"
                )

    def test_training_selects_for_ambient_false_accepts_first(self):
        config = CONFIG_MODULE.training_config(Path("work"), Path("trained"))
        self.assertEqual(config["minimization_metric"], "ambient_false_positives_per_hour")
        hard_negatives = [
            item
            for item in config["features"]
            if item["features_dir"].endswith("hard_negative")
        ]
        self.assertEqual(len(hard_negatives), 2)
        training_hard_negative, evaluation_hard_negative = hard_negatives
        self.assertFalse(training_hard_negative["truth"])
        self.assertGreaterEqual(training_hard_negative["penalty_weight"], 4.0)
        self.assertEqual(evaluation_hard_negative["sampling_weight"], 0.0)
        self.assertEqual(evaluation_hard_negative["truncation_strategy"], "split")
        self.assertLessEqual(config["eval_step_interval"], 1000)

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

    def test_evaluator_uses_the_feature_pipeline_holdout_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for group_name in ("hi_fi", "hee_fee"):
                group = root / group_name
                group.mkdir()
                for index in range(10):
                    wavfile.write(
                        group / f"{index}.wav",
                        16000,
                        np.zeros(1600, dtype=np.int16),
                    )
            grouped = EVALUATOR_MODULE.clips_by_group(root, "test", 231)
            held_out = [path for paths in grouped.values() for path in paths]
            self.assertEqual(len(held_out), 2)
            self.assertTrue(all(path.parent.name in {"hi_fi", "hee_fee"} for path in held_out))

    def test_streaming_model_state_can_be_reset_between_independent_clips(self):
        from microwakeword.inference import Model

        class Interpreter:
            reset_count = 0

            def reset_all_variables(self):
                self.reset_count += 1

        model = Model.__new__(Model)
        model.model = Interpreter()
        model.reset_states()
        self.assertEqual(model.model.reset_count, 1)


if __name__ == "__main__":
    unittest.main()
