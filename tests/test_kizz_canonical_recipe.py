import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "recipes" / "kizz" / "corpus.yaml"
CANONICAL = ROOT / "recipes" / "kizz" / "corpus.v32-canonical.yaml"


class KizzCanonicalRecipeTest(unittest.TestCase):
    def setUp(self):
        self.baseline = yaml.safe_load(BASELINE.read_text())
        self.recipe = yaml.safe_load(CANONICAL.read_text())

    def test_is_a_separate_canonical_only_positive_contract(self):
        self.assertNotEqual(self.recipe["name"], self.baseline["name"])
        self.assertEqual(
            self.recipe["positive_phrases"],
            [{"text": "Hi-Fi Kizz", "samples": 24000}],
        )

    def test_former_kids_aliases_are_negative_not_positive(self):
        positives = {
            item["text"].casefold() for item in self.recipe["positive_phrases"]
        }
        negatives = {
            item["text"].casefold() for item in self.recipe["hard_negative_phrases"]
        }
        former_aliases = {
            "hippy kids",
            "hi-fi kids",
            "high-fi kids",
            "hi fi kids",
            "high fi kids",
            "hee fee kids",
            "high fee kids",
            "hiffy kids",
        }
        self.assertTrue(former_aliases.isdisjoint(positives))
        self.assertTrue(former_aliases.issubset(negatives))

    def test_high_five_collision_remains_negative(self):
        negatives = {
            item["text"].casefold() for item in self.recipe["hard_negative_phrases"]
        }
        self.assertIn("high five kizz", negatives)
        self.assertIn("high five kids", negatives)

    def test_two_hundred_base_speakers_are_held_out(self):
        cohorts = self.recipe["generation"]["speaker_cohorts"]
        validation = (
            cohorts["validation"]["speaker_end"]
            - cohorts["validation"]["speaker_start"]
        )
        testing = cohorts["test"]["speaker_end"] - cohorts["test"]["speaker_start"]
        self.assertEqual(validation + testing, 200)

    def test_connected_speech_source_has_fixed_bounded_allocation(self):
        sources = self.recipe["connected_sentence_sources"]
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["samples_per_text"], 96)
        self.assertEqual(
            {sources[0][split] for split in ("train", "validation", "test")},
            {
                "connected-negative-sentences-train.txt",
                "connected-negative-sentences-validation.txt",
                "connected-negative-sentences-test.txt",
            },
        )


if __name__ == "__main__":
    unittest.main()
