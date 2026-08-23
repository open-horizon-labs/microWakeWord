import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from tools.add_labeled_voice_samples import add_samples, load_catalog, samples_for_class
from tools.build_recipe_features import validate_generated_corpus


def write_catalog(path: Path, voices: list[dict]) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "provider": "elevenlabs",
                "model_id": "test-model",
                "voices": voices,
            },
            sort_keys=False,
        )
    )


class LabeledVoiceSamplesTest(unittest.TestCase):
    def test_class_specific_sample_counts_override_legacy_default(self):
        voice = {
            "samples_per_phrase": 2,
            "positive_samples_per_phrase": 9,
            "hard_negative_samples_per_phrase": 3,
        }
        self.assertEqual(samples_for_class(voice, "positive"), 9)
        self.assertEqual(samples_for_class(voice, "hard_negative"), 3)

    def test_rejects_voice_identity_crossing_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            catalog = Path(temporary) / "voices.yaml"
            write_catalog(
                catalog,
                [
                    {
                        "name": "one",
                        "voice_id": "same",
                        "split": "train",
                        "age_group": "adult",
                    },
                    {
                        "name": "two",
                        "voice_id": "same",
                        "split": "test",
                        "age_group": "adult",
                    },
                ],
            )

            with self.assertRaisesRegex(ValueError, "voice identity crosses"):
                load_catalog(catalog)

    def test_adds_labeled_pcm_with_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recipe = root / "recipe.yaml"
            recipe.write_text(
                yaml.safe_dump(
                    {
                        "random_seed": 23,
                        "positive_phrases": [{"text": "Wake", "samples": 1}],
                        "hard_negative_phrases": [{"text": "Wait", "samples": 1}],
                    }
                )
            )
            generated = root / "generated"
            generated.mkdir()
            (generated / "generation-manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "recipe_sha256": hashlib.sha256(
                            recipe.read_bytes()
                        ).hexdigest(),
                        "plan": [],
                    }
                )
            )
            catalog = root / "voices.yaml"
            write_catalog(
                catalog,
                [
                    {
                        "name": "adult train",
                        "voice_id": "adult-1",
                        "split": "train",
                        "age_group": "adult",
                        "samples_per_phrase": 2,
                    }
                ],
            )
            requests = []

            def fake_synthesize(*args):
                requests.append(args)
                return b"\x00\x00" * 160

            manifest = add_samples(
                recipe, generated, catalog, "test-key", fake_synthesize
            )

            self.assertEqual(len(requests), 4)
            self.assertEqual(len(manifest["plan"]), 2)
            for item in manifest["plan"]:
                self.assertEqual(item["speaker_id"], "adult-1")
                self.assertEqual(item["age_group"], "adult")
                metadata = [
                    json.loads(line)
                    for line in (Path(item["output"]) / "synthesis-metadata.jsonl")
                    .read_text()
                    .splitlines()
                ]
                self.assertEqual(len(metadata), 2)
                self.assertEqual(len(list(Path(item["output"]).glob("*.wav"))), 2)

    def test_feature_validation_requires_declared_age_cohorts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recipe = root / "recipe.yaml"
            recipe.write_text(
                yaml.safe_dump(
                    {
                        "generation": {
                            "labeled_voice_requirements": {
                                "age_groups": ["adult", "child"],
                                "minimum_voices_per_split": {
                                    "train": 1,
                                    "validation": 0,
                                    "test": 0,
                                },
                            }
                        }
                    }
                )
            )
            generated = root / "generated"
            (generated / "positive").mkdir(parents=True)
            (generated / "hard_negative").mkdir()
            (generated / "generation-manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "recipe_sha256": hashlib.sha256(
                            recipe.read_bytes()
                        ).hexdigest(),
                        "plan": [],
                    }
                )
            )

            with self.assertRaisesRegex(ValueError, "train/adult.*train/child"):
                validate_generated_corpus(recipe, generated)


if __name__ == "__main__":
    unittest.main()
