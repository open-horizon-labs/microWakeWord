import hashlib
import json
import tempfile
import unittest
import subprocess
import sys
from pathlib import Path

from tools.reuse_labeled_voice_samples import reuse_labeled_samples


class ReuseLabeledVoiceSamplesTest(unittest.TestCase):
    def test_script_entrypoint_resolves_sibling_imports(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(
                    Path(__file__).parents[1]
                    / "tools"
                    / "reuse_labeled_voice_samples.py"
                ),
                "--help",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_reuses_audio_and_can_move_a_former_positive_to_negative(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recipe = root / "recipe.yaml"
            recipe.write_text(
                "random_seed: 1\n"
                "positive_phrases:\n  - text: Hi-Fi Kizz\n    samples: 1\n"
                "hard_negative_phrases:\n  - text: Hi-Fi Kids\n    samples: 1\n"
            )
            target = root / "target"
            source = root / "source"
            target.mkdir()
            source.mkdir()
            target_manifest = {
                "schema_version": 2,
                "recipe_sha256": hashlib.sha256(recipe.read_bytes()).hexdigest(),
                "plan": [{"class": "positive", "text": None}],
            }
            (target / "generation-manifest.json").write_text(
                json.dumps(target_manifest)
            )

            plan = []
            for text, source_class in (
                ("Hi-Fi Kizz", "positive"),
                ("Hi-Fi Kids", "positive"),
            ):
                output = source / text.replace(" ", "_")
                output.mkdir()
                (output / "0000.wav").write_bytes(text.encode())
                (output / "synthesis-metadata.jsonl").write_text(
                    json.dumps({"file": "0000.wav", "text": text}) + "\n"
                )
                plan.append(
                    {
                        "class": source_class,
                        "text": text,
                        "split": "train",
                        "samples": 1,
                        "speaker_id": f"voice-{text}",
                        "speaker_name": f"speaker-{text}",
                        "provider": "elevenlabs",
                        "age_group": "adult",
                        "output": str(output),
                    }
                )
            (source / "generation-manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "plan": plan,
                        "labeled_voice_catalog": "voices.yaml",
                        "labeled_voice_catalog_sha256": "catalog-hash",
                    }
                )
            )

            result = reuse_labeled_samples(recipe, target, source)

            labeled = [item for item in result["plan"] if item.get("speaker_id")]
            by_text = {item["text"]: item for item in labeled}
            self.assertEqual(by_text["Hi-Fi Kizz"]["class"], "positive")
            self.assertEqual(by_text["Hi-Fi Kids"]["class"], "hard_negative")
            self.assertTrue(Path(by_text["Hi-Fi Kids"]["output"]).is_dir())
            self.assertEqual(result["labeled_voice_catalog_sha256"], "catalog-hash")

    def test_fails_when_a_recipe_text_has_no_labeled_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recipe = root / "recipe.yaml"
            recipe.write_text(
                "positive_phrases:\n  - text: Missing\n    samples: 1\n"
                "hard_negative_phrases: []\n"
            )
            for name, payload in (
                (
                    "target",
                    {
                        "schema_version": 2,
                        "recipe_sha256": hashlib.sha256(
                            recipe.read_bytes()
                        ).hexdigest(),
                        "plan": [],
                    },
                ),
                ("source", {"schema_version": 2, "plan": []}),
            ):
                directory = root / name
                directory.mkdir()
                (directory / "generation-manifest.json").write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "source lacks"):
                reuse_labeled_samples(recipe, root / "target", root / "source")


if __name__ == "__main__":
    unittest.main()
