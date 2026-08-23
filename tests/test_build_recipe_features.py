import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from scipy.io import wavfile

from tools.build_recipe_features import (
    _plan_speakers,
    augmentation_for_split,
    generate_class_features,
    selected_phrase_directories,
    staged_clip_source,
    validate_generated_corpus,
)


class SelectedPhraseFeaturesTest(unittest.TestCase):
    def test_augmentation_is_training_only(self):
        augmenter = object()
        self.assertIs(augmentation_for_split(augmenter, "training"), augmenter)
        self.assertIsNone(augmentation_for_split(augmenter, "validation"))
        self.assertIsNone(augmentation_for_split(augmenter, "testing"))

    def test_labeled_tts_speaker_has_provider_scoped_identity(self):
        self.assertEqual(
            _plan_speakers({"provider": "elevenlabs", "speaker_id": "voice-1"}),
            {"elevenlabs:voice-1"},
        )

    def test_selects_exact_recipe_phrases_in_requested_order(self):
        manifest = {
            "plan": [
                {"class": "positive", "text": "Hi-Fi Kizz", "output": "/a"},
                {"class": "positive", "text": "Hippy Kizz", "output": "/b"},
                {"class": "hard_negative", "text": "Kizz", "output": "/c"},
            ]
        }
        self.assertEqual(
            selected_phrase_directories(
                manifest, "positive", ["Hippy Kizz", "Hi-Fi Kizz"]
            ),
            [Path("/b"), Path("/a")],
        )

    def test_rejects_unknown_phrase(self):
        with self.assertRaisesRegex(ValueError, "unknown positive phrase"):
            selected_phrase_directories({"plan": []}, "positive", ["Nope"])

    def test_selects_hard_negative_phrase(self):
        manifest = {
            "plan": [
                {
                    "class": "hard_negative",
                    "text": "high five kids",
                    "output": "/negative",
                }
            ]
        }
        self.assertEqual(
            selected_phrase_directories(manifest, "hard_negative", ["high five kids"]),
            [Path("/negative")],
        )

    def test_stages_selected_wavs_without_name_collisions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            (first / "000.wav").touch()
            (second / "000.wav").touch()
            with staged_clip_source([first, second]) as staged:
                names = sorted(path.name for path in staged.glob("*.wav"))
                self.assertEqual(names, ["first--000.wav", "second--000.wav"])

    def test_staging_excludes_rejected_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            phrase = root / "phrase"
            phrase.mkdir()
            accepted = phrase / "accepted.wav"
            rejected = phrase / "rejected.wav"
            accepted.touch()
            rejected.touch()

            with staged_clip_source([phrase], {rejected.resolve()}) as staged:
                names = [path.name for path in staged.glob("*.wav")]

            self.assertEqual(names, ["phrase--accepted.wav"])

    def test_can_rebuild_only_one_feature_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            spectrograms = MagicMock()
            spectrograms.spectrogram_generator.return_value = iter(())
            with (
                patch("tools.build_recipe_features.Clips"),
                patch(
                    "tools.build_recipe_features.SpectrogramGeneration",
                    return_value=spectrograms,
                ),
                patch(
                    "tools.build_recipe_features.RaggedMmap.from_generator"
                ) as build_mmap,
            ):
                generate_class_features(
                    [Path(temporary) / "source"],
                    Path(temporary) / "features",
                    MagicMock(),
                    "testing",
                )

            spectrograms.spectrogram_generator.assert_called_once_with(
                split=None, repeat=1
            )
            self.assertEqual(
                build_mmap.call_args.kwargs["out_dir"],
                str(Path(temporary) / "features/testing/wakeword_mmap"),
            )

    def test_rejects_synthetic_speaker_leakage_across_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recipe = root / "recipe.yaml"
            recipe.write_text("name: test\n")
            generated = root / "generated"
            plan = []
            for split in ("train", "test"):
                output = generated / "positive" / "wake" / split
                output.mkdir(parents=True)
                wavfile.write(output / "0.wav", 16000, np.zeros(1600, dtype=np.int16))
                (output / "synthesis-metadata.jsonl").write_text(
                    json.dumps({"file": "0.wav", "speaker_1": 0, "speaker_2": 0}) + "\n"
                )
                plan.append(
                    {
                        "class": "positive",
                        "text": "Wake",
                        "group": "wake",
                        "split": split,
                        "samples": 1,
                        "speaker_start": 0,
                        "speaker_end": 1,
                        "output": str(output),
                    }
                )
            (generated / "generation-manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "recipe_sha256": hashlib.sha256(
                            recipe.read_bytes()
                        ).hexdigest(),
                        "plan": plan,
                    }
                )
            )

            with self.assertRaisesRegex(ValueError, "synthetic speakers cross"):
                validate_generated_corpus(recipe, generated)


if __name__ == "__main__":
    unittest.main()
