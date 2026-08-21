import tempfile
import unittest
from pathlib import Path

from tools.build_recipe_features import (
    selected_phrase_directories,
    staged_clip_source,
)


class SelectedPhraseFeaturesTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
