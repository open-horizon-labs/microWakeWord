import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.screen_kizz_wake_phrase_candidates import (
    derive_ipa,
    discover_positive_audio,
    minimum_subsequence_edit_distance,
    parse_candidate,
    parse_candidates,
    positive_render_dimensions,
    positive_threshold,
    token_edit_distance,
)


class ScreenKizzWakePhraseCandidateTests(unittest.TestCase):
    def test_parse_candidate_preserves_equals_in_text(self):
        self.assertEqual(
            parse_candidate("alpha=Hi Phi Kizz=please"), ("alpha", "Hi Phi Kizz=please")
        )

    def test_parse_candidate_rejects_bad_values_and_duplicates(self):
        with self.assertRaises(ValueError):
            parse_candidate("=Hi Phi")
        with self.assertRaises(ValueError):
            parse_candidate("bad id=Hi Phi")
        with self.assertRaises(ValueError):
            parse_candidates(["a=one", "a=two"])

    def test_token_edit_distance_is_token_based(self):
        self.assertEqual(token_edit_distance(["h", "aɪ", "k"], ["h", "k"]), 1)
        self.assertEqual(token_edit_distance([], ["h", "aɪ"]), 2)

    def test_minimum_subsequence_edit_distance_ignores_clip_context(self):
        target = ["h", "aɪ", "f", "aɪ", "k"]
        observed = ["noise", "h", "aɪ", "f", "aɪ", "k", "speech"]
        self.assertEqual(minimum_subsequence_edit_distance(target, observed), 0)
        self.assertEqual(minimum_subsequence_edit_distance(target, []), len(target))

    def test_ipa_comes_from_tokenizer_phonemizer(self):
        class FakeTokenizer:
            def phonemize(self, text):
                self.text = text
                return " h  aɪ  k "

        tokenizer = FakeTokenizer()
        self.assertEqual(derive_ipa(tokenizer, "Hi Kizz"), ("h", "aɪ", "k"))
        self.assertEqual(tokenizer.text, "Hi Kizz")

    def test_threshold_uses_only_positive_scores(self):
        result = positive_threshold([0.2, 0.5, 0.9, float("-inf")], 0.75)
        self.assertEqual(result["required_count"], 3)
        self.assertEqual(result["threshold"], 0.2)
        self.assertAlmostEqual(result["achieved_recall"], 0.75)
        self.assertNotIn("false_accepts", result)

    def test_positive_discovery_is_candidate_scoped_and_extension_case_insensitive(
        self,
    ):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "one--a.wav",
                "one--b.AIFF",
                "two--a.wav",
                "one.wav",
                "one--notes.txt",
            ):
                (root / name).touch()
            found = discover_positive_audio(root, ["one", "two"])
            self.assertEqual(
                [path.name for path in found["one"]], ["one--a.wav", "one--b.AIFF"]
            )
            self.assertEqual([path.name for path in found["two"]], ["two--a.wav"])

    def test_positive_render_dimensions_require_voice_and_rate(self):
        paths = [
            Path("one--Alice--175.aiff"),
            Path("one--Bob--220.wav"),
            Path("one--Alice--220.wav"),
            Path("one--Bob--175.wav"),
        ]
        self.assertEqual(
            positive_render_dimensions(paths, "one"),
            {"voices": ["Alice", "Bob"], "rates": ["175", "220"]},
        )
        with self.assertRaisesRegex(ValueError, "VOICE--RATE"):
            positive_render_dimensions([Path("one--Alice.wav")], "one")
        with self.assertRaisesRegex(ValueError, "matrix is incomplete"):
            positive_render_dimensions(paths[:-1], "one")
        with self.assertRaisesRegex(ValueError, "matrix repeats"):
            positive_render_dimensions([paths[0], Path("one--Alice--175.wav")], "one")


if __name__ == "__main__":
    unittest.main()
