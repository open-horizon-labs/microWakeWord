import unittest

from tools.apply_phrase_spans import apply_phrase_spans


class ApplyPhraseSpansTest(unittest.TestCase):
    def test_updates_named_captures_without_dropping_misses(self):
        manifest = {
            "captures": [
                {"capture_id": "miss", "detected": False},
                {"capture_id": "other", "detected": True},
            ]
        }

        updated = apply_phrase_spans(
            manifest,
            {"miss": {"start_ms": 120, "end_ms": 920}},
        )

        self.assertEqual(updated, 1)
        self.assertEqual(
            manifest["captures"][0]["phrase_span"],
            {"start_ms": 120, "end_ms": 920},
        )
        self.assertFalse(manifest["captures"][0]["detected"])
        self.assertNotIn("phrase_span", manifest["captures"][1])

    def test_rejects_unknown_capture_ids(self):
        with self.assertRaisesRegex(ValueError, "unknown capture"):
            apply_phrase_spans(
                {"captures": []},
                {"missing": {"start_ms": 0, "end_ms": 100}},
            )


if __name__ == "__main__":
    unittest.main()
