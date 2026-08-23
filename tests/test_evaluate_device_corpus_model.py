import unittest

from tools.evaluate_device_corpus_model import capture_dimensions, qualification_scope


class DeviceCorpusEvaluationTest(unittest.TestCase):
    def test_reports_speaker_and_session_by_truth(self):
        item = {
            "device_profile": "m5stack_stackchan_v1",
            "speaker_id": "speaker-b",
            "session_id": "session-2",
            "phrase": "Hi-Fi Kizz",
            "pronunciation": "hi-fi-kizz",
            "detected": False,
        }

        dimensions = dict(capture_dimensions(item, "positive"))

        self.assertEqual(dimensions["speaker_id"], "speaker-b")
        self.assertEqual(dimensions["speaker_id_by_truth"], "speaker-b:positive")
        self.assertEqual(dimensions["session_id"], "session-2")
        self.assertEqual(dimensions["session_id_by_truth"], "session-2:positive")
        self.assertEqual(dimensions["source_detector_outcome"], "provisional_missed")

    def test_all_split_cannot_be_reported_as_qualification(self):
        manifest = {
            "speakers": {
                "speaker-a": {"kind": "human", "age_group": "adult"},
            },
            "captures": [
                {
                    "speaker_id": "speaker-a",
                    "session_id": "session-a",
                    "split": "train",
                    "truth": "positive",
                }
            ],
        }

        scope = qualification_scope(manifest, "all")

        self.assertTrue(scope["includes_training_data"])
        self.assertFalse(scope["qualification_eligible"])
        self.assertIn("qualification requires the test split", scope["issues"])

    def test_complete_test_scope_is_qualification_eligible(self):
        manifest = {
            "speakers": {
                "speaker-b": {"kind": "human", "age_group": "adult"},
                "room": {"kind": "ambient", "age_group": "not_applicable"},
            },
            "captures": [
                {
                    "speaker_id": "speaker-b",
                    "session_id": "session-b",
                    "split": "test",
                    "truth": truth,
                }
                for truth in ("positive", "hard_negative")
            ]
            + [
                {
                    "speaker_id": "room",
                    "session_id": "session-room",
                    "split": "test",
                    "truth": "ambient_negative",
                }
            ],
        }

        self.assertTrue(qualification_scope(manifest, "test")["qualification_eligible"])

        scope = qualification_scope(
            manifest, "test", required_age_groups=("adult", "child")
        )
        self.assertFalse(scope["qualification_eligible"])
        self.assertIn(
            "test split has no registered child human speaker", scope["issues"]
        )


if __name__ == "__main__":
    unittest.main()
