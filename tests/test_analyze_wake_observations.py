import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from tools.analyze_wake_observations import analyze


class AnalyzeWakeObservationsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.corpus = Path(self.temporary.name)
        (self.corpus / "observations" / "wakes").mkdir(parents=True)
        (self.corpus / "observations" / "false-wakes").mkdir(parents=True)

    def tearDown(self):
        self.temporary.cleanup()

    def write_observation(self, directory, name, received_at, outcome):
        path = self.corpus / "observations" / directory / f"{name}.json"
        audio_path = self.corpus / "observations" / directory / f"{name}.wav"
        audio_path.write_bytes(b"test-pcm")
        path.write_text(
            json.dumps(
                {
                    "observation_id": name,
                    "received_at": received_at,
                    "outcome": outcome,
                    "wake_probability": 0.71,
                    "path": f"observations/{directory}/{name}.wav",
                    "sha256": hashlib.sha256(b"test-pcm").hexdigest(),
                }
            )
        )

    def test_correlates_command_transcript_and_keeps_weak_label(self):
        self.write_observation("wakes", "wake-command", 100.5, "command_speech")
        report = analyze(
            self.corpus,
            {
                "recent": [
                    {
                        "timestamp": 101.0,
                        "turn_id": 7,
                        "provider": "deepgram",
                        "outcome": "completed",
                        "detail": "Next",
                        "latency_ms": 320,
                    }
                ]
            },
            window_seconds=10,
        )

        item = report["observations"][0]
        self.assertEqual(item["weak_label"], "stt_command_candidate")
        self.assertEqual(item["matched_turn_id"], 7)
        self.assertEqual(item["audio_sha256"], hashlib.sha256(b"test-pcm").hexdigest())
        self.assertTrue(item["metadata_sha256"])
        self.assertTrue(report["human_review_required"])
        self.assertFalse(report["training_eligible"])

    def test_no_command_observation_does_not_need_stt_to_be_quarantined(self):
        self.write_observation("false-wakes", "wake-no-command", 200, "no_command")
        report = analyze(self.corpus, {"recent": []}, window_seconds=10)

        item = report["observations"][0]
        self.assertEqual(item["weak_label"], "false_wake_no_command")
        self.assertIsNone(item["matched_turn_id"])

    def test_distant_transcript_is_not_cross_turn_matched(self):
        self.write_observation("wakes", "wake-unconfirmed", 300, "command_speech")
        report = analyze(
            self.corpus,
            {
                "recent": [
                    {
                        "timestamp": 311,
                        "turn_id": 9,
                        "outcome": "completed",
                        "detail": "Next",
                    }
                ]
            },
            window_seconds=10,
        )

        item = report["observations"][0]
        self.assertEqual(item["weak_label"], "speech_unconfirmed")
        self.assertIsNone(item["matched_turn_id"])

    def test_rejects_audio_changed_after_capture(self):
        self.write_observation("false-wakes", "wake-tampered", 400, "no_command")
        audio_path = self.corpus / "observations" / "false-wakes" / "wake-tampered.wav"
        audio_path.write_bytes(b"different")

        with self.assertRaisesRegex(ValueError, "audio hash mismatch"):
            analyze(self.corpus, {"recent": []}, window_seconds=10)


if __name__ == "__main__":
    unittest.main()
