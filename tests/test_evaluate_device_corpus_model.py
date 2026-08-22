import unittest

from tools.evaluate_device_corpus_model import capture_dimensions


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


if __name__ == "__main__":
    unittest.main()
