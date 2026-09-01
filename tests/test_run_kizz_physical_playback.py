import json
import tempfile
import unittest
from pathlib import Path

from tools.run_kizz_physical_playback import _schedule, classify, parse_load


class PhysicalPlaybackTests(unittest.TestCase):
    def test_classifies_cascade_events(self):
        self.assertEqual(
            classify("I Kizz compact verifier: score=1 pass=true"), "compact"
        )
        self.assertEqual(
            classify("I Kizz ordered verifier: score=-1 pass=false"), "ordered"
        )
        self.assertEqual(
            classify("I m5_platform: Kizz wake detected on-device: custom model"),
            "accepted",
        )
        self.assertIsNone(classify("I Kizz Control detected locally"))
        self.assertEqual(classify("abort() was called at PC 1"), "crash")

    def test_parses_perf_load_integer_fields(self):
        value = parse_load(
            "I KIZZ_PERF load detector_candidates=12 verifier_runs=11 "
            "compact=11/2/9/0 audio_dropped=512"
        )
        self.assertEqual(value["detector_candidates"], 12)
        self.assertEqual(value["verifier_runs"], 11)
        self.assertEqual(value["audio_dropped"], 512)
        self.assertNotIn("compact", value)

    def test_schedule_requires_positive_serial_wait_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "wake.wav"
            audio.write_bytes(b"not decoded by schedule validation")
            schedule = root / "schedule.json"
            schedule.write_text(
                json.dumps(
                    {
                        "segments": [
                            {
                                "label": "wake",
                                "duration_seconds": 1,
                                "sources": [str(audio)],
                                "wait_for_serial": "Kizz command armed",
                            }
                        ]
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "wait_timeout_seconds"):
                _schedule(schedule)

            value = json.loads(schedule.read_text())
            value["segments"][0]["wait_timeout_seconds"] = 4
            schedule.write_text(json.dumps(value))
            _, rows = _schedule(schedule)
            self.assertEqual(rows[0]["wait_for_serial"], "Kizz command armed")
            self.assertEqual(rows[0]["wait_timeout_seconds"], 4)


if __name__ == "__main__":
    unittest.main()
