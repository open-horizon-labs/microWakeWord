import json
import tempfile
import unittest
from pathlib import Path

from tools.qualify_kizz_phoneme_teacher import _wake_context_metadata


class QualifyKizzPhonemeTeacherTests(unittest.TestCase):
    def test_wake_context_is_bound_to_audio_hash_id_and_trigger(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = Path(directory) / "wake.json"
            metadata.write_text(
                json.dumps(
                    {
                        "observation_id": "wake-1",
                        "sha256": "audio-hash",
                        "pre_wake_ms": 3000,
                        "firmware_sha": "firmware",
                        "device_profile": "device",
                    }
                )
            )
            row = {
                "source_id": "false-wake:wake-1",
                "audio_sha256": "audio-hash",
                "metadata_path": str(metadata),
            }
            result = _wake_context_metadata(row, 3.25)
            self.assertEqual(result["wake_trigger_seconds"], 3.0)
            self.assertEqual(result["firmware_sha"], "firmware")
            with self.assertRaisesRegex(ValueError, "audio hash differs"):
                _wake_context_metadata({**row, "audio_sha256": "other"}, 3.25)
            with self.assertRaisesRegex(ValueError, "outside the recording"):
                _wake_context_metadata(row, 2.99)


if __name__ == "__main__":
    unittest.main()
