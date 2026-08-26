import json
import tempfile
import unittest
import wave
from pathlib import Path

from tools.export_kizz_device_replays import export_rows


class ExportKizzDeviceReplaysTests(unittest.TestCase):
    def test_export_retains_voice_and_source_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "audio" / "capture.wav"
            audio.parent.mkdir()
            with wave.open(str(audio), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b"\0\0" * 160)
            import hashlib

            digest = hashlib.sha256(audio.read_bytes()).hexdigest()
            manifest = {
                "captures": [
                    {
                        "capture_id": "capture",
                        "path": "audio/capture.wav",
                        "truth": "positive",
                        "split": "test",
                        "samples": 160,
                        "sha256": digest,
                        "speaker_id": "replay-provider-voice",
                        "session_id": "session",
                        "phrase": "Kizz Control",
                        "device_id": "kizz-1",
                        "device_profile": "stackchan",
                        "firmware_sha": "abc",
                        "conditions": {
                            "evidence_role": "reserved_target_channel_positive",
                            "source_audio_sha256": "a" * 64,
                            "source_descriptor_sha256": "b" * 64,
                            "source_provider": "provider",
                            "source_voice": "voice",
                        },
                    }
                ]
            }
            (root / "device-corpus.json").write_text(json.dumps(manifest))
            rows = export_rows(root)
            self.assertEqual(rows[0]["source_audio_sha256"], "a" * 64)
            self.assertEqual(rows[0]["conditions"]["source_voice"], "voice")
            self.assertFalse(rows[0]["training_eligible"])


if __name__ == "__main__":
    unittest.main()
