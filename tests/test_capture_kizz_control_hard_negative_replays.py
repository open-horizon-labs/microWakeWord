import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.capture_kizz_control_hard_negative_replays import (
    _capture_id,
    load_sources,
)


class HardNegativeReplayCaptureTest(unittest.TestCase):
    def test_loads_hash_bound_sources_and_ids_bind_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "negative.wav"
            audio.write_bytes(b"source-audio")
            digest = hashlib.sha256(audio.read_bytes()).hexdigest()
            manifest = root / "sources.json"
            manifest.write_text(
                json.dumps(
                    {
                        "examples": [
                            {
                                "source_id": "near-kids-01",
                                "speaker_id": "near-kids-voice",
                                "path": audio.name,
                                "audio_sha256": digest,
                                "truth": "hard_negative",
                                "source": "synthetic_playback",
                                "phrase": "Kids control the volume",
                            }
                        ]
                    }
                )
            )
            row = load_sources(manifest)[0]
            self.assertEqual(Path(row["path"]), audio.resolve())
            first = _capture_id(row, 0.35, 1)
            self.assertEqual(first, _capture_id(row, 0.35, 1))
            self.assertNotEqual(first, _capture_id(row, 0.45, 1))
            self.assertNotEqual(first, _capture_id(row, 0.35, 2))

    def test_rejects_ambient_truth_as_synthetic_playback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "negative.wav"
            audio.write_bytes(b"source-audio")
            manifest = root / "sources.json"
            manifest.write_text(
                json.dumps(
                    {
                        "examples": [
                            {
                                "source_id": "music-01",
                                "speaker_id": "music-source",
                                "path": audio.name,
                                "audio_sha256": hashlib.sha256(
                                    audio.read_bytes()
                                ).hexdigest(),
                                "truth": "ambient_negative",
                                "source": "synthetic_playback",
                                "phrase": "Classical music",
                            }
                        ]
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "requires source=ambient"):
                load_sources(manifest)


if __name__ == "__main__":
    unittest.main()
