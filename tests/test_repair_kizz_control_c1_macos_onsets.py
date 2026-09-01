import hashlib
import json
import tempfile
import unittest
import wave
from pathlib import Path

from tools.repair_kizz_control_c1_macos_onsets import repair


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RepairKizzControlC1MacosOnsetsTests(unittest.TestCase):
    def test_prepends_context_without_mutating_parent_pcm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_audio = root / "source.wav"
            parent_pcm = b"\x01\x02" * 160
            with wave.open(str(source_audio), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16_000)
                wav.writeframes(parent_pcm)
            parent_hash = sha256(source_audio)
            other_audio = root / "other.wav"
            other_audio.write_bytes(b"untouched")
            manifest = root / "source.json"
            manifest.write_text(
                json.dumps(
                    {
                        "examples": [
                            {
                                "source_id": "macos-1",
                                "provider": "macos-say",
                                "split": "train",
                                "path": str(source_audio),
                                "audio_sha256": parent_hash,
                                "duration_seconds": 0.01,
                            },
                            {
                                "source_id": "other-1",
                                "provider": "kokoro",
                                "split": "train",
                                "path": str(other_audio),
                                "audio_sha256": sha256(other_audio),
                            },
                        ]
                    }
                )
            )
            payload = repair(
                manifest,
                root / "repaired",
                root / "repaired-manifest.json",
            )
            repaired = payload["examples"][0]
            output = Path(repaired["path"])
            with wave.open(str(output), "rb") as wav:
                output_pcm = wav.readframes(wav.getnframes())

            self.assertEqual(sha256(source_audio), parent_hash)
            self.assertEqual(output_pcm[: 3200 * 2], b"\0\0" * 3200)
            self.assertEqual(output_pcm[3200 * 2 :], parent_pcm)
            self.assertEqual(repaired["parent_audio_sha256"], parent_hash)
            self.assertEqual(repaired["audio_sha256"], sha256(output))
            self.assertEqual(repaired["audio_repair"]["leading_context_ms"], 200)
            self.assertEqual(payload["examples"][1]["path"], str(other_audio))
            self.assertFalse(payload["repair"]["raw_source_mutated"])


if __name__ == "__main__":
    unittest.main()
