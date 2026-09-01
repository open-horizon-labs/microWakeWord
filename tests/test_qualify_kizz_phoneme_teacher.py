import json
import tempfile
import unittest
from pathlib import Path

from microwakeword.kizz_phoneme_teacher import sha256_file
from tools.qualify_kizz_phoneme_teacher import (
    _validated_adaptation_metadata,
    _wake_context_metadata,
)


class QualifyKizzPhonemeTeacherTests(unittest.TestCase):
    def test_adaptation_report_binds_exact_best_weights_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "best"
            model.mkdir()
            weights = model / "model.safetensors"
            weights.write_bytes(b"weights")
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            report = root / "training.json"
            report.write_text(
                json.dumps(
                    {
                        "kind": "kizz_phoneme_teacher_adaptation",
                        "wake_phrase": {"phrase_id": "kizz-control"},
                        "manifest": {
                            "path": str(manifest),
                            "sha256": sha256_file(manifest),
                        },
                        "checkpoints": {
                            "best": {
                                "path": str(weights),
                                "file_sha256": sha256_file(weights),
                            }
                        },
                    }
                )
            )
            result = _validated_adaptation_metadata(
                report,
                model_directory=model,
                weights_path=weights,
                weights_sha256=sha256_file(weights),
                phrase_id="kizz-control",
            )
            self.assertEqual(result["checkpoint"]["file_sha256"], sha256_file(weights))
            weights.write_bytes(b"drift")
            with self.assertRaisesRegex(ValueError, "not bound"):
                _validated_adaptation_metadata(
                    report,
                    model_directory=model,
                    weights_path=weights,
                    weights_sha256=sha256_file(weights),
                    phrase_id="kizz-control",
                )

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
