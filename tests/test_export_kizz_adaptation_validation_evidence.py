import json
import tempfile
import unittest
from pathlib import Path

from microwakeword.kizz_phoneme_teacher import sha256_file
from tools.export_kizz_adaptation_validation_evidence import export


class ExportAdaptationValidationEvidenceTests(unittest.TestCase):
    def test_exports_only_locked_device_validation_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            examples = []
            providers = ("assemblyai", "deepgram", "elevenlabs", "kokoro")
            for index in range(12):
                audio = root / f"{index}.wav"
                audio.write_bytes(str(index).encode())
                examples.append(
                    {
                        "source_id": f"device:{index}",
                        "path": str(audio),
                        "audio_sha256": sha256_file(audio),
                        "split": "validation",
                        "label": 1,
                        "source_group": "device_channel_positive",
                        "provider": providers[index % len(providers)],
                    }
                )
            examples.append({**examples[0], "split": "train"})
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"examples": examples}))
            output = root / "evidence.json"
            result = export(manifest, output)
            self.assertEqual(result["counts"]["total"], 12)
            self.assertTrue(all(row["training_eligible"] is False for row in result["examples"]))


if __name__ == "__main__":
    unittest.main()
