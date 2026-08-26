import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from microwakeword.kizz_phoneme_teacher import sha256_file
from tools.audit_kizz_control_adaptation_replays import audit


class AuditKizzControlAdaptationReplaysTests(unittest.TestCase):
    def _fixture(self, root: Path):
        providers = ("assemblyai", "deepgram", "elevenlabs", "kokoro")
        source_rows = []
        captures = []
        audio = root / "audio"
        audio.mkdir()
        t = np.arange(6400, dtype=np.float32) / 16000.0
        for provider_index, provider in enumerate(providers):
            for variant in range(4):
                frequency = 300 + provider_index * 50 + variant * 7
                source_values = (
                    0.20
                    * np.sin(2 * np.pi * frequency * t)
                    * (0.25 + 0.75 * (np.sin(2 * np.pi * (5 + variant) * t) > 0))
                ).astype(np.float32)
                voice = f"{provider}-train-{variant}"
                source_path = root / f"source-{provider}-{variant}.wav"
                sf.write(source_path, source_values, 16000, subtype="PCM_16")
                source_hash = sha256_file(source_path)
                capture_path = audio / f"capture-{provider}-{variant}.wav"
                captured = np.concatenate(
                    (np.zeros(4800, dtype=np.float32), source_values, np.zeros(4800, dtype=np.float32))
                )
                sf.write(capture_path, captured, 16000, subtype="PCM_16")
                source_rows.append(
                    {
                        "provider": provider,
                        "voice": voice,
                        "path": str(source_path),
                        "audio_sha256": source_hash,
                    }
                )
                captures.append(
                    {
                        "capture_id": f"capture-{provider}-{variant}",
                        "truth": "positive",
                        "split": "train",
                        "path": str(capture_path.relative_to(root)),
                        "sha256": sha256_file(capture_path),
                        "conditions": {
                            "evidence_role": "teacher_adaptation_target_channel_positive",
                            "source_provider": provider,
                            "source_voice": voice,
                            "source_audio_sha256": source_hash,
                        },
                    }
                )
        corpus = root / "device-corpus.json"
        corpus.write_text(json.dumps({"captures": captures}))
        selection = root / "selection.json"
        selection.write_text(json.dumps({"selected_examples": source_rows}))
        qualification = root / "qualification.json"
        qualification.write_text(
            json.dumps(
                {
                    "examples": [
                        {
                            "conditions": {
                                "source_provider": provider,
                                "source_voice": f"{provider}-heldout",
                            }
                        }
                        for provider in providers
                    ]
                }
            )
        )
        return corpus, selection, qualification

    def test_exact_provider_voice_quality_and_overlap_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus, selection, qualification = self._fixture(Path(directory))
            report = audit(corpus, selection, qualification)
            self.assertTrue(report["qualified"], report["failure_reasons"])
            self.assertEqual(report["counts"]["providers"], {
                "assemblyai": 4, "deepgram": 4, "elevenlabs": 4, "kokoro": 4,
            })
            heldout = json.loads(qualification.read_text())
            heldout["examples"][0]["conditions"]["source_voice"] = "assemblyai-train-0"
            qualification.write_text(json.dumps(heldout))
            failed = audit(corpus, selection, qualification)
            self.assertFalse(failed["qualified"])
            self.assertTrue(
                any("qualification_voice_overlap" in reason for reason in failed["failure_reasons"])
            )


if __name__ == "__main__":
    unittest.main()
