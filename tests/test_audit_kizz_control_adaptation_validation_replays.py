import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.audit_kizz_control_adaptation_validation_replays import audit
from tools.capture_kizz_control_adaptation_validation_replays import _selection_payload


class AuditKizzControlAdaptationValidationReplaysTests(unittest.TestCase):
    def test_audits_dynamic_inventory_and_detects_capture_hash_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            providers = {"assemblyai": 4, "deepgram": 3, "elevenlabs": 2, "kokoro": 3}
            aligned_rows = []
            selected = []
            captures = []
            corpus = root / "validation-corpus"
            corpus.mkdir()
            audio_dir = corpus / "audio"
            audio_dir.mkdir()
            time = np.arange(6400, dtype=np.float32) / 16000.0
            for provider, count in providers.items():
                for index in range(count):
                    voice = f"validation-{index}"
                    source_values = (0.2 * np.sin(2 * np.pi * (280 + list(providers).index(provider) * 37 + index * 11) * time)).astype(np.float32)
                    source_path = root / f"source-{provider}-{index}.wav"
                    sf.write(source_path, source_values, 16000, subtype="PCM_16")
                    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
                    row = {"provider": provider, "voice": voice, "voice_id": f"tts:{provider}:{voice}", "source_id": f"source:{provider}:{voice}", "speaker_id": f"speaker:{provider}:{voice}", "session_id": f"session:{provider}:{voice}", "descriptor_sha256": f"descriptor-{provider}-{index}", "audio_sha256": source_hash, "path": str(source_path), "label": 1, "split": "validation", "target_id": "kizz-control", "training_eligible": True, "target_phones": ["k", "ɪ", "z"], "alignment": {"method": "wav2vec2_ipa_ctc_forced_alignment", "pronunciation_decision": {"accepted": True}}}
                    aligned_rows.append(row)
                    selected.append(row)
                    captured = np.concatenate((np.zeros(4800, dtype=np.float32), source_values, np.zeros(4800, dtype=np.float32)))
                    capture_path = audio_dir / f"capture-{provider}-{index}.wav"
                    sf.write(capture_path, captured, 16000, subtype="PCM_16")
                    captures.append({"capture_id": f"capture-{provider}-{index}", "speaker_id": f"replay-validation-{provider}-tts-{provider}-{voice}", "session_id": f"kc-adaptation-validation-{provider}-{voice}-v1", "truth": "positive", "split": "validation", "path": str(capture_path.relative_to(corpus)), "sha256": hashlib.sha256(capture_path.read_bytes()).hexdigest(), "conditions": {"evidence_role": "teacher_adaptation_target_channel_validation_positive", "source_provider": provider, "source_voice": voice, "source_audio_sha256": source_hash}})
            aligned = root / "aligned.json"
            aligned.write_text(json.dumps({"examples": aligned_rows}))
            target = root / "target.json"
            target.write_text(json.dumps({"examples": [{"provider": p, "voice": f"target-{p}", "audio_sha256": f"target-{p}"} for p in providers]}))
            train_corpus = root / "train.json"
            train_corpus.write_text(json.dumps({"captures": []}))
            train_selection = root / "train-selection.json"
            train_selection.write_text(json.dumps({"selected_examples": []}))
            selection = root / "selection.json"
            selection.write_text(json.dumps(_selection_payload(aligned, target, train_corpus, train_selection, selected, providers)))
            (corpus / "device-corpus.json").write_text(json.dumps({"corpus_id": "kizz-control-teacher-adaptation-validation-device-replays-v1", "captures": captures}))
            report = audit(corpus, selection, target, train_corpus, train_selection)
            self.assertTrue(report["qualified"], report["failure_reasons"])
            self.assertEqual(report["expected_voice_counts"], providers)
            captures[0]["sha256"] = "wrong"
            (corpus / "device-corpus.json").write_text(json.dumps({"corpus_id": "kizz-control-teacher-adaptation-validation-device-replays-v1", "captures": captures}))
            failed = audit(corpus, selection, target, train_corpus, train_selection)
            self.assertFalse(failed["qualified"])
            self.assertIn("capture_audio_hash_drift", failed["results"][0]["failure_reasons"])


if __name__ == "__main__":
    unittest.main()
