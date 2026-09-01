import hashlib
import json
import tempfile
import unittest
import wave
from pathlib import Path

from tools.compose_kizz_control_adaptation_validation_replays import (
    COMPOSITION_ALGORITHM,
    compose,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


class ComposeTests(unittest.TestCase):
    def _wav(self, path: Path, value: int) -> str:
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(16000); stream.writeframes((value.to_bytes(2, "little", signed=True)) * 160)
        return _sha(path)

    def _setup(self, root: Path, outcomes=(True, False)):
        source = root / "source.wav"; source_hash = self._wav(source, 100)
        selection = {"schema_version": 1, "selection_algorithm": "provider_voice_aligned_validation_round_robin_v1", "qualification_evidence_sha256": "evidence", "selected_examples": [{"provider": "deepgram", "voice": "v", "audio_sha256": source_hash, "path": str(source)}]}
        selection["selection_sha256"] = hashlib.sha256(_canonical({k: v for k, v in selection.items() if k != "selection_sha256"})).hexdigest()
        selection_path = root / "selection.json"; selection_path.write_text(json.dumps(selection))
        attempts = []
        for index, qualified in enumerate(outcomes):
            corpus = root / f"attempt-{index}"; audio = corpus / "audio" / f"capture-{index}.wav"; audio.parent.mkdir(parents=True)
            audio_hash = self._wav(audio, 200 + index)
            manifest = {"schema_version": 2, "corpus_id": "kizz-control-teacher-adaptation-validation-device-replays-v1", "device_profiles": {"stackchan": {"audio": {"sample_rate": 16000}}}, "speakers": {"s": {"provider": "deepgram", "voice": "v"}}, "captures": [{"capture_id": f"c{index}", "provider": "deepgram", "voice": "v", "path": str(audio.relative_to(corpus)), "sha256": audio_hash, "conditions": {"source_provider": "deepgram", "source_voice": "v", "source_audio_sha256": source_hash}}]}
            corpus.mkdir(exist_ok=True); manifest_path = corpus / "device-corpus.json"; manifest_path.write_text(json.dumps(manifest))
            report = {"schema_version": 1, "kind": "kizz_control_teacher_adaptation_device_replay_quality", "gate_scope": "validation_only_target_channel_positive_quality", "qualified": qualified, "inputs": {"corpus_sha256": _sha(manifest_path), "selection_sha256": _sha(selection_path), "qualification_evidence_sha256": "evidence"}, "results": [{"capture_id": f"c{index}", "provider": "deepgram", "voice": "v", "source_audio_sha256": source_hash, "audio_sha256": audio_hash, "qualified": qualified, "metric": index}]}
            report_path = root / f"report-{index}.json"; report_path.write_text(json.dumps(report)); attempts.append((corpus, report_path))
        return selection_path, attempts

    def test_deterministic_first_qualified_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            selection, attempts = self._setup(Path(directory), (True, True))
            payload = compose(selection, attempts, Path(directory) / "out")
            self.assertEqual(payload["composition"]["algorithm"], COMPOSITION_ALGORITHM)
            self.assertEqual(payload["composition"]["selected"][0]["selected_attempt_index"], 0)
            self.assertEqual(payload["captures"][0]["path"], str((Path(directory) / "attempt-0/audio/capture-0.wav").resolve()))

    def test_all_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            selection, attempts = self._setup(Path(directory), (False, False))
            with self.assertRaisesRegex(ValueError, "no qualified attempt"):
                compose(selection, attempts, Path(directory) / "out")

    def test_stale_report_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); selection, attempts = self._setup(root)
            report = json.loads(attempts[0][1].read_text()); report["inputs"]["corpus_sha256"] = "stale"; attempts[0][1].write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "stale corpus"):
                compose(selection, attempts, root / "out")

    def test_capture_hash_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); selection, attempts = self._setup(root)
            audio = root / "attempt-0/audio/capture-0.wav"; audio.write_bytes(audio.read_bytes() + b"drift")
            with self.assertRaisesRegex(ValueError, "capture/file hash drift"):
                compose(selection, attempts, root / "out")

    def test_capture_source_hash_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); selection, attempts = self._setup(root)
            manifest_path = attempts[0][0] / "device-corpus.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["captures"][0]["conditions"]["source_audio_sha256"] = "stale"
            manifest_path.write_text(json.dumps(manifest))
            report = json.loads(attempts[0][1].read_text())
            report["inputs"]["corpus_sha256"] = _sha(manifest_path)
            attempts[0][1].write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "capture/source hash mismatch"):
                compose(selection, attempts, root / "out")


if __name__ == "__main__":
    unittest.main()
