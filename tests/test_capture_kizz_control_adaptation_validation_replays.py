import json
import tempfile
import unittest
from pathlib import Path

from tools.capture_kizz_control_adaptation_validation_replays import (
    EVIDENCE_ROLE,
    PROVIDERS,
    build_capture_request,
    lock_selection,
    select_rows,
)


def _row(provider: str, voice: str, index: int, *, split: str = "validation") -> dict:
    return {
        "provider": provider,
        "voice": voice,
        "voice_id": f"tts:{provider}:{voice}",
        "source_id": f"source:{provider}:{voice}:{index}",
        "provenance_id": f"prov:{provider}:{voice}:{index}",
        "speaker_id": f"speaker:{provider}:{voice}",
        "session_id": f"session:{provider}:{voice}:{index}",
        "descriptor_sha256": f"descriptor-{provider}-{voice}-{index}",
        "audio_sha256": f"audio-{provider}-{voice}-{index}",
        "path": f"audio/{provider}-{voice}-{index}.wav",
        "render_text": f"Kizz Control {index}",
        "label": 1,
        "split": split,
        "target_id": "kizz-control",
        "training_eligible": True,
        "target_phones": ["k", "ɪ", "z"],
        "alignment": {
            "method": "wav2vec2_ipa_ctc_forced_alignment",
            "pronunciation_decision": {"accepted": True},
        },
    }


class CaptureKizzControlAdaptationValidationTests(unittest.TestCase):
    def _inputs(self, root: Path):
        rows = []
        # Deliberately model the real inventory: 4/3/2/3, with duplicate
        # render rows for one voice to test deterministic one-per-voice choice.
        counts = {"assemblyai": 4, "deepgram": 3, "elevenlabs": 2, "kokoro": 3}
        for provider in PROVIDERS:
            for voice_index in range(counts[provider]):
                rows.append(_row(provider, f"validation-{voice_index}", 1))
                rows.append(_row(provider, f"validation-{voice_index}", 0))
        rows.append(_row("assemblyai", "ignored", 0, split="train"))
        aligned = root / "aligned.json"
        aligned.write_text(json.dumps({"examples": rows}))

        target = root / "target.json"
        target.write_text(json.dumps({"examples": [{
            "provider": provider,
            "voice": f"target-{provider}",
            "audio_sha256": f"target-audio-{provider}",
            "source_id": f"target-source-{provider}",
        } for provider in PROVIDERS]}))
        train_corpus = root / "train-corpus.json"
        train_corpus.write_text(json.dumps({"captures": [{
            "provider": "assemblyai",
            "voice": "train-voice",
            "sha256": "train-capture",
            "conditions": {"source_provider": "assemblyai", "source_voice": "train-voice", "source_audio_sha256": "train-source"},
            "speaker_id": "train-speaker",
            "session_id": "train-session",
        }]}))
        train_selection = root / "train-selection.json"
        train_selection.write_text(json.dumps({"selected_examples": [{
            "provider": "deepgram", "voice": "train-selected", "audio_sha256": "train-selected-audio", "source_id": "train-selected-source",
        }]}))
        return aligned, target, train_corpus, train_selection, rows, counts

    def test_selects_every_distinct_eligible_validation_voice_from_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            aligned, target, train_corpus, train_selection, _, counts = self._inputs(Path(directory))
            selected, expected = select_rows(aligned, target, train_corpus, train_selection)
            self.assertEqual(expected, counts)
            self.assertEqual(len(selected), sum(counts.values()))
            self.assertEqual({p: sum(row["provider"] == p for row in selected) for p in PROVIDERS}, counts)
            self.assertEqual(len({(row["provider"], row["voice"]) for row in selected}), len(selected))
            self.assertEqual({row["render_text"] for row in selected if row["voice"] == "validation-0"}, {"Kizz Control 0"})

    def test_locked_selection_binds_inventory_counts_and_role(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aligned, target, train_corpus, train_selection, _, counts = self._inputs(root)
            selection_path = root / "selection.json"
            selected, expected = lock_selection(selection_path, aligned, target, train_corpus, train_selection)
            self.assertEqual(expected, counts)
            self.assertEqual(json.loads(selection_path.read_text())["expected_voice_counts"], counts)
            request = build_capture_request(selected[0], device_id="kizz-1", device_profile="stackchan", duration_ms=5000, volume=0.45, lead_seconds=0.55)
            self.assertEqual(request["split"], "validation")
            self.assertEqual(request["conditions"]["evidence_role"], EVIDENCE_ROLE)

    def test_rejects_source_overlap_with_train_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aligned, target, train_corpus, train_selection, rows, _ = self._inputs(root)
            selected = rows[0]
            train_selection.write_text(json.dumps({"selected_examples": [{"provider": selected["provider"], "voice": "other", "audio_sha256": selected["audio_sha256"]}]}))
            with self.assertRaisesRegex(ValueError, "audio/source hash overlap"):
                select_rows(aligned, target, train_corpus, train_selection)

    def test_rejects_provider_voice_overlap_with_target_qualification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aligned, target, train_corpus, train_selection, rows, _ = self._inputs(root)
            target.write_text(json.dumps({"examples": [{
                "provider": rows[0]["provider"],
                "voice": rows[0]["voice"],
                "audio_sha256": "target-audio",
                "source_id": "target-source",
            }]}))
            with self.assertRaisesRegex(ValueError, "target qualification provider/voice overlap"):
                select_rows(aligned, target, train_corpus, train_selection)


if __name__ == "__main__":
    unittest.main()
