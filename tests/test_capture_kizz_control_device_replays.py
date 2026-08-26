import json
import tempfile
import unittest
from pathlib import Path

from tools.capture_kizz_control_device_replays import (
    _ensure_capture_corpus,
    _speaker_id,
    _wait_for_pending_clear,
    lock_selected_evidence,
    replay_rows,
)


def _row(provider, voice, index, *, audio_hash=None):
    return {
        "provider": provider,
        "voice": voice,
        "voice_id": f"tts:{provider}:{voice}",
        "source_id": f"{provider}:{voice}:{index}",
        "descriptor_sha256": f"descriptor-{provider}-{voice}-{index}",
        "audio_sha256": audio_hash or f"audio-{provider}-{voice}-{index}",
        "render_text": f"Kizz Control {index}",
        "label": 1,
        "reserved_evidence_role": "target_channel_positive",
        "training_eligible": False,
    }


class CaptureKizzControlDeviceReplayTests(unittest.TestCase):
    def test_pending_wait_accepts_capture_persisted_at_timeout_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "device-corpus.json").write_text(
                json.dumps({"captures": [{"capture_id": "capture-1"}]})
            )
            _wait_for_pending_clear(
                root,
                "capture-1",
                service_url="http://unused.invalid",
                timeout=0.0,
            )

    def test_speaker_identity_preserves_provider_and_voice(self):
        row = _row("deepgram", "aura-2-arcas-en", 0)
        self.assertEqual(
            _speaker_id(row),
            "replay-deepgram-tts-deepgram-aura-2-arcas-en",
        )

    def test_initializes_voice_registered_device_corpus(self):
        rows = [
            _row("assemblyai", "voice-a", 0),
            _row("assemblyai", "voice-b", 0),
        ]
        audio = {
            "sample_rate": 16000,
            "channels": 1,
            "sample_format": "s16le",
            "frontend": "m5unified_mic",
            "gain_profile": "room_scale_4x",
            "preprocessing": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _ensure_capture_corpus(
                root,
                rows,
                device_profile="stackchan-v2",
                audio_profile=audio,
            )
            payload = json.loads((root / "device-corpus.json").read_text())
            self.assertEqual(payload["captures"], [])
            self.assertEqual(len(payload["speakers"]), 2)
            self.assertEqual(
                payload["device_profiles"]["stackchan-v2"]["audio"], audio
            )

    def test_stratifies_each_voice_before_taking_a_second_clip(self):
        rows = []
        for voice in ("voice-b", "voice-a", "voice-c"):
            rows.extend(_row("deepgram", voice, index) for index in range(2))
        rows.extend(_row("assemblyai", "only-voice", index) for index in range(3))
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "source.json"
            manifest.write_text(json.dumps({"examples": rows}))
            selected = replay_rows(manifest, ("deepgram",), per_provider=5)
        self.assertEqual(
            [(row["voice"], row["source_id"]) for row in selected],
            [
                ("voice-a", "deepgram:voice-a:0"),
                ("voice-b", "deepgram:voice-b:0"),
                ("voice-c", "deepgram:voice-c:0"),
                ("voice-a", "deepgram:voice-a:1"),
                ("voice-b", "deepgram:voice-b:1"),
            ],
        )

    def test_rejects_duplicate_audio_hashes(self):
        rows = [
            _row("assemblyai", "voice-a", 0, audio_hash="same"),
            _row("assemblyai", "voice-b", 0, audio_hash="same"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "source.json"
            manifest.write_text(json.dumps({"examples": rows}))
            with self.assertRaises(ValueError):
                replay_rows(manifest, ("assemblyai",), per_provider=2)

    def test_locked_selection_is_reused_and_not_reselected(self):
        rows = [_row("assemblyai", "voice-a", index) for index in range(2)]
        rows += [_row("assemblyai", "voice-b", index) for index in range(2)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "source.json"
            manifest.write_text(json.dumps({"examples": rows}))
            selection_path = root / "selected-evidence-v1.json"
            first = lock_selected_evidence(
                manifest,
                selection_path,
                ("assemblyai",),
                per_provider=2,
            )
            self.assertTrue(selection_path.is_file())
            reused = lock_selected_evidence(
                manifest,
                selection_path,
                ("assemblyai",),
                per_provider=2,
            )
            self.assertEqual(reused, first)

            # Change ordering and add a more attractive row. A locked selection
            # must remain the pre-scoring decision, not silently drift.
            rows.insert(0, _row("assemblyai", "voice-a", 99))
            manifest.write_text(json.dumps({"examples": rows}))
            with self.assertRaises(ValueError):
                lock_selected_evidence(
                    manifest,
                    selection_path,
                    ("assemblyai",),
                    per_provider=2,
                )
            self.assertEqual([row["source_id"] for row in first], [
                "assemblyai:voice-a:0",
                "assemblyai:voice-b:0",
            ])

    def test_lock_records_existing_v1_evidence_without_replacing_it(self):
        rows = [_row("assemblyai", "voice-a", index) for index in range(2)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "source.json"
            manifest.write_text(json.dumps({"examples": rows}))
            selection_path = root / "selected-evidence-v1.json"
            lock_selected_evidence(
                manifest,
                selection_path,
                ("assemblyai",),
                per_provider=1,
                existing_capture_ids=("old-capture",),
            )
            payload = json.loads(selection_path.read_text())
            self.assertEqual(payload["preserved_existing_v1_capture_ids"], ["old-capture"])
            self.assertTrue(payload["locked_before_teacher_scoring"])


if __name__ == "__main__":
    unittest.main()
