import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.evaluate_observation_manifest import (
    cutoff_contract,
    evaluate_records,
    observation_records,
)


class EvaluateObservationManifestTest(unittest.TestCase):
    def test_requires_validation_cutoff_bound_to_exact_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model.tflite"
            model.write_bytes(b"model")
            report = root / "cutoff.json"
            report.write_text(
                json.dumps(
                    {
                        "selection_split": "validation",
                        "model_sha256": hashlib.sha256(b"model").hexdigest(),
                        "selected_cutoff": 0.71,
                        "sliding_window": 5,
                        "ignore_initial": 25,
                        "clip_duration_ms": 2000,
                    }
                )
            )
            self.assertEqual(cutoff_contract(report, model)["cutoff"], 0.71)
            model.write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "does not match"):
                cutoff_contract(report, model)

    def test_quarantined_snapshot_is_self_contained_and_hash_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "observations" / "false-wakes" / "wake.wav"
            audio.parent.mkdir(parents=True)
            audio.write_bytes(b"pcm")
            manifest = root / "held-out-manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "training_eligible": False,
                        "source_corpus": str(root),
                        "observations": [
                            {
                                "observation_id": "wake",
                                "path": str(audio.relative_to(root)),
                                "audio_sha256": hashlib.sha256(b"pcm").hexdigest(),
                                "weak_label": "false_wake_no_command",
                            }
                        ],
                    }
                )
            )
            _, records = observation_records(manifest)
            self.assertEqual(records[0]["resolved_path"], audio.resolve())
            audio.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                observation_records(manifest)

    def test_unconfirmed_speech_is_reported_separately(self):
        root = Path("/tmp")
        records = [
            {
                "resolved_path": root / "reviewed.wav",
                "observation_id": "reviewed",
                "weak_label": "false_wake_no_command",
                "review": {"reviewer": "human"},
            },
            {
                "resolved_path": root / "unknown.wav",
                "observation_id": "unknown",
                "weak_label": "speech_unconfirmed",
                "review": None,
            },
        ]
        peaks = {"reviewed.wav": 0.8, "unknown.wav": 0.2}
        metrics, scored = evaluate_records(records, 0.7, lambda path: peaks[path.name])
        self.assertEqual(metrics["false_wake_no_command"]["accepted"], 1)
        self.assertEqual(metrics["speech_unconfirmed"]["accepted"], 0)
        self.assertTrue(scored[0]["human_reviewed"])
        self.assertFalse(scored[1]["human_reviewed"])


if __name__ == "__main__":
    unittest.main()
