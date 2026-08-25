import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tools.qualify_kizz_teacher_v3 import (
    _feature_provenance,
    _rows,
    choose_validation_threshold,
    dedupe_positive_records,
    score_logits,
    sha256_file,
    validate_false_wake_anchor_contract,
)


class FakeModel:
    def __init__(self, logits):
        self.logits = np.asarray(logits, dtype=np.float32)

    def predict(self, values, verbose=0):
        return self.logits[: len(values)]


class QualifyKizzTeacherV3Tests(unittest.TestCase):
    def test_rows_accepts_frozen_observation_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(
                json.dumps({"observations": [{"observation_id": "wake-1"}]})
            )
            self.assertEqual(_rows(path), [{"observation_id": "wake-1"}])

    def test_threshold_is_selected_from_validation_and_reports_failure(self):
        result = choose_validation_threshold(
            [0.9, 0.8, 0.7],
            [0.85, 0.75],
            7200,
            min_recall=2 / 3,
            max_faph=0.1,
        )
        self.assertFalse(result["qualified"])
        self.assertIsNone(result["threshold"])
        self.assertEqual(result["selection_scope"], "validation_only")
        self.assertEqual(result["threshold_at_recall_floor"], 0.7)

    def test_threshold_does_not_use_heldout_scores(self):
        result = choose_validation_threshold(
            [0.9, 0.8, 0.7],
            [0.1, 0.2],
            7200,
            min_recall=2 / 3,
            max_faph=0.1,
        )
        self.assertTrue(result["qualified"])
        self.assertEqual(result["threshold"], 0.7)
        self.assertEqual(result["selection_scope"], "validation_only")

    def test_positive_dedupe_uses_audio_identity_and_preserves_duplicate_audit(self):
        kept, duplicates = dedupe_positive_records(
            [
                {"source_id": "b", "audio_sha256": "same", "score": 2},
                {"source_id": "a", "audio_sha256": "same", "score": 3},
                {"source_id": "c", "audio_sha256": "other", "score": 1},
            ]
        )
        self.assertEqual(len(kept), 2)
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(duplicates[0]["stable_identity"], "audio_sha256:same")

    def test_duration_bounds_are_forwarded_to_existing_decoder(self):
        topology = mock.Mock(state_count=9)
        logits = np.zeros((1, 4, 9), dtype=np.float32)
        with mock.patch(
            "tools.qualify_kizz_teacher_v3.ordered_state_duration_score_numpy",
            return_value=np.asarray([1.25]),
        ) as decoder:
            result = score_logits(
                FakeModel(logits),
                np.zeros((1, 260, 40), dtype=np.float32),
                topology=topology,
                minimum_path_frames=24,
                maximum_path_frames=50,
                batch_size=1,
            )
        self.assertEqual(result.tolist(), [1.25])
        self.assertEqual(decoder.call_args.kwargs["minimum_path_frames"], 24)
        self.assertEqual(decoder.call_args.kwargs["maximum_path_frames"], 50)

    def test_false_wake_cache_requires_exact_locked_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "wake.wav"
            audio.write_bytes(b"locked wake")
            audio_sha = sha256_file(audio)
            manifest = root / "anchors.json"
            manifest.write_text(
                json.dumps(
                    {
                        "examples": [
                            {
                                "source_id": "false-wake:wake-1",
                                "audio_sha256": audio_sha,
                                "path": str(audio),
                                "label": 0,
                                "locked_deployment_anchor": True,
                                "training_eligible": False,
                            }
                        ]
                    }
                )
            )
            observations = {f"audio_sha256:{audio_sha}": {}}
            payload = {
                "manifest": str(manifest),
                "manifest_sha256": sha256_file(manifest),
            }
            result = validate_false_wake_anchor_contract(
                payload, observations, manifest, expected_count=1
            )
            self.assertEqual(result["anchor_audio_contract"]["count"], 1)
            with self.assertRaisesRegex(ValueError, "does not declare"):
                validate_false_wake_anchor_contract(
                    {}, observations, manifest, expected_count=1
                )
            with self.assertRaisesRegex(ValueError, "hash is stale"):
                validate_false_wake_anchor_contract(
                    {**payload, "manifest_sha256": "stale"},
                    observations,
                    manifest,
                    expected_count=1,
                )

    def test_feature_provenance_resolves_and_hashes_parent_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "positive.wav"
            audio.write_bytes(b"positive")
            audio_sha = sha256_file(audio)
            parents = root / "parents.json"
            parents.write_text(
                json.dumps(
                    {
                        "examples": [
                            {
                                "source_id": "parent-1",
                                "audio_sha256": audio_sha,
                                "path": str(audio),
                                "speaker_id": "speaker-1",
                                "split": "validation",
                            }
                        ]
                    }
                )
            )
            provenance = root / "features.json"
            provenance.write_text(
                json.dumps(
                    {
                        "positive_manifests": [
                            {"path": str(parents), "sha256": sha256_file(parents)}
                        ],
                        "examples": [
                            {
                                "source_id": "parent-1::clean",
                                "parent_source_id": "parent-1",
                                "source_audio_sha256": audio_sha,
                                "split": "validation",
                                "augmentation": None,
                            }
                        ],
                    }
                )
            )
            rows = _feature_provenance(provenance, "validation")
            self.assertEqual(rows[0]["speaker_id"], "speaker-1")
            self.assertEqual(rows[0]["feature_source_id"], "parent-1::clean")
            parents.write_text("{}")
            with self.assertRaisesRegex(ValueError, "stale positive"):
                _feature_provenance(provenance, "validation")

    def test_scope_has_no_training_write_path(self):
        source = Path(__file__).parents[1] / "tools" / "qualify_kizz_teacher_v3.py"
        text = source.read_text()
        self.assertNotIn("np.save(", text)
        self.assertNotIn("unlink(", text)
        self.assertIn('"training_data_modified": False', text)


if __name__ == "__main__":
    unittest.main()
