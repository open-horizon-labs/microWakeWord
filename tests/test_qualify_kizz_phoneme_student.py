import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tools.qualify_kizz_phoneme_student import (
    _forward_sum_batch_scores,
    _false_wake_context,
    _select_evidence_rows,
    _stream_window_scores,
    _validate_evidence,
    choose_validation_threshold,
    frontend_features,
    poisson_upper_95,
)


class QualifyKizzPhonemeStudentTests(unittest.TestCase):
    def test_threshold_is_validation_only_and_fail_closed_on_scoring_failure(self):
        rows = [
            {"label": 1, "score": 0.9, "duration_seconds": 1.0},
            {"label": 1, "score": 0.8, "duration_seconds": 1.0},
            {"label": 0, "score": 0.1, "duration_seconds": 3600.0},
            {"label": 0, "score": 0.2, "duration_seconds": 3600.0},
        ]
        point = choose_validation_threshold(rows, min_recall=0.9, max_faph=0.1)
        self.assertTrue(point["qualified"])
        self.assertEqual(point["selection"], "validation_only")
        self.assertEqual(point["threshold"], 0.8)
        self.assertEqual(point["zero_false_accept_recall"], 1.0)
        self.assertEqual(point["false_accepts_at_recall_floor"], 0)
        failed = choose_validation_threshold(rows + [{"label": 0, "score": None, "failure_reasons": ["io"]}], min_recall=0.9, max_faph=0.1)
        self.assertFalse(failed["qualified"])
        self.assertEqual(failed["reason"], "validation_scoring_failure")

    def test_poisson_zero_event_upper_bound_is_finite_and_conservative(self):
        self.assertAlmostEqual(poisson_upper_95(0, 100.0), -np.log(0.05) / 100.0)
        self.assertGreater(poisson_upper_95(1, 100.0), poisson_upper_95(0, 100.0))

    def test_raw_audio_path_uses_c_frontend(self):
        class Result:
            samples_read = 160
            features = [[1.0] * 40]

        fake = mock.Mock()
        fake.process_samples.return_value = Result()
        with mock.patch("tools.qualify_kizz_phoneme_student.MicroFrontend", return_value=fake):
            values = frontend_features(np.zeros(320, dtype=np.float32))
        self.assertEqual(values.shape, (3, 40))
        self.assertEqual(fake.process_samples.call_count, 3)

    def test_raw_audio_path_normalizes_flat_frontend_frames(self):
        class Result:
            samples_read = 160
            features = [1.0] * 40

        fake = mock.Mock()
        fake.process_samples.return_value = Result()
        with mock.patch("tools.qualify_kizz_phoneme_student.MicroFrontend", return_value=fake):
            values = frontend_features(np.zeros(320, dtype=np.float32))
        self.assertEqual(values.shape, (3, 40))

    def test_real_frontend_pads_sub_window_audio(self):
        values = frontend_features(np.zeros(418, dtype=np.float32))
        self.assertEqual(values.shape, (1, 40))

    def test_streaming_model_is_called_once_for_a_file(self):
        class FakeModel:
            output_frames = 66

            def __init__(self):
                self.calls = 0

            def stream_logits(self, features, contract):
                self.calls += 1
                return np.zeros((len(features) // 3, len(contract["tokens"])), dtype=np.float32)

        contract = {"tokens": ["blank", "a"], "canonical_path": [1], "collision_paths": {"x": [0]}, "blank_id": 0}
        model = FakeModel()
        scores, timestamps = _stream_window_scores(model, np.zeros((520, 40), dtype=np.float32), contract, beta=0.0)
        self.assertEqual(model.calls, 1)
        self.assertEqual(len(scores), len(timestamps))
        self.assertGreater(len(scores), 1)

    def test_forward_sum_reference_is_used_for_student_decisions(self):
        from tools import qualify_kizz_phoneme_student as qualifier

        expected = mock.Mock(eligible=True, canonical_fit=0.75)
        contract = {"tokens": ["blank", "a"], "canonical_path": [1], "collision_paths": {"x": [0]}, "blank_id": 0}
        with mock.patch.object(
            qualifier, "exhaustive_suffix_forward_score", return_value=expected
        ) as scorer:
            self.assertEqual(
                qualifier._decoder_score(
                    np.zeros((4, 2)),
                    contract,
                    beta=0.2,
                    decoder_algorithm="forward_sum_ctc",
                ),
                0.75,
            )
        args, kwargs = scorer.call_args
        np.testing.assert_array_equal(args[0], np.zeros((4, 2)))
        self.assertEqual(args[1], contract)
        self.assertEqual(kwargs, {"window_lengths": qualifier.WINDOW_LENGTHS, "beta": 0.2})

    def test_vectorized_forward_sum_matches_portable_suffix_reference(self):
        from microwakeword.ctc_forward import exhaustive_suffix_forward_score
        from microwakeword.phoneme_student import compact_phone_contract

        contract = compact_phone_contract()
        sequences = np.random.default_rng(231).normal(
            size=(3, 66, len(contract["tokens"]))
        ).astype(np.float32)
        actual = _forward_sum_batch_scores(sequences, contract, beta=0.0)
        expected = [
            exhaustive_suffix_forward_score(
                sequence,
                contract,
                window_lengths=(19, 23, 27, 32, 39, 47, 54),
                beta=0.0,
            ).canonical_fit
            for sequence in sequences
        ]
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_false_wake_context_is_strictly_pre_trigger(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "wake.wav"
            import soundfile as sf
            samples = np.arange(64_000, dtype=np.float32) / 64_000.0
            sf.write(audio, samples, 16_000, subtype="PCM_16")
            import hashlib
            digest = hashlib.sha256(audio.read_bytes()).hexdigest()
            metadata = root / "wake.json"
            metadata.write_text(json.dumps({"observation_id": "wake-1", "sha256": digest, "pre_wake_ms": 3000}))
            row = {"source_id": "false-wake:wake-1", "audio_sha256": digest, "metadata_path": str(metadata)}
            selected, context = _false_wake_context(row, samples, context_seconds=2.0)
            self.assertEqual(len(selected), 32_000)
            self.assertEqual(context["context_start_seconds"], 1.0)
            self.assertEqual(context["context_end_seconds"], 3.0)
            self.assertLessEqual((1.0 + len(selected) / 16_000.0), 3.0)

    def test_evidence_file_hashes_and_partition_overlap_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "a.raw"
            audio.write_bytes(b"audio")
            import hashlib
            digest = hashlib.sha256(audio.read_bytes()).hexdigest()

            def manifest(name, source_id, *, label=0, locked=False, split=None):
                path = root / f"{name}.json"
                row = {"path": str(audio), "audio_sha256": digest, "source_id": source_id, "label": label, "training_eligible": False, "locked_deployment_anchor": locked}
                if split is not None:
                    row["split"] = split
                path.write_text(json.dumps({"examples": [row]}))
                return path

            paths = {"validation": manifest("validation", "v", split="validation"), "test": manifest("test", "t", label=1, split="test"), "target": manifest("target", "n", label=1), "false_wakes": manifest("false", "f", locked=True)}
            with self.assertRaisesRegex(ValueError, "overlap"):
                _validate_evidence(paths)

    def test_shared_manifest_partitions_are_explicit(self):
        rows = [{"split": "validation", "label": 0}, {"split": "test", "label": 1}]
        self.assertEqual(_select_evidence_rows("validation", rows), [rows[0]])
        self.assertEqual(_select_evidence_rows("test", rows), [rows[1]])
        with self.assertRaisesRegex(ValueError, "no validation"):
            _select_evidence_rows("validation", [{"split": "train", "label": 0}])
        with self.assertRaisesRegex(ValueError, "only label=1"):
            _select_evidence_rows("test", [{"split": "test", "label": 0}])

    def test_uint8_is_the_deployed_output_contract(self):
        source = Path(__file__).parents[1] / "tools/qualify_kizz_phoneme_student.py"
        self.assertIn('get("output") or {}).get("dtype") != "uint8"', source.read_text())


if __name__ == "__main__":
    unittest.main()
