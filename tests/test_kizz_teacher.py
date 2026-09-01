import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import tensorflow as tf

from microwakeword.kizz_teacher import (
    NegativeSource,
    TeacherBatchSequence,
    build_teacher,
    teacher_loss,
)
from microwakeword.ordered_state import KIZZ_SINGLE_STATE_TOPOLOGY


class _FakeRaggedMmap:
    def __init__(self, path):
        self.items = [np.ones((280, 40), dtype=np.float32)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class _FakeUint16RaggedMmap:
    def __init__(self, path):
        self.items = [np.full((520, 40), 256, dtype=np.uint16)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class KizzTeacherTest(unittest.TestCase):
    def test_teacher_emits_full_context_state_logits(self):
        model = build_teacher(hidden_size=16, recurrent_layers=1)
        output = model(np.zeros((2, 260, 40), dtype=np.float32))
        self.assertEqual(tuple(output.shape), (2, 66, 23))

    def test_single_state_topology_emits_nine_states(self):
        model = build_teacher(
            hidden_size=16, recurrent_layers=1, topology=KIZZ_SINGLE_STATE_TOPOLOGY
        )
        output = model(np.zeros((2, 260, 40), dtype=np.float32))
        self.assertEqual(tuple(output.shape), (2, 66, 9))

    def test_states_per_phone_is_a_direct_topology_option(self):
        model = build_teacher(hidden_size=16, recurrent_layers=1, states_per_phone=1)
        self.assertEqual(model.output_shape[-1], 9)

    def test_teacher_loss_is_finite(self):
        model = build_teacher(hidden_size=16, recurrent_layers=1)
        logits = model(np.zeros((2, 260, 40), dtype=np.float32))
        targets = np.ones((2, 66), dtype=np.int32)
        labels = np.ones((2,), dtype=np.float32)
        value = teacher_loss(logits, targets, labels)
        self.assertTrue(np.isfinite(float(value.numpy())))

    def test_negative_frame_loss_accepts_background_or_silence(self):
        model = build_teacher(hidden_size=16, recurrent_layers=1)
        logits = model(np.zeros((2, 260, 40), dtype=np.float32))
        labels = np.asarray([0.0, 1.0], dtype=np.float32)
        targets_background = np.zeros((2, 66), dtype=np.int32)
        targets_silence = np.ones((2, 66), dtype=np.int32)
        background_loss = teacher_loss(logits, targets_background, labels)
        silence_loss = teacher_loss(logits, targets_silence, labels)
        self.assertAlmostEqual(
            float(background_loss.numpy()), float(silence_loss.numpy()), places=6
        )

    def test_batch_sequence_is_balanced_and_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            features = np.zeros((4, 260, 40), dtype=np.float32)
            targets = np.ones((4, 66), dtype=np.int32)
            np.save(base / "features.npy", features)
            np.save(base / "targets.npy", targets)
            with mock.patch(
                "microwakeword.kizz_teacher.open_feature_archive",
                side_effect=lambda path: _FakeRaggedMmap(path),
            ):
                sequence = TeacherBatchSequence(
                    base / "features.npy",
                    base / "targets.npy",
                    [NegativeSource("negative", base / "negative")],
                    batch_size=4,
                    steps_per_epoch=2,
                    seed=19,
                )
                first = sequence[0]
                second = sequence[0]
            np.testing.assert_array_equal(first[0], second[0])
            np.testing.assert_array_equal(first[1]["states"], second[1]["states"])
            np.testing.assert_array_equal(first[1]["label"], second[1]["label"])
            self.assertEqual(np.sum(first[1]["label"]), 2)

    def test_batch_sequence_accepts_fixed_negative_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            np.save(base / "features.npy", np.zeros((2, 260, 40), dtype=np.float32))
            np.save(base / "targets.npy", np.ones((2, 66), dtype=np.int32))
            np.save(base / "negative.npy", np.ones((3, 260, 40), dtype=np.float16))
            sequence = TeacherBatchSequence(
                base / "features.npy",
                base / "targets.npy",
                [NegativeSource("negative", base / "negative.npy")],
                batch_size=2,
                steps_per_epoch=1,
                seed=19,
            )
            features, batch = sequence[0]
            self.assertEqual(features.shape, (2, 260, 40))
            self.assertEqual(batch["states"].shape, (2, 66))
            self.assertEqual(np.sum(batch["label"]), 1)

    def test_batch_sequence_realizes_uniform_positive_provider_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            features = np.stack(
                [np.full((260, 40), index, dtype=np.float32) for index in range(8)]
            )
            np.save(base / "features.npy", features)
            np.save(base / "targets.npy", np.ones((8, 66), dtype=np.int32))
            np.save(base / "negative.npy", np.zeros((2, 260, 40), dtype=np.float32))
            families = ["assemblyai"] * 2 + ["deepgram"] * 2 + ["elevenlabs"] * 2 + ["kokoro"] * 2
            sequence = TeacherBatchSequence(
                base / "features.npy",
                base / "targets.npy",
                [NegativeSource("negative", base / "negative.npy")],
                batch_size=8,
                steps_per_epoch=4,
                seed=19,
                positive_source_families=families,
            )
            for step in range(4):
                sequence[step]
            self.assertEqual(
                sequence.positive_source_sample_counts,
                {
                    "assemblyai": 4,
                    "deepgram": 4,
                    "elevenlabs": 4,
                    "kokoro": 4,
                },
            )

    def test_batch_sequence_rejects_misaligned_provider_vector(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            np.save(base / "features.npy", np.zeros((2, 260, 40), dtype=np.float32))
            np.save(base / "targets.npy", np.ones((2, 66), dtype=np.int32))
            np.save(base / "negative.npy", np.zeros((2, 260, 40), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "must match positive feature rows"):
                TeacherBatchSequence(
                    base / "features.npy",
                    base / "targets.npy",
                    [NegativeSource("negative", base / "negative.npy")],
                    positive_source_families=["assemblyai"],
                )

    def test_batch_sequence_rejects_labels_outside_topology(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            np.save(base / "features.npy", np.zeros((2, 260, 40), dtype=np.float32))
            np.save(base / "targets.npy", np.full((2, 66), 9, dtype=np.int32))
            np.save(base / "negative.npy", np.ones((2, 260, 40), dtype=np.float16))
            with self.assertRaisesRegex(ValueError, "selected topology state count"):
                TeacherBatchSequence(
                    base / "features.npy",
                    base / "targets.npy",
                    [NegativeSource("negative", base / "negative.npy")],
                    batch_size=2,
                    steps_per_epoch=1,
                    topology=KIZZ_SINGLE_STATE_TOPOLOGY,
                )

    def test_loss_rejects_mismatched_topology(self):
        model = build_teacher(hidden_size=16, recurrent_layers=1)
        logits = model(np.zeros((2, 260, 40), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "state_logits and topology"):
            teacher_loss(
                logits,
                np.ones((2, 66), dtype=np.int32),
                np.ones((2,), dtype=np.float32),
                topology=KIZZ_SINGLE_STATE_TOPOLOGY,
            )

    def test_loss_rejects_target_labels_outside_topology(self):
        model = build_teacher(
            hidden_size=16,
            recurrent_layers=1,
            topology=KIZZ_SINGLE_STATE_TOPOLOGY,
        )
        logits = model(np.zeros((2, 260, 40), dtype=np.float32))
        with self.assertRaises(tf.errors.InvalidArgumentError):
            teacher_loss(
                logits,
                np.full((2, 66), 9, dtype=np.int32),
                np.ones((2,), dtype=np.float32),
                topology=KIZZ_SINGLE_STATE_TOPOLOGY,
            )

    def test_keyword_weight_is_per_example_and_finite(self):
        model = build_teacher(
            hidden_size=16,
            recurrent_layers=1,
            topology=KIZZ_SINGLE_STATE_TOPOLOGY,
        )
        logits = model(np.zeros((2, 260, 40), dtype=np.float32))
        targets = np.ones((2, 66), dtype=np.int32)
        targets[0, :7] = np.arange(2, 9)
        value = teacher_loss(
            logits,
            targets,
            np.asarray([1.0, 0.0], dtype=np.float32),
            keyword_frame_weight=3.0,
            topology=KIZZ_SINGLE_STATE_TOPOLOGY,
        )
        self.assertTrue(np.isfinite(float(value.numpy())))
        with self.assertRaisesRegex(ValueError, "keyword_frame_weight"):
            teacher_loss(
                logits,
                targets,
                np.asarray([1.0, 0.0], dtype=np.float32),
                keyword_frame_weight=0.0,
                topology=KIZZ_SINGLE_STATE_TOPOLOGY,
            )

    def test_negative_examples_use_rejection_state_and_weighted_source_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            np.save(base / "features.npy", np.zeros((2, 260, 40), dtype=np.float32))
            np.save(base / "targets.npy", np.ones((2, 66), dtype=np.int32))
            np.save(base / "negative.npy", np.ones((2, 260, 40), dtype=np.float16))
            sequence = TeacherBatchSequence(
                base / "features.npy",
                base / "targets.npy",
                [NegativeSource("negative", base / "negative.npy")],
                batch_size=2,
                steps_per_epoch=1,
                seed=19,
                negative_state=1,
                negative_source_weights=[1.0],
            )
            _, batch = sequence[0]
            self.assertTrue(np.all(batch["states"][batch["label"] == 0] == 1))

    def test_uint16_archive_negatives_are_decoded_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            np.save(base / "features.npy", np.zeros((2, 260, 40), dtype=np.float32))
            np.save(base / "targets.npy", np.ones((2, 66), dtype=np.int32))
            with mock.patch(
                "microwakeword.kizz_teacher.open_feature_archive",
                side_effect=lambda path: _FakeUint16RaggedMmap(path),
            ):
                sequence = TeacherBatchSequence(
                    base / "features.npy",
                    base / "targets.npy",
                    [NegativeSource("negative", base / "negative")],
                    batch_size=2,
                    steps_per_epoch=1,
                    seed=19,
                )
                features, batch = sequence[0]
            negative = features[batch["label"] == 0]
            self.assertTrue(np.allclose(negative, 10.0))


if __name__ == "__main__":
    unittest.main()
