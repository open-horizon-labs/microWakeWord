import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from microwakeword.kizz_teacher import (
    NegativeSource,
    TeacherBatchSequence,
    build_teacher,
    teacher_loss,
)


class _FakeRaggedMmap:
    def __init__(self, path):
        self.items = [np.ones((280, 40), dtype=np.float32)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class KizzTeacherTest(unittest.TestCase):
    def test_teacher_emits_full_context_state_logits(self):
        model = build_teacher(hidden_size=16, recurrent_layers=1)
        output = model(np.zeros((2, 260, 40), dtype=np.float32))
        self.assertEqual(tuple(output.shape), (2, 66, 23))

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
                "microwakeword.kizz_teacher.RaggedMmap", _FakeRaggedMmap
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


if __name__ == "__main__":
    unittest.main()
