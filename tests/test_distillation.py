import unittest

import numpy as np
import tensorflow as tf

from microwakeword.distillation import distillation_loss, teacher_kl_loss


class DistillationTest(unittest.TestCase):
    def test_teacher_kl_is_finite_and_zero_for_equal_logits(self):
        logits = tf.zeros((2, 66, 23), dtype=tf.float32)
        value = teacher_kl_loss(logits, logits)
        self.assertTrue(np.isfinite(float(value.numpy())))
        self.assertAlmostEqual(float(value.numpy()), 0.0, places=6)

    def test_distillation_loss_is_finite(self):
        student = tf.zeros((2, 66, 23), dtype=tf.float32)
        teacher = tf.ones((2, 66, 23), dtype=tf.float32)
        targets = tf.ones((2, 66), dtype=tf.int32)
        value = distillation_loss(student, teacher, targets)
        self.assertTrue(np.isfinite(float(value.numpy())))


if __name__ == "__main__":
    unittest.main()
