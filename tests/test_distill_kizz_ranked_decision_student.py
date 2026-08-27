import unittest

import numpy as np
import tensorflow as tf

from tools.distill_kizz_ranked_decision_student import (
    RankedBatcher,
    deployment_frame_scores,
    ranknet_loss,
    rolling_mean_scores,
)


class RankedDecisionStudentTest(unittest.TestCase):
    def test_rolling_score_rejects_single_frame_spike(self):
        logits = tf.constant([[[0.0], [0.0], [9.0], [0.0], [0.0]]])
        self.assertAlmostEqual(float(rolling_mean_scores(logits)[0]), 4.5)

    def test_collision_evidence_reduces_deployed_score(self):
        neutral = tf.constant([[[2.0, 3.0, 0.0, 0.0]]])
        collision = tf.constant([[[2.0, 0.0, 3.0, 0.0]]])
        self.assertGreater(
            float(deployment_frame_scores(neutral)[0, 0]),
            float(deployment_frame_scores(collision)[0, 0]),
        )

    def test_teacher_ordering_rewards_matching_student_order(self):
        teacher = tf.constant([2.0, 0.5, -1.0])
        mask = tf.ones(3)
        correct = ranknet_loss(tf.constant([3.0, 0.0, -3.0]), teacher, mask)
        reversed_ = ranknet_loss(tf.constant([-3.0, 0.0, 3.0]), teacher, mask)
        self.assertLess(float(correct), float(reversed_))

    def test_mask_excludes_examples_without_teacher_scores(self):
        base = ranknet_loss(
            tf.constant([2.0, -2.0]), tf.constant([1.0, -1.0]), tf.ones(2)
        )
        masked = ranknet_loss(
            tf.constant([2.0, -2.0, 100.0]),
            tf.constant([1.0, -1.0, -100.0]),
            tf.constant([1.0, 1.0, 0.0]),
        )
        np.testing.assert_allclose(float(base), float(masked), rtol=1e-6)

    def test_critical_collision_gets_distinct_auxiliary_label(self):
        batcher = object.__new__(RankedBatcher)
        batcher.batch_size = 2
        batcher.seed = 1
        batcher.features = np.zeros((2, 260, 40), np.float32)
        batcher.causal_scores = np.zeros((2, 66), np.float32)
        batcher.overlay_positive = np.zeros((1, 260, 40), np.float32)
        batcher.clean_positive = np.asarray([0])
        batcher.device_positive = np.asarray([0])
        batcher.critical_collision = np.asarray([1])
        batcher.student_hard_negative = np.asarray([1])
        batcher.negative = np.asarray([1])
        batcher.critical_collision_set = {1}
        batcher.expanded = np.zeros((1, 260, 40), np.float32)
        batcher.student_hard_expanded = np.asarray([0])
        batcher.expanded_order = np.asarray([0])
        batcher.noise = [np.zeros((1, 260, 40), np.float32)]
        _, labels, _, _, auxiliary = batcher.batch(0)
        self.assertEqual(
            set(zip(labels.tolist(), auxiliary.tolist())), {(1.0, 0), (0.0, 1)}
        )


if __name__ == "__main__":
    unittest.main()
