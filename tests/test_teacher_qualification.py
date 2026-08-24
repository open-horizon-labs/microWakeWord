import unittest

import numpy as np

from microwakeword.ordered_state import ordered_state_sequence_score_numpy
from tools.qualify_kizz_teacher import choose_operating_point, fast_sequence_scores


class TeacherQualificationTest(unittest.TestCase):
    def test_fast_score_matches_reference_viterbi(self):
        rng = np.random.default_rng(24113)
        logits = rng.normal(size=(4, 66, 23))
        np.testing.assert_allclose(
            fast_sequence_scores(logits),
            ordered_state_sequence_score_numpy(logits),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_selects_point_that_meets_both_limits(self):
        result = choose_operating_point(
            np.asarray([3.0, 2.0, 1.0]),
            np.asarray([0.0, -1.0]),
            7200.0,
            min_recall=0.90,
            max_faph=0.10,
        )
        self.assertTrue(result["qualified"])
        self.assertEqual(result["threshold"], 1.0)
        self.assertEqual(result["false_accepts"], 0)

    def test_rejects_when_high_recall_has_too_many_false_accepts(self):
        result = choose_operating_point(
            np.asarray([3.0, 2.0, 1.0]),
            np.asarray([2.5, 2.0]),
            7200.0,
            min_recall=0.90,
            max_faph=0.10,
        )
        self.assertFalse(result["qualified"])


if __name__ == "__main__":
    unittest.main()
