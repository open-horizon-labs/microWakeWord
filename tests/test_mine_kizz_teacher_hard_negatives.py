import unittest

import numpy as np

from microwakeword.ordered_state import (
    KIZZ_SINGLE_STATE_TOPOLOGY,
    ordered_state_sequence_score_numpy,
)
from tools.mine_kizz_teacher_hard_negatives import (
    candidate_windows,
    sequence_scores_from_logits,
)


class _Archive:
    def __init__(self):
        self.items = [np.zeros((500, 40), dtype=np.uint16) for _ in range(10)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class MineKizzTeacherHardNegativesTest(unittest.TestCase):
    def test_vectorized_scores_match_reference_recurrence(self):
        rng = np.random.default_rng(231)
        logits = rng.normal(size=(5, 17, 9)).astype(np.float32)
        expected = ordered_state_sequence_score_numpy(
            logits, KIZZ_SINGLE_STATE_TOPOLOGY
        )
        np.testing.assert_allclose(
            sequence_scores_from_logits(logits), expected, rtol=1e-6, atol=1e-6
        )

    def test_candidates_are_bounded_and_deterministic(self):
        first = candidate_windows(_Archive(), max_items=4, windows_per_item=2, seed=7)
        second = candidate_windows(_Archive(), max_items=4, windows_per_item=2, seed=7)
        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 8)
        self.assertTrue(all(0 <= start <= 240 for _, start in first))


if __name__ == "__main__":
    unittest.main()
