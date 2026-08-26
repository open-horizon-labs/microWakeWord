import unittest

import numpy as np

from microwakeword.ctc_forward import (
    exhaustive_sliding_forward_score,
    exhaustive_suffix_forward_score,
)
from microwakeword.phoneme_student import compact_phone_contract
from tools.cache_kizz_teacher_sequence_scores import forward_sum_sliding_scores


class TeacherSequenceScoreCacheTests(unittest.TestCase):
    def test_vectorized_scores_match_portable_forward_sum_reference(self):
        contract = compact_phone_contract()
        logits = np.random.default_rng(231).normal(
            size=(2, 40, len(contract["tokens"]))
        ).astype(np.float32)
        logits -= np.max(logits, axis=-1, keepdims=True)
        log_probs = logits - np.log(
            np.exp(logits).sum(axis=-1, keepdims=True)
        )
        scored = forward_sum_sliding_scores(
            log_probs,
            contract,
            window_lengths=(28, 34, 40),
            hop=3,
            beta=0.0,
            batch_size=1,
        )
        expected = [
            exhaustive_sliding_forward_score(
                sequence,
                contract,
                window_lengths=(28, 34, 40),
                hop=3,
                beta=0.0,
            )
            for sequence in log_probs
        ]
        np.testing.assert_array_equal(
            scored["eligible"], [item.eligible for item in expected]
        )
        for index, item in enumerate(expected):
            if item.eligible:
                self.assertEqual(
                    scored["deployment_start_frame"][index], item.start_frame
                )
                self.assertEqual(
                    scored["deployment_end_frame"][index], item.end_frame
                )
                self.assertAlmostEqual(
                    scored["deployment_canonical_fit"][index],
                    item.canonical_fit,
                    places=5,
                )
                self.assertAlmostEqual(
                    scored["deployment_collision_margin"][index],
                    item.collision_margin,
                    places=5,
                )

    def test_rejects_nonfinite_posteriors(self):
        contract = compact_phone_contract()
        values = np.zeros((1, 40, len(contract["tokens"])), dtype=np.float32)
        values[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            forward_sum_sliding_scores(values, contract)

    def test_suffix_reference_matches_vectorized_deployment_score(self):
        contract = compact_phone_contract()
        logits = np.random.default_rng(8128).normal(
            size=(1, 66, len(contract["tokens"]))
        ).astype(np.float32)
        expected = forward_sum_sliding_scores(
            logits,
            contract,
            window_lengths=(19, 23, 27, 32, 39, 47, 54),
            hop=1,
            beta=0.0,
            suffix_only=True,
        )
        actual = exhaustive_suffix_forward_score(
            logits[0],
            contract,
            window_lengths=(19, 23, 27, 32, 39, 47, 54),
            beta=0.0,
        )
        self.assertEqual(actual.eligible, bool(expected["eligible"][0]))
        if actual.eligible:
            self.assertAlmostEqual(
                actual.canonical_fit,
                expected["deployment_canonical_fit"][0],
                places=5,
            )
            self.assertAlmostEqual(
                actual.collision_margin,
                expected["deployment_collision_margin"][0],
                places=5,
            )


if __name__ == "__main__":
    unittest.main()
