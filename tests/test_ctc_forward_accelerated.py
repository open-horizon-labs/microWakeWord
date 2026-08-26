import unittest

import numpy as np

from microwakeword.ctc_forward import exhaustive_suffix_forward_score
from microwakeword.ctc_forward_accelerated import suffix_forward_sum_scores
from microwakeword.phoneme_student import compact_phone_contract


class AcceleratedForwardSumTests(unittest.TestCase):
    def test_matches_portable_suffix_reference(self):
        contract = compact_phone_contract()
        logits = np.random.default_rng(238).normal(
            size=(8, 66, len(contract["tokens"]))
        ).astype(np.float32)
        lengths = (19, 23, 27, 32, 39, 47, 54)
        actual = suffix_forward_sum_scores(
            logits, contract, window_lengths=lengths, beta=0.0
        )
        expected = [
            exhaustive_suffix_forward_score(
                sequence, contract, window_lengths=lengths, beta=0.0
            ).canonical_fit
            for sequence in logits
        ]
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_rejects_nonfinite_logits(self):
        contract = compact_phone_contract()
        logits = np.zeros((1, 66, len(contract["tokens"])), dtype=np.float32)
        logits[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "invalid accelerated"):
            suffix_forward_sum_scores(
                logits, contract, window_lengths=(19,), beta=0.0
            )


if __name__ == "__main__":
    unittest.main()
