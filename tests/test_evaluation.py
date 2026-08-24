import unittest

import numpy as np

from microwakeword.test import false_negative_rates_at_cutoffs


class FalseNegativeRatesTest(unittest.TestCase):
    def test_rejects_split_without_positives(self):
        with self.assertRaisesRegex(
            ValueError,
            "testing must contain at least one positive sample to compute ROC",
        ):
            false_negative_rates_at_cutoffs([], np.asarray([0.5]), "testing")

    def test_uses_strict_operating_cutoff(self):
        rates = false_negative_rates_at_cutoffs(
            [0.5, 0.75], np.asarray([0.5, 0.7, 0.8]), "testing"
        )
        np.testing.assert_allclose(rates, [0.5, 0.5, 1.0])


if __name__ == "__main__":
    unittest.main()
