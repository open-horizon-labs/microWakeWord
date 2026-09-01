import unittest

from tools.score_kizz_candidate_verifier_device_validation import (
    select_full_recall_threshold,
)


class ScoreKizzCandidateVerifierDeviceValidationTests(unittest.TestCase):
    def test_full_recall_threshold_is_lowest_validation_score(self):
        self.assertEqual(select_full_recall_threshold([2.5, -1.25, 0.0]), -1.25)

    def test_threshold_rejects_empty_or_nonfinite_scores(self):
        for values in ([], [float("nan")], [float("inf")]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    select_full_recall_threshold(values)


if __name__ == "__main__":
    unittest.main()
