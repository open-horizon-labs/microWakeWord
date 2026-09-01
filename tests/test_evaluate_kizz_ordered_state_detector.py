import unittest

import numpy as np

from tools.evaluate_kizz_ordered_state_detector import (
    _metrics,
    _qualification_reasons,
    _select_threshold,
)


class EvaluateKizzOrderedStateDetectorTests(unittest.TestCase):
    def test_validation_selection_minimizes_candidates_at_recall_floor(self):
        positives = np.asarray([0.9, 0.8, 0.7, 0.6])
        negatives = np.asarray([0.75, 0.65, 0.1, 0.0])
        selected = _select_threshold(positives, negatives, 0.75)
        self.assertEqual(selected["threshold"], 0.7)
        self.assertEqual(selected["opportunity_recall"], 0.75)
        self.assertEqual(selected["false_candidates"], 1)

    def test_frozen_threshold_metrics_report_recall_and_candidate_pressure(self):
        metrics = _metrics(
            np.asarray([0.9, 0.8, 0.1]),
            np.asarray([0.7, 0.2, 0.0]),
            0.5,
        )
        self.assertEqual(metrics["opportunity_recall"], 2 / 3)
        self.assertEqual(metrics["false_candidate_fraction"], 1 / 3)
        self.assertEqual(metrics["window_set_precision"], 2 / 3)

    def test_candidate_pressure_blocks_conversion_even_when_recall_passes(self):
        validation = {
            "opportunity_recall": 0.97,
            "false_candidate_fraction": 0.25,
        }
        test = {
            "opportunity_recall": 0.95,
            "false_candidate_fraction": 0.10,
        }
        self.assertEqual(
            _qualification_reasons(
                validation,
                test,
                minimum_recall=0.95,
                maximum_false_candidate_fraction=0.20,
            ),
            ["detector_false_candidate_fraction_above_limit"],
        )


if __name__ == "__main__":
    unittest.main()
