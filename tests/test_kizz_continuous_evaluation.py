import math
import unittest

from microwakeword.kizz_continuous_evaluation import (
    PositiveOpportunity,
    ScoreStream,
    ThresholdProvenance,
    detect_events,
    poisson_upper_95,
    qualify_test_streams,
    select_threshold,
)


class KizzContinuousEvaluationTest(unittest.TestCase):
    def test_separated_events_in_one_file_are_both_counted(self):
        events = detect_events(
            [0.0, 0.2, 1.0, 1.1, 2.4, 2.6],
            [0.1, 0.9, 0.1, 0.95, 0.1, 0.99],
            0.8,
            refractory_seconds=1.0,
        )
        self.assertEqual(len(events), 2)
        self.assertEqual([event.start_seconds for event in events], [0.2, 2.6])

    def test_continuous_high_run_is_one_event_and_duration_is_capped(self):
        events = detect_events(
            [0.0, 0.5, 1.0, 1.5],
            [0.9, 0.9, 0.9, 0.1],
            0.8,
            refractory_seconds=0.5,
            max_event_duration_seconds=0.75,
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].end_seconds, 0.75)

    def test_zero_false_accepts_over_too_little_exposure_fails_gate(self):
        validation = [
            ScoreStream("vp", "validation", "positive", (0.0, 1.0), (0.1, 0.9), 2.0),
            ScoreStream("vn", "validation", "negative", (0.0, 1.0), (0.1, 0.1), 3600.0),
        ]
        provenance = select_threshold(validation)
        result = qualify_test_streams(
            [
                ScoreStream("tp", "test", "positive", (0.0, 1.0), (0.1, 0.9), 2.0),
                ScoreStream("tn", "test", "negative", (0.0, 1.0), (0.1, 0.1), 3599.0),
            ],
            provenance,
        )
        self.assertFalse(result.qualified)
        self.assertIn("negative exposure", " ".join(result.reasons))
        self.assertAlmostEqual(
            result.false_accepts_per_hour_upper_95,
            -math.log(0.05) * 3600.0 / 3599.0,
            places=6,
        )

    def test_negative_events_are_counted_per_event_not_per_file(self):
        validation = [
            ScoreStream("vp", "validation", "positive", (0.0, 1.0), (0.1, 0.9), 2.0),
            ScoreStream("vn", "validation", "negative", (0.0, 1.0), (0.1, 0.1), 3600.0),
        ]
        provenance = select_threshold(validation)
        result = qualify_test_streams(
            [
                ScoreStream("tp", "test", "positive", (0.0, 1.0), (0.1, 0.9), 2.0),
                ScoreStream(
                    "tn",
                    "test",
                    "negative",
                    (0.0, 1.0, 2.0, 3.0),
                    (0.9, 0.1, 0.9, 0.1),
                    3600.0,
                ),
            ],
            provenance,
            refractory_seconds=0.5,
        )
        self.assertEqual(result.false_accepts, 2)

    def test_test_split_cannot_fit_threshold(self):
        with self.assertRaises(ValueError):
            select_threshold(
                [
                    ScoreStream("tp", "test", "positive", (0.0,), (0.9,), 1.0),
                    ScoreStream("tn", "test", "negative", (0.0,), (0.1,), 1.0),
                ]
            )

    def test_bad_threshold_provenance_is_rejected(self):
        with self.assertRaises(ValueError):
            qualify_test_streams(
                [
                    ScoreStream("tn", "test", "negative", (0.0,), (0.1,), 3600.0),
                    ScoreStream("tp", "test", "positive", (0.0,), (0.9,), 1.0),
                ],
                ThresholdProvenance("test", (), (), 0.8),
            )

    def test_threshold_provenance_must_name_validation_streams(self):
        with self.assertRaisesRegex(ValueError, "requires validation stream IDs"):
            qualify_test_streams(
                [
                    ScoreStream("tn", "test", "negative", (0.0,), (0.1,), 3600.0),
                    ScoreStream("tp", "test", "positive", (0.0,), (0.9,), 1.0),
                ],
                ThresholdProvenance("validation", (), (), 0.8),
            )

    def test_validation_and_test_stream_ids_must_be_disjoint(self):
        with self.assertRaisesRegex(ValueError, "overlap test streams"):
            qualify_test_streams(
                [
                    ScoreStream("tn", "test", "negative", (0.0,), (0.1,), 3600.0),
                    ScoreStream("tp", "test", "positive", (0.0,), (0.9,), 1.0),
                ],
                ThresholdProvenance("validation", ("tp",), ("vn",), 0.8),
            )

    def test_duplicate_stream_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "stream IDs must be unique"):
            select_threshold(
                [
                    ScoreStream(
                        "duplicate", "validation", "positive", (0.0,), (0.9,), 1.0
                    ),
                    ScoreStream(
                        "duplicate", "validation", "negative", (0.0,), (0.1,), 1.0
                    ),
                ]
            )

    def test_non_finite_scores_and_out_of_range_timestamps_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be finite"):
            ScoreStream("bad", "validation", "negative", (0.0,), (math.nan,), 1.0)
        with self.assertRaisesRegex(ValueError, "outside stream duration"):
            ScoreStream("bad", "validation", "negative", (2.0,), (0.1,), 1.0)

    def test_locked_anchor_accept_fails_even_with_long_exposure(self):
        provenance = ThresholdProvenance("validation", ("vp",), ("vn",), 0.8)
        result = qualify_test_streams(
            [
                ScoreStream("tp", "test", "positive", (0.0,), (0.9,), 1.0),
                ScoreStream("tn", "test", "negative", (0.0,), (0.1,), 100 * 3600.0),
                ScoreStream(
                    "anchor",
                    "test",
                    "anchor",
                    (0.0,),
                    (0.9,),
                    1.0,
                    locked_deployment_anchor=True,
                ),
            ],
            provenance,
        )
        self.assertFalse(result.qualified)
        self.assertEqual(result.locked_anchor_false_accepts, 1)

    def test_poisson_zero_event_bound_is_exact(self):
        self.assertAlmostEqual(
            poisson_upper_95(0, 100.0), -math.log(0.05) / 100.0, places=12
        )

    def test_poisson_bound_remains_finite_for_many_false_accepts(self):
        upper = poisson_upper_95(500, 100.0)
        self.assertTrue(math.isfinite(upper))
        self.assertGreater(upper, 5.0)


if __name__ == "__main__":
    unittest.main()
