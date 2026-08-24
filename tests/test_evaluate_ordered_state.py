import json
import math
import tempfile
import unittest
from pathlib import Path

from tools.evaluate_ordered_state import (
    apply_decoder_contract,
    evaluate_sources,
    load_sources,
    poisson_upper_bound_95,
)


class FakeDecoder:
    resets = 0
    rearm_calls = 0

    def __init__(self):
        self.ready = True

    def reset(self):
        type(self).resets += 1
        self.ready = True

    def update(self, score, timestamp):
        if self.ready and score >= 0.8:
            self.ready = False
            return {"score": score, "timestamp": timestamp}
        return False

    def rearm(self):
        type(self).rearm_calls += 1
        self.ready = True


class VectorDecoder:
    def reset(self):
        pass

    def step(self, frame, frame_index):
        if frame[0] > 0.5:
            return {
                "start_frame": frame_index - 1,
                "end_frame": frame_index,
                "score": 2.0,
            }
        return None


class EvaluateOrderedStateTest(unittest.TestCase):
    def setUp(self):
        FakeDecoder.resets = 0
        FakeDecoder.rearm_calls = 0

    def test_exposure_and_false_accept_math(self):
        report = evaluate_sources(
            [
                {
                    "id": "room",
                    "records": [
                        {"timestamp": 0, "score": 0.9},
                        {"timestamp": 3600, "score": 0.1},
                    ],
                }
            ],
            FakeDecoder,
        )
        self.assertEqual(report["exposure_seconds"], 3600)
        self.assertEqual(report["false_accepts"], 1)
        self.assertEqual(report["false_accepts_per_hour"], 1.0)

    def test_cooldown_suppresses_events_and_rearms(self):
        report = evaluate_sources(
            [
                {
                    "records": [
                        {"timestamp": 0, "score": 0.9},
                        {"timestamp": 1, "score": 0.9},
                        {"timestamp": 5, "score": 0.9},
                        {"timestamp": 6, "score": 0.1},
                    ],
                    "exposure_seconds": 6,
                }
            ],
            FakeDecoder,
            cooldown_seconds=3,
        )
        self.assertEqual(report["false_accepts"], 2)
        self.assertEqual(FakeDecoder.rearm_calls, 1)

    def test_source_boundaries_reset_streaming_state(self):
        report = evaluate_sources(
            [
                {"id": "a", "records": [{"timestamp": 0, "score": 0.9}]},
                {"id": "b", "records": [{"timestamp": 0, "score": 0.9}]},
            ],
            FakeDecoder,
        )
        self.assertEqual(FakeDecoder.resets, 2)
        self.assertEqual(
            [source["event_count"] for source in report["sources"]], [1, 1]
        )

    def test_positive_recall_and_coordinates(self):
        report = evaluate_sources(
            [
                {
                    "id": "hit",
                    "session_id": "s1",
                    "label": "positive",
                    "records": [{"timestamp": 2, "logit": math.log(4)}],
                },
                {
                    "id": "miss",
                    "session_id": "s2",
                    "label": "positive",
                    "records": [
                        {"timestamp": 0, "score": 0.1},
                        {"timestamp": 1, "score": 0.2},
                    ],
                },
            ],
            FakeDecoder,
        )
        self.assertEqual(report["positive_recall"], 0.5)
        self.assertEqual(report["false_rejection_rate"], 0.5)
        self.assertEqual(report["sources"][0]["positive_recall"], 1.0)
        self.assertEqual(report["sources"][1]["false_rejection_rate"], 1.0)
        event = report["sources"][0]["events"][0]
        self.assertEqual(event["timestamp"], 2)
        self.assertEqual(event["index"], 0)
        self.assertAlmostEqual(event["score"], 0.8)

    def test_zero_event_poisson_bound(self):
        self.assertAlmostEqual(poisson_upper_bound_95(0, 3600), -math.log(0.05))

    def test_qualification_can_require_declared_exposure(self):
        with self.assertRaisesRegex(ValueError, "needs exposure_seconds"):
            evaluate_sources(
                [{"records": [{"timestamp": 0, "score": 0.1}]}],
                FakeDecoder,
                require_declared_exposure=True,
            )

    def test_frame_step_includes_final_frame_and_validates_cadence(self):
        report = evaluate_sources(
            [
                {
                    "records": [
                        {"timestamp": 0.00, "score": 0.1},
                        {"timestamp": 0.03, "score": 0.1},
                    ]
                }
            ],
            FakeDecoder,
            frame_step_seconds=0.03,
        )
        self.assertAlmostEqual(report["exposure_seconds"], 0.06)
        with self.assertRaisesRegex(ValueError, "do not match"):
            evaluate_sources(
                [
                    {
                        "records": [
                            {"timestamp": 0.00, "score": 0.1},
                            {"timestamp": 0.04, "score": 0.1},
                        ]
                    }
                ],
                FakeDecoder,
                frame_step_seconds=0.03,
            )

    def test_rejects_double_cooldown_configuration(self):
        class InternalCooldownDecoder(FakeDecoder):
            cooldown_frames = 10

        with self.assertRaisesRegex(ValueError, "not both"):
            evaluate_sources(
                [{"records": [{"timestamp": 0.0, "score": 0.1}]}],
                InternalCooldownDecoder,
                cooldown_seconds=1.0,
            )

    def test_rejects_non_finite_or_empty_declared_exposure(self):
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            evaluate_sources(
                [
                    {
                        "exposure_seconds": float("nan"),
                        "records": [{"timestamp": 0.0, "score": 0.1}],
                    }
                ],
                FakeDecoder,
            )
        with self.assertRaisesRegex(ValueError, "no score records"):
            evaluate_sources(
                [{"exposure_seconds": 3600, "records": []}],
                FakeDecoder,
            )

    def test_artifact_decoder_contract_prevents_setting_drift(self):
        contract = {
            "schema_version": 1,
            "state_count": 23,
            "frame_step_seconds": 0.03,
            "decoder_args": {
                "from_logits": True,
                "self_loop_probability": 0.6,
            },
        }
        merged, frame_step = apply_decoder_contract(
            {"completion_margin": 12.0}, contract
        )
        self.assertTrue(merged["from_logits"])
        self.assertEqual(merged["completion_margin"], 12.0)
        self.assertEqual(frame_step, 0.03)
        with self.assertRaisesRegex(ValueError, "conflicts"):
            apply_decoder_contract({"self_loop_probability": 0.5}, contract)

    def test_nonzero_event_poisson_bound(self):
        bound = poisson_upper_bound_95(1, 3600)
        self.assertAlmostEqual(bound, 4.743864518390577, places=8)

    def test_reports_recall_by_session(self):
        report = evaluate_sources(
            [
                {
                    "session_id": "s",
                    "label": "positive",
                    "records": [{"timestamp": 0, "score": 0.9}],
                },
                {
                    "session_id": "s",
                    "label": "positive",
                    "records": [{"timestamp": 0, "score": 0.1}],
                },
            ],
            FakeDecoder,
        )
        self.assertEqual(report["sessions"][0]["positive_recall"], 0.5)

    def test_vector_decoder_coordinates_and_logit_records(self):
        report = evaluate_sources(
            [
                {
                    "id": "vector",
                    "records": [
                        {"timestamp": 10, "logit": [0.1, 0.2]},
                        {"timestamp": 11, "score": [0.9, 0.1]},
                    ],
                }
            ],
            VectorDecoder,
        )
        event = report["sources"][0]["events"][0]
        self.assertEqual(event["start_timestamp"], 10)
        self.assertEqual(event["end_timestamp"], 11)
        self.assertEqual(report["score_distributions"]["all"]["count"], 4)

    def test_loads_json_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.jsonl"
            path.write_text(json.dumps({"timestamp": 0, "score": 0.1}) + "\n")
            sources = load_sources(path)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["records"][0]["score"], 0.1)


if __name__ == "__main__":
    unittest.main()
