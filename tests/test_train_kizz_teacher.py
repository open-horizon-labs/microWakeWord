import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

from microwakeword.kizz_teacher import NegativeSource
from tools.train_kizz_teacher import (
    declared_mixture_summary,
    evaluate_validation_checkpoint,
    positive_source_balance_report,
    realized_mixture_ledger,
    resolve_topology,
    scheduled_sequence_weight,
    training_positive_source_families,
    validate_realized_positive_sampling,
)


class _MarkerModel:
    def __call__(self, features, training=False):
        del training
        features = np.asarray(features)
        logits = np.full((len(features), 66, 9), -5.0, dtype=np.float32)
        for row, feature in enumerate(features):
            if feature[0, 0] > 0.5:
                logits[row, :, 0:2] = 0.0
                for state in range(7):
                    logits[row, state * 2, 2 + state] = 5.0
            else:
                logits[row, :, 0] = 5.0
        return tf.convert_to_tensor(logits)


class TrainKizzTeacherTest(unittest.TestCase):
    def test_positive_source_balance_counts_parents_not_overlays(self):
        provenance = {
            "examples": [
                {
                    "split": "train",
                    "source_id": f"a::{variant}",
                    "parent_source_id": "a",
                    "provider": "kokoro",
                }
                for variant in ("clean", "overlay-0", "overlay-1")
            ]
            + [
                {
                    "split": "train",
                    "source_id": "b::clean",
                    "parent_source_id": "b",
                    "provider": "deepgram",
                }
            ]
        }
        report = positive_source_balance_report(
            provenance,
            {
                "splits": {
                    "train": {
                        "minimum_families": 2,
                        "maximum_family_share": 0.50,
                    }
                }
            },
        )
        self.assertTrue(report["qualified"])
        self.assertEqual(report["splits"]["train"]["unique_parent_count"], 2)

    def test_positive_source_balance_rejects_dominant_generator(self):
        provenance = {
            "examples": [
                {
                    "split": "test",
                    "source_id": str(index),
                    "parent_source_id": str(index),
                    "provider": "kokoro" if index < 9 else "device",
                }
                for index in range(10)
            ]
        }
        report = positive_source_balance_report(
            provenance,
            {
                "splits": {
                    "test": {
                        "minimum_families": 2,
                        "maximum_family_share": 0.60,
                    }
                }
            },
        )
        self.assertFalse(report["qualified"])
        self.assertIn(
            "source_family_overrepresented",
            {item["reason"] for item in report["violations"]},
        )

    def test_materialized_positive_families_preserve_feature_order(self):
        provenance = {
            "examples": [
                {"split": "validation", "provider": "unused"},
                {"split": "train", "provider": "deepgram"},
                {"split": "train", "provider": "elevenlabs"},
            ]
        }
        self.assertEqual(
            training_positive_source_families(provenance),
            ["deepgram", "elevenlabs"],
        )

    def test_realized_positive_sampling_fails_provider_tokenism(self):
        report = validate_realized_positive_sampling(
            {"assemblyai": 700, "deepgram": 100, "elevenlabs": 100, "kokoro": 100},
            {
                "mode": "uniform_family",
                "minimum_families": 4,
                "minimum_family_share": 0.24,
                "maximum_family_share": 0.26,
            },
        )
        self.assertFalse(report["qualified"])
        self.assertIn(
            "realized_positive_family_overrepresented",
            {item["reason"] for item in report["violations"]},
        )

    def test_realized_positive_sampling_accepts_uniform_mix(self):
        report = validate_realized_positive_sampling(
            {"assemblyai": 25, "deepgram": 25, "elevenlabs": 25, "kokoro": 25},
            {
                "mode": "uniform_family",
                "minimum_families": 4,
                "minimum_family_share": 0.24,
                "maximum_family_share": 0.26,
            },
        )
        self.assertTrue(report["qualified"])

    def test_sequence_schedule_supports_frame_warmup_and_every_step_training(self):
        self.assertEqual(
            scheduled_sequence_weight(98, weight=0.5, every=1, start_step=100), 0.0
        )
        self.assertEqual(
            scheduled_sequence_weight(99, weight=0.5, every=1, start_step=100), 0.5
        )
        self.assertEqual(
            scheduled_sequence_weight(100, weight=0.5, every=2, start_step=100), 0.0
        )

    def test_declared_and_realized_mixture_use_exact_source_groups(self):
        probabilities = {"speech": 0.6, "noise": 0.4}
        groups = {"speech": "far_field_speech", "noise": "background_noise"}
        declared = declared_mixture_summary(probabilities, groups)
        self.assertEqual(
            declared["groups"]["canonical_positive"]["sampling_share"], 0.5
        )
        self.assertEqual(declared["groups"]["far_field_speech"]["sampling_share"], 0.3)
        sequence = type(
            "Sequence",
            (),
            {
                "positive_sample_count": 10,
                "negative_source_sample_counts": {"speech": 6, "noise": 4},
            },
        )()
        guard = {"test": "bound"}
        ledger = realized_mixture_ledger(sequence, guard, groups)
        self.assertEqual(ledger["total_samples"], 20)
        self.assertEqual(ledger["realized_groups"]["far_field_speech"]["share"], 0.3)

    def test_single_topology_is_explicit_and_nine_state(self):
        args = type("Args", (), {"topology": "single", "states_per_phone": None})()
        self.assertEqual(resolve_topology(args).state_count, 9)

    def test_double_topology_is_explicit_and_sixteen_state(self):
        args = type("Args", (), {"topology": "double", "states_per_phone": None})()
        self.assertEqual(resolve_topology(args).state_count, 16)

    def test_validation_selection_is_deterministic_and_zero_fp_first(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            positive_features = np.ones((2, 260, 40), dtype=np.float32)
            negative_features = np.zeros((8, 260, 40), dtype=np.float32)
            targets = np.full((2, 66), 1, dtype=np.int32)
            np.save(root / "positive-features.npy", positive_features)
            np.save(root / "positive-targets.npy", targets)
            np.save(root / "negative.npy", negative_features)
            kwargs = {
                "positive_features": root / "positive-features.npy",
                "positive_targets": root / "positive-targets.npy",
                "negative_sources": [
                    NegativeSource("validation", root / "negative.npy")
                ],
                "topology": resolve_topology(
                    type("Args", (), {"topology": "single", "states_per_phone": None})()
                ),
                "negative_limit": 8,
                "batch_size": 2,
                "seed": 24103,
            }
            first = evaluate_validation_checkpoint(_MarkerModel(), **kwargs)
            second = evaluate_validation_checkpoint(_MarkerModel(), **kwargs)
            self.assertEqual(first, second)
            self.assertTrue(first["selected"]["zero_false_accepts"])
            self.assertEqual(first["selected"]["false_accepts"], 0)
            self.assertEqual(first["selected"]["opportunity_recall"], 1.0)
            self.assertTrue(first["ledger"])


if __name__ == "__main__":
    unittest.main()
