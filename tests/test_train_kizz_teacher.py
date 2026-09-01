import tempfile
import unittest
import json
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
    validate_feature_provenance,
    validate_realized_positive_sampling,
    validation_selection_key,
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
    def _write_feature_provenance_fixture(
        self, root: Path, *, overlay_count: int = 12
    ) -> tuple[Path, Path, Path, object]:
        topology = resolve_topology(
            type(
                "Args",
                (),
                {
                    "topology": "single",
                    "states_per_phone": None,
                    "phrase_id": "kizz-control",
                },
            )()
        )
        variants = ["clean"] + [
            f"overlay-{index}" for index in range(overlay_count)
        ]
        examples = [
            {
                "split": "train",
                "parent_source_id": "parent-a",
                "variant": variant,
                "augmentation": None if variant == "clean" else {"seed": index},
            }
            for index, variant in enumerate(variants)
        ]
        examples.extend(
            {
                "split": split,
                "parent_source_id": f"parent-{split}",
                "variant": "clean",
                "augmentation": None,
            }
            for split in ("validation", "test")
        )
        report = {
            "schema_version": 3,
            "recipe": "kizz_aligned_teacher_features_v3",
            "state_count": topology.state_count,
            "states_per_phone": topology.states_per_phone,
            "include_inherited_alignments": False,
            "positive_counts": {"train": len(variants)},
            "overlay_snr_db": [10.0] * overlay_count,
            "examples": examples,
        }
        provenance = root / "feature-provenance.json"
        features = root / "positive-features.npy"
        targets = root / "positive-targets.npy"
        provenance.write_text(json.dumps(report), encoding="utf-8")
        np.save(features, np.zeros((len(variants), 2, 2), dtype=np.float32))
        np.save(targets, np.zeros((len(variants), 2), dtype=np.int32))
        return provenance, features, targets, topology

    def test_feature_provenance_accepts_declared_twelve_overlay_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            provenance, features, targets, topology = (
                self._write_feature_provenance_fixture(Path(directory))
            )
            report = validate_feature_provenance(
                provenance, features, targets, topology
            )
            self.assertEqual(len(report["overlay_snr_db"]), 12)

    def test_feature_provenance_rejects_missing_parent_overlay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provenance, features, targets, topology = (
                self._write_feature_provenance_fixture(root)
            )
            report = json.loads(provenance.read_text(encoding="utf-8"))
            report["examples"] = [
                item
                for item in report["examples"]
                if not (
                    item["split"] == "train" and item["variant"] == "overlay-11"
                )
            ]
            provenance.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "variants are incomplete"):
                validate_feature_provenance(provenance, features, targets, topology)

    def test_feature_provenance_rejects_augmented_evaluation_example(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provenance, features, targets, topology = (
                self._write_feature_provenance_fixture(root)
            )
            report = json.loads(provenance.read_text(encoding="utf-8"))
            evaluation = next(
                item for item in report["examples"] if item["split"] == "validation"
            )
            evaluation["variant"] = "overlay-0"
            evaluation["augmentation"] = {"seed": 1}
            provenance.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "evaluation examples must be clean"):
                validate_feature_provenance(provenance, features, targets, topology)

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

    def test_detector_selection_minimizes_false_accepts_at_recall_floor(self):
        points = [
            {
                "zero_false_accepts": True,
                "opportunity_recall": 0.55,
                "false_accepts": 0,
                "separation": -1.0,
                "validation_loss": 0.1,
                "threshold": 2.0,
            },
            {
                "zero_false_accepts": False,
                "opportunity_recall": 0.97,
                "false_accepts": 25,
                "separation": -2.0,
                "validation_loss": 0.2,
                "threshold": 1.0,
            },
            {
                "zero_false_accepts": False,
                "opportunity_recall": 1.0,
                "false_accepts": 80,
                "separation": -3.0,
                "validation_loss": 0.3,
                "threshold": 0.0,
            },
        ]
        selected = max(
            points, key=lambda item: validation_selection_key(item, 0.95)
        )
        self.assertEqual(selected["opportunity_recall"], 0.97)
        self.assertEqual(selected["false_accepts"], 25)


if __name__ == "__main__":
    unittest.main()
