import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import yaml

from microwakeword.ordered_state import OrderedStateTopology
from microwakeword.wake_phrase import KIZZ_CONTROL
from tools.cache_kizz_teacher_logits import (
    main,
    sha256_artifact,
    sha256_file,
    sha256_json,
    validate_teacher_training,
)


class _FakeTeacher:
    def __init__(self, state_count, output_frames):
        self.state_count = state_count
        self.output_frames = output_frames
        self.loaded = None

    def load_weights(self, path):
        self.loaded = Path(path)

    def predict(self, features, verbose=0):
        del verbose
        return np.zeros(
            (len(features), self.output_frames, self.state_count), dtype=np.float32
        )


class CacheKizzTeacherLogitsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.topology = OrderedStateTopology(KIZZ_CONTROL.phones, 1)
        self.features = self.root / "positive_features.npy"
        self.targets = self.root / "positive_targets.npy"
        np.save(self.features, np.zeros((4, 260, 40), dtype=np.float32))
        np.save(self.targets, np.zeros((4, 87), dtype=np.int32))

        self.source_manifest = self.root / "source-manifest.json"
        self.source_manifest.write_text('{"source": "locked"}\n', encoding="utf-8")
        self.provenance = self.root / "feature-provenance.json"
        provenance = {
            "schema_version": 3,
            "recipe": "kizz_aligned_teacher_features_v3",
            "state_count": self.topology.state_count,
            "states_per_phone": 1,
            "include_inherited_alignments": False,
            "positive_counts": {"train": 4},
            "overlay_snr_db": [],
            "positive_manifests": [
                {
                    "path": str(self.source_manifest),
                    "sha256": sha256_file(self.source_manifest),
                }
            ],
            "examples": [
                {
                    "split": "train",
                    "source_id": f"{provider}::clean",
                    "parent_source_id": provider,
                    "provider": provider,
                    "variant": "clean",
                    "augmentation": None,
                }
                for provider in (
                    "assemblyai",
                    "deepgram",
                    "elevenlabs",
                    "kokoro",
                )
            ],
        }
        self.provenance.write_text(json.dumps(provenance), encoding="utf-8")

        self.best = self.root / "best.weights.h5"
        self.checkpoint = self.root / "checkpoint-000200.weights.h5"
        self.best.write_bytes(b"selected detector weights")
        self.checkpoint.write_bytes(self.best.read_bytes())
        self.training = self.root / "teacher-training.json"
        self.write_training()

        self.negative_groups = {
            "collision": "phonetic_collision",
            "speech": "public_speech",
            "music": "music",
            "noise": "background_noise",
        }
        self.negatives = {}
        for index, source_id in enumerate(self.negative_groups):
            path = self.root / f"negative-{source_id}.npy"
            np.save(path, np.full((2, 260, 40), index, dtype=np.float16))
            self.negatives[source_id] = path

        self.recipe = self.root / "batch-mixture.yaml"
        groups = {
            "canonical_positive": {
                "sample_share": 0.5,
                "weighted_pressure_share": 0.5,
            }
        }
        groups.update(
            {
                group: {
                    "sample_share": 0.125,
                    "weighted_pressure_share": 0.125,
                }
                for group in self.negative_groups.values()
            }
        )
        self.recipe.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "deployment_qualification": False,
                    "positive_sampling_guard": {
                        "mode": "uniform_family",
                        "minimum_families": 4,
                        "minimum_family_share": 0.24,
                        "maximum_family_share": 0.26,
                    },
                    "mixture_guard": {
                        "schema_version": 1,
                        "require_all_active_groups": True,
                        "minimum_realized_samples": 8,
                        "tolerances": {
                            "sample_share": 1.0,
                            "weighted_pressure_share": 1.0,
                        },
                        "expected": {
                            "classes": {
                                "positive": {
                                    "sample_share": 0.5,
                                    "weighted_pressure_share": 0.5,
                                },
                                "negative": {
                                    "sample_share": 0.5,
                                    "weighted_pressure_share": 0.5,
                                },
                            },
                            "groups": groups,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def training_payload(self):
        losing = {
            "threshold": 0.7,
            "false_accepts": 0,
            "opportunity_recall": 0.95,
            "separation": 0.1,
            "validation_loss": 0.4,
            "zero_false_accepts": True,
        }
        winning = {
            "threshold": 0.5,
            "false_accepts": 0,
            "opportunity_recall": 0.98,
            "separation": 0.2,
            "validation_loss": 0.3,
            "zero_false_accepts": True,
        }
        return {
            "schema_version": 1,
            "model": "kizz_offline_teacher",
            "input_shape": [260, 40],
            "output_shape": [87, self.topology.state_count],
            "topology": {
                "phones": list(KIZZ_CONTROL.phones),
                "states_per_phone": 1,
            },
            "hidden_size": 32,
            "recurrent_layers": 2,
            "selection_min_recall": 0.95,
            "checkpoint_selection": (
                "validation_min_false_accepts_subject_to_recall_floor"
            ),
            "checkpoint_selection_ledger": [
                {"step": 100, "selected": losing},
                {"step": 200, "selected": winning},
            ],
            "best_validation": winning,
            "feature_provenance": str(self.provenance),
            "feature_provenance_sha256": sha256_file(self.provenance),
            "positive_features": str(self.features),
            "positive_features_sha256": sha256_file(self.features),
            "positive_targets": str(self.targets),
            "positive_targets_sha256": sha256_file(self.targets),
        }

    def write_training(self, mutate=None):
        payload = self.training_payload()
        if mutate is not None:
            mutate(payload)
        self.training.write_text(json.dumps(payload), encoding="utf-8")

    def validate_training(self):
        return validate_teacher_training(
            self.training,
            teacher_weights=self.best,
            feature_provenance=self.provenance,
            positive_features=self.features,
            positive_targets=self.targets,
            topology=self.topology,
            phrase_id=KIZZ_CONTROL.phrase_id,
        )

    def argv(self):
        arguments = [
            "--teacher-weights",
            str(self.best),
            "--teacher-training",
            str(self.training),
            "--positive-features",
            str(self.features),
            "--positive-targets",
            str(self.targets),
            "--feature-provenance",
            str(self.provenance),
            "--batch-mixture-recipe",
            str(self.recipe),
            "--output",
            str(self.root / "cache"),
            "--steps",
            "20",
            "--batch-size",
            "8",
        ]
        for source_id, path in self.negatives.items():
            arguments.extend(("--negative-source", f"{source_id}={path}"))
            arguments.extend(("--negative-source-probability", f"{source_id}=1"))
            arguments.extend(
                (
                    "--negative-source-group",
                    f"{source_id}={self.negative_groups[source_id]}",
                )
            )
        return arguments

    def test_derives_and_binds_detector_winner_without_faph_qualification(self):
        result = self.validate_training()
        selected = result["selected_teacher"]
        self.assertEqual(selected["step"], 200)
        self.assertEqual(
            selected["best_weights"]["sha256"],
            selected["checkpoint_weights"]["sha256"],
        )
        self.assertEqual(selected["selection_min_recall"], 0.95)

    def test_rejects_best_weights_that_differ_from_derived_checkpoint(self):
        self.best.write_bytes(b"wrong weights")
        with self.assertRaisesRegex(ValueError, "not byte-identical"):
            self.validate_training()

    def test_rejects_report_best_validation_that_is_not_ledger_winner(self):
        self.write_training(
            lambda payload: payload.__setitem__(
                "best_validation",
                payload["checkpoint_selection_ledger"][0]["selected"],
            )
        )
        with self.assertRaisesRegex(ValueError, "derived ledger winner"):
            self.validate_training()

    def test_rejects_detector_recall_floor_below_95_percent(self):
        self.write_training(
            lambda payload: payload.__setitem__("selection_min_recall", 0.94)
        )
        with self.assertRaisesRegex(ValueError, "at least 0.95"):
            self.validate_training()

    def test_rejects_feature_provenance_hash_mismatch(self):
        self.write_training(
            lambda payload: payload.__setitem__(
                "feature_provenance_sha256", "0" * 64
            )
        )
        with self.assertRaisesRegex(ValueError, "feature provenance SHA-256"):
            self.validate_training()

    def test_main_emits_exact_bindings_and_provider_balanced_ledger(self):
        fake = _FakeTeacher(self.topology.state_count, 87)
        with mock.patch(
            "tools.cache_kizz_teacher_logits.build_teacher", return_value=fake
        ) as build:
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(self.argv()), 0)
        build.assert_called_once_with(
            hidden_size=32,
            recurrent_layers=2,
            output_frames=87,
            topology=self.topology,
        )
        metadata = json.loads((self.root / "cache.json").read_text())
        self.assertFalse(metadata["deployment_qualification"])
        self.assertEqual(metadata["topology"]["phrase_id"], "kizz-control")
        self.assertEqual(metadata["topology"]["states_per_phone"], 1)
        self.assertEqual(metadata["topology"]["state_count"], 12)
        self.assertEqual(
            metadata["realized_sampling_ledger"]["positive_source_samples"],
            {
                "assemblyai": 20,
                "deepgram": 20,
                "elevenlabs": 20,
                "kokoro": 20,
            },
        )
        self.assertEqual(
            metadata["realized_sampling_ledger_sha256"],
            sha256_json(metadata["realized_sampling_ledger"]),
        )
        for binding in metadata["negative_sources"]:
            digest, mode = sha256_artifact(Path(binding["path"]))
            self.assertEqual((binding["sha256"], binding["sha256_mode"]), (digest, mode))
        for binding in metadata["outputs"].values():
            self.assertEqual(binding["sha256"], sha256_file(Path(binding["path"])))

    def test_main_rejects_missing_explicit_negative_contract(self):
        arguments = self.argv()
        index = arguments.index("--negative-source-probability")
        del arguments[index : index + 2]
        with self.assertRaises(SystemExit):
            main(arguments)
        self.assertFalse((self.root / "cache.json").exists())

    def test_main_rejects_noncanonical_feature_provenance(self):
        payload = json.loads(self.provenance.read_text())
        payload["recipe"] = "legacy_features"
        self.provenance.write_text(json.dumps(payload), encoding="utf-8")
        self.write_training()
        with self.assertRaises(SystemExit):
            main(self.argv())
        self.assertFalse((self.root / "cache.json").exists())

    def test_main_rejects_declared_negative_mixture_drift(self):
        payload = yaml.safe_load(self.recipe.read_text())
        payload["mixture_guard"]["tolerances"] = {
            "sample_share": 0.1,
            "weighted_pressure_share": 0.1,
        }
        self.recipe.write_text(yaml.safe_dump(payload), encoding="utf-8")
        arguments = self.argv()
        probability = arguments.index("noise=1")
        arguments[probability] = "noise=4"
        with self.assertRaises(SystemExit):
            main(arguments)
        self.assertFalse((self.root / "cache.json").exists())

    def test_main_rejects_realized_provider_share_drift(self):
        payload = yaml.safe_load(self.recipe.read_text())
        payload["positive_sampling_guard"]["maximum_family_share"] = 0.24
        self.recipe.write_text(yaml.safe_dump(payload), encoding="utf-8")
        fake = _FakeTeacher(self.topology.state_count, 87)
        with mock.patch(
            "tools.cache_kizz_teacher_logits.build_teacher", return_value=fake
        ):
            with self.assertRaisesRegex(
                ValueError, "realized positive provider sampling"
            ):
                with redirect_stdout(io.StringIO()):
                    main(self.argv())
        self.assertFalse((self.root / "cache.json").exists())


if __name__ == "__main__":
    unittest.main()
