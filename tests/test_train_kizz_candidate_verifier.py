import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.train_kizz_candidate_verifier import (
    BalancedCandidateBatcher,
    DEFAULT_MODEL_VARIANT,
    DEVICE_ROBUSTNESS_PROFILES,
    MODEL_JSON_PROVENANCE_KEY,
    MODEL_VARIANT_CHANNELS,
    TensorFlowVerifierBackend,
    TensorFlowVerifierModel,
    _freeze_probability_threshold,
    _verify_output_bindings,
    bind_model_json_provenance,
    dscnn_spec,
    estimate_dscnn_cost,
    evaluate_operating_point,
    load_verified_dataset,
    sha256_file,
    train_candidate_verifier,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


class CandidateDatasetFixture:
    def __init__(self, root: Path):
        self.root = root
        self.dataset = root / "dataset"
        self.dataset.mkdir()
        self.bound_files = {}
        for name in (
            "artifact",
            "config",
            "threshold",
            "source-manifest",
            "source-features",
            "detector-traces",
            "locked-holdout",
        ):
            path = root / f"{name}.bin"
            path.write_bytes(f"bound:{name}".encode())
            self.bound_files[name] = path

        definitions = [
            ("train-positive-a", "train", 1, "provider-a", 0.1, 0.2),
            ("train-positive-b", "train", 1, "provider-b", 0.2, 0.1),
            ("train-negative-noise", "train", 0, "noise", -0.2, -0.1),
            ("train-negative-collision", "train", 0, "collision", -0.1, -0.2),
            ("validation-positive-a", "validation", 1, "provider-a", 3.0, 2.0),
            ("validation-positive-b", "validation", 1, "provider-b", 3.0, 2.0),
            ("validation-negative-noise", "validation", 0, "noise", 2.5, 3.0),
            ("validation-negative-collision", "validation", 0, "collision", 2.2, 1.5),
            ("test-positive-a", "test", 1, "provider-a", 3.0, 0.0),
            ("test-positive-b", "test", 1, "provider-b", 2.8, 0.0),
            ("test-negative-noise", "test", 0, "noise", 2.9, 0.0),
            ("test-negative-collision", "test", 0, "collision", -2.0, 0.0),
        ]
        features = np.zeros((len(definitions), 260, 40), dtype=np.float16)
        labels = np.zeros(len(definitions), dtype=np.int8)
        detector_scores = np.linspace(0.55, 0.99, len(definitions), dtype=np.float32)
        rows = []
        for index, (name, split, label, family, first, second) in enumerate(
            definitions
        ):
            features[index, 0, 0] = first
            features[index, 0, 1] = second
            labels[index] = label
            row = {
                "candidate_id": f"{name}::candidate",
                "source_id": f"{name}::candidate",
                "parent_source_id": name,
                "speaker_id": f"speaker:{name}",
                "audio_sha256": _digest(f"audio:{name}"),
                "split": split,
                "label": label,
                "detector_conditioned": True,
                "feature_index": index,
                "detector_score": float(detector_scores[index]),
            }
            if label:
                row["provider"] = family
            else:
                row["source_group"] = family
            rows.append(row)
        for index, row in enumerate(rows):
            row["candidate_feature_sha256"] = hashlib.sha256(
                np.ascontiguousarray(features[index]).tobytes()
            ).hexdigest()
        self.rows = rows
        self.features = self.dataset / "features.npy"
        self.labels = self.dataset / "labels.npy"
        self.detector_scores = self.dataset / "detector_scores.npy"
        self.extra = self.dataset / "detector_score_frames.npy"
        np.save(self.features, features, allow_pickle=False)
        np.save(self.labels, labels, allow_pickle=False)
        np.save(self.detector_scores, detector_scores, allow_pickle=False)
        np.save(self.extra, np.arange(len(rows), dtype=np.int32), allow_pickle=False)
        self.corpus_path = self.dataset / "corpus.json"
        self.corpus = {
            "schema_version": 1,
            "recipe": "kizz_control_candidate_conditioned_verifier_v1",
            "candidate_condition": "frozen_detector_trigger_only",
            "detector": {
                "artifact": self._binding("artifact"),
                "config": self._binding("config"),
                "threshold": {**self._binding("threshold"), "value": 0.5},
            },
            "bindings": {
                "source_manifest": self._binding("source-manifest"),
                "source_features": self._binding("source-features"),
                "detector_traces": self._binding("detector-traces"),
                "locked_holdout": self._binding("locked-holdout"),
            },
            "hard_negative_selection": {
                "ranking": "detector_score_descending_then_candidate_id",
                "top_k": 1,
                "group_by": "source",
                "scope": "train_only",
                "raw_training_count": 2,
                "selected_training_count": 2,
                "heldout_candidates_unfiltered": 4,
            },
            "counts": {
                "selected_candidates": len(rows),
                "selected_positives": 6,
                "selected_negatives": 6,
                "by_split": {
                    split: {
                        "raw_negative_candidates": 2,
                        "selected_negative_candidates": 2,
                    }
                    for split in ("train", "validation", "test")
                },
            },
            "examples": rows,
            "array_sha256": {
                path.name: sha256_file(path)
                for path in (
                    self.features,
                    self.labels,
                    self.detector_scores,
                    self.extra,
                )
            },
        }
        self.write_corpus()

    def _binding(self, name: str):
        path = self.bound_files[name]
        return {"path": str(path.resolve()), "sha256": sha256_file(path)}

    def write_corpus(self) -> str:
        self.corpus_path.write_text(
            json.dumps(self.corpus, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        return sha256_file(self.corpus_path)


class FakeModel:
    def __init__(self, model_variant=DEFAULT_MODEL_VARIANT):
        self.step = 0
        self.model_variant = model_variant


class FakeBackend:
    def __init__(self):
        self.score_calls = []
        self.loaded_steps = []

    def build_model(
        self,
        *,
        learning_rate,
        seed,
        model_variant=DEFAULT_MODEL_VARIANT,
    ):
        self.learning_rate = learning_rate
        self.seed = seed
        self.model_variant = model_variant
        return FakeModel(model_variant)

    @staticmethod
    def train_batch(model, features, labels):
        if features.shape[0] != labels.shape[0]:
            raise AssertionError("fake backend received mismatched batch")
        model.step += 1
        return 1.0 / model.step

    def score(self, model, features, *, batch_size, purpose):
        del batch_size
        self.score_calls.append((purpose, model.step, len(features)))
        column = 0 if model.step == 1 else 1
        return np.asarray(features[:, 0, column, 0], dtype=np.float64)

    @staticmethod
    def save_weights(model, path):
        path.write_text(str(model.step), encoding="utf-8")

    def load_weights(self, model, path):
        model.step = int(path.read_text(encoding="utf-8"))
        self.loaded_steps.append(model.step)

    @staticmethod
    def count_params(model):
        return estimate_dscnn_cost(model_variant=model.model_variant)[
            "parameter_estimate"
        ]

    @staticmethod
    def model_json(model):
        return json.dumps(
            {
                "fake": "fixed_window_dscnn",
                "model_variant": model.model_variant,
                "dscnn_spec": dscnn_spec(model.model_variant),
            },
            sort_keys=True,
        )


class TrainCandidateVerifierTests(unittest.TestCase):
    def test_frozen_threshold_never_rounds_above_selected_boundary(self):
        selected = 0.3538808349553079
        self.assertEqual(_freeze_probability_threshold(selected, 0.0), selected)
        self.assertLess(_freeze_probability_threshold(selected, 0.3), selected)

    def test_loader_verifies_candidate_contract_all_hashes_and_split_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CandidateDatasetFixture(Path(directory))
            verified = load_verified_dataset(
                fixture.dataset, expected_corpus_sha256=sha256_file(fixture.corpus_path)
            )
            self.assertEqual(len(verified.rows), 12)
            self.assertIn("detector_score_frames.npy", verified.array_bindings)
            self.assertGreaterEqual(len(verified.transitive_bindings), 7)

    def test_loader_fails_closed_on_provenance_and_heldout_adversaries(self):
        mutations = {
            "not_candidate_conditioned": lambda fixture: fixture.rows[0].update(
                detector_conditioned=False
            ),
            "identity_overlap": lambda fixture: fixture.rows[8].update(
                parent_source_id=fixture.rows[0]["parent_source_id"]
            ),
            "heldout_filtering": lambda fixture: fixture.corpus["counts"]["by_split"][
                "test"
            ].update(raw_negative_candidates=3),
            "non_train_hard_mining": lambda fixture: fixture.corpus[
                "hard_negative_selection"
            ].update(scope="all_splits"),
            "top_k_drift": lambda fixture: fixture.rows[3].update(
                parent_source_id=fixture.rows[2]["parent_source_id"]
            ),
        }
        expected_messages = {
            "not_candidate_conditioned": "detector_conditioned",
            "identity_overlap": "identity overlap",
            "heldout_filtering": "filtered",
            "non_train_hard_mining": "train_only",
            "top_k_drift": "exceeds declared top-K",
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                fixture = CandidateDatasetFixture(Path(directory))
                mutate(fixture)
                corpus_sha = fixture.write_corpus()
                with self.assertRaisesRegex(ValueError, expected_messages[name]):
                    load_verified_dataset(
                        fixture.dataset, expected_corpus_sha256=corpus_sha
                    )

    def test_loader_rejects_corpus_array_transitive_and_row_hash_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CandidateDatasetFixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "corpus hash drift"):
                load_verified_dataset(fixture.dataset, expected_corpus_sha256="0" * 64)

        with tempfile.TemporaryDirectory() as directory:
            fixture = CandidateDatasetFixture(Path(directory))
            values = np.load(fixture.features)
            values[0, 0, 0] += 1
            np.save(fixture.features, values, allow_pickle=False)
            with self.assertRaisesRegex(ValueError, "features.npy hash drift"):
                load_verified_dataset(
                    fixture.dataset,
                    expected_corpus_sha256=sha256_file(fixture.corpus_path),
                )

        with tempfile.TemporaryDirectory() as directory:
            fixture = CandidateDatasetFixture(Path(directory))
            fixture.bound_files["artifact"].write_bytes(b"drift")
            with self.assertRaisesRegex(ValueError, "hash drift"):
                load_verified_dataset(
                    fixture.dataset,
                    expected_corpus_sha256=sha256_file(fixture.corpus_path),
                )

        with tempfile.TemporaryDirectory() as directory:
            fixture = CandidateDatasetFixture(Path(directory))
            fixture.rows[0]["candidate_feature_sha256"] = "0" * 64
            corpus_sha = fixture.write_corpus()
            with self.assertRaisesRegex(ValueError, "candidate feature hash drift"):
                load_verified_dataset(
                    fixture.dataset, expected_corpus_sha256=corpus_sha
                )

    def test_batching_is_deterministic_and_emphasizes_triggered_negatives(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CandidateDatasetFixture(Path(directory))
            dataset = load_verified_dataset(
                fixture.dataset, expected_corpus_sha256=sha256_file(fixture.corpus_path)
            )
            first = BalancedCandidateBatcher(dataset, batch_size=8, seed=19)
            second = BalancedCandidateBatcher(dataset, batch_size=8, seed=19)
            first_features, first_labels = first.batch(0)
            second_features, second_labels = second.batch(0)
            np.testing.assert_array_equal(first_features, second_features)
            np.testing.assert_array_equal(first_labels, second_labels)
            self.assertEqual(int(np.sum(first_labels)), 2)
            self.assertEqual(int(np.sum(first_labels == 0)), 6)
            report = first.report()
            self.assertEqual(
                report["mode"],
                "bounded_negative_emphasis_uniform_group_round_robin",
            )
            self.assertEqual(report["negative_group_sampling"], "uniform_group")
            self.assertEqual(
                report["candidate_condition"], "frozen_detector_trigger_only"
            )
            self.assertEqual(report["sampling_split"], "train")
            self.assertEqual(report["configured_negative_sampling_share"], 0.75)
            self.assertEqual(report["realized_negative_sampling_share"], 0.75)
            self.assertEqual(
                report["samples_per_batch"], {"positive": 2, "negative": 6}
            )
            self.assertEqual(
                report["negative_sampling_share_bounds"],
                {"minimum": 0.5, "maximum": 0.75},
            )
            self.assertEqual(
                set(report["positive_provider_samples"]), {"provider-a", "provider-b"}
            )
            self.assertEqual(
                set(report["negative_group_samples"]), {"collision", "noise"}
            )
            self.assertTrue(
                all(
                    row["samples"] == 1
                    for row in report["positive_provider_samples"].values()
                )
            )
            self.assertTrue(
                all(
                    row["samples"] == 3
                    for row in report["negative_group_samples"].values()
                )
            )

    def test_negative_emphasis_is_bounded_exact_and_train_only(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CandidateDatasetFixture(Path(directory))
            dataset = load_verified_dataset(
                fixture.dataset, expected_corpus_sha256=sha256_file(fixture.corpus_path)
            )
            batcher = BalancedCandidateBatcher(
                dataset,
                batch_size=8,
                seed=29,
                negative_sampling_share=0.75,
            )
            features, labels = batcher.batch(3)
            self.assertEqual(int(np.sum(labels == 0)), 6)
            self.assertEqual(int(np.sum(labels == 1)), 2)
            train_values = {
                float(np.float32(value))
                for value in np.asarray(dataset.features[:4, 0, 0])
            }
            self.assertTrue(
                all(float(value) in train_values for value in features[:, 0, 0, 0])
            )

            for invalid in (0.49, 0.76, math.nan):
                with (
                    self.subTest(invalid=invalid),
                    self.assertRaisesRegex(ValueError, "negative_sampling_share"),
                ):
                    BalancedCandidateBatcher(
                        dataset,
                        batch_size=8,
                        seed=29,
                        negative_sampling_share=invalid,
                    )
            with self.assertRaisesRegex(ValueError, "must be an integer"):
                BalancedCandidateBatcher(
                    dataset,
                    batch_size=6,
                    seed=29,
                    negative_sampling_share=0.75,
                )

            output = Path(directory) / "invalid-output"
            with self.assertRaisesRegex(ValueError, "negative_sampling_share"):
                train_candidate_verifier(
                    fixture.dataset,
                    output,
                    expected_corpus_sha256=sha256_file(fixture.corpus_path),
                    steps=1,
                    batch_size=8,
                    negative_sampling_share=0.76,
                    backend=FakeBackend(),
                )
            self.assertFalse(output.exists())

    def test_proportional_negative_sampling_tracks_observed_candidate_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CandidateDatasetFixture(Path(directory))
            dataset = load_verified_dataset(
                fixture.dataset, expected_corpus_sha256=sha256_file(fixture.corpus_path)
            )
            first = BalancedCandidateBatcher(
                dataset,
                batch_size=8,
                seed=31,
                negative_group_sampling="proportional_example",
            )
            second = BalancedCandidateBatcher(
                dataset,
                batch_size=8,
                seed=31,
                negative_group_sampling="proportional_example",
            )
            for step in range(200):
                first_features, first_labels = first.batch(step)
                second_features, second_labels = second.batch(step)
                np.testing.assert_array_equal(first_features, second_features)
                np.testing.assert_array_equal(first_labels, second_labels)
            report = first.report()
            self.assertEqual(
                report["mode"], "bounded_negative_emphasis_proportional_example"
            )
            self.assertEqual(report["negative_group_sampling"], "proportional_example")
            # The fixture has one collision and one noise train negative, so the
            # proportional sampler remains approximately even without forcing
            # exact per-batch group balance.
            shares = [
                row["share_within_class"]
                for row in report["negative_group_samples"].values()
            ]
            self.assertTrue(all(0.4 < share < 0.6 for share in shares))
            with self.assertRaisesRegex(ValueError, "negative_group_sampling"):
                BalancedCandidateBatcher(
                    dataset,
                    batch_size=8,
                    seed=31,
                    negative_group_sampling="bogus",
                )

    def test_physical_hard_negative_share_is_bounded_and_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CandidateDatasetFixture(Path(directory))
            dataset = load_verified_dataset(
                fixture.dataset, expected_corpus_sha256=sha256_file(fixture.corpus_path)
            )
            dataset.rows[2]["capture_id"] = "hardneg-fixture"
            first = BalancedCandidateBatcher(
                dataset,
                batch_size=8,
                seed=37,
                negative_group_sampling="proportional_example",
                physical_hard_negative_share=0.5,
            )
            second = BalancedCandidateBatcher(
                dataset,
                batch_size=8,
                seed=37,
                negative_group_sampling="proportional_example",
                physical_hard_negative_share=0.5,
            )
            for step in range(20):
                first_features, first_labels = first.batch(step)
                second_features, second_labels = second.batch(step)
                np.testing.assert_array_equal(first_features, second_features)
                np.testing.assert_array_equal(first_labels, second_labels)
            report = first.report()
            self.assertEqual(
                report["configured_physical_hard_negative_share_within_negatives"],
                0.5,
            )
            self.assertEqual(
                report["negative_group_samples"]["noise"]["share_within_class"],
                0.5,
            )
            with self.assertRaisesRegex(ValueError, "requires proportional_example"):
                BalancedCandidateBatcher(
                    dataset,
                    batch_size=8,
                    seed=37,
                    negative_group_sampling="uniform_group",
                    physical_hard_negative_share=0.1,
                )
            with self.assertRaisesRegex(ValueError, "within"):
                BalancedCandidateBatcher(
                    dataset,
                    batch_size=8,
                    seed=37,
                    negative_group_sampling="proportional_example",
                    physical_hard_negative_share=0.51,
                )

    def test_feature_augmentation_is_training_only_seeded_and_nonnegative(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CandidateDatasetFixture(Path(directory))
            dataset = load_verified_dataset(
                fixture.dataset, expected_corpus_sha256=sha256_file(fixture.corpus_path)
            )
            first = BalancedCandidateBatcher(
                dataset, batch_size=4, seed=23, augmentation_profile="strong"
            )
            second = BalancedCandidateBatcher(
                dataset, batch_size=4, seed=23, augmentation_profile="strong"
            )
            first_features, first_labels = first.batch(7)
            second_features, second_labels = second.batch(7)
            np.testing.assert_array_equal(first_features, second_features)
            np.testing.assert_array_equal(first_labels, second_labels)
            self.assertTrue(np.all(first_features >= 0.0))
            self.assertEqual(
                first.report()["feature_augmentation"],
                {
                    "profile": "strong",
                    "level_offset": 1.5,
                    "noise_stddev": 0.35,
                    "max_time_shift_frames": 8,
                    "max_time_mask_frames": 12,
                    "max_frequency_mask_bins": 4,
                    "training_only": True,
                    "deterministic_seeded_by_step": True,
                },
            )

            unaugmented = BalancedCandidateBatcher(
                dataset, batch_size=4, seed=23, augmentation_profile="none"
            )
            plain_features, _ = unaugmented.batch(7)
            self.assertFalse(np.array_equal(first_features, plain_features))

    def test_rejects_unknown_feature_augmentation_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CandidateDatasetFixture(Path(directory))
            dataset = load_verified_dataset(
                fixture.dataset, expected_corpus_sha256=sha256_file(fixture.corpus_path)
            )
            with self.assertRaisesRegex(ValueError, "unknown feature augmentation"):
                BalancedCandidateBatcher(
                    dataset, batch_size=4, seed=23, augmentation_profile="extreme"
                )

    def test_validation_threshold_minimizes_false_candidates_at_recall_floor(self):
        rows = [
            {"candidate_id": "p1", "parent_source_id": "p1"},
            {"candidate_id": "p2", "parent_source_id": "p2"},
            {"candidate_id": "n1", "parent_source_id": "n1"},
            {"candidate_id": "n2", "parent_source_id": "n2"},
        ]
        logits = np.asarray([_logit(0.9), _logit(0.8), _logit(0.85), _logit(0.2)])
        labels = np.asarray([1, 1, 0, 0], dtype=np.int8)
        strict = evaluate_operating_point(
            logits, labels, rows, recall_floor=0.98, threshold=None
        )["selected"]
        self.assertEqual(strict["conditional_recall"], 1.0)
        self.assertEqual(strict["false_candidates"], 1)
        permissive = evaluate_operating_point(
            logits, labels, rows, recall_floor=0.5, threshold=None
        )["selected"]
        self.assertEqual(permissive["conditional_recall"], 0.5)
        self.assertEqual(permissive["false_candidates"], 0)

    def test_trainer_selects_only_on_validation_then_scores_test_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = CandidateDatasetFixture(root)
            backend = FakeBackend()
            evaluator_calls = []

            def evaluator(logits, labels, rows, *, recall_floor, threshold=None):
                evaluator_calls.append(
                    {
                        "split": rows[0]["split"],
                        "threshold_frozen": threshold is not None,
                    }
                )
                return evaluate_operating_point(
                    logits,
                    labels,
                    rows,
                    recall_floor=recall_floor,
                    threshold=threshold,
                )

            output = root / "output"
            report = train_candidate_verifier(
                fixture.dataset,
                output,
                expected_corpus_sha256=sha256_file(fixture.corpus_path),
                steps=2,
                batch_size=4,
                eval_every=1,
                conditional_recall_floor=0.98,
                seed=7,
                backend=backend,
                evaluator=evaluator,
            )
            self.assertEqual(report["winner"]["step"], 1)
            self.assertEqual(backend.loaded_steps, [1])
            self.assertEqual(
                [call[0] for call in backend.score_calls],
                [
                    "validation-checkpoint-1",
                    "validation-checkpoint-2",
                    "test-once-after-frozen-selection",
                ],
            )
            self.assertEqual(
                evaluator_calls,
                [
                    {"split": "validation", "threshold_frozen": False},
                    {"split": "validation", "threshold_frozen": False},
                    {"split": "validation", "threshold_frozen": True},
                    {"split": "test", "threshold_frozen": True},
                ],
            )
            self.assertFalse(report["selection_contract"]["test_used_for_selection"])
            self.assertEqual(report["selection_contract"]["test_score_passes"], 1)
            self.assertFalse(report["deployment_qualification"])
            self.assertEqual(report["architecture"]["output"], "one_logit")
            self.assertEqual(
                report["winner"]["checkpoint"]["sha256"],
                report["winner"]["best_weights"]["sha256"],
            )
            _verify_output_bindings(report["output_bindings"])
            artifact_manifest = json.loads(
                (output / "artifact-manifest.json").read_text(encoding="utf-8")
            )
            self.assertFalse(artifact_manifest["deployment_qualification"])
            _verify_output_bindings(artifact_manifest["bindings"])
            (output / "best.weights.h5").write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "output binding drift"):
                _verify_output_bindings(report["output_bindings"])

    def test_dscnn_contract_has_only_int8_friendly_core_ops_and_exact_cost(self):
        operations = {layer["op"] for layer in dscnn_spec()}
        self.assertEqual(
            operations,
            {"Conv2D", "DepthwiseConv2D", "Flatten", "Dense"},
        )
        pointwise = [
            layer
            for layer in dscnn_spec()
            if layer["op"] == "Conv2D" and layer["kernel"] == (1, 1)
        ]
        self.assertEqual(len(pointwise), 4)
        self.assertEqual(
            dscnn_spec()[-1], {"name": "verifier_logit", "op": "Dense", "units": 1}
        )
        cost = estimate_dscnn_cost()
        self.assertEqual(cost["parameter_estimate"], 15793)
        self.assertEqual(cost["mac_estimate"], 2801952)

    def test_compact_default_preserves_original_topology_contract(self):
        self.assertEqual(DEFAULT_MODEL_VARIANT, "compact")
        self.assertEqual(MODEL_VARIANT_CHANNELS["compact"], (24, 32, 48, 64, 96))
        self.assertEqual(dscnn_spec(), dscnn_spec("compact"))
        self.assertEqual(
            estimate_dscnn_cost(), estimate_dscnn_cost(model_variant="compact")
        )

        from tools.convert_kizz_candidate_verifier import _model_topology_sha256

        backend = TensorFlowVerifierBackend()
        implicit = backend.build_model(learning_rate=0.0005, seed=248)
        explicit = backend.build_model(
            learning_rate=0.0005, seed=248, model_variant="compact"
        )
        spec = dscnn_spec("compact")
        self.assertEqual(len(implicit.layers) - 1, len(spec))
        for keras_layer, expected in zip(implicit.layers[1:], spec):
            config = keras_layer.get_config()
            self.assertEqual(keras_layer.name, expected["name"])
            self.assertEqual(type(keras_layer).__name__, expected["op"])
            for expected_key, config_key in (
                ("filters", "filters"),
                ("kernel", "kernel_size"),
                ("strides", "strides"),
                ("activation", "activation"),
                ("units", "units"),
            ):
                if expected_key not in expected:
                    continue
                actual = config[config_key]
                if isinstance(expected[expected_key], tuple):
                    actual = tuple(actual)
                self.assertEqual(actual, expected[expected_key])
        self.assertEqual(
            _model_topology_sha256(backend.model_json(implicit)),
            _model_topology_sha256(backend.model_json(explicit)),
        )
        bound_json = bind_model_json_provenance(
            backend.model_json(implicit),
            model_variant="compact",
            cost=estimate_dscnn_cost(),
        )
        self.assertEqual(
            _model_topology_sha256(bound_json),
            _model_topology_sha256(backend.model_json(implicit)),
        )
        import tensorflow as tf

        reloaded = tf.keras.models.model_from_json(bound_json)
        self.assertEqual(reloaded.name, "kizz_candidate_verifier_dscnn")
        self.assertEqual(reloaded.count_params(), 15793)

    def test_wide_topology_is_deterministic_and_within_target_cost_band(self):
        wide = dscnn_spec("wide")
        self.assertEqual(MODEL_VARIANT_CHANNELS["wide"], (32, 48, 64, 80, 112))
        self.assertEqual(wide, dscnn_spec("wide"))
        self.assertEqual(
            [layer["filters"] for layer in wide if layer["op"] == "Conv2D"],
            [32, 48, 64, 80, 112],
        )
        self.assertEqual(
            {layer["op"] for layer in wide},
            {"Conv2D", "DepthwiseConv2D", "Flatten", "Dense"},
        )
        compact_cost = estimate_dscnn_cost(model_variant="compact")
        wide_cost = estimate_dscnn_cost(model_variant="wide")
        self.assertEqual(wide_cost["parameter_estimate"], 24081)
        self.assertEqual(wide_cost["mac_estimate"], 4310512)
        self.assertGreaterEqual(
            wide_cost["mac_estimate"] / compact_cost["mac_estimate"], 1.5
        )
        self.assertLess(wide_cost["mac_estimate"] / compact_cost["mac_estimate"], 2.0)

    def test_compact_relu6_keeps_compact_cost_and_bounds_all_convolutions(self):
        spec = dscnn_spec("compact_relu6")
        convolutions = [
            layer for layer in spec if layer["op"] in {"Conv2D", "DepthwiseConv2D"}
        ]
        self.assertTrue(convolutions)
        self.assertTrue(all(layer["activation"] == "relu6" for layer in convolutions))
        relu6_cost = estimate_dscnn_cost(model_variant="compact_relu6")
        compact_cost = estimate_dscnn_cost(model_variant="compact")
        self.assertEqual(
            relu6_cost["parameter_estimate"], compact_cost["parameter_estimate"]
        )
        self.assertEqual(relu6_cost["mac_estimate"], compact_cost["mac_estimate"])
        backend = TensorFlowVerifierBackend()
        model = backend.build_model(
            learning_rate=0.0005, seed=248, model_variant="compact_relu6"
        )
        model_convolutions = [
            layer
            for layer in model.layers
            if layer.__class__.__name__ in {"Conv2D", "DepthwiseConv2D"}
        ]
        self.assertTrue(
            all(layer.activation.__name__ == "relu6" for layer in model_convolutions)
        )

    def test_training_binds_variant_spec_cost_and_model_json_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = CandidateDatasetFixture(root)
            output = root / "wide-output"
            report = train_candidate_verifier(
                fixture.dataset,
                output,
                expected_corpus_sha256=sha256_file(fixture.corpus_path),
                steps=1,
                batch_size=4,
                eval_every=1,
                conditional_recall_floor=0.5,
                model_variant="wide",
                backend=FakeBackend(),
            )
            architecture = report["architecture"]
            self.assertEqual(architecture["variant"], "wide")
            self.assertEqual(architecture["channel_plan"], [32, 48, 64, 80, 112])
            self.assertEqual(architecture["input_shape"], [260, 40, 1])
            self.assertEqual(architecture["parameter_count"], 24081)
            self.assertEqual(architecture["mac_estimate"], 4310512)
            self.assertEqual(
                architecture["dscnn_spec"], json.loads(json.dumps(dscnn_spec("wide")))
            )
            model_json = json.loads((output / "model.json").read_text(encoding="utf-8"))
            self.assertEqual(model_json["model_variant"], "wide")
            model_provenance = model_json[MODEL_JSON_PROVENANCE_KEY]
            self.assertEqual(model_provenance["variant"], "wide")
            self.assertEqual(model_provenance["channel_plan"], [32, 48, 64, 80, 112])
            self.assertEqual(model_provenance["cost"]["parameter_estimate"], 24081)
            self.assertEqual(model_provenance["cost"]["mac_estimate"], 4310512)
            self.assertEqual(
                model_provenance["topology_sha256"],
                architecture["topology_sha256"],
            )
            manifest = json.loads(
                (output / "artifact-manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["model"]["variant"], "wide")
            self.assertEqual(manifest["model"]["parameter_count"], 24081)
            self.assertEqual(manifest["model"]["mac_count"], 4310512)
            self.assertEqual(
                manifest["model"]["model_json"]["sha256"],
                report["output_bindings"]["model_architecture"]["sha256"],
            )

    def test_temporal_variant_preserves_time_resolution_with_esp_nn_ops(self):
        temporal = dscnn_spec("temporal")
        depthwise = [layer for layer in temporal if layer["op"] == "DepthwiseConv2D"]
        self.assertEqual(
            [layer["strides"] for layer in depthwise],
            [(2, 2), (2, 2), (2, 2), (2, 2)],
        )
        self.assertEqual(temporal[0]["strides"], (1, 2))
        self.assertEqual(
            {layer["op"] for layer in temporal},
            {"Conv2D", "DepthwiseConv2D", "Flatten", "Dense"},
        )
        cost = estimate_dscnn_cost(model_variant="temporal")
        self.assertGreater(
            cost["mac_estimate"],
            estimate_dscnn_cost(model_variant="wide")["mac_estimate"],
        )
        self.assertLess(cost["mac_estimate"], 12_000_000)
        backend = TensorFlowVerifierBackend()
        model = backend.build_model(
            learning_rate=0.0005, seed=248, model_variant="temporal"
        )
        loss = backend.train_batch(
            model,
            np.zeros((2, 260, 40, 1), dtype=np.float32),
            np.asarray([1.0, 0.0], dtype=np.float32),
        )
        self.assertTrue(math.isfinite(loss))

    def test_real_tensorflow_backend_executes_one_training_batch(self):
        backend = TensorFlowVerifierBackend()
        model = backend.build_model(learning_rate=0.0005, seed=248)
        convolution_layers = [
            layer
            for layer in model.layers
            if layer.__class__.__name__ in {"Conv2D", "DepthwiseConv2D"}
        ]
        self.assertEqual(len(convolution_layers), 9)
        self.assertTrue(
            all(layer.activation.__name__ == "relu" for layer in convolution_layers)
        )
        features = np.zeros((2, 260, 40, 1), dtype=np.float32)
        labels = np.asarray([1.0, 0.0], dtype=np.float32)
        loss = backend.train_batch(model, features, labels)
        self.assertTrue(math.isfinite(loss))
        self.assertEqual(backend.count_params(model), 15793)

    def test_device_robustness_perturbs_only_shared_weight_training_graph(self):
        backend = TensorFlowVerifierBackend()
        model = backend.build_model(
            learning_rate=0.0005,
            seed=248,
            model_variant="compact_relu6",
            device_robustness_profile="int8_lsb1",
        )
        self.assertIsInstance(model, TensorFlowVerifierModel)
        self.assertEqual(backend.count_params(model), 15793)
        training_names = {layer.name for layer in model.training.layers}
        deployment_names = {layer.name for layer in model.deployment.layers}
        self.assertEqual(
            sum(name.endswith("_training_fake_quant") for name in training_names), 9
        )
        self.assertEqual(
            sum(name.endswith("_training_lsb_noise") for name in training_names), 9
        )
        self.assertFalse(
            any("training_fake_quant" in name for name in deployment_names)
        )
        self.assertFalse("training_lsb_noise" in backend.model_json(model))
        profile = DEVICE_ROBUSTNESS_PROFILES["int8_lsb1"]
        expected_stddev = (
            float(profile["activation_max"]) - float(profile["activation_min"])
        ) / ((1 << int(profile["activation_quantization_bits"])) - 1)
        noise_layers = [
            layer
            for layer in model.training.layers
            if layer.name.endswith("_training_lsb_noise")
        ]
        self.assertTrue(
            all(math.isclose(layer.stddev, expected_stddev) for layer in noise_layers)
        )
        loss = backend.train_batch(
            model,
            np.zeros((2, 260, 40, 1), dtype=np.float32),
            np.asarray([1.0, 0.0], dtype=np.float32),
        )
        self.assertTrue(math.isfinite(loss))

    def test_device_robustness_requires_bounded_relu6_variant(self):
        backend = TensorFlowVerifierBackend()
        with self.assertRaisesRegex(ValueError, "compact_relu6"):
            backend.build_model(
                learning_rate=0.0005,
                seed=248,
                model_variant="compact",
                device_robustness_profile="int8_lsb1",
            )


if __name__ == "__main__":
    unittest.main()
