import unittest

import numpy as np
import tensorflow as tf

from microwakeword.data import FeatureHandler
from microwakeword.train import (
    _evaluate_batches,
    configured_training_metrics,
    validate_nonstreaming,
)


class StreamingValidationDataTest(unittest.TestCase):
    def test_batches_are_bounded_and_preserve_provider_order(self):
        class Provider:
            label = 0.0
            penalty_weight = 1.0

            def get_feature_generator(self, mode, features_length, strategy):
                self.arguments = (mode, features_length, strategy)
                for value in range(5):
                    yield np.full((features_length, 2), value, dtype=np.float32)

        provider = Provider()
        handler = FeatureHandler.__new__(FeatureHandler)
        handler.feature_providers = [provider]
        handler.evaluation_enabled = [True]

        batches = list(
            handler.get_data_batches(
                "validation_ambient",
                batch_size=2,
                features_length=3,
                truncation_strategy="split",
            )
        )

        self.assertEqual([batch[0].shape[0] for batch in batches], [2, 2, 1])
        np.testing.assert_array_equal(
            np.concatenate([batch[0][:, 0, 0] for batch in batches]),
            np.arange(5),
        )
        self.assertEqual(provider.arguments, ("validation_ambient", 3, "split"))


class StreamingValidationMetricTest(unittest.TestCase):
    def test_accumulated_batch_metrics_match_one_shot_evaluation(self):
        model = tf.keras.Sequential(
            [tf.keras.layers.Input(shape=(1,)), tf.keras.layers.Dense(1)]
        )
        model.compile(
            optimizer="sgd",
            loss="binary_crossentropy",
            metrics=configured_training_metrics({}),
        )
        fingerprints = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]], dtype=np.float32)
        ground_truth = np.array([[0], [0], [1], [1], [1]], dtype=np.float32)

        one_shot = model.evaluate(
            fingerprints,
            ground_truth,
            batch_size=1024,
            return_dict=True,
            verbose=0,
        )
        model.reset_metrics()
        streamed = _evaluate_batches(
            model,
            (
                (fingerprints[:2], ground_truth[:2], None),
                (fingerprints[2:4], ground_truth[2:4], None),
                (fingerprints[4:], ground_truth[4:], None),
            ),
        )

        self.assertEqual(set(one_shot), set(streamed))
        for name in one_shot:
            np.testing.assert_allclose(streamed[name], one_shot[name], rtol=1e-6)

    def test_validate_nonstreaming_uses_ambient_batch_stream(self):
        class Handler:
            def get_data(self, mode, **kwargs):
                if mode.endswith("_ambient"):
                    raise AssertionError("ambient data must not be materialized")
                return (
                    np.zeros((2, 1), dtype=np.float32),
                    np.array([0, 1]),
                    np.ones(2),
                )

            def get_mode_size(self, mode):
                return 2 if mode == "validation_ambient" else 0

            def get_mode_duration(self, mode):
                return 3600.0

            def get_data_batches(self, mode, **kwargs):
                self.batch_size = kwargs["batch_size"]
                yield (
                    np.zeros((2, 1), dtype=np.float32),
                    np.array([0, 0]),
                    np.ones(2),
                )

        class Model:
            def reset_metrics(self):
                self.reset_count = getattr(self, "reset_count", 0) + 1

            def evaluate(self, *args, **kwargs):
                return self.result()

            def test_step(self, batch):
                self.seen_batch_sizes = getattr(self, "seen_batch_sizes", [])
                self.seen_batch_sizes.append(len(batch[0]))
                return self.result()

            @staticmethod
            def result():
                zeros = np.zeros(101)
                ones = np.ones(101)
                return {
                    "accuracy": 1.0,
                    "recall": 1.0,
                    "precision": 1.0,
                    "auc": 1.0,
                    "loss": 0.0,
                    "fp": zeros,
                    "tp": ones,
                    "fn": zeros,
                }

        handler = Handler()
        model = Model()
        metrics = validate_nonstreaming(
            {"batch_size": 2, "spectrogram_length": 1}, handler, model, "validation"
        )

        self.assertEqual(handler.batch_size, 1024)
        self.assertEqual(model.seen_batch_sizes, [2])
        self.assertEqual(metrics["ambient_false_positives"], 0)


if __name__ == "__main__":
    unittest.main()
