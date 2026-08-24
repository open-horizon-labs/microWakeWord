import unittest

import numpy as np
import tensorflow as tf

from microwakeword.train import (
    combined_sample_weights,
    configure_trainable_layers,
    configured_training_loss,
    configured_training_metrics,
    require_binary_validation,
)


class TrainabilityPolicyTest(unittest.TestCase):
    def test_combined_sample_weights_remain_one_dimensional(self):
        weights = combined_sample_weights(
            labels=np.array([1, 0, 0]),
            penalty_weights=np.array([1.0, 2.0, 3.0]),
            positive_class_weight=4,
            negative_class_weight=5,
        )

        np.testing.assert_array_equal(weights, np.array([4.0, 10.0, 15.0]))
        self.assertEqual(weights.shape, (3,))

    def test_checkpoint_selection_requires_both_validation_classes(self):
        class Processor:
            def get_mode_label_counts(self, mode):
                return {0: 12, 1: 0}

        with self.assertRaisesRegex(ValueError, "missing: positive"):
            require_binary_validation(Processor())

    def test_freeze_feature_extractor_leaves_only_classifier_trainable(self):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input((8,)),
                tf.keras.layers.Dense(4, name="feature_dense"),
                tf.keras.layers.BatchNormalization(name="feature_norm"),
                tf.keras.layers.Activation("relu", name="feature_activation"),
                tf.keras.layers.Dense(1, name="classifier"),
            ]
        )

        configure_trainable_layers(model, {"freeze_feature_extractor": True})

        self.assertFalse(model.get_layer("feature_dense").trainable)
        self.assertFalse(model.get_layer("feature_norm").trainable)
        self.assertFalse(model.get_layer("feature_activation").trainable)
        self.assertTrue(model.get_layer("classifier").trainable)

    def test_binary_crossentropy_remains_the_default(self):
        loss = configured_training_loss({})

        self.assertIsInstance(loss, tf.keras.losses.BinaryCrossentropy)

    def test_focal_loss_uses_declared_gamma_without_implicit_class_balance(self):
        loss = configured_training_loss(
            {
                "training_loss": {
                    "name": "binary_focal_crossentropy",
                    "gamma": 1.5,
                    "apply_class_balancing": False,
                }
            }
        )

        self.assertIsInstance(loss, tf.keras.losses.BinaryFocalCrossentropy)
        self.assertEqual(loss.gamma, 1.5)
        self.assertFalse(loss.apply_class_balancing)

    def test_ordered_state_endpoint_uses_binary_loss_on_the_sequence_score(self):
        loss = configured_training_loss(
            {"training_loss": {"name": "ordered_state_sequence"}}
        )
        self.assertIsInstance(loss, tf.keras.losses.BinaryCrossentropy)
        self.assertTrue(loss.from_logits)
        metrics = configured_training_metrics(
            {"training_loss": {"name": "ordered_state_sequence"}}
        )
        self.assertEqual(metrics[0].metric.threshold, 0.5)
        self.assertEqual(metrics[-2].metric.name, "auc")

    def test_rejects_unknown_loss(self):
        with self.assertRaisesRegex(ValueError, "unsupported training loss"):
            configured_training_loss({"training_loss": {"name": "mystery"}})


if __name__ == "__main__":
    unittest.main()
