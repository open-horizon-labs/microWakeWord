import unittest

import tensorflow as tf

from microwakeword.train import configure_trainable_layers, configured_training_loss


class TrainabilityPolicyTest(unittest.TestCase):
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

    def test_rejects_unknown_loss(self):
        with self.assertRaisesRegex(ValueError, "unsupported training loss"):
            configured_training_loss({"training_loss": {"name": "mystery"}})


if __name__ == "__main__":
    unittest.main()
