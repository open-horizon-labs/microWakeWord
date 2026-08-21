import unittest

import tensorflow as tf

from microwakeword.train import configure_trainable_layers


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


if __name__ == "__main__":
    unittest.main()
