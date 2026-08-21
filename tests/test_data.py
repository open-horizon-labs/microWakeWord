import unittest

import numpy as np

from microwakeword.data import MmapFeatureGenerator
from microwakeword.train import constrain_faph_by_negative_false_accepts


class SplitFeatureGeneratorTest(unittest.TestCase):
    @staticmethod
    def provider(spectrogram):
        provider = MmapFeatureGenerator.__new__(MmapFeatureGenerator)
        provider.truncation_strategy = "split"
        provider.loaded_features = [[spectrogram]]
        provider.feature_sets = {
            "validation_ambient": [{"loaded_feature_index": 0, "subindex": 0}]
        }
        provider.step = 0.01
        provider.stride = 3
        return provider

    def test_short_ambient_clip_yields_one_padded_window(self):
        spectrogram = np.ones((193, 40), dtype=np.float32)

        windows = list(
            self.provider(spectrogram).get_feature_generator(
                "validation_ambient", 220, "split"
            )
        )

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].shape, (220, 40))
        np.testing.assert_array_equal(windows[0][:27], 0)
        np.testing.assert_array_equal(windows[0][27:], 1)

    def test_exact_length_ambient_clip_yields_one_window(self):
        spectrogram = np.ones((220, 40), dtype=np.float32)

        windows = list(
            self.provider(spectrogram).get_feature_generator(
                "validation_ambient", 220, "split"
            )
        )

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].shape, (220, 40))

    def test_long_ambient_clip_yields_complete_windows(self):
        spectrogram = np.ones((281, 40), dtype=np.float32)

        windows = list(
            self.provider(spectrogram).get_feature_generator(
                "validation_ambient", 220, "split"
            )
        )

        self.assertEqual(len(windows), 3)
        self.assertTrue(all(window.shape == (220, 40) for window in windows))


class FalseAcceptConstraintTest(unittest.TestCase):
    def test_labeled_negative_accept_makes_cutoff_unusable(self):
        constrained = constrain_faph_by_negative_false_accepts(
            np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])
        )

        self.assertTrue(np.isinf(constrained[0]))
        np.testing.assert_array_equal(constrained[1:], [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
