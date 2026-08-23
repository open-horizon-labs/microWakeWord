import unittest
from collections import Counter

import numpy as np

from microwakeword.data import FeatureHandler, MmapFeatureGenerator, largest_remainder_counts
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


class StratifiedSamplingTest(unittest.TestCase):
    def test_allocates_exact_batch_by_declared_group_share(self):
        self.assertEqual(
            largest_remainder_counts(
                64,
                {
                    "piper": 0.4,
                    "designed": 0.3,
                    "device": 0.2,
                    "targeted_negative": 0.1,
                },
            ),
            {
                "piper": 26,
                "designed": 19,
                "device": 13,
                "targeted_negative": 6,
            },
        )

    def test_sampling_ledger_reports_realized_source_and_group_shares(self):
        handler = FeatureHandler.__new__(FeatureHandler)
        handler.sampling_group_weights = {"designed": 0.75, "device": 0.25}
        handler.training_sampling_counts = Counter(
            {
                ("designed", "adult"): 3,
                ("designed", "teen"): 3,
                ("device", "kizz"): 2,
            }
        )
        handler.training_weighted_pressure = Counter(
            {
                ("designed", "adult"): 3.0,
                ("designed", "teen"): 3.0,
                ("device", "kizz"): 4.0,
            }
        )
        handler.sampling_source_config = {
            "adult": {"truth": True, "penalty_weight": 1.0},
            "teen": {"truth": True, "penalty_weight": 1.0},
            "kizz": {"truth": True, "penalty_weight": 2.0},
        }

        ledger = handler.sampling_ledger()

        self.assertEqual(ledger["total_samples"], 8)
        self.assertEqual(ledger["realized_groups"]["designed"]["share"], 0.75)
        self.assertEqual(ledger["realized_sources"]["kizz"]["samples"], 2)
        self.assertEqual(
            ledger["realized_sources"]["kizz"]["weighted_pressure_share"], 0.4
        )


class FalseAcceptConstraintTest(unittest.TestCase):
    def test_labeled_negative_accept_makes_cutoff_unusable(self):
        constrained = constrain_faph_by_negative_false_accepts(
            np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])
        )

        self.assertTrue(np.isinf(constrained[0]))
        np.testing.assert_array_equal(constrained[1:], [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
