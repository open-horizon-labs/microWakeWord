import unittest

from tools.audit_source_ablation import class_exposure, features_by_source


class SourceAblationAuditTest(unittest.TestCase):
    def test_class_exposure_sums_group_weights_by_truth(self):
        config = {
            "sampling_groups": {"positive": 0.4, "negative_a": 0.3, "negative_b": 0.3},
            "features": [
                {"sampling_source": "wake", "sampling_group": "positive", "sampling_weight": 1, "truth": True},
                {"sampling_source": "foil-a", "sampling_group": "negative_a", "sampling_weight": 1, "truth": False},
                {"sampling_source": "foil-b", "sampling_group": "negative_b", "sampling_weight": 1, "truth": False},
            ],
        }

        self.assertEqual(class_exposure(config), {"positive": 0.4, "negative": 0.6})

    def test_rejects_a_group_with_mixed_truth_labels(self):
        config = {
            "sampling_groups": {"mixed": 1.0},
            "features": [
                {"sampling_source": "wake", "sampling_group": "mixed", "sampling_weight": 1, "truth": True},
                {"sampling_source": "foil", "sampling_group": "mixed", "sampling_weight": 1, "truth": False},
            ],
        }

        with self.assertRaisesRegex(ValueError, "one truth label"):
            class_exposure(config)

    def test_rejects_duplicate_source_names(self):
        config = {
            "features": [
                {"sampling_source": "same"},
                {"sampling_source": "same"},
            ]
        }

        with self.assertRaisesRegex(ValueError, "duplicate"):
            features_by_source(config)


if __name__ == "__main__":
    unittest.main()
