import unittest

from tools.compare_kizz_physical_hard_negative_verifiers import summarize_scores


class ComparePhysicalHardNegativeVerifierTests(unittest.TestCase):
    def test_summarizes_evaluation_and_physical_families(self):
        rows = [
            {"split": "validation", "label": 1},
            {"split": "validation", "label": 0},
            {"split": "test", "label": 1},
            {"split": "test", "label": 0},
            {"split": "train", "label": 0, "capture_id": "hardneg-a"},
            {"split": "train", "label": 0, "capture_id": "hardneg-b"},
        ]
        result = summarize_scores(
            rows,
            [1.0, -1.0, -0.1, 0.2, -0.3, 0.4],
            {"hardneg-a": "music", "hardneg-b": "speech"},
            threshold=0.0,
        )
        self.assertEqual(result["validation_positive"]["recall"], 1.0)
        self.assertEqual(result["test_positive"]["recall"], 0.0)
        self.assertEqual(result["physical_hard_negative"]["accepted"], 1)
        self.assertEqual(result["physical_by_family"]["music"]["rejected"], 1)
        self.assertEqual(result["physical_by_family"]["speech"]["accepted"], 1)

    def test_rejects_unbound_physical_capture(self):
        with self.assertRaisesRegex(ValueError, "unbound"):
            summarize_scores(
                [{"split": "train", "label": 0, "capture_id": "hardneg-x"}],
                [-1.0],
                {},
                threshold=0.0,
            )


if __name__ == "__main__":
    unittest.main()
