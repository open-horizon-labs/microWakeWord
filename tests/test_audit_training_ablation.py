import unittest

from tools.audit_training_ablation import differences, is_allowed


class TrainingAblationAuditTest(unittest.TestCase):
    def test_reports_nested_scalar_and_list_differences(self):
        found = differences(
            {"seed": 1, "schedule": [100, 200], "groups": {"positive": 0.7}},
            {"seed": 1, "schedule": [100, 300], "groups": {"positive": 0.5}},
        )

        self.assertEqual(
            [item["path"] for item in found],
            ["groups.positive", "schedule[1]"],
        )

    def test_allows_a_declared_mapping_and_its_children(self):
        self.assertTrue(is_allowed("sampling_groups.positive", ["sampling_groups"]))
        self.assertTrue(is_allowed("train_dir", ["train_dir"]))
        self.assertFalse(is_allowed("training_seed", ["train_dir"]))


if __name__ == "__main__":
    unittest.main()
