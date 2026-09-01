import unittest

from tools.compose_kizz_control_c1_manifest import reusable_frozen_negatives


class ComposeKizzControlC1ManifestTests(unittest.TestCase):
    def row(self, split, *, eligible, locked=False, group="background_noise"):
        return {
            "label": 0,
            "split": split,
            "training_eligible": eligible,
            "locked_deployment_anchor": locked,
            "source_group": group,
            "source_id": f"{split}-{eligible}-{locked}-{group}",
        }

    def test_retains_clean_evaluation_negatives_but_excludes_locked_holdout(self):
        rows = [
            self.row("train", eligible=True),
            self.row("validation", eligible=False),
            self.row("test", eligible=False),
            self.row("test", eligible=False, locked=True),
            self.row("test", eligible=False, group="unsupported"),
        ]
        selected = reusable_frozen_negatives(rows)
        self.assertEqual({row["split"] for row in selected}, {"train", "validation", "test"})
        self.assertFalse(any(row["locked_deployment_anchor"] for row in selected))

    def test_rejects_ambiguous_split_eligibility(self):
        with self.assertRaisesRegex(ValueError, "train negative"):
            reusable_frozen_negatives([self.row("train", eligible=False)])
        with self.assertRaisesRegex(ValueError, "evaluation negative"):
            reusable_frozen_negatives([self.row("validation", eligible=True)])


if __name__ == "__main__":
    unittest.main()
