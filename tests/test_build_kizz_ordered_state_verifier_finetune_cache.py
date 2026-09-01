import unittest

from tools.build_kizz_ordered_state_verifier_finetune_cache import (
    materialization_plan,
)


class OrderedStateVerifierFineTuneCacheTests(unittest.TestCase):
    def test_plan_is_train_only_balanced_and_deterministic(self):
        candidates = [
            {"split": "train", "label": 1},
            {"split": "train", "label": 0, "source_group": "ordinary"},
            {"split": "train", "label": 0, "source_group": "consumed_failure"},
            {"split": "validation", "label": 1},
            {"split": "test", "label": 0, "source_group": "consumed_failure"},
        ]
        physical = [{"split": "test", "label": 1}, {"split": "test", "label": 1}]
        first = materialization_plan(
            candidates,
            physical,
            positive_repeats=3,
            physical_repeats=2,
            seed=7,
            hard_negative_group="consumed_failure",
            hard_negative_repeats=4,
        )
        second = materialization_plan(
            candidates,
            physical,
            positive_repeats=3,
            physical_repeats=2,
            seed=7,
            hard_negative_group="consumed_failure",
            hard_negative_repeats=4,
        )
        self.assertEqual(first, second)
        self.assertEqual(first.count(("candidate", 0)), 3)
        self.assertEqual(first.count(("candidate", 1)), 1)
        self.assertEqual(first.count(("candidate", 2)), 4)
        self.assertNotIn(("candidate", 3), first)
        self.assertNotIn(("candidate", 4), first)
        self.assertEqual(first.count(("physical", 0)), 2)
        self.assertEqual(first.count(("physical", 1)), 2)


if __name__ == "__main__":
    unittest.main()
