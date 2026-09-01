import unittest

import numpy as np

from microwakeword.phoneme_student import compact_phone_contract
from tools.cache_kizz_teacher_causal_windows import (
    causal_suffix_score_grid,
    teacher_endpoint_frames,
)


class TeacherCausalWindowCacheTests(unittest.TestCase):
    def test_endpoint_mapping_never_uses_future_teacher_frames(self):
        ends = teacher_endpoint_frames(
            np.asarray([0.635, 0.665, 2.585]),
            teacher_frame_center_seconds=0.01246875,
            teacher_frame_stride_seconds=0.02,
            teacher_frame_count=130,
        )
        np.testing.assert_array_equal(ends, [32, 33, 129])
        centers = 0.01246875 + 0.02 * (ends - 1)
        self.assertTrue(np.all(centers <= np.asarray([0.635, 0.665, 2.585])))

    def test_score_grid_is_prefix_specific_and_shape_bound(self):
        contract = compact_phone_contract()
        rng = np.random.default_rng(231)
        logits = rng.normal(size=(3, 40, len(contract["tokens"]))).astype(np.float32)
        result = causal_suffix_score_grid(
            logits,
            contract,
            end_frames=np.asarray([28, 34, 40]),
            window_lengths=(28, 34, 40),
        )
        for values in result.values():
            self.assertEqual(values.shape, (3, 3))
        self.assertTrue(np.isfinite(result["decision_score"]).all())
        self.assertFalse(
            np.array_equal(
                result["decision_score"][:, 0], result["decision_score"][:, -1]
            )
        )

    def test_score_grid_rejects_endpoint_before_deployment_window(self):
        contract = compact_phone_contract()
        logits = np.zeros((1, 40, len(contract["tokens"])), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "causal suffix score grid"):
            causal_suffix_score_grid(
                logits,
                contract,
                end_frames=np.asarray([27]),
                window_lengths=(28, 34),
            )


if __name__ == "__main__":
    unittest.main()
