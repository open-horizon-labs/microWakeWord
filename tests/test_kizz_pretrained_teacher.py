import unittest

import numpy as np

from microwakeword.kizz_pretrained_teacher import fit_context, mix_positive_context


class PretrainedTeacherAudioTests(unittest.TestCase):
    def test_fit_context_is_fixed_length_and_deterministic(self):
        values = np.arange(5, dtype=np.float32)
        result = fit_context(values, context_samples=8, start=2)
        np.testing.assert_array_equal(result, np.array([0, 0, 0, 1, 2, 3, 4, 0], dtype=np.float32))

    def test_positive_context_does_not_use_zero_padding_shortcut(self):
        positive = np.ones(4, dtype=np.float32)
        background = np.full(8, 0.25, dtype=np.float32)
        mixed = mix_positive_context(
            positive, background, rng=np.random.default_rng(1), context_samples=8
        )
        self.assertTrue(np.all(np.isfinite(mixed)))
        self.assertGreater(float(np.max(mixed)), float(np.min(mixed)))


if __name__ == "__main__":
    unittest.main()
