import unittest

import numpy as np

from microwakeword.kizz_phoneme_teacher import (
    best_window_score,
    choose_validation_threshold,
    ctc_log_probability,
    ctc_log_probability_batch,
    score_window,
    resolve_hf_weights_path,
)


class KizzPhonemeTeacherTests(unittest.TestCase):
    def test_resolves_exactly_one_local_teacher_weights_file(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "model.safetensors"
            weights.write_bytes(b"weights")
            self.assertEqual(
                resolve_hf_weights_path(
                    str(root), revision="artifact-sha", local_files_only=True
                ),
                weights.resolve(),
            )
            (root / "pytorch_model.bin").write_bytes(b"other")
            with self.assertRaises(ValueError):
                resolve_hf_weights_path(
                    str(root), revision="artifact-sha", local_files_only=True
                )

    def test_ctc_accepts_blank_separated_canonical_path(self):
        # blank, h, blank, a, blank, i, blank; vocabulary is blank/h/a/i.
        logits = np.full((7, 4), -20.0)
        for frame, token in enumerate((0, 1, 0, 2, 0, 3, 0)):
            logits[frame, token] = 0.0
        log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        value = ctc_log_probability(log_probs, (1, 2, 3), blank_id=0)
        self.assertGreater(value, -1.0)

    def test_repeated_tokens_need_a_blank(self):
        logits = np.full((2, 2), -20.0)
        logits[:, 1] = 0.0
        log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        direct = ctc_log_probability(log_probs, (1, 1), blank_id=0)
        self.assertLess(direct, -10.0)
        logits = np.vstack(
            (logits[0], np.array([-20.0, 0.0]), np.array([0.0, -20.0]), logits[1])
        )
        log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        self.assertGreater(ctc_log_probability(log_probs, (1, 1), blank_id=0), -1.0)

    def test_batched_ctc_is_the_scalar_reference_on_varied_paths(self):
        rng = np.random.default_rng(231)
        logits = rng.normal(size=(7, 19, 8))
        log_probs = logits - np.logaddexp.reduce(logits, axis=2, keepdims=True)
        for path in ((1, 2, 3), (1, 1, 2), (4, 5, 4, 6)):
            expected = np.asarray(
                [
                    ctc_log_probability(window, path, blank_id=0)
                    for window in log_probs
                ]
            )
            actual = ctc_log_probability_batch(log_probs, path, blank_id=0)
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_collision_margin_is_an_independent_guard(self):
        logits = np.full((7, 3), -10.0)
        logits[:, 1] = 0.0  # canonical
        canonical = score_window(
            logits, canonical_tokens=(1,), collision_tokens=((2,),), blank_id=0
        )
        self.assertGreater(canonical.collision_margin, 5.0)
        rejected = best_window_score(
            logits,
            canonical_tokens=(1,),
            collision_tokens=((1,),),
            blank_id=0,
            window_lengths=(7,),
            hop=1,
            beta=0.1,
        )
        self.assertFalse(rejected.eligible)

    def test_sliding_windows_choose_best_eligible_window(self):
        logits = np.full((8, 3), -10.0)
        logits[4:, 1] = 0.0
        result = best_window_score(
            logits,
            canonical_tokens=(1,),
            collision_tokens=((2,),),
            blank_id=0,
            window_lengths=(4,),
            hop=2,
            beta=0.0,
        )
        self.assertEqual((result.start_frame, result.end_frame), (4, 8))

    def test_threshold_uses_validation_only(self):
        result = choose_validation_threshold(
            [0.9, 0.8, 0.7],
            [0.1, 0.2],
            negative_exposure_seconds=7200,
            min_recall=2 / 3,
            max_faph=0.1,
        )
        self.assertTrue(result["qualified"])
        self.assertGreaterEqual(result["threshold"], 0.7)


if __name__ == "__main__":
    unittest.main()
