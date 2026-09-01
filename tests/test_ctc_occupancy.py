import itertools
import unittest

import numpy as np

from microwakeword.ctc_occupancy import ctc_state_occupation_log_probs


def _collapse(sequence, blank):
    output = []
    previous = None
    for token in sequence:
        if token != previous and token != blank:
            output.append(token)
        previous = token
    return tuple(output)


def _exhaustive_occupation(probabilities, path, blank):
    frames, vocabulary = probabilities.shape
    accepted = []
    weights = []
    for sequence in itertools.product(range(vocabulary), repeat=frames):
        if _collapse(sequence, blank) == tuple(path):
            accepted.append(sequence)
            weights.append(
                np.prod([probabilities[t, token] for t, token in enumerate(sequence)])
            )
    weights = np.asarray(weights, dtype=np.float64)
    weights /= weights.sum()
    result = np.zeros_like(probabilities, dtype=np.float64)
    for sequence, weight in zip(accepted, weights, strict=True):
        for frame, token in enumerate(sequence):
            result[frame, token] += weight
    return result


class CtcOccupationTest(unittest.TestCase):
    def test_matches_exhaustive_alignment_marginals(self):
        probabilities = np.asarray(
            [
                [0.55, 0.35, 0.10],
                [0.25, 0.55, 0.20],
                [0.45, 0.15, 0.40],
                [0.50, 0.20, 0.30],
            ],
            dtype=np.float64,
        )
        expected = _exhaustive_occupation(probabilities, (1, 2), 0)
        actual = np.exp(
            ctc_state_occupation_log_probs(np.log(probabilities), (1, 2), 0)
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(actual.sum(axis=1), 1.0, atol=1e-7)

    def test_enforces_blank_between_repeated_tokens(self):
        probabilities = np.asarray(
            [
                [0.2, 0.8],
                [0.8, 0.2],
                [0.2, 0.8],
            ],
            dtype=np.float64,
        )
        actual = np.exp(
            ctc_state_occupation_log_probs(np.log(probabilities), (1, 1), 0)
        )
        np.testing.assert_allclose(actual[:, 1], [1.0, 0.0, 1.0], atol=1e-7)

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            ctc_state_occupation_log_probs(np.empty((0, 2)), (1,), 0)
        with self.assertRaises(ValueError):
            ctc_state_occupation_log_probs(np.zeros((1, 2)), (), 0)
        with self.assertRaises(ValueError):
            ctc_state_occupation_log_probs(np.zeros((1, 2)), (2,), 0)


if __name__ == "__main__":
    unittest.main()
