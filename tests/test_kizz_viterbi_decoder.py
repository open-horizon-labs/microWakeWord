import unittest

import numpy as np

from microwakeword.phoneme_student import compact_phone_contract
from microwakeword.kizz_viterbi_decoder import (
    StreamingViterbiCTCDecoder,
    exhaustive_suffix_score,
)


class KizzViterbiDecoderTests(unittest.TestCase):
    def setUp(self):
        self.contract = compact_phone_contract()
        self.blank = self.contract["blank_id"]
        self.canonical = self.contract["canonical_path"]
        self.collision = next(iter(self.contract["collision_paths"].values()))

    def _path_logits(self, path, frames_per_token=2, noise=0.01):
        rng = np.random.default_rng(42)
        rows = []
        for token in path:
            rows.extend([rng.normal(0, noise, len(self.contract["tokens"])) for _ in range(frames_per_token - 1)])
            row = rng.normal(-8, noise, len(self.contract["tokens"]))
            row[token] = 8
            rows.append(row)
        return np.asarray(rows)

    def test_streaming_scores_match_exhaustive_randomized_logits(self):
        rng = np.random.default_rng(1234)
        logits = rng.normal(size=(80, len(self.contract["tokens"])))
        threshold = -2.0
        cooldown_frames = 3
        decoder = StreamingViterbiCTCDecoder(self.contract, window_lengths=(7, 13, 21), threshold=threshold, beta=-20.0, cooldown_frames=cooldown_frames)
        history = []
        expected_cooldown = 0
        for frame in logits:
            history.append(frame)
            actual_event = decoder.step(frame)
            expected = exhaustive_suffix_score(history, self.contract, window_lengths=(7, 13, 21), beta=-20.0)
            actual = decoder.score()
            self.assertEqual(actual.start_frame, expected.start_frame)
            self.assertAlmostEqual(actual.canonical_fit, expected.canonical_fit, places=12)
            self.assertAlmostEqual(actual.collision_margin, expected.collision_margin, places=12)
            if expected_cooldown:
                expected_event = None
                expected_cooldown -= 1
            elif expected.canonical_fit >= threshold:
                expected_event = expected
                expected_cooldown = cooldown_frames
            else:
                expected_event = None
            self.assertEqual(actual_event is None, expected_event is None)

    def test_repeated_token_ctc_semantics_allow_blank_or_repeat(self):
        repeated = [self.blank, self.canonical[0], self.blank, self.canonical[0], self.blank]
        logits = self._path_logits(repeated, frames_per_token=1)
        repeated_contract = self.contract | {
            "canonical_path": [self.canonical[0], self.canonical[0]],
            "collision_paths": {"other": [self.canonical[0], self.canonical[1]]},
        }
        score = exhaustive_suffix_score(logits, repeated_contract, window_lengths=(len(logits),), beta=-100.0)
        self.assertTrue(np.isfinite(score.canonical_fit))

    def test_collision_is_rejected_by_margin(self):
        logits = self._path_logits(self.collision, frames_per_token=2)
        decoder = StreamingViterbiCTCDecoder(self.contract, window_lengths=(len(logits),), threshold=-100.0, beta=0.0, cooldown_frames=0)
        events = [decoder.step(frame) for frame in logits]
        self.assertTrue(all(event is None for event in events))

    def test_threshold_and_cooldown(self):
        logits = self._path_logits(self.canonical, frames_per_token=2)
        decoder = StreamingViterbiCTCDecoder(self.contract, window_lengths=(len(logits),), threshold=-100.0, beta=-100.0, cooldown_frames=3)
        events = [decoder.step(frame) for frame in logits]
        detected = [event for event in events if event is not None]
        self.assertGreaterEqual(len(detected), 1)
        for left, right in zip([index for index, event in enumerate(events) if event], [index for index, event in enumerate(events) if event][1:]):
            self.assertGreaterEqual(right - left, 4)

    def test_buffer_never_exceeds_largest_window(self):
        decoder = StreamingViterbiCTCDecoder(self.contract, window_lengths=(3, 11), threshold=100.0, beta=0.0, cooldown_frames=0)
        for frame in np.zeros((100, len(self.contract["tokens"]))) :
            decoder.step(frame)
            self.assertLessEqual(decoder.buffered_frames, 11)

    def test_malformed_input_fails_closed(self):
        with self.assertRaises(ValueError):
            StreamingViterbiCTCDecoder(self.contract, window_lengths=(0,), threshold=0.0, beta=0.0, cooldown_frames=0)
        decoder = StreamingViterbiCTCDecoder(self.contract, window_lengths=(3,), threshold=0.0, beta=0.0, cooldown_frames=0)
        with self.assertRaises(ValueError): decoder.step([0.0])
        with self.assertRaises(ValueError): decoder.step([float("nan")] * len(self.contract["tokens"]))
        with self.assertRaises(ValueError): exhaustive_suffix_score(np.zeros((2, len(self.contract["tokens"]) - 1)), self.contract, window_lengths=(2,), beta=0.0)


if __name__ == "__main__":
    unittest.main()
