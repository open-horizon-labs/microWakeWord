import math
import unittest

import numpy as np

from microwakeword.ordered_state import (
    KIZZ_TOPOLOGY,
    OrderedStateDecoder,
    OrderedStateEvent,
    OrderedStateMarginSweepDecoder,
)


def probability_frame(state, *, background=0.01, strength=0.98):
    values = np.full(
        KIZZ_TOPOLOGY.state_count,
        (1.0 - strength - background) / (KIZZ_TOPOLOGY.state_count - 2),
    )
    values[KIZZ_TOPOLOGY.background_index] = background
    values[KIZZ_TOPOLOGY.ordered_state_index(state)] = strength
    return values


def assert_event_equal(test_case, expected, actual):
    if expected is None or actual is None:
        test_case.assertIs(expected, actual)
        return
    test_case.assertIsInstance(actual, OrderedStateEvent)
    test_case.assertEqual(expected.start_frame, actual.start_frame)
    test_case.assertEqual(expected.end_frame, actual.end_frame)
    test_case.assertEqual(expected.score, actual.score)
    test_case.assertEqual(expected.rejection_score, actual.rejection_score)


class OrderedStateMarginSweepDecoderTest(unittest.TestCase):
    def assert_matches_independent_decoders(
        self, frames, margins, *, from_logits, cooldown_frames, evidence_floor, indices
    ):
        sweep = OrderedStateMarginSweepDecoder(
            completion_margins=margins,
            from_logits=from_logits,
            cooldown_frames=cooldown_frames,
            state_evidence_floor=evidence_floor,
            self_loop_probability=0.71,
            next_state_probability=0.29,
        )
        decoders = [
            OrderedStateDecoder(
                completion_margin=margin,
                from_logits=from_logits,
                cooldown_frames=cooldown_frames,
                state_evidence_floor=evidence_floor,
                self_loop_probability=0.71,
                next_state_probability=0.29,
            )
            for margin in margins
        ]
        for frame, frame_index in zip(frames, indices):
            expected = [decoder.step(frame, frame_index) for decoder in decoders]
            actual = sweep.step(frame, frame_index)
            for expected_event, actual_event in zip(expected, actual):
                assert_event_equal(self, expected_event, actual_event)
            np.testing.assert_array_equal(
                sweep.current_completion_scores,
                [decoder.current_completion_score for decoder in decoders],
            )
            np.testing.assert_array_equal(
                sweep.cooldown_remaining,
                [decoder.cooldown_remaining for decoder in decoders],
            )
            for expected_decoder, actual_scores in zip(decoders, sweep.scores):
                np.testing.assert_array_equal(expected_decoder.scores, actual_scores)

    def test_randomized_equivalence_for_probabilities_and_logits(self):
        rng = np.random.default_rng(241)
        margins = (-2.0, 0.0, 0.75, 3.0)
        probabilities = rng.dirichlet(np.ones(KIZZ_TOPOLOGY.state_count), size=140)
        logits = rng.normal(size=probabilities.shape)
        indices = np.arange(100, 100 + len(probabilities))
        for frames, from_logits in ((probabilities, False), (logits, True)):
            self.assert_matches_independent_decoders(
                frames,
                margins,
                from_logits=from_logits,
                cooldown_frames=7,
                evidence_floor=-0.2,
                indices=indices,
            )

    def test_trigger_outputs_scores_and_coordinates_match(self):
        margins = (-10.0, -2.0, 0.0, 2.0)
        frames = [probability_frame(state) for state in range(21)]
        self.assert_matches_independent_decoders(
            frames + frames,
            margins,
            from_logits=False,
            cooldown_frames=4,
            evidence_floor=0.0,
            indices=list(range(50, 92)),
        )

    def test_reset_and_rearm_match_independent_decoders(self):
        margins = (-1.0, 1.0)
        sweep = OrderedStateMarginSweepDecoder(
            completion_margins=margins, from_logits=False
        )
        decoders = [
            OrderedStateDecoder(completion_margin=margin, from_logits=False)
            for margin in margins
        ]
        first = probability_frame(0)
        sweep.step(first, 20)
        for decoder in decoders:
            decoder.step(first, 20)
        sweep.reset(90)
        for decoder in decoders:
            decoder.reset(90)
        for index, frame in enumerate(
            [probability_frame(state) for state in range(21)], 90
        ):
            expected = [decoder.step(frame, index) for decoder in decoders]
            actual = sweep.step(frame, index)
            for expected_event, actual_event in zip(expected, actual):
                assert_event_equal(self, expected_event, actual_event)
            np.testing.assert_array_equal(
                sweep.current_completion_scores,
                [decoder.current_completion_score for decoder in decoders],
            )

        sweep.rearm()
        for decoder in decoders:
            decoder.rearm()
        self.assertEqual(sweep._frame_index, decoders[0]._frame_index)
        self.assertEqual(sweep._frame_index, 111)
        for offset, frame in enumerate([probability_frame(0), probability_frame(1)]):
            expected = [decoder.step(frame) for decoder in decoders]
            actual = sweep.step(frame)
            for expected_event, actual_event in zip(expected, actual):
                assert_event_equal(self, expected_event, actual_event)
            np.testing.assert_array_equal(
                sweep.current_completion_scores,
                [decoder.current_completion_score for decoder in decoders],
            )

    def test_rearm_after_trigger_preserves_coordinate_and_cooldown_independence(self):
        margins = (-10.0, 10.0)
        sweep = OrderedStateMarginSweepDecoder(
            completion_margins=margins, from_logits=False, cooldown_frames=3
        )
        decoders = [
            OrderedStateDecoder(
                completion_margin=margin, from_logits=False, cooldown_frames=3
            )
            for margin in margins
        ]
        for index, frame in enumerate(
            [probability_frame(state) for state in range(21)]
        ):
            expected = [decoder.step(frame, 10 + index) for decoder in decoders]
            actual = sweep.step(frame, 10 + index)
            for expected_event, actual_event in zip(expected, actual):
                assert_event_equal(self, expected_event, actual_event)
        sweep.rearm()
        for decoder in decoders:
            decoder.rearm()
        self.assertEqual(sweep.cooldown_remaining.tolist(), [0, 0])
        self.assertEqual(sweep._frame_index, 31)

    def test_rejects_empty_nonfinite_and_nonsequence_margins(self):
        for margins in ([], [math.nan], [math.inf], [-math.inf], [[1.0]]):
            with self.subTest(margins=margins):
                with self.assertRaises(ValueError):
                    OrderedStateMarginSweepDecoder(completion_margins=margins)
        with self.assertRaises(ValueError):
            OrderedStateMarginSweepDecoder(completion_margins=1.0)

    def test_frame_index_cannot_move_backwards(self):
        decoder = OrderedStateMarginSweepDecoder(completion_margins=[0.0])
        decoder.step(np.zeros(KIZZ_TOPOLOGY.state_count), 10)
        with self.assertRaises(ValueError):
            decoder.step(np.ones(KIZZ_TOPOLOGY.state_count), 9)


if __name__ == "__main__":
    unittest.main()
