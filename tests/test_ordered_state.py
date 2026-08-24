import math
import unittest

import numpy as np

from microwakeword.ordered_state import (
    KIZZ_TOPOLOGY,
    OrderedStateDecoder,
    ordered_state_sequence_loss,
    ordered_state_sequence_score,
    ordered_state_sequence_score_numpy,
)


def frame(state, *, background=0.01, strength=0.98):
    values = np.full(
        KIZZ_TOPOLOGY.state_count,
        (1.0 - strength - background) / (KIZZ_TOPOLOGY.state_count - 2),
    )
    values[KIZZ_TOPOLOGY.background_index] = background
    values[KIZZ_TOPOLOGY.ordered_state_index(state)] = strength
    return values


def decoder(**kwargs):
    return OrderedStateDecoder(
        KIZZ_TOPOLOGY,
        from_logits=False,
        state_evidence_floor=0.0,
        **kwargs,
    )


class OrderedStateDecoderTest(unittest.TestCase):
    def test_declares_seven_phones_three_states_and_rejection_outputs(self):
        self.assertEqual(KIZZ_TOPOLOGY.ordered_state_count, 21)
        self.assertEqual(KIZZ_TOPOLOGY.state_count, 23)
        self.assertEqual(KIZZ_TOPOLOGY.background_index, 0)
        self.assertEqual(KIZZ_TOPOLOGY.silence_index, 1)
        self.assertEqual(KIZZ_TOPOLOGY.phone_state_index(0, 0), 2)
        self.assertEqual(KIZZ_TOPOLOGY.phone_state_index(6, 2), 22)

    def test_legal_progression_emits_coordinates(self):
        state_decoder = decoder(completion_margin=0.0)
        frames = [frame(0), frame(0), frame(1), frame(2)]
        frames += [frame(index) for index in range(3, 21)]
        events = state_decoder.decode(frames, start_frame=10)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].start_frame, 10)
        self.assertEqual(events[0].end_frame, 31)
        self.assertGreater(events[0].score, 0.0)

    def test_skip_and_reorder_do_not_complete(self):
        state_decoder = decoder()
        skipped = [frame(0), frame(2)] + [frame(index) for index in range(3, 21)]
        reordered = [frame(0), frame(1), frame(3), frame(2)] + [
            frame(index) for index in range(4, 21)
        ]
        self.assertEqual(state_decoder.decode(skipped), [])
        self.assertEqual(state_decoder.decode(reordered), [])

    def test_background_competes_with_partial_phrase(self):
        state_decoder = decoder(completion_margin=1.0)
        frames = [frame(index, background=0.8, strength=0.15) for index in range(21)]
        self.assertEqual(state_decoder.decode(frames), [])

    def test_reset_discards_partial_progress(self):
        state_decoder = decoder()
        state_decoder.step(frame(0))
        state_decoder.reset(50)
        frames = [frame(index) for index in range(1, 21)]
        self.assertEqual(state_decoder.decode(frames, start_frame=50), [])

    def test_cooldown_suppresses_immediate_second_event(self):
        state_decoder = decoder(cooldown_frames=30)
        phrase = [frame(index) for index in range(21)]
        events = state_decoder.decode(phrase + phrase, start_frame=0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].end_frame, 20)

    def test_cooldown_does_not_accumulate_a_hidden_partial_path(self):
        state_decoder = decoder(cooldown_frames=10)
        phrase = [frame(index) for index in range(21)]
        self.assertIsNotNone(state_decoder.decode(phrase)[0])
        for index in range(10):
            self.assertIsNone(state_decoder.step(frame(index), 21 + index))
        for index in range(10, 21):
            self.assertIsNone(state_decoder.step(frame(index), 21 + index))

    def test_sequence_score_cannot_start_at_an_interior_state(self):
        interior_only = np.asarray(
            [[[value for value in frame(index)]] for index in range(1, 21)]
        ).reshape(1, 20, KIZZ_TOPOLOGY.state_count)
        score = ordered_state_sequence_score_numpy(interior_only, from_logits=False)
        self.assertTrue(np.isneginf(score[0]))

    def test_sequence_score_prefers_the_complete_ordered_path(self):
        ordered = np.asarray([[frame(index) for index in range(21)]])
        reordered = np.asarray(
            [[frame(0), frame(2), frame(1), *[frame(index) for index in range(3, 21)]]]
        )
        ordered_score = ordered_state_sequence_score_numpy(ordered, from_logits=False)[
            0
        ]
        reordered_score = ordered_state_sequence_score_numpy(
            reordered, from_logits=False
        )[0]
        self.assertGreater(ordered_score, reordered_score)

    def test_batch_and_stream_are_equivalent(self):
        rng = np.random.default_rng(7)
        frames = rng.dirichlet(np.ones(KIZZ_TOPOLOGY.state_count), size=100)
        batch_decoder = decoder(completion_margin=2.0)
        stream_decoder = decoder(completion_margin=2.0)
        batch = batch_decoder.decode(frames, start_frame=4)
        stream = [
            event
            for index, item in enumerate(frames)
            if (event := stream_decoder.step(item, 4 + index))
        ]
        self.assertEqual(batch, stream)

    def test_batch_score_matches_streaming_viterbi_for_logits(self):
        rng = np.random.default_rng(231)
        logits = rng.normal(size=(80, KIZZ_TOPOLOGY.state_count))
        expected = ordered_state_sequence_score_numpy(logits[None, ...])[0]
        state_decoder = OrderedStateDecoder(completion_margin=math.inf)
        completions = []
        for item in logits:
            state_decoder.step(item)
            completions.append(state_decoder.current_completion_score)
        self.assertAlmostEqual(expected, max(completions), places=10)

    def test_batch_score_matches_streaming_viterbi_with_evidence_floor(self):
        probabilities = np.asarray([[frame(index) for index in range(21)]])
        expected = ordered_state_sequence_score_numpy(
            probabilities,
            from_logits=False,
            state_evidence_floor=0.0,
        )[0]
        state_decoder = decoder(completion_margin=math.inf)
        completions = []
        for item in probabilities[0]:
            state_decoder.step(item)
            completions.append(state_decoder.current_completion_score)
        self.assertAlmostEqual(expected, max(completions), places=10)


class OrderedStateTensorFlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import tensorflow as tf
        except ImportError:
            cls.tf = None
        else:
            cls.tf = tf

    @unittest.skipUnless(True, "TensorFlow availability is checked at runtime")
    def test_numpy_tensorflow_score_agree_when_tensorflow_is_installed(self):
        if self.tf is None:
            self.skipTest("TensorFlow is not installed")
        logits = self.tf.constant(
            np.asarray(
                [[frame(index, background=0.02) for index in range(21)]],
                dtype=np.float32,
            )
        )
        tf_score = ordered_state_sequence_score(logits, from_logits=False).numpy()
        np_score = ordered_state_sequence_score_numpy(logits.numpy(), from_logits=False)
        np.testing.assert_allclose(tf_score, np_score, rtol=1e-5, atol=1e-5)

    def test_numpy_tensorflow_score_agree_for_random_logits_and_transitions(self):
        if self.tf is None:
            self.skipTest("TensorFlow is not installed")
        logits = (
            np.random.default_rng(238)
            .normal(size=(3, 37, KIZZ_TOPOLOGY.state_count))
            .astype(np.float32)
        )
        arguments = {
            "self_loop_probability": 0.7,
            "next_state_probability": 0.3,
        }
        tf_score = ordered_state_sequence_score(
            self.tf.constant(logits), **arguments
        ).numpy()
        np_score = ordered_state_sequence_score_numpy(logits, **arguments)
        np.testing.assert_allclose(tf_score, np_score, rtol=1e-5, atol=1e-5)

    def test_sequence_loss_supports_optional_aligned_targets(self):
        if self.tf is None:
            self.skipTest("TensorFlow is not installed")
        logits = self.tf.zeros([2, 24, KIZZ_TOPOLOGY.state_count])
        labels = self.tf.constant([1.0, 0.0])
        targets = self.tf.zeros([2, 24], dtype=self.tf.int32)
        loss = ordered_state_sequence_loss(
            logits,
            labels,
            frame_state_targets=targets,
            sequence_weight=1.0,
            frame_weight=0.25,
        )
        self.assertTrue(np.isfinite(float(loss.numpy())))

    def test_sequence_loss_has_finite_nonzero_gradients(self):
        if self.tf is None:
            self.skipTest("TensorFlow is not installed")
        logits = self.tf.Variable(
            self.tf.random.normal([2, 24, KIZZ_TOPOLOGY.state_count], seed=231)
        )
        with self.tf.GradientTape() as tape:
            loss = ordered_state_sequence_loss(logits, [1.0, 0.0])
        gradient = tape.gradient(loss, logits)
        self.assertTrue(np.all(np.isfinite(gradient.numpy())))
        self.assertGreater(
            float(self.tf.reduce_sum(self.tf.abs(gradient)).numpy()), 0.0
        )


if __name__ == "__main__":
    unittest.main()
