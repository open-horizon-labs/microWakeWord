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
    values = np.full(KIZZ_TOPOLOGY.state_count, (1.0 - strength - background) / (KIZZ_TOPOLOGY.state_count - 2))
    values[KIZZ_TOPOLOGY.background_index] = background
    values[state] = strength
    return values


class OrderedStateDecoderTest(unittest.TestCase):
    def test_declares_seven_phones_three_states_and_rejection_outputs(self):
        self.assertEqual(KIZZ_TOPOLOGY.ordered_state_count, 21)
        self.assertEqual(KIZZ_TOPOLOGY.state_count, 23)
        self.assertEqual(KIZZ_TOPOLOGY.silence_index, 21)
        self.assertEqual(KIZZ_TOPOLOGY.background_index, 22)

    def test_legal_progression_emits_coordinates(self):
        decoder = OrderedStateDecoder(KIZZ_TOPOLOGY, completion_margin=0.0)
        frames = [frame(0), frame(0), frame(1), frame(2)]
        frames += [frame(index) for index in range(3, 21)]
        events = decoder.decode(frames, start_frame=10)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].start_frame, 10)
        self.assertEqual(events[0].end_frame, 31)
        self.assertGreater(events[0].score, 0.0)

    def test_skip_and_reorder_do_not_complete(self):
        decoder = OrderedStateDecoder(KIZZ_TOPOLOGY)
        skipped = [frame(0), frame(2)] + [frame(index) for index in range(3, 21)]
        reordered = [frame(0), frame(1), frame(3), frame(2)] + [frame(index) for index in range(4, 21)]
        self.assertEqual(decoder.decode(skipped), [])
        self.assertEqual(decoder.decode(reordered), [])

    def test_background_competes_with_partial_phrase(self):
        decoder = OrderedStateDecoder(KIZZ_TOPOLOGY, completion_margin=1.0)
        frames = [frame(index, background=0.8, strength=0.15) for index in range(21)]
        self.assertEqual(decoder.decode(frames), [])

    def test_reset_discards_partial_progress(self):
        decoder = OrderedStateDecoder(KIZZ_TOPOLOGY)
        decoder.step(frame(0))
        decoder.reset(50)
        frames = [frame(index) for index in range(1, 21)]
        self.assertEqual(decoder.decode(frames, start_frame=50), [])

    def test_cooldown_suppresses_immediate_second_event(self):
        decoder = OrderedStateDecoder(KIZZ_TOPOLOGY, cooldown_frames=30)
        phrase = [frame(index) for index in range(21)]
        events = decoder.decode(phrase + phrase, start_frame=0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].end_frame, 20)

    def test_batch_and_stream_are_equivalent(self):
        rng = np.random.default_rng(7)
        frames = rng.dirichlet(np.ones(KIZZ_TOPOLOGY.state_count), size=100)
        batch_decoder = OrderedStateDecoder(KIZZ_TOPOLOGY, completion_margin=2.0)
        stream_decoder = OrderedStateDecoder(KIZZ_TOPOLOGY, completion_margin=2.0)
        batch = batch_decoder.decode(frames, start_frame=4)
        stream = [event for index, item in enumerate(frames) if (event := stream_decoder.step(item, 4 + index))]
        self.assertEqual(batch, stream)


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
            np.asarray([[frame(index, background=0.02) for index in range(21)]], dtype=np.float32)
        )
        tf_score = ordered_state_sequence_score(logits, from_logits=False).numpy()
        np_score = ordered_state_sequence_score_numpy(logits.numpy(), from_logits=False)
        np.testing.assert_allclose(tf_score, np_score, rtol=1e-5, atol=1e-5)

    def test_sequence_loss_supports_optional_aligned_targets(self):
        if self.tf is None:
            self.skipTest("TensorFlow is not installed")
        logits = self.tf.zeros([2, 4, KIZZ_TOPOLOGY.state_count])
        labels = self.tf.constant([1.0, 0.0])
        targets = self.tf.zeros([2, 4], dtype=self.tf.int32)
        loss = ordered_state_sequence_loss(
            logits, labels, frame_state_targets=targets, sequence_weight=1.0, frame_weight=0.25
        )
        self.assertTrue(np.isfinite(float(loss.numpy())))


if __name__ == "__main__":
    unittest.main()
