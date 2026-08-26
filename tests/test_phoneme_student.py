import unittest
from types import SimpleNamespace

import numpy as np

from microwakeword.phoneme_student import (
    StreamingPhonemeDecoder,
    compact_phone_contract,
    resample_log_posteriors,
    student_output_times_seconds,
)


class PhonemeStudentTests(unittest.TestCase):
    def test_compact_contract_preserves_repeated_phone_and_collisions(self):
        contract = compact_phone_contract()
        self.assertEqual(contract["canonical_path"][0], contract["canonical_path"][3])
        self.assertEqual(contract["tokens"][0], "<blank>")
        self.assertEqual(contract["tokens"][-1], "OTHER")
        self.assertEqual(len(contract["collision_paths"]), 9)

    def test_output_timing_is_receptive_field_derived(self):
        flags = SimpleNamespace(
            first_conv_filters=48,
            first_conv_kernel_size=5,
            stride=3,
            repeat_in_block="1,1,1,1",
            mixconv_kernel_sizes="[3], [5], [7], [9]",
        )
        times = student_output_times_seconds(flags, 66)
        self.assertAlmostEqual(times[0], 0.655)
        self.assertAlmostEqual(times[1] - times[0], 0.030)
        self.assertAlmostEqual(times[-1], 2.605)

    def test_resampling_preserves_mass(self):
        source = np.log(np.asarray([[0.8, 0.2], [0.2, 0.8]], dtype=np.float32))
        result = resample_log_posteriors(
            source,
            teacher_frame_center_seconds=0.01,
            teacher_frame_stride_seconds=0.02,
            student_times_seconds=(0.01, 0.02, 0.03),
        )
        self.assertTrue(np.allclose(np.exp(result).sum(axis=1), 1.0))
        self.assertTrue(np.allclose(np.exp(result[1]), [0.5, 0.5]))

    def test_streaming_decoder_requires_canonical_over_collision(self):
        contract = {
            "tokens": ["<blank>", "a", "b", "c", "OTHER"],
            "blank_id": 0,
            "canonical_path": [1, 2],
            "collision_paths": {"collision": [1, 3]},
        }
        decoder = StreamingPhonemeDecoder(
            contract,
            window_lengths=(4,),
            threshold=-1.0,
            beta=0.0,
            cooldown_frames=2,
        )
        frames = [
            [4, 0, 0, 0, 0],
            [0, 4, 0, 0, 0],
            [0, 0, 4, 0, 0],
            [4, 0, 0, 0, 0],
        ]
        events = [decoder.step(frame) for frame in frames]
        self.assertTrue(any(event is not None for event in events))
        collision = StreamingPhonemeDecoder(
            contract,
            window_lengths=(4,),
            threshold=-1.0,
            beta=0.0,
            cooldown_frames=2,
        )
        collision_frames = [frames[0], frames[1], [0, 0, 0, 4, 0], frames[3]]
        self.assertTrue(all(collision.step(frame) is None for frame in collision_frames))


if __name__ == "__main__":
    unittest.main()
