import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from microwakeword.ordered_state import OrderedStateTopology
from microwakeword.wake_phrase import KIZZ_CONTROL
from tools.extend_kizz_detector_student_cache_with_device_replays import (
    _materialize_rows,
)


class ExtendKizzDetectorStudentCacheTests(unittest.TestCase):
    def test_device_materialization_is_deterministic_and_keeps_phone_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "capture.wav"
            values = np.zeros(80_000, dtype=np.float32)
            values[12_000:28_000] = 0.1 * np.sin(
                2 * np.pi * 300 * np.arange(16_000, dtype=np.float32) / 16_000
            )
            sf.write(audio, values, 16_000, subtype="PCM_16")
            phones = list(KIZZ_CONTROL.phones)
            starts = np.linspace(0.75, 1.55, len(phones), endpoint=False)
            row = {
                "source_id": "device:test",
                "path": str(audio),
                "audio_sha256": "f" * 64,
                "provider": "kokoro",
                "voice": "test",
                "split": "train",
                "phrase_span": {"start_s": 0.75, "end_s": 1.65},
                "phone_spans": [
                    {
                        "phone": phone,
                        "start_s": float(start),
                        "end_s": float(start + 0.08),
                    }
                    for phone, start in zip(phones, starts, strict=True)
                ],
            }
            topology = OrderedStateTopology(KIZZ_CONTROL.phones, 1)
            first = _materialize_rows(
                [row], topology, replicas=3, seed=42, augment=True
            )
            second = _materialize_rows(
                [row], topology, replicas=3, seed=42, augment=True
            )
            np.testing.assert_array_equal(first[0], second[0])
            np.testing.assert_array_equal(first[1], second[1])
            self.assertEqual(first[2], second[2])
            self.assertTrue(all(0.82 <= row["speed"] <= 1.18 for row in first[2]))
            self.assertEqual(first[0].shape, (3, 260, 40))
            self.assertEqual(first[1].shape, (3, 87))
            for state in range(2, topology.state_count):
                self.assertIn(state, first[1][0])


if __name__ == "__main__":
    unittest.main()
