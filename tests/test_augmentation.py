import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from microwakeword.audio.augmentation import Augmentation


class AugmentationTest(unittest.TestCase):
    def test_positive_snr_keeps_background_below_speech(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample_rate = 16000
            time = np.arange(sample_rate, dtype=np.float32) / sample_rate
            speech = 0.1 * np.sin(2 * np.pi * 440 * time)
            rng = np.random.default_rng(231)
            background = rng.normal(0, 0.1, sample_rate).astype(np.float32)
            wavfile.write(
                root / "background.wav",
                sample_rate,
                np.clip(background * 32767, -32768, 32767).astype(np.int16),
            )
            augmenter = Augmentation(
                augmentation_probabilities={"AddBackgroundNoise": 1.0},
                background_paths=[str(root)],
                background_min_snr_db=6,
                background_max_snr_db=6,
            )

            mixed = augmenter.augment_clip(speech.copy())
            added = mixed - speech
            observed_snr = 20 * math.log10(
                np.sqrt(np.mean(np.square(speech))) / np.sqrt(np.mean(np.square(added)))
            )

            self.assertAlmostEqual(observed_snr, 6.0, delta=0.25)


if __name__ == "__main__":
    unittest.main()
