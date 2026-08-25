import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.generate_kizz_kokoro_blends_v3 import TRAIN_VOICES, blend_pairs, generate


class KokoroBlendGenerationTest(unittest.TestCase):
    def test_pairs_are_train_only_and_complete(self):
        pairs = blend_pairs()
        self.assertEqual(len(pairs), len(TRAIN_VOICES) * (len(TRAIN_VOICES) - 1) // 2)
        self.assertTrue(
            all(left in TRAIN_VOICES and right in TRAIN_VOICES for left, right in pairs)
        )
        with self.assertRaisesRegex(ValueError, "held-out"):
            blend_pairs((TRAIN_VOICES[0], "am_onyx"))

    def test_generation_is_provenance_complete_and_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.pth"
            model.write_bytes(b"model")

            def synth(_phones, _voice, _speed):
                return np.zeros(2400, dtype=np.float32), 24_000

            first = generate(
                root / "audio",
                root / "manifest.json",
                model,
                speeds=(1.0,),
                synthesizer=synth,
            )
            second = generate(
                root / "audio",
                root / "manifest.json",
                model,
                speeds=(1.0,),
                synthesizer=lambda *_: self.fail("resume regenerated audio"),
            )
            self.assertEqual(first, second)
            self.assertEqual(len(first["examples"]), len(blend_pairs()))
            self.assertTrue(all(row["split"] == "train" for row in first["examples"]))
            self.assertTrue(all(row["base_voices"] for row in first["examples"]))
            self.assertEqual(
                json.loads((root / "manifest.json").read_text())["contract"][
                    "base_voice_split"
                ],
                "train",
            )


if __name__ == "__main__":
    unittest.main()
