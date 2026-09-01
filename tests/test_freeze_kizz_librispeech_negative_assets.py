import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.freeze_kizz_librispeech_negative_assets import _speaker_split, freeze


class FreezeLibriSpeechTests(unittest.TestCase):
    def test_partition_is_speaker_disjoint_and_excludes_mini_speakers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "full"
            mini = Path(temporary) / "mini"
            for speaker in range(1, 80):
                path = root / str(speaker) / "1" / f"{speaker}-1-0000.flac"
                path.parent.mkdir(parents=True)
                sf.write(path, np.zeros(1600, dtype=np.float32), 16_000)
            (mini / "1").mkdir(parents=True)
            output = Path(temporary) / "manifest.json"

            result = freeze(
                root,
                output,
                validation_fraction=0.25,
                exclude_speaker_root=mini,
            )

            self.assertEqual(result["excluded_files"], 1)
            self.assertGreater(result["by_split"]["train"], 0)
            self.assertGreater(result["by_split"]["validation"], 0)
            import json

            rows = json.loads(output.read_text())["examples"]
            train = {row["speaker_id"] for row in rows if row["split"] == "train"}
            validation = {
                row["speaker_id"] for row in rows if row["split"] == "validation"
            }
            self.assertFalse(train & validation)
            self.assertNotIn("librispeech-speaker:1", train | validation)

    def test_speaker_split_is_deterministic_and_validates_fraction(self):
        self.assertEqual(_speaker_split("42", 0.2), _speaker_split("42", 0.2))
        with self.assertRaisesRegex(ValueError, "validation_fraction"):
            _speaker_split("42", 0.0)


if __name__ == "__main__":
    unittest.main()
