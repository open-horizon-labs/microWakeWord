import csv
import subprocess
import tempfile
import unittest
from pathlib import Path

from scipy.io import wavfile
import numpy as np

from tools.prepare_background_corpus import prepare


class PrepareBackgroundCorpusTest(unittest.TestCase):
    def test_separates_training_and_stress_by_source_fold(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            esc50 = root / "esc50"
            (esc50 / "audio").mkdir(parents=True)
            (esc50 / "meta").mkdir()
            rows = [
                ("indoor.wav", 1, "keyboard_typing"),
                ("outdoor.wav", 5, "rain"),
                ("unused.wav", 1, "dog"),
            ]
            with (esc50 / "meta/esc50.csv").open("w", newline="") as target:
                writer = csv.writer(target)
                writer.writerow(["filename", "fold", "target", "category"])
                for index, (filename, fold, category) in enumerate(rows):
                    writer.writerow([filename, fold, index, category])
                    wavfile.write(
                        esc50 / "audio" / filename,
                        16000,
                        np.zeros(160, dtype=np.int16),
                    )
            subprocess.run(["git", "init", "-q"], cwd=esc50, check=True)
            subprocess.run(["git", "add", "."], cwd=esc50, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.com",
                    "commit",
                    "-qm",
                    "fixture",
                ],
                cwd=esc50,
                check=True,
            )

            manifest = prepare(root / "backgrounds", esc50=esc50)

            self.assertEqual(
                manifest["counts"],
                {
                    "indoor": {"train": 1, "stress": 0},
                    "outdoor": {"train": 0, "stress": 1},
                },
            )
            self.assertTrue((root / "backgrounds/indoor/train/indoor.wav").exists())
            self.assertTrue((root / "backgrounds/outdoor/stress/outdoor.wav").exists())
            self.assertFalse((root / "backgrounds/unused.wav").exists())


if __name__ == "__main__":
    unittest.main()
