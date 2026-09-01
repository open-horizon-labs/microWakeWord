import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from tools.freeze_kizz_fma_small_negative_assets import freeze


class FreezeFmaTests(unittest.TestCase):
    def test_binds_archive_before_scanning_audio(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tracks = root / "tracks.csv"
            tracks.write_text("not reached\n", encoding="utf-8")
            archive = root / "fma_small.zip"
            archive.write_bytes(b"archive")
            calls = []

            def binding(path):
                calls.append(Path(path).name)
                if Path(path).resolve() == archive.resolve():
                    raise OSError("volume disconnected")
                return {"path": str(path), "sha256": "0" * 64, "bytes": 1}

            with mock.patch(
                "tools.freeze_kizz_fma_small_negative_assets._binding",
                side_effect=binding,
            ):
                with self.assertRaisesRegex(OSError, "disconnected"):
                    freeze(tracks, root / "audio", archive, root / "output.json")

            self.assertEqual(calls, ["tracks.csv", "fma_small.zip"])

    def test_freezes_train_validation_and_preserves_official_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "audio"
            tracks = root / "tracks.csv"
            archive = root / "fma_small.zip"
            archive.write_bytes(b"bound archive")
            groups = ["", "album", "artist", "set", "set", "track", "track"]
            fields = ["", "id", "id", "split", "subset", "genre_top", "license"]
            rows = [
                [2, 20, 200, "training", "small", "Rock", "CC BY"],
                [3, 30, 300, "validation", "small", "Jazz", "CC BY-SA"],
                [4, 40, 400, "test", "small", "Pop", "CC BY"],
                [5, 50, 500, "training", "small", "Rock", "CC BY"],
                [6, 60, 600, "training", "small", "Rock", "CC BY"],
                [7, 60, 700, "validation", "small", "Jazz", "CC BY"],
                [8, 70, 800, "training", "small", "Rock", "CC BY"],
                [9, 70, 900, "test", "small", "Pop", "CC BY"],
            ]
            with tracks.open("w", newline="", encoding="utf-8") as target:
                writer = csv.writer(target)
                writer.writerow(groups)
                writer.writerow(fields)
                writer.writerow(["track_id", "", "", "", "", "", ""])
                writer.writerows(rows)
            for track_id in (2, 3):
                stem = f"{track_id:06d}"
                path = audio / stem[:3] / f"{stem}.mp3"
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, np.zeros(1600, dtype=np.float32), 16_000, format="WAV")
            corrupt = audio / "000" / "000005.mp3"
            corrupt.write_bytes(b"not audio")
            output = root / "manifest.json"

            result = freeze(tracks, audio, archive, output)

            self.assertEqual(result["by_split"], {"train": 1, "validation": 1})
            self.assertEqual(result["preserved_official_test_tracks"], 2)
            self.assertEqual(result["quarantined_audio_files"], 1)
            self.assertEqual(
                result["excluded_unlicensed_by_split"],
                {"train": 0, "validation": 0},
            )
            self.assertEqual(
                result["excluded_cross_split_album_by_split"],
                {"train": 2, "validation": 1},
            )
            payload = json.loads(output.read_text())
            self.assertEqual({row["split"] for row in payload["examples"]}, {"train", "validation"})
            self.assertTrue(payload["partition"]["official_test_preserved_unread"])
            self.assertEqual(payload["partition"]["cross_split_albums"], 2)

    def test_rejects_artist_overlap_across_official_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "audio"
            tracks = root / "tracks.csv"
            archive = root / "fma_small.zip"
            archive.write_bytes(b"archive")
            with tracks.open("w", newline="", encoding="utf-8") as target:
                writer = csv.writer(target)
                writer.writerow(["", "album", "artist", "set", "set", "track", "track"])
                writer.writerow(["", "id", "id", "split", "subset", "genre_top", "license"])
                writer.writerow(["track_id", "", "", "", "", "", ""])
                writer.writerows(
                    [
                        [2, 20, 200, "training", "small", "Rock", "CC BY"],
                        [3, 30, 200, "validation", "small", "Rock", "CC BY"],
                        [4, 40, 400, "test", "small", "Rock", "CC BY"],
                    ]
                )
            for track_id in (2, 3):
                stem = f"{track_id:06d}"
                path = audio / stem[:3] / f"{stem}.mp3"
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, np.zeros(1600), 16_000, format="WAV")
            with self.assertRaisesRegex(ValueError, "artist overlap"):
                freeze(tracks, audio, archive, root / "manifest.json")


if __name__ == "__main__":
    unittest.main()
