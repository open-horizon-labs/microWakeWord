import csv
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scipy.io import wavfile
import numpy as np

from tools.prepare_background_corpus import prepare


class PrepareBackgroundCorpusTest(unittest.TestCase):
    def make_esc50(self, root, rows):
        esc50 = root / "esc50"
        (esc50 / "audio").mkdir(parents=True)
        (esc50 / "meta").mkdir()
        with (esc50 / "meta/esc50.csv").open("w", newline="") as target:
            writer = csv.writer(target)
            writer.writerow(
                ["filename", "fold", "target", "category", "esc10", "src_file", "take"]
            )
            for index, (filename, fold, category, source_file_id) in enumerate(rows):
                writer.writerow(
                    [filename, fold, index, category, "False", source_file_id, "A"]
                )
                wavfile.write(
                    esc50 / "audio" / filename,
                    16000,
                    np.full(160, index + 1, dtype=np.int16),
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
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-qm",
                "fixture",
            ],
            cwd=esc50,
            check=True,
        )
        return esc50

    def make_device_corpus(self, root):
        corpus = root / "device-corpus"
        (corpus / "audio").mkdir(parents=True)
        audio = corpus / "audio/ambient.wav"
        wavfile.write(audio, 16000, np.full(160, 7, dtype=np.int16))
        audio_hash = hashlib.sha256(audio.read_bytes()).hexdigest()
        payload = {
            "schema_version": 2,
            "corpus_id": "ambient-fixture",
            "device_profiles": {
                "stackchan": {
                    "audio": {
                        "sample_rate": 16000,
                        "channels": 1,
                        "sample_format": "s16le",
                        "frontend": "m5unified_mic",
                        "gain_profile": "fixture",
                        "preprocessing": {},
                    }
                }
            },
            "speakers": {
                "room": {
                    "kind": "ambient",
                    "age_group": "not_applicable",
                    "split": "train",
                }
            },
            "captures": [
                {
                    "capture_id": "ambient-1",
                    "truth": "ambient_negative",
                    "source": "ambient",
                    "split": "train",
                    "speaker_id": "room",
                    "session_id": "room-session",
                    "device_id": "stackchan-1",
                    "device_profile": "stackchan",
                    "phrase": "ambient",
                    "pronunciation": "room-tone",
                    "detected": False,
                    "path": "audio/ambient.wav",
                    "samples": 160,
                    "sha256": audio_hash,
                }
            ],
        }
        (corpus / "device-corpus.json").write_text(json.dumps(payload))
        return corpus

    def test_separates_source_groups_into_train_validation_and_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            esc50 = self.make_esc50(
                root,
                [
                    ("indoor.wav", 1, "keyboard_typing", "indoor-source"),
                    ("validation.wav", 4, "clock_tick", "validation-source"),
                    ("outdoor.wav", 5, "rain", "outdoor-source"),
                    ("unused.wav", 1, "dog", "unused-source"),
                ],
            )

            manifest = prepare(root / "backgrounds", esc50=esc50)

            self.assertEqual(
                manifest["counts"],
                {
                    "indoor": {"train": 1, "validation": 1, "test": 0},
                    "outdoor": {"train": 0, "validation": 0, "test": 1},
                },
            )
            self.assertTrue((root / "backgrounds/indoor/train/indoor.wav").exists())
            self.assertTrue(
                (root / "backgrounds/indoor/validation/validation.wav").exists()
            )
            self.assertTrue((root / "backgrounds/outdoor/test/outdoor.wav").exists())
            self.assertFalse((root / "backgrounds/unused.wav").exists())
            eligibility = {
                row["split"]: row["training_eligible"]
                for row in manifest["examples"]
            }
            self.assertEqual(
                eligibility,
                {"train": True, "validation": False, "test": False},
            )

    def test_cross_fold_source_file_moves_whole_group_to_most_held_out_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            esc50 = self.make_esc50(
                root,
                [
                    ("shared-train.wav", 1, "clock_alarm", "shared"),
                    ("shared-validation.wav", 4, "clock_tick", "shared"),
                    ("shared-test-excluded.wav", 5, "dog", "shared"),
                    ("train.wav", 2, "keyboard_typing", "train-only"),
                ],
            )

            manifest = prepare(root / "backgrounds", esc50=esc50)

            shared_files = [
                row for row in manifest["files"] if row.get("source_file_id") == "shared"
            ]
            self.assertEqual(len(shared_files), 2)
            self.assertEqual({row["evidence_split"] for row in shared_files}, {"test"})
            shared_examples = [
                row
                for row in manifest["examples"]
                if row.get("source_file_id") == "shared"
            ]
            self.assertEqual({row["split"] for row in shared_examples}, {"test"})
            self.assertTrue(
                all(row["training_eligible"] is False for row in shared_examples)
            )

    def test_emits_compatible_files_and_canonical_examples_with_live_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            esc50 = self.make_esc50(
                root,
                [("background.wav", 2, "washing_machine", "source-42")],
            )
            device = self.make_device_corpus(root)

            manifest = prepare(root / "backgrounds", esc50=esc50, device_corpus=device)

            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(len(manifest["files"]), 2)
            esc_file = next(row for row in manifest["files"] if row["source"] == "esc50")
            self.assertTrue(
                {
                    "path",
                    "source",
                    "source_path",
                    "source_split",
                    "source_file_id",
                    "environment",
                    "evidence_split",
                    "category",
                    "sha256",
                }.issubset(esc_file)
            )
            self.assertEqual(esc_file["source_file_id"], "source-42")

            required = {
                "path",
                "audio_sha256",
                "duration_seconds",
                "label",
                "source_group",
                "source_id",
                "provenance_id",
                "parent_id",
                "ancestry_id",
                "speaker_id",
                "session_id",
                "split",
                "training_eligible",
            }
            self.assertTrue(all(required.issubset(row) for row in manifest["examples"]))
            for row in manifest["examples"]:
                path = Path(row["path"])
                self.assertTrue(path.is_absolute())
                self.assertEqual(
                    row["audio_sha256"], hashlib.sha256(path.read_bytes()).hexdigest()
                )
                self.assertAlmostEqual(row["duration_seconds"], 0.01)
                self.assertEqual(row["label"], 0)
                self.assertEqual(row["source_group"], "background_noise")
                self.assertTrue(row["training_eligible"])
            device_example = next(
                row for row in manifest["examples"] if row["source"] == "device_corpus"
            )
            self.assertEqual(device_example["split"], "train")
            self.assertEqual(device_example["speaker_id"], "room")
            self.assertEqual(device_example["session_id"], "room-session")

    def test_manifest_is_byte_deterministic_and_rejects_stale_destination_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            esc50 = self.make_esc50(
                root,
                [
                    ("train.wav", 1, "keyboard_typing", "stable-train"),
                    ("validation.wav", 4, "clock_tick", "stable-validation"),
                    ("test.wav", 5, "rain", "stable-test"),
                ],
            )
            output = root / "backgrounds"

            prepare(output, esc50=esc50)
            first = (output / "background-corpus.json").read_bytes()
            first_hash = hashlib.sha256(first).hexdigest()

            metadata = esc50 / "meta/esc50.csv"
            lines = metadata.read_text().splitlines()
            metadata.write_text("\n".join([lines[0], *reversed(lines[1:])]) + "\n")
            prepare(output, esc50=esc50)
            second = (output / "background-corpus.json").read_bytes()
            self.assertEqual(second, first)
            self.assertEqual(hashlib.sha256(second).hexdigest(), first_hash)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            esc50 = self.make_esc50(
                root,
                [("background.wav", 1, "keyboard_typing", "stable-source")],
            )
            destination = root / "backgrounds/indoor/train/background.wav"
            destination.parent.mkdir(parents=True)
            wavfile.write(destination, 16000, np.full(160, 99, dtype=np.int16))
            with self.assertRaisesRegex(ValueError, "existing background differs"):
                prepare(root / "backgrounds", esc50=esc50)


if __name__ == "__main__":
    unittest.main()
