import json
from pathlib import Path
import tempfile
import unittest
import wave

from tools.prepare_ordered_state_experiment import prepare


def write_wav(path: Path, seconds: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * int(16000 * seconds))


class PrepareOrderedStateExperimentTests(unittest.TestCase):
    def make_config(self, root: Path, sources: list[dict], **extra) -> Path:
        config = root / "experiment.json"
        config.write_text(
            json.dumps({"experiment": "test", "sources": sources, **extra})
        )
        return config

    def source(
        self,
        root: Path,
        source_id: str,
        split: str,
        speaker: str,
        session: str,
        category: str = "speech",
    ):
        source_root = root / source_id
        write_wav(source_root / "nested" / "clip.wav", 1.25)
        return {
            "source_id": source_id,
            "path": str(source_root),
            "split": split,
            "truth": "negative",
            "category": category,
            "speaker_id": speaker,
            "session_id": session,
        }

    def test_freezes_hashes_exposure_and_disjoint_manifest_without_copying(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = [
                self.source(
                    root,
                    "piper-train",
                    "train",
                    "piper-1",
                    "piper-s1",
                    "speech/TV",
                ),
                self.source(
                    root,
                    "piper-validation",
                    "validation",
                    "piper-2",
                    "piper-s2",
                    "music",
                ),
                self.source(
                    root,
                    "piper-test",
                    "test",
                    "piper-3",
                    "piper-s3",
                    "non-speech",
                ),
            ]
            output = root / "frozen" / "manifest.json"
            manifest = prepare(self.make_config(root, sources), output)

            self.assertTrue(output.is_file())
            self.assertEqual(manifest["counts"]["files"], 3)
            self.assertAlmostEqual(manifest["counts"]["exposure_seconds"], 3.75)
            self.assertEqual(
                manifest["counts"]["exposure_seconds_by_split_and_category"],
                {
                    "test": {"non-speech": 1.25},
                    "train": {"speech/TV": 1.25},
                    "validation": {"music": 1.25},
                },
            )
            self.assertEqual(manifest["sources"][0]["file_count"], 1)
            self.assertEqual(manifest["files"][0]["source_id"], "piper-test")
            self.assertTrue(manifest["files"][0]["sha256"])
            self.assertEqual(manifest["sources"][0]["category"], "non-speech")
            self.assertEqual(manifest["files"][0]["category"], "non-speech")
            self.assertTrue(manifest["inventory_sha256"])
            self.assertTrue(
                all(Path(item["source_root"]).is_dir() for item in manifest["files"])
            )
            self.assertFalse((output.parent / "audio").exists())

    def test_uses_declared_exposure_for_non_wav_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "archive"
            source_root.mkdir()
            (source_root / "features.bin").write_bytes(b"features")
            source = {
                "source_id": "archive",
                "path": str(source_root),
                "split": "train",
                "speaker_id": "archive-speaker",
                "session_id": "archive-session",
                "category": "non-speech",
                "exposure_seconds": 3600,
                "extensions": [".bin"],
            }
            sources = [source]
            for split in ("validation", "test"):
                split_root = root / f"archive-{split}"
                split_root.mkdir()
                (split_root / "features.bin").write_bytes(b"features")
                sources.append(
                    {
                        **source,
                        "source_id": f"archive-{split}",
                        "path": str(split_root),
                        "split": split,
                        "speaker_id": f"archive-speaker-{split}",
                        "session_id": f"archive-session-{split}",
                        "exposure_seconds": 1,
                    }
                )
            manifest = prepare(self.make_config(root, sources), root / "manifest.json")
            self.assertEqual(
                manifest["counts"]["exposure_seconds_by_split"]["train"], 3600
            )
            self.assertTrue(manifest["sources"][0]["exposure_declared"])

    def test_empty_extensions_includes_extensionless_ragged_mmap_payloads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = []
            for split in ("train", "validation", "test"):
                source_root = root / f"ragged-{split}"
                (source_root / "data").mkdir(parents=True)
                (source_root / "data" / "00000").write_bytes(b"payload")
                (source_root / "index").write_bytes(b"index")
                sources.append(
                    {
                        "source_id": f"ragged-{split}",
                        "path": str(source_root),
                        "split": split,
                        "speaker_id": f"speaker-{split}",
                        "session_id": f"session-{split}",
                        "category": "speech",
                        "exposure_seconds": 7200,
                        "extensions": [],
                    }
                )
            manifest = prepare(self.make_config(root, sources), root / "manifest.json")
            self.assertEqual(manifest["counts"]["files"], 6)
            self.assertEqual(
                manifest["counts"]["exposure_seconds_by_split_and_category"]["train"][
                    "speech"
                ],
                7200,
            )
            self.assertEqual(manifest["sources"][0]["file_count"], 2)
            self.assertIn("data/00000", {item["path"] for item in manifest["files"]})

    def test_preserves_optional_channel_and_source_family(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = []
            for split in ("train", "validation", "test"):
                item = self.source(
                    root,
                    f"source-{split}",
                    split,
                    f"speaker-{split}",
                    f"session-{split}",
                )
                item.update(channel="far-field", source_family="voices")
                sources.append(item)
            manifest = prepare(self.make_config(root, sources), root / "manifest.json")
            source = manifest["sources"][0]
            file_record = manifest["files"][0]
            self.assertEqual(source["channel"], "far-field")
            self.assertEqual(source["source_family"], "voices")
            self.assertEqual(file_record["channel"], "far-field")
            self.assertEqual(file_record["source_family"], "voices")

    def test_requires_non_empty_category(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.source(root, "source", "train", "speaker", "session")
            source.pop("category")
            with self.assertRaisesRegex(ValueError, "category"):
                prepare(self.make_config(root, [source]), root / "manifest.json")
            source["category"] = ""
            with self.assertRaisesRegex(ValueError, "category"):
                prepare(self.make_config(root, [source]), root / "manifest.json")

    def test_rejects_quarantined_evidence_path_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            forbidden = root / "observations" / "false-wakes" / "evidence"
            write_wav(forbidden / "wake.wav")
            config = self.make_config(
                root,
                [
                    {
                        "source_id": "false-wake",
                        "path": str(forbidden),
                        "split": "train",
                        "speaker_id": "unknown",
                        "session_id": "session",
                        "category": "speech",
                    }
                ],
            )
            output = root / "manifest.json"
            with self.assertRaisesRegex(ValueError, "forbidden"):
                prepare(config, output)
            self.assertFalse(output.exists())

    def test_rejects_each_forbidden_component_case_insensitively(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for component in ("observations", "false-wakes", "evidence"):
                directory = root / component.upper()
                write_wav(directory / "clip.wav")
                source = self.source(root, "other", "train", "speaker", "session")
                source["path"] = str(directory)
                with self.assertRaisesRegex(ValueError, "forbidden"):
                    prepare(
                        self.make_config(root, [source]), root / f"{component}.json"
                    )

    def test_rejects_identity_leakage_across_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = self.source(root, "train", "train", "same-speaker", "s1")
            validation = self.source(
                root, "validation", "validation", "same-speaker", "s2"
            )
            with self.assertRaisesRegex(ValueError, "speaker_id"):
                prepare(
                    self.make_config(root, [train, validation]), root / "manifest.json"
                )

    def test_rejects_session_leakage_and_wrong_threshold_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = self.source(root, "train", "train", "speaker-1", "same-session")
            test = self.source(root, "test", "test", "speaker-2", "same-session")
            with self.assertRaisesRegex(ValueError, "session_id"):
                prepare(self.make_config(root, [train, test]), root / "manifest.json")
            with self.assertRaisesRegex(ValueError, "threshold_selection_split"):
                prepare(
                    self.make_config(root, [train], threshold_selection_split="test"),
                    root / "wrong.json",
                )

    def test_rejects_missing_or_empty_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = {
                "source_id": "missing",
                "path": str(root / "missing"),
                "split": "train",
                "speaker_id": "speaker",
                "session_id": "session",
                "category": "speech",
            }
            with self.assertRaisesRegex(ValueError, "not a directory"):
                prepare(self.make_config(root, [missing]), root / "manifest.json")
            empty = root / "empty"
            empty.mkdir()
            missing["source_id"] = "empty"
            missing["path"] = str(empty)
            with self.assertRaisesRegex(ValueError, "no configured audio"):
                prepare(self.make_config(root, [missing]), root / "manifest.json")

    def test_requires_all_frozen_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = self.source(root, "train", "train", "speaker", "session")
            with self.assertRaisesRegex(ValueError, "missing"):
                prepare(self.make_config(root, [train]), root / "manifest.json")


if __name__ == "__main__":
    unittest.main()
