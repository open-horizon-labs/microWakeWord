import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
import wave

from tools.freeze_wake_observation_snapshot import freeze_snapshot
from tools.evaluate_observation_manifest import observation_records


class FreezeWakeObservationSnapshotTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.corpus = self.root / "corpus"
        for name in ("wakes", "false-wakes"):
            (self.corpus / "observations" / name).mkdir(parents=True)
        self.output = self.root / "snapshot"

    def tearDown(self):
        self.temporary.cleanup()

    def write_observation(self, directory, name, received_at, payload=b"pcm"):
        base = self.corpus / "observations" / directory / name
        audio_path = base.with_suffix(".wav")
        metadata_path = base.with_suffix(".json")
        with wave.open(str(audio_path), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(payload * 160)
        digest = hashlib.sha256(audio_path.read_bytes()).hexdigest()
        metadata_path.write_text(
            json.dumps(
                {
                    "observation_id": name,
                    "received_at": received_at,
                    "path": f"observations/{directory}/{name}.wav",
                    "sha256": digest,
                }
            )
        )
        return metadata_path, audio_path

    def test_copies_both_directories_and_applies_cutoff(self):
        self.write_observation("wakes", "before", 99)
        self.write_observation("wakes", "at-cutoff", 100)
        self.write_observation("false-wakes", "after", 101)
        self.write_observation("false-wakes", "later", 201)
        manifest = freeze_snapshot(
            self.corpus,
            self.output,
            datetime.fromtimestamp(100, timezone.utc),
            datetime.fromtimestamp(200, timezone.utc),
            "reviewer",
            "all selected observations are deployment evidence",
        )

        self.assertEqual(
            [item["observation_id"] for item in manifest["observations"]],
            ["at-cutoff", "after"],
        )
        self.assertFalse(manifest["training_eligible"])
        self.assertEqual(manifest["source_corpus"], str(self.corpus.resolve()))
        self.assertEqual(manifest["snapshot_root"], str(self.output.resolve()))
        for item in manifest["observations"]:
            self.assertTrue((self.output / item["audio_path"]).is_file())
            self.assertTrue((self.output / item["metadata_path"]).is_file())
            self.assertTrue(item["review"])
        loaded, records = observation_records(self.output / "manifest.json")
        self.assertFalse(loaded["training_eligible"])
        self.assertEqual(len(records), 2)

    def test_snapshot_hashes_are_self_contained_and_sources_unchanged(self):
        metadata_path, audio_path = self.write_observation(
            "false-wakes", "keep", 100, payload=b"audio"
        )
        source_metadata = metadata_path.read_bytes()
        source_audio = audio_path.read_bytes()
        freeze_snapshot(
            self.corpus,
            self.output,
            datetime.fromtimestamp(100, timezone.utc),
            datetime.fromtimestamp(200, timezone.utc),
            "reviewer",
            "review basis",
        )
        item = json.loads((self.output / "manifest.json").read_text())["observations"][
            0
        ]
        copied_audio = self.output / item["audio_path"]
        self.assertEqual(
            item["audio_sha256"], hashlib.sha256(copied_audio.read_bytes()).hexdigest()
        )
        self.assertEqual(source_metadata, metadata_path.read_bytes())
        self.assertEqual(source_audio, audio_path.read_bytes())
        self.assertEqual(source_audio, copied_audio.read_bytes())

    def test_rejects_duplicate_ids_and_manifest_replacement(self):
        self.write_observation("wakes", "one", 100)
        metadata_path, _ = self.write_observation("false-wakes", "one", 101)
        with self.assertRaisesRegex(ValueError, "duplicate observation_id"):
            self._freeze()

        metadata = json.loads(metadata_path.read_text())
        metadata["observation_id"] = "two"
        metadata["path"] = "observations/false-wakes/two.wav"
        metadata_path.rename(metadata_path.with_name("two.json"))
        audio_path = metadata_path.with_suffix(".wav")
        audio_path.rename(audio_path.with_name("two.wav"))
        metadata_path = metadata_path.with_name("two.json")
        metadata_path.write_text(json.dumps(metadata))
        self._freeze()
        with self.assertRaisesRegex(ValueError, "output manifest already exists"):
            self._freeze()

    def test_rejects_path_escape_and_empty_selection(self):
        metadata_path, _ = self.write_observation("wakes", "escape", 100)
        metadata = json.loads(metadata_path.read_text())
        metadata["path"] = "../outside.wav"
        metadata_path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "escapes corpus"):
            self._freeze()

        metadata_path.unlink()
        with self.assertRaisesRegex(ValueError, "selection is empty"):
            self._freeze(since=101)

    def _freeze(self, since=100):
        return freeze_snapshot(
            self.corpus,
            self.output,
            datetime.fromtimestamp(since, timezone.utc),
            datetime.fromtimestamp(200, timezone.utc),
            "reviewer",
            "review basis",
        )


if __name__ == "__main__":
    unittest.main()
