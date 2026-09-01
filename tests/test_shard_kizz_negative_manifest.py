import json
import tempfile
import unittest
from pathlib import Path

from tools.mine_kizz_librispeech_hard_negatives import sha256_file
from tools.shard_kizz_negative_manifest import shard_manifest


class ShardKizzNegativeManifestTests(unittest.TestCase):
    def _manifest(self, root: Path) -> Path:
        path = root / "source.json"
        examples = [
            {
                "source_id": "speaker-a-1",
                "ancestry_id": "speaker-a",
                "duration_seconds": 12.0,
                "split": "train",
                "label": 0,
                "training_eligible": True,
            },
            {
                "source_id": "speaker-a-2",
                "ancestry_id": "speaker-a",
                "duration_seconds": 8.0,
                "split": "train",
                "label": 0,
                "training_eligible": True,
            },
            {
                "source_id": "speaker-b",
                "ancestry_id": "speaker-b",
                "duration_seconds": 17.0,
                "split": "validation",
                "label": 0,
            },
            {
                "source_id": "speaker-c",
                "ancestry_id": "speaker-c",
                "duration_seconds": 11.0,
                "split": "train",
                "label": 0,
                "training_eligible": True,
            },
            {
                "source_id": "locked",
                "ancestry_id": "locked",
                "duration_seconds": 9.0,
                "split": "validation",
                "label": 0,
                "locked_holdout": True,
            },
            {
                "source_id": "positive",
                "ancestry_id": "positive",
                "duration_seconds": 4.0,
                "split": "train",
                "label": 1,
            },
            {
                "source_id": "test",
                "ancestry_id": "test",
                "duration_seconds": 10.0,
                "split": "test",
                "label": 0,
            },
        ]
        path.write_text(
            json.dumps({"schema_version": 7, "kind": "fixture", "examples": examples}),
            encoding="utf-8",
        )
        return path

    def test_shards_deterministically_without_splitting_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._manifest(root)
            first = root / "first"
            second = root / "second"

            result = shard_manifest(source, first, 2)
            shard_manifest(source, second, 2)

            self.assertEqual(result["eligible_examples"], 4)
            self.assertEqual(result["eligible_identities"], 3)
            self.assertEqual(result["duration_seconds"], 48.0)
            self.assertEqual(
                (first / "index.json").read_bytes(),
                (second / "index.json").read_bytes(),
            )
            for index in range(2):
                name = f"shard-{index:03d}-of-002.json"
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())

            payloads = [
                json.loads((first / f"shard-{index:03d}-of-002.json").read_text())
                for index in range(2)
            ]
            seen = {}
            source_ids = set()
            for index, payload in enumerate(payloads):
                self.assertEqual(payload["source_manifest"]["sha256"], sha256_file(source))
                for row in payload["examples"]:
                    source_ids.add(row["source_id"])
                    previous = seen.setdefault(row["ancestry_id"], index)
                    self.assertEqual(previous, index)
            self.assertEqual(
                source_ids,
                {"speaker-a-1", "speaker-a-2", "speaker-b", "speaker-c"},
            )

    def test_rejects_invalid_shard_count_and_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._manifest(root)
            with self.assertRaisesRegex(ValueError, "at least two"):
                shard_manifest(source, root / "one", 1)
            with self.assertRaisesRegex(ValueError, "exceeds"):
                shard_manifest(source, root / "too-many", 4)
            output = root / "existing"
            output.mkdir()
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                shard_manifest(source, output, 2)


if __name__ == "__main__":
    unittest.main()
