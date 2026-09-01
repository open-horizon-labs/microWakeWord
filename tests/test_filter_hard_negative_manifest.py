import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools.filter_hard_negative_manifest import filter_manifest


class FilterHardNegativeManifestTest(unittest.TestCase):
    def test_filters_exact_archives_and_preserves_random_reserve(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "training" / "source_mmap"
            archive.mkdir(parents=True)
            (archive / "data.ninja").write_bytes(b"source")
            digest = __import__("hashlib").sha256(b"source").hexdigest()
            manifest = root / "input.jsonl"
            rows = [
                {
                    "source_path": str(archive),
                    "source_hash": digest,
                    "reason": "high_score:0.9-1.01",
                    "split": "training",
                },
                {
                    "source_path": str(archive),
                    "source_hash": digest,
                    "reason": "random_reserve",
                    "split": "training",
                },
            ]
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
            output = root / "filtered"
            with (
                mock.patch(
                    "tools.filter_hard_negative_manifest._write_records",
                    side_effect=[4, 1],
                ) as write,
                mock.patch("tools.filter_hard_negative_manifest._write_manifest"),
            ):
                result = filter_manifest(manifest, output, [archive])
            self.assertEqual(write.call_count, 2)
            self.assertEqual(result["selected"], 4)
            self.assertEqual(result["random_reserve"], 1)
            self.assertTrue((output / "mining-metadata.json").is_file())

    def test_rejects_quarantine_and_missing_reserve(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            quarantine = root / "observations" / "false-wakes" / "source_mmap"
            quarantine.mkdir(parents=True)
            (quarantine / "data.ninja").write_bytes(b"source")
            manifest = root / "input.jsonl"
            manifest.write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "quarantined"):
                filter_manifest(manifest, root / "out", [quarantine])


if __name__ == "__main__":
    unittest.main()
