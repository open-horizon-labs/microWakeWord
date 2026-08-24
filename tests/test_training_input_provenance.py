from pathlib import Path
import tempfile
import unittest
from unittest import mock

from microwakeword.data import FeatureHandler
from microwakeword.ordered_state_training import validate_expected_file_hashes
from microwakeword.provenance import sha256_file, sha256_path


class TrainingInputProvenanceTest(unittest.TestCase):
    def test_directory_hash_follows_symlinked_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.mkdir()
            data = target / "data.bin"
            data.write_bytes(b"before")
            view = root / "view"
            view.mkdir()
            (view / "source").symlink_to(target, target_is_directory=True)
            before = sha256_path(view)
            data.write_bytes(b"after")
            self.assertNotEqual(before, sha256_path(view))

    def test_directory_hash_rejects_repeated_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.mkdir()
            (target / "data.bin").write_bytes(b"content")
            view = root / "view"
            view.mkdir()
            (view / "one").symlink_to(target, target_is_directory=True)
            (view / "two").symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "repeated or cyclic"):
                sha256_path(view)

    def test_feature_handler_rejects_content_hash_mismatch_before_loading(self):
        config = {
            "features": [
                {
                    "features_dir": "/not/loaded",
                    "expected_path_sha256": "expected",
                    "type": "mmap",
                }
            ]
        }
        with (
            mock.patch("microwakeword.data.sha256_path", return_value="actual"),
            self.assertRaisesRegex(ValueError, "content hash mismatch"),
        ):
            FeatureHandler(config)

    def test_frame_supervision_hashes_are_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            features = root / "features.npy"
            features.write_bytes(b"features")
            expected = {"features.npy": sha256_file(features)}
            validate_expected_file_hashes(root, expected)
            features.write_bytes(b"mutated")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_expected_file_hashes(root, expected)


if __name__ == "__main__":
    unittest.main()
