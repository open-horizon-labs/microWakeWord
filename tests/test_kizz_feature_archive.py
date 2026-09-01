import tempfile
import unittest
from pathlib import Path

import numpy as np

from microwakeword.kizz_feature_archive import (
    LegacyRaggedFeatureArchive,
    decode_frontend_features,
    open_feature_archive,
)


def write_legacy(root: Path, items: list[np.ndarray]) -> Path:
    root.mkdir()
    for name in ("starts", "ends", "shapes"):
        (root / name).mkdir()
    sizes = np.asarray([item.size for item in items], dtype="<i8")
    starts = np.concatenate([np.asarray([0], dtype="<i8"), np.cumsum(sizes)[:-1]])
    ends = np.cumsum(sizes)
    np.concatenate([item.reshape(-1) for item in items]).astype("<f4").tofile(
        root / "data.ninja"
    )
    starts.tofile(root / "starts" / "data.ninja")
    ends.tofile(root / "ends" / "data.ninja")
    np.asarray([item.shape for item in items], dtype="<i8").tofile(
        root / "shapes" / "data.ninja"
    )
    return root


class KizzFeatureArchiveTest(unittest.TestCase):
    def test_uint16_frontend_bins_decode_to_product_float_scale(self):
        stored = np.asarray([[0, 256] + [512] * 38], dtype=np.uint16)
        decoded = decode_frontend_features(stored)
        self.assertEqual(decoded.dtype, np.float32)
        self.assertAlmostEqual(float(decoded[0, 1]), 10.0)
        self.assertAlmostEqual(float(decoded[0, 2]), 20.0)

    def test_decoder_rejects_unsupported_integer_features(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            decode_frontend_features(np.zeros((2, 40), dtype=np.int16))

    def test_reads_validated_legacy_archive_without_mutating_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "legacy"
            expected = [
                np.arange(120, dtype=np.float32).reshape(3, 40),
                np.full((2, 40), 7, dtype=np.float32),
            ]
            write_legacy(root, expected)
            before = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
            archive = open_feature_archive(root)
            self.assertIsInstance(archive, LegacyRaggedFeatureArchive)
            self.assertEqual(len(archive), 2)
            np.testing.assert_array_equal(archive[0], expected[0])
            np.testing.assert_array_equal(archive[-1], expected[-1])
            after = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
            self.assertEqual(before, after)

    def test_rejects_shape_and_data_size_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "legacy"
            write_legacy(root, [np.zeros((3, 40), dtype=np.float32)])
            np.asarray([3, 39], dtype="<i8").tofile(root / "shapes" / "data.ninja")
            with self.assertRaisesRegex(ValueError, "offsets or shapes"):
                LegacyRaggedFeatureArchive(root)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "legacy"
            write_legacy(root, [np.zeros((3, 40), dtype=np.float32)])
            with (root / "data.ninja").open("ab") as output:
                output.write(b"bad")
            with self.assertRaisesRegex(ValueError, "data size"):
                LegacyRaggedFeatureArchive(root)

    def test_rejects_missing_or_empty_archives(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "legacy"
            root.mkdir()
            with self.assertRaisesRegex(ValueError, "missing"):
                open_feature_archive(root)


if __name__ == "__main__":
    unittest.main()
