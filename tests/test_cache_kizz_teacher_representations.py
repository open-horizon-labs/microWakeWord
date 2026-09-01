import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.cache_kizz_teacher_representations import load_representation_cache


class RepresentationCacheTest(unittest.TestCase):
    def _write_cache(self, root: Path) -> Path:
        prefix = root / "cache"
        matrix = np.arange(24, dtype=np.float16).reshape(3, 8)
        metadata = {
            "schema_version": 1,
            "shape": list(matrix.shape),
            "dtype": str(matrix.dtype),
        }
        digest = hashlib.sha256()
        digest.update(matrix.tobytes(order="C"))
        digest.update(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        )
        metadata["cache_sha256"] = digest.hexdigest()
        np.save(prefix.with_suffix(".npy"), matrix)
        prefix.with_suffix(".json").write_text(json.dumps(metadata))
        return prefix

    def test_loads_hash_bound_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = self._write_cache(Path(directory))
            metadata, matrix = load_representation_cache(prefix)
            self.assertEqual(metadata["shape"], [3, 8])
            np.testing.assert_array_equal(matrix, np.arange(24).reshape(3, 8))

    def test_rejects_corrupt_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = self._write_cache(Path(directory))
            matrix = np.load(prefix.with_suffix(".npy"))
            matrix[0, 0] = 99
            np.save(prefix.with_suffix(".npy"), matrix)
            with self.assertRaisesRegex(ValueError, "stale or corrupt"):
                load_representation_cache(prefix)


if __name__ == "__main__":
    unittest.main()
