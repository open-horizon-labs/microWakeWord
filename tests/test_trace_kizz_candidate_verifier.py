import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools import trace_kizz_candidate_verifier as tracer


def _sha(path):
    return tracer.sha256_file(path)


class TraceCandidateVerifierTests(unittest.TestCase):
    def test_evaluation_only_cli_is_available(self):
        with self.assertRaises(SystemExit) as raised:
            tracer.main(["--help"])
        self.assertEqual(raised.exception.code, 0)

    def test_quantization_accepts_converter_mapping_and_tflite_pair(self):
        self.assertEqual(tracer._quantization({"scale": 0.1, "zero_point": -3}, "q"), (0.1, -3))
        self.assertEqual(tracer._quantization((0.2, 4), "q"), (0.2, 4))

    def test_source_base_preserves_identity_and_truth(self):
        row = {
            "source_id": "s", "split": "test", "label": 1,
            "duration_seconds": 2.6, "audio_sha256": hashlib.sha256(b"a").hexdigest(),
            "speaker_id": "speaker",
        }
        value = tracer._source_base(row)
        self.assertEqual(value["truth"], "positive")
        self.assertEqual(value["speaker_id"], "speaker")
        self.assertEqual(value["events"], [])

    def test_source_base_accepts_continuous_corpus_audio_identity(self):
        digest = hashlib.sha256(b"capture").hexdigest()
        value = tracer._source_base(
            {
                "source_id": "physical",
                "split": "test",
                "label": 1,
                "duration_seconds": 4.62,
                "source_audio_sha256": digest,
            }
        )
        self.assertEqual(value["audio_sha256"], digest)

    def test_override_corpus_arrays_fail_closed_on_hash_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = root / "features.npy"
            np.save(features, np.zeros((1, 260, 40), dtype=np.float32))
            corpus_path = root / "corpus.json"
            corpus = {"array_sha256": {"features.npy": _sha(features)}}
            self.assertEqual(
                tracer._corpus_array_paths(corpus_path, corpus)["features.npy"],
                features,
            )
            corpus["array_sha256"]["features.npy"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "hash drift"):
                tracer._corpus_array_paths(corpus_path, corpus)

    def test_mined_candidate_reconstructs_missing_evaluation_source(self):
        digest = hashlib.sha256(b"source").hexdigest()
        candidate = {
            "candidate_id": "candidate",
            "parent_source_id": "external-source",
            "source_audio_sha256": digest,
            "split": "validation",
            "label": 0,
            "duration_seconds": 12.5,
        }
        rows = tracer._evaluation_source_rows([], [candidate, dict(candidate)])
        self.assertEqual(rows["external-source"]["audio_sha256"], digest)

        conflicting = dict(candidate, duration_seconds=12.6)
        with self.assertRaisesRegex(ValueError, "identity conflict"):
            tracer._evaluation_source_rows([], [candidate, conflicting])

    def test_binding_fails_closed_on_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x"
            path.write_bytes(b"x")
            value = {"path": str(path), "sha256": "0" * 64, "bytes": 1}
            with self.assertRaisesRegex(ValueError, "hash drift"):
                tracer._binding(value, Path(directory), "x")


if __name__ == "__main__":
    unittest.main()
