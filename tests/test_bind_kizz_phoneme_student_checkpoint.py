import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.bind_kizz_phoneme_student_checkpoint import bind_checkpoint


class BindKizzPhonemeStudentCheckpointTests(unittest.TestCase):
    def test_binds_exact_retained_checkpoint_without_changing_base(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "step-0500.weights.h5"
            weights.write_bytes(b"checkpoint")
            digest = hashlib.sha256(weights.read_bytes()).hexdigest()
            source = root / "distillation.json"
            original = {
                "student": {"best_step": 2400, "weights_sha256": "best"},
                "validation_ledger": [{
                    "step": 500,
                    "checkpoint": {"path": str(weights), "sha256": digest},
                }],
            }
            source.write_text(json.dumps(original))
            output = root / "candidate" / "distillation.json"

            result = bind_checkpoint(source, 500, output)

            self.assertEqual(result["student"]["selected_checkpoint"], "step-0500")
            self.assertEqual(result["student"]["weights_sha256"], digest)
            self.assertEqual(json.loads(source.read_text()), original)

    def test_rejects_checkpoint_hash_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "step.weights.h5"
            weights.write_bytes(b"checkpoint")
            source = root / "distillation.json"
            source.write_text(json.dumps({
                "student": {},
                "validation_ledger": [{
                    "step": 1,
                    "checkpoint": {"path": str(weights), "sha256": "0" * 64},
                }],
            }))
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                bind_checkpoint(source, 1, root / "candidate.json")


if __name__ == "__main__":
    unittest.main()
