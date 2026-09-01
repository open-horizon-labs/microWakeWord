import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.curate_kizz_pronunciation_accepted_qualification_inventory import curate
from tools.prepare_kizz_reserved_multisource_qualification_inventory import sha256_file


class CurateAcceptedInventoryTest(unittest.TestCase):
    def test_selects_only_accepted_unconsumed_rows_and_locks_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for provider in ("a", "b"):
                for voice in ("one", "two"):
                    for index in range(2):
                        audio = f"{provider}:{voice}:{index}".encode()
                        path = root / f"{provider}-{voice}-{index}.wav"
                        path.write_bytes(audio)
                        rows.append({
                            "source_id": f"source:{provider}:{voice}:{index}",
                            "audio_sha256": hashlib.sha256(audio).hexdigest(),
                            "path": str(path), "provider": provider, "voice": voice,
                            "label": 1, "split": "test", "training_eligible": True,
                        })
            source = root / "source.json"
            source.write_text(json.dumps({"examples": rows}))
            audit = root / "audit.json"
            audit.write_text(json.dumps({
                "gate_scope": "independent_source_pronunciation_qc",
                "source_manifest_sha256": sha256_file(source),
                "results": [
                    {"source_id": row["source_id"], "audio_sha256": row["audio_sha256"], "accepted": row is not rows[-1]}
                    for row in rows
                ],
            }))
            exclusion = root / "exclusion.json"
            exclusion.write_text(json.dumps({"examples": [{"audio_sha256": rows[0]["audio_sha256"]}]}))
            output = root / "output.json"
            payload = curate([(source, audit)], [exclusion], {"a": 2, "b": 2}, output)
            selected = payload["examples"]
            self.assertEqual(len(selected), 4)
            self.assertNotIn(rows[0]["audio_sha256"], {row["audio_sha256"] for row in selected})
            self.assertNotIn(rows[-1]["audio_sha256"], {row["audio_sha256"] for row in selected})
            self.assertTrue(payload["locked_before_scoring"])
            self.assertFalse(payload["selection_policy"]["model_scores_read_or_used"])
            self.assertTrue(all(row["training_eligible"] is False for row in selected))
            self.assertEqual(
                set(payload["reserved_evidence_contract"]["providers"]["a"]),
                {"count", "minimum_voices"},
            )

    def test_rejects_audit_source_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            source.write_text(json.dumps({"examples": []}))
            audit = root / "audit.json"
            audit.write_text(json.dumps({
                "gate_scope": "independent_source_pronunciation_qc",
                "source_manifest_sha256": "0" * 64,
                "results": [],
            }))
            with self.assertRaisesRegex(ValueError, "source binding drift"):
                curate([(source, audit)], [], {"a": 1}, root / "output.json")


if __name__ == "__main__":
    unittest.main()
