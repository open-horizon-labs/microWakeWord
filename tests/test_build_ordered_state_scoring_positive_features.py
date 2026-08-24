import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.build_ordered_state_scoring_positive_features import (
    build_positive_features,
)


class FakeRaggedMmap:
    writes = {}

    @classmethod
    def from_generator(cls, out_dir, sample_generator, batch_size, verbose=False):
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        items = list(sample_generator)
        cls.writes[str(path)] = items
        (path / "items.json").write_text(
            json.dumps([item.tolist() for item in items]), encoding="utf-8"
        )


def make_record(root, source_id, split, *, span=(0.02, 0.07), group=None):
    feature_path = root / f"{source_id}.npy"
    np.save(
        feature_path, np.full((20, 40), len(FakeRaggedMmap.writes), dtype=np.float32)
    )
    return {
        "source_id": source_id,
        "source_group": group or f"speaker:{source_id}",
        "split": split,
        "truth": True,
        "features_path": feature_path.name,
        "feature_frame_step_seconds": 0.01,
        "phrase_span": {"start_s": span[0], "end_s": span[1]},
    }


class BuildOrderedStateScoringPositiveFeaturesTest(unittest.TestCase):
    def setUp(self):
        FakeRaggedMmap.writes = {}

    def test_preserves_manifest_order_and_exact_spans(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            validation = root / "validation.json"
            testing = root / "testing.json"
            validation.write_text(
                json.dumps(
                    {
                        "records": [
                            make_record(root, "v-2", "validation", span=(0.11, 0.19)),
                            make_record(root, "v-1", "validation", span=(0.01, 0.09)),
                        ]
                    }
                ),
                encoding="utf-8",
            )
            testing.write_text(
                json.dumps({"records": [make_record(root, "t-1", "test")]}),
                encoding="utf-8",
            )
            output = root / "out"
            with patch(
                "tools.build_ordered_state_scoring_positive_features.RaggedMmap",
                FakeRaggedMmap,
            ):
                summary = build_positive_features([validation, testing], output)
            report = json.loads(
                (output / "ordered-state-positive-occurrences.json").read_text()
            )
            self.assertEqual(summary["validation_examples"], 2)
            self.assertEqual(summary["test_examples"], 1)
            occurrences = report["occurrences"]
            self.assertEqual(
                [(item["source_id"], item["item_index"]) for item in occurrences],
                [("v-2", 0), ("v-1", 1), ("t-1", 0)],
            )
            self.assertEqual(
                occurrences[0]["phrase_span"], {"start_s": 0.11, "end_s": 0.19}
            )
            self.assertEqual(
                [
                    item[0, 0]
                    for item in FakeRaggedMmap.writes[
                        str(output / "positive" / "validation" / "wakeword_mmap")
                    ]
                ],
                [0.0, 0.0],
            )

    def test_rejects_train_mixed_missing_duplicate_and_invalid_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid_v = make_record(root, "v", "validation")
            valid_t = make_record(root, "t", "test")

            def run(records):
                manifest = root / "input.json"
                manifest.write_text(json.dumps({"records": records}), encoding="utf-8")
                with patch(
                    "tools.build_ordered_state_scoring_positive_features.RaggedMmap",
                    FakeRaggedMmap,
                ):
                    return build_positive_features([manifest], root / "out")

            with self.assertRaisesRegex(ValueError, "train"):
                run([dict(valid_v, split="train")])
            with self.assertRaisesRegex(ValueError, "mixed or missing"):
                run([valid_v, valid_t])
            with self.assertRaisesRegex(ValueError, "duplicate source_id"):
                run([valid_v, dict(valid_v)])
            with self.assertRaisesRegex(ValueError, "phrase_span"):
                run([dict(valid_v, phrase_span={"start_s": 0.0, "end_s": 0.201})])
            with self.assertRaisesRegex(ValueError, "missing required split"):
                run([valid_v])

    def test_rejects_empty_and_quarantine_evidence_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            empty = root / "empty.json"
            empty.write_text(json.dumps({"records": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "empty"):
                build_positive_features([empty], root / "out")

            evidence = root / "evidence"
            evidence.mkdir()
            record = make_record(evidence, "v", "validation")
            record["features_path"] = str(evidence / "v.npy")
            manifest = root / "input.json"
            manifest.write_text(json.dumps({"records": [record]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "quarantine/evidence"):
                build_positive_features([manifest], root / "out")


if __name__ == "__main__":
    unittest.main()
