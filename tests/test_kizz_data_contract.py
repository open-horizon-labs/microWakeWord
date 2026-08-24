import json
import tempfile
import unittest
from pathlib import Path

from microwakeword.kizz_data_contract import validate_balance_manifest
from tools.curate_kizz_manifest import curate


CONTRACT = """
schema_version: 1
training_split: train
splits: [train, validation, test]
require_each_split: [positive, negative]
required_metadata: [speaker_id, session_id]
overall: {min_positive_fraction: 0.4, max_positive_fraction: 0.6}
classes:
  positive:
    min_source_groups: 3
    required_source_groups: [piper_synthetic, labeled_tts, device_replay]
    max_source_fraction: 0.5
    min_source_fractions: {labeled_tts: 0.1, device_replay: 0.1}
  negative:
    min_source_groups: 4
    required_source_groups: [piper_hard_negative, public_speech, background, device_false_wake]
    max_source_fraction: 0.5
    min_source_fractions: {public_speech: 0.05, background: 0.05, device_false_wake: 0.01}
split_disjoint: {speaker_id: true, session_id: true}
"""


class KizzDataContractTests(unittest.TestCase):
    def write_fixture(self, examples):
        root = Path(tempfile.mkdtemp())
        for item in examples:
            path = root / item["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
            item["path"] = str(path)
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"schema_version": 2, "examples": examples}))
        contract = root / "contract.yaml"
        contract.write_text(CONTRACT)
        return manifest, contract

    def complete_examples(self):
        examples = []
        positive_groups = ["piper_synthetic", "labeled_tts", "device_replay"]
        negative_groups = ["piper_hard_negative", "public_speech", "background", "device_false_wake"]
        for split, count in (("train", 10), ("validation", 1), ("test", 1)):
            for label, groups in ((1, positive_groups), (0, negative_groups)):
                for group in groups:
                    for index in range(count):
                        examples.append({
                            "path": f"{split}-{label}-{group}-{index}.wav",
                            "label": label,
                            "source_group": group,
                            "split": split,
                            "speaker_id": f"{group}-{split}-speaker-{index}",
                            "session_id": f"{group}-{split}-session-{index}",
                        })
        return examples

    def test_complete_manifest_qualifies(self):
        manifest, contract = self.write_fixture(self.complete_examples())
        report = validate_balance_manifest(manifest, contract)
        self.assertTrue(report["qualified"], report["violations"])

    def test_piper_only_manifest_is_rejected(self):
        examples = [
            {"path": f"positive-{i}.wav", "label": 1, "source_group": "piper_synthetic", "split": "train"}
            for i in range(90)
        ] + [
            {"path": f"negative-{i}.wav", "label": 0, "source_group": "piper_hard_negative", "split": "train"}
            for i in range(10)
        ]
        manifest, contract = self.write_fixture(examples)
        report = validate_balance_manifest(manifest, contract)
        self.assertFalse(report["qualified"])
        self.assertTrue(any("required source group" in item for item in report["violations"]))

    def test_split_counts_are_label_accurate(self):
        examples = self.complete_examples()
        examples.extend(
            {
                "path": f"validation-{index}.wav",
                "label": index % 2,
                "source_group": "device_replay" if index % 2 else "background",
                "split": "validation",
                "speaker_id": f"extra-speaker-{index}",
                "session_id": f"extra-session-{index}",
            }
            for index in range(4)
        )
        manifest, contract = self.write_fixture(examples)
        report = validate_balance_manifest(manifest, contract)
        self.assertEqual(report["split_counts"]["train"], {"positive": 30, "negative": 40})
        self.assertEqual(report["split_counts"]["validation"], {"positive": 5, "negative": 6})

    def test_curation_is_deterministic_and_preserves_heldout_splits(self):
        source_manifest, _ = self.write_fixture(self.complete_examples())
        output_a = source_manifest.parent / "curated-a.json"
        output_b = source_manifest.parent / "curated-b.json"
        curate(source_manifest, output_a, seed=7, caps={"piper_synthetic": 2})
        curate(source_manifest, output_b, seed=7, caps={"piper_synthetic": 2})
        self.assertEqual(output_a.read_bytes(), output_b.read_bytes())
        result = json.loads(output_a.read_text())
        self.assertEqual(
            sum(
                item["split"] == "train" and item["source_group"] == "piper_synthetic"
                for item in result["examples"]
            ),
            2,
        )
        self.assertTrue(any(item["split"] == "test" for item in result["examples"]))


if __name__ == "__main__":
    unittest.main()
