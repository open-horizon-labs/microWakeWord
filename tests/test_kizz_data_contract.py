import json
import tempfile
import unittest
import wave
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


GATED_CONTRACT = """
schema_version: 1
training_split: train
splits: [train, validation, test]
require_each_split: [positive, negative]
required_metadata: [speaker_id, session_id]
semantic_labels:
  enabled: true
  field: semantic_label
  canonical: canonical_exact
  prohibited_labels: [kids_variant, high_five_variant]
provenance:
  enabled: true
  required_fields: [source_id, provenance_id, parent_id, ancestry_id]
duration:
  enabled: true
  min_seconds: 0.25
  max_seconds: 4.0
overall:
  min_positive_fraction: 0.4
  max_positive_fraction: 0.6
classes:
  positive:
    min_source_groups: 3
    required_source_groups: [piper_synthetic, labeled_tts, device_replay]
    max_source_fraction: 0.5
    max_source_duration_fraction: 0.5
    min_source_fractions: {labeled_tts: 0.1, device_replay: 0.1}
  negative:
    min_source_groups: 4
    required_source_groups: [piper_hard_negative, public_speech, background, device_false_wake]
    max_source_fraction: 0.5
    min_source_fractions: {public_speech: 0.05, background: 0.05, device_false_wake: 0.01}
split_disjoint:
  speaker_id: true
  session_id: true
  ancestry_id: true
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

    def write_gated_fixture(self, examples, contract=GATED_CONTRACT):
        root = Path(tempfile.mkdtemp())
        for item in examples:
            path = root / item["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
            item["path"] = str(path)
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"schema_version": 2, "examples": examples}))
        contract_path = root / "contract.yaml"
        contract_path.write_text(contract)
        return manifest, contract_path

    def complete_examples(self):
        examples = []
        positive_groups = ["piper_synthetic", "labeled_tts", "device_replay"]
        negative_groups = [
            "piper_hard_negative",
            "public_speech",
            "background",
            "device_false_wake",
        ]
        for split, count in (("train", 10), ("validation", 1), ("test", 1)):
            for label, groups in ((1, positive_groups), (0, negative_groups)):
                for group in groups:
                    for index in range(count):
                        examples.append(
                            {
                                "path": f"{split}-{label}-{group}-{index}.wav",
                                "label": label,
                                "source_group": group,
                                "split": split,
                                "speaker_id": f"{group}-{split}-speaker-{index}",
                                "session_id": f"{group}-{split}-session-{index}",
                            }
                        )
        return examples

    def gated_examples(self):
        examples = self.complete_examples()
        for index, item in enumerate(examples):
            item.update(
                {
                    "semantic_label": (
                        "canonical_exact" if item["label"] else "negative"
                    ),
                    "source_id": f"source-{item['source_group']}-{item['split']}",
                    "provenance_id": f"provenance-{index}",
                    "parent_id": f"parent-{index}",
                    "ancestry_id": f"ancestry-{index}",
                    "duration_seconds": 1.0,
                }
            )
        return examples

    def test_complete_manifest_qualifies(self):
        manifest, contract = self.write_fixture(self.complete_examples())
        report = validate_balance_manifest(manifest, contract)
        self.assertTrue(report["qualified"], report["violations"])

    def test_legacy_contract_remains_compatible_without_new_gates(self):
        manifest, contract = self.write_fixture(self.complete_examples())
        report = validate_balance_manifest(manifest, contract)
        self.assertTrue(report["qualified"], report["violations"])

    def test_prohibited_kids_positive_is_rejected(self):
        examples = self.gated_examples()
        next(item for item in examples if item["label"] == 1)[
            "semantic_label"
        ] = "kids_variant"
        manifest, contract = self.write_gated_fixture(examples)
        report = validate_balance_manifest(manifest, contract)
        self.assertFalse(report["qualified"])
        self.assertTrue(
            any(
                "prohibited positive semantic label" in item
                for item in report["violations"]
            )
        )

    def test_prohibited_high_five_positive_is_rejected(self):
        examples = self.gated_examples()
        next(item for item in examples if item["label"] == 1)[
            "semantic_label"
        ] = "high_five_variant"
        manifest, contract = self.write_gated_fixture(examples)
        report = validate_balance_manifest(manifest, contract)
        self.assertFalse(report["qualified"])
        self.assertTrue(
            any(
                "prohibited positive semantic label" in item
                for item in report["violations"]
            )
        )

    def test_missing_provenance_is_rejected_fail_closed(self):
        examples = self.gated_examples()
        del examples[0]["parent_id"]
        manifest, contract = self.write_gated_fixture(examples)
        report = validate_balance_manifest(manifest, contract)
        self.assertFalse(report["qualified"])
        self.assertTrue(any("parent_id" in item for item in report["violations"]))

    def test_duplicate_ancestry_across_splits_is_rejected(self):
        examples = self.gated_examples()
        train_item = next(item for item in examples if item["split"] == "train")
        test_item = next(item for item in examples if item["split"] == "test")
        test_item["ancestry_id"] = train_item["ancestry_id"]
        manifest, contract = self.write_gated_fixture(examples)
        report = validate_balance_manifest(manifest, contract)
        self.assertFalse(report["qualified"])
        self.assertTrue(
            any(
                "ancestry_id values overlap train/test" in item
                for item in report["violations"]
            )
        )

    def test_duplicate_source_id_across_splits_is_rejected(self):
        examples = self.gated_examples()
        train_item = next(item for item in examples if item["split"] == "train")
        test_item = next(item for item in examples if item["split"] == "test")
        test_item["source_id"] = train_item["source_id"]
        manifest, contract = self.write_gated_fixture(examples)
        report = validate_balance_manifest(manifest, contract)
        self.assertFalse(report["qualified"])
        self.assertTrue(
            any(
                "source_id values overlap train/test" in item
                for item in report["violations"]
            )
        )

    def test_count_balanced_but_duration_skewed_source_is_rejected(self):
        examples = self.gated_examples()
        positives = [
            item for item in examples if item["split"] == "train" and item["label"] == 1
        ]
        for item in positives:
            item["duration_seconds"] = (
                100.0 if item["source_group"] == "piper_synthetic" else 1.0
            )
        contract = GATED_CONTRACT.replace(
            "max_source_fraction: 0.5", "max_source_fraction: 1.0"
        )
        manifest, contract_path = self.write_gated_fixture(examples, contract)
        report = validate_balance_manifest(manifest, contract_path)
        self.assertFalse(report["qualified"])
        self.assertTrue(
            any("duration fraction" in item for item in report["violations"])
        )

    def test_duration_can_be_computed_from_wav(self):
        examples = self.gated_examples()
        for item in examples:
            del item["duration_seconds"]
        root = Path(tempfile.mkdtemp())
        for item in examples:
            path = root / item["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(path), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16000)
                audio.writeframes(b"\0\0" * 16000)
            item["path"] = str(path)
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"schema_version": 2, "examples": examples}))
        contract = root / "contract.yaml"
        contract.write_text(GATED_CONTRACT)
        report = validate_balance_manifest(manifest, contract)
        self.assertTrue(report["qualified"], report["violations"])
        self.assertEqual(report["training"]["total_duration_seconds"], 70.0)

    def test_piper_only_manifest_is_rejected(self):
        examples = [
            {
                "path": f"positive-{i}.wav",
                "label": 1,
                "source_group": "piper_synthetic",
                "split": "train",
            }
            for i in range(90)
        ] + [
            {
                "path": f"negative-{i}.wav",
                "label": 0,
                "source_group": "piper_hard_negative",
                "split": "train",
            }
            for i in range(10)
        ]
        manifest, contract = self.write_fixture(examples)
        report = validate_balance_manifest(manifest, contract)
        self.assertFalse(report["qualified"])
        self.assertTrue(
            any("required source group" in item for item in report["violations"])
        )

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
        self.assertEqual(
            report["split_counts"]["train"], {"positive": 30, "negative": 40}
        )
        self.assertEqual(
            report["split_counts"]["validation"], {"positive": 5, "negative": 6}
        )

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
