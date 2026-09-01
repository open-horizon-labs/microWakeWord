import tempfile
import unittest
import hashlib
import json
from pathlib import Path

from tools.write_kizz_v32_training_config import (
    V19_CHECKPOINT_100_WEIGHTS_SHA256,
    experiment_plan,
    reject_quarantined_path,
    validate_promoted_device_features,
)


class KizzV32TrainingConfigTest(unittest.TestCase):
    def arguments(self, root: Path):
        return (
            root,
            root / "features" / "feature-build-manifest.json",
            root / "mined",
            root / "device",
            root / "trained",
        )

    def test_scratch_is_sixty_percent_negative_and_keeps_eval_untouched(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = experiment_plan(*self.arguments(root), mode="scratch")
            positive = sum(
                weight
                for name, weight in plan["sampling_groups"].items()
                if name in {"canonical_positive", "device_positive"}
            )
            self.assertAlmostEqual(positive, 0.40)
            self.assertEqual(
                plan["balance_guard"]["maximum_negative_sampling_share"], 0.60
            )
            dinner_eval = [
                source
                for source in plan["sources"]
                if source.get("source_name") == "untouched-dinner-party-evaluation"
            ][0]
            self.assertEqual(dinner_eval["within_group_weight"], 0)
            self.assertNotIn("group", dinner_eval)
            serialized = str(plan).casefold()
            self.assertNotIn("observations", serialized)
            self.assertNotIn("2026-08-23", serialized)
            self.assertNotIn("heldout-device-canonical-positive", serialized)

    def test_adaptation_requires_exact_deployed_checkpoint_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = root / "weights.h5"
            weights.write_bytes(b"wrong")
            with self.assertRaisesRegex(ValueError, "not deployed v19"):
                experiment_plan(
                    *self.arguments(root),
                    mode="v19-adapt",
                    initial_weights=weights,
                )
            self.assertEqual(len(V19_CHECKPOINT_100_WEIGHTS_SHA256), 64)

    def test_scratch_and_adaptation_use_different_schedules(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scratch = experiment_plan(*self.arguments(root), mode="scratch")
            self.assertEqual(
                scratch["config_overrides"]["training_steps"], [20000, 10000]
            )
            self.assertNotIn("initial_weights", scratch["config_overrides"])

    def test_candidate_remine_binds_arbitrary_candidate_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = root / "candidate.weights.h5"
            weights.write_bytes(b"candidate")
            plan = experiment_plan(
                *self.arguments(root),
                mode="candidate-remine",
                initial_weights=weights,
            )
            self.assertEqual(plan["config_overrides"]["training_steps"], [1000, 500])
            self.assertEqual(
                plan["initialization"]["weights_sha256"],
                hashlib.sha256(b"candidate").hexdigest(),
            )
            self.assertIn("candidate-union-mined-negative", str(plan))

    def test_quarantined_observation_paths_cannot_become_training_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments = list(self.arguments(root))
            arguments[2] = root / "evidence" / "2026-08-23" / "features"
            with self.assertRaisesRegex(ValueError, "quarantined evidence"):
                experiment_plan(*arguments, mode="scratch")

    def test_symlink_into_quarantine_cannot_become_a_training_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            quarantined = root / "evidence" / "features"
            quarantined.mkdir(parents=True)
            clean_link = root / "features"
            clean_link.symlink_to(quarantined, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "quarantined evidence"):
                reject_quarantined_path(clean_link, "test source")

    def test_promoted_feature_contract_binds_source_and_speaker_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            source.write_text(
                json.dumps(
                    {
                        "speaker_ids_by_split": {
                            "train": ["train-speaker"],
                            "test": ["test-speaker"],
                        }
                    }
                )
            )
            feature_manifest = {
                "source_manifest": str(source),
                "source_manifest_sha256": hashlib.sha256(
                    source.read_bytes()
                ).hexdigest(),
                "entries": [
                    {
                        "truth": "positive",
                        "text": "Hi-Fi Kizz",
                        "human_reviewed": True,
                        "training_eligible": True,
                        "split": split,
                    }
                    for split in ("train", "test")
                ],
            }
            (root / "promoted-audio-feature-build.json").write_text(
                json.dumps(feature_manifest)
            )
            self.assertEqual(
                validate_promoted_device_features(root)["entries"],
                feature_manifest["entries"],
            )
            source.write_text(
                json.dumps(
                    {
                        "speaker_ids_by_split": {
                            "train": ["same"],
                            "test": ["same"],
                        }
                    }
                )
            )
            feature_manifest["source_manifest_sha256"] = hashlib.sha256(
                source.read_bytes()
            ).hexdigest()
            (root / "promoted-audio-feature-build.json").write_text(
                json.dumps(feature_manifest)
            )
            with self.assertRaisesRegex(ValueError, "speakers overlap"):
                validate_promoted_device_features(root)

    def test_promoted_feature_contract_allows_train_only_aligned_positives(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            source.write_text(
                json.dumps({"speaker_ids_by_split": {"train": ["train-speaker"]}})
            )
            feature_manifest = {
                "source_manifest": str(source),
                "source_manifest_sha256": hashlib.sha256(
                    source.read_bytes()
                ).hexdigest(),
                "entries": [
                    {
                        "truth": "positive",
                        "text": "Hi-Fi Kizz",
                        "human_reviewed": True,
                        "training_eligible": True,
                        "split": "train",
                    }
                ],
            }
            (root / "promoted-audio-feature-build.json").write_text(
                json.dumps(feature_manifest)
            )
            self.assertEqual(
                validate_promoted_device_features(root)["entries"],
                feature_manifest["entries"],
            )


if __name__ == "__main__":
    unittest.main()
