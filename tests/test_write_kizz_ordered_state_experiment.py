import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools.write_kizz_ordered_state_experiment import (
    experiment_plan,
    validate_frozen_manifest,
)


class OrderedStateExperimentConfigTest(unittest.TestCase):
    def frozen(self, root: Path, validation=100, test=100):
        path = root / "frozen.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "threshold_selection_split": "validation",
                    "source_disjoint": {
                        "source_id": True,
                        "speaker_id": True,
                        "session_id": True,
                    },
                    "inventory_sha256": "abc",
                    "sources": [],
                    "counts": {
                        "exposure_seconds_by_split": {
                            "train": 1,
                            "validation": validation * 3600,
                            "test": test * 3600,
                        }
                    },
                }
            )
        )
        return path

    def test_requires_hundred_hour_validation_and_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            validate_frozen_manifest(self.frozen(root))
            with self.assertRaisesRegex(ValueError, "validation"):
                validate_frozen_manifest(self.frozen(root, validation=99))

    def test_ordered_candidate_requires_real_frame_arrays(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch(
                "tools.write_kizz_ordered_state_experiment.validate_frozen_manifest",
                return_value={"inventory_sha256": "abc"},
            ):
                with self.assertRaisesRegex(ValueError, "frame supervision"):
                    experiment_plan(
                        root,
                        root / "trained",
                        "ordered_state",
                        root / "frozen.json",
                        root / "hard",
                    )

    def test_zero_frame_weight_is_a_declared_sequence_only_ablation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root / "features.v32-canonical-connected"
            device = root / "promoted.v32-device-canonical-span-aligned" / "features"
            hard = root / "hard"
            negative = root / "ordered-v1-frozen-negative-view"
            for directory in (
                canonical / "positive",
                canonical / "hard_negative",
                device / "positive",
                hard / "mined",
                hard / "random_reserve",
                negative / "speech",
                negative / "non_speech",
            ):
                directory.mkdir(parents=True)
            (canonical / "feature-build-manifest.json").write_text("{}")
            (device / "promoted-audio-feature-build.json").write_text("{}")
            (hard / "mining-metadata.json").write_text("{}")
            with (
                mock.patch(
                    "tools.write_kizz_ordered_state_experiment.validate_frozen_manifest",
                    return_value={"inventory_sha256": "abc"},
                ),
                mock.patch(
                    "tools.write_kizz_ordered_state_experiment.sha256",
                    return_value="hash",
                ),
                mock.patch(
                    "tools.write_kizz_ordered_state_experiment.sha256_path",
                    return_value="directory-hash",
                ),
            ):
                plan = experiment_plan(
                    root,
                    root / "trained",
                    "ordered_state",
                    root / "frozen.json",
                    hard,
                    frame_weight=0.0,
                )
            self.assertEqual(plan["ordered_state_frame_weight"], 0.0)
            self.assertNotIn(
                "frame_supervision", plan["config_overrides"]["training_loss"]
            )

    def test_plan_content_binds_feature_directories_and_frame_arrays(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen = self.frozen(root)
            canonical = root / "features.v32-canonical-connected"
            device = root / "promoted.v32-device-canonical-span-aligned" / "features"
            hard = root / "hard"
            negative = root / "ordered-v1-frozen-negative-view"
            directories = (
                canonical / "positive",
                canonical / "hard_negative",
                device / "positive",
                hard / "mined",
                hard / "random_reserve",
                negative / "speech",
                negative / "non_speech",
            )
            for index, directory in enumerate(directories):
                directory.mkdir(parents=True)
                (directory / "data.bin").write_bytes(f"data-{index}".encode())
            (canonical / "feature-build-manifest.json").write_text("{}")
            (device / "promoted-audio-feature-build.json").write_text("{}")
            (hard / "mining-metadata.json").write_text("{}")
            frames = root / "frames"
            frames.mkdir()
            for filename in ("features.npy", "targets.npy", "weights.npy"):
                (frames / filename).write_bytes(filename.encode())

            plan = experiment_plan(
                root,
                root / "trained",
                "ordered_state",
                frozen,
                hard,
                frame_supervision=frames,
            )
            for source in plan["sources"]:
                self.assertEqual(
                    source["expected_path_sha256"],
                    plan["input_hashes"][source["features_dir"]],
                )
            frame_config = plan["config_overrides"]["training_loss"][
                "frame_supervision"
            ]
            self.assertEqual(
                set(frame_config["expected_files_sha256"]),
                {"features.npy", "targets.npy", "weights.npy"},
            )

            original_source_hash = plan["sources"][0]["expected_path_sha256"]
            original_frame_hash = frame_config["expected_files_sha256"]["features.npy"]
            (directories[0] / "data.bin").write_bytes(b"changed")
            (frames / "features.npy").write_bytes(b"changed-frames")
            changed = experiment_plan(
                root,
                root / "trained",
                "ordered_state",
                frozen,
                hard,
                frame_supervision=frames,
            )
            self.assertNotEqual(
                original_source_hash,
                changed["sources"][0]["expected_path_sha256"],
            )
            self.assertNotEqual(
                original_frame_hash,
                changed["config_overrides"]["training_loss"]["frame_supervision"][
                    "expected_files_sha256"
                ]["features.npy"],
            )

    def test_rejects_quarantined_frozen_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.frozen(root)
            value = json.loads(path.read_text())
            value["sources"] = [
                {
                    "source_id": "wake",
                    "path": str(root / "observations" / "false-wakes"),
                }
            ]
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "quarantined"):
                validate_frozen_manifest(path)


if __name__ == "__main__":
    unittest.main()
