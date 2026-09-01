import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import microwakeword.hard_negative_mining as mining
from microwakeword.hard_negative_mining import (
    deterministic_reserve_starts,
    discover_archives,
    effective_score_band_quota,
    local_maxima,
    merge_shards,
    mine,
    prediction_coordinates,
    score_band,
    shard_artifact_root,
    temporal_nms,
)


class FakeMmap:
    arrays = {}
    writes = {}

    def __init__(self, path):
        self.items = self.arrays[str(path)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    @classmethod
    def from_generator(cls, out_dir, sample_generator, batch_size, verbose=False):
        cls.writes[str(out_dir)] = list(sample_generator)


class FakeModel:
    created = []

    def __init__(self, name):
        self.name = name
        self.stride = 3
        self.input_feature_slices = 5
        self.resets = 0
        self.calls = []
        self.created.append(self)

    def reset_states(self):
        self.resets += 1

    def predict_spectrogram(self, spectrogram):
        self.calls.append(int(spectrogram[0, 0]))
        if self.name == "first":
            return [0.1, 0.2, 0.1, 0.95, 0.1, 0.1, 0.1, 0.1]
        return [0.1, 0.1, 0.1, 0.1, 0.1, 0.92, 0.1, 0.1]


def fixture_item(identity, frames=60):
    result = np.full((frames, 40), identity, dtype=np.uint16)
    return result


class MiningTest(unittest.TestCase):
    def setUp(self):
        FakeMmap.arrays = {}
        FakeMmap.writes = {}
        FakeModel.created = []

    def test_local_maxima_nms_and_score_bands(self):
        self.assertEqual(local_maxima([0.1, 0.8, 0.4, 0.9, 0.2], 0.5), [1, 3])
        candidates = [
            {"end_frame": 5, "score": 0.8, "model": "a"},
            {"end_frame": 10, "score": 0.9, "model": "a"},
            {"end_frame": 30, "score": 0.7, "model": "b"},
        ]
        self.assertEqual(
            [item["end_frame"] for item in temporal_nms(candidates, 20)],
            [10, 30],
        )
        self.assertEqual(score_band(0.51), "0.5-0.7")

    def test_default_band_allocation_can_fill_source_quota(self):
        self.assertEqual(effective_score_band_quota(128, 0.5, None), 32)
        self.assertEqual(effective_score_band_quota(128, 0.8, None), 64)
        self.assertEqual(effective_score_band_quota(128, 0.5, 17), 17)

    def test_reserve_starts_are_seeded_stable_and_jittered_within_buckets(self):
        first = deterministic_reserve_starts(231, "source", 4, 700, 220)
        self.assertEqual(
            first, deterministic_reserve_starts(231, "source", 4, 700, 220)
        )
        self.assertNotEqual(
            first, deterministic_reserve_starts(232, "source", 4, 700, 220)
        )
        self.assertEqual(len(first), 4)
        self.assertTrue(
            all(
                low <= value < min(700, low + 220)
                for low, value in zip(range(0, 700, 220), first)
            )
        )

    def test_prediction_coordinate_matches_model_consumption_for_later_peak(self):
        model = FakeModel("first")
        self.assertEqual(prediction_coordinates(model, 7, 100, 20), (6, 26))
        self.assertEqual(prediction_coordinates(model, 0, 3, 20), (0, 3))

    def test_discovery_excludes_evaluation_and_observations(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for relative in (
                "speech/training/ok_mmap",
                "dinner_party_eval/testing_ambient/bad_mmap",
                "observations/false-wakes/bad_mmap",
            ):
                path = root / relative
                path.mkdir(parents=True)
                (path / "data.ninja").write_bytes(relative.encode())
                for name in ("starts", "ends", "shapes"):
                    (path / name).mkdir()
            found = discover_archives([root])
            self.assertEqual([Path(item["path"]).name for item in found], ["ok_mmap"])
            self.assertEqual(found[0]["split"], "training")
            self.assertEqual(len(found[0]["source_layout_hash"]), 64)

    def test_corrupt_metadata_fails_loudly(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "speech" / "training" / "broken_mmap"
            path.mkdir(parents=True)
            (path / "data.ninja").write_bytes(b"data")
            (path / "starts").mkdir()
            with self.assertRaises(ValueError):
                discover_archives([Path(temp)])

    def _models_and_factory(self, root):
        first = root / "first.tflite"
        second = root / "second.tflite"
        first.write_bytes(b"first")
        second.write_bytes(b"second")

        def factory(path, stride):
            return FakeModel(path.stem)

        return [("first", first), ("second", second)], factory

    def test_model_union_coordinates_state_reset_split_and_output_shape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            models, factory = self._models_and_factory(root)
            FakeMmap.arrays = {
                "/fixture/train": [fixture_item(1)],
                "/fixture/validation": [fixture_item(2)],
            }
            archives = [
                {
                    "path": "/fixture/train",
                    "split": "training",
                    "archive": "train_mmap",
                    "source_hash": "train",
                    "source_layout_hash": "train-layout",
                },
                {
                    "path": "/fixture/validation",
                    "split": "validation",
                    "archive": "validation_mmap",
                    "source_hash": "validation",
                    "source_layout_hash": "validation-layout",
                },
            ]
            with (
                patch.object(mining, "RaggedMmap", FakeMmap),
                patch.object(mining, "discover_archives", return_value=archives),
            ):
                result = mine(
                    [root],
                    models,
                    root / "output",
                    context_frames=10,
                    nms_frames=3,
                    per_source_quota=8,
                    per_item_quota=4,
                    score_band_quota=4,
                    model_factory=factory,
                )
            artifact = Path(result["artifact_root"])
            records = [
                json.loads(line)
                for line in (artifact / "mining-manifest.jsonl")
                .read_text()
                .splitlines()
            ]
            high = [
                record
                for record in records
                if record["reason"].startswith("high_score:")
            ]
            self.assertTrue(
                any(
                    record["start_frame"] == 4 and record["end_frame"] == 14
                    for record in high
                )
            )
            self.assertTrue(
                any(
                    record["start_frame"] == 10 and record["end_frame"] == 20
                    for record in high
                )
            )
            self.assertTrue(any(record["scores"]["first"] > 0.9 for record in high))
            self.assertTrue(any(record["scores"]["second"] > 0.9 for record in high))
            self.assertEqual(
                {record["split"] for record in high}, {"training", "validation"}
            )
            self.assertTrue(all(model.resets == 2 for model in FakeModel.created))
            for windows in FakeMmap.writes.values():
                for window in windows:
                    self.assertEqual(window.shape, (10, 40))
                    self.assertEqual(window.dtype, np.uint16)

    def test_required_model_sha_is_enforced_and_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            models, factory = self._models_and_factory(root)
            archives = [
                {
                    "path": "/fixture/required",
                    "split": "training",
                    "archive": "required_mmap",
                    "source_hash": "source",
                    "source_layout_hash": "source-layout",
                }
            ]
            FakeMmap.arrays = {"/fixture/required": [fixture_item(1)]}
            with (
                patch.object(mining, "RaggedMmap", FakeMmap),
                patch.object(mining, "discover_archives", return_value=archives),
            ):
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    mine(
                        [root],
                        models,
                        root / "bad",
                        required_model_shas={"first": "0" * 64},
                        model_factory=factory,
                    )
                expected = mining.sha256_file(models[0][1])
                result = mine(
                    [root],
                    models,
                    root / "good",
                    context_frames=10,
                    nms_frames=3,
                    required_model_shas={"first": expected},
                    model_factory=factory,
                )
            metadata = json.loads(
                (Path(result["artifact_root"]) / "mining-metadata.json").read_text()
            )
            self.assertEqual(
                metadata["config"]["required_model_shas"], {"first": expected}
            )

    def test_shard_assignment_and_resume_are_stable_and_max_counts_scored_items(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            models, factory = self._models_and_factory(root)
            FakeMmap.arrays = {
                "/fixture/sharded": [fixture_item(index) for index in range(6)]
            }
            archives = [
                {
                    "path": "/fixture/sharded",
                    "split": "training",
                    "archive": "sharded_mmap",
                    "source_hash": "source",
                    "source_layout_hash": "source-layout",
                }
            ]
            kwargs = dict(
                roots=[root],
                model_paths=models,
                output=root / "output",
                context_frames=10,
                nms_frames=3,
                per_source_quota=20,
                per_item_quota=4,
                score_band_quota=20,
                shard_count=2,
                max_items=1,
                checkpoint_interval=50,
                model_factory=factory,
            )
            with (
                patch.object(mining, "RaggedMmap", FakeMmap),
                patch.object(mining, "discover_archives", return_value=archives),
            ):
                first = mine(shard_index=1, **kwargs)
                second = mine(shard_index=1, **kwargs)
                zero = mine(shard_index=0, **kwargs)

            shard_one_checkpoint = json.loads(
                (Path(second["artifact_root"]) / "mining-checkpoint.json").read_text()
            )
            shard_zero_checkpoint = json.loads(
                (Path(zero["artifact_root"]) / "mining-checkpoint.json").read_text()
            )
            self.assertEqual(
                shard_one_checkpoint["completed"],
                [["/fixture/sharded", 1], ["/fixture/sharded", 3]],
            )
            self.assertEqual(
                shard_zero_checkpoint["completed"], [["/fixture/sharded", 0]]
            )
            self.assertEqual(first["scored_this_run"], 1)
            self.assertEqual(second["scored_this_run"], 1)
            self.assertEqual(zero["scored_this_run"], 1)
            self.assertNotEqual(first["artifact_root"], zero["artifact_root"])
            self.assertEqual(
                Path(first["artifact_root"]),
                shard_artifact_root(root / "output", 1, 2),
            )

    def test_checkpoint_interval_and_final_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            models, factory = self._models_and_factory(root)
            FakeMmap.arrays = {
                "/fixture/checkpoints": [fixture_item(index) for index in range(5)]
            }
            archives = [
                {
                    "path": "/fixture/checkpoints",
                    "split": "training",
                    "archive": "checkpoints_mmap",
                    "source_hash": "source",
                    "source_layout_hash": "source-layout",
                }
            ]
            original = mining._checkpoint_write
            checkpoint_calls = []

            def tracking_write(path, payload):
                if Path(path).name == "mining-checkpoint.json":
                    checkpoint_calls.append(len(payload["completed"]))
                original(path, payload)

            with (
                patch.object(mining, "RaggedMmap", FakeMmap),
                patch.object(mining, "discover_archives", return_value=archives),
                patch.object(mining, "_checkpoint_write", side_effect=tracking_write),
            ):
                mine(
                    [root],
                    models,
                    root / "output",
                    context_frames=10,
                    nms_frames=3,
                    per_source_quota=40,
                    per_item_quota=4,
                    score_band_quota=40,
                    checkpoint_interval=2,
                    model_factory=factory,
                )
            self.assertEqual(checkpoint_calls, [2, 4, 5])

    def test_two_complete_shards_merge_to_one_training_compatible_corpus(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            models, factory = self._models_and_factory(root)
            FakeMmap.arrays = {
                "/fixture/merge": [fixture_item(index) for index in range(4)]
            }
            archives = [
                {
                    "path": "/fixture/merge",
                    "split": "training",
                    "archive": "merge_mmap",
                    "source_hash": "source",
                    "source_layout_hash": "source-layout",
                }
            ]
            with (
                patch.object(mining, "RaggedMmap", FakeMmap),
                patch.object(mining, "discover_archives", return_value=archives),
            ):
                for shard_index in range(2):
                    mine(
                        [root],
                        models,
                        root / "output",
                        context_frames=10,
                        nms_frames=3,
                        per_source_quota=20,
                        per_item_quota=4,
                        score_band_quota=20,
                        shard_index=shard_index,
                        shard_count=2,
                        checkpoint_interval=2,
                        model_factory=factory,
                    )
                merged = merge_shards(root / "output", 2)

            self.assertGreater(merged["selected"], 0)
            self.assertTrue((root / "output" / "mining-manifest.jsonl").exists())
            self.assertIn(
                str(root / "output" / "mined" / "training" / "wakeword_mmap"),
                FakeMmap.writes,
            )
            merged_records = [
                json.loads(line)
                for line in (root / "output" / "mining-manifest.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(
                {
                    record["item_index"]
                    for record in merged_records
                    if record["reason"].startswith("high_score:")
                },
                {0, 1, 2, 3},
            )


if __name__ == "__main__":
    unittest.main()
