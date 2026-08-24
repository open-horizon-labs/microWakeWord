import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tools.write_ordered_state_scoring_manifests import (
    build_manifest,
    sha256_path,
    write_manifests,
)


class _FakeRaggedMmap:
    items_by_path = {}

    def __init__(self, path):
        self.items = self.items_by_path[str(path)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class WriteOrderedStateScoringManifestsTest(unittest.TestCase):
    def _root(self, directory, name):
        path = Path(directory) / name
        path.mkdir(parents=True)
        return path

    def _frozen(self, path, *, validation_exposure=360_000, include_train=False):
        source_root = Path(path).parent
        sources = [
            {
                "source_id": "validation-negative",
                "split": "validation",
                "path": str(source_root / "validation-negative"),
                "path_sha256": "v" * 64,
                "exposure_seconds": validation_exposure,
                "category": "speech/TV",
                "channel": "evaluation-channel",
                "session_id": "validation-session",
                "speaker_id": "validation-speakers",
                "source_family": "CHiME-6-dev",
                "truth": "negative",
            },
            {
                "source_id": "test-negative",
                "split": "test",
                "path": str(source_root / "test-negative"),
                "path_sha256": "t" * 64,
                "exposure_seconds": 360_000,
                "category": "speech/TV",
                "channel": "evaluation-channel",
                "session_id": "test-session",
                "speaker_id": "test-speakers",
                "source_family": "VOiCES",
                "truth": "negative",
            },
        ]
        if include_train:
            sources.append(
                {
                    "source_id": "train-negative",
                    "split": "train",
                    "path": str(source_root / "train-negative"),
                    "path_sha256": "x" * 64,
                    "exposure_seconds": 360_000,
                    "category": "speech/TV",
                    "channel": "train",
                }
            )
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "experiment": "ordered-test",
                    "threshold_selection_split": "validation",
                    "sources": sources,
                }
            )
        )

    def _positive(self, directory):
        root = Path(directory) / "features"
        validation = root / "positive" / "validation" / "wakeword_mmap"
        testing = root / "positive" / "testing" / "wakeword_mmap"
        validation.mkdir(parents=True)
        testing.mkdir(parents=True)
        return root, validation, testing

    def test_split_mapping_exact_occurrences_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            frozen_path = base / "frozen.json"
            self._frozen(frozen_path, validation_exposure=360_000)
            for name in ("validation-negative", "test-negative"):
                (base / name).mkdir()
                (base / name / "data.ninja").write_bytes(b"negative")
            root, validation, testing = self._positive(base)
            _FakeRaggedMmap.items_by_path[str(validation.resolve())] = [
                np.zeros((3, 40), dtype=np.float32),
                np.zeros((7, 40), dtype=np.float32),
            ]
            _FakeRaggedMmap.items_by_path[str(testing.resolve())] = [
                np.zeros((5, 40), dtype=np.float32),
            ]
            with mock.patch(
                "tools.write_ordered_state_scoring_manifests.RaggedMmap",
                _FakeRaggedMmap,
            ):
                validation_manifest, test_manifest = write_manifests(
                    frozen_path,
                    root,
                    base / "validation.json",
                    base / "test.json",
                )
            self.assertEqual(
                validation_manifest["split_mapping"]["positive"], "validation"
            )
            self.assertEqual(test_manifest["split_mapping"]["positive"], "testing")
            self.assertTrue(validation_manifest["threshold_selection_allowed"])
            self.assertFalse(test_manifest["threshold_selection_allowed"])
            self.assertTrue(test_manifest["test_is_untouched"])
            positive = next(
                item
                for item in validation_manifest["sources"]
                if item["label"] == "positive"
            )
            self.assertEqual(
                [
                    (item["start_seconds"], item["end_seconds"])
                    for item in positive["occurrences"]
                ],
                [(0.0, 0.03), (0.0, 0.07)],
            )
            self.assertEqual(
                positive["occurrences"][1]["id"],
                "canonical-positive-validation-item-000001",
            )
            negative = next(
                item
                for item in validation_manifest["sources"]
                if item["label"] == "negative"
            )
            self.assertEqual(negative["exposure_seconds"], 360_000.0)
            self.assertEqual(negative["expected_path_sha256"], sha256_path(base / "validation-negative"))
            self.assertEqual(negative["frozen_path_sha256"], "v" * 64)
            self.assertEqual(
                validation_manifest["input_hashes"]["frozen_negative_manifest_sha256"],
                hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
            )

    def test_rejects_less_than_100_negative_hours(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            frozen_path = base / "frozen.json"
            self._frozen(frozen_path, validation_exposure=359_999)
            (base / "validation-negative").mkdir()
            root, validation, _ = self._positive(base)
            _FakeRaggedMmap.items_by_path[str(validation.resolve())] = [
                np.zeros((1, 40))
            ]
            with (
                mock.patch(
                    "tools.write_ordered_state_scoring_manifests.RaggedMmap",
                    _FakeRaggedMmap,
                ),
                self.assertRaisesRegex(ValueError, "100 hours"),
            ):
                build_manifest(frozen_path, root, "validation")

    def test_exact_occurrence_manifest_replaces_full_item_proxy(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            frozen_path = base / "frozen.json"
            self._frozen(frozen_path)
            (base / "validation-negative").mkdir()
            root, validation, testing = self._positive(base)
            validation.joinpath("data.ninja").write_bytes(b"validation")
            testing.joinpath("data.ninja").write_bytes(b"testing")
            _FakeRaggedMmap.items_by_path[str(validation.resolve())] = [
                np.zeros((20, 40))
            ]
            _FakeRaggedMmap.items_by_path[str(testing.resolve())] = [np.zeros((20, 40))]
            occurrences = base / "occurrences.json"
            occurrences.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "ragged_mmap_sha256": {
                            "validation": sha256_path(validation),
                            "test": sha256_path(testing),
                        },
                        "occurrences": [
                            {
                                "source_id": "v-exact",
                                "source_group": "v-speaker",
                                "split": "validation",
                                "item_index": 0,
                                "phrase_span": {"start_s": 0.04, "end_s": 0.16},
                            },
                            {
                                "source_id": "t-exact",
                                "source_group": "t-speaker",
                                "split": "test",
                                "item_index": 0,
                                "phrase_span": {"start_s": 0.03, "end_s": 0.17},
                            },
                        ],
                    }
                )
            )
            with mock.patch(
                "tools.write_ordered_state_scoring_manifests.RaggedMmap",
                _FakeRaggedMmap,
            ):
                manifest = build_manifest(frozen_path, root, "validation", occurrences)
            positive = next(
                item for item in manifest["sources"] if item["label"] == "positive"
            )
            self.assertEqual(
                manifest["positive_occurrence_geometry"], "exact_phrase_span"
            )
            self.assertEqual(positive["occurrences"][0]["start_seconds"], 0.04)
            self.assertEqual(positive["occurrences"][0]["end_seconds"], 0.16)

    def test_omits_train_and_rejects_quarantined_positive_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            frozen_path = base / "frozen.json"
            self._frozen(frozen_path, include_train=True)
            (base / "validation-negative").mkdir()
            (base / "train-negative").mkdir()
            root, validation, _ = self._positive(base)
            _FakeRaggedMmap.items_by_path[str(validation.resolve())] = [
                np.zeros((2, 40))
            ]
            with mock.patch(
                "tools.write_ordered_state_scoring_manifests.RaggedMmap",
                _FakeRaggedMmap,
            ):
                manifest = build_manifest(frozen_path, root, "validation")
            self.assertNotIn(
                "negative-train-negative", [item["id"] for item in manifest["sources"]]
            )

            self._frozen(frozen_path)
            quarantined = base / "observations" / "false-wakes" / "features"
            (quarantined / "positive" / "validation" / "wakeword_mmap").mkdir(
                parents=True
            )
            with self.assertRaisesRegex(ValueError, "quarantined"):
                build_manifest(frozen_path, quarantined, "validation")

    def test_rejects_non_feature_items_and_wrong_frozen_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            frozen_path = base / "frozen.json"
            self._frozen(frozen_path)
            (base / "validation-negative").mkdir()
            root, validation, _ = self._positive(base)
            _FakeRaggedMmap.items_by_path[str(validation.resolve())] = [
                np.zeros((2, 39))
            ]
            with (
                mock.patch(
                    "tools.write_ordered_state_scoring_manifests.RaggedMmap",
                    _FakeRaggedMmap,
                ),
                self.assertRaisesRegex(ValueError, r"\[N, 40\]"),
            ):
                build_manifest(frozen_path, root, "validation")
            payload = json.loads(frozen_path.read_text())
            payload["threshold_selection_split"] = "test"
            frozen_path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "thresholds on validation"):
                build_manifest(frozen_path, root, "validation")


if __name__ == "__main__":
    unittest.main()
