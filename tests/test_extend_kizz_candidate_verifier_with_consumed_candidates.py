import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.extend_kizz_candidate_verifier_with_consumed_candidates import extend
from tools.train_kizz_candidate_verifier import load_verified_dataset, sha256_file


ARRAY_NAMES = (
    "features.npy",
    "labels.npy",
    "detector_scores.npy",
    "detector_feature_frames.npy",
    "detector_score_frames.npy",
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class DatasetFactory:
    def __init__(self, root: Path):
        self.root = root
        self.detector_a = root / "detector-a.tflite"
        self.detector_b = root / "detector-b.tflite"
        self.detector_a.write_bytes(b"detector-a")
        self.detector_b.write_bytes(b"detector-b")
        base_rows = [
            ("train", 1, "base-train-positive"),
            ("train", 0, "base-train-negative"),
            ("validation", 1, "base-validation-positive"),
            ("validation", 0, "base-validation-negative"),
            ("test", 1, "base-test-positive"),
            ("test", 0, "base-test-negative"),
        ]
        self.base = self.dataset("base", base_rows, detector="a")

    def detector(self, name: str) -> dict:
        path = self.detector_a if name == "a" else self.detector_b
        return {
            "artifact": {"path": str(path), "sha256": sha256_file(path)},
            "threshold": {"value": -10.0, "sha256_value": _sha(b"threshold-a")},
            "score_geometry": {"feature_stride_frames": 3},
        }

    def dataset(
        self,
        name: str,
        specifications: list[tuple[str, int, str]],
        *,
        detector: str = "a",
        overrides: dict[str, dict] | None = None,
        feature_values: dict[str, float] | None = None,
        top_k: int = 1,
    ) -> tuple[Path, str]:
        path = self.root / name
        path.mkdir()
        overrides = overrides or {}
        feature_values = feature_values or {}
        rows = []
        features = []
        labels = []
        scores = []
        feature_frames = []
        score_frames = []
        for index, (split, label, candidate_id) in enumerate(specifications):
            value = feature_values.get(candidate_id, float(len(list(self.root.iterdir())) * 100 + index + 1))
            feature = np.full((260, 40), value, dtype=np.float16)
            feature_hash = _sha(np.ascontiguousarray(feature).tobytes())
            audio_hash = _sha(f"audio:{name}:{candidate_id}".encode())
            row = {
                "source_id": candidate_id,
                "candidate_id": candidate_id,
                "parent_source_id": f"parent:{name}:{candidate_id}",
                "source_parent_source_id": f"parent:{name}:{candidate_id}",
                "speaker_id": f"speaker:{name}:{candidate_id}",
                "session_id": f"session:{name}:{candidate_id}",
                "ancestry_id": f"ancestry:{name}:{candidate_id}",
                "audio_sha256": audio_hash,
                "source_audio_sha256": audio_hash,
                "parent_source_audio_sha256": audio_hash,
                "source_group": "positive" if label else "negative",
                "semantic_label": "wake_word" if label else "non_wake",
                "provider": f"provider:{name}",
                "split": split,
                "label": label,
                "duration_seconds": 1.0,
                "detector_conditioned": True,
                "detector_score": float(index + 1),
                "detector_feature_frame_index": index + 10,
                "detector_score_frame_index": index + 20,
                "candidate_feature_sha256": feature_hash,
                "feature_sha256": feature_hash,
                "feature_index": index,
                "detector_event": {"score": float(index + 1)},
            }
            row.update(copy.deepcopy(overrides.get(candidate_id, {})))
            rows.append(row)
            features.append(feature)
            labels.append(label)
            scores.append(float(index + 1))
            feature_frames.append(index + 10)
            score_frames.append(index + 20)
        arrays = {
            "features.npy": np.asarray(features, dtype=np.float16),
            "labels.npy": np.asarray(labels, dtype=np.int8),
            "detector_scores.npy": np.asarray(scores, dtype=np.float32),
            "detector_feature_frames.npy": np.asarray(feature_frames, dtype=np.int32),
            "detector_score_frames.npy": np.asarray(score_frames, dtype=np.int32),
        }
        for array_name, values in arrays.items():
            np.save(path / array_name, values, allow_pickle=False)
        split_counts = {}
        for split in ("train", "validation", "test"):
            selected = [row for row in rows if row["split"] == split]
            positives = sum(row["label"] == 1 for row in selected)
            negatives = sum(row["label"] == 0 for row in selected)
            split_counts[split] = {
                "source_examples": len(selected),
                "exposure_seconds": float(len(selected)),
                "negative_exposure_seconds": float(negatives),
                "raw_detector_candidates": len(selected),
                "raw_positive_candidates": positives,
                "raw_negative_candidates": negatives,
                "selected_positive_candidates": positives,
                "selected_negative_candidates": negatives,
                "detector_missed_positives": 0,
                "detector_positive_source_recall": 1.0 if positives else None,
                "raw_candidate_rate_per_second": 1.0 if selected else 0.0,
                "raw_candidate_rate_per_hour": 3600.0 if selected else 0.0,
                "raw_negative_candidate_rate_per_hour": 3600.0 if negatives else 0.0,
            }
        train_negatives = sum(
            row["split"] == "train" and row["label"] == 0 for row in rows
        )
        heldout_negatives = sum(
            row["split"] != "train" and row["label"] == 0 for row in rows
        )
        corpus = {
            "schema_version": 1,
            "recipe": "kizz_control_candidate_conditioned_verifier_v1",
            "candidate_condition": "frozen_detector_trigger_only",
            "input_shape": [260, 40],
            "detector": self.detector(detector),
            "bindings": {},
            "hard_negative_selection": {
                "ranking": "detector_score_descending_then_candidate_id",
                "group_by": "source",
                "scope": "train_only",
                "top_k": top_k,
                "raw_training_count": train_negatives,
                "selected_training_count": train_negatives,
                "heldout_candidates_unfiltered": heldout_negatives,
            },
            "counts": {
                "selected_candidates": len(rows),
                "selected_positives": sum(row["label"] == 1 for row in rows),
                "selected_negatives": sum(row["label"] == 0 for row in rows),
                "detector_missed_positives": 0,
                "by_split": split_counts,
            },
            "examples": rows,
            "array_sha256": {
                array_name: sha256_file(path / array_name)
                for array_name in ARRAY_NAMES
            },
        }
        (path / "corpus.json").write_text(json.dumps(corpus, sort_keys=True))
        return path, sha256_file(path / "corpus.json")


class ExtendConsumedCandidatesTests(unittest.TestCase):
    def test_appends_roles_as_train_and_preserves_base_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            factory = DatasetFactory(Path(directory))
            consumed = factory.dataset(
                "consumed",
                [("validation", 1, "consumed-p"), ("test", 0, "ignored-n")],
            )
            auxiliary = factory.dataset(
                "auxiliary",
                [("train", 0, "aux-n"), ("validation", 0, "ignored-heldout")],
                detector="b",
            )
            output = Path(directory) / "output"
            result = extend(factory.base[0], factory.base[1], [consumed], [auxiliary], output)
            verified = load_verified_dataset(
                output, expected_corpus_sha256=result["corpus_sha256"]
            )
            base_corpus = json.loads((factory.base[0] / "corpus.json").read_text())
            self.assertEqual(list(verified.rows[:6]), base_corpus["examples"])
            for name in ARRAY_NAMES:
                np.testing.assert_array_equal(
                    np.load(output / name, allow_pickle=False)[:6],
                    np.load(factory.base[0] / name, allow_pickle=False),
                )
            appended = {row["candidate_id"]: row for row in verified.rows[6:]}
            self.assertEqual(set(appended), {"aux-n", "consumed-p"})
            self.assertEqual({row["split"] for row in appended.values()}, {"train"})
            self.assertEqual(appended["aux-n"]["original_split"], "train")
            self.assertEqual(appended["consumed-p"]["original_split"], "validation")
            self.assertEqual(appended["consumed-p"]["evidence_role"], "consumed_positive")
            self.assertEqual(appended["consumed-p"]["original_provider"], "provider:consumed")
            self.assertEqual(appended["consumed-p"]["provider"], "consumed_stackchan_physical")
            self.assertEqual(
                appended["consumed-p"]["source_candidate_corpus_sha256"], consumed[1]
            )
            self.assertEqual(result["consumed_positives"], 1)
            self.assertEqual(result["auxiliary_negatives"], 1)
            output_corpus = json.loads((output / "corpus.json").read_text())
            self.assertEqual(
                output_corpus["counts"]["by_split"]["validation"],
                base_corpus["counts"]["by_split"]["validation"],
            )
            self.assertEqual(
                output_corpus["counts"]["by_split"]["test"],
                base_corpus["counts"]["by_split"]["test"],
            )

    def test_is_deterministic_across_output_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            factory = DatasetFactory(Path(directory))
            one = factory.dataset("one", [("test", 1, "p-one")])
            two = factory.dataset("two", [("train", 0, "n-two")], detector="b")
            first = extend(factory.base[0], factory.base[1], [one], [two], Path(directory) / "out-a")
            second = extend(factory.base[0], factory.base[1], [one], [two], Path(directory) / "out-b")
            self.assertEqual(first["corpus_sha256"], second["corpus_sha256"])
            self.assertEqual(first["provenance_sha256"], second["provenance_sha256"])
            for name in (*ARRAY_NAMES, "corpus.json", "provenance.json"):
                self.assertEqual(
                    sha256_file(Path(directory) / "out-a" / name),
                    sha256_file(Path(directory) / "out-b" / name),
                )

    def test_consumed_positive_requires_exact_detector(self):
        with tempfile.TemporaryDirectory() as directory:
            factory = DatasetFactory(Path(directory))
            consumed = factory.dataset(
                "wrong-detector", [("test", 1, "positive")], detector="b"
            )
            with self.assertRaisesRegex(ValueError, "consumed_positive detector mismatch"):
                extend(factory.base[0], factory.base[1], [consumed], [], Path(directory) / "output")

    def test_auxiliary_detector_mismatch_is_permitted_and_marked(self):
        with tempfile.TemporaryDirectory() as directory:
            factory = DatasetFactory(Path(directory))
            auxiliary = factory.dataset(
                "old-negatives", [("train", 0, "old-negative")], detector="b"
            )
            output = Path(directory) / "output"
            extend(factory.base[0], factory.base[1], [], [auxiliary], output)
            row = json.loads((output / "corpus.json").read_text())["examples"][-1]
            self.assertEqual(row["source_detector_relation"], "mismatch")
            self.assertTrue(row["detector_mismatch_permitted"])
            self.assertEqual(
                row["candidate_distribution_role"],
                "older_detector_auxiliary_negative",
            )

    def test_rejects_identity_or_hash_overlap_with_base_heldout(self):
        with tempfile.TemporaryDirectory() as directory:
            factory = DatasetFactory(Path(directory))
            base_rows = json.loads((factory.base[0] / "corpus.json").read_text())["examples"]
            heldout = base_rows[2]
            identity = factory.dataset(
                "identity-overlap",
                [("train", 0, "identity-negative")],
                detector="b",
                overrides={
                    "identity-negative": {"parent_source_id": heldout["parent_source_id"]}
                },
            )
            with self.assertRaisesRegex(ValueError, "identity overlaps base validation/test"):
                extend(factory.base[0], factory.base[1], [], [identity], Path(directory) / "identity-output")

            hash_overlap = factory.dataset(
                "hash-overlap",
                [("test", 1, "hash-positive")],
                overrides={"hash-positive": {"audio_sha256": heldout["audio_sha256"]}},
            )
            with self.assertRaisesRegex(ValueError, "hash overlaps base validation/test"):
                extend(factory.base[0], factory.base[1], [hash_overlap], [], Path(directory) / "hash-output")

    def test_dedupes_candidate_ids_and_feature_hashes_against_train(self):
        with tempfile.TemporaryDirectory() as directory:
            factory = DatasetFactory(Path(directory))
            base_features = np.load(factory.base[0] / "features.npy", allow_pickle=False)
            duplicate_id = factory.dataset(
                "duplicate-id",
                [("train", 0, "base-train-negative"), ("train", 0, "unique-negative")],
                detector="b",
            )
            duplicate_feature = factory.dataset(
                "duplicate-feature",
                [("test", 1, "duplicate-feature-positive")],
                feature_values={"duplicate-feature-positive": float(base_features[0, 0, 0])},
            )
            output = Path(directory) / "output"
            result = extend(
                factory.base[0], factory.base[1], [duplicate_feature], [duplicate_id], output
            )
            self.assertEqual(result["appended_rows"], 1)
            self.assertEqual(result["deduplicated"], 2)
            self.assertEqual(
                json.loads((output / "corpus.json").read_text())["examples"][-1]["candidate_id"],
                "unique-negative",
            )

    def test_updates_top_k_to_actual_grouping_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            factory = DatasetFactory(Path(directory))
            shared = "shared-old-source"
            auxiliary = factory.dataset(
                "grouped-old-negatives",
                [("train", 0, "n-1"), ("train", 0, "n-2"), ("train", 0, "n-3")],
                detector="b",
                overrides={
                    candidate: {"parent_source_id": shared}
                    for candidate in ("n-1", "n-2", "n-3")
                },
                top_k=3,
            )
            output = Path(directory) / "output"
            result = extend(factory.base[0], factory.base[1], [], [auxiliary], output)
            corpus = json.loads((output / "corpus.json").read_text())
            self.assertEqual(corpus["hard_negative_selection"]["top_k"], 3)
            self.assertEqual(corpus["hard_negative_selection"]["selected_training_count"], 4)
            self.assertEqual(corpus["hard_negative_selection"]["raw_training_count"], 4)
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                extend(factory.base[0], factory.base[1], [], [auxiliary], output)
            load_verified_dataset(output, expected_corpus_sha256=result["corpus_sha256"])

    def test_hash_drift_fails_without_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            factory = DatasetFactory(Path(directory))
            source = factory.dataset("source", [("test", 1, "positive")])
            values = np.load(source[0] / "features.npy", allow_pickle=False)
            values[0, 0, 0] += 1
            np.save(source[0] / "features.npy", values, allow_pickle=False)
            output = Path(directory) / "output"
            with self.assertRaisesRegex(ValueError, "features.npy hash drift"):
                extend(factory.base[0], factory.base[1], [source], [], output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
