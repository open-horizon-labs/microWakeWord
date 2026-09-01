import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import tools.merge_kizz_candidate_verifier_extensions as merger
from tools.merge_kizz_candidate_verifier_extensions import ARRAY_NAMES, merge_extensions
from tools.train_kizz_candidate_verifier import load_verified_dataset, sha256_file


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


class DatasetFactory:
    def __init__(self, root: Path):
        self.root = root
        self.bound = root / "bound"
        self.bound.mkdir()
        self.detector_files = {}
        for name in ("artifact", "config", "threshold", "source", "features", "traces", "holdout"):
            path = self.bound / f"{name}.bin"
            path.write_bytes(f"bound:{name}".encode())
            self.detector_files[name] = path
        self.base = self._write_base()

    @staticmethod
    def _file_binding(path: Path):
        return {"path": str(path.resolve()), "sha256": sha256_file(path)}

    @staticmethod
    def _row(name: str, split: str, label: int, feature: np.ndarray, score: float):
        source = f"source:{name}"
        audio_hash = _digest(f"audio:{name}")
        feature_hash = hashlib.sha256(np.ascontiguousarray(feature).tobytes()).hexdigest()
        row = {
            "candidate_id": f"candidate:{name}",
            "source_id": f"candidate:{name}",
            "parent_source_id": source,
            "source_parent_source_id": source,
            "speaker_id": f"speaker:{name}",
            "session_id": f"session:{name}",
            "ancestry_id": source,
            "audio_sha256": audio_hash,
            "source_audio_sha256": audio_hash,
            "parent_source_audio_sha256": audio_hash,
            "feature_sha256": feature_hash,
            "candidate_feature_sha256": feature_hash,
            "split": split,
            "label": label,
            "detector_conditioned": True,
            "detector_score": score,
        }
        if label:
            row["provider"] = f"provider:{name}"
        else:
            row["source_group"] = f"negative:{name}"
        return row

    def _write_dataset(self, path: Path, corpus: dict, arrays: dict[str, np.ndarray]):
        path.mkdir()
        for name, values in arrays.items():
            np.save(path / name, values, allow_pickle=False)
        corpus["array_sha256"] = {
            name: sha256_file(path / name) for name in ARRAY_NAMES
        }
        _write_json(path / "corpus.json", corpus)
        return path, sha256_file(path / "corpus.json")

    def _write_base(self):
        definitions = [
            ("train-positive", "train", 1),
            ("train-negative", "train", 0),
            ("validation-positive", "validation", 1),
            ("validation-negative", "validation", 0),
            ("test-positive", "test", 1),
            ("test-negative", "test", 0),
        ]
        count = len(definitions)
        features = np.zeros((count, 260, 40), dtype=np.float16)
        labels = np.zeros(count, dtype=np.int8)
        scores = np.zeros(count, dtype=np.float32)
        rows = []
        for index, (name, split, label) in enumerate(definitions):
            features[index, 0, 0] = index + 1
            labels[index] = label
            scores[index] = 0.6 + index / 100
            row = self._row(name, split, label, features[index], float(scores[index]))
            row["feature_index"] = index
            rows.append(row)
        arrays = {
            "features.npy": features,
            "labels.npy": labels,
            "detector_scores.npy": scores,
            "detector_feature_frames.npy": np.arange(count, dtype=np.int32) + 100,
            "detector_score_frames.npy": np.arange(count, dtype=np.int32) + 200,
        }
        corpus = {
            "schema_version": 1,
            "recipe": "kizz_control_candidate_conditioned_verifier_v1",
            "candidate_condition": "frozen_detector_trigger_only",
            "detector": {
                "artifact": self._file_binding(self.detector_files["artifact"]),
                "config": self._file_binding(self.detector_files["config"]),
                "threshold": {
                    **self._file_binding(self.detector_files["threshold"]),
                    "value": 0.5,
                },
            },
            "bindings": {
                "source_manifest": self._file_binding(self.detector_files["source"]),
                "source_features": self._file_binding(self.detector_files["features"]),
                "detector_traces": self._file_binding(self.detector_files["traces"]),
                "locked_holdout": self._file_binding(self.detector_files["holdout"]),
            },
            "hard_negative_selection": {
                "ranking": "detector_score_descending_then_candidate_id",
                "top_k": 4,
                "group_by": "source",
                "scope": "train_only",
                "raw_training_count": 1,
                "selected_training_count": 1,
                "heldout_candidates_unfiltered": 2,
            },
            "counts": {
                "selected_candidates": count,
                "selected_positives": 3,
                "selected_negatives": 3,
                "detector_missed_positives": 0,
                "by_split": {
                    split: {
                        "source_examples": 2,
                        "exposure_seconds": 20.0,
                        "raw_detector_candidates": 2,
                        "raw_positive_candidates": 1,
                        "raw_negative_candidates": 1,
                        "selected_positive_candidates": 1,
                        "selected_negative_candidates": 1,
                        "detector_missed_positives": 0,
                        "detector_positive_source_recall": 1.0,
                        "raw_candidate_rate_per_second": 0.1,
                        "raw_candidate_rate_per_hour": 360.0,
                        "negative_exposure_seconds": 10.0,
                        "raw_negative_candidate_rate_per_hour": 360.0,
                    }
                    for split in ("train", "validation", "test")
                },
            },
            "examples": rows,
        }
        return self._write_dataset(self.root / "base", corpus, arrays)

    def extension(
        self,
        name: str,
        *,
        split: str,
        value: float,
        candidate_id: str | None = None,
        identity: str | None = None,
        source_hash: str | None = None,
        feature_from: np.ndarray | None = None,
    ):
        base_path, base_hash = self.base
        corpus = json.loads((base_path / "corpus.json").read_text())
        arrays = {
            array_name: np.load(base_path / array_name, allow_pickle=False)
            for array_name in ARRAY_NAMES
        }
        feature = np.zeros((260, 40), dtype=np.float16)
        feature[0, 0] = value
        if feature_from is not None:
            feature = np.asarray(feature_from, dtype=np.float16).copy()
        score = np.float32(0.9 + value / 1000)
        row = self._row(name, split, 0, feature, float(score))
        if candidate_id is not None:
            row["candidate_id"] = candidate_id
            row["source_id"] = candidate_id
        if identity is not None:
            row["parent_source_id"] = identity
            row["source_parent_source_id"] = identity
            row["speaker_id"] = identity
            row["session_id"] = identity
            row["ancestry_id"] = identity
        if source_hash is not None:
            row["audio_sha256"] = source_hash
            row["source_audio_sha256"] = source_hash
            row["parent_source_audio_sha256"] = source_hash
        row["feature_index"] = len(corpus["examples"])
        corpus["examples"].append(row)
        arrays["features.npy"] = np.concatenate([arrays["features.npy"], feature[None]])
        arrays["labels.npy"] = np.concatenate(
            [arrays["labels.npy"], np.asarray([0], dtype=np.int8)]
        )
        arrays["detector_scores.npy"] = np.concatenate(
            [arrays["detector_scores.npy"], np.asarray([score], dtype=np.float32)]
        )
        arrays["detector_feature_frames.npy"] = np.concatenate(
            [arrays["detector_feature_frames.npy"], np.asarray([300], dtype=np.int32)]
        )
        arrays["detector_score_frames.npy"] = np.concatenate(
            [arrays["detector_score_frames.npy"], np.asarray([400], dtype=np.int32)]
        )

        corpus["counts"]["selected_candidates"] += 1
        corpus["counts"]["selected_negatives"] += 1
        counts = corpus["counts"]["by_split"][split]
        counts["source_examples"] += 1
        counts["exposure_seconds"] += 10.0
        counts["negative_exposure_seconds"] += 10.0
        counts["raw_detector_candidates"] += 1
        counts["raw_negative_candidates"] += 1
        counts["selected_negative_candidates"] += 1
        counts["raw_candidate_rate_per_second"] = (
            counts["raw_detector_candidates"] / counts["exposure_seconds"]
        )
        counts["raw_candidate_rate_per_hour"] = (
            counts["raw_candidate_rate_per_second"] * 3600
        )
        counts["raw_negative_candidate_rate_per_hour"] = (
            counts["raw_negative_candidates"]
            * 3600
            / counts["negative_exposure_seconds"]
        )
        if split == "train":
            corpus["hard_negative_selection"]["raw_training_count"] += 1
            corpus["hard_negative_selection"]["selected_training_count"] += 1
        else:
            corpus["hard_negative_selection"]["heldout_candidates_unfiltered"] += 1

        ledger = self.bound / f"ledger-{name}.json"
        _write_json(ledger, {"extension": name, "split": split})
        corpus["bindings"]["base_candidate_corpus"] = self._file_binding(
            base_path / "corpus.json"
        )
        corpus["bindings"]["extension_source_ledger"] = self._file_binding(ledger)
        corpus["extension"] = {"name": name, "appended_rows": 1}
        path, digest = self._write_dataset(self.root / name, corpus, arrays)
        return path, digest, ledger


class MergeCandidateVerifierExtensionsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.factory = DatasetFactory(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def test_merges_tails_deterministically_and_recomputes_counts(self):
        train = self.factory.extension("fma", split="train", value=11.0)
        validation = self.factory.extension(
            "librispeech", split="validation", value=12.0
        )
        base_path, base_hash = self.factory.base
        first = self.root / "merged-first"
        second = self.root / "merged-second"
        result = merge_extensions(
            base_path,
            base_hash,
            [(train[0], train[1]), (validation[0], validation[1])],
            first,
        )
        merge_extensions(
            base_path,
            base_hash,
            [(validation[0], validation[1]), (train[0], train[1])],
            second,
        )

        self.assertEqual(result["appended_rows"], 2)
        self.assertEqual((first / "corpus.json").read_bytes(), (second / "corpus.json").read_bytes())
        for name in ARRAY_NAMES:
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
        corpus = json.loads((first / "corpus.json").read_text())
        expected_extensions = sorted([train, validation], key=lambda item: (item[1], str(item[0].resolve())))
        self.assertEqual(
            [row["candidate_id"] for row in corpus["examples"][6:]],
            [f"candidate:{item[0].name}" for item in expected_extensions],
        )
        self.assertEqual(
            [row["feature_index"] for row in corpus["examples"]], list(range(8))
        )
        self.assertEqual(corpus["counts"]["selected_candidates"], 8)
        self.assertEqual(corpus["counts"]["selected_positives"], 3)
        self.assertEqual(corpus["counts"]["selected_negatives"], 5)
        self.assertEqual(corpus["hard_negative_selection"]["raw_training_count"], 2)
        self.assertEqual(corpus["hard_negative_selection"]["selected_training_count"], 2)
        self.assertEqual(corpus["hard_negative_selection"]["heldout_candidates_unfiltered"], 3)
        for split in ("train", "validation"):
            counts = corpus["counts"]["by_split"][split]
            self.assertEqual(counts["source_examples"], 3)
            self.assertEqual(counts["exposure_seconds"], 30.0)
            self.assertEqual(counts["negative_exposure_seconds"], 20.0)
            self.assertEqual(counts["raw_detector_candidates"], 3)
            self.assertEqual(counts["raw_negative_candidates"], 2)
            self.assertEqual(counts["raw_candidate_rate_per_hour"], 360.0)
            self.assertEqual(counts["raw_negative_candidate_rate_per_hour"], 360.0)
        verified = load_verified_dataset(first, expected_corpus_sha256=result["corpus_sha256"])
        self.assertEqual(len(verified.rows), 8)

        train[2].write_text("drift\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash drift"):
            load_verified_dataset(first, expected_corpus_sha256=result["corpus_sha256"])

    def test_rejects_changed_base_rows_or_any_base_array(self):
        good = self.factory.extension("good", split="train", value=20.0)
        row_drift = self.factory.extension("row-drift", split="train", value=21.0)
        corpus_path = row_drift[0] / "corpus.json"
        corpus = json.loads(corpus_path.read_text())
        corpus["examples"][0]["provider"] = "drifted-provider"
        _write_json(corpus_path, corpus)
        row_drift = (row_drift[0], sha256_file(corpus_path), row_drift[2])
        with self.assertRaisesRegex(ValueError, "inherited base rows differ"):
            merge_extensions(
                self.factory.base[0], self.factory.base[1],
                [(good[0], good[1]), (row_drift[0], row_drift[1])],
                self.root / "row-drift-output",
            )

        array_drift = self.factory.extension("array-drift", split="validation", value=22.0)
        frames_path = array_drift[0] / "detector_feature_frames.npy"
        frames = np.load(frames_path, allow_pickle=False)
        frames[0] += 1
        np.save(frames_path, frames, allow_pickle=False)
        corpus_path = array_drift[0] / "corpus.json"
        corpus = json.loads(corpus_path.read_text())
        corpus["array_sha256"][frames_path.name] = sha256_file(frames_path)
        _write_json(corpus_path, corpus)
        array_drift = (array_drift[0], sha256_file(corpus_path), array_drift[2])
        with self.assertRaisesRegex(ValueError, "inherited detector_feature_frames.npy differs"):
            merge_extensions(
                self.factory.base[0], self.factory.base[1],
                [(good[0], good[1]), (array_drift[0], array_drift[1])],
                self.root / "array-drift-output",
            )

        short_tail = self.factory.extension("short-tail", split="validation", value=23.0)
        score_frames_path = short_tail[0] / "detector_score_frames.npy"
        score_frames = np.load(score_frames_path, allow_pickle=False)[:-1]
        np.save(score_frames_path, score_frames, allow_pickle=False)
        corpus_path = short_tail[0] / "corpus.json"
        corpus = json.loads(corpus_path.read_text())
        corpus["array_sha256"][score_frames_path.name] = sha256_file(score_frames_path)
        _write_json(corpus_path, corpus)
        short_tail = (short_tail[0], sha256_file(corpus_path), short_tail[2])
        with self.assertRaisesRegex(ValueError, "must contain one value per corpus row"):
            merge_extensions(
                self.factory.base[0], self.factory.base[1],
                [(good[0], good[1]), (short_tail[0], short_tail[1])],
                self.root / "short-tail-output",
            )

    def test_rejects_cross_extension_duplicates_and_split_leakage(self):
        first = self.factory.extension("first", split="train", value=30.0)
        duplicate_id = self.factory.extension(
            "duplicate-id", split="validation", value=31.0,
            candidate_id="candidate:first",
        )
        with self.assertRaisesRegex(ValueError, "duplicate or empty candidate IDs"):
            merge_extensions(
                self.factory.base[0], self.factory.base[1],
                [(first[0], first[1]), (duplicate_id[0], duplicate_id[1])],
                self.root / "duplicate-id-output",
            )

        duplicate_feature = self.factory.extension(
            "duplicate-feature", split="validation", value=32.0,
            feature_from=np.load(first[0] / "features.npy", allow_pickle=False)[-1],
        )
        with self.assertRaisesRegex(ValueError, "duplicate or empty candidate feature hashes"):
            merge_extensions(
                self.factory.base[0], self.factory.base[1],
                [(first[0], first[1]), (duplicate_feature[0], duplicate_feature[1])],
                self.root / "duplicate-feature-output",
            )

        shared_identity = "shared-cross-extension-identity"
        train_identity = self.factory.extension(
            "train-identity", split="train", value=33.0, identity=shared_identity
        )
        validation_identity = self.factory.extension(
            "validation-identity", split="validation", value=34.0,
            identity=shared_identity,
        )
        with self.assertRaisesRegex(ValueError, "train/validation identity overlap"):
            merge_extensions(
                self.factory.base[0], self.factory.base[1],
                [(train_identity[0], train_identity[1]), (validation_identity[0], validation_identity[1])],
                self.root / "identity-leak-output",
            )

        shared_hash = _digest("shared-cross-extension-audio")
        train_hash = self.factory.extension(
            "train-hash", split="train", value=35.0, source_hash=shared_hash
        )
        test_hash = self.factory.extension(
            "test-hash", split="test", value=36.0, source_hash=shared_hash
        )
        with self.assertRaisesRegex(ValueError, "train/test hash overlap"):
            merge_extensions(
                self.factory.base[0], self.factory.base[1],
                [(train_hash[0], train_hash[1]), (test_hash[0], test_hash[1])],
                self.root / "hash-leak-output",
            )

    def test_verifies_hashes_refuses_overwrite_and_leaves_no_partial_output(self):
        first = self.factory.extension("hash-first", split="train", value=40.0)
        second = self.factory.extension("hash-second", split="validation", value=41.0)
        bad_features = second[0] / "features.npy"
        values = np.load(bad_features, allow_pickle=False)
        values[-1, 0, 1] = 99
        np.save(bad_features, values, allow_pickle=False)
        with self.assertRaisesRegex(ValueError, "features.npy hash drift"):
            merge_extensions(
                self.factory.base[0], self.factory.base[1],
                [(first[0], first[1]), (second[0], second[1])],
                self.root / "hash-output",
            )

        second = self.factory.extension("atomic-second", split="validation", value=42.0)
        existing = self.root / "existing"
        existing.mkdir()
        sentinel = existing / "sentinel"
        sentinel.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
            merge_extensions(
                self.factory.base[0], self.factory.base[1],
                [(first[0], first[1]), (second[0], second[1])], existing,
            )
        self.assertEqual(sentinel.read_text(), "keep")

        output = self.root / "atomic-output"
        with mock.patch.object(merger, "_atomic_npy", side_effect=RuntimeError("write failed")):
            with self.assertRaisesRegex(RuntimeError, "write failed"):
                merge_extensions(
                    self.factory.base[0], self.factory.base[1],
                    [(first[0], first[1]), (second[0], second[1])], output,
                )
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob(".atomic-output.*")), [])


if __name__ == "__main__":
    unittest.main()
