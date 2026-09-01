import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.build_kizz_candidate_verifier_dataset import (
    build_candidate_verifier_dataset,
    candidate_window,
    sha256_file,
    threshold_region_events,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _feature_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


class CandidateVerifierFixture:
    def __init__(self, root: Path, rows, traces, *, event_policy="recorded_events"):
        self.root = root
        self.features = root / "source-features.npy"
        values = np.arange(len(rows) * 8 * 2, dtype=np.float32).reshape(len(rows), 8, 2)
        np.save(self.features, values)
        self.source = root / "source.json"
        source_rows = []
        for index, row in enumerate(rows):
            item = {
                "feature_index": index,
                "duration_seconds": 0.08,
                "speaker_id": f"speaker-{index}",
                "session_id": f"session-{index}",
                "ancestry_id": f"ancestry-{index}",
                "audio_sha256": _hash(f"audio-{index}"),
                "feature_sha256": _feature_hash(values[index]),
                **row,
            }
            source_rows.append(item)
        self.source.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "array_sha256": {self.features.name: sha256_file(self.features)},
                    "examples": source_rows,
                },
                sort_keys=True,
            )
        )
        self.artifact = root / "detector.tflite"
        self.config = root / "detector-config.json"
        self.threshold = root / "detector-threshold.json"
        self.artifact.write_bytes(b"frozen-detector")
        self.config.write_text('{"stride":1}')
        self.threshold.write_text('{"threshold":0.5}')
        self.lock = root / "locked-holdout.json"
        self.lock.write_text(
            json.dumps(
                {
                    "gate_scope": "locked_untouched_continuous_negative_corpus",
                    "locked_before_scoring": True,
                    "examples": [],
                },
                sort_keys=True,
            )
        )
        self.traces = root / "detector-traces.json"
        detector = {
            "artifact": self._binding(self.artifact),
            "config": self._binding(self.config),
            "threshold": {**self._binding(self.threshold), "value": 0.5},
            "event_policy": event_policy,
            "score_geometry": {
                "feature_stride_frames": 1,
                "feature_offset_frames": 0,
                "feature_hop_ms": 10,
            },
        }
        source_by_id = {row["source_id"]: row for row in source_rows}
        bound_traces = []
        for trace in traces:
            source = source_by_id[trace["source_id"]]
            bound_traces.append(
                {
                    **trace,
                    "feature_index": source["feature_index"],
                    "split": source["split"],
                    "label": int(source["label"]),
                    "source_feature_sha256": source["feature_sha256"],
                }
            )
        self.traces.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "source_manifest": self._binding(self.source),
                    "source_features": self._binding(self.features),
                    "detector": detector,
                    "examples": bound_traces,
                },
                sort_keys=True,
            )
        )

    @staticmethod
    def _binding(path: Path):
        return {"path": str(path.resolve()), "sha256": sha256_file(path)}


class BuildCandidateVerifierDatasetTests(unittest.TestCase):
    def test_identical_derived_features_do_not_imply_cross_split_source_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = CandidateVerifierFixture(
                root,
                [
                    {"source_id": "train-silence", "split": "train", "label": 0},
                    {
                        "source_id": "validation-silence",
                        "split": "validation",
                        "label": 0,
                    },
                ],
                [
                    {"source_id": "train-silence", "scores": [0.8]},
                    {"source_id": "validation-silence", "scores": [0.8]},
                ],
                event_policy="threshold_regions",
            )
            features = np.load(fixture.features, allow_pickle=False)
            features[1] = features[0]
            np.save(fixture.features, features)
            payload = json.loads(fixture.source.read_text())
            shared_hash = _feature_hash(features[0])
            for row in payload["examples"]:
                row["feature_sha256"] = shared_hash
            payload["array_sha256"][fixture.features.name] = sha256_file(
                fixture.features
            )
            fixture.source.write_text(json.dumps(payload, sort_keys=True))
            trace_payload = json.loads(fixture.traces.read_text())
            trace_payload["source_manifest"] = fixture._binding(fixture.source)
            trace_payload["source_features"] = fixture._binding(fixture.features)
            for row in trace_payload["examples"]:
                row["source_feature_sha256"] = shared_hash
            fixture.traces.write_text(json.dumps(trace_payload, sort_keys=True))

            report = build_candidate_verifier_dataset(
                fixture.source,
                fixture.features,
                fixture.traces,
                root / "out",
                locked_holdout_manifest=fixture.lock,
                pre_context_frames=7,
                post_context_frames=0,
            )
            self.assertEqual(report["counts"]["selected_candidates"], 2)

    def test_only_detector_triggers_become_rows_and_positive_misses_remain_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = CandidateVerifierFixture(
                root,
                [
                    {"source_id": "train-positive", "split": "train", "label": 1},
                    {
                        "source_id": "validation-positive-miss",
                        "split": "validation",
                        "label": 1,
                    },
                    {"source_id": "train-negative", "split": "train", "label": 0},
                    {"source_id": "test-negative", "split": "test", "label": 0},
                ],
                [
                    {
                        "source_id": "train-positive",
                        "scores": [0.1, 0.2, 0.8, 0.1],
                        "events": [{"score_frame_index": 2, "score": 0.8}],
                    },
                    {
                        "source_id": "validation-positive-miss",
                        "scores": [0.1, 0.2, 0.49, 0.1],
                        "events": [],
                    },
                    {
                        "source_id": "train-negative",
                        "scores": [0.2, 0.7, 0.1, 0.6],
                        "events": [
                            {"score_frame_index": 1, "score": 0.7},
                            {"score_frame_index": 3, "score": 0.6},
                        ],
                    },
                    {
                        "source_id": "test-negative",
                        "scores": [0.6, 0.2, 0.7, 0.4],
                        "events": [
                            {"score_frame_index": 0, "score": 0.6},
                            {"score_frame_index": 2, "score": 0.7},
                        ],
                    },
                ],
            )
            report = build_candidate_verifier_dataset(
                fixture.source,
                fixture.features,
                fixture.traces,
                root / "output",
                locked_holdout_manifest=fixture.lock,
                pre_context_frames=1,
                post_context_frames=1,
                hard_negative_top_k=1,
            )

            self.assertEqual(report["counts"]["selected_candidates"], 4)
            self.assertEqual([row["label"] for row in report["examples"]], [1, 0, 0, 0])
            self.assertTrue(all(row["detector_conditioned"] for row in report["examples"]))
            self.assertTrue(
                all(len(row["candidate_feature_sha256"]) == 64 for row in report["examples"])
            )
            self.assertEqual(
                [row["parent_source_id"] for row in report["examples"]],
                ["train-positive", "train-negative", "test-negative", "test-negative"],
            )
            self.assertEqual(len(report["detector_misses"]), 1)
            miss = report["detector_misses"][0]
            self.assertEqual(miss["source_id"], "validation-positive-miss")
            self.assertTrue(miss["detector_miss"])
            self.assertAlmostEqual(miss["maximum_detector_score"], 0.49)
            self.assertEqual(
                report["counts"]["by_split"]["train"]["raw_detector_candidates"], 3
            )
            self.assertEqual(
                report["counts"]["by_split"]["test"]["selected_negative_candidates"],
                2,
            )
            self.assertEqual(
                report["counts"]["by_split"]["test"][
                    "raw_negative_candidate_rate_per_hour"
                ],
                90000.0,
            )
            self.assertEqual(
                report["counts"]["by_split"]["validation"][
                    "detector_positive_source_recall"
                ],
                0.0,
            )
            self.assertEqual(
                report["hard_negative_selection"]["heldout_candidates_unfiltered"], 2
            )
            output_features = np.load(root / "output" / "features.npy")
            source_features = np.load(fixture.features)
            np.testing.assert_array_equal(output_features[0], source_features[0, 1:4])
            self.assertEqual(
                report["examples"][0]["candidate_feature_sha256"],
                hashlib.sha256(
                    np.ascontiguousarray(source_features[0, 1:4].astype(np.float16)).tobytes()
                ).hexdigest(),
            )
            self.assertEqual(
                json.loads((root / "output" / "corpus.json").read_text()), report
            )

    def test_hard_negative_top_k_is_score_ranked_and_output_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = CandidateVerifierFixture(
                root,
                [{"source_id": "negative", "split": "train", "label": 0}],
                [
                    {
                        "source_id": "negative",
                        "scores": [0.7, 0.95, 0.8, 0.1],
                        "events": [
                            {"score_frame_index": 2, "score": 0.8},
                            {"score_frame_index": 0, "score": 0.7},
                            {"score_frame_index": 1, "score": 0.95},
                        ],
                    }
                ],
            )
            first = root / "first"
            second = root / "second"
            first_report = build_candidate_verifier_dataset(
                fixture.source,
                fixture.features,
                fixture.traces,
                first,
                locked_holdout_manifest=fixture.lock,
                pre_context_frames=1,
                post_context_frames=1,
                hard_negative_top_k=2,
            )
            second_report = build_candidate_verifier_dataset(
                fixture.source,
                fixture.features,
                fixture.traces,
                second,
                locked_holdout_manifest=fixture.lock,
                pre_context_frames=1,
                post_context_frames=1,
                hard_negative_top_k=2,
            )

            self.assertEqual(
                sorted(row["detector_score"] for row in first_report["examples"]),
                [0.8, 0.95],
            )
            self.assertEqual(first_report, second_report)
            self.assertEqual((first / "corpus.json").read_bytes(), (second / "corpus.json").read_bytes())
            for name in first_report["array_sha256"]:
                self.assertEqual(sha256_file(first / name), sha256_file(second / name))

    def test_session_scoped_top_k_competes_across_sources_and_preserves_parent_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = CandidateVerifierFixture(
                root,
                [
                    {
                        "source_id": "negative-a",
                        "parent_source_id": "ancestor-a",
                        "session_id": "shared-session",
                        "split": "train",
                        "label": 0,
                    },
                    {
                        "source_id": "negative-b",
                        "parent_source_id": "ancestor-b",
                        "session_id": "shared-session",
                        "split": "train",
                        "label": 0,
                    },
                ],
                [
                    {
                        "source_id": "negative-a",
                        "scores": [0.7],
                        "events": [{"score_frame_index": 0, "score": 0.7}],
                    },
                    {
                        "source_id": "negative-b",
                        "scores": [0.9],
                        "events": [{"score_frame_index": 0, "score": 0.9}],
                    },
                ],
            )
            report = build_candidate_verifier_dataset(
                fixture.source,
                fixture.features,
                fixture.traces,
                root / "output",
                locked_holdout_manifest=fixture.lock,
                pre_context_frames=0,
                post_context_frames=0,
                hard_negative_top_k=1,
                hard_negative_group_by="session",
            )
            self.assertEqual(len(report["examples"]), 1)
            candidate = report["examples"][0]
            self.assertEqual(candidate["parent_source_id"], "negative-b")
            self.assertEqual(candidate["source_parent_source_id"], "ancestor-b")
            self.assertEqual(candidate["session_id"], "shared-session")

    def test_rejects_split_identity_or_hash_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for overlap in ("audio_sha256", "speaker_id"):
                case = root / overlap
                case.mkdir()
                shared = _hash("shared") if overlap == "audio_sha256" else "shared-speaker"
                rows = [
                    {
                        "source_id": "train-row",
                        "split": "train",
                        "label": 0,
                        overlap: shared,
                    },
                    {
                        "source_id": "validation-row",
                        "split": "validation",
                        "label": 0,
                        overlap: shared,
                    },
                ]
                fixture = CandidateVerifierFixture(
                    case,
                    rows,
                    [
                        {"source_id": "train-row", "scores": [0.6], "events": [{"frame_index": 0}]},
                        {
                            "source_id": "validation-row",
                            "scores": [0.6],
                            "events": [{"frame_index": 0}],
                        },
                    ],
                )
                with self.assertRaisesRegex(ValueError, "overlap"):
                    build_candidate_verifier_dataset(
                        fixture.source,
                        fixture.features,
                        fixture.traces,
                        case / "output",
                        locked_holdout_manifest=fixture.lock,
                        pre_context_frames=0,
                        post_context_frames=0,
                    )

    def test_rejects_locked_holdout_and_below_threshold_fake_event(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = CandidateVerifierFixture(
                root,
                [{"source_id": "train-negative", "split": "train", "label": 0}],
                [
                    {
                        "source_id": "train-negative",
                        "scores": [0.4],
                        "events": [{"score_frame_index": 0, "score": 0.4}],
                    }
                ],
            )
            with self.assertRaisesRegex(ValueError, "locked_holdout_manifest is required"):
                build_candidate_verifier_dataset(
                    fixture.source,
                    fixture.features,
                    fixture.traces,
                    root / "missing-lock-output",
                    pre_context_frames=0,
                    post_context_frames=0,
                )
            with self.assertRaisesRegex(ValueError, "below threshold"):
                build_candidate_verifier_dataset(
                    fixture.source,
                    fixture.features,
                    fixture.traces,
                    root / "fake-event-output",
                    locked_holdout_manifest=fixture.lock,
                    pre_context_frames=0,
                    post_context_frames=0,
                )

            trace_payload = json.loads(fixture.traces.read_text())
            trace_payload["examples"][0]["scores"] = [0.6]
            trace_payload["examples"][0]["events"] = [{"score_frame_index": 0, "score": 0.6}]
            fixture.traces.write_text(json.dumps(trace_payload, sort_keys=True))
            source_row = json.loads(fixture.source.read_text())["examples"][0]
            lock = root / "locked.json"
            lock.write_text(
                json.dumps(
                    {
                        "gate_scope": "locked_untouched_continuous_negative_corpus",
                        "locked_before_scoring": True,
                        "examples": [
                            {
                                "source_id": "locked-copy",
                                "audio_sha256": source_row["audio_sha256"],
                            }
                        ],
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "locked holdout"):
                build_candidate_verifier_dataset(
                    fixture.source,
                    fixture.features,
                    fixture.traces,
                    root / "locked-output",
                    locked_holdout_manifest=lock,
                    pre_context_frames=0,
                    post_context_frames=0,
                )

    def test_threshold_regions_and_padding_are_deterministic(self):
        self.assertEqual(
            threshold_region_events([0.6, 0.8, 0.7, 0.1, 0.9], 0.5),
            [
                {"score_frame_index": 1, "score": 0.8},
                {"score_frame_index": 4, "score": 0.9},
            ],
        )
        features = np.arange(6, dtype=np.float32).reshape(3, 2)
        window, geometry = candidate_window(features, 0, 2, 1)
        np.testing.assert_array_equal(
            window,
            np.asarray([[0, 0], [0, 0], [0, 1], [2, 3]], dtype=np.float32),
        )
        self.assertEqual(geometry["left_padding_frames"], 2)
        self.assertEqual(geometry["right_padding_frames"], 0)
        with self.assertRaisesRegex(ValueError, "beyond source feature"):
            candidate_window(features, 3, 0, 0)


if __name__ == "__main__":
    unittest.main()
