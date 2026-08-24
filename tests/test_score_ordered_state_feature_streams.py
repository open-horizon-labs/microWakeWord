import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tools.score_ordered_state_feature_streams import (
    MAX_EVENT_RECORDS_PER_SOURCE_MARGIN,
    QuantizedStreamingModel,
    _aggregate,
    _score_feature_source_margins,
    load_manifest,
    run_manifest,
    score_feature_source,
    sha256_path,
)


class _FakeMmap:
    sources = {}

    def __init__(self, path):
        self.items = self.sources[str(path)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class _FakeModel:
    def __init__(self):
        self.input = {"shape": np.array([1, 3, 40])}
        self.output = {"shape": np.array([1, 1, 23])}
        self.resets = 0
        self.steps = []

    def reset(self):
        self.resets += 1

    def step(self, features):
        self.steps.append(np.array(features, copy=True))
        return np.zeros(23, dtype=np.float32)


class _FakeDecoder:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.margins = tuple(kwargs["completion_margins"])
        self.reset_count = 0
        self.current_completion_scores = np.full(
            len(self.margins), float(len(self.instances))
        )
        self.__class__.instances.append(self)

    def reset(self):
        self.reset_count += 1

    def step(self, _logits, frame_index):
        return [None] * len(self.margins)


class _FakeInterpreter:
    def __init__(self, *_args, **_kwargs):
        self.tensor = None

    def allocate_tensors(self):
        pass

    def get_input_details(self):
        return [
            {
                "index": 1,
                "shape": np.array([1, 3, 40]),
                "dtype": np.uint8,
                "quantization": (0.5, 128),
            }
        ]

    def get_output_details(self):
        return [
            {
                "index": 2,
                "shape": np.array([1, 1, 23]),
                "dtype": np.uint8,
                "quantization": (0.25, 100),
            }
        ]

    def set_tensor(self, _index, tensor):
        self.tensor = tensor

    def invoke(self):
        pass

    def get_tensor(self, _index):
        return np.full((1, 1, 23), 104, dtype=np.uint8)


class FeatureStreamScorerTest(unittest.TestCase):
    def setUp(self):
        _FakeDecoder.instances = []

    def _source(self, directory, name="source"):
        path = Path(directory) / name
        path.mkdir()
        return path

    def test_file_hash_is_standard_sha256(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.tflite"
            path.write_bytes(b"artifact")
            self.assertEqual(sha256_path(path), hashlib.sha256(b"artifact").hexdigest())

    def test_preserves_state_inside_item_and_resets_at_item_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._source(directory)
            _FakeMmap.sources[str(path)] = [
                np.arange(6 * 40, dtype=np.float32).reshape(6, 40),
                np.full((3, 40), 7, dtype=np.float32),
            ]
            model = _FakeModel()
            with (
                mock.patch(
                    "tools.score_ordered_state_feature_streams.RaggedMmap", _FakeMmap
                ),
                mock.patch(
                    "tools.score_ordered_state_feature_streams.OrderedStateMarginSweepDecoder",
                    _FakeDecoder,
                ),
            ):
                report = score_feature_source(
                    {
                        "id": "s",
                        "path": str(path),
                        "split": "validation",
                        "label": "negative",
                        "category": "speech/TV",
                        "exposure_seconds": 0.09,
                    },
                    model,
                    {"from_logits": True},
                    0.03,
                    "m",
                    "d",
                    0.0,
                    2,
                )
            self.assertEqual(model.resets, 2)
            self.assertEqual(len(_FakeDecoder.instances), 2)
            self.assertEqual(
                [item.reset_count for item in _FakeDecoder.instances], [1, 1]
            )
            self.assertEqual(len(model.steps), 3)
            np.testing.assert_array_equal(
                model.steps[0], _FakeMmap.sources[str(path)][0][:3]
            )
            self.assertEqual(report["stored_feature_frames"], 9)
            self.assertEqual(report["scored_feature_frames"], 9)
            self.assertEqual(report["category"], "speech/TV")

    def test_positive_recall_requires_complete_event_containment(self):
        class EventDecoder(_FakeDecoder):
            def step(self, _logits, frame_index):
                return [
                    {"start_frame": 0, "end_frame": 0, "score": 1.0}
                    for _ in self.margins
                ]

        with tempfile.TemporaryDirectory() as directory:
            path = self._source(directory)
            _FakeMmap.sources[str(path)] = [np.zeros((3, 40), dtype=np.float32)]
            with (
                mock.patch(
                    "tools.score_ordered_state_feature_streams.RaggedMmap", _FakeMmap
                ),
                mock.patch(
                    "tools.score_ordered_state_feature_streams.OrderedStateMarginSweepDecoder",
                    EventDecoder,
                ),
            ):
                report = score_feature_source(
                    {
                        "id": "positive",
                        "path": str(path),
                        "split": "validation",
                        "label": "positive",
                        "occurrences": [
                            {
                                "id": "contained",
                                "item_index": 0,
                                "start_seconds": 0.0,
                                "end_seconds": 0.03,
                            },
                            {
                                "id": "edge-only",
                                "item_index": 0,
                                "start_seconds": 0.02,
                                "end_seconds": 0.03,
                            },
                        ],
                    },
                    _FakeModel(),
                    {},
                    0.03,
                    "m",
                    "d",
                    0.0,
                    0,
                )
            self.assertEqual(report["positive_occurrence_count"], 2)
            self.assertEqual(report["detected_positive_occurrence_count"], 1)
            self.assertEqual(report["positive_occurrence_recall"], 0.5)

    def test_exposure_accounting_and_partial_stride_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._source(directory)
            _FakeMmap.sources[str(path)] = [np.zeros((4, 40), dtype=np.float32)]
            with (
                mock.patch(
                    "tools.score_ordered_state_feature_streams.RaggedMmap", _FakeMmap
                ),
                mock.patch(
                    "tools.score_ordered_state_feature_streams.OrderedStateMarginSweepDecoder",
                    _FakeDecoder,
                ),
            ):
                report = score_feature_source(
                    {
                        "id": "s",
                        "path": str(path),
                        "split": "test",
                        "label": "negative",
                        "exposure_seconds": 0.04,
                    },
                    _FakeModel(),
                    {},
                    0.03,
                    "m",
                    "d",
                    0.0,
                    0,
                )
            self.assertEqual(report["trailing_feature_frames"], 1)
            self.assertEqual(report["scored_exposure_seconds"], 0.03)
            with self.assertRaisesRegex(ValueError, "declared exposure"):
                score_feature_source(
                    {
                        "id": "s",
                        "path": str(path),
                        "split": "test",
                        "label": "negative",
                        "exposure_seconds": 0.03,
                    },
                    _FakeModel(),
                    {},
                    0.03,
                    "m",
                    "d",
                    0.0,
                    0,
                )

    def test_expected_source_hash_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._source(directory)
            _FakeMmap.sources[str(path)] = [np.zeros((3, 40), dtype=np.float32)]
            with (
                mock.patch(
                    "tools.score_ordered_state_feature_streams.RaggedMmap", _FakeMmap
                ),
                mock.patch(
                    "tools.score_ordered_state_feature_streams.OrderedStateMarginSweepDecoder",
                    _FakeDecoder,
                ),
                self.assertRaisesRegex(ValueError, "source hash"),
            ):
                score_feature_source(
                    {
                        "id": "s",
                        "path": str(path),
                        "split": "validation",
                        "label": "negative",
                        "exposure_seconds": 0.03,
                        "expected_path_sha256": "not-the-real-hash",
                    },
                    _FakeModel(),
                    {},
                    0.03,
                    "m",
                    "d",
                    0.0,
                    0,
                )

    def test_model_steps_are_independent_of_margin_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._source(directory)
            _FakeMmap.sources[str(path)] = [np.zeros((6, 40), dtype=np.float32)]
            model = _FakeModel()
            with (
                mock.patch(
                    "tools.score_ordered_state_feature_streams.RaggedMmap", _FakeMmap
                ),
                mock.patch(
                    "tools.score_ordered_state_feature_streams.OrderedStateMarginSweepDecoder",
                    _FakeDecoder,
                ),
            ):
                reports = _score_feature_source_margins(
                    {
                        "id": "s",
                        "path": str(path),
                        "split": "validation",
                        "label": "negative",
                        "exposure_seconds": 0.06,
                    },
                    model,
                    {},
                    0.03,
                    "m",
                    "d",
                    [0.0, 1.0, 2.0],
                    0,
                )
            self.assertEqual(len(reports), 3)
            self.assertEqual(len(model.steps), 2)
            self.assertEqual(model.resets, 1)
            self.assertEqual(
                [report["completion_score_stats"]["count"] for report in reports],
                [2, 2, 2],
            )

    def test_event_records_are_bounded_without_losing_event_count(self):
        class EventDecoder(_FakeDecoder):
            def step(self, _logits, frame_index):
                return [
                    {
                        "start_frame": frame_index,
                        "end_frame": frame_index,
                        "score": 1.0,
                    }
                    for _ in self.margins
                ]

        with tempfile.TemporaryDirectory() as directory:
            path = self._source(directory)
            frames = (MAX_EVENT_RECORDS_PER_SOURCE_MARGIN + 2) * 3
            _FakeMmap.sources[str(path)] = [np.zeros((frames, 40), dtype=np.float32)]
            with (
                mock.patch(
                    "tools.score_ordered_state_feature_streams.RaggedMmap", _FakeMmap
                ),
                mock.patch(
                    "tools.score_ordered_state_feature_streams.OrderedStateMarginSweepDecoder",
                    EventDecoder,
                ),
            ):
                report = score_feature_source(
                    {
                        "id": "s",
                        "path": str(path),
                        "split": "validation",
                        "label": "negative",
                        "exposure_seconds": frames * 0.01,
                    },
                    _FakeModel(),
                    {},
                    0.03,
                    "m",
                    "d",
                    0.0,
                    0,
                )
            self.assertEqual(
                report["event_count"], MAX_EVENT_RECORDS_PER_SOURCE_MARGIN + 2
            )
            self.assertEqual(len(report["events"]), MAX_EVENT_RECORDS_PER_SOURCE_MARGIN)
            self.assertEqual(report["event_records_truncated"], 2)
            self.assertEqual(report["events"][0]["end_timestamp"], 0.03)
            self.assertEqual(report["events"][0]["duration_seconds"], 0.03)

    def test_manifest_rejects_undeclared_split_and_exposure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "sources": [
                            {"path": "x", "split": "validation", "exposure_seconds": 1}
                        ]
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "explicit.*label"):
                load_manifest(path)

            path.write_text(
                json.dumps(
                    {
                        "feature_step_seconds": 0.02,
                        "sources": [
                            {
                                "path": "x",
                                "split": "validation",
                                "label": "negative",
                                "exposure_seconds": 1,
                            }
                        ],
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "non-10 ms"):
                load_manifest(path)

            path.write_text(
                json.dumps(
                    {
                        "sources": [
                            {
                                "path": "x",
                                "split": "validation",
                                "label": "positive",
                            }
                        ]
                    }
                )
            )
            with self.assertRaisesRegex(
                ValueError, "positive source needs occurrences"
            ):
                load_manifest(path)

    def test_faph_uses_only_negative_scored_exposure(self):
        negative = {
            "label": "negative",
            "negative_scored_exposure_seconds": 3.0,
            "negative_event_count": 1,
            "event_count": 1,
            "positive_occurrence_count": 0,
            "detected_positive_occurrence_count": 0,
            "completion_score_stats": {
                "count": 1,
                "finite_count": 1,
                "finite_minimum": 1.0,
                "finite_maximum": 1.0,
                "finite_mean": 1.0,
                "negative_infinity_count": 0,
                "positive_infinity_count": 0,
            },
        }
        positive = {
            "label": "positive",
            "negative_scored_exposure_seconds": 0.0,
            "negative_event_count": 0,
            "event_count": 7,
            "positive_occurrence_count": 2,
            "detected_positive_occurrence_count": 2,
            "completion_score_stats": negative["completion_score_stats"],
        }
        aggregate = _aggregate([negative, positive], 0.0, 4)
        self.assertEqual(aggregate["negative_scored_exposure_seconds"], 3.0)
        self.assertEqual(aggregate["negative_event_count"], 1)
        self.assertAlmostEqual(aggregate["faph"], 1200.0)
        self.assertGreater(aggregate["poisson_upper_bound_95_per_hour"], 1200.0)
        self.assertEqual(aggregate["positive_occurrence_recall"], 1.0)

    def test_sweep_requires_negative_exposure_and_positive_occurrence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(directory)
            _FakeMmap.sources[str(source.resolve())] = [
                np.zeros((3, 40), dtype=np.float32)
            ]
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "positive_occurrence_geometry": "exact_phrase_span",
                        "sources": [
                            {
                                "path": "source",
                                "split": "validation",
                                "label": "positive",
                                "occurrences": [
                                    {
                                        "item_index": 0,
                                        "start_seconds": 0.0,
                                        "end_seconds": 0.03,
                                    }
                                ],
                            }
                        ],
                    }
                )
            )
            model_path = root / "model.tflite"
            contract_path = root / "contract.json"
            model_path.write_bytes(b"model")
            contract_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "state_count": 23,
                        "frame_step_seconds": 0.03,
                        "decoder_args": {},
                    }
                )
            )
            with mock.patch(
                "tools.score_ordered_state_feature_streams.QuantizedStreamingModel",
                return_value=_FakeModel(),
            ):
                with self.assertRaisesRegex(ValueError, "negative exposure"):
                    run_manifest(
                        manifest,
                        model_path,
                        contract_path,
                        root / "out.json",
                        [0.0, 1.0],
                        0,
                        0.1,
                        0.9,
                    )

    def test_selection_maximizes_recall_then_faph_then_conservative_margin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            negative_path = self._source(directory, "negative")
            positive_path = self._source(directory, "positive")
            _FakeMmap.sources[str(negative_path.resolve())] = [
                np.zeros((3, 40), dtype=np.float32)
            ]
            _FakeMmap.sources[str(positive_path.resolve())] = [
                np.zeros((3, 40), dtype=np.float32)
            ]
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "positive_occurrence_geometry": "exact_phrase_span",
                        "sources": [
                            {
                                "path": "negative",
                                "split": "validation",
                                "label": "negative",
                                "exposure_seconds": 0.03,
                            },
                            {
                                "path": "positive",
                                "split": "validation",
                                "label": "positive",
                                "occurrences": [
                                    {
                                        "item_index": 0,
                                        "start_seconds": 0.0,
                                        "end_seconds": 0.03,
                                    }
                                ],
                            },
                        ],
                    }
                )
            )
            model_path = root / "model.tflite"
            contract_path = root / "contract.json"
            model_path.write_bytes(b"model")
            contract_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "state_count": 23,
                        "frame_step_seconds": 0.03,
                        "decoder_args": {},
                    }
                )
            )

            def fake_score(source, _model, _args, _step, *_rest):
                margins = _rest[-2]
                stats = {
                    "count": 0,
                    "finite_count": 0,
                    "finite_minimum": None,
                    "finite_maximum": None,
                    "finite_mean": None,
                    "negative_infinity_count": 0,
                    "positive_infinity_count": 0,
                }
                reports = []
                for margin in margins:
                    if source["label"] == "negative":
                        faph = {
                            0.0: 0.05,
                            1.0: 0.08,
                            2.0: 0.02,
                            3.0: 0.02,
                            4.0: 0.001,
                        }[margin]
                        reports.append(
                            {
                                "label": "negative",
                                "negative_scored_exposure_seconds": 360000.0,
                                "negative_event_count": int(faph * 100),
                                "event_count": int(faph * 100),
                                "positive_occurrence_count": 0,
                                "detected_positive_occurrence_count": 0,
                                "completion_score_stats": stats,
                            }
                        )
                    else:
                        recall = {
                            0.0: 0.9,
                            1.0: 1.0,
                            2.0: 1.0,
                            3.0: 1.0,
                            4.0: 0.8,
                        }[margin]
                        reports.append(
                            {
                                "label": "positive",
                                "negative_scored_exposure_seconds": 0.0,
                                "negative_event_count": 0,
                                "event_count": 0,
                                "positive_occurrence_count": 10,
                                "detected_positive_occurrence_count": int(recall * 10),
                                "completion_score_stats": stats,
                            }
                        )
                return reports

            with (
                mock.patch(
                    "tools.score_ordered_state_feature_streams.QuantizedStreamingModel",
                    return_value=_FakeModel(),
                ),
                mock.patch(
                    "tools.score_ordered_state_feature_streams._score_feature_source_margins",
                    side_effect=fake_score,
                ),
            ):
                result = run_manifest(
                    manifest,
                    model_path,
                    contract_path,
                    root / "out.json",
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                    0,
                    0.1,
                    0.9,
                )
            self.assertEqual(
                result["selected_operating_point"]["completion_margin"], 3.0
            )
            self.assertIn(
                "unclassified",
                result["completion_margin_sweep"][0]["negative_categories"],
            )

    def test_test_split_single_point_is_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(directory)
            _FakeMmap.sources[str(source.resolve())] = [
                np.zeros((3, 40), dtype=np.float32)
            ]
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "sources": [
                            {
                                "path": "source",
                                "split": "test",
                                "label": "negative",
                                "exposure_seconds": 0.03,
                            }
                        ]
                    }
                )
            )
            model_path = root / "model.tflite"
            contract_path = root / "contract.json"
            model_path.write_bytes(b"model")
            contract_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "state_count": 23,
                        "frame_step_seconds": 0.03,
                        "decoder_args": {},
                    }
                )
            )
            with (
                mock.patch(
                    "tools.score_ordered_state_feature_streams.QuantizedStreamingModel",
                    return_value=_FakeModel(),
                ),
                mock.patch(
                    "tools.score_ordered_state_feature_streams.RaggedMmap", _FakeMmap
                ),
                mock.patch(
                    "tools.score_ordered_state_feature_streams.OrderedStateMarginSweepDecoder",
                    _FakeDecoder,
                ),
            ):
                result = run_manifest(
                    manifest,
                    model_path,
                    contract_path,
                    root / "out.json",
                    [0.0],
                    0,
                    0.1,
                    0.9,
                )
            self.assertEqual(result["declared_split_counts"]["test"], 1)
            self.assertEqual(len(result["reports"]), 1)

    def test_real_quantized_adapter_contract_with_fake_interpreter(self):
        model = QuantizedStreamingModel(
            Path("model.tflite"), interpreter_factory=_FakeInterpreter
        )
        logits = model.step(np.ones((3, 40), dtype=np.float32))
        np.testing.assert_allclose(logits, np.full(23, 1.0))
        np.testing.assert_array_equal(
            model.interpreter.tensor, np.full((1, 3, 40), 130, dtype=np.uint8)
        )

    def test_quantization_adapter_and_validation_only_sweep(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_path = self._source(directory)
            _FakeMmap.sources[str(feature_path.resolve())] = [
                np.zeros((3, 40), dtype=np.uint16)
            ]
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "positive_occurrence_geometry": "exact_phrase_span",
                        "sources": [
                            {
                                "id": "v",
                                "path": "source",
                                "split": "validation",
                                "label": "negative",
                                "exposure_seconds": 0.03,
                            },
                            {
                                "id": "p",
                                "path": "source",
                                "split": "validation",
                                "label": "positive",
                                "occurrences": [
                                    {
                                        "id": "wake",
                                        "item_index": 0,
                                        "start_seconds": 0.0,
                                        "end_seconds": 0.03,
                                    }
                                ],
                            },
                        ],
                    }
                )
            )
            model_path = root / "model.tflite"
            contract_path = root / "contract.json"
            model_path.write_bytes(b"model")
            contract_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "state_count": 23,
                        "frame_step_seconds": 0.03,
                        "decoder_args": {"from_logits": True},
                    }
                )
            )
            fake_model = _FakeModel()
            with (
                mock.patch(
                    "tools.score_ordered_state_feature_streams.QuantizedStreamingModel",
                    return_value=fake_model,
                ),
                mock.patch(
                    "tools.score_ordered_state_feature_streams.RaggedMmap", _FakeMmap
                ),
                mock.patch(
                    "tools.score_ordered_state_feature_streams.OrderedStateMarginSweepDecoder",
                    _FakeDecoder,
                ),
            ):
                result = run_manifest(
                    manifest_path,
                    model_path,
                    contract_path,
                    root / "out.json",
                    [0.0, 1.0],
                    3,
                    0.1,
                    0.9,
                )
            self.assertEqual(len(result["completion_margin_sweep"]), 2)
            self.assertEqual(result["declared_split_counts"]["test"], 0)
            self.assertEqual(len(fake_model.steps), 2)
            np.testing.assert_array_equal(
                fake_model.steps[0], np.zeros((3, 40), dtype=np.float32)
            )


if __name__ == "__main__":
    unittest.main()
