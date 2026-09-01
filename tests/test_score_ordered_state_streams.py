import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from tools.score_ordered_state_streams import (
    QuantizedStreamingModel,
    event_matches_occurrence,
    frontend_features,
    score_source,
    sha256_file,
)


class _FrontendResult:
    def __init__(self, features, samples_read=160):
        self.features = features
        self.samples_read = samples_read


class _FakeFrontend:
    calls = 0

    def process_samples(self, raw):
        self.calls += 1
        return _FrontendResult(np.arange(40, dtype=np.float32))


class _FakeInterpreter:
    def __init__(self, *_args, **_kwargs):
        self.tensor = None
        self.calls = 0

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
        self.calls += 1

    def get_tensor(self, _index):
        return np.full((1, 1, 23), 100 + self.calls, dtype=np.uint8)


class ScoreOrderedStateStreamsTest(unittest.TestCase):
    def _wav(self, directory, name, seconds=0.04):
        path = Path(directory) / name
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(np.zeros(int(16000 * seconds), dtype="<i2").tobytes())
        return path

    def test_frontend_is_persistent_and_uses_repository_10ms_calls(self):
        frontend = _FakeFrontend()
        with mock.patch(
            "tools.score_ordered_state_streams.MicroFrontend", return_value=frontend
        ):
            features = list(frontend_features(np.zeros(320, dtype=np.int16)))
        self.assertEqual(len(features), 2)
        self.assertEqual(frontend.calls, 2)
        self.assertEqual(features[0].shape, (40,))

    def test_quantized_model_dequantizes_23_logits_and_quantizes_input(self):
        model = QuantizedStreamingModel(
            Path("model.tflite"), interpreter_factory=_FakeInterpreter
        )
        logits = model.step(np.ones((3, 40), dtype=np.float32))
        np.testing.assert_allclose(logits, np.full(23, 0.25))
        self.assertEqual(model.interpreter.tensor.dtype, np.uint8)
        np.testing.assert_array_equal(
            model.interpreter.tensor, np.full((1, 3, 40), 130, dtype=np.uint8)
        )

    def test_score_source_resets_decoder_only_at_source_boundary_and_emits_occurrence(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            path = self._wav(temporary, "positive.wav", seconds=0.1)
            model = mock.Mock()
            model.step.side_effect = [np.zeros(23), np.zeros(23)]
            model.input = {"shape": np.array([1, 1, 40])}
            decoder = mock.Mock()
            decoder.step.side_effect = [
                None,
                SimpleNamespace(
                    start_frame=0, end_frame=1, score=2.0, rejection_score=0.0
                ),
            ]
            decoder.current_completion_score = -np.inf
            decoder_args = {"from_logits": True}
            with (
                mock.patch(
                    "tools.score_ordered_state_streams.OrderedStateDecoder",
                    return_value=decoder,
                ),
                mock.patch(
                    "tools.score_ordered_state_streams.frontend_features",
                    return_value=[np.zeros(40), np.zeros(40)],
                ),
            ):
                records = score_source(
                    {
                        "id": "p1",
                        "path": str(path),
                        "label": "positive",
                        "session_id": "s1",
                        "occurrences": [{"start_seconds": 0.0, "end_seconds": 0.06}],
                    },
                    model,
                    decoder_args,
                    0.03,
                    "model-hash",
                    "contract-hash",
                    completion_margin=1.25,
                    cooldown_frames=7,
                )
        decoder.reset.assert_called_once_with()
        model.reset.assert_called_once_with()
        self.assertEqual(decoder.step.call_count, 2)
        self.assertFalse(any(item["type"] == "score" for item in records))
        occurrence = next(
            item for item in records if item["type"] == "positive_occurrence"
        )
        self.assertEqual(occurrence["session_id"], "s1")
        self.assertEqual(occurrence["model_sha256"], "model-hash")
        self.assertTrue(occurrence["detected"])
        self.assertEqual(records[-1]["positive_occurrence_recall"], 1.0)
        self.assertEqual(records[-1]["completion_score_stats"]["finite_maximum"], 2.0)
        self.assertEqual(records[-1]["exposure_seconds"], 0.1)
        event = next(item for item in records if item["type"] == "event")
        self.assertEqual(event["start_timestamp"], 0.0)
        self.assertEqual(event["end_timestamp"], 0.06)
        self.assertEqual(event["duration_seconds"], 0.06)

    def test_occurrence_match_requires_complete_event_containment(self):
        event = {"start_timestamp": 0.0, "end_timestamp": 0.63}

        self.assertTrue(event_matches_occurrence(event, 0.0, 0.63))
        self.assertFalse(event_matches_occurrence(event, 0.62, 1.0))
        self.assertFalse(event_matches_occurrence(event, 0.01, 0.62))
        self.assertTrue(event_matches_occurrence(event, 0.01, 0.62, 0.01))

    def test_compact_output_does_not_grow_with_stream_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self._wav(temporary, "negative.wav", seconds=0.04)
            model = mock.Mock()
            model.input = {"shape": np.array([1, 1, 40])}
            model.step.return_value = np.zeros(23)
            decoder = mock.Mock()
            decoder.current_completion_score = -np.inf
            decoder.step.return_value = None
            with (
                mock.patch(
                    "tools.score_ordered_state_streams.OrderedStateDecoder",
                    return_value=decoder,
                ),
                mock.patch(
                    "tools.score_ordered_state_streams.frontend_features",
                    return_value=[np.zeros(40) for _ in range(1000)],
                ),
            ):
                records = score_source(
                    {"id": "n1", "path": str(path), "label": "negative"},
                    model,
                    {},
                    0.03,
                    "model-hash",
                    "contract-hash",
                )
        self.assertEqual([item["type"] for item in records], ["source_summary"])
        self.assertEqual(records[0]["frame_count"], 1000)
        self.assertEqual(records[0]["completion_score_stats"]["count"], 1000)

    def test_runtime_operating_point_is_passed_to_decoder(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self._wav(temporary, "negative.wav", seconds=0.04)
            model = mock.Mock()
            model.input = {"shape": np.array([1, 1, 40])}
            model.step.return_value = np.zeros(23)
            with (
                mock.patch(
                    "tools.score_ordered_state_streams.OrderedStateDecoder"
                ) as decoder_class,
                mock.patch(
                    "tools.score_ordered_state_streams.frontend_features",
                    return_value=[np.zeros(40)],
                ),
            ):
                decoder_class.return_value.current_completion_score = -np.inf
                decoder_class.return_value.step.return_value = None
                score_source(
                    {"id": "n1", "path": str(path), "label": "negative"},
                    model,
                    {"from_logits": True},
                    0.03,
                    "model-hash",
                    "contract-hash",
                    completion_margin=2.5,
                    cooldown_frames=11,
                )
        decoder_class.assert_called_once_with(
            from_logits=True, completion_margin=2.5, cooldown_frames=11
        )

    def test_hash_is_stable(self):
        with tempfile.NamedTemporaryFile() as handle:
            handle.write(b"artifact")
            handle.flush()
            self.assertEqual(
                sha256_file(Path(handle.name)), sha256_file(Path(handle.name))
            )


if __name__ == "__main__":
    unittest.main()
