import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.build_kizz_candidate_verifier_dataset import (
    build_candidate_verifier_dataset,
)
from tools.trace_kizz_ordered_state_detector import (
    EXPECTED_DECODER_ARGUMENTS,
    MODEL_FILENAME,
    VERIFIER_MODEL_FILENAME,
    feature_sha256,
    sha256_file,
    sha256_json,
    trace_detector,
)
from microwakeword.ordered_state import OrderedStateTopology
from microwakeword.wake_phrase import KIZZ_CONTROL


class FakeInterpreter:
    def __init__(self, *, input_observer=None):
        self.calls = 0
        self.input_observer = input_observer
        self.current = None

    def allocate_tensors(self):
        return None

    def get_input_details(self):
        return [
            {
                "index": 3,
                "shape": np.asarray([1, 3, 40], dtype=np.int32),
                "dtype": np.int8,
                "quantization": (0.5, -1),
            }
        ]

    def get_output_details(self):
        return [
            {
                "index": 7,
                "shape": np.asarray([1, 1, 12], dtype=np.int32),
                "dtype": np.uint8,
                "quantization": (0.25, 128),
            }
        ]

    def set_tensor(self, index, value):
        if index != 3:
            raise AssertionError("wrong input tensor")
        self.current = np.asarray(value).copy()
        if self.input_observer is not None:
            self.input_observer(self.current, self.calls)

    def invoke(self):
        if self.current is None:
            raise AssertionError("invoke before input")

    def get_tensor(self, index):
        if index != 7:
            raise AssertionError("wrong output tensor")
        raw = np.full((1, 1, 12), 128, dtype=np.uint8)
        is_positive = float(np.mean(self.current)) > -1.0
        if is_positive:
            raw[0, 0, 0:2] = 120
            # The converter discards 20 warmup calls.  The aligned timeline
            # therefore begins with the first ordered phone state.
            state = ((self.calls - 20) % 10) + 2
            raw[0, 0, state] = 200
        else:
            raw[0, 0, 0] = 200
        self.calls += 1
        return raw


class TraceFixture:
    def __init__(self, root: Path, rows=None, values=None, *, observer=None):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.output = root / "trace"
        self.threshold = root / "deployed-int8-threshold.json"
        self.model = root / MODEL_FILENAME
        self.model.write_bytes(b"deployed-stateful-int8-model")
        if rows is None:
            rows = [
                {"source_id": "validation-positive", "split": "validation", "label": 1},
                {"source_id": "validation-negative", "split": "validation", "label": 0},
                {"source_id": "test-positive", "split": "test", "label": 1},
                {"source_id": "test-negative", "split": "test", "label": 0},
            ]
        if values is None:
            values = np.stack(
                [
                    np.full((260, 40), 2.0, dtype=np.float32),
                    np.full((260, 40), -2.0, dtype=np.float32),
                    np.full((260, 40), 3.0, dtype=np.float32),
                    np.full((260, 40), -3.0, dtype=np.float32),
                ]
            )
        self.features = root / "source-features.npy"
        np.save(self.features, values)
        bound_rows = []
        for index, row in enumerate(rows):
            bound_rows.append(
                {
                    "feature_index": index,
                    "feature_sha256": feature_sha256(values[index]),
                    "duration_seconds": 2.6,
                    "session_id": f"session-{index}",
                    "ancestry_id": f"ancestry-{index}",
                    **row,
                }
            )
        self.source = root / "source-manifest.json"
        self.source.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "array_sha256": {self.features.name: sha256_file(self.features)},
                    "examples": bound_rows,
                },
                sort_keys=True,
            )
        )
        topology = OrderedStateTopology(tuple(KIZZ_CONTROL.phones), 1)
        topology_payload = {
            "phrase_id": KIZZ_CONTROL.phrase_id,
            "text": KIZZ_CONTROL.text,
            "phones": list(topology.phones),
            "states_per_phone": 1,
            "state_count": topology.state_count,
            "state_names": list(topology.state_names),
            "background_index": 0,
            "silence_index": 1,
            "first_ordered_state_index": 2,
        }
        decoder_contract = {
            "topology": topology_payload,
            "algorithm": "ordered_state_sequence_score_numpy",
            "arguments": EXPECTED_DECODER_ARGUMENTS,
        }
        reference = Path(__file__).resolve().parents[1] / "microwakeword/ordered_state.py"
        equivalence = {
            "algorithm": "generic_ordered_state_sequence_score_numpy_v1",
            "from_logits": True,
            "offline_shape": [66, 12],
            "streaming_calls_per_example": 86,
            "streaming_warmup_outputs_discarded": 20,
        }
        equivalence["evidence_sha256"] = sha256_json(equivalence)
        self.artifact = root / "firmware-artifact.json"
        metadata = {
            "schema_version": 2,
            "kind": "kizz_control_ordered_state_detector_streaming_int8",
            "student_role": "permissive_detector_candidate_generator",
            "deployment_qualification": False,
            "artifact": {
                "filename": self.model.name,
                "bytes": self.model.stat().st_size,
                "sha256": sha256_file(self.model),
            },
            "source": {},
            "topology": topology_payload,
            "decoder": {
                "algorithm": "ordered_state_sequence_score_numpy",
                "contract_version": 1,
                "arguments": EXPECTED_DECODER_ARGUMENTS,
                "score_semantics": "maximum_complete_left_to_right_log_odds_path",
                "contract_sha256": sha256_json(decoder_contract),
                "reference_module": str(reference),
                "reference_module_sha256": sha256_file(reference),
            },
            "timeline": {
                "frontend_feature_step_seconds": 0.01,
                "frontend_window_seconds": 0.03,
                "offline_input_frames": 260,
                "offline_output_frames": 66,
                "stream_input_frames_per_call": 3,
                "stream_hop_seconds": 0.03,
                "stream_phase_offset_frames": 0,
                "stream_phase_priming": "zero_prefix_then_observed_prefix",
                "streaming_calls_per_260_frame_example": 86,
                "streaming_warmup_outputs_discarded": 20,
                "offline_output_times_seconds": [0.03 * index for index in range(66)],
                "causal_tail_alignment": "derived_from_calls_minus_offline_output_frames",
            },
            "tensor_contracts": {
                "input": {
                    "shape": [1, 3, 40],
                    "dtype": "int8",
                    "quantization": [0.5, -1],
                },
                "output": {
                    "shape": [1, 1, 12],
                    "dtype": "uint8",
                    "quantization": [0.25, 128],
                },
                "output_semantics": "unnormalized_ordered_state_logits",
            },
            "static_memory_contract": {
                "batch_size": 1,
                "fixed_input_shape": True,
                "fixed_output_shape": True,
                "dynamic_tensor_shapes_forbidden": True,
                "external_state_tensor_count": 0,
                "persistent_state": "internal_tflite_variables",
                "tensor_audit": {
                    "input_count": 1,
                    "output_count": 1,
                    "dynamic_shape_tensor_count": 0,
                },
            },
            "equivalence": equivalence,
        }
        self.artifact.write_text(json.dumps(metadata, sort_keys=True))
        self.instances = []
        self.observer = observer

    def factory(self, model_path):
        self.assert_model_path = model_path
        instance = FakeInterpreter(input_observer=self.observer)
        self.instances.append(instance)
        return instance

    def run(self, **overrides):
        arguments = {
            "threshold_output": self.threshold,
            "minimum_recall": 0.95,
            "maximum_false_candidate_fraction": 0.20,
            "interpreter_factory": self.factory,
        }
        arguments.update(overrides)
        return trace_detector(
            self.artifact,
            self.model,
            self.source,
            self.features,
            self.output,
            **arguments,
        )


class TraceOrderedStateDetectorTests(unittest.TestCase):
    def test_evaluation_only_does_not_invoke_train_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = [
                {"source_id": "train-positive", "split": "train", "label": 1},
                {"source_id": "validation-positive", "split": "validation", "label": 1},
                {"source_id": "validation-negative", "split": "validation", "label": 0},
                {"source_id": "test-positive", "split": "test", "label": 1},
                {"source_id": "test-negative", "split": "test", "label": 0},
            ]
            values = np.stack(
                [
                    np.full((260, 40), 4.0, dtype=np.float32),
                    np.full((260, 40), 2.0, dtype=np.float32),
                    np.full((260, 40), -2.0, dtype=np.float32),
                    np.full((260, 40), 3.0, dtype=np.float32),
                    np.full((260, 40), -3.0, dtype=np.float32),
                ]
            )
            fixture = TraceFixture(Path(directory), rows=rows, values=values)

            report = fixture.run(evaluation_only=True)

            self.assertEqual(report["counts"]["examples"], 4)
            self.assertEqual(len(fixture.instances), 4)
            self.assertNotIn("train-positive", {row["source_id"] for row in report["examples"]})
            self.assertTrue(report["detector"]["source_context"]["evaluation_only"])

    def test_evaluation_only_accepts_a_test_only_physical_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = [
                {"source_id": "test-positive", "split": "test", "label": 1},
                {"source_id": "test-negative", "split": "test", "label": 0},
            ]
            values = np.stack(
                [
                    np.full((260, 40), 3.0, dtype=np.float32),
                    np.full((260, 40), -3.0, dtype=np.float32),
                ]
            )
            fixture = TraceFixture(Path(directory), rows=rows, values=values)

            report = fixture.run(
                evaluation_only=True,
                threshold_output=None,
                threshold=0.0,
            )

            self.assertEqual(report["counts"]["examples"], 2)
            self.assertEqual(
                set(report["evaluation"]["by_split"]),
                {"train", "validation", "test"},
            )

    def test_opt_in_continuous_context_extends_the_same_streaming_model(self):
        with tempfile.TemporaryDirectory() as directory:
            values = np.stack(
                [
                    np.full((560, 40), 2.0, dtype=np.float32),
                    np.full((560, 40), -2.0, dtype=np.float32),
                    np.full((560, 40), 3.0, dtype=np.float32),
                    np.full((560, 40), -3.0, dtype=np.float32),
                ]
            )
            fixture = TraceFixture(Path(directory), values=values)

            with self.assertRaisesRegex(ValueError, r"\[N,260,40\]"):
                fixture.run()
            report = fixture.run(allow_continuous_context=True)

            self.assertEqual(
                report["detector"]["source_context"],
                {
                    "mode": "continuous_fixed_length",
                    "input_frames": 560,
                    "evaluation_only": False,
                },
            )
            self.assertTrue(all(instance.calls == 186 for instance in fixture.instances[-4:]))
            self.assertTrue(all(len(row["scores"]) == 157 for row in report["examples"]))

    def test_candidate_verifier_validates_candidate_window_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = TraceFixture(Path(directory))
            verifier_model = fixture.root / VERIFIER_MODEL_FILENAME
            verifier_model.write_bytes(fixture.model.read_bytes())
            fixture.model = verifier_model

            metadata = json.loads(fixture.artifact.read_text())
            metadata.update(
                {
                    "kind": "kizz_control_ordered_state_candidate_verifier_streaming_int8",
                    "student_role": "detector_conditioned_ordered_state_candidate_verifier",
                    "candidate_conditioned": True,
                }
            )
            metadata["artifact"] = {
                "filename": verifier_model.name,
                "bytes": verifier_model.stat().st_size,
                "sha256": sha256_file(verifier_model),
            }
            fixture.artifact.write_text(json.dumps(metadata, sort_keys=True))

            source = json.loads(fixture.source.read_text())
            for index, row in enumerate(source["examples"]):
                row["candidate_feature_sha256"] = row["feature_sha256"]
                row["feature_sha256"] = hashlib.sha256(
                    f"parent-feature-{index}".encode()
                ).hexdigest()
            fixture.source.write_text(json.dumps(source, sort_keys=True))

            report = fixture.run()
            self.assertEqual(report["counts"]["examples"], 4)

            source = json.loads(fixture.source.read_text())
            source["examples"][0]["candidate_feature_sha256"] = "0" * 64
            fixture.source.write_text(json.dumps(source, sort_keys=True))
            with self.assertRaisesRegex(ValueError, "row feature SHA drift"):
                fixture.run()

    def test_candidate_verifier_continuous_context_uses_source_feature_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            values = np.stack(
                [
                    np.full((560, 40), 2.0, dtype=np.float32),
                    np.full((560, 40), -2.0, dtype=np.float32),
                    np.full((560, 40), 3.0, dtype=np.float32),
                    np.full((560, 40), -3.0, dtype=np.float32),
                ]
            )
            fixture = TraceFixture(Path(directory), values=values)
            verifier_model = fixture.root / VERIFIER_MODEL_FILENAME
            verifier_model.write_bytes(fixture.model.read_bytes())
            fixture.model = verifier_model
            metadata = json.loads(fixture.artifact.read_text())
            metadata.update(
                {
                    "kind": "kizz_control_ordered_state_candidate_verifier_streaming_int8",
                    "student_role": "detector_conditioned_ordered_state_candidate_verifier",
                    "candidate_conditioned": True,
                }
            )
            metadata["artifact"] = {
                "filename": verifier_model.name,
                "bytes": verifier_model.stat().st_size,
                "sha256": sha256_file(verifier_model),
            }
            fixture.artifact.write_text(json.dumps(metadata, sort_keys=True))

            report = fixture.run(allow_continuous_context=True)

            self.assertEqual(report["counts"]["examples"], 4)

    def test_representative_positive_negative_fit_then_builder_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            threshold_seen_before_test = []

            def observer(value, call):
                if call == 0 and abs(float(np.mean(value))) > 5:
                    threshold_seen_before_test.append((root / "deployed-int8-threshold.json").is_file())

            fixture = TraceFixture(root, observer=observer)
            report = fixture.run()

            self.assertTrue(all(threshold_seen_before_test))
            frozen = json.loads(fixture.threshold.read_text())
            self.assertEqual(frozen["selection"]["fit_split"], "validation")
            self.assertFalse(frozen["selection"]["test_used_for_selection"])
            self.assertTrue(frozen["selection"]["qualified"])
            self.assertIsNone(frozen["test_metrics"])
            self.assertFalse(report["deployment_qualification"])
            self.assertEqual(report["evaluation"]["by_split"]["test"]["opportunity_recall"], 1.0)
            self.assertEqual(report["evaluation"]["by_split"]["test"]["false_candidates"], 0)
            self.assertEqual(len(fixture.instances), 4)
            self.assertTrue(all(instance.calls == 86 for instance in fixture.instances))
            for trace in report["examples"]:
                self.assertEqual(len(trace["scores"]), 57)
                self.assertEqual(len(trace["feature_frame_indexes"]), 57)
                self.assertTrue(all(np.isfinite(trace["scores"])))

            lock = root / "locked.json"
            lock.write_text(
                json.dumps(
                    {
                        "gate_scope": "locked_untouched_continuous_negative_corpus",
                        "locked_before_scoring": True,
                        "examples": [],
                    }
                )
            )
            built = build_candidate_verifier_dataset(
                fixture.source,
                fixture.features,
                fixture.output / "detector-traces.json",
                root / "candidates",
                locked_holdout_manifest=lock,
                pre_context_frames=1,
                post_context_frames=1,
            )
            self.assertGreaterEqual(built["counts"]["selected_positives"], 2)
            self.assertEqual(
                built["detector"]["config"]["path"], str(fixture.artifact.resolve())
            )
            self.assertEqual(
                built["detector"]["threshold"]["path"], str(fixture.threshold.resolve())
            )

    def test_corrupt_artifact_and_source_feature_hashes_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = TraceFixture(root)
            fixture.model.write_bytes(fixture.model.read_bytes() + b"corrupt")
            with self.assertRaisesRegex(ValueError, "TFLite artifact hash drift"):
                fixture.run()

            fixture = TraceFixture(root / "source-corrupt")
            payload = json.loads(fixture.source.read_text())
            payload["examples"][0]["feature_sha256"] = "0" * 64
            fixture.source.write_text(json.dumps(payload, sort_keys=True))
            with self.assertRaisesRegex(ValueError, "row feature SHA drift"):
                fixture.run()

    def test_wrong_timeline_tensor_and_decoder_contracts_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, mutate, message in (
                (
                    "timeline",
                    lambda value: value["timeline"].__setitem__("stream_input_frames_per_call", 4),
                    "timeline contract drift",
                ),
                (
                    "tensor",
                    lambda value: value["tensor_contracts"]["output"].__setitem__("dtype", "int8"),
                    "tensor topology contract drift",
                ),
                (
                    "decoder",
                    lambda value: value["decoder"]["arguments"].__setitem__("self_loop_probability", 0.7),
                    "decoder semantics drift",
                ),
            ):
                case = root / name
                fixture = TraceFixture(case)
                metadata = json.loads(fixture.artifact.read_text())
                mutate(metadata)
                fixture.artifact.write_text(json.dumps(metadata, sort_keys=True))
                with self.assertRaisesRegex(ValueError, message):
                    fixture.run()

    def test_fresh_interpreter_resets_state_between_identical_examples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = np.stack(
                [
                    np.full((260, 40), 2.0, dtype=np.float32),
                    np.full((260, 40), 2.0, dtype=np.float32),
                ]
            )
            fixture = TraceFixture(
                root,
                rows=[
                    {"source_id": "a", "split": "validation", "label": 1},
                    {"source_id": "b", "split": "validation", "label": 0},
                ],
                values=values,
            )
            report = fixture.run(maximum_false_candidate_fraction=1.0)
            self.assertEqual(len(fixture.instances), 2)
            self.assertEqual(report["examples"][0]["scores"], report["examples"][1]["scores"])
            self.assertEqual(
                report["detector"]["state_reset"],
                "fresh_interpreter_and_allocate_tensors_per_example",
            )

    def test_int8_input_quantization_clips_and_uint8_logits_dequantize(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observed = []

            def observer(value, call):
                if call == 0:
                    observed.append(value.copy())

            values = np.stack(
                [
                    np.full((260, 40), 2.0, dtype=np.float32),
                    np.full((260, 40), -2.0, dtype=np.float32),
                ]
            )
            values[0, 0, 0] = 1000.0
            values[0, 0, 1] = -1000.0
            fixture = TraceFixture(
                root,
                rows=[
                    {"source_id": "positive", "split": "validation", "label": 1},
                    {"source_id": "negative", "split": "validation", "label": 0},
                ],
                values=values,
                observer=observer,
            )
            fixture.run()
            positive = observed[1] if float(np.mean(observed[0])) < -1 else observed[0]
            self.assertEqual(int(positive[0, 0, 0]), 127)
            self.assertEqual(int(positive[0, 0, 1]), -128)
            logits = np.load(fixture.output / "state-logits.npy")
            self.assertEqual(logits.dtype, np.float32)
            self.assertTrue(np.allclose(logits / 0.25, np.rint(logits / 0.25)))

    def test_validation_report_rejects_test_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = TraceFixture(root)
            report_path = root / "bad-threshold.json"
            report_path.write_text(
                json.dumps(
                    {
                        "deployment_qualification": False,
                        "selection": {
                            "fit_split": "test",
                            "test_used_for_selection": True,
                            "threshold": 0.0,
                        },
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "validation only"):
                fixture.run(
                    threshold_output=None,
                    threshold_report=report_path,
                )

    def test_output_is_deterministic_and_binds_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = TraceFixture(root)
            first = fixture.run()
            first_manifest = (fixture.output / "detector-traces.json").read_bytes()
            first_threshold = fixture.threshold.read_bytes()
            first_hashes = {
                name: sha256_file(fixture.output / name)
                for name in ("state-logits.npy", "scores.npy", "feature-frame-indexes.npy")
            }
            fixture.instances.clear()
            second = fixture.run()
            self.assertEqual(first, second)
            self.assertEqual(first_manifest, (fixture.output / "detector-traces.json").read_bytes())
            self.assertEqual(first_threshold, fixture.threshold.read_bytes())
            self.assertEqual(
                first_hashes,
                {name: sha256_file(fixture.output / name) for name in first_hashes},
            )
            for name, binding in second["arrays"].items():
                self.assertEqual(binding["sha256"], sha256_file(fixture.output / name))


if __name__ == "__main__":
    unittest.main()
