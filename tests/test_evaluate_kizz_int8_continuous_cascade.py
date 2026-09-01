import hashlib
import json
import math
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from microwakeword.ordered_state import OrderedStateTopology
from microwakeword.wake_phrase import KIZZ_CONTROL
from tools import evaluate_kizz_int8_continuous_cascade as evaluator
from tools.trace_kizz_ordered_state_detector import (
    EXPECTED_DECODER_ARGUMENTS,
    sha256_json as detector_sha256_json,
)


def write_json(path: Path, value):
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def binding(path: Path):
    return {
        "path": str(path.resolve()),
        "sha256": evaluator.sha256_file(path),
        "bytes": path.stat().st_size,
    }


def write_wav(path: Path, samples: int = 1600):
    values = (np.arange(samples, dtype=np.int32) % 101 - 50).astype("<i2")
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16_000)
        target.writeframes(values.tobytes())


class FakeRuntime:
    def __init__(self, role, contract, *, verifier_raw=20):
        self.role = role
        self.contract = contract
        self.reset_count = 0
        self.invoke_count = 0
        self.tensor_bytes = 1
        self.tensor_count = 1
        self.verifier_raw = verifier_raw

    def reset(self):
        self.reset_count += 1

    def invoke(self, values):
        self.invoke_count += 1
        expected = tuple(self.contract.input_contract["shape"])[1:]
        self.asserted_shape = np.asarray(values).shape
        if self.role == "detector":
            if self.asserted_shape != expected:
                raise AssertionError((self.asserted_shape, expected))
            return np.full((1, 1, 12), 128, dtype=np.uint8)
        if self.role == "ordered_verifier":
            if self.asserted_shape != expected:
                raise AssertionError((self.asserted_shape, expected))
            output = np.full((1, 1, 12), 128, dtype=np.uint8)
            decoded_call = self.invoke_count - 22
            if 0 <= decoded_call < 10:
                output[0, 0, 2 + decoded_call] = 255
            return output
        if self.asserted_shape not in (expected, tuple(self.contract.input_contract["shape"])):
            raise AssertionError((self.asserted_shape, expected))
        return np.asarray([[self.verifier_raw]], dtype=np.int8)


class RuntimeFactory:
    def __init__(self, *, verifier_raw=20):
        self.instances = {}
        self.verifier_raw = verifier_raw

    def __call__(self, role, artifact, contract):
        self.artifact = artifact
        instance = FakeRuntime(role, contract, verifier_raw=self.verifier_raw)
        self.instances[role] = instance
        return instance


class FakeScorer:
    def __init__(self, scores):
        self.scores = iter(scores)

    def step(self, logits):
        if np.asarray(logits).shape != (12,):
            raise AssertionError("wrong deployed detector output")
        return float(next(self.scores))


class ScorerFactory:
    def __init__(self, scores=(2.0, 3.0, -1.0, 4.0, -1.0)):
        self.scores = tuple(scores)
        self.instances = 0

    def __call__(self, topology, contract):
        self.instances += 1
        return FakeScorer(self.scores)


class FeatureFactory:
    def __init__(self, *, fail_source=None):
        self.calls = []
        self.fail_source = fail_source

    def __call__(self, path):
        self.calls.append(path.name)
        for index in range(75):  # 20 warmup calls + five scored 30-ms hops.
            if self.fail_source == path.stem and index == 1:
                raise RuntimeError("synthetic interruption")
            yield np.full((40,), index / 100.0, dtype=np.float32)


class CascadeFixture:
    def __init__(self, root: Path, *, audio_count=1):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.detector_model = root / "kizz_control_detector_ordered_state_streaming_int8.tflite"
        self.verifier_model = root / "kizz_control_candidate_verifier_int8.tflite"
        self.detector_model.write_bytes(b"detector-tflite")
        self.verifier_model.write_bytes(b"verifier-tflite")

        topology = OrderedStateTopology(tuple(KIZZ_CONTROL.phones), 1)
        topology_value = {
            "phrase_id": KIZZ_CONTROL.phrase_id,
            "text": KIZZ_CONTROL.text,
            "phones": list(topology.phones),
            "states_per_phone": 1,
            "state_count": topology.state_count,
            "state_names": list(topology.state_names),
            "background_index": topology.background_index,
            "silence_index": topology.silence_index,
            "first_ordered_state_index": topology.first_ordered_state_index,
        }
        decoder_contract = {
            "topology": topology_value,
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
        equivalence["evidence_sha256"] = detector_sha256_json(equivalence)
        self.detector_metadata = root / "detector-firmware-artifact.json"
        self.detector_payload = {
            "schema_version": 2,
            "kind": "kizz_control_ordered_state_detector_streaming_int8",
            "student_role": "permissive_detector_candidate_generator",
            "deployment_qualification": False,
            "artifact": {
                "filename": self.detector_model.name,
                "sha256": evaluator.sha256_file(self.detector_model),
                "bytes": self.detector_model.stat().st_size,
            },
            "source": {},
            "topology": topology_value,
            "decoder": {
                "algorithm": "ordered_state_sequence_score_numpy",
                "contract_version": 1,
                "arguments": EXPECTED_DECODER_ARGUMENTS,
                "score_semantics": "maximum_complete_left_to_right_log_odds_path",
                "contract_sha256": detector_sha256_json(decoder_contract),
                "reference_module": str(reference),
                "reference_module_sha256": evaluator.sha256_file(reference),
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
                "input": {"shape": [1, 3, 40], "dtype": "int8", "quantization": [0.5, -1]},
                "output": {"shape": [1, 1, 12], "dtype": "uint8", "quantization": [0.25, 128]},
                "output_semantics": "unnormalized_ordered_state_logits",
            },
            "static_memory_contract": {
                "batch_size": 1,
                "fixed_input_shape": True,
                "fixed_output_shape": True,
                "dynamic_tensor_shapes_forbidden": True,
                "external_state_tensor_count": 0,
                "persistent_state": "internal_tflite_variables",
                "tensor_audit": {"input_count": 1, "output_count": 1, "dynamic_shape_tensor_count": 0},
            },
            "equivalence": equivalence,
        }
        write_json(self.detector_metadata, self.detector_payload)

        self.detector_threshold = root / "detector-threshold.json"
        self.threshold_payload = {
            "schema_version": 1,
            "kind": "kizz_control_deployed_int8_validation_threshold",
            "deployment_qualification": False,
            "selection": {
                "fit_split": "validation",
                "test_used_for_selection": False,
                "qualified": True,
                "threshold": 1.0,
            },
            "threshold": 1.0,
            "bindings": {
                "artifact": binding(self.detector_model),
                "config": binding(self.detector_metadata),
            },
            "topology": topology_value,
            "decoder_contract_sha256": self.detector_payload["decoder"]["contract_sha256"],
        }
        write_json(self.detector_threshold, self.threshold_payload)

        self.training_report = root / "verifier-training.json"
        write_json(
            self.training_report,
            {
                "selection_contract": {
                    "selection_split": "validation",
                    "test_used_for_selection": False,
                },
                "winner": {
                    "frozen_threshold": 1.0 / (1.0 + math.exp(-0.75))
                },
            },
        )
        self.candidate_corpus = root / "candidate-corpus.json"
        write_json(
            self.candidate_corpus,
            {
                "schema_version": 1,
                "recipe": "kizz_control_candidate_conditioned_verifier_v1",
                "candidate_condition": "frozen_detector_trigger_only",
                "window_contract": {
                    "pre_context_frames": 220,
                    "trigger_frames": 1,
                    "post_context_frames": 39,
                    "padding": "zero",
                },
                "detector": {
                    "artifact": binding(self.detector_model),
                    "config": binding(self.detector_metadata),
                },
            },
        )
        self.verifier_metadata = root / "verifier-firmware-artifact.json"
        self.verifier_payload = {
            "schema_version": 1,
            "kind": "kizz_control_candidate_verifier_fixed_window_int8",
            "model_role": "detector_conditioned_candidate_verifier",
            "candidate_conditioned": True,
            "deployment_qualification": False,
            "artifact": {
                "filename": self.verifier_model.name,
                "sha256": evaluator.sha256_file(self.verifier_model),
                "bytes": self.verifier_model.stat().st_size,
            },
            "inputs": {
                "training_report": binding(self.training_report),
                "candidate_corpus": binding(self.candidate_corpus),
            },
            "threshold_contract": {
                "training_probability_threshold": 1.0
                / (1.0 + math.exp(-0.75)),
                "training_logit_threshold": 0.75,
                "deployed_logit_threshold": 0.75,
                "deployment_logit_bound": None,
                "quantization_logit_safety_margin": 0.0,
                "fit_split": "validation",
                "test_used_for_selection": False,
                "int8_threshold_retuning_performed": False,
            },
            "tensor_contracts": {
                "input": {
                    "shape": [1, 260, 40, 1],
                    "dtype": "int8",
                    "quantization": {"scale": 0.125, "zero_point": -3},
                },
                "output": {
                    "shape": [1, 1],
                    "dtype": "int8",
                    "quantization": {"scale": 0.1, "zero_point": 0},
                },
                "output_semantics": "unnormalized_candidate_verifier_logit",
                "fully_integer": True,
            },
            "static_memory_audit": {
                "fixed_input_shape": True,
                "fixed_output_shape": True,
                "dynamic_tensor_shapes_forbidden": True,
                "tensor_audit": {
                    "dynamic_shape_tensor_count": 0,
                    "variable_tensor_count": 0,
                },
            },
            "equivalence": {"passed": True},
            "provenance": {},
        }
        write_json(self.verifier_metadata, self.verifier_payload)

        self.audio = []
        examples = []
        for index in range(audio_count):
            name = chr(ord("a") + index)
            path = root / f"{name}.wav"
            write_wav(path, 1600 + index * 160)
            duration = (1600 + index * 160) / 16_000.0
            self.audio.append(path)
            examples.append(
                {
                    "source_id": name,
                    "path": str(path),
                    "sha256": evaluator.sha256_file(path),
                    "audio_sha256": evaluator.sha256_file(path),
                    "duration_seconds": duration,
                    "category": "speech" if index % 2 == 0 else "music",
                    "source": "MUSAN",
                    "split": "test",
                }
            )
        exposure = sum(item["duration_seconds"] for item in examples)
        self.locked_manifest = root / "locked.json"
        write_json(
            self.locked_manifest,
            {
                "schema_version": 2,
                "gate_scope": "locked_untouched_continuous_negative_corpus",
                "locked_before_scoring": True,
                "training_eligible": False,
                "counts": {
                    "files": len(examples),
                    "exposure_seconds": exposure,
                    "exposure_hours": exposure / 3600.0,
                    "categories": dict(
                        sorted(
                            {
                                category: sum(item["category"] == category for item in examples)
                                for category in {item["category"] for item in examples}
                            }.items()
                        )
                    ),
                },
                "examples": examples,
            },
        )
        self.checkpoint = root / "checkpoint.json"
        self.output = root / "report.json"

    def evaluate(self, **overrides):
        values = {
            "runtime_factory": RuntimeFactory(),
            "feature_stream_factory": FeatureFactory(),
            "scorer_factory": ScorerFactory(),
            "_minimum_exposure_hours": 0.0,
        }
        values.update(overrides)
        checkpoint = values.pop("checkpoint", self.checkpoint)
        output = values.pop("output", self.output)
        return evaluator.evaluate_shard(
            self.locked_manifest,
            self.detector_metadata,
            self.detector_model,
            self.detector_threshold,
            self.verifier_metadata,
            self.verifier_model,
            checkpoint,
            output,
            **values,
        )


class ContinuousCascadeTests(unittest.TestCase):
    def test_physical_recall_threshold_is_bound_and_cannot_tune_on_locked_audio(self):
        with tempfile.TemporaryDirectory() as raw:
            fixture = CascadeFixture(Path(raw))
            report_path = Path(raw) / "physical-threshold.json"
            report = {
                "schema_version": 1,
                "kind": "kizz_control_candidate_verifier_physical_recall_threshold",
                "deployment_qualification": False,
                "locked_audio_used_for_tuning": False,
                "test_scored_after_threshold_frozen": True,
                "threshold": -8.0,
                "selection": {
                    "qualified": True,
                    "fit_split": "physical_microphone_replay",
                    "test_used_for_selection": False,
                    "meets_minimum_recall": True,
                    "threshold": -8.0,
                    "detector_candidates": 13,
                    "accepted_candidates": 13,
                },
                "bindings": {
                    "artifact": binding(fixture.verifier_model),
                    "config": binding(fixture.verifier_metadata),
                },
            }
            write_json(report_path, report)
            self.assertEqual(
                evaluator._verifier_threshold_from_report(
                    report_path,
                    fixture.verifier_metadata,
                    fixture.verifier_model,
                    evaluator.sha256_file(fixture.verifier_metadata),
                    evaluator.sha256_file(fixture.verifier_model),
                ),
                -8.0,
            )

            report["locked_audio_used_for_tuning"] = True
            write_json(report_path, report)
            with self.assertRaisesRegex(ValueError, "physical verifier threshold evidence"):
                evaluator._verifier_threshold_from_report(
                    report_path,
                    fixture.verifier_metadata,
                    fixture.verifier_model,
                    evaluator.sha256_file(fixture.verifier_metadata),
                    evaluator.sha256_file(fixture.verifier_model),
                )

            report["locked_audio_used_for_tuning"] = False
            report["selection"]["accepted_candidates"] = 12
            write_json(report_path, report)
            with self.assertRaisesRegex(ValueError, "physical verifier threshold evidence"):
                evaluator._verifier_threshold_from_report(
                    report_path,
                    fixture.verifier_metadata,
                    fixture.verifier_model,
                    evaluator.sha256_file(fixture.verifier_metadata),
                    evaluator.sha256_file(fixture.verifier_model),
                )

    def test_ordered_int8_threshold_report_is_artifact_bound_and_validation_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "firmware-artifact.json"
            model = root / "ordered.tflite"
            report = root / "threshold.json"
            metadata.write_text("metadata\n", encoding="utf-8")
            model.write_bytes(b"ordered-model")
            payload = {
                "schema_version": 1,
                "kind": "kizz_control_ordered_state_candidate_verifier_int8_validation_threshold",
                "deployment_qualification": False,
                "test_scored_after_threshold_frozen": True,
                "decoder_contract_sha256": "d" * 64,
                "threshold": -24.5,
                "selection": {
                    "qualified": True,
                    "fit_split": "validation",
                    "test_used_for_selection": False,
                    "meets_minimum_recall": True,
                    "threshold": -24.5,
                },
                "bindings": {
                    "artifact": binding(model),
                    "config": binding(metadata),
                },
            }
            write_json(report, payload)
            self.assertEqual(
                evaluator._ordered_threshold_from_report(
                    report,
                    metadata,
                    model,
                    evaluator.sha256_file(metadata),
                    evaluator.sha256_file(model),
                    "d" * 64,
                ),
                -24.5,
            )
            payload["selection"]["test_used_for_selection"] = True
            write_json(report, payload)
            with self.assertRaisesRegex(ValueError, "validation-only"):
                evaluator._ordered_threshold_from_report(
                    report,
                    metadata,
                    model,
                    evaluator.sha256_file(metadata),
                    evaluator.sha256_file(model),
                    "d" * 64,
                )

    def test_ordered_verifier_discards_21_warmup_calls_and_exits_after_complete_path(self):
        topology = OrderedStateTopology(tuple(KIZZ_CONTROL.phones), 1)
        artifact = evaluator.FirmwareArtifact(
            "ordered_verifier",
            Path("metadata.json"),
            "0" * 64,
            {},
            Path("model.tflite"),
            "1" * 64,
            {"shape": [1, 3, 40], "dtype": "int8", "quantization": [0.1, -128]},
            {"shape": [1, 1, 12], "dtype": "uint8", "quantization": [0.1, 128]},
            tuple(),
        )
        contract = {
            "stride": 3,
            "phase_offset": 2,
            "warmup": 21,
            "calls": 87,
            "decoder_arguments": {
                "from_logits": True,
                "state_evidence_floor": None,
                "self_loop_probability": 0.6,
                "next_state_probability": 0.4,
            },
        }
        inputs = SimpleNamespace(
            ordered_verifier=artifact,
            ordered_verifier_topology=topology,
            ordered_verifier_contract=contract,
            ordered_verifier_threshold=-100.0,
        )
        runtime = FakeRuntime("ordered_verifier", artifact)
        accepted, score, calls = evaluator._score_ordered_verifier(
            np.zeros((260, 40), dtype=np.float32), inputs, runtime
        )
        self.assertTrue(accepted)
        self.assertGreaterEqual(score, -100.0)
        self.assertEqual(calls, 31)
        self.assertEqual(runtime.invoke_count, 31)
        self.assertEqual(runtime.reset_count, 1)

    def test_verifier_window_accepts_bound_causal_259_plus_trigger(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CascadeFixture(Path(directory))
            corpus_path = Path(
                fixture.verifier_payload["inputs"]["candidate_corpus"]["path"]
            )
            corpus = json.loads(corpus_path.read_text())
            corpus["window_contract"] = {
                "pre_context_frames": 259,
                "trigger_frames": 1,
                "post_context_frames": 0,
                "padding": "zero",
            }
            write_json(corpus_path, corpus)
            fixture.verifier_payload["inputs"]["candidate_corpus"]["sha256"] = evaluator.sha256_file(corpus_path)
            fixture.verifier_payload["inputs"]["candidate_corpus"]["bytes"] = corpus_path.stat().st_size
            write_json(fixture.verifier_metadata, fixture.verifier_payload)
            inputs = evaluator.validate_inputs(
                fixture.locked_manifest,
                fixture.detector_metadata,
                fixture.detector_model,
                fixture.detector_threshold,
                fixture.verifier_metadata,
                fixture.verifier_model,
                minimum_exposure_hours=0.0,
            )
            self.assertEqual(
                inputs.candidate_window,
                {"pre_context_frames": 259, "trigger_frames": 1, "post_context_frames": 0},
            )

    def test_threshold_regions_are_deduplicated_and_verifier_is_candidate_only(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CascadeFixture(Path(directory))
            factory = RuntimeFactory()
            report = fixture.evaluate(runtime_factory=factory)
            record = report["files"][0]
            self.assertEqual(record["detector_hops"], 25)
            self.assertEqual(record["detector_aligned_hops"], 5)
            self.assertEqual(record["detector_candidates"], 2)
            self.assertEqual(record["verifier_invocations"], 2)
            self.assertEqual(record["accepted_false_wakes"], 2)
            self.assertEqual(factory.instances["detector"].invoke_count, 25)
            self.assertEqual(factory.instances["verifier"].invoke_count, 2)
            self.assertFalse(report["threshold_policy"]["selection_performed"])
            self.assertFalse(report["threshold_policy"]["locked_audio_used_for_tuning"])
            self.assertFalse(report["deployment_qualification"])
            self.assertFalse(report["physical_hardware_proof"]["present"])

    def test_verifier_window_accepts_extension_metadata_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CascadeFixture(Path(directory))
            corpus_path = Path(
                fixture.verifier_payload["inputs"]["candidate_corpus"]["path"]
            )
            corpus = json.loads(corpus_path.read_text())
            corpus["detector"]["metadata"] = corpus["detector"].pop("config")
            write_json(corpus_path, corpus)
            fixture.verifier_payload["inputs"]["candidate_corpus"] = binding(
                corpus_path
            )
            write_json(fixture.verifier_metadata, fixture.verifier_payload)
            inputs = evaluator.validate_inputs(
                fixture.locked_manifest,
                fixture.detector_metadata,
                fixture.detector_model,
                fixture.detector_threshold,
                fixture.verifier_metadata,
                fixture.verifier_model,
                minimum_exposure_hours=0.0,
            )
            self.assertEqual(inputs.candidate_window["pre_context_frames"], 220)

    def test_compact_verifier_can_be_skipped_for_direct_ordered_cascade(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CascadeFixture(Path(directory))
            base = evaluator.validate_inputs(
                fixture.locked_manifest,
                fixture.detector_metadata,
                fixture.detector_model,
                fixture.detector_threshold,
                fixture.verifier_metadata,
                fixture.verifier_model,
                minimum_exposure_hours=0.0,
            )
            topology = OrderedStateTopology(tuple(KIZZ_CONTROL.phones), 1)
            ordered = evaluator.FirmwareArtifact(
                "ordered_verifier",
                Path("ordered-metadata.json"),
                "0" * 64,
                {},
                Path("ordered-model.tflite"),
                "1" * 64,
                {"shape": [1, 3, 40], "dtype": "int8", "quantization": [0.1, -128]},
                {"shape": [1, 1, 12], "dtype": "uint8", "quantization": [0.1, 128]},
                tuple(),
            )
            values = dict(base.__dict__)
            values.update(
                run_compact_verifier=False,
                ordered_verifier=ordered,
                ordered_verifier_topology=topology,
                ordered_verifier_contract={
                    "stride": 3,
                    "phase_offset": 2,
                    "warmup": 21,
                    "calls": 87,
                    "decoder_arguments": {
                        "from_logits": True,
                        "state_evidence_floor": None,
                        "self_loop_probability": 0.6,
                        "next_state_probability": 0.4,
                    },
                },
                ordered_verifier_threshold=-100.0,
            )
            inputs = SimpleNamespace(**values)
            detector_runtime = FakeRuntime("detector", inputs.detector)
            ordered_runtime = FakeRuntime("ordered_verifier", ordered)
            record = evaluator._score_file(
                inputs.locked.rows[0],
                inputs,
                detector_runtime,
                None,
                ordered_runtime,
                FeatureFactory(),
                ScorerFactory(),
            )
            self.assertEqual(record["detector_candidates"], 2)
            self.assertEqual(record["verifier_invocations"], 0)
            self.assertEqual(record["compact_verifier_accepts"], 0)
            self.assertEqual(record["ordered_verifier_runs"], 2)
            self.assertEqual(record["accepted_false_wakes"], 2)

    def test_verifier_compares_raw_logit_to_raw_logit_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CascadeFixture(Path(directory))
            threshold = -1.0
            probability = 1.0 / (1.0 + math.exp(-threshold))
            write_json(
                fixture.training_report,
                {
                    "selection_contract": {
                        "selection_split": "validation",
                        "test_used_for_selection": False,
                    },
                    "winner": {"frozen_threshold": probability},
                },
            )
            fixture.verifier_payload["inputs"]["training_report"] = binding(
                fixture.training_report
            )
            fixture.verifier_payload["threshold_contract"].update(
                {
                    "training_probability_threshold": probability,
                    "training_logit_threshold": threshold,
                    "deployed_logit_threshold": threshold,
                }
            )
            write_json(fixture.verifier_metadata, fixture.verifier_payload)

            report = fixture.evaluate(
                runtime_factory=RuntimeFactory(verifier_raw=-20)
            )

            self.assertEqual(report["files"][0]["verifier_invocations"], 2)
            self.assertEqual(report["files"][0]["accepted_false_wakes"], 0)
            self.assertEqual(
                report["threshold_policy"]["verifier"]["score_transform"],
                "dequantized_raw_logit",
            )

    def test_monotonic_bounded_verifier_compares_raw_output_to_transformed_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CascadeFixture(Path(directory))
            training_logit = -1.0
            probability = 1.0 / (1.0 + math.exp(-training_logit))
            bound = 4.0
            margin = 0.1
            deployed_threshold = bound * math.tanh(training_logit / bound) - margin
            write_json(
                fixture.training_report,
                {
                    "selection_contract": {
                        "selection_split": "validation",
                        "test_used_for_selection": False,
                    },
                    "winner": {"frozen_threshold": probability},
                },
            )
            fixture.verifier_payload["inputs"]["training_report"] = binding(
                fixture.training_report
            )
            fixture.verifier_payload["model"] = {
                "deployment_logit_transform": {
                    "kind": "bound_times_tanh_logit_over_bound",
                    "bound": bound,
                    "monotonic": True,
                }
            }
            fixture.verifier_payload["tensor_contracts"]["output_semantics"] = (
                "monotonic_bounded_candidate_verifier_logit"
            )
            fixture.verifier_payload["threshold_contract"].update(
                {
                    "training_probability_threshold": probability,
                    "training_logit_threshold": training_logit,
                    "deployed_logit_threshold": deployed_threshold,
                    "deployment_logit_bound": bound,
                    "quantization_logit_safety_margin": margin,
                }
            )
            write_json(fixture.verifier_metadata, fixture.verifier_payload)

            report = fixture.evaluate(runtime_factory=RuntimeFactory(verifier_raw=-20))

            self.assertEqual(report["files"][0]["accepted_false_wakes"], 0)
            self.assertAlmostEqual(
                report["threshold_policy"]["verifier"]["value"],
                deployed_threshold,
            )

            fixture.verifier_payload.pop("model")
            write_json(fixture.verifier_metadata, fixture.verifier_payload)
            with self.assertRaisesRegex(
                ValueError, "bounded-logit transform contract drift"
            ):
                fixture.evaluate(runtime_factory=RuntimeFactory(verifier_raw=-20))

    def test_frontend_detector_and_decoder_state_are_fresh_for_every_file(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CascadeFixture(Path(directory), audio_count=2)
            runtime = RuntimeFactory()
            frontend = FeatureFactory()
            scorers = ScorerFactory()
            report = fixture.evaluate(
                runtime_factory=runtime,
                feature_stream_factory=frontend,
                scorer_factory=scorers,
            )
            self.assertEqual(len(report["files"]), 2)
            self.assertEqual(len(frontend.calls), 2)
            self.assertEqual(scorers.instances, 2)
            self.assertEqual(runtime.instances["detector"].reset_count, 2)
            self.assertEqual(runtime.instances["verifier"].reset_count, 6)

    def test_corrupt_audio_and_model_hashes_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CascadeFixture(Path(directory))
            fixture.audio[0].write_bytes(fixture.audio[0].read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "audio hash drift"):
                fixture.evaluate()

            fixture = CascadeFixture(Path(directory) / "artifact")
            fixture.detector_model.write_bytes(b"tampered-model")
            with self.assertRaisesRegex(ValueError, "TFLite artifact hash drift"):
                fixture.evaluate()

    def test_incomplete_exposure_is_rejected_by_production_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CascadeFixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "exposure is incomplete"):
                evaluator.load_locked_manifest(fixture.locked_manifest)

    def test_interrupted_file_is_not_committed_and_resume_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CascadeFixture(Path(directory), audio_count=2)
            with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                fixture.evaluate(feature_stream_factory=FeatureFactory(fail_source="b"))
            checkpoint = json.loads(fixture.checkpoint.read_text())
            self.assertEqual([row["source_id"] for row in checkpoint["completed_files"]], ["a"])
            with self.assertRaisesRegex(ValueError, "checkpoint provenance drift"):
                fixture.evaluate(acceptance_ceiling=0.2)

            runtime = RuntimeFactory()
            report = fixture.evaluate(runtime_factory=runtime)
            self.assertEqual({row["source_id"] for row in report["files"]}, {"a", "b"})
            self.assertEqual(runtime.instances["detector"].reset_count, 1)

    def test_merge_rejects_missing_and_duplicate_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CascadeFixture(Path(directory), audio_count=3)
            shard = fixture.root / "shard-0.json"
            fixture.evaluate(shard_count=2, shard_index=0, output=shard)
            with self.assertRaisesRegex(ValueError, "missing shard reports"):
                evaluator.merge_shards([shard], fixture.root / "merged.json", _minimum_exposure_hours=0.0)
            with self.assertRaisesRegex(ValueError, "duplicate or invalid shard index"):
                evaluator.merge_shards([shard, shard], fixture.root / "merged.json", _minimum_exposure_hours=0.0)

    def test_exact_confidence_bounds_are_reported(self):
        interval = evaluator._poisson_interval(0, 100.0)
        self.assertAlmostEqual(
            interval["one_sided_upper_95_per_hour"],
            -np.log(0.05) / 100.0,
            places=12,
        )
        binomial = evaluator._binomial_interval(0, 10)
        self.assertEqual(binomial["lower"], 0.0)
        self.assertGreater(binomial["upper"], 0.0)

    def test_direct_cli_help(self):
        script = Path(evaluator.__file__).resolve()
        completed = subprocess.run(
            [sys.executable, str(script), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--merge-shards", completed.stdout)
        self.assertIn("--detector-threshold-report", completed.stdout)


if __name__ == "__main__":
    unittest.main()
