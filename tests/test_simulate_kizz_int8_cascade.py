import hashlib
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools import simulate_kizz_int8_cascade as simulator


def _feature_hash(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


class CascadeFixture:
    def __init__(self, root: Path):
        self.root = root
        self.detector_tflite = root / "detector.tflite"
        self.verifier_tflite = root / "verifier.tflite"
        self.detector_tflite.write_bytes(b"fake-stateful-int8-detector")
        self.verifier_tflite.write_bytes(b"fake-int8-verifier")
        self.threshold = root / "detector-threshold.json"
        self.threshold.write_text('{"threshold":-12.5}\n', encoding="utf-8")

        self.source_features = root / "source-features.npy"
        source_values = np.zeros((2, 260, 40), dtype=np.float32)
        source_values[0, :, 0] = np.arange(260, dtype=np.float32) / 260.0
        source_values[1, :, 1] = -0.5
        np.save(self.source_features, source_values, allow_pickle=False)
        self.source_manifest = root / "source-manifest.json"
        self.source_rows = [
            {
                "source_id": "train-positive",
                "split": "train",
                "label": 1,
                "feature_index": 0,
                "feature_sha256": _feature_hash(source_values[0]),
                "duration_seconds": 2.6,
            },
            {
                "source_id": "test-negative",
                "split": "test",
                "label": 0,
                "feature_index": 1,
                "feature_sha256": _feature_hash(source_values[1]),
                "duration_seconds": 2.6,
            },
        ]
        self.source_payload = {
            "schema_version": 1,
            "recipe": "kizz_detector_scoring_corpus_v1",
            "examples": self.source_rows,
            "array_sha256": {
                self.source_features.name: simulator.sha256_file(self.source_features)
            },
        }
        self.write_json(self.source_manifest, self.source_payload)

        self.detector_metadata = root / "detector-firmware-artifact.json"
        self.detector_payload = {
            "schema_version": 2,
            "kind": "kizz_control_ordered_state_detector_streaming_int8",
            "student_role": "permissive_detector_candidate_generator",
            "deployment_qualification": False,
            "artifact": {
                "filename": self.detector_tflite.name,
                "sha256": simulator.sha256_file(self.detector_tflite),
                "bytes": self.detector_tflite.stat().st_size,
            },
            "source": {},
            "topology": {"state_count": 12},
            "timeline": {
                "stream_input_frames_per_call": 3,
                "stream_hop_seconds": 0.03,
                "stream_phase_offset_frames": 1,
            },
            "tensor_contracts": {
                "input": {
                    "shape": [1, 3, 40],
                    "dtype": "int8",
                    "quantization": [0.25, 0],
                },
                "output": {
                    "shape": [1, 1, 12],
                    "dtype": "uint8",
                    "quantization": [0.125, 128],
                },
            },
            "static_memory_contract": {
                "tensor_audit": {
                    "tensor_count": 9,
                    "declared_tensor_bytes_sum": 4096,
                }
            },
        }
        self.write_json(self.detector_metadata, self.detector_payload)

        self.detector_trace = root / "detector-traces.json"
        self.trace_rows = [
            {
                "source_id": row["source_id"],
                "split": row["split"],
                "label": row["label"],
                "feature_index": row["feature_index"],
                "source_feature_sha256": row["feature_sha256"],
                "events": [
                    {
                        "score_frame_index": 3 + index,
                        "feature_frame_index": 230 + index,
                        "score": -10.0 + index,
                    }
                ],
            }
            for index, row in enumerate(self.source_rows)
        ]
        self.trace_payload = {
            "schema_version": 1,
            "recipe": "kizz_control_ordered_state_deployed_int8_trace_v1",
            "deployment_qualification": False,
            "source_manifest": self.binding(self.source_manifest),
            "source_features": self.binding(self.source_features),
            "detector": {
                "artifact": self.binding(self.detector_tflite),
                "config": self.binding(self.detector_metadata),
                "threshold": {**self.binding(self.threshold), "value": -12.5},
                "event_policy": "recorded_events",
                "score_geometry": {
                    "feature_stride_frames": 3,
                    "feature_offset_frames": 2,
                    "feature_hop_ms": 10.0,
                },
            },
            "counts": {"candidate_events": 2},
            "evaluation": {
                "threshold_frozen_before_test_scoring": True,
                "test_used_for_selection": False,
            },
            "examples": self.trace_rows,
        }
        self.write_json(self.detector_trace, self.trace_payload)

        self.candidate_root = root / "candidate-dataset"
        self.candidate_root.mkdir()
        self.candidate_features = self.candidate_root / "features.npy"
        candidate_values = np.zeros((2, 260, 40), dtype=np.float16)
        for index in range(2):
            feature_frame = 230 + index
            start = feature_frame - 220
            stop = min(260, feature_frame + 40)
            copied = source_values[index, start:stop]
            candidate_values[index, : len(copied)] = copied.astype(np.float16)
        np.save(self.candidate_features, candidate_values, allow_pickle=False)
        self.candidate_rows = [
            {
                "candidate_id": f"{row['source_id']}::candidate",
                "source_id": f"{row['source_id']}::candidate",
                "parent_source_id": row["source_id"],
                "split": row["split"],
                "label": row["label"],
                "feature_index": index,
                "detector_conditioned": True,
                "detector_score_frame_index": 3 + index,
                "candidate_feature_sha256": _feature_hash(candidate_values[index]),
            }
            for index, row in enumerate(self.source_rows)
        ]
        self.candidate_corpus = self.candidate_root / "corpus.json"
        self.candidate_payload = {
            "schema_version": 1,
            "recipe": "kizz_control_candidate_conditioned_verifier_v1",
            "candidate_condition": "frozen_detector_trigger_only",
            "window_contract": {
                "pre_context_frames": 220,
                "trigger_frames": 1,
                "post_context_frames": 39,
                "padding": "zero",
            },
            "bindings": {
                "source_manifest": self.binding(self.source_manifest),
                "source_features": self.binding(self.source_features),
                "detector_traces": self.binding(self.detector_trace),
            },
            "detector": {
                "artifact": self.binding(self.detector_tflite),
                "config": self.binding(self.detector_metadata),
            },
            "counts": {"selected_candidates": 2},
            "examples": self.candidate_rows,
            "array_sha256": {
                "features.npy": simulator.sha256_file(self.candidate_features)
            },
        }
        self.write_json(self.candidate_corpus, self.candidate_payload)

        self.verifier_metadata = root / "verifier-firmware-artifact.json"
        self.verifier_payload = {
            "schema_version": 1,
            "kind": "kizz_control_candidate_verifier_fixed_window_int8",
            "model_role": "detector_conditioned_candidate_verifier",
            "candidate_conditioned": True,
            "deployment_qualification": False,
            "artifact": {
                "filename": self.verifier_tflite.name,
                "sha256": simulator.sha256_file(self.verifier_tflite),
                "bytes": self.verifier_tflite.stat().st_size,
            },
            "inputs": {"candidate_corpus": self.binding(self.candidate_corpus)},
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
                    "quantization": [0.125, -3],
                },
                "output": {
                    "shape": [1, 1],
                    "dtype": "int8",
                    "quantization": [0.02, 0],
                },
            },
            "static_memory_audit": {
                "tensor_audit": {
                    "tensor_count": 14,
                    "declared_tensor_bytes_sum": 8192,
                }
            },
        }
        self.write_json(self.verifier_metadata, self.verifier_payload)

    @staticmethod
    def write_json(path: Path, payload):
        path.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def binding(path: Path):
        return {
            "path": str(path.resolve()),
            "sha256": simulator.sha256_file(path),
        }

    def simulate(self, **kwargs):
        return simulator.simulate_cascade(
            self.detector_metadata,
            self.verifier_metadata,
            self.source_manifest,
            self.source_features,
            self.detector_trace,
            self.candidate_root,
            **kwargs,
        )


class FakeRuntime:
    def __init__(self, role, contract):
        self.role = role
        self.contract = contract
        self.tensor_count = 7 if role == "detector" else 11
        self.tensor_bytes = 3000 if role == "detector" else 6000
        self.reset_count = 0
        self.invoke_count = 0
        self.state_count = 0

    def reset(self):
        self.reset_count += 1
        self.state_count = 0

    def invoke(self, values):
        self.invoke_count += 1
        self.state_count += 1
        if self.role == "detector":
            return np.full((1, 1, 12), self.state_count % 251, dtype=np.uint8)
        return np.asarray([[self.state_count % 127]], dtype=np.int8)


class RuntimeFactory:
    def __init__(self):
        self.runtimes = {}

    def __call__(self, role, artifact, contract):
        self.asserted_artifact = artifact
        runtime = FakeRuntime(role, contract)
        self.runtimes[role] = runtime
        return runtime


class DurationClock:
    def __init__(self, durations_ns):
        self.durations = iter(durations_ns)
        self.at_start = True
        self.value = 0

    def __call__(self):
        if self.at_start:
            self.at_start = False
            return self.value
        self.value += next(self.durations)
        self.at_start = True
        return self.value


class ElapsedClock:
    def __init__(self, elapsed_ns):
        self.values = iter((0, elapsed_ns))

    def __call__(self):
        return next(self.values)


class SimulateKizzInt8CascadeTests(unittest.TestCase):
    def test_tensor_contract_accepts_converter_quantization_mapping(self):
        value = simulator._tensor_contract(
            {
                "shape": [1, 260, 40, 1],
                "dtype": "int8",
                "quantization": {"scale": 0.1, "zero_point": -128},
            },
            "verifier input",
        )
        self.assertEqual(value["quantization"], [0.1, -128])

    def test_replays_exact_hops_candidates_resets_and_excludes_warmup(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CascadeFixture(Path(directory))
            factory = RuntimeFactory()
            detector_hops = 174  # 87 stateful calls for each 260-frame source.
            clock = DurationClock([1_000_000] * detector_hops + [5_000_000] * 2)
            report = fixture.simulate(
                warmup_detector_hops=3,
                warmup_verifier_candidates=2,
                runtime_factory=factory,
                clock_ns=clock,
                elapsed_clock_ns=ElapsedClock(1_000_000_000),
            )
            functional = report["functional_replay"]
            self.assertEqual(functional["detector_hops"], detector_hops)
            self.assertEqual(functional["recorded_detector_candidates"], 2)
            self.assertEqual(functional["verifier_candidate_invocations"], 2)
            self.assertEqual(functional["candidate_dataset_selected_examples"], 2)
            self.assertEqual(functional["detector_state_resets"], 2)
            self.assertEqual(functional["verifier_state_resets"], 2)
            self.assertEqual(factory.runtimes["detector"].invoke_count, detector_hops + 3)
            self.assertEqual(factory.runtimes["verifier"].invoke_count, 2 + 2)
            self.assertEqual(report["host_timing"]["detector_hop"]["count"], detector_hops)
            self.assertEqual(report["host_timing"]["detector_hop"]["p99_ms"], 1.0)
            self.assertEqual(report["host_timing"]["verifier_candidate"]["p95_ms"], 5.0)
            self.assertEqual(report["host_timing"]["benchmark_elapsed_seconds"], 1.0)
            self.assertEqual(report["host_timing"]["end_to_end_audio_replay_x_realtime"], 5.2)
            self.assertAlmostEqual(
                report["rates_and_duty_cycle"]["candidate_probability_per_detector_hop"],
                2 / detector_hops,
            )
            proxy = report["analytical_scheduler_budget"]["host_p95_proxy"]
            self.assertAlmostEqual(proxy["combined_ms_per_hop"], 1 + 5 * 2 / detector_hops)
            self.assertFalse(report["deployment_qualification"])
            self.assertFalse(report["host_timing"]["portable_to_esp32_s3"])
            self.assertFalse(
                report["analytical_scheduler_budget"]["hardware_qualification"]
            )

    def test_functional_digests_are_stable_when_timing_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CascadeFixture(Path(directory))

            def run(detector_ns, verifier_ns, elapsed, detector_warmup, verifier_warmup):
                return fixture.simulate(
                    warmup_detector_hops=detector_warmup,
                    warmup_verifier_candidates=verifier_warmup,
                    runtime_factory=RuntimeFactory(),
                    clock_ns=DurationClock([detector_ns] * 174 + [verifier_ns] * 2),
                    elapsed_clock_ns=ElapsedClock(elapsed),
                )

            first = run(1_000_000, 2_000_000, 800_000_000, 0, 0)
            second = run(7_000_000, 9_000_000, 2_000_000_000, 4, 3)
            self.assertEqual(
                first["functional_replay"]["detector_raw_output_sha256"],
                second["functional_replay"]["detector_raw_output_sha256"],
            )
            self.assertEqual(
                first["functional_replay"]["verifier_raw_output_sha256"],
                second["functional_replay"]["verifier_raw_output_sha256"],
            )
            self.assertNotEqual(
                first["host_timing"]["detector_hop"]["p50_ms"],
                second["host_timing"]["detector_hop"]["p50_ms"],
            )
            self.assertFalse(first["threshold_policy"]["selection_performed"])
            self.assertFalse(first["threshold_policy"]["test_used_for_selection"])

    def test_allows_training_only_superset_rows_not_in_current_detector_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CascadeFixture(Path(directory))
            existing = np.load(fixture.candidate_features, allow_pickle=False)
            supplemental = np.full((1, 260, 40), 0.25, dtype=np.float16)
            combined = np.concatenate((existing, supplemental), axis=0)
            np.save(fixture.candidate_features, combined, allow_pickle=False)
            fixture.candidate_payload["examples"].append(
                {
                    "candidate_id": "legacy-hard-negative::candidate",
                    "source_id": "legacy-hard-negative::candidate",
                    "parent_source_id": "train-positive",
                    "split": "train",
                    "label": 0,
                    "feature_index": 2,
                    "detector_conditioned": True,
                    "detector_score_frame_index": 999,
                    "candidate_feature_sha256": _feature_hash(supplemental[0]),
                }
            )
            fixture.candidate_payload["counts"]["selected_candidates"] = 3
            fixture.candidate_payload["array_sha256"]["features.npy"] = (
                simulator.sha256_file(fixture.candidate_features)
            )
            fixture.write_json(fixture.candidate_corpus, fixture.candidate_payload)
            fixture.verifier_payload["inputs"]["candidate_corpus"] = fixture.binding(
                fixture.candidate_corpus
            )
            fixture.write_json(fixture.verifier_metadata, fixture.verifier_payload)

            report = fixture.simulate(runtime_factory=RuntimeFactory())
            functional = report["functional_replay"]
            self.assertEqual(functional["verifier_candidate_invocations"], 2)
            self.assertEqual(functional["candidate_dataset_selected_examples"], 3)
            self.assertEqual(
                functional["candidate_dataset_supplemental_training_examples"], 1
            )
            self.assertFalse(
                report["rates_and_duty_cycle"][
                    "candidate_dataset_is_exact_benchmark_invocation_set"
                ]
            )

    def test_percentiles_use_observed_higher_order_statistic(self):
        result = simulator._quantiles([1_000_000, 2_000_000, 3_000_000, 8_000_000])
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["p50_ms"], 3.0)
        self.assertEqual(result["p95_ms"], 8.0)
        self.assertEqual(result["p99_ms"], 8.0)
        self.assertEqual(result["max_ms"], 8.0)
        self.assertEqual(result["total_ms"], 14.0)

    def test_rejects_corrupt_detector_artifact_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CascadeFixture(Path(directory))
            fixture.detector_tflite.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "detector TFLite artifact hash drift"):
                fixture.simulate(runtime_factory=RuntimeFactory())

    def test_rejects_trace_rebound_to_different_source_features(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CascadeFixture(Path(directory))
            fixture.trace_payload["source_features"]["sha256"] = "0" * 64
            fixture.write_json(fixture.detector_trace, fixture.trace_payload)
            # Preserve the candidate->trace binding so the failure is specifically transitive.
            fixture.candidate_payload["bindings"]["detector_traces"] = fixture.binding(
                fixture.detector_trace
            )
            fixture.write_json(fixture.candidate_corpus, fixture.candidate_payload)
            fixture.verifier_payload["inputs"]["candidate_corpus"] = fixture.binding(
                fixture.candidate_corpus
            )
            fixture.write_json(fixture.verifier_metadata, fixture.verifier_payload)
            with self.assertRaisesRegex(ValueError, "trace source features hash drift"):
                fixture.simulate(runtime_factory=RuntimeFactory())

    def test_rejects_candidate_dataset_not_bound_by_verifier(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CascadeFixture(Path(directory))
            fixture.verifier_payload["inputs"]["candidate_corpus"]["sha256"] = "f" * 64
            fixture.write_json(fixture.verifier_metadata, fixture.verifier_payload)
            with self.assertRaisesRegex(ValueError, "verifier.inputs.candidate_corpus hash drift"):
                fixture.simulate(runtime_factory=RuntimeFactory())

    def test_rejects_test_fitted_verifier_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CascadeFixture(Path(directory))
            fixture.verifier_payload["threshold_contract"]["test_used_for_selection"] = True
            fixture.write_json(fixture.verifier_metadata, fixture.verifier_payload)
            with self.assertRaisesRegex(ValueError, "validation-only threshold"):
                fixture.simulate(runtime_factory=RuntimeFactory())

    def test_rejects_detector_tensor_contract_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CascadeFixture(Path(directory))
            fixture.detector_payload["tensor_contracts"]["input"]["shape"] = [1, 2, 40]
            fixture.write_json(fixture.detector_metadata, fixture.detector_payload)
            with self.assertRaisesRegex(ValueError, "streaming timeline"):
                fixture.simulate(runtime_factory=RuntimeFactory())

    def test_negative_warmup_and_backwards_timing_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CascadeFixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "warmup"):
                fixture.simulate(
                    warmup_detector_hops=-1,
                    runtime_factory=RuntimeFactory(),
                )

            class BackwardsClock:
                def __init__(self):
                    self.values = iter((5, 4))

                def __call__(self):
                    return next(self.values)

            with self.assertRaisesRegex(ValueError, "moved backwards"):
                fixture.simulate(
                    warmup_detector_hops=0,
                    warmup_verifier_candidates=0,
                    runtime_factory=RuntimeFactory(),
                    clock_ns=BackwardsClock(),
                    elapsed_clock_ns=ElapsedClock(1),
                )

    def test_cli_help_does_not_import_tensorflow_or_write_external_data(self):
        script = Path(simulator.__file__).resolve()
        completed = subprocess.run(
            [sys.executable, str(script), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--detector-metadata", completed.stdout)
        self.assertIn("--candidate-dataset", completed.stdout)


if __name__ == "__main__":
    unittest.main()
