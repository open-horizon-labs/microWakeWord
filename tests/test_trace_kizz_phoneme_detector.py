import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.build_kizz_candidate_verifier_dataset import (
    build_candidate_verifier_dataset,
)
from tools.trace_kizz_phoneme_detector import (
    SCORE_FLOOR,
    WINDOW_LENGTHS,
    generate_detector_traces,
    sha256_file,
)


def _canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _row_hash(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


class FakeScorer:
    def __init__(self, score_vectors):
        self.score_vectors = [list(values) for values in score_vectors]
        self.observed = []

    def score(self, features):
        self.observed.append(np.array(features, copy=True))
        return self.score_vectors.pop(0)


class TraceFixture:
    def __init__(self, root: Path, *, frames=266):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.reference = root / "decoder.py"
        self.reference.write_text("# frozen decoder\n")
        self.artifact = root / "student.tflite"
        self.artifact.write_bytes(b"frozen-causal-int8-student")
        self.contract = {
            "schema_version": 1,
            "phrase_id": "kizz-control",
            "tokens": ["<blank>", "k", "OTHER"],
            "blank_id": 0,
            "other_id": 2,
            "canonical_path": [1],
            "collision_paths": {"kids control": [1, 2]},
        }
        self.decoder_contract = {
            "type": "kizz_ctc_phone_decoder",
            "implementation": "microwakeword.ctc_forward.exhaustive_sliding_forward_score",
            "algorithm": "forward_sum_ctc",
            "score": "summed_ctc_alignment_log_probability_divided_by_path_token_count",
            "window_lengths_frames": list(WINDOW_LENGTHS),
            "beta": 0.0,
            "selection": "maximum_canonical_fit_then_collision_margin",
            "compact_phone_contract_sha256": _canonical_hash(self.contract),
        }
        decoder_hash = _canonical_hash(self.decoder_contract)
        self.config = root / "firmware-artifact.json"
        self.config.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "artifact": {
                        "filename": self.artifact.name,
                        "sha256": sha256_file(self.artifact),
                        "bytes": self.artifact.stat().st_size,
                    },
                    "compact_phone_contract": self.contract,
                    "architecture": {"architecture_id": "control_mixconv"},
                    "timeline": {
                        "feature_step_seconds": 0.01,
                        "output_step_seconds": 0.03,
                        "output_frames": 66,
                        "stream_phase_offset_frames": 2,
                        "stream_phase_priming": "zero_prefix_then_observed_prefix",
                        "causal_warmup_derived": True,
                    },
                    "input": {"shape": [1, 3, 40], "dtype": "int8"},
                    "output": {"shape": [1, 1, 3], "dtype": "uint8"},
                    "decoder": {
                        "type": "deterministic_suffix_forward_sum_ctc",
                        "algorithm": "forward_sum_ctc",
                        "contract_sha256": decoder_hash,
                        "distillation_decoder_contract": self.decoder_contract,
                        "distillation_decoder_contract_sha256": decoder_hash,
                        "reference_module": str(self.reference),
                        "reference_module_sha256": sha256_file(self.reference),
                    },
                },
                sort_keys=True,
            )
        )
        self.threshold = root / "qualification.json"
        self.threshold.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "threshold": {
                        "threshold": 0.5,
                        "selection": "validation_only",
                    },
                    "artifact_metadata": {
                        "sha256": sha256_file(self.config),
                        "artifact_sha256": sha256_file(self.artifact),
                    },
                    "decoder": {"contract_sha256": decoder_hash},
                },
                sort_keys=True,
            )
        )
        self.audio = []
        for index in range(3):
            path = root / f"audio-{index}.wav"
            path.write_bytes(f"audio-{index}".encode())
            self.audio.append(path)
        self.features = root / "features.npy"
        values = np.arange(3 * frames * 40, dtype=np.float32).reshape(3, frames, 40)
        np.save(self.features, values)
        self.source = root / "source.json"
        self.rows = [
            self._row(0, "b-train", "train", 1, values[0]),
            self._row(1, "a-validation", "validation", 0, values[1]),
            self._row(2, "c-test", "test", 1, values[2]),
        ]
        self.source.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "array_sha256": {self.features.name: sha256_file(self.features)},
                    "examples": self.rows,
                },
                sort_keys=True,
            )
        )

    def _row(self, index, source_id, split, label, features):
        return {
            "source_id": source_id,
            "feature_index": index,
            "split": split,
            "label": label,
            "path": str(self.audio[index]),
            "audio_sha256": sha256_file(self.audio[index]),
            "feature_sha256": _row_hash(features),
            "speaker_id": f"speaker-{index}",
            "session_id": f"session-{index}",
            "ancestry_id": f"ancestry-{index}",
            "duration_seconds": features.shape[0] * 0.01,
        }

    def run(self, output: Path, score_vectors):
        scorer = FakeScorer(score_vectors)
        report = generate_detector_traces(
            self.artifact,
            self.config,
            self.threshold,
            self.source,
            self.features,
            output,
            scorer_factory=lambda artifact, contract: scorer,
        )
        return report, scorer


class TraceKizzPhonemeDetectorTests(unittest.TestCase):
    def test_emits_complete_deployed_geometry_and_threshold_region_events(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = TraceFixture(Path(directory))
            output = fixture.root / "trace.json"
            # Sources are scored in source-id order: validation, train, test.
            report, scorer = fixture.run(
                output,
                [
                    [0.1, 0.6, 0.8],
                    [0.9, 0.7, 0.1],
                    [0.2, 0.5, 0.1],
                ],
            )

            self.assertEqual([row["source_id"] for row in report["examples"]], [
                "a-validation", "b-train", "c-test"
            ])
            first = report["examples"][0]
            self.assertEqual(first["scores"], [0.1, 0.6, 0.8])
            self.assertEqual(first["feature_frame_indexes"], [259, 262, 265])
            self.assertEqual(
                first["events"],
                [
                    {
                        "score_frame_index": 2,
                        "feature_frame_index": 265,
                        "score": 0.8,
                        "threshold_region_start_score_frame_index": 1,
                        "threshold_region_end_score_frame_index": 2,
                        "threshold_region_start_feature_frame_index": 262,
                        "threshold_region_end_feature_frame_index": 265,
                    }
                ],
            )
            self.assertEqual(first["split"], "validation")
            self.assertEqual(first["label"], 0)
            self.assertEqual(first["path"], str(fixture.audio[1].resolve()))
            self.assertEqual(first["audio_sha256"], sha256_file(fixture.audio[1]))
            self.assertEqual(
                first["source_feature_sha256"], fixture.rows[1]["feature_sha256"]
            )
            self.assertEqual(len(scorer.observed), 3)
            self.assertEqual(
                report["detector"]["score_geometry"]["feature_offset_frames"], 259
            )
            self.assertEqual(
                report["detector"]["score_geometry"]["feature_stride_frames"], 3
            )
            self.assertEqual(json.loads(output.read_text()), report)

    def test_output_is_deterministic_and_consumable_by_candidate_builder(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = TraceFixture(Path(directory))
            first = fixture.root / "trace-a.json"
            second = fixture.root / "trace-b.json"
            vectors = [[0.1, 0.7, 0.1], [0.8, 0.1, 0.1], [0.1, 0.6, 0.1]]
            first_report, _ = fixture.run(first, vectors)
            second_report, _ = fixture.run(second, vectors)
            self.assertEqual(first_report, second_report)
            self.assertEqual(first.read_bytes(), second.read_bytes())

            lock = fixture.root / "lock.json"
            lock.write_text(
                json.dumps(
                    {
                        "gate_scope": "locked_untouched_continuous_negative_corpus",
                        "locked_before_scoring": True,
                        "examples": [],
                    },
                    sort_keys=True,
                )
            )
            candidate = build_candidate_verifier_dataset(
                fixture.source,
                fixture.features,
                first,
                fixture.root / "candidates",
                locked_holdout_manifest=lock,
                pre_context_frames=1,
                post_context_frames=0,
                hard_negative_top_k=1,
            )
            self.assertEqual(candidate["counts"]["selected_candidates"], 3)
            self.assertEqual(
                sorted(row["detector_feature_frame_index"] for row in candidate["examples"]),
                [259, 262, 262],
            )

    def test_short_sources_are_right_padded_but_keep_causal_endpoint_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = TraceFixture(Path(directory), frames=40)
            report, _ = fixture.run(
                fixture.root / "trace.json", [[0.1], [0.6], [0.2]]
            )
            self.assertTrue(all(len(row["scores"]) == 1 for row in report["examples"]))
            self.assertTrue(
                all(row["feature_frame_indexes"] == [259] for row in report["examples"])
            )

    def test_negative_infinity_is_losslessly_below_threshold_for_json_consumer(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = TraceFixture(Path(directory), frames=260)
            report, _ = fixture.run(
                fixture.root / "trace.json", [[-np.inf], [0.6], [0.1]]
            )
            self.assertEqual(report["examples"][0]["scores"], [SCORE_FLOOR])
            self.assertEqual(
                report["counts"]["negative_infinity_scores_serialized"], 1
            )
            self.assertEqual(report["examples"][0]["events"], [])

    def test_rejects_artifact_config_threshold_and_decoder_hash_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for case in ("artifact", "config", "threshold", "decoder"):
                fixture = TraceFixture(root / case)
                if case == "artifact":
                    fixture.artifact.write_bytes(b"changed")
                elif case == "config":
                    payload = json.loads(fixture.config.read_text())
                    payload["input"]["shape"] = [1, 1, 40]
                    fixture.config.write_text(json.dumps(payload, sort_keys=True))
                elif case == "threshold":
                    payload = json.loads(fixture.threshold.read_text())
                    payload["artifact_metadata"]["sha256"] = "0" * 64
                    fixture.threshold.write_text(json.dumps(payload, sort_keys=True))
                else:
                    fixture.reference.write_text("# changed decoder\n")
                with self.assertRaises(ValueError):
                    fixture.run(fixture.root / "trace.json", [[0.1] * 3] * 3)

    def test_rejects_source_array_row_audio_shape_and_split_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = ("array", "row", "audio", "shape", "split", "leakage")
            for case in cases:
                fixture = TraceFixture(root / case)
                if case == "array":
                    values = np.load(fixture.features)
                    values[0, 0, 0] += 1
                    np.save(fixture.features, values)
                elif case == "row":
                    payload = json.loads(fixture.source.read_text())
                    payload["examples"][0]["feature_sha256"] = "0" * 64
                    fixture.source.write_text(json.dumps(payload, sort_keys=True))
                elif case == "audio":
                    fixture.audio[0].write_bytes(b"changed")
                elif case == "shape":
                    values = np.zeros((3, 266, 39), dtype=np.float32)
                    np.save(fixture.features, values)
                    payload = json.loads(fixture.source.read_text())
                    payload["array_sha256"][fixture.features.name] = sha256_file(
                        fixture.features
                    )
                    fixture.source.write_text(json.dumps(payload, sort_keys=True))
                elif case == "split":
                    payload = json.loads(fixture.source.read_text())
                    payload["examples"][0]["split"] = "evaluation"
                    fixture.source.write_text(json.dumps(payload, sort_keys=True))
                else:
                    payload = json.loads(fixture.source.read_text())
                    payload["examples"][1]["speaker_id"] = payload["examples"][0][
                        "speaker_id"
                    ]
                    fixture.source.write_text(json.dumps(payload, sort_keys=True))
                with self.assertRaises((ValueError, TypeError)):
                    fixture.run(fixture.root / "trace.json", [[0.1] * 3] * 3)

    def test_rejects_scorer_count_nan_and_positive_infinity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, vector in (
                ("count", [0.1, 0.2]),
                ("nan", [0.1, np.nan, 0.2]),
                ("positive-infinity", [0.1, np.inf, 0.2]),
            ):
                fixture = TraceFixture(root / name)
                with self.assertRaises(ValueError):
                    fixture.run(
                        fixture.root / "trace.json",
                        [vector, [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]],
                    )


if __name__ == "__main__":
    unittest.main()
