import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import evaluate_kizz_cascade as cascade


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _audio_hash(source_id: str) -> str:
    return _sha_bytes(f"audio:{source_id}".encode())


def _event(candidate_id: str, detector_score: float, verifier_score: float, *, start=0.1, end=0.9, timestamp=0.5):
    common = {
        "candidate_id": candidate_id,
        "start_seconds": start,
        "end_seconds": end,
        "timestamp_seconds": timestamp,
        "window_sha256": _sha_bytes(f"window:{candidate_id}".encode()),
    }
    return ({**common, "score": detector_score}, {**common, "score": verifier_score})


def _source(source_id: str, split: str, truth: str, pairs, *, duration=None, audio_sha256=None, speaker_id=None):
    detector_events, verifier_events = zip(*pairs) if pairs else ((), ())
    base = {
        "source_id": source_id,
        "split": split,
        "truth": truth,
        "duration_seconds": duration if duration is not None else (360_000.0 if truth == "negative" else 1.0),
        "audio_sha256": audio_sha256 or _audio_hash(source_id),
        "speaker_id": speaker_id or source_id,
    }
    return (
        {**base, "events": list(detector_events)},
        {**base, "events": list(verifier_events)},
    )


class TraceFixture:
    def __init__(self, root: Path, source_pairs):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        detector_artifact = root / "detector.tflite"
        verifier_artifact = root / "verifier.tflite"
        detector_artifact.write_bytes(b"frozen-detector")
        verifier_artifact.write_bytes(b"frozen-verifier")
        detector_sources, verifier_sources = zip(*source_pairs)
        self.detector_payload = {
            "schema_version": 1,
            "trace_kind": "detector",
            "artifact": {
                "path": str(detector_artifact),
                "sha256": cascade.sha256_file(detector_artifact),
            },
            "sources": list(detector_sources),
        }
        self.verifier_payload = {
            "schema_version": 1,
            "trace_kind": "verifier",
            "artifact": {
                "path": str(verifier_artifact),
                "sha256": cascade.sha256_file(verifier_artifact),
            },
            "sources": list(verifier_sources),
        }
        self.detector_path = root / "detector-trace.json"
        self.verifier_path = root / "verifier-trace.json"
        self.write()

    def write(self):
        self.detector_path.write_text(json.dumps(self.detector_payload, sort_keys=True) + "\n")
        self.verifier_path.write_text(json.dumps(self.verifier_payload, sort_keys=True) + "\n")
        self.detector_sha = cascade.sha256_file(self.detector_path)
        self.verifier_sha = cascade.sha256_file(self.verifier_path)

    def evaluate(self, **kwargs):
        return cascade.evaluate_cascade(
            self.detector_path,
            self.verifier_path,
            detector_trace_sha256=self.detector_sha,
            verifier_trace_sha256=self.verifier_sha,
            **kwargs,
        )


def _standard_sources():
    values = []
    for split, prefix in (("validation", "v"), ("test", "t")):
        values.extend(
            [
                _source(f"{prefix}p1", split, "positive", [_event("c1", 0.95, 0.92)]),
                _source(f"{prefix}p2", split, "positive", [_event("c2", 0.90, 0.88)]),
                _source(f"{prefix}p3", split, "positive", [_event("c3", 0.85, 0.84)]),
                _source(f"{prefix}p4", split, "positive", [_event("c4", 0.80, 0.80)]),
                _source(f"{prefix}n", split, "negative", []),
            ]
        )
    return values


class EvaluateKizzCascadeTests(unittest.TestCase):
    def test_validation_only_thresholds_and_test_is_evaluated_once(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = TraceFixture(Path(directory), _standard_sources())
            original_metrics = cascade._metrics
            evaluated_splits = []

            def recording_metrics(evidence, detector_threshold, verifier_threshold):
                evaluated_splits.append(evidence["sources"][0].split)
                return original_metrics(evidence, detector_threshold, verifier_threshold)

            with mock.patch.object(cascade, "_metrics", side_effect=recording_metrics):
                report = fixture.evaluate()
            self.assertTrue(report["qualified"])
            self.assertEqual(evaluated_splits.count("test"), 1)
            self.assertEqual(report["protocol"]["test_evaluations"], 1)
            self.assertTrue(report["protocol"]["test_scored_once_at_frozen_thresholds"])
            self.assertFalse(report["protocol"]["test_used_for_threshold_selection"])

    def test_test_scores_cannot_change_frozen_thresholds(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = TraceFixture(Path(directory), _standard_sources())
            first = fixture.evaluate()
            for row in fixture.verifier_payload["sources"]:
                if row["split"] == "test" and row["truth"] == "positive":
                    for event in row["events"]:
                        event["score"] = 0.0
            fixture.write()
            second = fixture.evaluate()
            self.assertEqual(
                first["threshold_selection"]["thresholds"],
                second["threshold_selection"]["thresholds"],
            )
            self.assertTrue(first["qualified"])
            self.assertFalse(second["qualified"])
            self.assertIn("test_cascade_recall_below_target", second["failure_reasons"])

    def test_cascade_recall_is_detector_recall_times_conditional_recall(self):
        with tempfile.TemporaryDirectory() as directory:
            sources = []
            for split, prefix in (("validation", "v"), ("test", "t")):
                sources.extend(
                    [
                        _source(f"{prefix}p1", split, "positive", [_event("p1", 0.9, 0.9)]),
                        _source(f"{prefix}p2", split, "positive", [_event("p2", 0.8, 0.1)]),
                        _source(f"{prefix}p3", split, "positive", []),
                        _source(f"{prefix}p4", split, "positive", []),
                    ]
                )
                negatives = [
                    _event(
                        f"n{index}",
                        0.8,
                        0.5,
                        start=index * 2.0,
                        end=index * 2.0 + 0.5,
                        timestamp=index * 2.0 + 0.25,
                    )
                    for index in range(10)
                ]
                sources.append(_source(f"{prefix}n", split, "negative", negatives))
            report = TraceFixture(Path(directory), sources).evaluate(
                detector_recall_target=0.5,
                cascade_recall_target=0.25,
            )
            metrics = report["validation"]
            self.assertEqual(metrics["detector_recall"], 0.5)
            self.assertEqual(metrics["conditional_verifier_recall"], 0.5)
            self.assertEqual(metrics["joint_recall"], 0.25)
            self.assertAlmostEqual(
                metrics["joint_recall"],
                metrics["detector_recall"] * metrics["conditional_verifier_recall"],
            )
            self.assertEqual(report["threshold_selection"]["thresholds"]["verifier"], 0.9)

    def test_temporally_overlapping_candidates_are_deduplicated_before_sweep(self):
        with tempfile.TemporaryDirectory() as directory:
            sources = []
            for split, prefix in (("validation", "v"), ("test", "t")):
                overlapping = [
                    _event("peak-detector", 0.9, 0.8, start=0.1, end=0.7, timestamp=0.4),
                    _event("peak-verifier", 0.8, 0.99, start=0.6, end=0.9, timestamp=0.7),
                ]
                sources.extend(
                    [
                        _source(f"{prefix}p", split, "positive", overlapping),
                        _source(f"{prefix}n", split, "negative", []),
                    ]
                )
            report = TraceFixture(Path(directory), sources).evaluate()
            self.assertEqual(
                report["deduplication"]["validation"],
                {"raw_candidates": 2, "deduplicated_candidates": 1},
            )
            self.assertEqual(report["validation"]["detector_candidates"], 1)
            self.assertEqual(report["validation"]["verifier_invocations"], 1)
            self.assertEqual(report["threshold_selection"]["thresholds"]["verifier"], 0.8)

    def test_no_positive_or_insufficient_exposure_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            no_positive = TraceFixture(
                root / "no-positive",
                [
                    _source("vn", "validation", "negative", []),
                    _source("tn", "test", "negative", []),
                ],
            )
            with self.assertRaisesRegex(ValueError, "no positive wake opportunities"):
                no_positive.evaluate()

        with tempfile.TemporaryDirectory() as directory:
            sources = _standard_sources()
            for detector, verifier in sources:
                if detector["truth"] == "negative":
                    detector["duration_seconds"] = verifier["duration_seconds"] = 3600.0
            fixture = TraceFixture(Path(directory), sources)
            with self.assertRaisesRegex(ValueError, "negative exposure.*below required"):
                fixture.evaluate()

    def test_validation_test_hash_or_identity_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            sources = _standard_sources()
            validation_hash = sources[0][0]["audio_sha256"]
            sources[5][0]["audio_sha256"] = validation_hash
            sources[5][1]["audio_sha256"] = validation_hash
            fixture = TraceFixture(Path(directory), sources)
            with self.assertRaisesRegex(ValueError, "groups overlap"):
                fixture.evaluate()

    def test_trace_drift_is_rejected_against_frozen_sha256(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = TraceFixture(Path(directory), _standard_sources())
            expected = fixture.detector_sha
            fixture.detector_payload["sources"][0]["events"][0]["score"] = 0.123
            fixture.detector_path.write_text(
                json.dumps(fixture.detector_payload, sort_keys=True) + "\n"
            )
            with self.assertRaisesRegex(ValueError, "trace hash drift"):
                cascade.evaluate_cascade(
                    fixture.detector_path,
                    fixture.verifier_path,
                    detector_trace_sha256=expected,
                    verifier_trace_sha256=fixture.verifier_sha,
                )

    def test_detector_verifier_candidate_or_identity_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = TraceFixture(Path(directory), _standard_sources())
            fixture.verifier_payload["sources"][0]["events"][0]["timestamp_seconds"] = 0.6
            fixture.write()
            with self.assertRaisesRegex(ValueError, "event metadata drift"):
                fixture.evaluate()

        with tempfile.TemporaryDirectory() as directory:
            fixture = TraceFixture(Path(directory), _standard_sources())
            fixture.verifier_payload["sources"][0]["speaker_id"] = "drifted-speaker"
            fixture.write()
            with self.assertRaisesRegex(ValueError, "identity metadata drift"):
                fixture.evaluate()

    def test_report_is_deterministic_and_preserves_provenance_and_exact_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = TraceFixture(Path(directory), _standard_sources())
            first = fixture.evaluate()
            second = fixture.evaluate()
            self.assertEqual(
                json.dumps(first, sort_keys=True, allow_nan=False),
                json.dumps(second, sort_keys=True, allow_nan=False),
            )
            self.assertEqual(
                first["provenance"]["detector_trace"]["sha256"], fixture.detector_sha
            )
            self.assertEqual(
                first["provenance"]["verifier_trace"]["sha256"], fixture.verifier_sha
            )
            self.assertAlmostEqual(
                first["test"]["false_accepts_per_hour_upper_95"],
                -math.log(0.05) / 100.0,
            )
            self.assertEqual(
                first["test"]["confidence"]["poisson_method"],
                "one_sided_exact_poisson_upper",
            )
            self.assertEqual(
                first["test"]["confidence"]["binomial_method"],
                "one_sided_exact_clopper_pearson_lower",
            )


if __name__ == "__main__":
    unittest.main()
