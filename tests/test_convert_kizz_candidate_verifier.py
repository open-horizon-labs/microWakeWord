import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.test_train_kizz_candidate_verifier import (
    CandidateDatasetFixture,
    FakeBackend as TrainingFakeBackend,
)
from tools.convert_kizz_candidate_verifier import (
    ARTIFACT_FILENAME,
    EXPECTED_OP_COUNTS,
    METADATA_FILENAME,
    ConversionRuntime,
    TensorSpec,
    _apply_logit_bound,
    _rebuild_bound_training_model,
    convert,
    sha256_file,
    validate_inputs,
)
from tools.train_kizz_candidate_verifier import train_candidate_verifier


class LogitBoundTests(unittest.TestCase):
    def test_deployment_logit_bound_is_monotonic_symmetric_and_bounded(self):
        values = np.asarray([-100.0, -5.0, 0.0, 5.0, 100.0])
        transformed = _apply_logit_bound(values, 4.0)
        self.assertTrue(np.all(np.diff(transformed) > 0.0))
        self.assertTrue(np.all(np.abs(transformed) <= 4.0))
        self.assertAlmostEqual(float(transformed[2]), 0.0)
        self.assertAlmostEqual(float(transformed[1]), -float(transformed[3]))
        np.testing.assert_array_equal(_apply_logit_bound(values, None), values)


class FakeConversionBackend:
    def __init__(
        self,
        *,
        input_shape=(1, 260, 40, 1),
        output_shape=(1, 1),
        input_dtype="int8",
        output_dtype="int8",
        int8_delta=0.0,
        dynamic_tensors=0,
        variable_tensors=0,
        float_tensors=0,
        tensor_dtypes=None,
        parameter_delta=0,
        model_hash_drift=False,
        op_counts=None,
        artifact=b"\x1c\x00\x00\x00TFL3-fake-fixed-window-verifier",
    ):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        self.int8_delta = float(int8_delta)
        self.dynamic_tensors = dynamic_tensors
        self.variable_tensors = variable_tensors
        self.float_tensors = float_tensors
        self.tensor_dtypes = dict(tensor_dtypes or {"int8": 20, "int32": 11})
        self.parameter_delta = parameter_delta
        self.model_hash_drift = model_hash_drift
        self.op_counts = dict(op_counts or EXPECTED_OP_COUNTS)
        self.artifact = artifact
        self.calibration_markers = []
        self.float_markers = []
        self.int8_markers = []

    def convert(self, *, validated, calibration_features, work_dir):
        del work_dir
        self.calibration_markers = [
            (float(sample[0, 0, 0]), float(sample[0, 1, 0]))
            for sample in calibration_features
        ]

        validation_indices = [
            index
            for index, row in enumerate(validated.dataset.rows)
            if row["split"] == "validation"
        ]
        validation_features = np.asarray(
            validated.dataset.features[validation_indices], dtype=np.float32
        )[..., None]
        validation_logits = np.load(validated.validation_logits.path)
        reference_by_marker = {
            (float(sample[0, 0, 0]), float(sample[0, 1, 0])): float(logit)
            for sample, logit in zip(validation_features, validation_logits, strict=True)
        }

        def score(sample):
            marker = (float(sample[0, 0, 0]), float(sample[0, 1, 0]))
            return reference_by_marker[marker]

        def run_float(sample):
            self.float_markers.append(
                (float(sample[0, 0, 0]), float(sample[0, 1, 0]))
            )
            return score(sample)

        def run_int8(sample):
            self.int8_markers.append(
                (float(sample[0, 0, 0]), float(sample[0, 1, 0]))
            )
            return score(sample) + self.int8_delta

        operators = [
            name for name, count in sorted(self.op_counts.items()) for _ in range(count)
        ]
        return ConversionRuntime(
            artifact=self.artifact,
            input_spec=TensorSpec(self.input_shape, self.input_dtype, (0.125, -3)),
            output_spec=TensorSpec(self.output_shape, self.output_dtype, (0.0625, 1)),
            run_float=run_float,
            run_int8=run_int8,
            tensor_audit={
                "tensor_count": 31,
                "declared_tensor_bytes_sum": 65536,
                "dynamic_shape_tensor_count": self.dynamic_tensors,
                "variable_tensor_count": self.variable_tensors,
                "float_tensor_count": self.float_tensors,
                "tensor_dtypes": self.tensor_dtypes,
                "input_count": 1,
                "output_count": 1,
            },
            operator_audit={"operators": operators, "counts": self.op_counts},
            parameter_count=validated.expected_parameter_count + self.parameter_delta,
            model_json_sha256=(
                "0" * 64
                if self.model_hash_drift
                else validated.model_topology_sha256
            ),
            framework={"fake": "1"},
        )


class ConvertKizzCandidateVerifierTests(unittest.TestCase):
    def fixture(self, root: Path, *, model_variant="compact"):
        candidate = CandidateDatasetFixture(root)
        output = root / "training"
        train_candidate_verifier(
            candidate.dataset,
            output,
            expected_corpus_sha256=sha256_file(candidate.corpus_path),
            steps=2,
            batch_size=4,
            eval_every=1,
            conditional_recall_floor=0.5,
            model_variant=model_variant,
            backend=TrainingFakeBackend(),
        )
        report = output / "training-report.json"
        weights = output / "best.weights.h5"
        return candidate, argparse.Namespace(
            training_report=report,
            weights=weights,
            output=root / "package",
            calibration_examples=99,
            equivalence_examples=99,
            max_logit_absolute_error=0.75,
            mean_logit_absolute_error=0.15,
            max_probability_absolute_error=0.05,
            threshold_decision_mismatch_fraction=0.0,
        )

    @staticmethod
    def rewrite_report(path: Path, mutation):
        report = json.loads(path.read_text(encoding="utf-8"))
        mutation(report)
        path.write_text(json.dumps(report), encoding="utf-8")

    def test_emits_bound_nonqualified_firmware_package(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate, args = self.fixture(Path(directory))
            backend = FakeConversionBackend()
            report = convert(args, backend=backend)
            artifact = args.output / ARTIFACT_FILENAME
            metadata = args.output / METADATA_FILENAME

            self.assertEqual(report["model_role"], "detector_conditioned_candidate_verifier")
            self.assertFalse(report["deployment_qualification"])
            self.assertFalse(report["cascade_qualification"])
            self.assertEqual(report["model"]["parameter_count"], 15793)
            self.assertEqual(report["model"]["mac_count"], 2801952)
            self.assertEqual(report["model"]["variant"], "compact")
            self.assertEqual(report["tensor_contracts"]["input"]["shape"], [1, 260, 40, 1])
            self.assertEqual(report["tensor_contracts"]["output"]["shape"], [1, 1])
            self.assertEqual(report["tensor_contracts"]["input"]["dtype"], "int8")
            self.assertEqual(report["tensor_contracts"]["output"]["dtype"], "int8")
            self.assertEqual(
                report["artifact"]["sha256"], hashlib.sha256(artifact.read_bytes()).hexdigest()
            )
            self.assertEqual(
                report["inputs"]["candidate_corpus"]["sha256"],
                sha256_file(candidate.corpus_path),
            )
            self.assertTrue(
                report["static_memory_audit"]["hardware_high_water_measurement_required"]
            )
            self.assertEqual(json.loads(metadata.read_text()), report)

    def test_wide_provenance_validates_rebuilds_and_packages_as_wide(self):
        with tempfile.TemporaryDirectory() as directory:
            _, args = self.fixture(Path(directory), model_variant="wide")
            validated = validate_inputs(args.training_report, args.weights)
            self.assertEqual(validated.model_variant, "wide")
            self.assertEqual(validated.expected_parameter_count, 24081)
            self.assertEqual(validated.expected_mac_count, 4310512)

            rebuild_backend = TrainingFakeBackend()
            _, rebuilt, _ = _rebuild_bound_training_model(
                validated, builder=rebuild_backend
            )
            self.assertEqual(rebuild_backend.model_variant, "wide")
            self.assertEqual(rebuilt.model_variant, "wide")

            report = convert(args, backend=FakeConversionBackend())
            self.assertEqual(report["model"]["variant"], "wide")
            self.assertEqual(report["model"]["channel_plan"], [32, 48, 64, 80, 112])
            self.assertEqual(report["model"]["parameter_count"], 24081)
            self.assertEqual(report["model"]["mac_count"], 4310512)

    def test_converter_rejects_missing_or_cross_variant_provenance(self):
        cases = (
            (
                lambda report: report["architecture"].pop("variant"),
                "model variant",
            ),
            (
                lambda report: report["architecture"].__setitem__(
                    "variant", "wide"
                ),
                "channel plan",
            ),
            (
                lambda report: report["architecture"].__setitem__(
                    "dscnn_spec", report["architecture"]["dscnn_spec"][:-1]
                ),
                "specification",
            ),
        )
        for mutation, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                _, args = self.fixture(Path(directory))
                self.rewrite_report(args.training_report, mutation)
                with self.assertRaisesRegex(ValueError, message):
                    validate_inputs(args.training_report, args.weights)

    def test_converter_accepts_exact_legacy_compact_architecture_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            _, args = self.fixture(Path(directory))

            def make_legacy(report):
                architecture = report["architecture"]
                architecture.pop("variant")
                architecture.pop("channel_plan")
                architecture.pop("dscnn_spec")
                architecture["input_shape"] = [260, 40]

            self.rewrite_report(args.training_report, make_legacy)
            validated = validate_inputs(args.training_report, args.weights)
            self.assertEqual(validated.model_variant, "compact")
            self.assertEqual(validated.expected_parameter_count, 15793)
            self.assertEqual(validated.expected_mac_count, 2801952)

    def test_calibration_is_train_only_and_equivalence_is_validation_only(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate, args = self.fixture(Path(directory))
            backend = FakeConversionBackend()
            report = convert(args, backend=backend)
            train_ids = {
                row["candidate_id"] for row in candidate.rows if row["split"] == "train"
            }
            validation_ids = {
                row["candidate_id"]
                for row in candidate.rows
                if row["split"] == "validation"
            }
            # CandidateDatasetFixture stores the feature file on ``features``.
            all_features = np.load(candidate.features)
            test_markers = {
                (float(all_features[index, 0, 0]), float(all_features[index, 0, 1]))
                for index, row in enumerate(candidate.rows)
                if row["split"] == "test"
            }
            self.assertEqual(set(report["calibration"]["candidate_ids"]), train_ids)
            self.assertEqual(set(report["equivalence"]["candidate_ids"]), validation_ids)
            self.assertEqual(report["equivalence"]["test_examples_scored"], 0)
            self.assertTrue(set(backend.calibration_markers).isdisjoint(test_markers))
            self.assertTrue(set(backend.float_markers).isdisjoint(test_markers))
            self.assertTrue(set(backend.int8_markers).isdisjoint(test_markers))

    def test_rejects_training_role_selection_and_architecture_drift(self):
        cases = (
            (lambda report: report.__setitem__("candidate_conditioned", False), "detector-conditioned"),
            (lambda report: report.__setitem__("deployment_qualification", True), "non-deployment"),
            (
                lambda report: report["selection_contract"].__setitem__(
                    "selection_split", "test"
                ),
                "validation-only",
            ),
            (
                lambda report: report["architecture"].__setitem__(
                    "parameter_count", 1
                ),
                "parameter count",
            ),
            (
                lambda report: report["architecture"].__setitem__("mac_estimate", 1),
                "mac_estimate",
            ),
            (
                lambda report: report["architecture"]["layers"][0].__setitem__(
                    "filters", 25
                ),
                "layers",
            ),
        )
        for mutation, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                _, args = self.fixture(Path(directory))
                self.rewrite_report(args.training_report, mutation)
                with self.assertRaisesRegex(ValueError, message):
                    validate_inputs(args.training_report, args.weights)
                self.assertFalse(args.output.exists())

    def test_rejects_weight_corpus_array_transitive_and_model_binding_drift(self):
        mutations = (
            ("weights", lambda candidate, args: args.weights.write_bytes(b"tampered")),
            ("corpus", lambda candidate, args: candidate.corpus_path.write_text("{}")),
            ("array", lambda candidate, args: candidate.features.write_bytes(b"tampered")),
            (
                "transitive",
                lambda candidate, args: candidate.bound_files["artifact"].write_bytes(b"tampered"),
            ),
            (
                "model",
                lambda candidate, args: (
                    Path(json.loads(args.training_report.read_text())["output_bindings"]["model_architecture"]["path"])
                    .write_text("tampered")
                ),
            ),
        )
        for label, mutation in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                candidate, args = self.fixture(Path(directory))
                mutation(candidate, args)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    validate_inputs(args.training_report, args.weights)
                self.assertFalse(args.output.exists())

    def test_rejects_rebuilt_model_hash_and_parameter_drift(self):
        for backend, message in (
            (FakeConversionBackend(model_hash_drift=True), "model topology hash"),
            (FakeConversionBackend(parameter_delta=1), "parameter count"),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                _, args = self.fixture(Path(directory))
                with self.assertRaisesRegex(ValueError, message):
                    convert(args, backend=backend)
                self.assertFalse(args.output.exists())

    def test_rejects_tensor_quantization_static_memory_and_operator_drift(self):
        bad_ops = dict(EXPECTED_OP_COUNTS)
        bad_ops["CONV_2D"] -= 1
        unsupported_ops = dict(EXPECTED_OP_COUNTS)
        unsupported_ops["CUSTOM"] = 1
        cases = (
            (FakeConversionBackend(input_shape=(1, 259, 40, 1)), "input shape"),
            (FakeConversionBackend(output_shape=(1, 2)), "output shape"),
            (FakeConversionBackend(input_dtype="float32"), "input must be int8"),
            (FakeConversionBackend(output_dtype="uint8"), "output must be int8"),
            (FakeConversionBackend(dynamic_tensors=1), "dynamic tensor"),
            (FakeConversionBackend(variable_tensors=1), "variable tensors"),
            (FakeConversionBackend(float_tensors=1), "floating-point tensors"),
            (
                FakeConversionBackend(tensor_dtypes={"int8": 20, "int64": 1}),
                "unsupported tensor dtypes",
            ),
            (FakeConversionBackend(op_counts=bad_ops), "CONV_2D operator count"),
            (FakeConversionBackend(op_counts=unsupported_ops), "unsupported operators"),
        )
        for backend, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                _, args = self.fixture(Path(directory))
                with self.assertRaisesRegex(ValueError, message):
                    convert(args, backend=backend)
                self.assertFalse(args.output.exists())

    def test_rejects_int8_equivalence_drift_without_retuning(self):
        with tempfile.TemporaryDirectory() as directory:
            _, args = self.fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "equivalence failed"):
                convert(args, backend=FakeConversionBackend(int8_delta=2.0))
            self.assertFalse(args.output.exists())

        with tempfile.TemporaryDirectory() as directory:
            _, args = self.fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "not a TFLite"):
                convert(args, backend=FakeConversionBackend(artifact=b"not-flatbuffer"))
            self.assertFalse(args.output.exists())

    def test_conversion_metadata_is_deterministic_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, args = self.fixture(root)
            first = convert(args, backend=FakeConversionBackend())
            args.output = root / "second-package"
            second = convert(args, backend=FakeConversionBackend())
            self.assertEqual(first, second)
            args.output = root / "package"
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                convert(args, backend=FakeConversionBackend())


if __name__ == "__main__":
    unittest.main()
