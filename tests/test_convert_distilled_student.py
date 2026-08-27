import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from microwakeword.phoneme_student import compact_phone_contract
from tools.convert_distilled_student import (
    _causal_slice,
    _decoder_decision,
    _spread_indices,
    _stream_input_chunks,
    architecture_contract,
    equivalence_report,
    load_distillation_contract,
    require_equivalence,
)
from tools.distill_kizz_phoneme_student import (
    student_decoder_contract,
    student_decoder_contract_hash,
)
from tools.distill_kizz_student import student_flags


class ConvertDistilledStudentTests(unittest.TestCase):
    def _metadata(self):
        contract = compact_phone_contract()
        return {
            "recipe": "kizz_control_compact_ctc_distillation_v6",
            "compact_phone_contract": contract,
            "architecture": architecture_contract(contract),
            "decoder": {
                "contract": student_decoder_contract(contract, "forward_sum_ctc"),
                "contract_sha256": student_decoder_contract_hash(
                    contract, "forward_sum_ctc"
                ),
            },
        }

    def test_contract_drives_output_count_and_rejects_token_drift(self):
        self.assertEqual(architecture_contract(self._metadata()["compact_phone_contract"])["output_count"], len(compact_phone_contract()["tokens"]))
        payload = self._metadata(); payload["compact_phone_contract"]["tokens"] = payload["compact_phone_contract"]["tokens"][:-1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); metadata = root / "distillation.json"; weights = root / "weights.h5"
            metadata.write_text(json.dumps(payload)); weights.write_bytes(b"weights")
            with self.assertRaisesRegex(ValueError, "contract"):
                load_distillation_contract(metadata, weights)

    def test_dilated_memory_architecture_contract_is_explicit(self):
        contract = compact_phone_contract()
        architecture = architecture_contract(contract, "dilated_temporal_memory")
        self.assertEqual(architecture["architecture_id"], "dilated_temporal_memory")
        self.assertEqual(architecture["temporal_dilations"], [1, 2, 4, 8, 16])
        self.assertEqual(architecture["warmup_output_drop"], 20)
        self.assertEqual(architecture["output_count"], len(contract["tokens"]))

    def test_exact_weight_hash_is_checked_when_declared(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); weights = root / "weights.h5"; weights.write_bytes(b"weights")
            payload = self._metadata(); payload["student"] = {"weights_sha256": "0" * 64}
            metadata = root / "distillation.json"; metadata.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "weights"):
                load_distillation_contract(metadata, weights)

    def test_exact_weight_hash_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); weights = root / "weights.h5"; weights.write_bytes(b"weights")
            metadata = root / "distillation.json"; metadata.write_text(json.dumps(self._metadata()))
            with self.assertRaisesRegex(ValueError, "bind exact student weights"):
                load_distillation_contract(metadata, weights)

    def test_equivalence_reports_numerical_and_decision_drift(self):
        offline = np.asarray([[[-2.0, 2.0], [2.0, -2.0]]], dtype=np.float32)
        report = equivalence_report(offline, offline.copy(), offline + np.asarray([[[0.0, 0.0], [-5.0, 5.0]]]))
        self.assertEqual(report["paths"]["tf_streaming"]["decision_mismatch_count"], 0)
        self.assertEqual(report["paths"]["int8_tflite"]["decision_mismatch_count"], 1)
        with self.assertRaisesRegex(ValueError, "int8_tflite"):
            require_equivalence(report, max_abs=0.1, max_mean_abs=0.1, max_decision_mismatch=0.0)

    def test_decoder_equivalence_uses_finite_zero_collision_margin(self):
        contract = compact_phone_contract()
        logits = np.zeros((54, len(contract["tokens"])), dtype=np.float32)
        self.assertIsInstance(_decoder_decision(logits, contract), bool)

    def test_causal_slice_accounts_for_two_frame_stream_phase(self):
        flags = student_flags(len(compact_phone_contract()["tokens"]))
        values = np.arange(87, dtype=np.float32)[:, None]
        selected = _causal_slice(values, flags, 66, 2)
        np.testing.assert_array_equal(selected[:, 0], np.arange(21, 87))

    def test_stream_phase_primer_preserves_the_observed_prefix(self):
        features = np.arange(7, dtype=np.float32)[:, None]
        chunks = list(_stream_input_chunks(features, stride=3, phase_offset=2))
        np.testing.assert_array_equal(chunks[0][:, 0], [0, 0, 1])
        np.testing.assert_array_equal(chunks[1][:, 0], [2, 3, 4])

    def test_calibration_sample_spans_the_entire_corpus(self):
        indices = _spread_indices(3556, 500)
        self.assertEqual((indices[0], indices[-1]), (0, 3555))
        self.assertEqual(len(indices), 500)
        self.assertGreater(len(indices[indices >= 500]), 400)


if __name__ == "__main__":
    unittest.main()
