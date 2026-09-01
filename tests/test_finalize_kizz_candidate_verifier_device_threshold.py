import unittest

from tools.finalize_kizz_candidate_verifier_device_threshold import (
    validate_reserved_evidence,
)


class FinalizeKizzCandidateVerifierDeviceThresholdTests(unittest.TestCase):
    def test_reserved_evidence_requires_exact_test_only_coverage(self):
        captures = [{"sha256": "a" * 64}, {"sha256": "b" * 64}]
        evidence = {
            "kind": "kizz_control_voice_stratified_device_replay_qualification_evidence",
            "training_eligible": False,
            "examples": [
                {"audio_sha256": "a" * 64, "split": "test", "training_eligible": False},
                {"audio_sha256": "b" * 64, "split": "test", "training_eligible": False},
            ],
        }
        validate_reserved_evidence(captures, evidence)
        evidence["examples"][1]["audio_sha256"] = "c" * 64
        with self.assertRaises(ValueError):
            validate_reserved_evidence(captures, evidence)


if __name__ == "__main__":
    unittest.main()
