import tempfile
import unittest
from pathlib import Path

from microwakeword.kizz_evaluation_contract import (
    group_identity_overlaps,
    require_disjoint_groups,
    sha256_file,
    validate_audio_rows,
)


class KizzEvaluationContractTests(unittest.TestCase):
    def test_audio_aliases_cross_field_names_and_fail_closed(self):
        groups = {
            "validation": [{"audio_sha256": "same"}],
            "test": [{"provenance_id": "audio-sha256:same"}],
        }
        self.assertEqual(len(group_identity_overlaps(groups)), 1)
        with self.assertRaisesRegex(
            ValueError, "qualification evidence groups overlap"
        ):
            require_disjoint_groups(groups)

    def test_partition_identity_detects_cross_split_speaker(self):
        groups = {
            "validation": [{"audio_sha256": "a", "speaker_id": "speaker-1"}],
            "heldout": [{"audio_sha256": "b", "speaker_id": "speaker-1"}],
        }
        self.assertEqual(group_identity_overlaps(groups), [])
        with self.assertRaisesRegex(ValueError, "speaker_id:speaker-1"):
            require_disjoint_groups(groups, include_partition_identity=True)

    def test_audio_contract_rejects_stale_hash_and_unlocked_anchor(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "audio.wav"
            audio.write_bytes(b"immutable evidence")
            row = {
                "source_id": "wake-1",
                "audio_sha256": sha256_file(audio),
                "path": str(audio),
                "label": 0,
                "locked_deployment_anchor": True,
                "training_eligible": False,
            }
            report = validate_audio_rows(
                [row], group="false_wake", require_locked_anchor=True
            )
            self.assertEqual(report["unique_audio_sha256"], 1)
            with self.assertRaisesRegex(ValueError, "audio hash mismatch"):
                validate_audio_rows(
                    [{**row, "audio_sha256": "0" * 64}], group="false_wake"
                )
            with self.assertRaisesRegex(ValueError, "not a locked"):
                validate_audio_rows(
                    [{**row, "locked_deployment_anchor": False}],
                    group="false_wake",
                    require_locked_anchor=True,
                )


if __name__ == "__main__":
    unittest.main()
