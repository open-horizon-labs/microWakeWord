import json
import tempfile
import unittest
from pathlib import Path

from microwakeword.kizz_continuous_evaluation import poisson_upper_95
from microwakeword.kizz_phoneme_teacher import sha256_file
from tools.rebind_kizz_phoneme_teacher_operating_point import (
    rebind_clip_report,
    rebind_continuous_report,
)


def scored(score, *, split="validation", label=1):
    return {
        "score": score,
        "collision_margin": 1.0,
        "accepted": True,
        "failure_reasons": [],
        "split": split,
        "label": label,
    }


class RebindTeacherOperatingPointTests(unittest.TestCase):
    def fixtures(self, root: Path, *, bound_threshold=-0.25):
        weights_hash = "a" * 64
        adaptation = root / "training.json"
        adaptation.write_text(
            json.dumps(
                {
                    "kind": "kizz_phoneme_teacher_adaptation",
                    "wake_phrase": {"phrase_id": "kizz-control"},
                    "checkpoint_selection": {"selected_step": 10},
                    "checkpoints": {"best": {"file_sha256": weights_hash}},
                    "validation_ledger": [
                        {
                            "step": 10,
                            "detector_selection": {
                                "checkpoint": {"file_sha256": weights_hash},
                                "metrics": {
                                    "threshold": bound_threshold,
                                    "qualified_clean_operating_point": True,
                                },
                            },
                        }
                    ],
                }
            )
        )
        false_wakes = [
            {
                **scored(-0.5, split="test", label=0),
                "wake_context": {"best_window_is_pre_wake": True},
            }
            for _ in range(62)
        ]
        clip = root / "clip.json"
        clip.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "gate_scope": "teacher_clip_and_anchor_prequalification",
                    "qualified": True,
                    "model": {"revision": weights_hash, "weights_sha256": weights_hash},
                    "limits": {"min_recall": 0.9, "max_faph": 0.1, "minimum_natural_positives": 1},
                    "scoring": {"threshold": -1.0, "collision_margin_beta": 0.0},
                    "counts": {"validation_negative_exposure_seconds": 3600.0},
                    "results": {
                        "aligned": [
                            scored(0.0),
                            scored(-0.1),
                            scored(0.0, split="test"),
                        ],
                        "validation_negative": [scored(-0.5, label=0)],
                        "natural_positive": [scored(-0.2, split="test")],
                        "false_wake_anchors": false_wakes,
                    },
                }
            )
        )
        continuous = root / "continuous.json"
        continuous.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "gate_scope": "untouched_continuous_qualification",
                    "qualified": False,
                    "failure_reasons": ["old"],
                    "model": {"weights_sha256": weights_hash},
                    "teacher_qualification": {"report_sha256": sha256_file(clip)},
                    "scoring": {"threshold": -1.0},
                    "limits": {"min_exposure_hours": 100.0, "max_faph_upper_95": 0.1},
                    "counts": {"exposure_hours": 100.0, "false_accepts": 1},
                    "categories": {"speech": {"events": 1}},
                    "members": [
                        {"category": "speech", "events": [{"peak_score": -0.5}]}
                    ],
                }
            )
        )
        return clip, continuous, adaptation

    def test_tightening_uses_adaptation_threshold_and_filters_stored_events(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clip, continuous, adaptation = self.fixtures(root)
            rebound = rebind_clip_report(clip, adaptation, min_natural_recall=0.875)
            self.assertTrue(rebound["qualified"])
            self.assertEqual(rebound["scoring"]["threshold"], -0.25)
            self.assertEqual(rebound["counts"]["false_wake_accepted"], 0)
            rebound_path = root / "rebound.json"
            rebound_path.write_text(json.dumps(rebound))
            result = rebind_continuous_report(
                continuous, clip, rebound_path, rebound
            )
            self.assertTrue(result["qualified"])
            self.assertEqual(result["counts"]["false_accepts"], 0)
            self.assertEqual(
                result["counts"]["faph_upper_95"], poisson_upper_95(0, 100.0)
            )

    def test_rejects_threshold_loosening(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clip, _, adaptation = self.fixtures(root, bound_threshold=-1.5)
            with self.assertRaisesRegex(ValueError, "only tighten"):
                rebind_clip_report(clip, adaptation, min_natural_recall=0.875)


if __name__ == "__main__":
    unittest.main()
