import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.build_ordered_state_frame_supervision import build_frame_supervision

PHONES = ("h", "aɪ", "f", "aɪ", "k", "ɪ", "z")


def record(root, *, source_id="clip", group="session", split="train", **changes):
    features_path = root / f"{source_id}.npy"
    np.save(features_path, np.arange(12 * 40, dtype=np.float32).reshape(12, 40))
    value = {
        "source_id": source_id,
        "source_group": group,
        "split": split,
        "truth": True,
        "duration_s": 0.72,
        "features_path": features_path.name,
        "feature_frame_step_seconds": 0.01,
        "target_frame_times_s": [0.04, 0.07, 0.10, 0.13],
        "alignment": {
            "method": "ctc_forced_alignment",
            "timing_source": "reviewed-alignment.json",
            "pronunciation_decision": {"accepted": True},
        },
        "phrase_span": {"start_s": 0.03, "end_s": 0.66},
        "phone_spans": [
            {"phone": phone, "start_s": 0.03 + i * 0.09, "end_s": 0.12 + i * 0.09}
            for i, phone in enumerate(PHONES)
        ],
    }
    value.update(changes)
    return value


class BuildOrderedStateFrameSupervisionTest(unittest.TestCase):
    def test_rejects_empty_records(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            self.assertRaisesRegex(ValueError, "is empty"),
        ):
            build_frame_supervision([], Path(temporary), Path(temporary) / "out")

    def test_writes_trainer_compatible_arrays_from_measured_phone_spans(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "aligned"
            summary = build_frame_supervision(
                [record(root)],
                root,
                output,
                expected_feature_frames=12,
                expected_target_frames=4,
            )
            self.assertEqual(summary["feature_shape"], [1, 12, 40])
            np.testing.assert_array_equal(
                np.load(output / "targets.npy"), [[2, 3, 4, 5]]
            )
            self.assertEqual(np.load(output / "weights.npy").tolist(), [1.0])

    def test_rejects_unqualified_ctc_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = record(root)
            value["alignment"] = {
                "method": "ctc_forced_alignment",
                "timing_source": "alignment.json",
                "pronunciation_decision": {"accepted": False},
            }
            with self.assertRaisesRegex(ValueError, "acoustically qualified"):
                build_frame_supervision(
                    [value], root, root / "out", expected_target_frames=4
                )

    def test_rejects_missing_phone_alignment_and_wrong_sequence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = record(root)
            value.pop("phone_spans")
            with self.assertRaisesRegex(ValueError, "phone spans"):
                build_frame_supervision(
                    [value], root, root / "out", expected_target_frames=4
                )
            value = record(root)
            value["phone_spans"][2]["phone"] = "v"
            with self.assertRaisesRegex(ValueError, "exactly match canonical"):
                build_frame_supervision(
                    [value], root, root / "out", expected_target_frames=4
                )

    def test_rejects_nonfinite_bounds_overlaps_and_wrong_cadence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = record(root)
            value["phone_spans"][0]["end_s"] = float("nan")
            with self.assertRaisesRegex(ValueError, "span must"):
                build_frame_supervision(
                    [value], root, root / "out", expected_target_frames=4
                )
            value = record(root)
            value["phone_spans"][1]["start_s"] = 0.04
            with self.assertRaisesRegex(ValueError, "non-overlapping"):
                build_frame_supervision(
                    [value], root, root / "out", expected_target_frames=4
                )
            value = record(root)
            value["target_frame_times_s"][2] = 0.11
            with self.assertRaisesRegex(ValueError, "wrong cadence"):
                build_frame_supervision(
                    [value], root, root / "out", expected_target_frames=4
                )

    def test_rejects_split_leakage_and_shape_or_nonfinite_features(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = record(root, source_id="one", group="same-session", split="train")
            second = record(
                root, source_id="two", group="same-session", split="validation"
            )
            with self.assertRaisesRegex(ValueError, "leaks across splits"):
                build_frame_supervision(
                    [first, second], root, root / "out", expected_target_frames=4
                )
            bad = record(root, source_id="bad")
            np.save(root / "bad.npy", np.full((12, 39), np.nan, dtype=np.float32))
            with self.assertRaisesRegex(ValueError, r"shape \[time, 40\]"):
                build_frame_supervision(
                    [bad], root, root / "out", expected_target_frames=4
                )

    def test_accepts_jsonl_and_inherited_ctc_timing_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = record(root)
            value["alignment"] = {
                "method": "inherited_ctc_forced_alignment",
                "timing_source": "parent-source-id",
                "pronunciation_decision": {"accepted": True},
            }
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps(value) + "\n", encoding="utf-8")
            from tools.build_ordered_state_frame_supervision import _load_manifest

            summary = build_frame_supervision(
                _load_manifest(manifest), root, root / "out", expected_target_frames=4
            )
            self.assertEqual(summary["examples"], 1)

    def test_one_state_per_phone_emits_nine_state_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = build_frame_supervision(
                [record(root)],
                root,
                root / "out",
                expected_target_frames=4,
                states_per_phone=1,
            )
            np.testing.assert_array_equal(
                np.load(root / "out" / "targets.npy"), [[2, 2, 2, 3]]
            )
            self.assertEqual(summary["state_count"], 9)

    def test_rejects_non_ctc_alignment_method(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = record(root)
            value["alignment"]["method"] = "synthesizer"
            with self.assertRaisesRegex(ValueError, "ctc_forced_alignment"):
                build_frame_supervision([value], root, root / "out")

    def test_measured_synthesizer_escape_hatch_is_explicit_and_hash_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = record(root)
            value["alignment"] = {
                "method": "synthesizer",
                "timing_source": "piper-manifest.json",
                "timing_record": {"measured_token_samples": True},
            }
            with self.assertRaisesRegex(ValueError, "source audio hash"):
                build_frame_supervision(
                    [value],
                    root,
                    root / "out",
                    allow_measured_synthesizer_timing=True,
                )
            value["alignment"]["timing_record"]["source_wav_sha256"] = "a" * 64
            summary = build_frame_supervision(
                [value],
                root,
                root / "out",
                allow_measured_synthesizer_timing=True,
            )
            self.assertEqual(summary["alignment_policy"], "ctc_or_measured_synthesizer")

    def test_rejects_model_shape_and_feature_cadence_mismatches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = record(root)
            with self.assertRaisesRegex(ValueError, "feature frame count"):
                build_frame_supervision(
                    [value],
                    root,
                    root / "out",
                    expected_feature_frames=11,
                    expected_target_frames=4,
                )
            with self.assertRaisesRegex(ValueError, "incompatible with the model"):
                build_frame_supervision(
                    [value],
                    root,
                    root / "out",
                    expected_feature_frames=12,
                    expected_target_frames=5,
                )
            value = record(root)
            value["feature_frame_step_seconds"] = 0.02
            with self.assertRaisesRegex(ValueError, "feature frame cadence"):
                build_frame_supervision(
                    [value], root, root / "out", expected_target_frames=4
                )


if __name__ == "__main__":
    unittest.main()
