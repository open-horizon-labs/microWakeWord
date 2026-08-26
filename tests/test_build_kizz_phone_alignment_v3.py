import itertools
import unittest

import numpy as np

from tools.build_kizz_phone_alignment_v3 import (
    CANONICAL_TRANSCRIPT,
    COLLISION_TRANSCRIPTS,
    _crop_for_alignment,
    _inherit_overlay,
    phone_spans_from_token_spans,
    pronunciation_decision,
    select_provider_balanced,
)


def losses(canonical, collision):
    values = {CANONICAL_TRANSCRIPT: canonical * len(CANONICAL_TRANSCRIPT)}
    values.update(
        {
            transcript: collision * len(transcript)
            for transcript in COLLISION_TRANSCRIPTS
        }
    )
    return values


class BuildKizzPhoneAlignmentV3Test(unittest.TestCase):
    def test_pronunciation_gate_requires_absolute_fit_and_collision_margin(self):
        accepted = pronunciation_decision(
            losses(1.0, 1.2),
            minimum_margin_per_token=0.1,
            maximum_canonical_nll_per_token=3.5,
        )
        self.assertTrue(accepted["accepted"])

        ambiguous = pronunciation_decision(
            losses(1.0, 1.05),
            minimum_margin_per_token=0.1,
            maximum_canonical_nll_per_token=3.5,
        )
        self.assertFalse(ambiguous["accepted"])
        self.assertIn("collision_not_separated", ambiguous["reasons"])

        weak = pronunciation_decision(
            losses(4.0, 4.5),
            minimum_margin_per_token=0.1,
            maximum_canonical_nll_per_token=3.5,
        )
        self.assertFalse(weak["accepted"])
        self.assertIn("canonical_fit_too_weak", weak["reasons"])

        canonical_wins = pronunciation_decision(
            losses(1.0, 1.0001),
            minimum_margin_per_token=0.0,
            maximum_canonical_nll_per_token=3.5,
        )
        self.assertTrue(canonical_wins["accepted"])
        collision_wins = pronunciation_decision(
            losses(1.0001, 1.0),
            minimum_margin_per_token=0.0,
            maximum_canonical_nll_per_token=3.5,
        )
        self.assertFalse(collision_wins["accepted"])

    def test_phone_regions_come_from_measured_centers_and_are_contiguous(self):
        token_spans = [
            {"start": start, "end": start + 1} for start in (2, 4, 7, 11, 16, 22, 29)
        ]
        phrase, phones = phone_spans_from_token_spans(
            token_spans,
            waveform_samples=16_000,
            emission_frames=40,
            crop_offset_seconds=0.25,
        )
        self.assertEqual(
            [item["phone"] for item in phones], ["h", "aɪ", "f", "aɪ", "k", "ɪ", "z"]
        )
        self.assertEqual(phrase["start_s"], phones[0]["start_s"])
        self.assertEqual(phrase["end_s"], phones[-1]["end_s"])
        self.assertTrue(
            all(
                left["end_s"] == right["start_s"]
                for left, right in itertools.pairwise(phones)
            )
        )
        durations = [item["end_s"] - item["start_s"] for item in phones]
        self.assertGreater(durations[-1], durations[0])

    def test_device_phrase_crop_accepts_millisecond_span_metadata(self):
        samples = np.zeros(48_000, dtype="float32")
        cropped, offset = _crop_for_alignment(
            {
                "source_group": "device_replay",
                "phrase_span": {"start_ms": 700, "end_ms": 1500},
            },
            samples,
        )
        self.assertAlmostEqual(offset, 0.5)
        self.assertEqual(len(cropped), 19_200)

    def test_overlay_cannot_outlive_rejected_parent(self):
        overlay = {
            "source_id": "overlay-1",
            "path": "/overlay.wav",
            "audio_sha256": "overlay-hash",
            "split": "train",
            "source_group": "noisy_overlay",
            "render_text": "Hi-Fi Kizz",
            "duration_seconds": 1.0,
        }
        selected, audit = _inherit_overlay(overlay, None, model_sha256="model")
        self.assertIsNone(selected)
        self.assertEqual(audit["reasons"], ["parent_not_acoustically_qualified"])

    def test_qualified_overlay_inherits_exact_parent_timing(self):
        parent = {
            "source_id": "parent-1",
            "audio_sha256": "parent-hash",
            "duration_seconds": 1.0,
            "phrase_span": {"start_s": 0.1, "end_s": 0.9},
            "phone_spans": [{"phone": "h", "start_s": 0.1, "end_s": 0.2}],
            "alignment": {"pronunciation_decision": {"accepted": True}},
        }
        overlay = {
            "source_id": "overlay-1",
            "path": "/overlay.wav",
            "audio_sha256": "overlay-hash",
            "split": "train",
            "source_group": "noisy_overlay",
            "render_text": "Hi-Fi Kizz",
            "duration_seconds": 1.0,
        }
        selected, audit = _inherit_overlay(overlay, parent, model_sha256="model")
        self.assertTrue(audit["accepted"])
        self.assertEqual(selected["phone_spans"], parent["phone_spans"])
        self.assertEqual(
            selected["alignment"]["method"], "inherited_ctc_forced_alignment"
        )

    def test_post_alignment_balance_excludes_failed_and_dominant_providers(self):
        rows = []
        counts = {"a": 8, "b": 4, "c": 4, "d": 4, "bad": 20}
        for split in ("train", "validation", "test"):
            for provider, count in counts.items():
                for index in range(count):
                    rows.append(
                        {
                            "source_id": f"{split}:{provider}:{index}",
                            "source_group": provider,
                            "provider": provider,
                            "split": split,
                        }
                    )
        selected, report = select_provider_balanced(
            rows,
            required_providers=("a", "b", "c", "d"),
            maximum_provider_share=0.35,
            seed=1,
        )
        self.assertTrue(report["qualified"])
        self.assertNotIn("bad", {row["provider"] for row in selected})
        for split in report["splits"].values():
            self.assertLessEqual(max(split["selected_shares"].values()), 0.35)

    def test_post_alignment_balance_fails_when_provider_disappears(self):
        rows = [
            {
                "source_id": f"train:{provider}",
                "source_group": provider,
                "provider": provider,
                "split": "train",
            }
            for provider in ("a", "b", "c")
        ]
        _, report = select_provider_balanced(
            rows,
            required_providers=("a", "b", "c", "d"),
            maximum_provider_share=0.35,
            seed=1,
        )
        self.assertFalse(report["qualified"])
        self.assertEqual(
            report["violations"][0]["reason"],
            "required_provider_missing_after_acoustic_gate",
        )


if __name__ == "__main__":
    unittest.main()
