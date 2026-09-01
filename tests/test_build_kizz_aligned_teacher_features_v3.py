import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from microwakeword.ordered_state import KIZZ_PHONES
from tools.build_kizz_aligned_teacher_features_v3 import (
    CONTEXT_SAMPLES,
    apply_room_impulse_response,
    build,
    mix_at_snr,
    place_phrase_context,
    validate_aligned_positive,
)


def aligned_row(path: Path, source_id: str, split: str) -> dict:
    phones = []
    boundaries = np.linspace(0.4, 1.1, 8)
    for phone, start, end in zip(KIZZ_PHONES, boundaries[:-1], boundaries[1:]):
        phones.append({"phone": phone, "start_s": float(start), "end_s": float(end)})
    return {
        "path": str(path),
        "label": 1,
        "training_eligible": True,
        "semantic_label": "canonical_exact",
        "source_id": source_id,
        "source_group": "test_synthesis",
        "provider": "test-provider",
        "split": split,
        "target_phones": list(KIZZ_PHONES),
        "phrase_span": {"start_s": 0.4, "end_s": 1.1},
        "phone_spans": phones,
        "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "alignment": {
            "method": "ctc_forced_alignment",
            "timing_source": "pinned/model.pt",
            "pronunciation_decision": {"accepted": True},
        },
    }


class BuildKizzAlignedTeacherFeaturesV3Test(unittest.TestCase):
    def test_phrase_context_translates_spans_for_pad_and_crop(self):
        short = np.zeros(16_000, dtype=np.float32)
        context, shift = place_phrase_context(
            short, (0.2, 0.8), desired_phrase_center_s=1.4
        )
        self.assertEqual(context.shape, (CONTEXT_SAMPLES,))
        self.assertAlmostEqual((0.2 + shift + 0.8 + shift) / 2, 1.4, places=4)
        long = np.zeros(80_000, dtype=np.float32)
        context, shift = place_phrase_context(
            long, (2.0, 2.8), desired_phrase_center_s=1.0
        )
        self.assertEqual(context.shape, (CONTEXT_SAMPLES,))
        self.assertAlmostEqual((2.0 + shift + 2.8 + shift) / 2, 1.0, places=4)

    def test_alignment_gate_rejects_unqualified_or_locked_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.wav"
            sf.write(path, np.zeros(16_000, dtype=np.float32), 16_000)
            row = aligned_row(path, "source", "train")
            validate_aligned_positive(row)
            row["alignment"]["method"] = "wav2vec2_ipa_ctc_forced_alignment"
            validate_aligned_positive(row)
            row["alignment"]["pronunciation_decision"]["accepted"] = False
            with self.assertRaisesRegex(ValueError, "failed acoustic"):
                validate_aligned_positive(row)
            row["alignment"]["pronunciation_decision"]["accepted"] = True
            row["locked_deployment_anchor"] = True
            with self.assertRaisesRegex(ValueError, "locked"):
                validate_aligned_positive(row)

    def test_build_keeps_eval_clean_and_writes_nine_state_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "positive.wav"
            sf.write(
                audio,
                np.sin(np.linspace(0, 200, 24_000)).astype(np.float32) * 0.1,
                16_000,
            )
            rows = [
                aligned_row(audio, f"source-{split}", split)
                for split in ("train", "validation", "test")
            ]
            manifest = root / "positives.json"
            manifest.write_text(json.dumps({"examples": rows}))
            background = root / "background.wav"
            sf.write(background, np.ones(8_000, dtype=np.float32) * 0.01, 16_000)
            background_manifest = root / "backgrounds.json"
            background_manifest.write_text(
                json.dumps(
                    {
                        "examples": [
                            {
                                "path": str(background),
                                "label": 0,
                                "split": "train",
                                "training_eligible": True,
                                "locked_deployment_anchor": False,
                                "source_group": "music",
                                "source_id": "music-1",
                                "audio_sha256": hashlib.sha256(
                                    background.read_bytes()
                                ).hexdigest(),
                            }
                        ]
                    }
                )
            )
            fake_features = np.zeros((260, 40), dtype=np.float32)
            with mock.patch(
                "tools.build_kizz_aligned_teacher_features_v3.frontend",
                return_value=fake_features,
            ):
                report = build(
                    [manifest],
                    root / "out",
                    background_manifest=background_manifest,
                    overlay_snr_db=(10.0,),
                    seed=7,
                )
            self.assertEqual(
                report["positive_counts"], {"train": 2, "validation": 1, "test": 1}
            )
            train_targets = np.load(root / "out" / "positive_targets-train.npy")
            self.assertEqual(train_targets.shape, (2, 87))
            self.assertGreaterEqual(int(train_targets.min()), 1)
            self.assertLess(int(train_targets.max()), 9)
            variants = [row["variant"] for row in report["examples"]]
            self.assertEqual(variants.count("overlay-0"), 1)
            self.assertTrue(
                all(
                    row["source_group"] == "test_synthesis"
                    for row in report["examples"]
                )
            )
            self.assertTrue(
                all(row["provider"] == "test-provider" for row in report["examples"])
            )
            overlay = next(
                row for row in report["examples"] if row["variant"] == "overlay-0"
            )
            self.assertEqual(overlay["augmentation"]["background_source_group"], "music")
            self.assertIn("background_crop_start_sample", overlay["augmentation"])
            self.assertIn("foreground_gain_db", overlay["augmentation"])
            self.assertIsNone(overlay["augmentation"]["rir_source_id"])

    def test_train_overlay_uses_speech_rir_gain_and_keeps_eval_clean(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            positive = root / "positive.wav"
            sf.write(
                positive,
                np.sin(np.linspace(0, 200, 24_000)).astype(np.float32) * 0.1,
                16_000,
            )
            positives = root / "positives.json"
            positives.write_text(
                json.dumps(
                    {
                        "examples": [
                            aligned_row(positive, f"source-{split}", split)
                            for split in ("train", "validation", "test")
                        ]
                    }
                )
            )
            speech = root / "speech.wav"
            sf.write(speech, np.ones(60_000, dtype=np.float32) * 0.01, 16_000)
            backgrounds = root / "backgrounds.json"
            backgrounds.write_text(
                json.dumps(
                    {
                        "examples": [
                            {
                                "path": str(speech),
                                "label": 0,
                                "split": "train",
                                "training_eligible": True,
                                "locked_deployment_anchor": False,
                                "source_group": "public_speech",
                                "source_id": "speech-1",
                                "audio_sha256": hashlib.sha256(
                                    speech.read_bytes()
                                ).hexdigest(),
                            }
                        ]
                    }
                )
            )
            rir = root / "rir.wav"
            impulse = np.zeros(2_000, dtype=np.float32)
            impulse[120] = 1.0
            impulse[400] = 0.3
            sf.write(rir, impulse, 16_000, subtype="FLOAT")
            rirs = root / "rirs.json"
            rirs.write_text(
                json.dumps(
                    {
                        "examples": [
                            {
                                "path": str(rir),
                                "split": "train",
                                "training_eligible": True,
                                "locked_deployment_anchor": False,
                                "source_id": "rir-1",
                                "stratum": "real",
                                "audio_sha256": hashlib.sha256(
                                    rir.read_bytes()
                                ).hexdigest(),
                            }
                        ]
                    }
                )
            )
            fake_features = np.zeros((260, 40), dtype=np.float32)
            with mock.patch(
                "tools.build_kizz_aligned_teacher_features_v3.frontend",
                return_value=fake_features,
            ):
                report = build(
                    [positives],
                    root / "out",
                    background_manifest=backgrounds,
                    rir_manifest=rirs,
                    overlay_snr_db=(7.0,),
                    gain_db_range=(-4.0, -4.0),
                    seed=13,
                )
            self.assertEqual(
                report["positive_counts"], {"train": 2, "validation": 1, "test": 1}
            )
            overlay = next(row for row in report["examples"] if row["variant"] == "overlay-0")
            augmentation = overlay["augmentation"]
            self.assertEqual(augmentation["background_source_group"], "public_speech")
            self.assertEqual(augmentation["rir_source_id"], "rir-1")
            self.assertEqual(augmentation["rir_stratum"], "real")
            self.assertEqual(augmentation["rir_arrival_trim_samples"], 120)
            self.assertEqual(augmentation["foreground_gain_db"], -4.0)
            self.assertEqual(
                [row["variant"] for row in report["examples"] if row["split"] != "train"],
                ["clean", "clean"],
            )
            self.assertEqual(report["rir_manifest"]["eligible_count"], 1)

    def test_rejects_background_hash_drift_and_locked_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            positive = root / "positive.wav"
            sf.write(positive, np.zeros(24_000, dtype=np.float32), 16_000)
            positives = root / "positives.json"
            positives.write_text(
                json.dumps(
                    {
                        "examples": [
                            aligned_row(positive, f"source-{split}", split)
                            for split in ("train", "validation", "test")
                        ]
                    }
                )
            )
            background = root / "background.wav"
            sf.write(background, np.ones(16_000, dtype=np.float32), 16_000)
            manifest = root / "backgrounds.json"
            manifest.write_text(
                json.dumps(
                    {
                        "examples": [
                            {
                                "path": str(background),
                                "label": 0,
                                "split": "train",
                                "training_eligible": True,
                                "locked_deployment_anchor": False,
                                "source_group": "background_noise",
                                "source_id": "drifted",
                                "audio_sha256": "0" * 64,
                            }
                        ]
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "background audio hash drift"):
                build(
                    [positives],
                    root / "out",
                    background_manifest=manifest,
                    overlay_snr_db=(10.0,),
                )

            payload = json.loads(manifest.read_text())
            payload["examples"][0].update(
                {
                    "audio_sha256": hashlib.sha256(background.read_bytes()).hexdigest(),
                    "split": "test",
                    "training_eligible": False,
                    "locked_deployment_anchor": True,
                }
            )
            manifest.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "training overlays require"):
                build(
                    [positives],
                    root / "locked",
                    background_manifest=manifest,
                    overlay_snr_db=(10.0,),
                )

    def test_two_states_per_phone_produce_sixteen_state_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "positive.wav"
            sf.write(audio, np.zeros(24_000, dtype=np.float32), 16_000)
            rows = [
                aligned_row(audio, f"source-{split}", split)
                for split in ("train", "validation", "test")
            ]
            manifest = root / "positives.json"
            manifest.write_text(json.dumps({"examples": rows}))
            fake_features = np.zeros((260, 40), dtype=np.float32)
            with mock.patch(
                "tools.build_kizz_aligned_teacher_features_v3.frontend",
                return_value=fake_features,
            ):
                report = build(
                    [manifest],
                    root / "out",
                    overlay_snr_db=(),
                    states_per_phone=2,
                )
            targets = np.load(root / "out" / "positive_targets-train.npy")
            self.assertEqual(report["state_count"], 16)
            self.assertLess(int(targets.max()), 16)

    def test_pronunciation_audit_filters_overlay_parents_and_is_source_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "positive.wav"
            sf.write(audio, np.zeros(24_000, dtype=np.float32), 16_000)
            rows = [
                aligned_row(audio, "accepted-train", "train"),
                aligned_row(audio, "rejected-train", "train"),
                aligned_row(audio, "accepted-validation", "validation"),
                aligned_row(audio, "accepted-test", "test"),
            ]
            aligned_manifest = root / "aligned.json"
            aligned_manifest.write_text(json.dumps({"examples": rows}))
            source_manifest = root / "source.json"
            source_manifest.write_text(json.dumps({"examples": []}))
            accepted = {
                "accepted-train",
                "accepted-validation",
                "accepted-test",
            }
            audit = root / "audit.json"
            audit.write_text(
                json.dumps(
                    {
                        "gate_scope": "independent_source_pronunciation_qc",
                        "qualified": True,
                        "source_manifest_sha256": hashlib.sha256(
                            source_manifest.read_bytes()
                        ).hexdigest(),
                        "scope": {
                            "gate_mode": "all",
                            "splits": ["train", "validation", "test"],
                        },
                        "results": [
                            {
                                "source_id": row["source_id"],
                                "accepted": row["source_id"] in accepted,
                            }
                            for row in rows
                        ],
                    }
                )
            )
            fake_features = np.zeros((260, 40), dtype=np.float32)
            with mock.patch(
                "tools.build_kizz_aligned_teacher_features_v3.frontend",
                return_value=fake_features,
            ):
                report = build(
                    [aligned_manifest],
                    root / "out",
                    source_pronunciation_audit=audit,
                    source_manifest=source_manifest,
                    overlay_snr_db=(),
                )
            self.assertEqual(
                report["positive_counts"],
                {"train": 1, "validation": 1, "test": 1},
            )
            self.assertEqual(
                report["source_pronunciation_audit"]["excluded_aligned_count"], 1
            )

            source_manifest.write_text(json.dumps({"examples": [{"drift": True}]}))
            with self.assertRaisesRegex(ValueError, "bound all-split gate"):
                build(
                    [aligned_manifest],
                    root / "drifted",
                    source_pronunciation_audit=audit,
                    source_manifest=source_manifest,
                    overlay_snr_db=(),
                )

    def test_negative_materialization_uses_separate_split_aware_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "audio.wav"
            sf.write(audio, np.zeros(24_000, dtype=np.float32), 16_000)
            positive_manifest = root / "positives.json"
            positive_manifest.write_text(
                json.dumps(
                    {
                        "examples": [
                            aligned_row(audio, f"positive-{split}", split)
                            for split in ("train", "validation", "test")
                        ]
                    }
                )
            )
            negative_manifest = root / "negatives.json"
            negative_manifest.write_text(
                json.dumps(
                    {
                        "examples": [
                            {
                                "path": str(audio),
                                "label": 0,
                                "split": split,
                                "source_group": "collision",
                            }
                            for split in ("train", "validation", "test")
                        ]
                    }
                )
            )
            fake_features = np.zeros((260, 40), dtype=np.float32)
            with mock.patch(
                "tools.build_kizz_aligned_teacher_features_v3.frontend",
                return_value=fake_features,
            ):
                report = build(
                    [positive_manifest],
                    root / "out",
                    negative_manifest=negative_manifest,
                    negative_groups=("collision",),
                    overlay_snr_db=(),
                )
            self.assertEqual(
                report["negative_counts"],
                {
                    "train": {"collision": 1},
                    "validation": {"collision": 1},
                    "test": {"collision": 1},
                },
            )
            self.assertEqual(
                report["negative_manifest"]["path"],
                str(negative_manifest.resolve()),
            )

    def test_negative_groups_may_be_train_only_when_each_eval_split_remains_nonempty(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "audio.wav"
            sf.write(audio, np.zeros(24_000, dtype=np.float32), 16_000)
            positives = root / "positives.json"
            positives.write_text(
                json.dumps(
                    {
                        "examples": [
                            aligned_row(audio, f"positive-{split}", split)
                            for split in ("train", "validation", "test")
                        ]
                    }
                )
            )
            negatives = root / "negatives.json"
            negatives.write_text(
                json.dumps(
                    {
                        "examples": [
                            {
                                "path": str(audio),
                                "label": 0,
                                "split": split,
                                "source_group": "background_noise",
                            }
                            for split in ("train", "validation", "test")
                        ]
                        + [
                            {
                                "path": str(audio),
                                "label": 0,
                                "split": "train",
                                "source_group": "music",
                            }
                        ]
                    }
                )
            )
            with mock.patch(
                "tools.build_kizz_aligned_teacher_features_v3.frontend",
                return_value=np.zeros((260, 40), dtype=np.float32),
            ):
                report = build(
                    [positives],
                    root / "out",
                    negative_manifest=negatives,
                    negative_groups=("background_noise", "music"),
                    overlay_snr_db=(),
                )
            self.assertEqual(
                report["negative_counts"],
                {
                    "train": {"background_noise": 1, "music": 1},
                    "validation": {"background_noise": 1},
                    "test": {"background_noise": 1},
                },
            )

    def test_snr_mixer_preserves_shape_and_is_finite(self):
        foreground = np.ones(CONTEXT_SAMPLES, dtype=np.float32) * 0.1
        background = np.ones(CONTEXT_SAMPLES, dtype=np.float32) * 0.02
        mixed = mix_at_snr(foreground, background, (0.4, 1.0), 10.0)
        self.assertEqual(mixed.shape, foreground.shape)
        self.assertTrue(np.all(np.isfinite(mixed)))

    def test_rir_trims_prearrival_and_rejects_silence(self):
        foreground = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)
        foreground[100] = 0.5
        rir = np.zeros(500, dtype=np.float32)
        rir[42] = 1.0
        rir[90] = 0.25
        convolved, trim = apply_room_impulse_response(foreground, rir)
        self.assertEqual(trim, 42)
        self.assertEqual(convolved.shape, foreground.shape)
        self.assertGreater(float(np.abs(convolved).sum()), 0.5)
        with self.assertRaisesRegex(ValueError, "usable impulse"):
            apply_room_impulse_response(foreground, np.zeros(500, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
