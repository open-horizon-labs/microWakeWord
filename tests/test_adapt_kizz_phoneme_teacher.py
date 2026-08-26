import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tools.adapt_kizz_phoneme_teacher import (
    APPROVED_PROVIDERS,
    DeterministicBatchMixture,
    adaptation_loss,
    augment_positive_waveform,
    _checkpoint_record_rank,
    _checkpoint_rank,
    _checkpoint_selection_metadata,
    _detector_selection_metrics,
    _discard_evaluated_checkpoint,
    _evaluate_checkpoint,
    load_adaptation_manifest,
    make_adaptation_models,
)
from microwakeword.wake_phrase import KIZZ_CONTROL

try:
    import torch  # noqa: F401
except ImportError:  # pragma: no cover - the focused command supplies torch
    torch = None


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AdaptKizzPhonemeTeacherTests(unittest.TestCase):
    def _manifest(self, directory, *, rows=None):
        rows = rows or []
        for provider in APPROVED_PROVIDERS:
            rows.append({"source_id": f"clean-{provider}", "split": "train", "label": 1, "provider": provider, "source_group": "clean_positive"})
            rows.append({"source_id": f"dev-{provider}", "split": "validation", "label": 1, "provider": provider, "source_group": "clean_positive"})
        for provider in APPROVED_PROVIDERS:
            rows.append({"source_id": f"device-{provider}", "split": "train", "label": 1, "provider": provider, "source_group": "device_channel_positive"})
            rows.append({"source_id": f"device-dev-{provider}", "split": "validation", "label": 1, "provider": provider, "source_group": "device_channel_positive"})
        for group in ("kizz_control_phonetic_collision", "device_collision", "public_speech"):
            rows.append({"source_id": group, "split": "train", "label": 0, "source_group": group})
            rows.append({"source_id": f"dev-{group}", "split": "validation", "label": 0, "source_group": group})
        payload = {
            "base_teacher": {"model_id": "facebook/wav2vec2-lv-60-espeak-cv-ft", "revision": "ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4"},
            "wake_phrase": {
                "phrase_id": KIZZ_CONTROL.phrase_id,
                "phones": list(KIZZ_CONTROL.phones),
                "collision_paths": {
                    name: list(phones)
                    for name, phones in zip(
                        KIZZ_CONTROL.collision_transcripts,
                        KIZZ_CONTROL.collision_phones,
                        strict=True,
                    )
                },
            },
            "examples": rows,
        }
        path = Path(directory) / "manifest.json"; path.write_text(json.dumps(payload, sort_keys=True))
        return path

    def test_manifest_hash_and_required_groups_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._manifest(tmp)
            payload, digest = load_adaptation_manifest(path, expected_sha256=_hash(path))
            self.assertEqual(len(payload["examples"]), 22)
            self.assertEqual(digest, _hash(path))
            with self.assertRaises(ValueError):
                load_adaptation_manifest(path, expected_sha256="0" * 64)
            bad = json.loads(path.read_text()); bad["examples"] = [r for r in bad["examples"] if r["source_group"] != "public_speech"]; path.write_text(json.dumps(bad, sort_keys=True))
            with self.assertRaises(ValueError):
                load_adaptation_manifest(path, expected_sha256=_hash(path))

    def test_manifest_requires_every_approved_clean_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._manifest(tmp)
            payload = json.loads(path.read_text())
            payload["examples"] = [row for row in payload["examples"] if row.get("provider") != "kokoro"]
            path.write_text(json.dumps(payload, sort_keys=True))
            with self.assertRaisesRegex(ValueError, "kokoro"):
                load_adaptation_manifest(path, expected_sha256=_hash(path))

    def test_mixture_is_deterministic_balanced_and_contains_all_negative_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = []
            path = self._manifest(tmp, rows=rows)
            rows = json.loads(path.read_text())["examples"]
            a = DeterministicBatchMixture(rows, batch_size=4, seed=17)
            b = DeterministicBatchMixture(rows, batch_size=4, seed=17)
            self.assertEqual(a.batch(9), b.batch(9))
            seen = {row["source_group"] for step in range(9) for row in a.batch(step)}
            self.assertTrue({"device_channel_positive", "kizz_control_phonetic_collision", "device_collision", "public_speech"} <= seen)
            providers = [row["provider"] for step in range(8) for row in a.batch(step) if row.get("source_group") == "clean_positive"]
            self.assertEqual(len(set(providers)), 4)
            self.assertEqual(
                {provider: providers.count(provider) for provider in APPROVED_PROVIDERS},
                {provider: 2 for provider in APPROVED_PROVIDERS},
            )
            device_providers = [row["provider"] for step in range(8) for row in a.batch(step) if row.get("source_group") == "device_channel_positive"]
            self.assertEqual(
                {provider: device_providers.count(provider) for provider in APPROVED_PROVIDERS},
                {provider: 2 for provider in APPROVED_PROVIDERS},
            )

    def test_augmentation_is_finite_and_length_bounded_and_can_preserve(self):
        wave = np.ones(800, dtype=np.float32) * 0.1
        kept = augment_positive_waveform(wave, np.random.default_rng(1), preserve_probability=1.0)
        changed = augment_positive_waveform(wave, np.random.default_rng(2), background=np.ones(13, dtype=np.float32) * .01, preserve_probability=0.0)
        self.assertEqual(len(kept), len(wave)); self.assertEqual(len(changed), len(wave))
        self.assertTrue(np.isfinite(changed).all()); self.assertLessEqual(np.max(np.abs(changed)), 1.0)

    def test_loss_directions_for_positive_negative_and_collision(self):
        if torch is None:
            self.skipTest("torch is supplied by the focused trainer test command")
        # Two frames, blank=0, canonical=1, collision=2.
        logits = torch.tensor([[[0., 5., -2.], [0., 5., -2.]], [[0., -2., 5.], [0., -2., 5.]]], requires_grad=True)
        lengths = torch.tensor([2, 2]); labels = torch.tensor([1., 0.])
        base = torch.log_softmax(logits.detach(), dim=-1); mask = torch.ones((2, 2), dtype=torch.bool)
        loss, parts = adaptation_loss(logits, lengths, labels, base, mask, canonical_path=(1,), collision_paths={"hifi_kiss": (2,)}, blank_id=0, collision_mask=[False, True])
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(parts["positive_ctc"], 0.0)
        self.assertGreaterEqual(parts["negative_suppression"], 0.0)
        self.assertGreaterEqual(parts["collision_margin"], 0.0)

    def test_positive_collision_margin_is_symmetric_with_negative_collision_margin(self):
        if torch is None:
            self.skipTest("torch is supplied by the focused trainer test command")
        # The positive row incorrectly favors collision token 2. The negative
        # collision row incorrectly favors canonical token 1. Both directions
        # must contribute even though only the negative has collision_mask.
        logits = torch.tensor(
            [
                [[0.0, -2.0, 5.0], [0.0, -2.0, 5.0]],
                [[0.0, 5.0, -2.0], [0.0, 5.0, -2.0]],
            ],
            requires_grad=True,
        )
        lengths = torch.tensor([2, 2])
        labels = torch.tensor([1.0, 0.0])
        base = torch.log_softmax(logits.detach(), dim=-1)
        mask = torch.ones((2, 2), dtype=torch.bool)
        _, parts = adaptation_loss(
            logits,
            lengths,
            labels,
            base,
            mask,
            canonical_path=(1,),
            collision_paths={"kiss": (2,)},
            blank_id=0,
            collision_mask=[False, True],
        )
        self.assertGreater(parts["positive_collision_margin"], 0.0)
        self.assertGreater(parts["negative_collision_margin"], 0.0)

    def test_freeze_contract_keeps_base_immutable(self):
        if torch is None:
            self.skipTest("torch is supplied by the focused trainer test command")
        class Layer(torch.nn.Module):
            def __init__(self): super().__init__(); self.weight = torch.nn.Parameter(torch.ones(1))
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.lm_head = torch.nn.Linear(1, 2); self.wav2vec2 = SimpleNamespace(feature_extractor=Layer(), feature_projection=torch.nn.Linear(1, 1), encoder=SimpleNamespace(layers=torch.nn.ModuleList([Layer(), Layer(), Layer()]))); self.gradient_checkpointing = False
            def parameters(self): return iter([*self.lm_head.parameters(), *self.wav2vec2.feature_extractor.parameters(), *self.wav2vec2.feature_projection.parameters(), *[p for l in self.wav2vec2.encoder.layers for p in l.parameters()]])
            def gradient_checkpointing_enable(self): self.gradient_checkpointing = True
        adapted, frozen = make_adaptation_models(
            Tiny(),
            last_n_encoder_layers=1,
            train_feature_projection=True,
            gradient_checkpointing=True,
        )
        self.assertTrue(all(p.requires_grad for p in adapted.lm_head.parameters()))
        self.assertTrue(all(not p.requires_grad for p in adapted.wav2vec2.feature_extractor.parameters()))
        self.assertTrue(all(p.requires_grad for p in adapted.wav2vec2.feature_projection.parameters()))
        self.assertTrue(all(not p.requires_grad for p in adapted.wav2vec2.encoder.layers[0].parameters()))
        self.assertTrue(all(p.requires_grad for p in adapted.wav2vec2.encoder.layers[-1].parameters()))
        self.assertTrue(adapted.gradient_checkpointing)
        self.assertTrue(all(not p.requires_grad for p in frozen.parameters()))

    def test_detector_selection_keeps_device_family_out_of_threshold(self):
        rows = []
        for index, score in enumerate((0.9, 0.8, 0.7, 0.6)):
            rows.append({"source_id": f"clean-{index}", "split": "validation", "label": 1, "source_group": "clean_positive", "provider": "assemblyai", "duration_seconds": 1})
        for index, score in enumerate((0.95, 0.2, 0.1, 0.0)):
            rows.append({"source_id": f"device-{index}", "split": "validation", "label": 1, "source_group": "device_channel_positive", "provider": "assemblyai", "duration_seconds": 1})
        for index, score in enumerate((-1.0, -2.0, -3.0)):
            rows.append({"source_id": f"negative-{index}", "split": "validation", "label": 0, "source_group": "public_speech", "duration_seconds": 3600})

        def fake_score(row, **kwargs):
            index = int(row["source_id"].rsplit("-", 1)[-1])
            scores = {
                "clean": (0.9, 0.8, 0.7, 0.6),
                "device": (0.95, 0.2, 0.1, 0.0),
                "negative": (-1.0, -2.0, -3.0),
            }
            family = row["source_id"].split("-", 1)[0]
            return {"source_id": row["source_id"], "score": scores[family][index], "collision_margin": 1.0, "duration_seconds": row["duration_seconds"]}

        with patch("tools.adapt_kizz_phoneme_teacher._score_row", side_effect=fake_score):
            metrics = _detector_selection_metrics(
                object(), object(), rows, device="cpu",
                token_ids={"canonical": (1,), "collisions": ((2,),)}, blank_id=0,
                window_lengths=(0.56,), hop=0.06, beta=0.0,
                min_recall=0.90, max_faph=0.10,
            )
        self.assertTrue(metrics["qualified_clean_operating_point"])
        self.assertEqual(metrics["threshold"], 0.6)
        self.assertEqual(metrics["clean"]["recall"], 1.0)
        self.assertEqual(metrics["device_channel"]["recall"], 0.25)
        self.assertEqual(metrics["negative"]["accepted"], 0)
        self.assertEqual(
            [row["accepted"] for row in metrics["rows"]["device_channel_positive"]],
            [True, False, False, False],
        )

    def test_evaluated_checkpoint_pruning_is_scoped_and_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkpoints"
            candidate = root / "step-000050"
            candidate.mkdir(parents=True)
            weights = candidate / "model.safetensors"
            weights.write_bytes(b"weights")
            checkpoint = {"path": str(weights), "file_sha256": _hash(weights)}
            _discard_evaluated_checkpoint(checkpoint, root)
            self.assertFalse(candidate.exists())
            self.assertFalse(checkpoint["retained"])
            outside = Path(tmp) / "outside" / "model.safetensors"
            outside.parent.mkdir(); outside.write_bytes(b"weights")
            with self.assertRaisesRegex(ValueError, "outside the run directory"):
                _discard_evaluated_checkpoint({"path": str(outside)}, root)

    def test_checkpoint_rank_privileges_clean_qualification_then_device_recall(self):
        def selection(qualified, device_recall, clean_recall=1.0, false_accepts=0, faph=0.0):
            return {
                "qualified_clean_operating_point": qualified,
                "device_channel": {"recall": device_recall},
                "clean": {"recall": clean_recall},
                "negative": {"accepted": false_accepts, "faph": faph},
            }
        self.assertGreater(
            _checkpoint_rank(selection(True, 0.1), 10.0),
            _checkpoint_rank(selection(False, 1.0), 0.01),
        )
        self.assertGreater(
            _checkpoint_rank(selection(True, 0.9), 10.0),
            _checkpoint_rank(selection(True, 0.8), 0.01),
        )

    def test_step_zero_checkpoint_is_evaluated_and_hashed_before_training(self):
        detector = {
            "qualified_clean_operating_point": True,
            "device_channel": {"recall": 0.75},
            "clean": {"recall": 1.0},
            "negative": {"accepted": 0, "faph": 0.0},
        }
        args = SimpleNamespace(
            negative_target=-4.0,
            collision_margin=0.2,
            validation_batch_size=4,
            validation_max_per_bucket=32,
            window_length=[0.56, 0.68],
            hop=0.06,
            beta=0.0,
            min_recall=0.90,
            max_faph=0.10,
        )
        checkpoint = {
            "path": "/tmp/step-000000/model.safetensors",
            "file_sha256": "a" * 64,
            "state_sha256": "b" * 64,
        }
        with (
            patch("tools.adapt_kizz_phoneme_teacher.evaluate_validation", return_value={"loss": 0.5, "bucket_losses": {}}),
            patch("tools.adapt_kizz_phoneme_teacher._detector_selection_metrics", return_value=detector),
            patch("tools.adapt_kizz_phoneme_teacher._save_checkpoint", return_value=checkpoint),
        ):
            record = _evaluate_checkpoint(
                object(), object(), object(), [], step=0,
                checkpoint_directory=Path("/tmp/step-000000"), device="cpu",
                canonical_path=(1,), collision_paths={"collision": (2,)},
                blank_id=0, args=args,
                loss_weights={"positive_weight": 1.0},
            )
        self.assertEqual(record["step"], 0)
        self.assertEqual(record["detector_selection"]["metrics"], detector)
        self.assertEqual(record["detector_selection"]["checkpoint"]["file_sha256"], "a" * 64)

    def test_unadapted_step_zero_can_remain_selected(self):
        def record(step, device_recall, loss, digest):
            selection = {
                "qualified_clean_operating_point": True,
                "device_channel": {"recall": device_recall},
                "clean": {"recall": 1.0},
                "negative": {"accepted": 0, "faph": 0.0},
            }
            return {
                "step": step,
                "validation": {"loss": loss},
                "detector_selection": {
                    "rank": list(_checkpoint_rank(selection, loss)),
                    "checkpoint": {"path": f"/tmp/{step}", "file_sha256": digest, "state_sha256": digest},
                    "metrics": selection,
                },
            }

        baseline = record(0, 0.90, 1.0, "a" * 64)
        adapted = record(50, 0.80, 0.01, "b" * 64)
        best = max((baseline, adapted), key=_checkpoint_record_rank)
        metadata = _checkpoint_selection_metadata(
            [baseline, adapted], best, baseline["detector_selection"]["checkpoint"]
        )
        self.assertEqual(best["step"], 0)
        self.assertEqual(metadata["evaluated_steps"], [0, 50])
        self.assertEqual(metadata["selected_step"], 0)
        self.assertEqual(metadata["baseline"]["step"], 0)
        self.assertEqual(metadata["baseline"]["checkpoint"]["file_sha256"], "a" * 64)


if __name__ == "__main__":
    unittest.main()
