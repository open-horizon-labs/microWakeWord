import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.build_kizz_phoneme_distillation_corpus import (
    deterministic_negative_context,
    hard_phone_targets,
    load_device_training_rows,
    load_pronunciation_acceptances,
    select_negative_rows,
    student_test_positive_evidence,
)


class BuildKizzPhonemeDistillationCorpusTests(unittest.TestCase):
    def test_pronunciation_allowlist_requires_bound_all_split_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            source.write_text('{"examples": []}\n')
            from tools.build_kizz_phoneme_distillation_corpus import sha256_file
            audit = root / "audit.json"
            audit.write_text(json.dumps({
                "gate_scope": "independent_source_pronunciation_qc",
                "source_manifest_sha256": sha256_file(source),
                "scope": {"gate_mode": "all", "splits": ["train", "validation", "test"]},
                "results": [
                    {"source_id": "good", "accepted": True},
                    {"source_id": "bad", "accepted": False},
                ],
            }))
            self.assertEqual(load_pronunciation_acceptances(audit, source), {"good"})
            payload = json.loads(audit.read_text())
            payload["scope"]["gate_mode"] = "reserved"
            audit.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "all-split"):
                load_pronunciation_acceptances(audit, source)

    def test_device_training_rows_require_exact_qualified_provider_voice_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected = []
            captures = []
            results = []
            heldout = []
            providers = ("assemblyai", "deepgram", "elevenlabs", "kokoro")
            phones = ("k", "ɪ", "z", "k", "ə", "n", "t", "ɹ", "oʊ", "l")
            for provider in providers:
                for index in range(4):
                    case = providers.index(provider) * 4 + index + 1
                    source_audio = root / f"source-{provider}-{index}.wav"
                    source_values = np.zeros(16_000, dtype=np.float32)
                    source_values[0] = 0.01 + case / 1000.0
                    sf.write(
                        source_audio,
                        source_values,
                        16_000,
                        subtype="PCM_16",
                    )
                    source_hash = hashlib.sha256(source_audio.read_bytes()).hexdigest()
                    descriptor = hashlib.sha256(
                        f"{provider}-{index}".encode()
                    ).hexdigest()
                    spans = [
                        {
                            "phone": phone,
                            "start_s": 0.10 + phone_index * 0.05,
                            "end_s": 0.15 + phone_index * 0.05,
                        }
                        for phone_index, phone in enumerate(phones)
                    ]
                    selected.append(
                        {
                            "source_id": f"source:{provider}:{index}",
                            "path": str(source_audio),
                            "audio_sha256": source_hash,
                            "descriptor_sha256": descriptor,
                            "provider": provider,
                            "voice": f"train-{index}",
                            "target_id": "kizz-control",
                            "label": 1,
                            "split": "train",
                            "training_eligible": True,
                            "semantic_label": "canonical_exact",
                            "target_phones": list(phones),
                            "phrase_span": {"start_s": 0.1, "end_s": 0.6},
                            "phone_spans": spans,
                            "alignment": {
                                "method": "wav2vec2_ipa_ctc_forced_alignment",
                                "pronunciation_decision": {"accepted": True}
                            },
                        }
                    )
                    captured = root / f"capture-{provider}-{index}.wav"
                    capture_values = np.zeros(48_000, dtype=np.float32)
                    capture_values[0] = 0.10 + case / 1000.0
                    sf.write(
                        captured,
                        capture_values,
                        16_000,
                        subtype="PCM_16",
                    )
                    audio_hash = hashlib.sha256(captured.read_bytes()).hexdigest()
                    capture_id = f"capture-{provider}-{index}"
                    captures.append(
                        {
                            "capture_id": capture_id,
                            "path": captured.name,
                            "sha256": audio_hash,
                            "split": "train",
                            "truth": "positive",
                            "phrase": "Kizz Control",
                            "speaker_id": f"speaker-{provider}-{index}",
                            "conditions": {
                                "source_audio_sha256": source_hash,
                                "source_descriptor_sha256": descriptor,
                                "source_provider": provider,
                                "source_voice": f"train-{index}",
                            },
                        }
                    )
                    results.append(
                        {
                            "capture_id": capture_id,
                            "audio_sha256": audio_hash,
                            "source_audio_sha256": source_hash,
                            "provider": provider,
                            "voice": f"train-{index}",
                            "playback_lag_seconds": 0.5,
                            "qualified": True,
                            "failure_reasons": [],
                        }
                    )
                    heldout_audio = root / f"heldout-{provider}-{index}.wav"
                    heldout_values = np.zeros(16_000, dtype=np.float32)
                    heldout_values[0] = 0.20 + case / 1000.0
                    sf.write(
                        heldout_audio,
                        heldout_values,
                        16_000,
                        subtype="PCM_16",
                    )
                    heldout.append(
                        {
                            "path": str(heldout_audio),
                            "audio_sha256": hashlib.sha256(
                                heldout_audio.read_bytes()
                            ).hexdigest(),
                            "provider": provider,
                            "voice": f"heldout-{index}",
                        }
                    )

            selection = root / "selection.json"
            selection.write_text(
                json.dumps(
                    {
                        "kind": "kizz_control_teacher_adaptation_device_replay_selection",
                        "locked_before_teacher_adaptation": True,
                        "selected_count": 16,
                        "selected_examples": selected,
                    }
                )
            )
            corpus = root / "device-corpus.json"
            corpus.write_text(json.dumps({"captures": captures}))
            qualification = root / "qualification.json"
            qualification.write_text(json.dumps({"examples": heldout}))
            quality = root / "quality.json"
            quality.write_text(
                json.dumps(
                    {
                        "kind": "kizz_control_teacher_adaptation_device_replay_quality",
                        "gate_scope": "train_only_target_channel_positive_quality",
                        "qualified": True,
                        "counts": {
                            "providers": {provider: 4 for provider in providers}
                        },
                        "inputs": {
                            "corpus": str(corpus),
                            "corpus_sha256": hashlib.sha256(
                                corpus.read_bytes()
                            ).hexdigest(),
                            "selection": str(selection),
                            "selection_sha256": hashlib.sha256(
                                selection.read_bytes()
                            ).hexdigest(),
                            "qualification_evidence": str(qualification),
                            "qualification_evidence_sha256": hashlib.sha256(
                                qualification.read_bytes()
                            ).hexdigest(),
                        },
                        "results": results,
                    }
                )
            )
            rows = load_device_training_rows(quality)
            self.assertEqual(len(rows), 16)
            self.assertEqual(
                {
                    provider: sum(
                        row["provider"] == provider for row in rows
                    )
                    for provider in providers
                },
                {provider: 4 for provider in providers},
            )
            self.assertTrue(
                all(
                    row["source_group"] == "device_channel_positive"
                    and row["split"] == "train"
                    for row in rows
                )
            )
            bad = json.loads(quality.read_text())
            bad["counts"]["providers"]["kokoro"] = 3
            quality.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "4x4"):
                load_device_training_rows(quality)

    def test_negative_selection_excludes_locked_hashes(self):
        rows = []
        for index in range(3):
            rows.append(
                {
                    "label": 0,
                    "split": "train",
                    "audio_sha256": str(index) * 64,
                    "source_id": f"speech-{index}",
                    "source_group": "public_speech",
                    "training_eligible": True,
                }
            )
        selected = select_negative_rows(
            rows, {"0" * 64}, public_per_split={"train": 3}
        )
        self.assertEqual([row["source_id"] for row in selected], ["speech-1", "speech-2"])

    def test_negative_context_is_hash_deterministic(self):
        samples = np.arange(50_000, dtype=np.float32)
        first = deterministic_negative_context(samples, "1" * 64)
        second = deterministic_negative_context(samples, "1" * 64)
        self.assertEqual(first.shape, (41_920,))
        self.assertTrue(np.array_equal(first, second))

    def test_hard_targets_cover_all_canonical_phone_ids(self):
        phones = ("k", "ɪ", "z", "k", "ə", "n", "t", "ɹ", "oʊ", "l")
        boundaries = np.linspace(0.7, 2.0, len(phones) + 1)
        spans = [
            {"phone": phone, "start_s": float(start), "end_s": float(end)}
            for phone, start, end in zip(phones, boundaries[:-1], boundaries[1:])
        ]
        targets = hard_phone_targets(spans, 0.0)
        self.assertEqual(targets.shape, (66,))
        self.assertGreater(len(set(targets.tolist())), 8)

    def test_student_test_evidence_is_positive_only_and_locked_before_training(self):
        evidence = student_test_positive_evidence([
            {"source_id": "positive", "split": "test", "label": 1, "training_eligible": False},
            {"source_id": "negative", "split": "test", "label": 0, "training_eligible": False},
            {"source_id": "train", "split": "train", "label": 1, "training_eligible": True},
        ])
        self.assertTrue(evidence["locked_before_student_training"])
        self.assertFalse(evidence["training_eligible"])
        self.assertEqual([row["source_id"] for row in evidence["examples"]], ["positive"])
        with self.assertRaisesRegex(ValueError, "training-ineligible"):
            student_test_positive_evidence([
                {"source_id": "bad", "split": "test", "label": 1, "training_eligible": True}
            ])


if __name__ == "__main__":
    unittest.main()
