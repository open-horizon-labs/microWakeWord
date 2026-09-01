import hashlib
import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

from tools.extend_kizz_candidate_verifier_with_device_corpus import extend
from tools.train_kizz_candidate_verifier import load_verified_dataset, sha256_file


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_wav(path: Path, samples: int = 16_000) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\0\0" * samples)
    return sha256_file(path)


class ExtensionFixture:
    def __init__(self, root: Path):
        self.root = root
        self.base = root / "base"
        self.base.mkdir()
        self.features = np.arange(6 * 260 * 40, dtype=np.float16).reshape(6, 260, 40)
        self.labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int8)
        self.scores = np.asarray([-3.0, 2.0, -2.0, 1.0, -1.0, 3.0], dtype=np.float32)
        self.feature_frames = np.arange(6, dtype=np.int32)
        self.score_frames = np.arange(6, dtype=np.int32)
        for name, values in {
            "features.npy": self.features,
            "labels.npy": self.labels,
            "detector_scores.npy": self.scores,
            "detector_feature_frames.npy": self.feature_frames,
            "detector_score_frames.npy": self.score_frames,
        }.items():
            np.save(self.base / name, values, allow_pickle=False)
        rows = []
        for index, label in enumerate(self.labels):
            split = ("train", "validation", "test")[index // 2]
            audio_hash = _sha(f"base-audio-{index}".encode())
            feature_hash = _sha(np.ascontiguousarray(self.features[index]).tobytes())
            rows.append(
                {
                    "source_id": f"base-{index}",
                    "candidate_id": f"base-{index}",
                    "parent_source_id": f"base-source-{index}",
                    "source_parent_source_id": f"base-source-{index}",
                    "speaker_id": f"base-speaker-{index}",
                    "session_id": f"base-session-{index}",
                    "ancestry_id": f"base-ancestry-{index}",
                    "audio_sha256": audio_hash,
                    "source_audio_sha256": audio_hash,
                    "parent_source_audio_sha256": audio_hash,
                    "source_group": "base_positive" if label else "base_negative",
                    "semantic_label": "wake_word" if label else "non_wake",
                    "provider": "base-provider",
                    "split": split,
                    "label": int(label),
                    "duration_seconds": 1.0,
                    "detector_conditioned": True,
                    "detector_score": float(self.scores[index]),
                    "detector_feature_frame_index": int(index),
                    "detector_score_frame_index": int(index),
                    "candidate_feature_sha256": feature_hash,
                    "feature_sha256": feature_hash,
                    "feature_index": index,
                    "detector_event": {"score": float(self.scores[index])},
                }
            )
        corpus = {
            "schema_version": 1,
            "recipe": "kizz_control_candidate_conditioned_verifier_v1",
            "candidate_condition": "frozen_detector_trigger_only",
            "input_shape": [260, 40],
            "hard_negative_selection": {
                "ranking": "detector_score_descending_then_candidate_id",
                "group_by": "source",
                "scope": "train_only",
                "top_k": 4,
                "raw_training_count": 1,
                "selected_training_count": 1,
                "heldout_candidates_unfiltered": 2,
            },
            "counts": {
                "selected_candidates": 6,
                "selected_positives": 3,
                "selected_negatives": 3,
                "detector_missed_positives": 0,
                "by_split": {
                    split: {
                        "source_examples": 2,
                        "exposure_seconds": 2.0,
                        "negative_exposure_seconds": 1.0,
                        "raw_detector_candidates": 2,
                        "raw_positive_candidates": 1,
                        "raw_negative_candidates": 1,
                        "selected_positive_candidates": 1,
                        "selected_negative_candidates": 1,
                        "detector_missed_positives": 0,
                        "detector_positive_source_recall": 1.0,
                    }
                    for split in ("train", "validation", "test")
                },
            },
            "examples": rows,
            "array_sha256": {
                name: sha256_file(self.base / name)
                for name in ("features.npy", "labels.npy", "detector_scores.npy")
            },
        }
        (self.base / "corpus.json").write_text(json.dumps(corpus, sort_keys=True))
        self.base_sha = sha256_file(self.base / "corpus.json")
        self.metadata = root / "detector-metadata.json"
        self.model = root / "detector.tflite"
        self.threshold = root / "threshold.json"
        self.metadata.write_text("metadata")
        self.model.write_bytes(b"model")
        self.threshold.write_text("threshold")
        self.device = root / "device-corpus.json"

    def capture(
        self,
        capture_id: str,
        truth: str,
        path: str,
        audio_hash: str,
        *,
        split: str = "train",
        samples: int = 16_000,
        **extra,
    ) -> dict:
        return {
            "capture_id": capture_id,
            "truth": truth,
            "source": "human" if truth == "positive" else "ambient",
            "phrase": "Kizz Control",
            "speaker_id": f"device-speaker-{capture_id}",
            "session_id": f"device-session-{capture_id}",
            "split": split,
            "detected": False,
            "path": path,
            "sha256": audio_hash,
            "samples": samples,
            **extra,
        }

    def write_device(self, captures: list[dict]) -> None:
        self.device.write_text(json.dumps({"schema_version": 2, "captures": captures}))

    def run(self, *, mine, captures, top_k=3, quality=None):
        self.write_device(captures)
        patches = {
            "tools.extend_kizz_candidate_verifier_with_device_corpus._validate_artifact": mock.Mock(
                return_value=({}, {}, {"stride": 1, "phase_offset": 0, "warmup": 0})
            ),
            "tools.extend_kizz_candidate_verifier_with_device_corpus._threshold_from_report": mock.Mock(
                return_value=(-1.0, {"mode": "test"})
            ),
            "tools.extend_kizz_candidate_verifier_with_device_corpus.load_firmware_artifact": mock.Mock(
                return_value={}
            ),
            "tools.extend_kizz_candidate_verifier_with_device_corpus.TFLiteRuntime": mock.Mock(
                return_value=object()
            ),
            "tools.extend_kizz_candidate_verifier_with_device_corpus._mine_file": mine,
        }
        with mock.patch.multiple(
            "tools.extend_kizz_candidate_verifier_with_device_corpus",
            **{key.rsplit(".", 1)[1]: value for key, value in patches.items()},
        ):
            return extend(
                self.base,
                self.base_sha,
                self.device,
                self.metadata,
                self.model,
                self.threshold,
                self.root / "output",
                top_k_per_file=top_k,
                device_quality_report=quality,
            )


class ExtendDeviceCorpusTests(unittest.TestCase):
    def test_quality_report_filters_rejected_capture_before_audio_access(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ExtensionFixture(Path(directory))
            accepted = Path(directory) / "accepted.wav"
            accepted_hash = _write_wav(accepted)
            captures = [
                fixture.capture("accepted", "positive", "accepted.wav", accepted_hash),
                fixture.capture("rejected", "positive", "missing.wav", "a" * 64),
            ]
            fixture.write_device(captures)
            quality = Path(directory) / "quality.json"
            quality.write_text(json.dumps({
                "schema_version": 1,
                "kind": "kizz_control_teacher_adaptation_device_replay_quality",
                "gate_scope": "train_only_target_channel_positive_quality",
                "inputs": {"corpus_sha256": sha256_file(fixture.device)},
                "captures": [],
                "results": [
                    {"capture_id": "accepted", "qualified": True},
                    {"capture_id": "rejected", "qualified": False},
                ],
            }))
            opened = []

            def mine(path, *args, **kwargs):
                opened.append(Path(path))
                return [(1.0, 10, np.ones((260, 40), dtype=np.float32))], 10, 10

            fixture.run(mine=mine, captures=captures, quality=quality)
            self.assertEqual(opened, [accepted.resolve()])
            ledger = json.loads(
                (Path(directory) / "output" / "device-corpus-candidate-extension-ledger.json").read_text()
            )
            self.assertEqual(
                ledger["quarantined_captures"][0]["reason"],
                "device_quality_gate_rejected",
            )

    def test_test_audio_is_quarantined_without_open_or_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ExtensionFixture(Path(directory))
            train_path = Path(directory) / "train.wav"
            train_hash = _write_wav(train_path)
            test_path = Path(directory) / "does-not-exist.wav"
            test_hash = "a" * 64
            captures = [
                fixture.capture("train", "hard_negative", "train.wav", train_hash),
                fixture.capture(
                    "locked-test",
                    "positive",
                    "does-not-exist.wav",
                    test_hash,
                    split="test",
                    locked_deployment_anchor=True,
                ),
            ]
            opened: list[Path] = []
            original_hash = sha256_file

            def mine(path, *args, **kwargs):
                opened.append(Path(path))
                return [], 10, 10

            with mock.patch(
                "tools.extend_kizz_candidate_verifier_with_device_corpus.sha256_file",
                side_effect=lambda path: (
                    opened.append(Path(path)) or original_hash(path)
                ),
            ):
                result = fixture.run(mine=mine, captures=captures)
            self.assertEqual(result["quarantined_captures"], 1)
            self.assertNotIn(test_path, opened)
            ledger = json.loads(
                (Path(directory) / "output" / "device-corpus-candidate-extension-ledger.json").read_text()
            )
            self.assertFalse(ledger["selection_policy"]["test_audio_opened"])
            self.assertFalse(ledger["selection_policy"]["test_audio_hashed"])

    def test_detector_miss_positive_is_not_fabricated(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ExtensionFixture(Path(directory))
            path = Path(directory) / "positive.wav"
            audio_hash = _write_wav(path)

            def mine(*args, **kwargs):
                return [], 10, 10

            fixture.run(
                mine=mine,
                captures=[fixture.capture("miss", "positive", "positive.wav", audio_hash)],
            )
            output = Path(directory) / "output"
            dataset = load_verified_dataset(
                output, expected_corpus_sha256=sha256_file(output / "corpus.json")
            )
            self.assertEqual(len(dataset.rows), 6)
            self.assertEqual(json.loads((output / "corpus.json").read_text())["counts"]["detector_missed_positives"], 1)

    def test_non_target_positive_is_quarantined_without_audio_access(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ExtensionFixture(Path(directory))
            capture = fixture.capture(
                "wrong-phrase",
                "positive",
                "must-not-be-opened.wav",
                "a" * 64,
                phrase="Hi-Fi Kizz",
            )
            opened = []
            original_hash = sha256_file
            with mock.patch(
                "tools.extend_kizz_candidate_verifier_with_device_corpus.sha256_file",
                side_effect=lambda path: opened.append(Path(path)) or original_hash(path),
            ):
                result = fixture.run(
                    mine=lambda *args, **kwargs: self.fail("mismatched positive was mined"),
                    captures=[capture],
                )
            self.assertEqual(result["quarantined_captures"], 1)
            self.assertNotIn(Path(directory) / "must-not-be-opened.wav", opened)
            ledger = json.loads(
                (Path(directory) / "output" / "device-corpus-candidate-extension-ledger.json").read_text()
            )
            self.assertEqual(
                ledger["quarantined_captures"][0]["reason"],
                "positive_phrase_mismatch",
            )

    def test_positive_uses_one_candidate_and_truths_map_to_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ExtensionFixture(Path(directory))
            positive = Path(directory) / "positive.wav"
            negative = Path(directory) / "negative.wav"
            positive_hash = _write_wav(positive)
            negative_hash = _write_wav(negative)
            calls = []
            feature = np.ones((260, 40), dtype=np.float32)

            def mine(path, *args, **kwargs):
                calls.append(kwargs["top_k"])
                selected = feature if "positive" in str(path) else feature * 2
                return [(3.0, 20, selected), (2.0, 40, selected * 2)], 10, 10

            fixture.run(
                mine=mine,
                captures=[
                    fixture.capture("p", "positive", "positive.wav", positive_hash),
                    fixture.capture("n", "ambient_negative", "negative.wav", negative_hash),
                ],
                top_k=2,
            )
            output = Path(directory) / "output"
            corpus = json.loads((output / "corpus.json").read_text())
            self.assertEqual(calls, [1, 2])
            self.assertEqual([row["label"] for row in corpus["examples"][-3:]], [1, 0, 0])
            self.assertEqual(
                {row["capture_id"]: row["label"] for row in corpus["examples"][-3:]},
                {"p": 1, "n": 0},
            )

    def test_hash_and_split_contracts_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ExtensionFixture(Path(directory))
            path = Path(directory) / "train.wav"
            audio_hash = _write_wav(path)
            bad_hash = fixture.capture("bad-hash", "hard_negative", "train.wav", "b" * 64)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                fixture.run(mine=lambda *args, **kwargs: ([], 0, 0), captures=[bad_hash])
            bad_split = fixture.capture("bad-split", "hard_negative", "train.wav", audio_hash, split="holdout")
            with self.assertRaisesRegex(ValueError, "unsupported split"):
                fixture.run(mine=lambda *args, **kwargs: ([], 0, 0), captures=[bad_split])
            locked = fixture.capture(
                "locked", "hard_negative", "train.wav", audio_hash, locked_holdout=True
            )
            with self.assertRaisesRegex(ValueError, "locked_holdout"):
                fixture.run(mine=lambda *args, **kwargs: ([], 0, 0), captures=[locked])

    def test_feature_hash_dedup_preserves_base_prefix_and_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ExtensionFixture(Path(directory))
            path = Path(directory) / "negative.wav"
            audio_hash = _write_wav(path)
            duplicate_feature = fixture.features[0].astype(np.float32)
            unique_feature = np.full((260, 40), 7, dtype=np.float32)

            def mine(*args, **kwargs):
                return [(1.0, 10, duplicate_feature), (0.5, 20, unique_feature)], 10, 10

            fixture.run(
                mine=mine,
                captures=[fixture.capture("dedupe", "hard_negative", "negative.wav", audio_hash)],
            )
            output = Path(directory) / "output"
            dataset = load_verified_dataset(
                output, expected_corpus_sha256=sha256_file(output / "corpus.json")
            )
            self.assertEqual(len(dataset.rows), 7)
            np.testing.assert_array_equal(dataset.features[:6], fixture.features)
            self.assertEqual(json.loads((output / "provenance.json").read_text())["output_contract"]["base_rows_preserved_as_prefix"], True)


if __name__ == "__main__":
    unittest.main()
