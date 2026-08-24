import hashlib
import tempfile
import unittest
import wave
from pathlib import Path

from tools.write_kizz_v32_device_promotion import build_promotion_manifest


def write_wav(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, 16000, 160, "NONE", "not compressed"))
        output.writeframes(b"\0" * 320)
    return hashlib.sha256(path.read_bytes()).hexdigest()


class KizzV32DevicePromotionTest(unittest.TestCase):
    def test_selects_only_canonical_train_and_test_with_original_phrase_span(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / "corpus"
            (corpus / "device-corpus.json").parent.mkdir(parents=True)
            (corpus / "device-corpus.json").write_text("{}")
            original_hash = write_wav(corpus / "audio" / "canonical.wav")
            test_hash = write_wav(corpus / "audio" / "test.wav")
            broad_hash = write_wav(corpus / "audio" / "broad.wav")
            manifest = {
                "corpus_id": "device-v1",
                "captures": [
                    {
                        "capture_id": "canonical",
                        "truth": "positive",
                        "split": "train",
                        "pronunciation": "hi_fi_kizz",
                        "source": "synthetic_playback",
                        "speaker_id": "train-speaker",
                        "path": "audio/canonical.wav",
                        "sha256": original_hash,
                        "phrase_span": {"start_ms": 0, "end_ms": 10},
                    },
                    {
                        "capture_id": "test",
                        "truth": "positive",
                        "split": "test",
                        "pronunciation": "hi_fi",
                        "source": "synthetic_playback",
                        "speaker_id": "test-speaker",
                        "path": "audio/test.wav",
                        "sha256": test_hash,
                        "phrase_span": {"start_ms": 0, "end_ms": 10},
                    },
                    {
                        "capture_id": "broad",
                        "truth": "positive",
                        "split": "train",
                        "pronunciation": "hiffy_kizz",
                        "source": "synthetic_playback",
                        "speaker_id": "broad-speaker",
                        "path": "audio/broad.wav",
                        "sha256": broad_hash,
                        "phrase_span": {"start_ms": 0, "end_ms": 10},
                    },
                ],
            }
            result = build_promotion_manifest(corpus, manifest)
            self.assertEqual(
                [entry["id"] for entry in result["entries"]], ["canonical", "test"]
            )
            canonical = result["entries"][0]
            self.assertEqual(canonical["sha256"], original_hash)
            self.assertEqual(
                Path(canonical["wav_path"]),
                (corpus / "audio" / "canonical.wav").resolve(),
            )
            self.assertEqual(canonical["phrase_span"], {"start_ms": 0, "end_ms": 10})
            self.assertEqual(canonical["text"], "Hi-Fi Kizz")
            self.assertTrue(canonical["training_eligible"])
            self.assertEqual(
                result["speaker_ids_by_split"],
                {"test": ["test-speaker"], "train": ["train-speaker"]},
            )

    def test_rejects_speaker_overlap_between_train_and_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / "corpus"
            corpus.mkdir()
            (corpus / "device-corpus.json").write_text("{}")
            digest = write_wav(corpus / "clip.wav")
            captures = []
            for split in ("train", "test"):
                captures.append(
                    {
                        "capture_id": split,
                        "truth": "positive",
                        "split": split,
                        "pronunciation": "hi_fi",
                        "source": "human",
                        "speaker_id": "same-speaker",
                        "path": "clip.wav",
                        "sha256": digest,
                        "phrase_span": {"start_ms": 0, "end_ms": 10},
                    }
                )
            with self.assertRaisesRegex(ValueError, "speakers overlap"):
                build_promotion_manifest(
                    corpus, {"corpus_id": "overlap", "captures": captures}
                )

    def test_rejects_an_empty_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / "corpus"
            corpus.mkdir()
            (corpus / "device-corpus.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "no canonical"):
                build_promotion_manifest(
                    corpus,
                    {"corpus_id": "empty", "captures": []},
                )


if __name__ == "__main__":
    unittest.main()
