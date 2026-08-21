import hashlib
import json
import tempfile
import unittest
import wave
import importlib.util
import sys
from pathlib import Path

from microwakeword.device_corpus import validate_device_corpus

ROOT = Path(__file__).resolve().parents[1]
FEATURE_SPEC = importlib.util.spec_from_file_location(
    "build_device_corpus_features", ROOT / "tools" / "build_device_corpus_features.py"
)
FEATURE_MODULE = importlib.util.module_from_spec(FEATURE_SPEC)
sys.modules[FEATURE_SPEC.name] = FEATURE_MODULE
FEATURE_SPEC.loader.exec_module(FEATURE_MODULE)


def write_wav(path: Path, samples: int = 32000) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\0\0" * samples)


class DeviceCorpusTest(unittest.TestCase):
    def make_corpus(self, root: Path, captures: list[dict]) -> None:
        for item in captures:
            path = root / item["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            write_wav(path)
            item["samples"] = 32000
            item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        (root / "device-corpus.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "corpus_id": "hiphi-device-v1",
                    "device_profiles": {
                        "m5stack_stackchan_k151_cores3_v1": {
                            "audio": {
                                "sample_rate": 16000,
                                "channels": 1,
                                "sample_format": "s16le",
                                "frontend": "m5unified_mic",
                                "gain_profile": "default",
                                "preprocessing": {},
                            }
                        }
                    },
                    "captures": captures,
                }
            )
        )

    def capture(self, capture_id: str, detected: bool = False, **changes) -> dict:
        item = {
            "capture_id": capture_id,
            "path": f"audio/{capture_id}.wav",
            "truth": "positive",
            "phrase": "Hi-Fi Kizz",
            "pronunciation": "hi_fi",
            "speaker_id": "speaker-a",
            "session_id": "session-a",
            "split": "train",
            "detected": detected,
            "device_id": "kizz",
            "device_profile": "m5stack_stackchan_k151_cores3_v1",
            "firmware_sha": "18433e0",
            "conditions": {},
        }
        item.update(changes)
        return item

    def test_retains_positive_attempt_even_when_detector_misses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_corpus(root, [self.capture("missed", detected=False)])
            manifest = validate_device_corpus(root)
            self.assertFalse(manifest["captures"][0]["detected"])
            self.assertEqual(manifest["captures"][0]["truth"], "positive")

    def test_rejects_speaker_leakage_across_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_corpus(
                root,
                [
                    self.capture("train"),
                    self.capture("test", session_id="session-b", split="test"),
                ],
            )
            with self.assertRaisesRegex(ValueError, "speaker speaker-a crosses"):
                validate_device_corpus(root)

    def test_rejects_audio_mutated_after_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_corpus(root, [self.capture("changed")])
            with (root / "audio/changed.wav").open("ab") as wav:
                wav.write(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                validate_device_corpus(root)

    def test_rejects_capture_with_unregistered_device_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_corpus(
                root,
                [
                    self.capture(
                        "dial", device_id="dial-a", device_profile="m5stack_dial_v1"
                    )
                ],
            )
            with self.assertRaisesRegex(ValueError, "unknown device_profile"):
                validate_device_corpus(root)

    def test_feature_source_preserves_explicit_manifest_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_corpus(
                root,
                [
                    self.capture("train"),
                    self.capture(
                        "validation",
                        speaker_id="speaker-b",
                        session_id="session-b",
                        split="validation",
                    ),
                    self.capture(
                        "test",
                        speaker_id="speaker-c",
                        session_id="session-c",
                        split="test",
                    ),
                ],
            )
            manifest = validate_device_corpus(root)
            clips = FEATURE_MODULE.explicit_clips(root, manifest, "positive")
            self.assertEqual(
                {
                    split: len(clips.split_clips[split])
                    for split in ("train", "validation", "test")
                },
                {"train": 1, "validation": 1, "test": 1},
            )

    def test_ambient_holdouts_feed_false_accept_metrics(self):
        self.assertEqual(
            FEATURE_MODULE.feature_split_directory("ambient_negative", "validation"),
            "validation_ambient",
        )
        self.assertEqual(
            FEATURE_MODULE.feature_split_directory("ambient_negative", "test"),
            "testing_ambient",
        )
        self.assertEqual(
            FEATURE_MODULE.feature_split_directory("positive", "test"), "testing"
        )


if __name__ == "__main__":
    unittest.main()
