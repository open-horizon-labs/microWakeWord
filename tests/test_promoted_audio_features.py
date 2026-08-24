import hashlib
import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

from microwakeword.promoted_audio import (
    OUTPUT_MANIFEST_NAME,
    build_features,
    phrase_aligned_pcm,
    select_entries,
    validate_manifest,
)


class PromotedAudioFeaturesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.wav = self.root / "clean" / "clip.wav"
        self.wav.parent.mkdir()
        with wave.open(str(self.wav), "wb") as output:
            output.setparams((1, 2, 16000, 1600, "NONE", "not compressed"))
            output.writeframes(b"\0" * 3200)

    def tearDown(self):
        self.temp.cleanup()

    def manifest(self, **changes):
        entry = {
            "id": "clip-1",
            "wav_path": str(self.wav.resolve()),
            "sha256": hashlib.sha256(self.wav.read_bytes()).hexdigest(),
            "truth": "positive",
            "split": "train",
            "text": "Hi-Fi Kizz",
            "phrase_span": {"start_ms": 0, "end_ms": 100},
            "provenance": "reviewed-device-capture",
            "human_reviewed": True,
            "training_eligible": True,
        }
        entry.update(changes)
        path = self.root / "manifest.json"
        path.write_text(json.dumps({"schema_version": 1, "entries": [entry]}))
        return path

    def test_validates_hash_audio_and_identity(self):
        manifest = validate_manifest(self.manifest())
        self.assertEqual(manifest["entries"][0]["id"], "clip-1")

    def test_rejects_hash_quarantine_and_review_failures(self):
        for changes, message in (
            ({"sha256": "0" * 64}, "sha256 mismatch"),
            ({"training_eligible": False}, "training_eligible"),
            ({"human_reviewed": False}, "human_reviewed"),
        ):
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                validate_manifest(self.manifest(**changes))
        quarantined = self.root / "evidence" / "clip.wav"
        quarantined.parent.mkdir()
        quarantined.write_bytes(self.wav.read_bytes())
        with self.assertRaisesRegex(ValueError, "quarantined"):
            validate_manifest(
                self.manifest(
                    wav_path=str(quarantined),
                    sha256=hashlib.sha256(quarantined.read_bytes()).hexdigest(),
                )
            )
        quarantined_manifest = self.root / "observations" / "manifest.json"
        quarantined_manifest.parent.mkdir()
        quarantined_manifest.write_text(self.manifest().read_text())
        with self.assertRaisesRegex(ValueError, "quarantined"):
            validate_manifest(quarantined_manifest)

    def test_rejects_symlinks_that_resolve_into_quarantine(self):
        quarantined = self.root / "observations"
        quarantined.mkdir()
        quarantined_wav = quarantined / "clip.wav"
        quarantined_wav.write_bytes(self.wav.read_bytes())
        clean_link = self.root / "clean-link.wav"
        clean_link.symlink_to(quarantined_wav)
        with self.assertRaisesRegex(ValueError, "quarantined"):
            validate_manifest(
                self.manifest(
                    wav_path=str(clean_link),
                    sha256=hashlib.sha256(clean_link.read_bytes()).hexdigest(),
                )
            )

        quarantined_manifest = quarantined / "manifest.json"
        quarantined_manifest.write_text(self.manifest().read_text())
        clean_manifest_link = self.root / "clean-manifest.json"
        clean_manifest_link.symlink_to(quarantined_manifest)
        with self.assertRaisesRegex(ValueError, "quarantined"):
            validate_manifest(clean_manifest_link)

    def test_rejects_missing_and_duplicate_entries(self):
        path = self.manifest()
        payload = json.loads(path.read_text())
        payload["entries"].append(dict(payload["entries"][0]))
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_manifest(path)
        payload["entries"] = [dict(payload["entries"][0], id="")]
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "non-empty id"):
            validate_manifest(path)
        payload = json.loads(self.manifest().read_text())
        payload["entries"][0]["unexpected"] = "does not mask a missing required field"
        del payload["entries"][0]["provenance"]
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            validate_manifest(path)

    def test_exact_canonical_filter_keeps_hard_negatives(self):
        manifest = {
            "entries": [
                {"truth": "positive", "text": "Hi-Fi Kizz"},
                {"truth": "positive", "text": "Hippy Kizz"},
                {"truth": "hard_negative", "text": "Hippy Kizz"},
            ]
        }
        self.assertEqual(len(select_entries(manifest, "Hi-Fi Kizz")), 2)

    def test_build_maps_splits_and_writes_provenance_contract(self):
        manifest_path = self.manifest()
        output = self.root / "features"
        fake_spectrograms = MagicMock()
        fake_spectrograms.spectrogram_generator.return_value = iter(())
        fake_mmap = MagicMock()
        with (
            patch("microwakeword.promoted_audio._clips", return_value=MagicMock()),
            patch("microwakeword.promoted_audio._augmenter", return_value=MagicMock()),
            patch(
                "microwakeword.promoted_audio.SpectrogramGeneration",
                return_value=fake_spectrograms,
            ),
            patch("microwakeword.promoted_audio.RaggedMmap", new=fake_mmap),
        ):
            build_features(manifest_path, output)
        self.assertEqual(
            fake_mmap.from_generator.call_args.kwargs["out_dir"],
            str(output / "positive" / "training" / "wakeword_mmap"),
        )
        saved = json.loads((output / OUTPUT_MANIFEST_NAME).read_text())
        self.assertTrue((output / "positive" / "training").is_dir())
        self.assertEqual(saved["entries"][0]["id"], "clip-1")
        self.assertEqual(saved["entries"][0]["provenance"], "reviewed-device-capture")
        self.assertTrue(saved["entries"][0]["training_eligible"])

    def test_dry_run_does_not_generate(self):
        report = build_features(self.manifest(), self.root / "features", dry_run=True)
        self.assertEqual(len(report["entries"]), 1)

    def test_positive_alignment_preserves_phrase_and_fixes_window_length(self):
        import numpy as np

        pcm = np.arange(64000, dtype=np.int16)
        aligned = phrase_aligned_pcm(pcm, 1500, 2200)
        self.assertEqual(aligned.size, 32000)
        self.assertTrue(np.any(aligned == pcm[24000]))
        self.assertTrue(np.any(aligned == pcm[35199]))

    def test_rejects_positive_without_phrase_span(self):
        with self.assertRaisesRegex(ValueError, "phrase_span"):
            validate_manifest(self.manifest(phrase_span=None))


if __name__ == "__main__":
    unittest.main()
