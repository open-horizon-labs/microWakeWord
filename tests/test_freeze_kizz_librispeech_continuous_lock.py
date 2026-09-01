import json
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.freeze_kizz_librispeech_continuous_lock import freeze, sha256_file


def _candidate(path: Path, *, speaker="other-speaker", ancestry="other-ancestry", audio="a" * 64):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "examples": [
                    {
                        "split": split,
                        "speaker_id": speaker,
                        "ancestry_id": ancestry,
                        "audio_sha256": audio,
                    }
                    for split in ("train", "validation", "test")
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _flac(root: Path, speaker: int, chapter: int, utterance: int, seconds: float = 1.0):
    path = root / str(speaker) / str(chapter) / f"{speaker}-{chapter}-{utterance:04d}.flac"
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(seconds * 16_000)
    offset = ((speaker * 17 + chapter * 7 + utterance) % 50) / 500.0
    values = np.linspace(-0.1 + offset, 0.1 + offset, frames, dtype=np.float32)
    sf.write(path, values, 16_000, format="FLAC")
    return path


class FreezeLibriSpeechContinuousLockTests(unittest.TestCase):
    def test_small_fixture_freezes_deterministically_and_keeps_whole_speakers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "train-clean-360"
            for speaker in range(10, 14):
                _flac(root, speaker, 1, 0, seconds=2.0)
                _flac(root, speaker, 1, 1, seconds=1.0)
            candidate = Path(temporary) / "corpus.json"
            _candidate(candidate)
            archive = Path(temporary) / "train-clean-360.tar.gz"
            archive.write_bytes(b"openslr archive")
            first = Path(temporary) / "first.json"
            second = Path(temporary) / "second.json"

            # The public CLI enforces the real 100-hour gate.  This focused
            # test exercises selection through an intentionally tiny target.
            with self.assertRaisesRegex(ValueError, "at least 100.0"):
                freeze(root, candidate, first, minimum_hours=0.001)

            # A focused unit test cannot use a 100-hour raw-audio fixture, so
            # lower only the module's production guard for this test process.
            from unittest import mock
            import tools.freeze_kizz_librispeech_continuous_lock as tool

            with mock.patch.object(tool, "DEFAULT_MINIMUM_HOURS", 0.001):
                archive_md5 = hashlib.md5(archive.read_bytes(), usedforsecurity=False).hexdigest()
                freeze(root, candidate, first, source_archive=archive, source_archive_md5=archive_md5, minimum_hours=0.002, margin_hours=0.0, seed=7)
                freeze(root, candidate, second, source_archive=archive, source_archive_md5=archive_md5, minimum_hours=0.002, margin_hours=0.0, seed=7)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            payload = json.loads(first.read_text())
            self.assertEqual(payload["schema_version"], 2)
            self.assertEqual(payload["gate_scope"], "locked_untouched_continuous_negative_corpus")
            self.assertTrue(payload["locked_before_scoring"])
            self.assertFalse(payload["training_eligible"])
            self.assertEqual(payload["counts"]["categories"], {"speech": payload["counts"]["files"]})
            self.assertEqual(payload["bindings"]["candidate_corpus"]["sha256"], sha256_file(candidate))
            self.assertEqual(payload["bindings"]["source_archive"]["sha256"], sha256_file(archive))
            self.assertEqual(payload["bindings"]["source_archive"]["md5"], archive_md5)
            self.assertEqual(payload["bindings"]["source_archive"]["expected_md5"], archive_md5)
            by_speaker = {}
            for row in payload["examples"]:
                by_speaker.setdefault(row["speaker_id"], set()).add(row["session_id"])
                self.assertEqual(row["split"], "test")
                self.assertEqual(row["category"], "speech")
                self.assertEqual(row["source"], "OpenSLR SLR12 LibriSpeech train-clean-360")
            self.assertTrue(by_speaker)
            self.assertEqual(payload["overlap_proof"]["speaker_overlap"], 0)
            self.assertEqual(payload["overlap_proof"]["audio_sha256_overlap"], 0)
            self.assertEqual(payload["overlap_proof"]["ancestry_overlap"], 0)
            from tools.evaluate_kizz_int8_continuous_cascade import load_locked_manifest

            loaded = load_locked_manifest(first, minimum_exposure_hours=0.0)
            self.assertEqual(len(loaded.rows), payload["counts"]["files"])
            self.assertEqual(
                {row.manifest_row["speaker_id"] for row in loaded.rows},
                set(by_speaker),
            )

    def test_rejects_candidate_speaker_audio_and_ancestry_overlap_without_replacing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "train-clean-360"
            audio = _flac(root, 10, 1, 0, seconds=4.0)
            candidate = Path(temporary) / "corpus.json"
            output = Path(temporary) / "lock.json"
            output.write_text("sentinel\n", encoding="utf-8")
            from unittest import mock
            import tools.freeze_kizz_librispeech_continuous_lock as tool

            cases = (
                ("librispeech-speaker:10", "other-ancestry", "a" * 64, "speaker_ids"),
                ("librispeech-mini:10", "other-ancestry", "a" * 64, "speaker_ids"),
                ("other-speaker", "other-ancestry", sha256_file(audio), "audio_sha256"),
                ("other-speaker", "librispeech-speaker:10", "a" * 64, "ancestry_ids"),
            )
            for speaker, ancestry, digest, label in cases:
                _candidate(candidate, speaker=speaker, ancestry=ancestry, audio=digest)
                with mock.patch.object(tool, "DEFAULT_MINIMUM_HOURS", 0.001):
                    with self.assertRaisesRegex(ValueError, label):
                        freeze(root, candidate, output, minimum_hours=0.001, margin_hours=0.0)
                self.assertEqual(output.read_text(encoding="utf-8"), "sentinel\n")

    def test_rejects_non_mono_or_insufficient_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "train-clean-360"
            bad = root / "10" / "1" / "10-1-0000.flac"
            bad.parent.mkdir(parents=True)
            sf.write(bad, np.zeros((160, 2), dtype=np.float32), 16_000, format="FLAC")
            candidate = Path(temporary) / "corpus.json"
            _candidate(candidate)
            with self.assertRaisesRegex(ValueError, "16 kHz mono"):
                freeze(root, candidate, Path(temporary) / "lock.json")

    def test_rejects_bad_official_archive_checksum_without_writing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "train-clean-360"
            _flac(root, 10, 1, 0, seconds=4.0)
            candidate = Path(temporary) / "corpus.json"
            _candidate(candidate)
            archive = Path(temporary) / "train-clean-360.tar.gz"
            archive.write_bytes(b"not the expected archive")
            output = Path(temporary) / "lock.json"
            with self.assertRaisesRegex(ValueError, "archive MD5 mismatch"):
                freeze(
                    root,
                    candidate,
                    output,
                    source_archive=archive,
                    source_archive_md5="0" * 32,
                )
            self.assertFalse(output.exists())

    def test_train_other_subset_has_distinct_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "train-other-500"
            _flac(root, 10, 1, 0, seconds=4.0)
            candidate = Path(temporary) / "corpus.json"
            _candidate(candidate)
            output = Path(temporary) / "lock.json"
            from unittest import mock
            import tools.freeze_kizz_librispeech_continuous_lock as tool

            with mock.patch.object(tool, "DEFAULT_MINIMUM_HOURS", 0.001):
                freeze(
                    root,
                    candidate,
                    output,
                    subset="train-other-500",
                    minimum_hours=0.001,
                    margin_hours=0.0,
                )
            payload = json.loads(output.read_text())
            self.assertEqual(
                payload["kind"],
                "kizz_control_librispeech_train_other_500_continuous_negative_lock",
            )
            self.assertEqual(
                payload["source"], "OpenSLR SLR12 LibriSpeech train-other-500"
            )
            self.assertTrue(
                payload["examples"][0]["source_id"].startswith(
                    "librispeech-train-other-500:"
                )
            )

    def test_excluded_lock_forces_a_speaker_disjoint_second_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "train-other-500"
            for speaker in range(10, 14):
                _flac(root, speaker, 1, 0, seconds=2.0)
                _flac(root, speaker, 1, 1, seconds=1.0)
            candidate = Path(temporary) / "corpus.json"
            _candidate(candidate)
            first = Path(temporary) / "first.json"
            second = Path(temporary) / "second.json"
            from unittest import mock
            import tools.freeze_kizz_librispeech_continuous_lock as tool

            with mock.patch.object(tool, "DEFAULT_MINIMUM_HOURS", 0.0001):
                freeze(
                    root,
                    candidate,
                    first,
                    subset="train-other-500",
                    minimum_hours=0.0005,
                    margin_hours=0.0,
                    seed=17,
                )
                freeze(
                    root,
                    candidate,
                    second,
                    subset="train-other-500",
                    exclude_locked_manifest=first,
                    minimum_hours=0.0005,
                    margin_hours=0.0,
                    seed=17,
                )
            first_payload = json.loads(first.read_text())
            second_payload = json.loads(second.read_text())
            first_speakers = {row["speaker_id"] for row in first_payload["examples"]}
            second_speakers = {row["speaker_id"] for row in second_payload["examples"]}
            self.assertFalse(first_speakers & second_speakers)
            self.assertEqual(
                second_payload["bindings"]["excluded_continuous_lock"]["sha256"],
                sha256_file(first),
            )
            self.assertEqual(
                second_payload["overlap_proof"]["excluded_continuous_lock"],
                {"speaker": 0, "audio": 0, "ancestry": 0},
            )

    def test_repeated_excluded_locks_union_all_prior_speaker_identities(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "train-other-500"
            for speaker in range(10, 16):
                _flac(root, speaker, 1, 0, seconds=2.0)
                _flac(root, speaker, 1, 1, seconds=1.0)
            candidate = Path(temporary) / "corpus.json"
            _candidate(candidate)
            locks = [Path(temporary) / f"lock-{index}.json" for index in range(3)]
            from unittest import mock
            import tools.freeze_kizz_librispeech_continuous_lock as tool

            with mock.patch.object(tool, "DEFAULT_MINIMUM_HOURS", 0.0001):
                freeze(
                    root, candidate, locks[0], subset="train-other-500",
                    minimum_hours=0.0005, margin_hours=0.0, seed=17,
                )
                freeze(
                    root, candidate, locks[1], subset="train-other-500",
                    exclude_locked_manifest=locks[0], minimum_hours=0.0005,
                    margin_hours=0.0, seed=17,
                )
                freeze(
                    root, candidate, locks[2], subset="train-other-500",
                    exclude_locked_manifest=locks[:2], minimum_hours=0.0005,
                    margin_hours=0.0, seed=17,
                )
            payloads = [json.loads(path.read_text()) for path in locks]
            speaker_sets = [
                {row["speaker_id"] for row in payload["examples"]}
                for payload in payloads
            ]
            self.assertFalse(speaker_sets[2] & speaker_sets[0])
            self.assertFalse(speaker_sets[2] & speaker_sets[1])
            self.assertEqual(
                [row["sha256"] for row in payloads[2]["bindings"]["excluded_continuous_locks"]],
                [sha256_file(locks[0]), sha256_file(locks[1])],
            )
            self.assertEqual(payloads[2]["selection_policy"]["excluded_lock_count"], 2)


if __name__ == "__main__":
    unittest.main()
