import io
import hashlib
import json
import tarfile
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
import soundfile as sf

from microwakeword.kizz_continuous_evaluation import poisson_upper_95
from tools.qualify_kizz_phoneme_continuous import (
    _load_teacher_qualification,
    _scan_chunk,
    qualify_archive,
)
from microwakeword.kizz_phoneme_teacher import score_window


def _wav(seconds: float, sample_rate: int = 16_000) -> bytes:
    frames = int(seconds * sample_rate)
    payload = (np.zeros(frames, dtype="<i2")).tobytes()
    output = io.BytesIO()
    with wave.open(output, "wb") as value:
        value.setnchannels(1)
        value.setsampwidth(2)
        value.setframerate(sample_rate)
        value.writeframes(payload)
    return output.getvalue()


def _archive(path: Path) -> None:
    with tarfile.open(path, "w:gz") as output:
        for name, seconds in (("speech/a.wav", 3.0), ("music/b.wav", 3.0), ("noise/c.wav", 3.0)):
            payload = _wav(seconds)
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            output.addfile(info, io.BytesIO(payload))


class QualifyKizzPhonemeContinuousTests(unittest.TestCase):
    def test_vectorized_threshold_pruning_matches_exhaustive_ctc_decisions(self):
        rng = np.random.default_rng(238)
        logits = rng.normal(size=(53, 9))
        log_probs = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
        canonical = (1, 2, 3, 1)
        collisions = ((1, 2, 4, 1), (5, 2, 3, 1))
        lengths = (11, 17, 23)
        hop = 3
        beta = 0.15
        threshold = -4.0
        expected = {}
        for requested in lengths:
            length = min(requested, len(log_probs))
            starts = list(range(0, len(log_probs) - length + 1, hop))
            tail = len(log_probs) - length
            if not starts or starts[-1] != tail:
                starts.append(tail)
            for start in starts:
                item = score_window(
                    log_probs[start : start + length],
                    canonical_tokens=canonical,
                    collision_tokens=collisions,
                    blank_id=0,
                    start_frame=start,
                )
                score = (
                    item.canonical_fit
                    if item.canonical_fit >= threshold
                    and item.collision_margin >= beta
                    else -np.inf
                )
                expected[start] = max(expected.get(start, -np.inf), score)
        actual = _scan_chunk(
            log_probs,
            canonical_tokens=canonical,
            collision_tokens=collisions,
            blank_id=0,
            window_lengths=lengths,
            hop=hop,
            beta=beta,
            minimum_score=threshold,
        )
        self.assertEqual(set(actual), set(expected))
        np.testing.assert_allclose(
            [actual[key] for key in sorted(actual)],
            [expected[key] for key in sorted(expected)],
            rtol=1e-12,
            atol=1e-12,
        )
    def test_streams_tar_without_extracting_and_counts_categories(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "musan.tar.gz"
            _archive(archive)
            calls = []

            def infer(waveform):
                calls.append(len(waveform))
                # Blank is dominant, so every score is below the frozen threshold.
                return np.zeros((max(2, len(waveform) // 160), 3), dtype=np.float32)

            report = qualify_archive(
                archive,
                threshold=0.0,
                beta=0.0,
                window_lengths_seconds=(0.20,),
                hop_seconds=0.10,
                chunk_seconds=0.50,
                min_exposure_hours=0.002,
                infer_logits=infer,
                token_ids={"canonical": (1,), "collisions": ((2,),)},
                blank_id=0,
            )
            self.assertTrue(calls)
            self.assertEqual(report["counts"]["files"], 3)
            self.assertEqual(report["counts"]["false_accepts"], 0)
            self.assertAlmostEqual(report["counts"]["exposure_seconds"], 9.0)
            self.assertEqual(set(report["categories"]), {"speech", "music", "noise"})
            self.assertEqual(report["source"]["archive_sha256"], __import__("hashlib").sha256(archive.read_bytes()).hexdigest())

    def test_overlap_dedupes_one_long_high_run_before_event_counting(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "musan.tar.gz"
            with tarfile.open(archive, "w:gz") as output:
                payload = _wav(1.1)
                info = tarfile.TarInfo("noise/long.wav")
                info.size = len(payload)
                output.addfile(info, io.BytesIO(payload))

            calls = []

            def infer(waveform):
                calls.append(len(waveform))
                logits = np.zeros((max(2, len(waveform) // 100), 3), dtype=np.float32)
                if len(calls) == 2:
                    logits[:, 1] = 5.0  # canonical token wins only in chunk two
                    logits[:, 2] = -5.0
                else:
                    logits[:, 1] = -5.0
                    logits[:, 2] = 5.0  # collision margin rejects these windows
                return logits

            report = qualify_archive(
                archive,
                threshold=-10.0,
                beta=0.0,
                window_lengths_seconds=(0.20,),
                hop_seconds=0.05,
                chunk_seconds=0.50,
                refractory_seconds=1.0,
                min_exposure_hours=0.0001,
                infer_logits=infer,
                token_ids={"canonical": (1,), "collisions": ((2,),)},
                blank_id=0,
            )
            self.assertEqual(report["counts"]["false_accepts"], 1)
            self.assertEqual(len(report["members"][0]["events"]), 1)
            self.assertGreater(len(calls), 1)
            self.assertAlmostEqual(report["members"][0]["events"][0]["start_seconds"], 0.30, places=5)
            self.assertAlmostEqual(report["counts"]["exposure_seconds"], 1.1, places=6)

    def test_locked_manifest_normalizes_connected_speech_and_rejects_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "spoken.wav"
            audio.write_bytes(_wav(0.75))
            manifest = root / "locked.json"
            row = {
                "path": str(audio),
                "sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
                "duration_seconds": 0.75,
                "category": "connected_speech",
            }
            manifest.write_text(json.dumps([row]))
            report = qualify_archive(
                None,
                manifest=manifest,
                threshold=0.0,
                beta=0.0,
                window_lengths_seconds=(0.20,),
                hop_seconds=0.10,
                chunk_seconds=0.50,
                min_exposure_hours=0.0001,
                infer_logits=lambda waveform: np.zeros((max(2, len(waveform) // 160), 3)),
                token_ids={"canonical": (1,), "collisions": ((2,),)},
                blank_id=0,
            )
            self.assertEqual(report["categories"]["speech"]["files"], 1)
            self.assertNotIn("connected_speech", report["categories"])
            self.assertEqual(
                report["source"]["manifest_sha256"],
                hashlib.sha256(manifest.read_bytes()).hexdigest(),
            )

            audio.write_bytes(_wav(0.80))
            with self.assertRaisesRegex(ValueError, "manifest hash drift"):
                qualify_archive(
                    None,
                    manifest=manifest,
                    threshold=0.0,
                    beta=0.0,
                    window_lengths_seconds=(0.20,),
                    chunk_seconds=0.50,
                    infer_logits=lambda waveform: np.zeros((2, 3)),
                    token_ids={"canonical": (1,), "collisions": ((2,),)},
                )

            row["sha256"] = hashlib.sha256(audio.read_bytes()).hexdigest()
            manifest.write_text(json.dumps([row]))
            with self.assertRaisesRegex(ValueError, "manifest duration drift"):
                qualify_archive(
                    None,
                    manifest=manifest,
                    threshold=0.0,
                    beta=0.0,
                    window_lengths_seconds=(0.20,),
                    chunk_seconds=0.50,
                    infer_logits=lambda waveform: np.zeros((2, 3)),
                    token_ids={"canonical": (1,), "collisions": ((2,),)},
                )

    def test_locked_examples_manifest_streams_flac(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "speech.flac"
            sf.write(audio, np.zeros(17_600, dtype=np.float32), 16_000, format="FLAC", subtype="PCM_16")
            manifest = root / "locked.json"
            manifest.write_text(json.dumps({
                "counts": {"files": 1},
                "examples": [{
                    "path": str(audio),
                    "sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
                    "duration_seconds": 1.1,
                    "category": "connected_speech",
                }],
            }))
            report = qualify_archive(
                None,
                manifest=manifest,
                threshold=0.0,
                beta=0.0,
                window_lengths_seconds=(0.20,),
                hop_seconds=0.10,
                chunk_seconds=0.50,
                min_exposure_hours=0.0001,
                infer_logits=lambda waveform: np.zeros((max(2, len(waveform) // 160), 3)),
                token_ids={"canonical": (1,), "collisions": ((2,),)},
                blank_id=0,
            )
            self.assertEqual(report["counts"]["files"], 1)
            self.assertAlmostEqual(report["counts"]["exposure_seconds"], 1.1, places=6)
            self.assertEqual(report["members"][0]["category"], "speech")

    def test_gate_math_uses_one_sided_poisson_upper_bound_and_exposure(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "musan.tar.gz"
            _archive(archive)
            report = qualify_archive(
                archive,
                threshold=0.0,
                beta=0.0,
                window_lengths_seconds=(0.20,),
                hop_seconds=0.10,
                chunk_seconds=0.50,
                min_exposure_hours=0.001,
                max_faph_upper_95=poisson_upper_95(0, 9.0 / 3600.0) + 0.001,
                infer_logits=lambda waveform: np.zeros((max(2, len(waveform) // 160), 3)),
                token_ids={"canonical": (1,), "collisions": ((2,),)},
                blank_id=0,
            )
            self.assertEqual(report["counts"]["faph_upper_95"], poisson_upper_95(0, 9.0 / 3600.0))
            self.assertTrue(report["qualified"])
            self.assertEqual(report["scoring"]["threshold"], 0.0)
            self.assertEqual(report["model"]["revision"], "ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4")

    def test_teacher_qualification_freezes_operating_point_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "musan.tar.gz"
            _archive(archive)
            qualification = {
                "qualified": True,
                "phones": {"phrase_id": "kizz-control"},
                "scoring": {"threshold": -0.25, "collision_margin_beta": 0.125},
                "model": {
                    "id": "teacher-id",
                    "revision": "teacher-revision",
                    "weights_sha256": "weights",
                    "config_sha256": "config",
                    "tokenizer_vocab_sha256": "vocab",
                },
            }
            qualification_path = root / "qualification.json"
            qualification_path.write_text(json.dumps(qualification))
            loaded, report_sha = _load_teacher_qualification(qualification_path)
            report = qualify_archive(
                archive,
                threshold=999.0,
                beta=999.0,
                window_lengths_seconds=(0.20,),
                hop_seconds=0.10,
                chunk_seconds=0.50,
                min_exposure_hours=0.001,
                infer_logits=lambda waveform: np.zeros((max(2, len(waveform) // 160), 3)),
                token_ids={"canonical": (1,), "collisions": ((2,),)},
                blank_id=0,
                teacher_qualification=loaded,
                teacher_qualification_sha256=report_sha,
            )
            self.assertEqual(report["scoring"]["threshold"], -0.25)
            self.assertEqual(report["scoring"]["collision_margin_beta"], 0.125)
            self.assertEqual(report["gate_scope"], "untouched_continuous_qualification")
            self.assertEqual(report["teacher_qualification"]["report_sha256"], report_sha)
            self.assertEqual(report["model"]["weights_sha256"], "weights")


if __name__ == "__main__":
    unittest.main()
