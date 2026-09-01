import tempfile
import unittest
import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.mine_kizz_manifest_hard_negatives import (
    _base_selection,
    _copy_relative_binding_files,
    _effective_split,
    _eligible_rows,
)
from tools.mine_kizz_librispeech_hard_negatives import _stream_training_frontend


class EligibleRowsTests(unittest.TestCase):
    def test_copies_and_verifies_relative_binding_sidecars(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            sidecar = source / "provenance.json"
            sidecar.write_bytes(b'{"source":"device"}\n')
            digest = hashlib.sha256(sidecar.read_bytes()).hexdigest()
            corpus = {
                "bindings": {
                    "provenance": {
                        "path": "provenance.json",
                        "sha256": digest,
                    }
                }
            }

            copied = _copy_relative_binding_files(
                corpus, source_root=source, output_root=output
            )

            self.assertEqual(copied, ["provenance.json"])
            self.assertEqual((output / "provenance.json").read_bytes(), sidecar.read_bytes())

            sidecar.write_bytes(b"drift")
            with self.assertRaisesRegex(ValueError, "hash drift"):
                _copy_relative_binding_files(
                    corpus, source_root=source, output_root=output
                )

    def test_selects_only_unlocked_train_label_zero_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for name in ("eligible.wav", "validation.wav", "positive.wav", "locked.wav"):
                path = root / name
                path.write_bytes(b"fixture")
                paths.append(path)

            manifest = {
                "examples": [
                    {
                        "source_id": "eligible",
                        "path": str(paths[0]),
                        "split": "train",
                        "label": 0,
                        "training_eligible": True,
                    },
                    {
                        "source_id": "validation",
                        "path": str(paths[1]),
                        "split": "validation",
                        "label": 0,
                    },
                    {
                        "source_id": "positive",
                        "path": str(paths[2]),
                        "split": "train",
                        "label": 1,
                    },
                    {
                        "source_id": "locked",
                        "path": str(paths[3]),
                        "split": "train",
                        "label": 0,
                        "locked_holdout": True,
                    },
                ]
            }

            selected = _eligible_rows(manifest)

            self.assertEqual([row["source_id"] for row in selected], ["eligible"])
            self.assertEqual(selected[0]["path"], str(paths[0].resolve()))

    def test_rejects_manifest_without_eligible_train_negatives(self):
        with self.assertRaisesRegex(ValueError, "no eligible development negatives"):
            _eligible_rows({"examples": []})

    def test_optional_validation_is_selected_but_locked_anchors_are_not(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            validation = root / "validation.wav"
            anchor = root / "anchor.wav"
            validation.write_bytes(b"fixture")
            anchor.write_bytes(b"fixture")
            manifest = {
                "examples": [
                    {
                        "source_id": "validation",
                        "path": str(validation),
                        "split": "validation",
                        "label": 0,
                        "training_eligible": False,
                    },
                    {
                        "source_id": "anchor",
                        "path": str(anchor),
                        "split": "validation",
                        "label": 0,
                        "locked_deployment_anchor": True,
                    },
                ]
            }

            with self.assertRaisesRegex(ValueError, "no eligible development negatives"):
                _eligible_rows(manifest)
            selected = _eligible_rows(manifest, include_validation=True)

            self.assertEqual([row["source_id"] for row in selected], ["validation"])

    def test_validation_only_requires_validation_opt_in(self):
        with self.assertRaisesRegex(ValueError, "requires include_validation"):
            _eligible_rows({"examples": []}, validation_only=True)

    def test_train_development_holdout_is_deterministic_and_source_level(self):
        source = {"source_id": "same-source", "split": "train", "path": "/unused"}
        first = _effective_split(source, 0.25)
        second = _effective_split(dict(reversed(list(source.items()))), 0.25)
        self.assertEqual(first, second)
        self.assertIn(first, {"train", "validation"})
        self.assertEqual(_effective_split({**source, "split": "validation"}, 0.25), "validation")
        self.assertEqual(_effective_split(source, 0.0), "train")
        related = {
            "source_id": "different-file",
            "ancestry_id": "shared-recording",
            "split": "train",
            "path": "/unused-b",
        }
        first_related = {**source, "ancestry_id": "shared-recording"}
        self.assertEqual(
            _effective_split(first_related, 0.25),
            _effective_split(related, 0.25),
        )

    def test_base_provider_exclusion_is_exact_and_case_insensitive(self):
        rows = [
            {"candidate_id": "say", "provider": "macos-say"},
            {"candidate_id": "keep", "provider": "Kokoro"},
        ]
        kept, removed = _base_selection(rows, ["MACOS-SAY"])
        np.testing.assert_array_equal(kept, np.asarray([1], dtype=np.int64))
        self.assertEqual([row["candidate_id"] for row in removed], ["say"])

    def test_training_frontend_resamples_source_assets_to_forty_bins(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.wav"
            rate = 44_100
            times = np.arange(rate, dtype=np.float32) / rate
            stereo = np.stack(
                [0.1 * np.sin(2 * np.pi * 440 * times), np.zeros_like(times)],
                axis=1,
            )
            sf.write(path, stereo, rate, subtype="PCM_16")

            frames = list(_stream_training_frontend(path))

            self.assertGreater(len(frames), 20)
            self.assertTrue(all(frame.shape == (40,) for frame in frames))
            self.assertTrue(all(np.all(np.isfinite(frame)) for frame in frames))


if __name__ == "__main__":
    unittest.main()
