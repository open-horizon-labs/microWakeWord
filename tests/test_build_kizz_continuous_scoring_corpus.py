import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.build_kizz_continuous_scoring_corpus import (
    SAMPLE_RATE,
    _select_background,
    build,
)
from tools.trace_kizz_ordered_state_detector import feature_sha256, sha256_file


class ContinuousScoringCorpusTests(unittest.TestCase):
    def test_background_selection_is_deterministic_and_split_local(self):
        pools = {
            "train": [{"id": "a"}, {"id": "b"}],
            "validation": [{"id": "v"}],
        }
        first = _select_background(
            pools, split="train", source_id="wake", seed=7
        )
        second = _select_background(
            pools, split="train", source_id="wake", seed=7
        )
        self.assertEqual(first, second)
        self.assertIn(first, pools["train"])
        self.assertEqual(
            _select_background(
                pools, split="validation", source_id="wake", seed=7
            )["id"],
            "v",
        )

    def test_build_emits_product_features_and_reproducible_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_audio = root / "source.wav"
            background_audio = root / "background.wav"
            times = np.arange(round(2.62 * SAMPLE_RATE)) / SAMPLE_RATE
            sf.write(
                source_audio,
                (0.15 * np.sin(2 * np.pi * 440 * times)).astype(np.float32),
                SAMPLE_RATE,
                subtype="PCM_16",
            )
            background_times = np.arange(5 * SAMPLE_RATE) / SAMPLE_RATE
            sf.write(
                background_audio,
                (0.05 * np.sin(2 * np.pi * 120 * background_times)).astype(np.float32),
                SAMPLE_RATE,
                subtype="PCM_16",
            )
            source_manifest = root / "source.json"
            source_features = root / "source-features.npy"
            source_tensor = np.full((260, 40), 3.0, dtype=np.float32)
            np.save(source_features, source_tensor[None])
            source_manifest.write_text(
                json.dumps(
                    {
                        "array_sha256": {
                            source_features.name: sha256_file(source_features)
                        },
                        "examples": [
                            {
                                "source_id": "wake",
                                "path": str(source_audio),
                                "audio_sha256": sha256_file(source_audio),
                                "feature_index": 0,
                                "feature_sha256": feature_sha256(source_tensor),
                                "split": "train",
                                "label": 1,
                                "duration_seconds": 1.75,
                                "provider": "fixture",
                                "source_group": "fixture_positive",
                            }
                        ],
                    }
                )
            )
            background_manifest = root / "background.json"
            background_manifest.write_text(
                json.dumps(
                    {
                        "files": [
                            {
                                "path": background_audio.name,
                                "sha256": sha256_file(background_audio),
                                "evidence_split": "train",
                                "source": "fixture",
                            }
                        ]
                    }
                )
            )

            report = build(
                source_manifest,
                source_features,
                background_manifest,
                root / "output",
                prefix_seconds=2.0,
                seed=11,
            )

            features = np.load(report["source_features"])
            manifest = json.loads(Path(report["source_manifest"]).read_text())
            self.assertEqual(features.shape, (1, 460, 40))
            self.assertEqual(manifest["input_shape"], [460, 40])
            self.assertEqual(manifest["context_duration_seconds"], 4.62)
            self.assertEqual(
                manifest["examples"][0]["foreground_duration_seconds"], 1.75
            )
            np.testing.assert_array_equal(features[0, 200:], source_tensor)
            self.assertEqual(
                manifest["examples"][0]["composition"]["background"]["evidence_split"],
                "train",
            )
            self.assertEqual(manifest["examples"][0]["window"] if "window" in manifest["examples"][0] else None, None)

    def test_build_accepts_physical_capture_path_and_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture = root / "capture.wav"
            background = root / "background.wav"
            sf.write(capture, np.zeros(round(2.62 * SAMPLE_RATE)), SAMPLE_RATE)
            sf.write(background, np.zeros(5 * SAMPLE_RATE), SAMPLE_RATE)
            features_path = root / "features.npy"
            tensor = np.zeros((260, 40), dtype=np.float32)
            np.save(features_path, tensor[None])
            source_manifest = root / "source.json"
            source_manifest.write_text(json.dumps({
                "array_sha256": {features_path.name: sha256_file(features_path)},
                "examples": [{
                    "source_id": "physical",
                    "capture_path": str(capture),
                    "capture_audio_sha256": sha256_file(capture),
                    "source_audio_sha256": "original-source-is-not-the-capture",
                    "source_duration_seconds": 1.25,
                    "feature_index": 0,
                    "feature_sha256": feature_sha256(tensor),
                    "split": "test",
                    "label": 1,
                }],
            }))
            background_manifest = root / "background.json"
            background_manifest.write_text(json.dumps({"files": [{
                "path": background.name,
                "sha256": sha256_file(background),
                "evidence_split": "test",
            }]}))

            report = build(
                source_manifest,
                features_path,
                background_manifest,
                root / "output",
            )
            manifest = json.loads(Path(report["source_manifest"]).read_text())
            self.assertEqual(manifest["examples"][0]["source_audio_sha256"], sha256_file(capture))
            self.assertEqual(manifest["examples"][0]["foreground_duration_seconds"], 1.25)


if __name__ == "__main__":
    unittest.main()
