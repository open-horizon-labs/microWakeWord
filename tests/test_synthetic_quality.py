import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile

from microwakeword.synthetic_quality import (
    QualityBounds,
    load_quality_mask,
    quality_reasons,
    reference_bounds,
)
from tools.build_synthetic_quality_mask import build_report


class SyntheticQualityTest(unittest.TestCase):
    def test_report_compares_recorded_and_synthetic_spans(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recipe = root / "recipe.yaml"
            recipe.write_text("clip_duration_ms: 2000\n")

            generated = root / "generated"
            phrase = generated / "positive" / "wake"
            phrase.mkdir(parents=True)
            for name in ("accepted.wav", "long.wav"):
                wavfile.write(phrase / name, 16000, np.zeros(16000, np.int16))
            (generated / "generation-manifest.json").write_text(
                json.dumps(
                    {
                        "recipe_sha256": hashlib.sha256(
                            recipe.read_bytes()
                        ).hexdigest(),
                        "plan": [
                            {
                                "class": "positive",
                                "text": "Wake",
                                "output": str(phrase),
                            }
                        ],
                    }
                )
            )

            corpus = root / "device-corpus"
            audio = corpus / "audio"
            audio.mkdir(parents=True)
            captures = []
            for index, span_ms in enumerate((600, 800, 900)):
                path = audio / f"human-{index}.wav"
                wavfile.write(path, 16000, np.zeros(32000, np.int16))
                captures.append(
                    {
                        "capture_id": f"human-{index}",
                        "path": f"audio/{path.name}",
                        "truth": "positive",
                        "source": "human",
                        "phrase": "Wake",
                        "speaker_id": "speaker-a",
                        "session_id": "session-a",
                        "split": "train",
                        "detected": False,
                        "device_id": "device-a",
                        "device_profile": "test_mic_v1",
                        "samples": 32000,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "phrase_span": {"start_ms": 100, "end_ms": 100 + span_ms},
                    }
                )
            (corpus / "device-corpus.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "corpus_id": "test-corpus",
                        "device_profiles": {
                            "test_mic_v1": {
                                "audio": {
                                    "sample_rate": 16000,
                                    "channels": 1,
                                    "sample_format": "s16le",
                                    "frontend": "test",
                                    "gain_profile": "default",
                                    "preprocessing": {},
                                }
                            }
                        },
                        "captures": captures,
                    }
                )
            )

            metrics = [
                {
                    "duration_ms": 1000,
                    "speech_span_ms": 800,
                    "rms_dbfs": -20.0,
                    "clipped_fraction": 0.0,
                },
                {
                    "duration_ms": 1800,
                    "speech_span_ms": 800,
                    "rms_dbfs": -20.0,
                    "clipped_fraction": 0.0,
                },
            ]
            with patch(
                "tools.build_synthetic_quality_mask.audio_metrics",
                side_effect=metrics,
            ):
                report = build_report(recipe, generated, corpus, 300)

            self.assertEqual(report["reference_positive_spans_ms"]["median"], 800)
            self.assertEqual(
                report["synthetic_by_truth"]["positive"]["speech_span_ms"]["median"],
                800,
            )
            self.assertEqual(report["accepted_clips"], 1)
            self.assertEqual(report["rejected_clips"], 1)

    def test_reference_bounds_are_broad_but_cannot_cross_truncation_limit(self):
        bounds = reference_bounds(
            [560, 720, 800, 800, 880, 880],
            clip_duration_ms=2000,
            maximum_jitter_ms=300,
        )

        self.assertLessEqual(bounds.minimum_speech_ms, 560)
        self.assertGreaterEqual(bounds.maximum_speech_ms, 880)
        self.assertEqual(bounds.maximum_source_ms, 1700)
        self.assertLessEqual(bounds.maximum_speech_ms, bounds.maximum_source_ms)

    def test_positive_span_outlier_and_truncation_risk_are_rejected(self):
        bounds = QualityBounds(
            minimum_speech_ms=300,
            maximum_speech_ms=1500,
            maximum_source_ms=1700,
            maximum_clipped_fraction=0.001,
            minimum_rms_dbfs=-50.0,
        )

        self.assertEqual(
            quality_reasons(
                {
                    "duration_ms": 1750,
                    "speech_span_ms": 900,
                    "clipped_fraction": 0.0,
                    "rms_dbfs": -20.0,
                },
                "positive",
                bounds,
            ),
            ["source_would_be_truncated"],
        )
        self.assertEqual(
            quality_reasons(
                {
                    "duration_ms": 1200,
                    "speech_span_ms": 1600,
                    "clipped_fraction": 0.0,
                    "rms_dbfs": -20.0,
                },
                "positive",
                bounds,
            ),
            ["speech_span_too_long"],
        )

    def test_reference_span_bounds_do_not_filter_hard_negative_phrases(self):
        bounds = QualityBounds(300, 1500, 1700, 0.001, -50.0)
        reasons = quality_reasons(
            {
                "duration_ms": 1200,
                "speech_span_ms": 120,
                "clipped_fraction": 0.0,
                "rms_dbfs": -20.0,
            },
            "hard_negative",
            bounds,
        )
        self.assertEqual(reasons, [])

    def test_mask_is_bound_to_recipe_and_generation_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recipe = root / "recipe.yaml"
            generation = root / "generation-manifest.json"
            mask = root / "quality-mask.json"
            recipe.write_text("name: test\n")
            generation.write_text('{"plan": []}\n')
            mask.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "recipe_sha256": hashlib.sha256(
                            recipe.read_bytes()
                        ).hexdigest(),
                        "generation_manifest_sha256": hashlib.sha256(
                            generation.read_bytes()
                        ).hexdigest(),
                        "rejected": {"positive/bad.wav": ["clipped_audio"]},
                    }
                )
            )

            loaded = load_quality_mask(mask, recipe, generation)
            self.assertEqual(
                loaded["rejected"],
                {"positive/bad.wav": ["clipped_audio"]},
            )

            recipe.write_text("name: changed\n")
            with self.assertRaisesRegex(ValueError, "recipe hash"):
                load_quality_mask(mask, recipe, generation)


if __name__ == "__main__":
    unittest.main()
