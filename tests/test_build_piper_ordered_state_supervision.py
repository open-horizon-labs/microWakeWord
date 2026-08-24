import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from tools.build_piper_ordered_state_supervision import (
    ATOMIC_KIZZ_PHONES,
    build_piper_supervision,
    canonical_phone_spans,
    deterministic_offset,
)


def timing_record():
    tokens = []
    cursor = 0

    def add(kind, base, samples, stress=None):
        nonlocal cursor
        start = cursor
        cursor += samples
        tokens.append(
            {
                "kind": kind,
                "phoneme_base": base,
                "stress": stress,
                "start_s": start / 22050,
                "end_s": cursor / 22050,
            }
        )

    add("boundary", None, 256)
    add("separator", None, 256)
    for index, phone in enumerate(ATOMIC_KIZZ_PHONES):
        if index in {1, 4, 7}:
            add("phoneme", "", 256, "primary")
            add("separator", None, 256)
        add("phoneme", phone, 1024)
        add("separator", None, 256)
        if index == 5:
            add("phoneme", " ", 256)
            add("separator", None, 256)
    add("boundary", None, 256)
    return {"sample_rate": 22050, "total_samples": cursor, "tokens": tokens}


class BuildPiperOrderedStateSupervisionTest(unittest.TestCase):
    def test_measured_trace_maps_to_exact_canonical_sequence(self):
        spans = canonical_phone_spans(timing_record())

        self.assertEqual(
            [span["phone"] for span in spans],
            list(("h", "aɪ", "f", "aɪ", "k", "ɪ", "z")),
        )
        self.assertTrue(
            all(
                left["end_s"] == right["start_s"]
                for left, right in zip(spans, spans[1:])
            )
        )
        self.assertNotEqual(
            len({round(span["end_s"] - span["start_s"], 6) for span in spans}),
            1,
        )

    def test_rejects_a_noncanonical_trace(self):
        timing = timing_record()
        next(token for token in timing["tokens"] if token.get("phoneme_base") == "z")[
            "phoneme_base"
        ] = "s"
        with self.assertRaisesRegex(ValueError, "exactly match canonical"):
            canonical_phone_spans(timing)

    def test_offset_is_deterministic_and_keeps_phrase_in_output_timeline(self):
        first = deterministic_offset("sample", 241, 0.9, 0.03, 0.8, 260, 66)
        second = deterministic_offset("sample", 241, 0.9, 0.03, 0.8, 260, 66)

        self.assertEqual(first, second)
        self.assertGreaterEqual(first + 0.03, 0.655)
        self.assertLessEqual(first + 0.8, 2.605)

    def test_builds_model_shaped_arrays_from_measured_piper_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "audio"
            audio.mkdir()
            timing = timing_record()
            samples = np.zeros(timing["total_samples"], dtype=np.int16)
            samples[100:-100] = 800
            wavfile.write(audio / "0.wav", 22050, samples)
            metadata = root / "metadata.jsonl"
            metadata.write_text(
                json.dumps(
                    {
                        "file": "0.wav",
                        "text": "Hi-Fi Kizz",
                        "speaker_1": 7,
                        "speaker_2": 8,
                        "phoneme_timing": timing,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summary = build_piper_supervision(metadata, audio, root / "output")

            self.assertEqual(summary["feature_shape"], [1, 260, 40])
            self.assertEqual(summary["target_shape"], [1, 66])
            self.assertEqual(summary["rejected_examples"], 0)
            targets = np.load(root / "output" / "arrays" / "targets.npy")
            self.assertTrue(np.any(targets >= 2))
            manifest = json.loads(
                (root / "output" / "frame-supervision-manifest.json").read_text()
            )
            record = manifest["records"][0]
            self.assertEqual(record["source_group"], "piper:7+8")
            self.assertTrue(
                record["alignment"]["timing_record"]["measured_token_samples"]
            )

    def test_duplicate_basenames_get_distinct_feature_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "audio"
            timing = timing_record()
            samples = np.zeros(timing["total_samples"], dtype=np.int16)
            records = []
            for directory in ("a", "b"):
                (audio / directory).mkdir(parents=True)
                wavfile.write(audio / directory / "0.wav", 22050, samples)
                records.append(
                    {
                        "file": f"{directory}/0.wav",
                        "text": "Hi-Fi Kizz",
                        "speaker_1": 7,
                        "speaker_2": 8,
                        "phoneme_timing": timing,
                    }
                )
            metadata = root / "metadata.jsonl"
            metadata.write_text("\n".join(json.dumps(item) for item in records) + "\n")

            summary = build_piper_supervision(metadata, audio, root / "output")

            self.assertEqual(summary["feature_shape"], [2, 260, 40])
            feature_paths = {
                item["features_path"]
                for item in json.loads(
                    (root / "output" / "frame-supervision-manifest.json").read_text()
                )["records"]
            }
            self.assertEqual(len(feature_paths), 2)

    def test_rejects_audio_paths_outside_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "audio"
            audio.mkdir()
            outside = root / "outside.wav"
            timing = timing_record()
            wavfile.write(
                outside, 22050, np.zeros(timing["total_samples"], dtype=np.int16)
            )
            for invalid in ("../outside.wav", str(outside.resolve())):
                metadata = root / "metadata.jsonl"
                metadata.write_text(
                    json.dumps(
                        {
                            "file": invalid,
                            "text": "Hi-Fi Kizz",
                            "speaker_1": 7,
                            "speaker_2": 8,
                            "phoneme_timing": timing,
                        }
                    )
                    + "\n"
                )
                with self.assertRaisesRegex(ValueError, "audio_root"):
                    build_piper_supervision(metadata, audio, root / "output")

            symlink = audio / "linked.wav"
            os.symlink(outside, symlink)
            metadata = root / "metadata.jsonl"
            metadata.write_text(
                json.dumps(
                    {
                        "file": "linked.wav",
                        "text": "Hi-Fi Kizz",
                        "speaker_1": 7,
                        "speaker_2": 8,
                        "phoneme_timing": timing,
                    }
                )
                + "\n"
            )
            with self.assertRaisesRegex(ValueError, "escapes audio_root"):
                build_piper_supervision(metadata, audio, root / "output")


if __name__ == "__main__":
    unittest.main()
