import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

from tools.generate_kizz_kokoro_phoneme_v3 import (
    ENGLISH_VOICES,
    MODEL_ID,
    RAW_PHONES,
    SPLIT_VOICES,
    TARGET_ID,
    TARGET_PHONES,
    generate,
    normalize_pcm16,
    split_for_voice,
)


class GenerateKizzKokoroPhonemeV3Test(unittest.TestCase):
    def fake_model(self, root):
        model = root / "model.bin"
        model.write_bytes(b"deterministic model")
        return model

    def fake_synth(self, phones, voice, speed):
        self.assertEqual(phones, RAW_PHONES)
        return np.full(1600, 0.1, dtype=np.float32), 16000

    def test_manifest_records_fixed_voice_disjoint_splits_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            payload = generate(
                root / "audio",
                manifest,
                self.fake_model(root),
                speeds=(1.0,),
                synthesizer=self.fake_synth,
            )
            self.assertEqual(len(payload["examples"]), len(ENGLISH_VOICES))
            self.assertEqual(payload["contract"]["model_id"], MODEL_ID)
            self.assertEqual(payload["contract"]["raw_phones"], RAW_PHONES)
            self.assertEqual(payload["contract"]["target_id"], TARGET_ID)
            self.assertEqual(payload["contract"]["target_phones"], list(TARGET_PHONES))
            self.assertEqual(
                payload["contract"]["source_group"], "kokoro_phoneme_synthetic"
            )
            self.assertEqual(payload["contract"]["role"], "positive")
            self.assertEqual(
                ENGLISH_VOICES,
                (
                    "af_alloy",
                    "af_aoede",
                    "af_bella",
                    "af_heart",
                    "af_jessica",
                    "af_kore",
                    "af_nicole",
                    "af_nova",
                    "af_river",
                    "af_sarah",
                    "af_sky",
                    "am_adam",
                    "am_echo",
                    "am_eric",
                    "am_fenrir",
                    "am_liam",
                    "am_michael",
                    "am_onyx",
                    "am_puck",
                    "am_santa",
                    "bf_alice",
                    "bf_emma",
                    "bf_isabella",
                    "bf_lily",
                    "bm_daniel",
                    "bm_fable",
                    "bm_george",
                    "bm_lewis",
                ),
            )
            self.assertEqual(
                set(payload["contract"]["voice_split_map"]), set(ENGLISH_VOICES)
            )
            for row in payload["examples"]:
                self.assertEqual(row["split"], split_for_voice(row["voice"]))
                self.assertEqual(row["sample_rate"], 16000)
                self.assertEqual(row["channels"], 1)
                self.assertEqual(row["audio_sha256"], row["output_hash"])
                self.assertEqual(row["source_group"], "kokoro_phoneme_synthetic")
                self.assertEqual(row["target_id"], TARGET_ID)
                self.assertEqual(row["target_phones"], list(TARGET_PHONES))
                self.assertEqual(row["role"], "positive")
                self.assertEqual(len(row["voice_sha256"]), 64)
                with wave.open(row["path"], "rb") as audio:
                    self.assertEqual(
                        (
                            audio.getframerate(),
                            audio.getnchannels(),
                            audio.getsampwidth(),
                        ),
                        (16000, 1, 2),
                    )
            self.assertEqual(
                {row["voice"] for row in payload["examples"] if row["split"] == "test"},
                set(SPLIT_VOICES["test"]),
            )

    def test_unknown_voice_and_output_collision_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = self.fake_model(root)
            with self.assertRaisesRegex(ValueError, "unknown"):
                split_for_voice("af_future")
            output = root / "audio"
            output.mkdir()
            source_id = (
                __import__("hashlib")
                .sha256(
                    (
                        "canonical-v3-kokoro-phoneme-1" + "\0" + "af_alloy" + "\0" + "1"
                    ).encode()
                )
                .hexdigest()
            )
            (output / f"{source_id}.wav").write_bytes(b"collision")
            with self.assertRaises(FileExistsError):
                generate(
                    output,
                    root / "manifest.json",
                    model,
                    speeds=(1.0,),
                    synthesizer=self.fake_synth,
                )

    def test_incompatible_manifest_is_rejected_before_rendering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = self.fake_model(root)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"generator": "other", "examples": []}))
            synth = mock.Mock(side_effect=self.fake_synth)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                generate(
                    root / "audio", manifest, model, speeds=(1.0,), synthesizer=synth
                )
            synth.assert_not_called()

    def test_non_16k_input_is_explicitly_normalized_to_16k(self):
        pcm = normalize_pcm16(np.zeros(24, dtype=np.float32), 24000)
        self.assertEqual(len(pcm), 16 * 2)

    def test_default_speed_contract_is_five_deterministic_speeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = generate(
                root / "audio",
                root / "manifest.json",
                self.fake_model(root),
                synthesizer=self.fake_synth,
            )
            self.assertEqual(
                payload["contract"]["speeds"], [0.82, 0.91, 1.0, 1.09, 1.18]
            )
            self.assertEqual(len(payload["examples"]), 28 * 5)


if __name__ == "__main__":
    unittest.main()
