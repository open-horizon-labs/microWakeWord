import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tools.cache_kizz_phoneme_teacher_posteriors import (
    cache_manifest,
    compact_log_posteriors,
    compact_vocabulary,
    load_cache,
    processor_vocab_hash,
    teacher_timing_metadata,
)


class FakeTokenizer:
    def __init__(self):
        self._vocab = {"<pad>": 0, "k": 1, "ɪ": 2, "z": 3, "ə": 4, "n": 5, "t": 6, "ɹ": 7, "oʊ": 8, "l": 9, "d": 10, "s": 11, "ð": 12, "h": 13, "p": 14, "ɐ": 15, "tʃ": 16, "æ": 17, "ɚ": 18, "<unk>": 19}
        self.pad_token_id = 0
        self.unk_token_id = 19
        self.bos_token_id = None
        self.eos_token_id = None

    def get_vocab(self):
        return dict(self._vocab)

    def convert_tokens_to_ids(self, token):
        return self._vocab.get(token, self.unk_token_id)


class FakeProcessor:
    def __call__(self, waveform, sampling_rate, return_tensors):
        self.last = (len(waveform), sampling_rate, return_tensors)
        return SimpleNamespace(input_values=np.asarray(waveform, dtype=np.float32)[None, :])


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(conv_kernel=(10, 3), conv_stride=(5, 2), inputs_to_logits_ratio=10)

    def __call__(self, **kwargs):
        length = int(kwargs["input_values"].shape[-1])
        frames = max(1, length // 10)
        logits = np.arange(frames * 20, dtype=np.float32).reshape(1, frames, 20)
        return SimpleNamespace(logits=logits)


class CacheKizzPhonemeTeacherPosteriorsTests(unittest.TestCase):
    def test_compact_mapping_preserves_mass_and_repeated_phone_path(self):
        tokenizer = FakeTokenizer()
        vocabulary = compact_vocabulary(tokenizer, phrase_id="kizz-control")
        self.assertEqual(vocabulary["canonical_path"][0], vocabulary["canonical_path"][3])
        self.assertNotEqual(vocabulary["canonical_path"][0], vocabulary["canonical_path"][1])
        logits = np.zeros((2, 20), dtype=np.float64)
        compact = compact_log_posteriors(logits, vocabulary)
        self.assertTrue(np.allclose(np.exp(compact).sum(axis=1), 1.0, atol=1e-6))
        self.assertAlmostEqual(float(np.exp(compact[0, vocabulary["other_compact_id"]])), 1 / 20)
        self.assertAlmostEqual(float(np.exp(compact[0, 0])), 1 / 20)
        self.assertEqual(vocabulary["tokens"][0], "<blank>")
        self.assertEqual(vocabulary["tokens"][-1], "OTHER")
        self.assertEqual(vocabulary["teacher_token_ids"]["k"], 1)

    def test_timing_is_derived_without_offset(self):
        timing = teacher_timing_metadata(FakeModel())
        self.assertEqual(timing["frame_stride_samples"], 10)
        self.assertEqual(timing["receptive_field_samples"], 20)
        self.assertFalse(timing["arbitrary_offset_applied"])
        self.assertAlmostEqual(timing["frame_center_seconds"], 19 / 32000)

    def test_cache_records_hashes_and_rejects_stale_source(self):
        try:
            import soundfile as sf
        except ImportError:
            self.skipTest("optional audio/model test dependencies unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "example.wav"
            sf.write(audio, np.zeros(100, dtype=np.float32), 16000)
            source_hash = __import__("hashlib").sha256(audio.read_bytes()).hexdigest()
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"examples": [{"path": str(audio), "audio_sha256": source_hash, "source_id": "x", "label": 1}]}))
            prefix = root / "cache"
            metadata = cache_manifest(manifest, prefix, model=FakeModel(), processor=FakeProcessor(), tokenizer=FakeTokenizer(), device="cpu", phrase_id="kizz-control")
            loaded, arrays = load_cache(prefix, expected_processor_vocab_sha256=processor_vocab_hash(FakeTokenizer()), expected_source_audio_sha256=[source_hash])
            self.assertEqual(loaded["model"]["revision"], "ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4")
            self.assertEqual(arrays["offsets"].tolist(), [0, arrays["log_posteriors"].shape[0]])
            with self.assertRaises(ValueError):
                load_cache(prefix, expected_source_audio_sha256=["stale"])


if __name__ == "__main__":
    unittest.main()
