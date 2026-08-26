import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.audit_kizz_control_source_pronunciations import build_report, has_canonical_prefix


class AuditKizzControlSourcePronunciationsTests(unittest.TestCase):
    def test_accepts_only_canonical_initial_phone_sequence(self):
        self.assertTrue(has_canonical_prefix("k ɪ z k ə n t ɹ o w l"))
        for value in (
            "k ɪ s k ə n t ɹ o w l",
            "k ɪ d z k ə n t ɹ o w l",
            "h ɪ z k ə n t ɹ o w l",
            "ɪ z k ə n t ɹ o w l",
        ):
            self.assertFalse(has_canonical_prefix(value), value)

    def test_all_split_gate_audits_every_requested_positive(self):
        class Recognizer:
            def recognize(self, path, language):
                return "k ɪ z k ə n t ɹ o w l" if "good" in path else "k ɪ d z k ə n t ɹ o w l"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = root / "good.wav"
            bad = root / "bad.wav"
            good.write_bytes(b"good")
            bad.write_bytes(b"bad")
            manifest = root / "manifest.json"
            from tools.audit_kizz_control_source_pronunciations import sha256_file
            manifest.write_text(json.dumps({"examples": [
                {"source_id": "good", "audio_sha256": sha256_file(good), "path": str(good), "label": 1, "split": "train", "provider": "kokoro", "voice": "a", "render_text": "Kizz Control"},
                {"source_id": "bad", "audio_sha256": sha256_file(bad), "path": str(bad), "label": 1, "split": "validation", "provider": "elevenlabs", "voice": "b", "render_text": "Kizz Control"},
                {"source_id": "ignored", "label": 1, "split": "test", "provider": "deepgram"},
            ]}))
            output = root / "report.json"
            fake_app = types.ModuleType("allosaurus.app")
            fake_app.__file__ = str(root / "fake_allosaurus" / "app.py")
            fake_app.read_recognizer = lambda model: Recognizer()
            fake_package = types.ModuleType("allosaurus")
            fake_package.app = fake_app
            with patch.dict(
                sys.modules,
                {"allosaurus": fake_package, "allosaurus.app": fake_app},
            ), patch(
                "tools.audit_kizz_control_source_pronunciations.sha256_tree",
                return_value="m" * 64,
            ), patch(
                "importlib.metadata.version", return_value="1.0.2"
            ):
                report = build_report(
                    manifest,
                    output,
                    splits=("train", "validation"),
                    gate_mode="all",
                )
        self.assertFalse(report["qualified"])
        self.assertEqual(report["counts"]["audited"], 2)
        self.assertEqual(report["counts"]["gated_rejected"], 1)
        self.assertEqual(report["scope"]["splits"], ["train", "validation"])


if __name__ == "__main__":
    unittest.main()
