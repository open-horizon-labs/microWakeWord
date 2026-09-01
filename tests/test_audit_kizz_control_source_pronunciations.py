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

    def test_training_eligible_gate_allows_explicit_quarantine_and_audits_macos(self):
        class Recognizer:
            def recognize(self, path, language):
                return (
                    "k ɪ d z k ə n t ɹ o w l"
                    if "bad" in path
                    else "k ɪ z k ə n t ɹ o w l"
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            from tools.audit_kizz_control_source_pronunciations import sha256_file

            for provider in ("assemblyai", "deepgram", "elevenlabs", "kokoro"):
                for index in range(6):
                    path = root / f"good-{provider}-{index}.wav"
                    path.write_bytes(f"{provider}-{index}".encode())
                    rows.append(
                        {
                            "source_id": f"{provider}-{index}",
                            "audio_sha256": sha256_file(path),
                            "path": str(path),
                            "label": 1,
                            "split": "test",
                            "provider": provider,
                            "voice": f"voice-{index % 2}",
                            "render_text": "Kizz Control",
                            "training_eligible": False,
                            "reserved_evidence_role": "target_channel_positive",
                        }
                    )
            good_macos = root / "good-macos.wav"
            bad_macos = root / "bad-macos.wav"
            good_macos.write_bytes(b"good-macos")
            bad_macos.write_bytes(b"bad-macos")
            for path, source_id, eligible in (
                (good_macos, "macos-good", True),
                (bad_macos, "macos-bad", False),
            ):
                rows.append(
                    {
                        "source_id": source_id,
                        "audio_sha256": sha256_file(path),
                        "path": str(path),
                        "label": 1,
                        "split": "train",
                        "provider": "macos-say",
                        "voice": source_id,
                        "render_text": "Kizz Control",
                        "training_eligible": eligible,
                    }
                )
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"examples": rows}))
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
            ), patch("importlib.metadata.version", return_value="1.0.2"):
                report = build_report(
                    manifest,
                    output,
                    splits=("train", "test"),
                    gate_mode="training_eligible",
                )
        self.assertTrue(report["qualified"])
        self.assertEqual(report["counts"]["audited"], 26)
        self.assertEqual(report["counts"]["gated"], 25)
        self.assertEqual(report["counts"]["gated_rejected"], 0)
        self.assertEqual(report["counts"]["training_eligible_rejected"], 0)

    def test_manifest_declared_single_provider_fresh_contract(self):
        class Recognizer:
            def recognize(self, path, language):
                return "k ɪ z k ə n t ɹ o w l"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            from tools.audit_kizz_control_source_pronunciations import sha256_file

            for index in range(24):
                path = root / f"fresh-{index}.wav"
                path.write_bytes(f"fresh-{index}".encode())
                rows.append(
                    {
                        "source_id": f"fresh-{index}",
                        "audio_sha256": sha256_file(path),
                        "path": str(path),
                        "label": 1,
                        "split": "test",
                        "provider": "kokoro",
                        "voice": f"voice-{index % 12}",
                        "render_text": "Kizz Control",
                        "training_eligible": False,
                        "reserved_evidence_role": "target_channel_positive",
                    }
                )
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "reserved_evidence_contract": {
                            "role": "target_channel_positive",
                            "locked_before_scoring": True,
                            "total_count": 24,
                            "providers": {
                                "kokoro": {"count": 24, "minimum_voices": 12}
                            },
                        },
                        "examples": rows,
                    }
                )
            )
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
            ), patch("importlib.metadata.version", return_value="1.0.2"):
                report = build_report(manifest, output)

        self.assertTrue(report["qualified"])
        self.assertEqual(
            report["reserved_contract_mode"],
            "manifest_declared_fresh_qualification_v1",
        )
        self.assertEqual(report["reserved_provider_contract"]["kokoro"]["count"], 24)
        self.assertEqual(len(report["reserved_provider_contract"]["kokoro"]["voices"]), 12)


if __name__ == "__main__":
    unittest.main()
