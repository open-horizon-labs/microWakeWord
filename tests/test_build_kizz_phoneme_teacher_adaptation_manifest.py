import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.build_kizz_phoneme_teacher_adaptation_manifest import build_manifest


class AdaptationManifestTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.paths = {}
        positives = []
        for provider in ("assemblyai", "deepgram", "elevenlabs", "kokoro"):
            positives.append(self.row(f"pos-{provider}-a", 1, f"{provider}_synthetic", provider, speaker=f"{provider}-a"))
            positives.append(self.row(f"pos-{provider}-b", 1, f"{provider}_synthetic", provider, speaker=f"{provider}-b"))
        negatives = [
            self.row("public-a", 0, "public_speech", None, speaker="public-a"),
            self.row("public-b", 0, "public_speech", None, speaker="public-b"),
            self.row("phonetic-a", 0, "kizz_control_phonetic_collision", None, speaker="phonetic-a"),
            self.row("phonetic-b", 0, "kizz_control_phonetic_collision", None, speaker="phonetic-b"),
            self.row("device-collision-a", 0, "device_collision", None, speaker="collision-a"),
            self.row("device-collision-b", 0, "device_collision", None, speaker="collision-b"),
        ]
        materialized_device = self.row(
            "materialized-device", 1, "device_channel_positive", "assemblyai"
        )
        corpus = positives + negatives + [
            materialized_device,
            self.row("heldout", 1, "assemblyai_synthetic", "assemblyai", split="validation"),
        ]
        self.write("corpus.json", {"examples": corpus})
        self.write("teacher-manifest.json", {"examples": corpus})
        captures = []
        for provider in ("assemblyai", "deepgram", "elevenlabs", "kokoro"):
            for variant in range(4):
                audio_path = self.root / "audio" / f"train-{provider}-{variant}.wav"
                audio_path.parent.mkdir(exist_ok=True)
                audio_path.write_bytes(f"device-{provider}-{variant}".encode())
                captures.append({
                    "capture_id": f"train-{provider}-{variant}", "truth": "positive", "split": "train",
                    "path": f"audio/train-{provider}-{variant}.wav", "sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
                    "speaker_id": f"replay-{provider}-{variant}", "session_id": f"session-{provider}-{variant}",
                    "device_id": "kizz-1", "device_profile": "stackchan",
                    "conditions": {"evidence_role": "teacher_adaptation_target_channel_positive", "source_provider": provider, "source_voice": f"train-{provider}-{variant}", "source_audio_sha256": f"source-{provider}-{variant}"},
                })
        self.write("device-training.json", {"captures": captures})
        validation_captures = []
        validation_counts = {"assemblyai": 2, "deepgram": 1, "elevenlabs": 3, "kokoro": 1}
        for provider, count in validation_counts.items():
            for variant in range(count):
                audio_path = self.root / "validation-audio" / f"validation-{provider}-{variant}.wav"
                audio_path.parent.mkdir(exist_ok=True)
                audio_path.write_bytes(f"validation-{provider}-{variant}".encode())
                validation_captures.append({
                    "capture_id": f"validation-{provider}-{variant}", "truth": "positive", "split": "validation",
                    "path": f"validation-audio/validation-{provider}-{variant}.wav", "sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
                    "speaker_id": f"validation-replay-{provider}-{variant}", "session_id": f"validation-session-{provider}-{variant}",
                    "device_id": "kizz-1", "device_profile": "stackchan",
                    "conditions": {"evidence_role": "teacher_adaptation_target_channel_validation_positive", "source_provider": provider, "source_voice": f"validation-{provider}-{variant}", "source_audio_sha256": f"validation-source-{provider}-{variant}"},
                })
        self.write("device-validation.json", {"captures": validation_captures})
        current = self.row("current-device", 1, "device", "assemblyai")
        current["conditions"] = {"source_provider": "assemblyai", "source_voice": "heldout"}
        self.write("current-evidence.json", {"examples": [current]})
        self.write("current-report.json", {"results": {"natural_positive": [{"source_id": "current-device", "audio_sha256": "hash-current-device", "accepted": True}]}})
        self.write("device-quality.json", {
            "kind": "kizz_control_teacher_adaptation_device_replay_quality",
            "qualified": True,
            "inputs": {
                "corpus_sha256": hashlib.sha256(self.paths["device-training.json"].read_bytes()).hexdigest(),
                "qualification_evidence_sha256": hashlib.sha256(self.paths["current-evidence.json"].read_bytes()).hexdigest(),
            },
            "counts": {
                "providers": {provider: 4 for provider in ("assemblyai", "deepgram", "elevenlabs", "kokoro")},
                "voices": {provider: [f"train-{provider}-{variant}" for variant in range(4)] for provider in ("assemblyai", "deepgram", "elevenlabs", "kokoro")},
            },
        })
        self.write("device-validation-quality.json", {
            "kind": "kizz_control_teacher_adaptation_device_replay_quality",
            "qualified": True,
            "inputs": {
                "corpus_sha256": hashlib.sha256(self.paths["device-validation.json"].read_bytes()).hexdigest(),
                "qualification_evidence_sha256": hashlib.sha256(self.paths["current-evidence.json"].read_bytes()).hexdigest(),
            },
            "counts": {
                "providers": validation_counts,
                "voices": {provider: [f"validation-{provider}-{variant}" for variant in range(count)] for provider, count in validation_counts.items()},
            },
        })
        self.write("continuous.json", {"examples": [{"source_id": "locked", "sha256": "hash-locked", "path": str(self.root / "locked.wav")} ]})
        self.write("false.json", {"examples": [{"source_id": "false", "audio_sha256": "hash-false", "path": str(self.root / "false.wav")} ]})

    def row(self, source_id, label, group, provider, split="train", speaker=None):
        return {"source_id": source_id, "path": str(self.root / f"{source_id}.wav"), "audio_sha256": f"hash-{source_id}", "label": label, "split": split, "source_group": group, "provider": provider, "speaker_id": speaker or source_id}

    def write(self, name, payload):
        path = self.root / name
        path.write_text(json.dumps(payload, sort_keys=True))
        self.paths[name] = path
        return path

    def build(self):
        return build_manifest(
            self.paths["corpus.json"], self.paths["teacher-manifest.json"],
            self.paths["device-training.json"],
            self.paths["device-quality.json"],
            self.paths["device-validation.json"],
            self.paths["device-validation-quality.json"],
            self.paths["current-evidence.json"], self.paths["current-report.json"],
            self.paths["continuous.json"], self.paths["false.json"],
        )

    def test_device_training_and_stable_output(self):
        report = self.build()
        self.assertIn("device-adaptation:train-assemblyai-0", {row["source_id"] for row in report["examples"]})
        self.assertEqual(report["counts"]["device_train"]["total"], 16)
        self.assertEqual(report["counts"]["device_validation"]["total"], 7)
        self.assertEqual(report["counts"]["device_validation"]["providers"]["elevenlabs"], 3)
        validation = {row["source_id"]: row for row in report["examples"] if row["source_id"].startswith("device-adaptation:validation-")}
        self.assertTrue(validation)
        self.assertTrue(all(row["split"] == "validation" for row in validation.values()))
        self.assertTrue(all(not row["training_eligible"] for row in report["examples"] if row["split"] == "validation"))
        self.assertTrue(all(row["training_eligible"] for row in report["examples"] if row["split"] == "train"))
        self.assertEqual(
            report["contract"]["adaptation_validation_policy"],
            "speaker_disjoint_train_inventory_plus_physical_device_validation_v1",
        )
        self.assertEqual(report["counts"]["source_groups"]["device_channel_positive"], 23)
        self.assertEqual(report["contract"]["materialized_device_rows_excluded"], 1)
        self.assertNotIn(
            "materialized-device", {row["source_id"] for row in report["examples"]}
        )
        self.assertEqual(set(report["counts"]["splits"]), {"train", "validation"})
        again = self.build()
        self.assertEqual(json.dumps(report, sort_keys=True), json.dumps(again, sort_keys=True))

    def test_leakage_rejection(self):
        corpus = json.loads(self.paths["corpus.json"].read_text())
        corpus["examples"][0]["audio_sha256"] = "hash-current-device"
        self.paths["corpus.json"].write_text(json.dumps(corpus))
        self.paths["teacher-manifest.json"].write_text(json.dumps(corpus))
        with self.assertRaisesRegex(ValueError, "overlaps excluded"):
            self.build()

    def test_provider_loss_is_rejected(self):
        corpus = json.loads(self.paths["corpus.json"].read_text())
        corpus["examples"] = [row for row in corpus["examples"] if row.get("provider") != "kokoro"]
        self.paths["corpus.json"].write_text(json.dumps(corpus))
        self.paths["teacher-manifest.json"].write_text(json.dumps(corpus))
        with self.assertRaisesRegex(ValueError, "provider"):
            self.build()

    def test_validation_rows_are_not_rewritten(self):
        report = self.build()
        rows = [row for row in report["examples"] if row["source_id"].startswith("device-adaptation:validation-")]
        self.assertEqual({row["source_split"] for row in rows}, {"validation"})
        self.assertTrue(all(row["split"] == "validation" for row in rows))
        self.assertTrue(all(row["training_eligible"] is False for row in rows))

    def test_device_train_validation_overlap_is_rejected(self):
        payload = json.loads(self.paths["device-validation.json"].read_text())
        payload["captures"][0]["conditions"]["source_audio_sha256"] = "source-assemblyai-0"
        self.paths["device-validation.json"].write_text(json.dumps(payload))
        quality = json.loads(self.paths["device-validation-quality.json"].read_text())
        quality["inputs"]["corpus_sha256"] = hashlib.sha256(self.paths["device-validation.json"].read_bytes()).hexdigest()
        self.paths["device-validation-quality.json"].write_text(json.dumps(quality))
        with self.assertRaisesRegex(ValueError, "device train/validation overlap"):
            self.build()


if __name__ == "__main__":
    unittest.main()
