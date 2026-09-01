import csv
import hashlib
import json
import tempfile
import unittest
import wave
from pathlib import Path

from tools.compose_kizz_canonical_v3_manifest import compose


class ComposeKizzCanonicalV3ManifestTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.counter = 1

    def audio(self, name, *, seconds=1.0):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        sample_count = int(16_000 * seconds)
        value = self.counter
        self.counter += 1
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16_000)
            output.writeframes(
                int(value).to_bytes(2, "little", signed=True) * sample_count
            )
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def write_inputs(self):
        synthesis_rows = []
        for index, text in enumerate(
            ("Hi-Fi Kizz", "Hi Phi Kizz", "Hi-Fi Kids", "High Five Kizz")
        ):
            audio, digest = self.audio(f"synthesis/{index}.wav")
            synthesis_rows.append(
                {
                    "path": str(audio),
                    "text": text,
                    "provider": "deepgram",
                    "voice": "voice-train",
                    "split": "train",
                    "duration": 1.0,
                    "output_hash": digest,
                    "source_hash": f"source-{index}",
                }
            )
        synthesis = self.root / "synthesis.json"
        synthesis.write_text(
            json.dumps({"schema_version": 1, "examples": synthesis_rows})
        )

        overlay_rows = []
        for index, base in enumerate(synthesis_rows):
            audio, digest = self.audio(f"overlay/{index}.wav")
            overlay_rows.append(
                {
                    "path": str(audio),
                    "split": base["split"],
                    "duration_s": 1.0,
                    "derived_from_positive": base["path"],
                    "positive_hash": base["output_hash"],
                    "background_hash": f"background-{index}",
                    "background_category": "noise",
                    "snr_db": 6.0,
                    "output_hash": digest,
                }
            )
        overlays = self.root / "overlays.json"
        overlays.write_text(json.dumps({"schema_version": 1, "examples": overlay_rows}))

        captures = []
        for capture_id, source, phrase, pronunciation in (
            ("device-exact", "synthetic_playback", "Hi-Fi Kizz", "hi_fi_kizz"),
            ("device-collision", "synthetic_playback", "Hiffy Kizz", "hiffy_kizz"),
            ("human-exact", "human", "HiPhi Kizz", "hi_fi"),
            ("human-repeated", "human", "Hi-Fi Kizz", "hi_fi_repeated"),
        ):
            audio, digest = self.audio(f"device/{capture_id}.wav")
            captures.append(
                {
                    "capture_id": capture_id,
                    "truth": "positive",
                    "source": source,
                    "phrase": phrase,
                    "pronunciation": pronunciation,
                    "split": "train",
                    "speaker_id": capture_id,
                    "session_id": f"session-{capture_id}",
                    "samples": 16_000,
                    "path": str(audio.relative_to(self.root)),
                    "sha256": digest,
                    "device_profile": "fixture",
                    "firmware_sha": "abc123",
                    "conditions": {"source_wav_sha256": f"source-{capture_id}"},
                }
            )
        device = self.root / "device-corpus.json"
        device.write_text(json.dumps({"schema_version": 2, "captures": captures}))

        public_audio, public_hash = self.audio("public/speech.wav", seconds=2.0)
        musan_audio, musan_hash = self.audio("public/musan.wav", seconds=2.0)
        short_audio, short_hash = self.audio("public/short.wav", seconds=0.1)
        public = self.root / "public.csv"
        with public.open("w", newline="") as output:
            writer = csv.DictWriter(
                output,
                fieldnames=(
                    "path",
                    "sha256",
                    "duration_s",
                    "category",
                    "split",
                    "speaker_or_session",
                    "source",
                    "license",
                    "provenance",
                    "eligible_for_target",
                ),
            )
            writer.writeheader()
            writer.writerow(
                {
                    "path": public_audio,
                    "sha256": public_hash,
                    "duration_s": 2.0,
                    "category": "connected_speech",
                    "split": "train",
                    "speaker_or_session": "speaker-1",
                    "source": "fixture",
                    "license": "CC0",
                    "provenance": "fixture",
                    "eligible_for_target": "True",
                }
            )
            writer.writerow(
                {
                    "path": musan_audio,
                    "sha256": musan_hash,
                    "duration_s": 2.0,
                    "category": "connected_speech",
                    "split": "test",
                    "speaker_or_session": "librivox",
                    "source": "MUSAN",
                    "license": "CC-BY-4.0",
                    "provenance": "fixture",
                    "eligible_for_target": "True",
                }
            )
            writer.writerow(
                {
                    "path": short_audio,
                    "sha256": short_hash,
                    "duration_s": 0.1,
                    "category": "noise",
                    "split": "train",
                    "speaker_or_session": "free-sound",
                    "source": "MUSAN",
                    "license": "CC-BY-4.0",
                    "provenance": "fixture",
                    "eligible_for_target": "True",
                }
            )

        false_audio, false_hash = self.audio("false/observations/false.wav")
        false_metadata = self.root / "false/observations/false.json"
        false_metadata.write_text(
            json.dumps(
                {
                    "samples": 16_000,
                    "device_id": "kizz-1",
                    "firmware_sha": "def456",
                }
            )
        )
        false_wakes = self.root / "false/manifest.json"
        false_wakes.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "training_eligible": False,
                    "observations": [
                        {
                            "observation_id": "false-1",
                            "audio_path": str(
                                false_audio.relative_to(false_wakes.parent)
                            ),
                            "metadata_path": str(
                                false_metadata.relative_to(false_wakes.parent)
                            ),
                            "audio_sha256": false_hash,
                            "human_review_basis": "fixture review",
                            "review": {"reviewer": "fixture"},
                        }
                    ],
                }
            )
        )
        return synthesis, overlays, device, public, false_wakes

    def run_compose(self):
        synthesis, overlays, device, public, false_wakes = self.write_inputs()
        output = self.root / "output"
        compose(
            synthesis_manifest=synthesis,
            overlay_manifest=overlays,
            device_corpus=device,
            public_negative_manifest=public,
            false_wake_manifest=false_wakes,
            output=output,
        )
        return output

    def test_only_declared_phone_equivalent_texts_are_positive(self):
        output = self.run_compose()
        manifest = json.loads((output / "manifest.json").read_text())
        rendered = {
            item.get("render_text"): (item["label"], item["semantic_label"])
            for item in manifest["examples"]
            if item["source_group"] != "public_speech"
        }
        self.assertEqual(rendered["Hi-Fi Kizz"], (1, "canonical_exact"))
        self.assertEqual(rendered["Hi Phi Kizz"], (1, "canonical_exact"))
        self.assertEqual(rendered["Hi-Fi Kids"], (0, "kids_collision"))
        self.assertEqual(rendered["High Five Kizz"], (0, "high_five_collision"))
        self.assertFalse(
            any(
                item["label"] == 1 and item["semantic_label"] != "canonical_exact"
                for item in manifest["examples"]
            )
        )

    def test_overlays_inherit_parent_semantics_and_ancestry(self):
        output = self.run_compose()
        examples = json.loads((output / "manifest.json").read_text())["examples"]
        direct = next(
            item
            for item in examples
            if item.get("render_text") == "Hi-Fi Kids"
            and item["source_group"] == "phonetic_collision"
        )
        overlay = next(
            item
            for item in examples
            if item.get("render_text") == "Hi-Fi Kids"
            and item["source_group"] == "collision_overlay"
        )
        self.assertEqual(overlay["label"], 0)
        self.assertEqual(overlay["ancestry_id"], direct["ancestry_id"])

    def test_household_captures_are_locked_anchors_not_training_examples(self):
        output = self.run_compose()
        manifest = json.loads((output / "manifest.json").read_text())
        positive_anchors = json.loads(
            (output / "locked-positive-anchors.json").read_text()
        )
        false_anchors = json.loads(
            (output / "locked-false-wake-anchors.json").read_text()
        )
        self.assertFalse(
            any(
                item["source_id"] == "device-capture:human-exact"
                for item in manifest["examples"]
            )
        )
        self.assertEqual(len(positive_anchors["examples"]), 1)
        self.assertTrue(positive_anchors["examples"][0]["locked_deployment_anchor"])
        self.assertFalse(positive_anchors["examples"][0]["training_eligible"])
        self.assertEqual(len(false_anchors["examples"]), 1)
        self.assertEqual(false_anchors["examples"][0]["role"], "anchor")
        self.assertFalse(false_anchors["examples"][0]["training_eligible"])
        report = json.loads((output / "composition-report.json").read_text())
        self.assertEqual(
            report["device_exclusions"],
            {"human_repeated_opportunity_geometry_unreviewed": 1},
        )
        self.assertEqual(
            report["public_negative_exclusions"], {"shorter_than_200ms": 1}
        )

    def test_musan_is_training_only_when_identity_metadata_is_not_reliable(self):
        output = self.run_compose()
        examples = json.loads((output / "manifest.json").read_text())["examples"]
        musan = [item for item in examples if item.get("source_collection") == "MUSAN"]
        self.assertEqual(len(musan), 1)
        self.assertEqual(musan[0]["source_split"], "test")
        self.assertEqual(musan[0]["split"], "train")


if __name__ == "__main__":
    unittest.main()
