import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.curate_kizz_control_c1_pronunciations import curate


PROVIDERS = ("assemblyai", "deepgram", "elevenlabs", "kokoro")
FAILURES = {"assemblyai": 1, "deepgram": 1, "elevenlabs": 0, "kokoro": 2}


class CurateKizzControlC1PronunciationsTests(unittest.TestCase):
    def test_quarantines_rejections_and_repairs_four_reserves_same_voice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            results = []
            failed_ids = set()
            replacement_ids = set()
            for provider in PROVIDERS:
                for index in range(6):
                    source_id = f"{provider}-reserved-{index}"
                    accepted = index >= FAILURES[provider]
                    if not accepted:
                        failed_ids.add(source_id)
                    row = {
                        "source_id": source_id,
                        "audio_sha256": hashlib.sha256(source_id.encode()).hexdigest(),
                        "path": str(root / f"{source_id}.wav"),
                        "label": 1,
                        "split": "test",
                        "provider": provider,
                        "voice": f"voice-{index % 2}",
                        "render_text": "Kizz Control",
                        "training_eligible": False,
                        "reserved_evidence_role": "target_channel_positive",
                    }
                    rows.append(row)
                    results.append(
                        {
                            "source_id": source_id,
                            "audio_sha256": row["audio_sha256"],
                            "accepted": accepted,
                            "reserved": True,
                            "phones": "k ɪ z k" if accepted else "k ɪ s k",
                        }
                    )
                for voice_index in range(2):
                    source_id = f"{provider}-candidate-{voice_index}"
                    replacement_ids.add(source_id)
                    row = {
                        "source_id": source_id,
                        "audio_sha256": hashlib.sha256(source_id.encode()).hexdigest(),
                        "path": str(root / f"{source_id}.wav"),
                        "label": 1,
                        "split": "test",
                        "provider": provider,
                        "voice": f"voice-{voice_index}",
                        "render_text": "Kizz Control",
                        "training_eligible": True,
                    }
                    rows.append(row)
                    results.append(
                        {
                            "source_id": source_id,
                            "audio_sha256": row["audio_sha256"],
                            "accepted": True,
                            "reserved": False,
                            "phones": "k ɪ z k",
                        }
                    )
            macos_id = "macos-train"
            macos_row = {
                "source_id": macos_id,
                "audio_sha256": hashlib.sha256(macos_id.encode()).hexdigest(),
                "path": str(root / "macos.wav"),
                "label": 1,
                "split": "train",
                "provider": "macos-say",
                "voice": "macos-voice",
                "render_text": "Kizz Control",
                "training_eligible": True,
            }
            rows.append(macos_row)
            results.append(
                {
                    "source_id": macos_id,
                    "audio_sha256": macos_row["audio_sha256"],
                    "accepted": True,
                    "reserved": False,
                    "phones": "k ɪ z k",
                }
            )
            source = root / "source.json"
            source.write_text(json.dumps({"schema_version": 2, "examples": rows}))
            audit = root / "audit.json"
            audit.write_text(
                json.dumps(
                    {
                        "gate_scope": "independent_source_pronunciation_qc",
                        "source_manifest_sha256": hashlib.sha256(
                            source.read_bytes()
                        ).hexdigest(),
                        "scope": {
                            "gate_mode": "all",
                            "splits": ["train", "validation", "test"],
                        },
                        "model": {"name": "eng2102"},
                        "results": results,
                    }
                )
            )
            with mock.patch(
                "tools.curate_kizz_control_c1_pronunciations.corpus_mix_report",
                return_value={"qualified": True, "violations": []},
            ):
                curated, quarantine, report = curate(
                    source,
                    audit,
                    root / "curated.json",
                    root / "quarantine.json",
                    root / "report.json",
                    excluded_positive_providers=("macos-say",),
                )

        by_id = {row["source_id"]: row for row in curated["examples"]}
        self.assertEqual(quarantine["counts"]["quarantined"], 4)
        self.assertEqual(quarantine["counts"]["reserved_quarantined"], 4)
        self.assertEqual(report["counts"]["reserved_replacements"], 4)
        self.assertTrue(report["qualified"])
        self.assertTrue(
            all(
                by_id[source_id].get("reserved_evidence_role") is None
                and by_id[source_id]["training_eligible"] is False
                and by_id[source_id]["exclusion_reason"]
                == "pronunciation_qc_rejected"
                for source_id in failed_ids
            )
        )
        self.assertFalse(by_id[macos_id]["training_eligible"])
        self.assertEqual(
            by_id[macos_id]["exclusion_reason"], "source_audit_only_provider"
        )
        self.assertEqual(report["counts"]["source_audit_only_exclusions"], 1)
        selected_replacements = {
            item["replacement_source_id"] for item in report["replacements"]
        }
        self.assertTrue(selected_replacements <= replacement_ids)
        self.assertTrue(all(item["same_voice"] for item in report["replacements"]))
        self.assertTrue(
            all(
                by_id[source_id]["training_eligible"] is False
                and by_id[source_id]["reserved_evidence_role"]
                == "target_channel_positive"
                for source_id in selected_replacements
            )
        )


if __name__ == "__main__":
    unittest.main()
