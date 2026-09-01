import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.prepare_kizz_reserved_multisource_qualification_inventory import (
    prepare,
    sha256_file,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


class InventoryFactory:
    def __init__(self, root: Path):
        self.root = root

    def row(
        self,
        provider: str,
        voice: str,
        index: int,
        *,
        label: int = 1,
        split: str = "test",
        training_eligible: bool = False,
        score: float = 0.0,
    ) -> dict:
        audio = f"audio:{provider}:{voice}:{index}".encode()
        path = self.root / "audio" / f"{provider}-{voice}-{index}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(audio)
        audio_hash = hashlib.sha256(audio).hexdigest()
        return {
            "source_id": f"source:{provider}:{voice}:{index}",
            "path": str(path),
            "audio_sha256": audio_hash,
            "provider": provider,
            "voice": voice,
            "voice_id": f"tts:{provider}:{voice}",
            "label": label,
            "split": split,
            "training_eligible": training_eligible,
            "model_score": score,
        }

    def manifest(self, name: str, rows: list[dict]) -> Path:
        path = self.root / f"{name}.json"
        _write_json(path, {"schema_version": 1, "examples": rows})
        return path


class PrepareReservedMultisourceInventoryTests(unittest.TestCase):
    def test_builds_balanced_locked_inventory_with_exact_bindings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            factory = InventoryFactory(root)
            rows = [
                factory.row(provider, voice, index)
                for provider in ("elevenlabs", "deepgram", "assemblyai")
                for voice in ("a", "b", "c")
                for index in range(2)
            ]
            consumed = rows[0]
            unqualified = factory.row("elevenlabs", "z", 99)
            macos = factory.row("macos-say", "Alex", 1)
            source_one = factory.manifest("source-one", rows[:9] + [unqualified])
            source_two = factory.manifest("source-two", rows[9:] + [macos])
            qualified = factory.manifest("qualified", rows + [macos])
            exclusion = root / "excluded.json"
            _write_json(
                exclusion,
                {
                    "captures": [
                        {
                            "sha256": "f" * 64,
                            "conditions": {
                                "source_audio_sha256": consumed["audio_sha256"]
                            },
                        }
                    ]
                },
            )
            output = root / "inventory.json"
            payload = prepare(
                [source_two, source_one],
                [exclusion],
                qualified,
                output,
                count=12,
                minimum_providers=3,
                minimum_voices=8,
            )

            self.assertEqual(payload["purpose"], "fresh_target_channel_positive_candidate_inventory")
            self.assertTrue(payload["locked_before_scoring"])
            self.assertFalse(payload["training_eligible"])
            self.assertEqual(payload["counts"]["selected"], 12)
            self.assertEqual(set(payload["provider_counts"]), {"assemblyai", "deepgram", "elevenlabs"})
            self.assertLessEqual(
                max(payload["provider_counts"].values()) - min(payload["provider_counts"].values()),
                1,
            )
            self.assertGreaterEqual(payload["counts"]["voices"], 8)
            self.assertNotIn(consumed["audio_sha256"], {row["audio_sha256"] for row in payload["examples"]})
            self.assertNotIn("macos-say", payload["provider_counts"])
            self.assertNotIn(unqualified["source_id"], {row["source_id"] for row in payload["examples"]})
            self.assertEqual(
                [row["candidate_inventory_selection_index"] for row in payload["examples"]],
                list(range(12)),
            )
            for row in payload["examples"]:
                self.assertEqual(row["reserved_evidence_role"], "target_channel_positive")
                self.assertEqual(row["evidence_status"], "reserved")
                self.assertTrue(row["locked_before_scoring"])
                self.assertFalse(row["training_eligible"])
            persisted = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(persisted, payload)
            source_bindings = payload["inputs"]["source_manifests"]
            self.assertEqual(
                [item["path"] for item in source_bindings],
                sorted([str(source_one.resolve()), str(source_two.resolve())]),
            )
            for binding in source_bindings + payload["inputs"]["exclude_manifests"]:
                path = Path(binding["path"])
                self.assertEqual(binding["sha256"], sha256_file(path))
                self.assertEqual(binding["bytes"], path.stat().st_size)
            self.assertFalse(payload["selection_policy"]["model_scores_read_or_used"])

    def test_selection_ignores_model_scores_and_source_argument_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            factory = InventoryFactory(root)
            rows = [
                factory.row(provider, voice, index, score=float(index))
                for provider in ("a", "b")
                for voice in ("one", "two")
                for index in range(3)
            ]
            first = factory.manifest("first", rows[:6])
            second = factory.manifest("second", rows[6:])
            qualified = factory.manifest("qualified", rows)
            payload_a = prepare(
                [first, second], [], qualified, root / "a.json",
                count=6, minimum_providers=2, minimum_voices=4,
            )
            for row in rows:
                row["model_score"] = 1000.0 - row["model_score"]
            first_changed = factory.manifest("first-changed", rows[:6])
            second_changed = factory.manifest("second-changed", rows[6:])
            qualified_changed = factory.manifest("qualified-changed", rows)
            payload_b = prepare(
                [second_changed, first_changed], [], qualified_changed, root / "b.json",
                count=6, minimum_providers=2, minimum_voices=4,
            )
            self.assertEqual(
                [row["source_id"] for row in payload_a["examples"]],
                [row["source_id"] for row in payload_b["examples"]],
            )

    def test_excludes_parent_hash_recursively_from_examples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            factory = InventoryFactory(root)
            rows = [factory.row("provider", f"v{i}", i) for i in range(3)]
            rows[0]["parent_source_audio_sha256"] = "a" * 64
            source = factory.manifest("source", rows)
            qualified = factory.manifest("qualified", rows)
            exclusion = root / "exclusion.json"
            _write_json(
                exclusion,
                {"examples": [{"nested": {"parent_source_audio_sha256": "a" * 64}}]},
            )
            payload = prepare(
                [source], [exclusion], qualified, root / "output.json",
                count=2, minimum_providers=1, minimum_voices=2,
            )
            self.assertNotIn(rows[0]["source_id"], {row["source_id"] for row in payload["examples"]})

    def test_rejects_audio_hash_drift_without_writing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            factory = InventoryFactory(root)
            row = factory.row("provider", "voice", 1)
            source = factory.manifest("source", [row])
            qualified = factory.manifest("qualified", [row])
            Path(row["path"]).write_bytes(b"drift")
            output = root / "output.json"
            with self.assertRaisesRegex(ValueError, "source audio hash drift"):
                prepare(
                    [source], [], qualified, output,
                    count=1, minimum_providers=1, minimum_voices=1,
                )
            self.assertFalse(output.exists())

    def test_fails_when_minimum_diversity_or_count_is_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            factory = InventoryFactory(root)
            rows = [factory.row("provider", f"v{i}", i) for i in range(3)]
            source = factory.manifest("source", rows)
            qualified = factory.manifest("qualified", rows)
            with self.assertRaisesRegex(ValueError, "qualified providers"):
                prepare(
                    [source], [], qualified, root / "providers.json",
                    count=2, minimum_providers=2, minimum_voices=2,
                )
            with self.assertRaisesRegex(ValueError, "qualified unconsumed candidates"):
                prepare(
                    [source], [], qualified, root / "count.json",
                    count=4, minimum_providers=1, minimum_voices=1,
                )

    def test_output_is_fail_if_exists_and_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            factory = InventoryFactory(root)
            row = factory.row("provider", "voice", 1)
            source = factory.manifest("source", [row])
            qualified = factory.manifest("qualified", [row])
            output = root / "output.json"
            output.write_bytes(b"do not replace")
            with self.assertRaises(FileExistsError):
                prepare(
                    [source], [], qualified, output,
                    count=1, minimum_providers=1, minimum_voices=1,
                )
            self.assertEqual(output.read_bytes(), b"do not replace")

    def test_can_reserve_training_eligible_source_only_when_unconsumed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            factory = InventoryFactory(root)
            consumed = factory.row("provider", "voice-a", 1, training_eligible=True)
            fresh = factory.row("provider", "voice-b", 2, training_eligible=True)
            source = factory.manifest("source", [consumed, fresh])
            qualified = factory.manifest("qualified", [consumed, fresh])
            exclusion = factory.manifest("exclusion", [consumed])

            with self.assertRaisesRegex(ValueError, "qualified unconsumed candidates"):
                prepare(
                    [source], [exclusion], qualified, root / "default.json",
                    count=1, minimum_providers=1, minimum_voices=1,
                )

            payload = prepare(
                [source], [exclusion], qualified, root / "allowed.json",
                count=1, minimum_providers=1, minimum_voices=1,
                allow_unconsumed_training_eligible=True,
            )
            self.assertEqual(payload["examples"][0]["source_id"], fresh["source_id"])
            self.assertFalse(payload["examples"][0]["training_eligible"])
            self.assertEqual(
                payload["selection_policy"]["candidate_requirements"]["source_training_eligible"],
                "either_but_not_present_in_any_exclusion_manifest",
            )


if __name__ == "__main__":
    unittest.main()
