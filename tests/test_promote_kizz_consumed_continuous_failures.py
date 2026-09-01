import json
import tempfile
import unittest
from pathlib import Path

from tools.mine_kizz_librispeech_hard_negatives import _binding, sha256_file
from tools.promote_kizz_consumed_continuous_failures import promote


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


class PromoteConsumedFailuresTests(unittest.TestCase):
    def test_promotes_only_accepted_sources_and_revalidates_lock_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = []
            rows = []
            for index in range(2):
                path = root / f"{index}.wav"
                path.write_bytes(bytes([index + 1]))
                audio.append(path)
                rows.append(
                    {
                        "source_id": f"source-{index}",
                        "path": str(path),
                        "sha256": sha256_file(path),
                        "audio_sha256": sha256_file(path),
                        "duration_seconds": 1.0,
                        "category": "speech",
                        "source": "fixture",
                        "speaker_id": f"speaker-{index}",
                        "session_id": f"session-{index}",
                        "ancestry_id": f"speaker-{index}",
                        "split": "test",
                    }
                )
            lock = root / "lock.json"
            write_json(
                lock,
                {"schema_version": 2, "locked_before_scoring": True, "examples": rows},
            )
            report = root / "report.json"
            write_json(
                report,
                {
                    "schema_version": 1,
                    "kind": "kizz_control_int8_continuous_negative_cascade_v1",
                    "shard": {"complete": True, "count": 1, "index": 0},
                    "bindings": {"locked_manifest": _binding(lock)},
                    "files": [
                        {**rows[0], "accepted_false_wakes": 1},
                        {**rows[1], "accepted_false_wakes": 0},
                    ],
                },
            )
            output = root / "promoted.json"
            result = promote(
                lock,
                [report],
                output,
                source_group="consumed_fresh_failure",
            )
            payload = json.loads(output.read_text())
            self.assertEqual(result["source_files"], 1)
            self.assertEqual(payload["examples"][0]["source_id"], "source-0")
            self.assertEqual(payload["examples"][0]["split"], "train")
            self.assertTrue(payload["examples"][0]["training_eligible"])
            self.assertEqual(
                payload["examples"][0]["source_group"],
                "consumed_fresh_failure",
            )

            all_output = root / "promoted-all.json"
            from unittest import mock
            import tools.promote_kizz_consumed_continuous_failures as tool

            with mock.patch.object(tool, "_binding", wraps=_binding) as bind:
                all_result = promote(
                    lock,
                    [report],
                    all_output,
                    source_group="consumed_broad_speech",
                    selection="all",
                )
            report_binding_calls = [
                call
                for call in bind.call_args_list
                if call.args and Path(call.args[0]).resolve() == report.resolve()
            ]
            self.assertEqual(len(report_binding_calls), 1)
            all_payload = json.loads(all_output.read_text())
            self.assertEqual(all_result["source_files"], 2)
            self.assertFalse(
                all_payload["selection_policy"]["accepted_false_wakes_required"]
            )
            self.assertEqual(
                [row["speaker_id"] for row in all_payload["examples"]],
                ["speaker-0", "speaker-1"],
            )

            rows[0]["duration_seconds"] = 2.0
            write_json(lock, {"schema_version": 2, "locked_before_scoring": True, "examples": rows})
            with self.assertRaisesRegex(ValueError, "hash drift"):
                promote(lock, [report], root / "rejected.json")


if __name__ == "__main__":
    unittest.main()
