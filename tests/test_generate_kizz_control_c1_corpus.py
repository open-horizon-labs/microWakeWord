import unittest
from pathlib import Path

from tools.generate_kizz_control_c1_corpus import (
    DEFAULT_ENV_FILE,
    EXPECTED_PROVIDERS,
    RESERVED_REPLAY_VARIANTS,
    RESERVED_REPLAYS_PER_PROVIDER,
    RUNTIME_POSITIVE_PROVIDERS,
    Task,
    _deepgram_payload_text,
    _tasks,
    causal_negative_decision,
    corpus_mix_report,
    is_reserved_replay_task,
    retain_active_rows,
)


class GenerateKizzControlC1CorpusTest(unittest.TestCase):
    def test_default_env_file_is_home_relative(self):
        self.assertEqual(
            DEFAULT_ENV_FILE,
            Path.home() / ".config" / "open-horizon-labs" / "voice.env",
        )

    def realized_rows(self):
        rows = []
        for index, task in enumerate(_tasks()):
            reserved = is_reserved_replay_task(task)
            rows.append(
                {
                    "provider": task.provider,
                    "voice": task.voice,
                    "split": task.split,
                    "label": task.label,
                    "training_eligible": not reserved,
                    "descriptor_sha256": task.descriptor,
                    "audio_sha256": f"{index:064x}",
                    **(
                        {"reserved_evidence_role": "target_channel_positive"}
                        if reserved
                        else {}
                    ),
                }
            )
        return rows

    def test_plan_uses_every_provider_in_every_positive_split(self):
        report = corpus_mix_report(self.realized_rows())
        self.assertTrue(report["qualified"], report["violations"])
        for split in ("train", "validation", "test"):
            counts = report["splits"][split]["positive_provider_counts"]
            self.assertEqual(set(counts), set(EXPECTED_PROVIDERS))
            self.assertTrue(all(counts.values()))

    def test_realized_contract_rejects_nominal_but_unused_provider(self):
        rows = [
            row for row in self.realized_rows() if row["provider"] != "deepgram"
        ]
        report = corpus_mix_report(rows)
        self.assertFalse(report["qualified"])
        self.assertIn(
            "positive_provider_missing",
            {violation["reason"] for violation in report["violations"]},
        )

    def test_explicit_runtime_subset_can_keep_macos_as_audit_only(self):
        rows = [
            row for row in self.realized_rows() if row["provider"] != "macos-say"
        ]
        report = corpus_mix_report(
            rows,
            expected_positive_providers=sorted(RUNTIME_POSITIVE_PROVIDERS),
        )
        self.assertNotIn(
            "positive_provider_missing",
            {violation["reason"] for violation in report["violations"]},
        )
        self.assertEqual(
            set(report["expected_positive_providers"]),
            set(RUNTIME_POSITIVE_PROVIDERS),
        )

    def test_replay_holdout_realizes_provider_and_voice_diversity(self):
        report = corpus_mix_report(self.realized_rows())
        self.assertTrue(report["qualified"], report["violations"])
        contract = report["reserved_replay_contract"]
        self.assertEqual(set(contract), set(RUNTIME_POSITIVE_PROVIDERS))
        for provider, item in contract.items():
            self.assertEqual(item["count"], RESERVED_REPLAYS_PER_PROVIDER, provider)
            self.assertGreaterEqual(len(item["voices"]), 2, provider)
            self.assertEqual(
                len(RESERVED_REPLAY_VARIANTS[provider]),
                RESERVED_REPLAYS_PER_PROVIDER,
            )

    def test_replay_contract_rejects_single_voice_provider(self):
        rows = self.realized_rows()
        provider = "kokoro"
        reserved = [
            row
            for row in rows
            if row.get("provider") == provider
            and row.get("reserved_evidence_role") == "target_channel_positive"
        ]
        for row in reserved:
            row["voice"] = "collapsed"
        report = corpus_mix_report(rows)
        self.assertFalse(report["qualified"])
        self.assertIn(
            "reserved_replay_voice_collapse",
            {violation["reason"] for violation in report["violations"]},
        )

    def test_voice_identity_never_crosses_splits(self):
        rows = self.realized_rows()
        extra = dict(rows[0])
        extra["split"] = "test"
        extra["descriptor_sha256"] = "f" * 64
        extra["audio_sha256"] = "e" * 64
        report = corpus_mix_report(rows + [extra])
        self.assertFalse(report["qualified"])
        self.assertIn(
            "voice_crosses_splits",
            {violation["reason"] for violation in report["violations"]},
        )

    def test_recipe_change_prunes_stale_manifest_descriptors(self):
        tasks = _tasks()
        rows = [
            {"descriptor_sha256": tasks[0].descriptor},
            {"descriptor_sha256": "obsolete"},
        ]
        retained, pruned = retain_active_rows(rows, tasks)
        self.assertEqual(retained, rows[:1])
        self.assertEqual(pruned, 1)

    def test_deepgram_positive_uses_explicit_ipa_but_collision_does_not(self):
        base = dict(
            provider="deepgram",
            model="aura-2-thalia-en",
            voice="aura-2-thalia-en",
            provider_voice_id="aura-2-thalia-en",
            split="train",
            variant_index=0,
            settings={"speed": 1.0},
        )
        positive = Task(label=1, text="Kizz Control", **base)
        collision = Task(label=0, text="Kids Control", **base)
        text, expected = _deepgram_payload_text(positive)
        self.assertEqual(expected, 1)
        self.assertIn('"pronounce": "kɪz"', text)
        self.assertTrue(text.startswith(r"\{"))
        self.assertEqual(_deepgram_payload_text(collision), ("Kids Control", 0))

    def test_complete_wake_prefix_cannot_be_a_streaming_negative(self):
        for text in ("Kizz controller", "Kizz controlled"):
            decision = causal_negative_decision(text)
            self.assertFalse(decision["qualified"])
            self.assertEqual(
                decision["reason"], "causally_unlearnable_suffix_extension"
            )
        for text in ("Kids Control", "Kizz patrol", "The kids control it"):
            self.assertTrue(causal_negative_decision(text)["qualified"])

    def test_mix_rejects_eligible_causal_impossibility(self):
        rows = self.realized_rows()
        row = next(item for item in rows if item["label"] == 0)
        row["render_text"] = "Kizz controller"
        report = corpus_mix_report(rows)
        self.assertFalse(report["qualified"])
        self.assertIn(
            "causally_unlearnable_negative_is_eligible",
            {violation["reason"] for violation in report["violations"]},
        )


if __name__ == "__main__":
    unittest.main()
