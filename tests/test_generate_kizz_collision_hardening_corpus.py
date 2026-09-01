import unittest
from pathlib import Path

from tools.generate_kizz_collision_hardening_corpus import (
    CRITICAL_TEXTS,
    DEFAULT_ENV_FILE,
    planned_tasks,
)
from tools.generate_kizz_control_c1_corpus import causal_negative_decision


class CollisionHardeningCorpusTest(unittest.TestCase):
    def test_every_provider_is_train_only_and_contains_critical_confusions(self):
        for provider in ("assemblyai", "deepgram", "elevenlabs", "kokoro"):
            tasks = planned_tasks(provider, contexts_per_voice=3)
            self.assertTrue(tasks)
            self.assertEqual({task.split for task in tasks}, {"train"})
            self.assertEqual({task.label for task in tasks}, {0})
            by_voice = {}
            for task in tasks:
                by_voice.setdefault(task.voice, set()).add(task.text)
            for texts in by_voice.values():
                self.assertTrue(set(CRITICAL_TEXTS) <= texts)

    def test_no_planned_phrase_contains_canonical_prefix(self):
        for task in planned_tasks("kokoro", contexts_per_voice=8):
            self.assertTrue(causal_negative_decision(task.text)["qualified"])

    def test_context_count_is_bounded(self):
        with self.assertRaises(ValueError):
            planned_tasks("kokoro", contexts_per_voice=10_000)

    def test_default_env_file_is_home_relative(self):
        self.assertEqual(
            DEFAULT_ENV_FILE,
            Path.home() / ".config" / "open-horizon-labs" / "voice.env",
        )


if __name__ == "__main__":
    unittest.main()
