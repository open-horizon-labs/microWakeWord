import unittest

from tools.lock_kizz_continuous_corpus import lock_rows


def row(source, split, category, identity, hours):
    return {
        "path": f"/{identity}.wav",
        "sha256": identity * 64,
        "duration_s": str(hours * 3600.0),
        "category": category,
        "split": split,
        "source": source,
    }


class LockKizzContinuousCorpusTests(unittest.TestCase):
    def test_lock_is_mixed_deterministic_and_validation_disjoint(self):
        rows = [
            row("MUSAN", "train", "connected_speech", "a", 40),
            row("MUSAN", "test", "music", "b", 40),
            row("ESC-50-derived-backgrounds-v1", "unassigned", "noise", "c", 10),
            row("LibriSpeech-train-clean-100", "train", "connected_speech", "d", 11),
            row("LibriSpeech-train-clean-100", "validation", "connected_speech", "e", 100),
        ]
        selected = lock_rows(rows, minimum_hours=100)
        self.assertEqual({item["sha256"] for item in selected}, {"a" * 64, "b" * 64, "c" * 64, "d" * 64})
        self.assertTrue(all(item["split"] != "validation" for item in selected))

    def test_gate_cannot_be_weakened_below_one_hundred_hours(self):
        with self.assertRaisesRegex(ValueError, "below 100 hours"):
            lock_rows([], minimum_hours=99.9)


if __name__ == "__main__":
    unittest.main()
