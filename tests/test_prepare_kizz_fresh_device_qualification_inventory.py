import unittest

from tools.prepare_kizz_fresh_device_qualification_inventory import (
    PROVIDERS,
    select_fresh_rows,
)


class PrepareKizzFreshDeviceQualificationInventoryTests(unittest.TestCase):
    def test_selection_excludes_prior_hashes_and_round_robins_voices(self):
        examples = []
        results = []
        excluded = set()
        for provider in PROVIDERS:
            for index in range(4):
                audio_hash = f"{provider}-{index}"
                examples.append(
                    {
                        "provider": provider,
                        "voice": f"voice-{index % 2}",
                        "audio_sha256": audio_hash,
                        "split": "test",
                        "label": 1,
                        "target_id": "kizz-control",
                        "render_text": f"Kizz Control {index}",
                    }
                )
                results.append({"audio_sha256": audio_hash, "accepted": True})
            excluded.add(f"{provider}-0")
        selected = select_fresh_rows(
            {"examples": examples},
            {"results": results},
            excluded,
            per_provider=3,
        )
        self.assertEqual(len(selected), 12)
        self.assertFalse(excluded.intersection(row["audio_sha256"] for row in selected))
        self.assertTrue(all(row["training_eligible"] is False for row in selected))
        self.assertTrue(all(row["fresh_qualification_holdout"] is True for row in selected))


if __name__ == "__main__":
    unittest.main()
