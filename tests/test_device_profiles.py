import json
import tempfile
import unittest
from pathlib import Path

from microwakeword.device_profiles import load_device_profiles, microphone_targets

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "device-profiles.json"


class DeviceProfilesTest(unittest.TestCase):
    def test_every_current_product_is_classified(self):
        catalog = load_device_profiles(CATALOG)
        self.assertEqual(
            {target["target_id"] for target in catalog["targets"]},
            {
                "hiphi_dial",
                "hiphi_frame",
                "hiphi_rlcd",
                "hiphi_joy",
                "hiphi_tough",
                "hiphi_m5dial",
                "hiphi_sticks3",
                "hiphi_stopwatch",
                "hiphi_kizz",
            },
        )

    def test_all_seven_microphone_targets_have_profiles(self):
        catalog = load_device_profiles(CATALOG)
        targets = microphone_targets(catalog)
        self.assertEqual(
            {target["target_id"] for target in targets},
            {
                "hiphi_dial",
                "hiphi_frame",
                "hiphi_rlcd",
                "hiphi_tough",
                "hiphi_sticks3",
                "hiphi_stopwatch",
                "hiphi_kizz",
            },
        )
        self.assertEqual(len({t["enrollment"]["device_profile"] for t in targets}), 7)

    def test_non_microphone_target_cannot_claim_profile(self):
        catalog = json.loads(CATALOG.read_text())
        joy = next(t for t in catalog["targets"] if t["target_id"] == "hiphi_joy")
        joy["enrollment"]["device_profile"] = "invented_microphone_v1"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.json"
            path.write_text(json.dumps(catalog))
            with self.assertRaisesRegex(ValueError, "cannot enroll"):
                load_device_profiles(path)

    def test_catalog_does_not_claim_uncollected_real_corpora(self):
        catalog = load_device_profiles(CATALOG)
        self.assertTrue(
            all(
                target["enrollment"]["corpus_status"] == "not_collected"
                for target in microphone_targets(catalog)
            )
        )


if __name__ == "__main__":
    unittest.main()
