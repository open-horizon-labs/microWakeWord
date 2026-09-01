import hashlib
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
import wave

from tools.freeze_kizz_control_c1_rirs import freeze, inventory_rirs


def write_wav(path: Path, value: int, *, frames: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = [value + (index % 7) for index in range(frames)]
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(struct.pack(f"<{frames}h", *samples))


def digest(path: Path, name: str) -> str:
    value = hashlib.new(name)
    value.update(path.read_bytes())
    return value.hexdigest()


class FreezeKizzControlC1RirsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.root = self.base / "RIRS_NOISES"
        self.real = self.root / "real_rirs_isotropic_noises"
        self.simulated = self.root / "simulated_rirs"
        self.archive = self.base / "rirs_noises.zip"
        self.archive.write_bytes(b"synthetic SLR28 archive fixture")
        self.archive_md5 = digest(self.archive, "md5")
        self.archive_sha256 = digest(self.archive, "sha256")
        self.discovery = []

        real_names = (
            "RVB2014_type1_rir_smallroom1_near_angla.wav",
            "RVB2014_type1_rir_mediumroom1_far_anglb.wav",
            "RWCP_type1_rir_circle_e1c_imp010.wav",
            "air_type1_air_binaural_booth_1_2.wav",
            "RWCP_type4_rir_p30l.wav",
        )
        value = 100
        for name in real_names:
            path = self.real / name
            write_wav(path, value)
            value += 100
            self.discovery.append(path)
        # This is a documented noise neighbor in the real SLR28 directory. It
        # must be ignored even if discovery places it first.
        self.real_noise = self.real / "RVB2014_type1_noise_smallroom1_1.wav"
        write_wav(self.real_noise, 900)
        self.discovery.insert(0, self.real_noise)

        for stratum in ("smallroom", "mediumroom", "largeroom"):
            for room_number in range(1, 6):
                room = f"Room{room_number:03d}"
                for response_number in range(2):
                    path = (
                        self.simulated
                        / stratum
                        / room
                        / f"{room}-{response_number:05d}.wav"
                    )
                    write_wav(path, value)
                    value += 100
                    self.discovery.append(path)

    def tearDown(self):
        self.temporary.cleanup()

    def run_freeze(self, discovery=None):
        output = self.base / "rir-manifest.json"
        report = self.base / "rir-report.json"
        result = freeze(
            self.root,
            self.archive,
            output,
            report,
            per_stratum=4,
            seed=231,
            expected_archive_md5=self.archive_md5,
            expected_archive_sha256=self.archive_sha256,
            discovered_paths=self.discovery if discovery is None else discovery,
        )
        return output, report, result

    def test_freezes_balanced_train_only_live_rirs_with_grouping_and_no_noise(self):
        output, report_path, (manifest, report) = self.run_freeze()
        self.assertEqual(
            manifest["counts"]["by_stratum"],
            {"real": 4, "smallroom": 4, "mediumroom": 4, "largeroom": 4},
        )
        self.assertTrue(report["qualified"])
        self.assertEqual(report["validation"]["noise_examples"], 0)
        self.assertEqual(len(manifest["examples"]), 16)
        self.assertNotIn(
            str(self.real_noise.resolve()),
            {row["path"] for row in manifest["examples"]},
        )
        for row in manifest["examples"]:
            self.assertTrue(Path(row["path"]).is_absolute())
            self.assertEqual(row["sha256"], digest(Path(row["path"]), "sha256"))
            self.assertEqual(row["audio_sha256"], row["sha256"])
            self.assertEqual(row["split"], "train")
            self.assertTrue(row["training_eligible"])
            self.assertGreater(row["duration_seconds"], 0)
            self.assertEqual(row["sample_rate_hz"], 16_000)
            self.assertEqual(row["channels"], 1)
            self.assertTrue(row["room_id"])
            self.assertTrue(row["source_identity"])
            self.assertTrue(row["source_group_id"].startswith("openslr28:"))
            self.assertEqual(row["archive_md5"], self.archive_md5)
            self.assertEqual(row["archive_sha256"], self.archive_sha256)
            self.assertNotIn("noise", Path(row["path"]).stem.lower().split("_"))
        for stratum in ("smallroom", "mediumroom", "largeroom"):
            rooms = {
                row["room_id"]
                for row in manifest["examples"]
                if row["stratum"] == stratum
            }
            self.assertEqual(len(rooms), 4)
        self.assertEqual(
            hashlib.sha256(output.read_bytes()).hexdigest(),
            json.loads(report_path.read_text())["manifest_sha256"],
        )

    def test_output_is_byte_deterministic_under_reordered_discovery(self):
        output, report, _ = self.run_freeze(self.discovery)
        first_manifest = output.read_bytes()
        first_report = report.read_bytes()
        self.run_freeze(reversed(self.discovery))
        self.assertEqual(output.read_bytes(), first_manifest)
        self.assertEqual(report.read_bytes(), first_report)

    def test_default_discovery_ignores_neighboring_pointsource_noise_tree(self):
        neighbor = self.root / "pointsource_noises" / "noise.wav"
        write_wav(neighbor, 1200)
        rows = inventory_rirs(self.root)
        self.assertNotIn(str(neighbor.resolve()), {row["path"] for row in rows})
        self.assertEqual(len(rows), len(self.discovery) - 1)

    def test_rejects_duplicate_paths_and_live_audio_hashes(self):
        with self.subTest("duplicate discovery path"):
            with self.assertRaisesRegex(ValueError, "duplicate RIR path"):
                inventory_rirs(
                    self.root,
                    discovered_paths=[*self.discovery, self.discovery[-1]],
                )
        duplicate = self.real / "RWCP_type1_rir_circle_e1c_imp020.wav"
        shutil.copyfile(self.real / "RWCP_type1_rir_circle_e1c_imp010.wav", duplicate)
        with self.subTest("duplicate audio hash"):
            with self.assertRaisesRegex(ValueError, "duplicate RIR audio hash"):
                inventory_rirs(self.root, discovered_paths=[*self.discovery, duplicate])

    def test_rejects_malformed_empty_and_accidental_simulated_noise_audio(self):
        cases = []
        malformed = self.real / "RWCP_type1_rir_circle_e1c_imp030.wav"
        malformed.write_bytes(b"not a WAV")
        cases.append((malformed, "malformed RIR audio"))

        silent = self.real / "RWCP_type1_rir_circle_e1c_imp040.wav"
        write_wav(silent, 0, frames=1)
        cases.append((silent, "empty/silent RIR audio"))

        noise = self.simulated / "smallroom" / "Room999" / "Room999-noise.wav"
        write_wav(noise, 700)
        cases.append((noise, "noise file found in simulated RIR tree"))

        for path, message in cases:
            with self.subTest(path=path.name):
                with self.assertRaisesRegex(ValueError, message):
                    inventory_rirs(self.root, discovered_paths=[*self.discovery, path])

    def test_rejects_archive_drift_and_ambiguous_outputs(self):
        with self.assertRaisesRegex(ValueError, "archive MD5 mismatch"):
            freeze(
                self.root,
                self.archive,
                self.base / "out.json",
                self.base / "report.json",
                per_stratum=1,
                seed=1,
                expected_archive_md5="0" * 32,
                expected_archive_sha256=self.archive_sha256,
                discovered_paths=self.discovery,
            )
        with self.assertRaisesRegex(ValueError, "must be distinct"):
            freeze(
                self.root,
                self.archive,
                self.base / "same.json",
                self.base / "same.json",
                per_stratum=1,
                seed=1,
                expected_archive_md5=self.archive_md5,
                expected_archive_sha256=self.archive_sha256,
                discovered_paths=self.discovery,
            )


if __name__ == "__main__":
    unittest.main()
