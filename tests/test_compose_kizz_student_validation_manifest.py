import json
import tempfile
import unittest
from pathlib import Path

from tools.compose_kizz_student_validation_manifest import compose


class ComposeStudentValidationManifestTests(unittest.TestCase):
    def test_combines_clean_validation_and_twelve_device_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean = root / "clean.json"
            device = root / "device.json"
            clean.write_text(json.dumps({"examples": [{"split": "validation", "label": 1}, {"split": "train", "label": 1}]}))
            device.write_text(json.dumps({"examples": [{"split": "validation", "label": 1} for _ in range(12)]}))
            result = compose(clean, device, root / "combined.json")
            self.assertEqual(result["counts"], {"clean": 1, "device_positive": 12, "total": 13})
            self.assertEqual(len(result["examples"]), 13)


if __name__ == "__main__":
    unittest.main()
