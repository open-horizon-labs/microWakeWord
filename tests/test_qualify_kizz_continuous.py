import unittest

from tools.qualify_kizz_continuous import validate_input_contract


class QualifyKizzContinuousTests(unittest.TestCase):
    def test_requires_model_bound_untouched_streams(self):
        payload = {
            "schema_version": 1,
            "model_sha256": "a" * 64,
            "test_is_untouched": True,
            "streams": [{"id": "validation-positive"}],
        }
        validate_input_contract(payload)
        with self.assertRaisesRegex(ValueError, "model_sha256"):
            validate_input_contract({**payload, "model_sha256": "missing"})
        with self.assertRaisesRegex(ValueError, "test_is_untouched"):
            validate_input_contract({**payload, "test_is_untouched": False})
        with self.assertRaisesRegex(ValueError, "score streams"):
            validate_input_contract({**payload, "streams": []})


if __name__ == "__main__":
    unittest.main()
