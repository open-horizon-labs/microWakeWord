import base64
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib import error

import yaml

from tools.design_elevenlabs_voice_catalog import (
    design_catalog,
    post_json,
    resolve_api_key,
)


class ElevenLabsVoiceDesignTest(unittest.TestCase):
    def test_accepts_both_common_api_key_environment_names(self):
        with mock.patch.dict(os.environ, {"ELEVEN_LABS_API_KEY": "key"}, clear=True):
            self.assertEqual(resolve_api_key(), "key")

    def test_provider_error_includes_the_response_detail(self):
        failure = error.HTTPError(
            "https://example.invalid",
            403,
            "Forbidden",
            {},
            io.BytesIO(b'{"detail":"paid plan required"}'),
        )
        with mock.patch(
            "tools.design_elevenlabs_voice_catalog.request.urlopen",
            side_effect=failure,
        ):
            with self.assertRaisesRegex(RuntimeError, "paid plan required"):
                post_json("https://example.invalid", "key", {})

    def test_designs_and_resolves_catalog_with_saved_previews(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = root / "designs.yaml"
            spec.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "preview_text": "A" * 100,
                        "voices": [
                            {
                                "name": "adult-test",
                                "split": "test",
                                "age_group": "adult",
                                "description": "A natural adult test voice.",
                            }
                        ],
                    }
                )
            )
            calls = []

            def fake_post(endpoint, api_key, body):
                calls.append((endpoint, api_key, body))
                if endpoint.endswith("/design"):
                    return {
                        "previews": [
                            {
                                "generated_voice_id": "preview-id",
                                "audio_base_64": base64.b64encode(b"mp3").decode(),
                            }
                        ]
                    }
                return {"voice_id": "voice-id"}

            output = root / "catalog.yaml"
            catalog = design_catalog(spec, output, root / "previews", "key", fake_post)

            self.assertEqual(len(calls), 2)
            self.assertEqual(catalog["voices"][0]["voice_id"], "voice-id")
            self.assertEqual((root / "previews/adult-test-0.mp3").read_bytes(), b"mp3")


if __name__ == "__main__":
    unittest.main()
