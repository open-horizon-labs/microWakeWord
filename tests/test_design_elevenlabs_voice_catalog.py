import base64
import tempfile
import unittest
from pathlib import Path

import yaml

from tools.design_elevenlabs_voice_catalog import design_catalog


class ElevenLabsVoiceDesignTest(unittest.TestCase):
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
