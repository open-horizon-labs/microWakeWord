import json
import tempfile
import unittest
from pathlib import Path

from tools.capture_kizz_control_adaptation_replays import select_rows


class CaptureKizzControlAdaptationReplaysTests(unittest.TestCase):
    def test_selection_round_robins_multiple_renders_per_voice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aligned = root / "aligned.json"
            rows = []
            for voice in ("voice-a", "voice-b"):
                for variant in range(2):
                    rows.append(
                        {
                            "provider": "deepgram",
                            "voice": voice,
                            "audio_sha256": f"{voice}-{variant}",
                            "descriptor_sha256": f"descriptor-{voice}-{variant}",
                            "render_text": f"Kizz Control {variant}",
                            "label": 1,
                            "split": "train",
                            "target_id": "kizz-control",
                            "training_eligible": True,
                            "alignment": {
                                "pronunciation_decision": {"accepted": True}
                            },
                        }
                    )
            aligned.write_text(json.dumps({"examples": rows}))
            qualification = root / "qualification.json"
            qualification.write_text(
                json.dumps(
                    {
                        "examples": [
                            {
                                "conditions": {
                                    "source_provider": "deepgram",
                                    "source_voice": "heldout",
                                }
                            }
                        ]
                    }
                )
            )
            selected = select_rows(
                aligned,
                qualification,
                providers=("deepgram",),
                per_provider=3,
            )
            self.assertEqual(
                [(row["voice"], row["render_text"]) for row in selected],
                [
                    ("voice-a", "Kizz Control 0"),
                    ("voice-b", "Kizz Control 0"),
                    ("voice-a", "Kizz Control 1"),
                ],
            )


if __name__ == "__main__":
    unittest.main()
