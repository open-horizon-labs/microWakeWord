import tempfile
import unittest
from pathlib import Path

from tools.evaluate_recipe_model import clips_by_group


class EvaluateRecipeModelTest(unittest.TestCase):
    def test_connected_sentence_source_without_phrase_group_is_evaluable(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            clip = output / "sentence.wav"
            clip.touch()
            manifest = {
                "schema_version": 2,
                "plan": [
                    {
                        "class": "hard_negative",
                        "split": "validation",
                        "text": None,
                        "text_source": "connected-household-speech",
                        "output": str(output),
                    }
                ],
            }

            grouped = clips_by_group(
                output,
                "validation",
                231,
                generation_manifest=manifest,
                class_name="hard_negative",
            )

            self.assertEqual(grouped, {"connected-household-speech": [clip]})


if __name__ == "__main__":
    unittest.main()
