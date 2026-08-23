import json
from pathlib import Path
import tempfile
import unittest

from tools.select_recipe_cutoff import (
    cutoff_for_false_accept_rate,
    piper_speakers,
    summarize,
)


class RecipeCutoffSelectionTest(unittest.TestCase):
    def test_zero_false_accept_budget_uses_maximum_negative(self):
        cutoff = cutoff_for_false_accept_rate([0.1, 0.3, 0.2], 0.0)

        self.assertEqual(cutoff, 0.3)
        self.assertEqual(summarize([0.1, 0.3, 0.2], cutoff)["accepted"], 0)

    def test_nonzero_budget_uses_validation_order_statistic(self):
        peaks = [0.1, 0.2, 0.3, 0.4, 0.5]
        cutoff = cutoff_for_false_accept_rate(peaks, 0.2)

        self.assertEqual(cutoff, 0.4)
        self.assertEqual(summarize(peaks, cutoff)["accepted"], 1)

    def test_rejects_an_invalid_budget(self):
        with self.assertRaisesRegex(ValueError, "must be in"):
            cutoff_for_false_accept_rate([0.1], 1.0)

    def test_summary_reports_distribution_quantiles(self):
        summary = summarize([0.0, 0.25, 0.5, 0.75, 1.0], 0.5)

        self.assertEqual(
            summary["peak_probability_quantiles"],
            {"p05": 0.05, "p25": 0.25, "p50": 0.5, "p75": 0.75, "p95": 0.95},
        )

    def test_piper_speakers_reads_synthesis_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            metadata = [
                {"file": "0.wav", "speaker_1": 12, "speaker_2": 34},
                {"file": "1.wav", "speaker_1": 56},
            ]
            (output / "synthesis-metadata.jsonl").write_text(
                "\n".join(json.dumps(item) for item in metadata) + "\n"
            )

            self.assertEqual(
                piper_speakers(output),
                {"0.wav": "piper:12+34", "1.wav": "piper:56"},
            )


if __name__ == "__main__":
    unittest.main()
