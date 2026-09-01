import json
import tempfile
import unittest
from pathlib import Path

from tools.generate_recipe_samples import (
    connected_generation_plan,
    generation_signature,
    reusable_connected_source,
    read_connected_text_source,
    validate_realized_speaker_isolation,
    validate_connected_sentence_sources,
    verify_connected_output,
)


ROOT = Path(__file__).parents[1]
KIZZ = ROOT / "recipes" / "kizz"


def generation_config():
    return {
        "max_speakers": 100,
        "length_scales": [0.9, 1.0],
        "noise_scales": [0.75],
        "noise_scale_ws": [0.8],
        "slerp_weights": [0.0, 0.5],
        "speaker_cohorts": {
            "train": {
                "speaker_start": 0,
                "speaker_end": 40,
                "sample_fraction": 0.8,
                "age_group": "adult",
            },
            "validation": {
                "speaker_start": 40,
                "speaker_end": 45,
                "sample_fraction": 0.1,
                "age_group": "adult",
            },
            "test": {
                "speaker_start": 45,
                "speaker_end": 50,
                "sample_fraction": 0.1,
                "age_group": "adult",
            },
        },
    }


class ConnectedSentenceGenerationTests(unittest.TestCase):
    def source_entries(self, samples_per_text=3):
        return [
            {
                "name": "connected-negative-sentences",
                "train": str(KIZZ / "connected-negative-sentences-train.txt"),
                "validation": str(KIZZ / "connected-negative-sentences-validation.txt"),
                "test": str(KIZZ / "connected-negative-sentences-test.txt"),
                "samples_per_text": samples_per_text,
            }
        ]

    def test_sources_have_exact_counts_and_no_text_or_speaker_overlap(self):
        sources = validate_connected_sentence_sources(self.source_entries())
        self.assertEqual([s["sources"]["train"]["line_count"] for s in sources], [32])
        self.assertEqual(sources[0]["sources"]["validation"]["line_count"], 8)
        self.assertEqual(sources[0]["sources"]["test"]["line_count"], 8)
        plan = connected_generation_plan(
            sources, generation_config(), Path("/tmp/model.pt"), Path("/tmp/out"), 4, 17
        )
        self.assertEqual([item["samples"] for item in plan], [96, 24, 24])
        speaker_ranges = [(item["speaker_start"], item["speaker_end"]) for item in plan]
        self.assertEqual(len(speaker_ranges), len(set(speaker_ranges)))
        self.assertEqual(
            {item["split"] for item in plan}, {"train", "validation", "test"}
        )
        self.assertEqual(
            len(
                {
                    line["id"]
                    for split in sources[0]["sources"].values()
                    for line in split["lines"]
                }
            ),
            48,
        )

    def test_plan_hashes_and_commands_are_deterministic(self):
        sources = validate_connected_sentence_sources(self.source_entries())
        first = connected_generation_plan(
            sources, generation_config(), Path("/tmp/model.pt"), Path("/tmp/out"), 8, 99
        )
        second = connected_generation_plan(
            sources, generation_config(), Path("/tmp/model.pt"), Path("/tmp/out"), 8, 99
        )
        self.assertEqual(first, second)
        self.assertEqual(
            first[0]["source_sha256"],
            read_connected_text_source(Path(first[0]["source_path"]))["sha256"],
        )
        self.assertEqual(first[0]["command"][3], first[0]["source_path"])
        self.assertEqual(first[0]["seed"], 99)

    def test_reuse_signature_ignores_output_specific_metadata_path(self):
        first = generation_signature(
            [
                "/old/python",
                "-m",
                "piper_sample_generator",
                "sentences.txt",
                "--output-dir",
                "/old/output",
                "--metadata-file",
                "/old/output/synthesis-metadata.jsonl",
            ]
        )
        second = generation_signature(
            [
                "/new/python",
                "-m",
                "piper_sample_generator",
                "sentences.txt",
                "--output-dir",
                "/new/output",
                "--metadata-file",
                "/new/output/synthesis-metadata.jsonl",
            ]
        )
        self.assertEqual(first, second)

    def test_piper_metadata_maps_each_wav_to_a_source_sentence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "sentences.txt"
            source_path.write_text(
                "High five kids, dinner is ready.\nThe Wi-Fi is working.\n",
                encoding="utf-8",
            )
            source = read_connected_text_source(source_path)
            output = root / "generated"
            output.mkdir()
            records = []
            for index, line in enumerate(source["lines"] * 2):
                filename = f"{index:04d}.wav"
                (output / filename).touch()
                records.append({"file": filename, "text": line["text"]})
            (output / "synthesis-metadata.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            result = verify_connected_output(output, source, 2)
            self.assertEqual(result["wav_count"], 4)
            self.assertEqual(
                result["line_ids"], sorted(line["id"] for line in source["lines"])
            )

    def test_realized_piper_speakers_must_match_and_remain_disjoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "sentences.txt"
            source_path.write_text("A connected sentence.\n", encoding="utf-8")
            source = read_connected_text_source(source_path)
            output = root / "generated"
            output.mkdir()
            (output / "0.wav").touch()
            (output / "synthesis-metadata.jsonl").write_text(
                json.dumps(
                    {
                        "file": "0.wav",
                        "text": "A connected sentence.",
                        "speaker_1": 12,
                        "speaker_2": 18,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            verified = verify_connected_output(output, source, 1, 10, 20)
            self.assertEqual(verified["realized_speaker_ids"], [12, 18])
            with self.assertRaisesRegex(ValueError, "outside"):
                verify_connected_output(output, source, 1, 20, 30)
            with self.assertRaisesRegex(ValueError, "overlap"):
                validate_realized_speaker_isolation(
                    [
                        {
                            "text_source": "one",
                            "split": "train",
                            "realized_speaker_ids": [12],
                        },
                        {
                            "text_source": "one",
                            "split": "validation",
                            "realized_speaker_ids": [12],
                        },
                    ]
                )

    def test_reuse_requires_matching_source_identity_and_verified_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "sentences.txt"
            source_path.write_text(
                "A sentence for the first split.\n", encoding="utf-8"
            )
            source = read_connected_text_source(source_path)
            generated = root / "old-generated"
            generated.mkdir()
            (generated / "0000.wav").touch()
            (generated / "synthesis-metadata.jsonl").write_text(
                json.dumps(
                    {
                        "file": "0000.wav",
                        "text": source["lines"][0]["text"],
                        "speaker_1": 1,
                        "speaker_2": 2,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            item = {
                "text_source": "source",
                "source_path": str(source_path),
                "source_sha256": source["sha256"],
                "split": "train",
                "speaker_start": 0,
                "speaker_end": 10,
                "samples_per_text": 1,
                "normalized_line_ids": [source["lines"][0]["id"]],
                "normalized_texts": [source["lines"][0]["text"]],
                "command": [
                    "python",
                    "-m",
                    "piper_sample_generator",
                    str(source_path),
                    "--output-dir",
                    str(root / "new"),
                ],
            }
            manifest = {
                "generator_model_sha256": None,
                "plan": [dict(item, output=str(generated))],
            }
            self.assertEqual(
                reusable_connected_source([manifest], item, None), generated
            )
            item["source_sha256"] = "changed"
            self.assertIsNone(reusable_connected_source([manifest], item, None))

    def test_rejects_blank_duplicate_overlap_and_zero_allocations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def files(train, validation="different sentence", test="third sentence"):
                paths = {}
                for split, text in (
                    ("train", train),
                    ("validation", validation),
                    ("test", test),
                ):
                    paths[split] = root / f"{split}.txt"
                    paths[split].write_text(text, encoding="utf-8")
                return paths

            paths = files("one sentence\n\nsecond sentence")
            with self.assertRaisesRegex(ValueError, "blank"):
                validate_connected_sentence_sources(
                    [
                        {
                            "name": "bad",
                            **{k: str(v) for k, v in paths.items()},
                            "samples_per_text": 1,
                        }
                    ]
                )

            paths = files("one sentence\none sentence")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                validate_connected_sentence_sources(
                    [
                        {
                            "name": "bad",
                            **{k: str(v) for k, v in paths.items()},
                            "samples_per_text": 1,
                        }
                    ]
                )

            paths = files("shared sentence", "shared sentence")
            with self.assertRaisesRegex(ValueError, "overlaps"):
                validate_connected_sentence_sources(
                    [
                        {
                            "name": "bad",
                            **{k: str(v) for k, v in paths.items()},
                            "samples_per_text": 1,
                        }
                    ]
                )

            paths = files("one sentence")
            with self.assertRaisesRegex(ValueError, "samples_per_text"):
                validate_connected_sentence_sources(
                    [
                        {
                            "name": "bad",
                            **{k: str(v) for k, v in paths.items()},
                            "samples_per_text": 0,
                        }
                    ]
                )

    def test_sources_contain_connected_collisions_but_no_canonical_phrase(self):
        sentences = [
            sentence.casefold()
            for path in (
                KIZZ / "connected-negative-sentences-train.txt",
                KIZZ / "connected-negative-sentences-validation.txt",
                KIZZ / "connected-negative-sentences-test.txt",
            )
            for sentence in path.read_text(encoding="utf-8").splitlines()
        ]
        text = "\n".join(sentences)
        self.assertIn("high five", text)
        self.assertIn("wi-fi is", text)
        self.assertIn("if he is", text)
        self.assertTrue(any(line.startswith("high five kids") for line in sentences))
        self.assertTrue(
            any(
                "high five kids" in line
                and not line.startswith("high five kids")
                and not line.rstrip(".!?").endswith("high five kids")
                for line in sentences
            )
        )
        self.assertTrue(
            any(line.rstrip(".!?").endswith("high five kids") for line in sentences)
        )
        self.assertNotRegex(text, r"hi[- ]?fi\s+kizz")
        self.assertNotIn("high five kizz", text)


if __name__ == "__main__":
    unittest.main()
