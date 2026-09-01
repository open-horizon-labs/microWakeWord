import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.build_kizz_detector_scoring_corpus import (
    OUTPUT_FEATURES_NAME,
    OUTPUT_MANIFEST_NAME,
    _canonical_hash,
    build_detector_scoring_corpus,
    sha256_file,
)
from tools.trace_kizz_phoneme_detector import validate_sources


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _row_hash(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(values).tobytes(order="C")
    ).hexdigest()


class ScoringCorpusFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.features = root / "features"
        self.features.mkdir()
        self.audio = root / "audio"
        self.audio.mkdir()

        self.upstream_rows = [
            self._upstream_positive("parent-train", "train", "train-speaker"),
            self._upstream_positive(
                "parent-validation", "validation", "validation-speaker"
            ),
            self._upstream_positive("parent-test", "test", "test-speaker"),
        ]
        self.positive_manifest = root / "aligned-positives.json"
        _write_json(self.positive_manifest, {"examples": self.upstream_rows})

        # Deliberately interleaved and not lexicographically ordered.  Each
        # negative array must follow traversal order only within its block.
        self.negative_rows = [
            self._negative("ignored-positive", "train", "a", label=1),
            self._negative("train-z-1", "train", "z"),
            self._negative("validation-a-1", "validation", "a"),
            self._negative("train-a-2", "train", "a"),
            self._negative("test-a-1", "test", "a"),
            self._negative("train-a-1", "train", "a"),
            self._negative("ignored-group", "train", "unused"),
        ]
        self.negative_manifest = root / "negative-manifest.json"
        _write_json(self.negative_manifest, {"examples": self.negative_rows})

        train_parent = self.upstream_rows[0]
        validation_parent = self.upstream_rows[1]
        test_parent = self.upstream_rows[2]
        self.ledger = [
            # This is canonical-v3 ledger order: test sorts before train, then
            # validation.  The output must still use explicit split order.
            self._positive_variant(test_parent, "clean"),
            self._positive_variant(train_parent, "clean"),
            self._positive_variant(train_parent, "overlay-0", augmented=True),
            self._positive_variant(validation_parent, "clean"),
        ]
        self.provenance_payload = {
            "schema_version": 3,
            "recipe": "kizz_aligned_teacher_features_v3",
            "input_shape": [260, 40],
            "positive_manifests": [self._binding(self.positive_manifest)],
            "negative_manifest": self._binding(self.negative_manifest),
            "positive_counts": {"train": 2, "validation": 1, "test": 1},
            "negative_counts": {
                "train": {"z": 1, "a": 2},
                "validation": {"a": 1},
                "test": {"a": 1},
            },
            "examples": self.ledger,
        }
        self.provenance = self.features / "feature-provenance.json"
        self._write_provenance()

        self.array_values = {
            "positive_features-train.npy": [11.0, 12.0],
            "negative-train-a.npy": [13.0, 14.0],
            "negative-train-z.npy": [15.0],
            "positive_features-validation.npy": [21.0],
            "negative-validation-a.npy": [22.0],
            "positive_features-test.npy": [31.0],
            "negative-test-a.npy": [32.0],
        }
        for filename, values in self.array_values.items():
            np.save(
                self.features / filename,
                np.stack(
                    [np.full((260, 40), value, dtype=np.float32) for value in values]
                ),
            )

    def _audio_file(self, source_id: str) -> Path:
        path = self.audio / f"{source_id}.wav"
        path.write_bytes(f"live-audio:{source_id}".encode())
        return path

    def _upstream_positive(self, source_id: str, split: str, speaker: str) -> dict:
        path = self._audio_file(f"source-{source_id}")
        return {
            "source_id": source_id,
            "label": 1,
            "split": split,
            "path": str(path),
            "audio_sha256": sha256_file(path),
            "duration_seconds": 1.25,
            "provider": "provider",
            "speaker_id": speaker,
            "session_id": f"{speaker}-session",
            "ancestry_id": f"{speaker}-ancestry",
            "voice_id": speaker,
            "source_group": "synthetic-positive",
        }

    def _positive_variant(
        self, parent: dict, variant: str, *, augmented: bool = False
    ) -> dict:
        source_id = f"{parent['source_id']}::{variant}"
        path = self._audio_file(f"waveform-{source_id}")
        return {
            "source_id": source_id,
            "parent_source_id": parent["source_id"],
            "split": parent["split"],
            "variant": variant,
            "path": str(path),
            "audio_sha256": sha256_file(path),
            "source_audio_sha256": parent["audio_sha256"],
            "provider": parent["provider"],
            "speaker_id": parent["speaker_id"],
            # session_id is intentionally absent: the builder must recover it
            # from the exact bound positive manifest.
            "ancestry_id": parent["ancestry_id"],
            "voice_id": parent["voice_id"],
            "source_group": parent["source_group"],
            "augmentation": (
                {"seed": 7, "background_source_id": "background"}
                if augmented
                else None
            ),
        }

    def _negative(
        self, source_id: str, split: str, group: str, *, label: int = 0
    ) -> dict:
        path = self._audio_file(f"negative-{source_id}")
        return {
            "source_id": source_id,
            "label": label,
            "split": split,
            "source_group": group,
            "path": str(path),
            "audio_sha256": sha256_file(path),
            "duration_seconds": 4.0,
            "provider": "negative-provider",
            "speaker_id": f"{source_id}-speaker",
            "session_id": f"{source_id}-session",
            "ancestry_id": f"{source_id}-ancestry",
        }

    @staticmethod
    def _binding(path: Path) -> dict:
        return {"path": str(path.resolve()), "sha256": sha256_file(path)}

    def _write_provenance(self) -> None:
        _write_json(self.provenance, self.provenance_payload)

    def rebind_positive_manifest(self) -> None:
        _write_json(self.positive_manifest, {"examples": self.upstream_rows})
        self.provenance_payload["positive_manifests"] = [
            self._binding(self.positive_manifest)
        ]
        self._write_provenance()

    def run(self, output: Path) -> dict:
        return build_detector_scoring_corpus(
            self.provenance,
            self.features,
            self.negative_manifest,
            output,
        )


class BuildKizzDetectorScoringCorpusTests(unittest.TestCase):
    def test_reconstructs_exact_rows_and_is_downstream_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ScoringCorpusFixture(Path(directory))
            output = fixture.root / "output"
            report = fixture.run(output)

            expected_ids = [
                "parent-train::clean",
                "parent-train::overlay-0",
                "train-a-2",
                "train-a-1",
                "train-z-1",
                "parent-validation::clean",
                "validation-a-1",
                "parent-test::clean",
                "test-a-1",
            ]
            self.assertEqual(
                [row["source_id"] for row in report["examples"]], expected_ids
            )
            self.assertEqual(
                [float(row[0, 0]) for row in np.load(output / OUTPUT_FEATURES_NAME)],
                [11.0, 12.0, 13.0, 14.0, 15.0, 21.0, 22.0, 31.0, 32.0],
            )
            self.assertEqual(
                report["examples"][0]["session_id"], "train-speaker-session"
            )
            self.assertEqual(
                [report["examples"][index]["negative_manifest_index"] for index in (2, 3)],
                [3, 5],
            )
            for index, row in enumerate(report["examples"]):
                self.assertEqual(row["feature_index"], index)
                self.assertEqual(
                    row["feature_sha256"],
                    _row_hash(np.load(output / OUTPUT_FEATURES_NAME)[index]),
                )
                self.assertEqual(
                    row["feature_sha256"], row["source_feature"]["feature_sha256"]
                )
            self.assertEqual(
                report["array_sha256"][OUTPUT_FEATURES_NAME],
                sha256_file(output / OUTPUT_FEATURES_NAME),
            )
            self.assertEqual(
                report["outputs"]["source_features"]["sha256"],
                sha256_file(output / OUTPUT_FEATURES_NAME),
            )
            without_self_hash = dict(report)
            self_hash = without_self_hash.pop("manifest_payload_sha256")
            self.assertEqual(self_hash, _canonical_hash(without_self_hash))
            self.assertEqual(
                json.loads((output / OUTPUT_MANIFEST_NAME).read_text()), report
            )

            values = np.load(output / OUTPUT_FEATURES_NAME, mmap_mode="r")
            validated = validate_sources(
                output / OUTPUT_MANIFEST_NAME,
                report,
                output / OUTPUT_FEATURES_NAME,
                values,
            )
            self.assertEqual(len(validated), len(expected_ids))

    def test_outputs_are_byte_deterministic_and_bind_every_input_array(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ScoringCorpusFixture(Path(directory))
            first = fixture.root / "first"
            second = fixture.root / "second"
            first_report = fixture.run(first)
            second_report = fixture.run(second)

            self.assertEqual(first_report, second_report)
            self.assertEqual(
                (first / OUTPUT_FEATURES_NAME).read_bytes(),
                (second / OUTPUT_FEATURES_NAME).read_bytes(),
            )
            self.assertEqual(
                (first / OUTPUT_MANIFEST_NAME).read_bytes(),
                (second / OUTPUT_MANIFEST_NAME).read_bytes(),
            )
            bound_names = {
                Path(binding["path"]).name
                for binding in first_report["inputs"]["feature_arrays"]
            }
            self.assertEqual(bound_names, set(fixture.array_values))
            for binding in first_report["inputs"]["feature_arrays"]:
                self.assertEqual(
                    binding["sha256"], sha256_file(Path(binding["path"]))
                )
            self.assertEqual(
                first_report["inputs"]["feature_provenance"]["sha256"],
                sha256_file(fixture.provenance),
            )
            self.assertEqual(
                first_report["inputs"]["upstream"]["negative_manifest"]["sha256"],
                sha256_file(fixture.negative_manifest),
            )

    def test_rejects_negative_manifest_binding_drift_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ScoringCorpusFixture(Path(directory))
            fixture.negative_manifest.write_text(
                fixture.negative_manifest.read_text() + " \n"
            )
            output = fixture.root / "output"
            with self.assertRaisesRegex(ValueError, "negative manifest binding drifted"):
                fixture.run(output)
            self.assertFalse(output.exists())

    def test_rejects_live_audio_drift_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ScoringCorpusFixture(Path(directory))
            Path(fixture.ledger[0]["path"]).write_bytes(b"tampered-waveform")
            output = fixture.root / "output"
            with self.assertRaisesRegex(ValueError, "live audio hash drifted"):
                fixture.run(output)
            self.assertFalse(output.exists())

    def test_rejects_augmented_heldout_positive(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ScoringCorpusFixture(Path(directory))
            fixture.ledger[0]["variant"] = "overlay-0"
            fixture.ledger[0]["source_id"] = "parent-test::overlay-0"
            fixture.ledger[0]["augmentation"] = {"seed": 9}
            fixture._write_provenance()
            with self.assertRaisesRegex(ValueError, "held-out positive is not clean-only"):
                fixture.run(fixture.root / "output")

    def test_rejects_cross_split_identity_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ScoringCorpusFixture(Path(directory))
            validation = fixture.upstream_rows[1]
            validation["speaker_id"] = "train-speaker"
            validation["voice_id"] = "train-speaker"
            fixture.ledger[3]["speaker_id"] = "train-speaker"
            fixture.ledger[3]["voice_id"] = "train-speaker"
            fixture.rebind_positive_manifest()
            with self.assertRaisesRegex(ValueError, "identity leakage"):
                fixture.run(fixture.root / "output")

    def test_rejects_cross_split_source_audio_hash_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ScoringCorpusFixture(Path(directory))
            train = fixture.upstream_rows[0]
            validation = fixture.upstream_rows[1]
            validation["path"] = train["path"]
            validation["audio_sha256"] = train["audio_sha256"]
            fixture.ledger[3]["source_audio_sha256"] = train["audio_sha256"]
            fixture.rebind_positive_manifest()
            with self.assertRaisesRegex(ValueError, "hash leakage"):
                fixture.run(fixture.root / "output")

    def test_rejects_array_count_drift_and_stale_scoring_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ScoringCorpusFixture(Path(directory))
            np.save(
                fixture.features / "negative-test-a.npy",
                np.zeros((2, 260, 40), dtype=np.float32),
            )
            with self.assertRaisesRegex(ValueError, "expected float32"):
                fixture.run(fixture.root / "count-output")

        with tempfile.TemporaryDirectory() as directory:
            fixture = ScoringCorpusFixture(Path(directory))
            np.save(
                fixture.features / "negative-test-stale.npy",
                np.zeros((1, 260, 40), dtype=np.float32),
            )
            with self.assertRaisesRegex(ValueError, "missing or stale scoring arrays"):
                fixture.run(fixture.root / "stale-output")

    def test_rejects_cross_split_feature_hash_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ScoringCorpusFixture(Path(directory))
            train = np.load(fixture.features / "positive_features-train.npy")
            test = np.load(fixture.features / "positive_features-test.npy")
            test[0] = train[0]
            np.save(fixture.features / "positive_features-test.npy", test)
            with self.assertRaisesRegex(ValueError, "nonzero feature hash collision"):
                fixture.run(fixture.root / "output")


if __name__ == "__main__":
    unittest.main()
