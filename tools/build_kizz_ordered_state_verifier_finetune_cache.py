#!/usr/bin/env python3
"""Build a train-only ordered-state verifier fine-tuning cache.

The cache replays frozen detector candidates with the qualified offline teacher
and deliberately promotes consumed StackChan captures into training.  Validation
and test candidate rows are never materialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from microwakeword.kizz_teacher import build_teacher
from microwakeword.ordered_state import OrderedStateTopology


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _binding(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path)}


def _candidate_data(
    corpus_path: Path, features_path: Path, *, physical: bool
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    corpus_path = corpus_path.expanduser().resolve()
    features_path = features_path.expanduser().resolve()
    corpus = _load_object(corpus_path, "candidate corpus")
    rows = corpus.get("examples")
    if not isinstance(rows, list) or not rows:
        raise ValueError("candidate corpus examples are missing")
    features = np.load(features_path, mmap_mode="r", allow_pickle=False)
    if features.shape != (len(rows), 260, 40):
        raise ValueError("candidate features must have shape [N,260,40]")
    if corpus.get("array_sha256", {}).get(features_path.name) != sha256_file(
        features_path
    ):
        raise ValueError("candidate feature-array hash drift")
    for row in rows:
        if row.get("split") not in {"train", "validation", "test"} or row.get(
            "label"
        ) not in (0, 1, False, True):
            raise ValueError("candidate split/label drift")
        if physical and int(row["label"]) != 1:
            raise ValueError("consumed physical cache must contain only positives")
    return corpus, [dict(row) for row in rows], features


def materialization_plan(
    candidate_rows: Sequence[Mapping[str, Any]],
    physical_rows: Sequence[Mapping[str, Any]],
    *,
    positive_repeats: int,
    physical_repeats: int,
    seed: int,
    hard_negative_group: str | None = None,
    hard_negative_repeats: int = 1,
) -> list[tuple[str, int]]:
    if positive_repeats < 1 or physical_repeats < 1 or hard_negative_repeats < 1:
        raise ValueError("repeat counts must be positive")
    hard_negative_group = (
        hard_negative_group.strip() if hard_negative_group is not None else None
    )
    if hard_negative_group == "":
        hard_negative_group = None
    if hard_negative_repeats > 1 and hard_negative_group is None:
        raise ValueError("hard-negative repeats require a source group")
    train = [
        index for index, row in enumerate(candidate_rows) if row.get("split") == "train"
    ]
    if not train:
        raise ValueError("candidate corpus has no training rows")
    positives = [index for index in train if int(candidate_rows[index]["label"]) == 1]
    negatives = [index for index in train if int(candidate_rows[index]["label"]) == 0]
    if not positives or not negatives or not physical_rows:
        raise ValueError("fine-tuning requires synthetic positives, negatives, and physical positives")
    plan = [("candidate", index) for index in negatives]
    hard_negatives = [
        index
        for index in negatives
        if hard_negative_group is not None
        and candidate_rows[index].get("source_group") == hard_negative_group
    ]
    if hard_negative_group is not None and not hard_negatives:
        raise ValueError(f"hard-negative source group is absent: {hard_negative_group}")
    plan.extend(
        ("candidate", index)
        for _ in range(hard_negative_repeats - 1)
        for index in hard_negatives
    )
    plan.extend(
        ("candidate", index)
        for _ in range(positive_repeats)
        for index in positives
    )
    plan.extend(
        ("physical", index)
        for _ in range(physical_repeats)
        for index in range(len(physical_rows))
    )
    rng = np.random.default_rng(seed)
    rng.shuffle(plan)
    return plan


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def build(
    candidate_corpus: Path,
    candidate_features: Path,
    physical_corpus: Path,
    physical_features: Path,
    detector_teacher_gate: Path,
    output_prefix: Path,
    *,
    positive_repeats: int = 4,
    physical_repeats: int = 32,
    hard_negative_group: str | None = None,
    hard_negative_repeats: int = 1,
    alignment_offset: int = 21,
    student_output_frames: int = 66,
    batch_size: int = 128,
    seed: int = 6053,
) -> dict[str, Any]:
    output_prefix = output_prefix.expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = {
        name: output_prefix.with_name(name)
        for name in ("features.npy", "targets.npy", "labels.npy", "teacher_logits.npy")
    }
    metadata_path = output_prefix.with_suffix(".json")
    if metadata_path.exists() or any(path.exists() for path in outputs.values()):
        raise FileExistsError("refusing to overwrite fine-tuning cache")
    if batch_size < 1 or alignment_offset < 0 or student_output_frames < 1:
        raise ValueError("batch and alignment geometry must be positive")

    candidate_payload, candidate_rows, candidate_values = _candidate_data(
        candidate_corpus, candidate_features, physical=False
    )
    _, physical_rows, physical_values = _candidate_data(
        physical_corpus, physical_features, physical=True
    )
    plan = materialization_plan(
        candidate_rows,
        physical_rows,
        positive_repeats=positive_repeats,
        physical_repeats=physical_repeats,
        seed=seed,
        hard_negative_group=hard_negative_group,
        hard_negative_repeats=hard_negative_repeats,
    )

    gate_path = detector_teacher_gate.expanduser().resolve()
    gate = _load_object(gate_path, "detector teacher gate")
    if (
        gate.get("qualified") is not True
        or gate.get("eligible_for_detector_distillation") is not True
    ):
        raise ValueError("detector teacher gate is not qualified")
    checkpoint = gate.get("selected_checkpoint", {})
    training_binding = gate.get("training_report", {})
    teacher_weights = Path(str(checkpoint.get("best_weights_path", ""))).resolve()
    teacher_training = Path(str(training_binding.get("path", ""))).resolve()
    if (
        not teacher_weights.is_file()
        or sha256_file(teacher_weights) != checkpoint.get("best_weights_sha256")
        or not teacher_training.is_file()
        or sha256_file(teacher_training) != training_binding.get("sha256")
    ):
        raise ValueError("detector teacher gate binding drift")
    training = _load_object(teacher_training, "teacher training")
    topology_payload = gate.get("topology", {})
    topology = OrderedStateTopology(
        tuple(str(phone) for phone in topology_payload.get("phones", ())),
        int(topology_payload.get("states_per_phone", 0)),
    )
    if alignment_offset + student_output_frames > int(training["output_shape"][0]):
        raise ValueError("teacher/student alignment exceeds teacher output")
    teacher = build_teacher(
        hidden_size=int(training["hidden_size"]),
        recurrent_layers=int(training["recurrent_layers"]),
        output_frames=int(training["output_shape"][0]),
        topology=topology,
    )
    teacher.load_weights(teacher_weights)

    temporary_paths = {
        name: path.with_name(f".{path.name}.building") for name, path in outputs.items()
    }
    arrays = {
        "features": np.lib.format.open_memmap(
            temporary_paths["features.npy"], mode="w+", dtype=np.float16,
            shape=(len(plan), 260, 40),
        ),
        "targets": np.lib.format.open_memmap(
            temporary_paths["targets.npy"], mode="w+", dtype=np.int8,
            shape=(len(plan), student_output_frames),
        ),
        "labels": np.lib.format.open_memmap(
            temporary_paths["labels.npy"], mode="w+", dtype=np.int8,
            shape=(len(plan),),
        ),
        "teacher_logits": np.lib.format.open_memmap(
            temporary_paths["teacher_logits.npy"], mode="w+", dtype=np.float16,
            shape=(len(plan), student_output_frames, topology.state_count),
        ),
    }
    try:
        for start in range(0, len(plan), batch_size):
            selected = plan[start : start + batch_size]
            batch = np.stack(
                [
                    np.asarray(
                        candidate_values[index]
                        if source == "candidate"
                        else physical_values[index],
                        dtype=np.float32,
                    )
                    for source, index in selected
                ]
            )
            labels = np.asarray(
                [
                    int(
                        candidate_rows[index]["label"]
                        if source == "candidate"
                        else physical_rows[index]["label"]
                    )
                    for source, index in selected
                ],
                dtype=np.int8,
            )
            logits = np.asarray(teacher.predict(batch, verbose=0), dtype=np.float32)[
                :, alignment_offset : alignment_offset + student_output_frames
            ]
            stop = start + len(selected)
            arrays["features"][start:stop] = batch
            arrays["labels"][start:stop] = labels
            arrays["teacher_logits"][start:stop] = logits
            arrays["targets"][start:stop] = np.argmax(logits, axis=-1).astype(np.int8)
        for value in arrays.values():
            value.flush()
        del arrays
        for name, path in outputs.items():
            os.replace(temporary_paths[name], path)
    except BaseException:
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)
        raise

    output_bindings = {
        name.removesuffix(".npy"): _binding(path) for name, path in outputs.items()
    }
    counts = {
        "candidate_train_negatives": sum(
            row["split"] == "train" and int(row["label"]) == 0
            for row in candidate_rows
        ),
        "candidate_train_positives": sum(
            row["split"] == "train" and int(row["label"]) == 1
            for row in candidate_rows
        ),
        "consumed_physical_positives": len(physical_rows),
        "candidate_train_hard_negatives": sum(
            row["split"] == "train"
            and int(row["label"]) == 0
            and hard_negative_group is not None
            and row.get("source_group") == hard_negative_group
            for row in candidate_rows
        ),
        "materialized_negative": sum(
            source == "candidate" and int(candidate_rows[index]["label"]) == 0
            for source, index in plan
        ),
        "materialized_positive": sum(
            source == "physical" or int(candidate_rows[index]["label"]) == 1
            for source, index in plan
        ),
    }
    metadata = {
        "schema_version": 2,
        "deployment_qualification": False,
        "cache_role": "detector_student_distillation",
        "cache_specialization": "detector_conditioned_ordered_state_verifier_train_only_v2",
        "split_policy": {
            "included": ["train"],
            "excluded": ["validation", "test"],
            "test_used_for_training": False,
        },
        "recipe": "corrected_causal_candidates_plus_weighted_hard_negatives_and_stackchan_replay_v2",
        "seed": seed,
        "sample_count": len(plan),
        "feature_shape": [len(plan), 260, 40],
        "target_shape": [len(plan), student_output_frames],
        "teacher_logit_shape": [len(plan), student_output_frames, topology.state_count],
        "alignment_offset_frames": alignment_offset,
        "student_output_frames": student_output_frames,
        "positive_repeats": positive_repeats,
        "physical_repeats": physical_repeats,
        "hard_negative_replay": {
            "source_group": hard_negative_group,
            "repeats": hard_negative_repeats,
            "test_used_for_training": False,
        },
        "counts": counts,
        "topology": {
            "phrase_id": "kizz-control",
            "text": "Kizz Control",
            "phones": list(topology.phones),
            "states_per_phone": topology.states_per_phone,
            "state_count": topology.state_count,
        },
        "selected_teacher": {
            "best_weights": _binding(teacher_weights),
        },
        "teacher_training": _binding(teacher_training),
        "detector_teacher_gate": _binding(gate_path),
        "source_candidate_corpus": _binding(candidate_corpus),
        "source_candidate_features": _binding(candidate_features),
        "consumed_physical_corpus": _binding(physical_corpus),
        "consumed_physical_features": _binding(physical_features),
        "source_candidate_corpus_sha256": sha256_file(candidate_corpus),
        "outputs": output_bindings,
        "cache_sha256": sha256_json(output_bindings),
        "limitations": [
            "consumed physical captures are training evidence, not qualification evidence",
            "validation and test candidate rows were excluded from materialization",
        ],
    }
    _atomic_json(metadata_path, metadata)
    return {
        "cache": str(metadata_path),
        "cache_sha256": sha256_file(metadata_path),
        "sample_count": len(plan),
        "counts": counts,
        "outputs": output_bindings,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-corpus", type=Path, required=True)
    parser.add_argument("--candidate-features", type=Path, required=True)
    parser.add_argument("--physical-corpus", type=Path, required=True)
    parser.add_argument("--physical-features", type=Path, required=True)
    parser.add_argument("--detector-teacher-gate", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--positive-repeats", type=int, default=4)
    parser.add_argument("--physical-repeats", type=int, default=32)
    parser.add_argument("--hard-negative-group")
    parser.add_argument("--hard-negative-repeats", type=int, default=1)
    parser.add_argument("--alignment-offset", type=int, default=21)
    parser.add_argument("--student-output-frames", type=int, default=66)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=6053)
    args = parser.parse_args(argv)
    try:
        report = build(**vars(args))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
