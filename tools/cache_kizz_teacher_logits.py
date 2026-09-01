#!/usr/bin/env python3
"""Freeze provenance-bound teacher logits for a detector-student run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from microwakeword.kizz_batch_mixture import (
    validate_declared_mixture,
    validate_realized_mixture,
)
from microwakeword.kizz_teacher import NegativeSource, TeacherBatchSequence, build_teacher
from microwakeword.ordered_state import OrderedStateTopology
from microwakeword.wake_phrase import KIZZ_CONTROL, WAKE_PHRASES, get_wake_phrase
if __package__:
    from tools.train_kizz_teacher import (
        declared_mixture_summary,
        realized_mixture_ledger,
        training_positive_source_families,
        validate_feature_provenance,
        validate_realized_positive_sampling,
        validation_selection_key,
    )
else:
    from train_kizz_teacher import (
        declared_mixture_summary,
        realized_mixture_ledger,
        training_positive_source_families,
        validate_feature_provenance,
        validate_realized_positive_sampling,
        validation_selection_key,
    )


DETECTOR_MINIMUM_RECALL = 0.95


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_artifact(path: Path) -> tuple[str, str]:
    """Hash a file directly or a directory as a stable relative-path tree."""
    path = path.resolve()
    if path.is_file():
        return sha256_file(path), "file_sha256_v1"
    if not path.is_dir():
        raise ValueError(f"artifact does not exist: {path}")
    digest = hashlib.sha256()
    digest.update(b"kizz-directory-tree-sha256-v1\0")
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"artifact directory is empty: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest(), "directory_tree_sha256_v1"


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_source(value: str) -> NegativeSource:
    if "=" not in value:
        raise argparse.ArgumentTypeError("negative source must be ID=PATH")
    source_id, raw_path = value.split("=", 1)
    path = Path(raw_path).resolve()
    if not source_id or not (
        path.is_dir() or (path.is_file() and path.suffix == ".npy")
    ):
        raise argparse.ArgumentTypeError(
            "source must be ID=existing RaggedMmap directory or .npy cache"
        )
    return NegativeSource(source_id, path)


def parse_probability(value: str) -> tuple[str, float]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("source probability must be ID=WEIGHT")
    source_id, raw_probability = value.split("=", 1)
    try:
        probability = float(raw_probability)
    except ValueError as error:
        raise argparse.ArgumentTypeError("source probability must be numeric") from error
    if not source_id or not math.isfinite(probability) or probability < 0:
        raise argparse.ArgumentTypeError(
            "source probability must be finite and non-negative"
        )
    return source_id, probability


def parse_source_group(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("source group must be ID=GROUP")
    source_id, group = value.split("=", 1)
    if not source_id or not group:
        raise argparse.ArgumentTypeError("source group must be ID=GROUP")
    return source_id, group


def _unique_map(values: Sequence[tuple[str, Any]], label: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise ValueError(f"duplicate {label} for {key!r}")
        result[key] = value
    return result


def _resolved_declared_path(raw_path: object, owner: Path, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} path is missing")
    path = Path(raw_path)
    return (path if path.is_absolute() else owner.parent / path).resolve()


def _require_file_binding(
    payload: Mapping[str, Any],
    *,
    path_key: str,
    hash_key: str,
    actual_path: Path,
    owner: Path,
    label: str,
) -> dict[str, str]:
    declared_path = _resolved_declared_path(payload.get(path_key), owner, label)
    actual_path = actual_path.resolve()
    if declared_path != actual_path:
        raise ValueError(f"{label} path does not match the selected input")
    declared_hash = payload.get(hash_key)
    actual_hash = sha256_file(actual_path)
    if declared_hash != actual_hash:
        raise ValueError(f"{label} SHA-256 does not match the selected input")
    return {"path": str(actual_path), "sha256": actual_hash}


def validate_teacher_training(
    path: Path,
    *,
    teacher_weights: Path,
    feature_provenance: Path,
    positive_features: Path,
    positive_targets: Path,
    topology: OrderedStateTopology,
    phrase_id: str,
) -> dict[str, Any]:
    """Validate a detector selection without applying a single-stage FAPH gate."""
    path = path.resolve()
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("schema_version") != 1
        or report.get("model") != "kizz_offline_teacher"
    ):
        raise ValueError("teacher training report has an unsupported schema or model")

    selection_min_recall = report.get("selection_min_recall")
    if (
        isinstance(selection_min_recall, bool)
        or not isinstance(selection_min_recall, (int, float))
        or not math.isfinite(float(selection_min_recall))
    ):
        raise ValueError("detector selection requires a finite selection_min_recall")
    selection_min_recall = float(selection_min_recall)
    if not DETECTOR_MINIMUM_RECALL <= selection_min_recall <= 1.0:
        raise ValueError(
            f"detector selection_min_recall must be at least {DETECTOR_MINIMUM_RECALL}"
        )
    if report.get("checkpoint_selection") != (
        "validation_min_false_accepts_subject_to_recall_floor"
    ):
        raise ValueError("teacher training report is not a detector-role selection")
    ledger = report.get("checkpoint_selection_ledger")
    if not isinstance(ledger, list) or not ledger:
        raise ValueError("detector selection lacks a checkpoint selection ledger")
    for entry in ledger:
        if (
            not isinstance(entry, Mapping)
            or isinstance(entry.get("step"), bool)
            or not isinstance(entry.get("step"), int)
            or entry["step"] < 1
            or not isinstance(entry.get("selected"), Mapping)
        ):
            raise ValueError("checkpoint selection ledger contains an invalid entry")
    winner = max(
        ledger,
        key=lambda entry: validation_selection_key(
            entry["selected"], selection_min_recall
        ),
    )
    selected_validation = report.get("best_validation")
    if selected_validation != winner["selected"]:
        raise ValueError("best_validation does not match the derived ledger winner")
    selected_recall = selected_validation.get("opportunity_recall")
    if (
        isinstance(selected_recall, bool)
        or not isinstance(selected_recall, (int, float))
        or not math.isfinite(float(selected_recall))
        or float(selected_recall) < selection_min_recall
    ):
        raise ValueError("selected teacher does not meet its detector recall floor")

    checkpoint_path = path.parent / f"checkpoint-{winner['step']:06d}.weights.h5"
    best_path = path.parent / "best.weights.h5"
    if teacher_weights.resolve() != best_path.resolve():
        raise ValueError("teacher weights input must be the selected best.weights.h5")
    checkpoint_hash = sha256_file(checkpoint_path)
    best_hash = sha256_file(best_path)
    if checkpoint_hash != best_hash:
        raise ValueError("best.weights.h5 is not byte-identical to the ledger winner")

    provenance_binding = _require_file_binding(
        report,
        path_key="feature_provenance",
        hash_key="feature_provenance_sha256",
        actual_path=feature_provenance,
        owner=path,
        label="feature provenance",
    )
    positive_features_binding = _require_file_binding(
        report,
        path_key="positive_features",
        hash_key="positive_features_sha256",
        actual_path=positive_features,
        owner=path,
        label="positive features",
    )
    positive_targets_binding = _require_file_binding(
        report,
        path_key="positive_targets",
        hash_key="positive_targets_sha256",
        actual_path=positive_targets,
        owner=path,
        label="positive targets",
    )

    phrase = get_wake_phrase(phrase_id)
    declared_topology = report.get("topology")
    if not isinstance(declared_topology, Mapping) or (
        tuple(declared_topology.get("phones", ())) != phrase.phones
        or declared_topology.get("states_per_phone") != topology.states_per_phone
    ):
        raise ValueError("teacher training topology does not match cache topology")
    target_frames = int(np.load(positive_targets, mmap_mode="r").shape[1])
    if report.get("input_shape") != [260, 40] or report.get("output_shape") != [
        target_frames,
        topology.state_count,
    ]:
        raise ValueError("teacher training tensor contract does not match cache inputs")
    if phrase_id == KIZZ_CONTROL.phrase_id and topology.states_per_phone != 1:
        raise ValueError("Kizz Control detector cache requires states_per_phone=1")
    for field in ("hidden_size", "recurrent_layers"):
        value = report.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"teacher training report lacks valid {field}")

    return {
        "report": report,
        "report_binding": {"path": str(path), "sha256": sha256_file(path)},
        "selected_teacher": {
            "role": "detector",
            "step": winner["step"],
            "best_weights": {"path": str(best_path.resolve()), "sha256": best_hash},
            "checkpoint_weights": {
                "path": str(checkpoint_path.resolve()),
                "sha256": checkpoint_hash,
            },
            "selection_min_recall": selection_min_recall,
            "selected_opportunity_recall": float(selected_recall),
        },
        "feature_provenance": provenance_binding,
        "positive_features": positive_features_binding,
        "positive_targets": positive_targets_binding,
    }


def validate_provenance_manifest_bindings(
    provenance: Mapping[str, Any], owner: Path
) -> list[dict[str, str]]:
    """Re-hash every path/hash manifest pair carried by canonical-v3 provenance."""
    candidates: list[tuple[str, Mapping[str, Any], str, str]] = []
    for index, item in enumerate(provenance.get("positive_manifests", ())):
        if not isinstance(item, Mapping):
            raise ValueError("positive manifest binding must be an object")
        candidates.append((f"positive_manifests[{index}]", item, "path", "sha256"))
    for name in (
        "negative_manifest",
        "background_manifest",
        "rir_manifest",
        "source_pronunciation_audit",
    ):
        item = provenance.get(name)
        if item is not None:
            if not isinstance(item, Mapping):
                raise ValueError(f"{name} binding must be an object")
            candidates.append((name, item, "path", "sha256"))
    audit = provenance.get("source_pronunciation_audit")
    if isinstance(audit, Mapping) and (
        "source_manifest" in audit or "source_manifest_sha256" in audit
    ):
        candidates.append(
            (
                "source_pronunciation_audit.source_manifest",
                audit,
                "source_manifest",
                "source_manifest_sha256",
            )
        )

    bindings = []
    for label, item, path_key, hash_key in candidates:
        declared_path = _resolved_declared_path(item.get(path_key), owner, label)
        declared_hash = item.get(hash_key)
        actual_hash = sha256_file(declared_path)
        if declared_hash != actual_hash:
            raise ValueError(f"{label} SHA-256 does not match its declared manifest")
        bindings.append(
            {"id": label, "path": str(declared_path), "sha256": actual_hash}
        )
    return bindings


def _source_bindings(
    sources: Sequence[NegativeSource], groups: Mapping[str, str]
) -> list[dict[str, str]]:
    bindings = []
    for source in sources:
        digest, mode = sha256_artifact(source.path)
        bindings.append(
            {
                "id": source.source_id,
                "path": str(source.path.resolve()),
                "group": groups[source.source_id],
                "sha256": digest,
                "sha256_mode": mode,
            }
        )
    return bindings


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-weights", type=Path, required=True)
    parser.add_argument("--teacher-training", type=Path, required=True)
    parser.add_argument("--positive-features", type=Path, required=True)
    parser.add_argument("--positive-targets", type=Path, required=True)
    parser.add_argument("--feature-provenance", type=Path, required=True)
    parser.add_argument(
        "--negative-source", type=parse_source, action="append", required=True
    )
    parser.add_argument(
        "--negative-source-probability",
        type=parse_probability,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--negative-source-group",
        type=parse_source_group,
        action="append",
        required=True,
    )
    parser.add_argument("--batch-mixture-recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=24104)
    parser.add_argument(
        "--alignment-offset",
        type=int,
        default=21,
        help="Teacher 30-ms frames to skip before the student's first output.",
    )
    parser.add_argument("--student-output-frames", type=int, default=66)
    parser.add_argument(
        "--phrase-id",
        choices=tuple(sorted(WAKE_PHRASES)),
        default=KIZZ_CONTROL.phrase_id,
    )
    parser.add_argument("--states-per-phone", type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args(argv)
    if args.steps < 1 or args.batch_size < 2 or args.batch_size % 2:
        parser.error("steps must be positive and batch-size must be even")

    try:
        source_ids = [source.source_id for source in args.negative_source]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("negative source IDs must be unique")
        probabilities = _unique_map(
            args.negative_source_probability, "negative source probability"
        )
        groups = _unique_map(args.negative_source_group, "negative source group")
        if set(probabilities) != set(source_ids) or set(groups) != set(source_ids):
            raise ValueError(
                "every negative source needs exactly one explicit probability and group"
            )
        if sum(probabilities.values()) <= 0:
            raise ValueError("negative source probabilities need positive mass")

        positive_target_width = int(
            np.load(args.positive_targets, mmap_mode="r").shape[1]
        )
        if args.alignment_offset < 0 or (
            args.alignment_offset + args.student_output_frames > positive_target_width
        ):
            raise ValueError("teacher alignment slice does not fit the teacher timeline")
        phrase_spec = get_wake_phrase(args.phrase_id)
        topology = OrderedStateTopology(phrase_spec.phones, args.states_per_phone)
        provenance = validate_feature_provenance(
            args.feature_provenance,
            args.positive_features,
            args.positive_targets,
            topology,
        )
        positive_families = training_positive_source_families(provenance)
        if len(positive_families) != len(
            np.load(args.positive_features, mmap_mode="r")
        ):
            raise ValueError("positive provenance does not match feature-array order")

        training = validate_teacher_training(
            args.teacher_training,
            teacher_weights=args.teacher_weights,
            feature_provenance=args.feature_provenance,
            positive_features=args.positive_features,
            positive_targets=args.positive_targets,
            topology=topology,
            phrase_id=args.phrase_id,
        )
        mixture_recipe = yaml.safe_load(
            args.batch_mixture_recipe.read_text(encoding="utf-8")
        )
        if not isinstance(mixture_recipe, Mapping):
            raise ValueError("batch mixture recipe must be an object")
        mixture_guard = mixture_recipe.get("mixture_guard")
        positive_sampling_guard = mixture_recipe.get("positive_sampling_guard")
        if not isinstance(mixture_guard, Mapping):
            raise ValueError("batch mixture recipe must contain mixture_guard")
        if not isinstance(positive_sampling_guard, Mapping):
            raise ValueError(
                "batch mixture recipe must contain positive_sampling_guard"
            )
        if positive_sampling_guard.get("mode") != "uniform_family":
            raise ValueError("positive sampling mode must be uniform_family")
        validate_declared_mixture(
            declared_mixture_summary(probabilities, groups), mixture_guard
        )
        if len(set(positive_families)) < int(
            positive_sampling_guard.get("minimum_families", 1)
        ):
            raise ValueError("too few positive families for uniform-family sampling")
        manifest_bindings = validate_provenance_manifest_bindings(
            provenance, args.feature_provenance.resolve()
        )
        negative_bindings = _source_bindings(args.negative_source, groups)
    except (OSError, ValueError, TypeError, json.JSONDecodeError, yaml.YAMLError) as error:
        parser.error(str(error))

    report = training["report"]
    teacher = build_teacher(
        hidden_size=report["hidden_size"],
        recurrent_layers=report["recurrent_layers"],
        output_frames=positive_target_width,
        topology=topology,
    )
    teacher.load_weights(args.teacher_weights)
    sequence = TeacherBatchSequence(
        args.positive_features,
        args.positive_targets,
        args.negative_source,
        batch_size=args.batch_size,
        seed=args.seed,
        steps_per_epoch=args.steps,
        negative_source_weights=[
            probabilities[source.source_id] for source in args.negative_source
        ],
        positive_source_families=positive_families,
        topology=topology,
    )
    total = args.steps * args.batch_size
    features = np.empty((total, 260, 40), dtype=np.float16)
    targets = np.empty((total, args.student_output_frames), dtype=np.int8)
    labels = np.empty((total,), dtype=np.float16)
    logits = np.empty(
        (total, args.student_output_frames, topology.state_count), dtype=np.float16
    )
    for step in range(args.steps):
        x, batch = sequence[step]
        start = step * args.batch_size
        end = start + args.batch_size
        features[start:end] = x
        targets[start:end] = batch["states"][
            :, args.alignment_offset : args.alignment_offset + args.student_output_frames
        ]
        labels[start:end] = batch["label"]
        teacher_logits = teacher.predict(x, verbose=0)
        logits[start:end] = teacher_logits[
            :, args.alignment_offset : args.alignment_offset + args.student_output_frames
        ]
        if (step + 1) % 50 == 0 or step == 0:
            print(json.dumps({"step": step + 1, "total": args.steps}), flush=True)

    sampling_ledger = realized_mixture_ledger(sequence, mixture_guard, groups)
    validate_realized_mixture(sampling_ledger, mixture_guard)
    positive_sampling = validate_realized_positive_sampling(
        sequence.positive_source_sample_counts, positive_sampling_guard
    )
    if not positive_sampling["qualified"]:
        raise ValueError(
            "realized positive provider sampling violated its contract: "
            + json.dumps(positive_sampling["violations"], sort_keys=True)
        )
    sampling_ledger["realized_positive_sampling"] = positive_sampling

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "features": args.output.with_name("features.npy"),
        "targets": args.output.with_name("targets.npy"),
        "labels": args.output.with_name("labels.npy"),
        "teacher_logits": args.output.with_name("teacher_logits.npy"),
    }
    for name, values in (
        ("features", features),
        ("targets", targets),
        ("labels", labels),
        ("teacher_logits", logits),
    ):
        np.save(output_paths[name], values)
    output_bindings = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in output_paths.items()
    }
    declared = declared_mixture_summary(probabilities, groups)
    metadata = {
        "schema_version": 2,
        "deployment_qualification": False,
        "cache_role": "detector_student_distillation",
        "sample_count": total,
        "feature_shape": list(features.shape),
        "target_shape": list(targets.shape),
        "teacher_logit_shape": list(logits.shape),
        "teacher_training": training["report_binding"],
        "selected_teacher": training["selected_teacher"],
        "positive_features": training["positive_features"],
        "positive_targets": training["positive_targets"],
        "feature_provenance": training["feature_provenance"],
        "feature_provenance_recipe": provenance["recipe"],
        "provenance_manifest_bindings": manifest_bindings,
        "negative_sources": negative_bindings,
        "negative_source_probabilities": dict(sorted(probabilities.items())),
        "batch_mixture_recipe": {
            "path": str(args.batch_mixture_recipe.resolve()),
            "sha256": sha256_file(args.batch_mixture_recipe),
        },
        "declared_mixture": declared,
        "declared_mixture_sha256": sha256_json(declared),
        "realized_sampling_ledger": sampling_ledger,
        "realized_sampling_ledger_sha256": sha256_json(sampling_ledger),
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "teacher_output_frames": positive_target_width,
        "student_output_frames": args.student_output_frames,
        "alignment_offset_frames": args.alignment_offset,
        "alignment_basis": "student_valid_receptive_field_offset_64_frames_div_3",
        "topology": {
            "phrase_id": phrase_spec.phrase_id,
            "text": phrase_spec.text,
            "phones": list(phrase_spec.phones),
            "states_per_phone": topology.states_per_phone,
            "state_count": topology.state_count,
        },
        "outputs": output_bindings,
    }
    metadata["cache_sha256"] = sha256_json(output_bindings)
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
