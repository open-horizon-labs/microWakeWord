#!/usr/bin/env python3
"""Train a new offline full-context Kizz teacher.

The teacher is deliberately separate from the firmware student. It learns
frame-level Kizz state logits from the training split and writes a provenance
bound checkpoint plus a JSON training report.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import tensorflow as tf

from microwakeword.kizz_batch_mixture import (
    validate_declared_mixture,
    validate_realized_mixture,
)
from microwakeword.kizz_data_contract import sha256_file as balance_sha256_file
from microwakeword.kizz_data_contract import validate_balance_manifest
from microwakeword.kizz_feature_archive import (
    decode_frontend_features,
    open_feature_archive,
)
from microwakeword.kizz_teacher import (
    NegativeSource,
    TeacherBatchSequence,
    build_teacher,
    teacher_loss,
)
from microwakeword.ordered_state import (
    KIZZ_SINGLE_STATE_TOPOLOGY,
    KIZZ_TOPOLOGY,
    OrderedStateTopology,
)
from microwakeword.wake_phrase import HI_FI_KIZZ, WAKE_PHRASES, get_wake_phrase

sha256_file = balance_sha256_file


def resolve_topology(args: argparse.Namespace) -> OrderedStateTopology:
    phrase_id = getattr(args, "phrase_id", HI_FI_KIZZ.phrase_id)
    phones = get_wake_phrase(phrase_id).phones
    if args.topology == "single" and args.states_per_phone not in (None, 1):
        raise ValueError("--topology single requires --states-per-phone 1 or omission")
    if args.topology == "legacy" and args.states_per_phone not in (None, 3):
        raise ValueError("--topology legacy requires --states-per-phone 3 or omission")
    if args.topology == "double" and args.states_per_phone not in (None, 2):
        raise ValueError("--topology double requires --states-per-phone 2 or omission")
    if args.states_per_phone is not None:
        return OrderedStateTopology(phones, args.states_per_phone)
    if args.topology == "single":
        return OrderedStateTopology(phones, 1)
    if args.topology == "double":
        return OrderedStateTopology(phones, 2)
    return OrderedStateTopology(phones, 3)


def parse_source(value: str) -> NegativeSource:
    if "=" not in value:
        raise argparse.ArgumentTypeError("negative source must be ID=PATH")
    source_id, raw_path = value.split("=", 1)
    if not source_id or not raw_path:
        raise argparse.ArgumentTypeError("negative source must be ID=PATH")
    path = Path(raw_path).resolve()
    if not (path.is_dir() or (path.is_file() and path.suffix == ".npy")):
        raise argparse.ArgumentTypeError(
            f"negative source does not exist or is not a .npy cache: {path}"
        )
    return NegativeSource(source_id, path)


def parse_probability(value: str) -> tuple[str, float]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("source probability must be ID=PROBABILITY")
    source_id, raw_probability = value.split("=", 1)
    try:
        probability = float(raw_probability)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "source probability must be numeric"
        ) from error
    if not source_id or probability < 0:
        raise argparse.ArgumentTypeError("source probability must be non-negative")
    return source_id, probability


def parse_source_group(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("source group must be ID=GROUP")
    source_id, group = value.split("=", 1)
    if not source_id or not group:
        raise argparse.ArgumentTypeError("source group must be ID=GROUP")
    return source_id, group


def validate_feature_provenance(
    path: Path,
    positive_features: Path,
    positive_targets: Path,
    topology: OrderedStateTopology,
) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("schema_version") != 3
        or report.get("recipe") != "kizz_aligned_teacher_features_v3"
    ):
        raise ValueError("feature provenance is not the aligned canonical-v3 recipe")
    if (
        report.get("state_count") != topology.state_count
        or report.get("states_per_phone") != topology.states_per_phone
    ):
        raise ValueError("feature provenance topology does not match training")
    if report.get("include_inherited_alignments") is not False:
        raise ValueError("clean-slate teacher may not consume inherited alignments")
    features = np.load(positive_features, mmap_mode="r")
    targets = np.load(positive_targets, mmap_mode="r")
    expected = int(report.get("positive_counts", {}).get("train", -1))
    if len(features) != expected or len(targets) != expected:
        raise ValueError(
            "feature provenance positive count does not match training arrays"
        )
    if any(
        item.get("split") == "train"
        and item.get("variant")
        not in {"clean", "overlay-0", "overlay-1", "overlay-2", "overlay-3"}
        for item in report.get("examples", [])
    ):
        raise ValueError("feature provenance contains an undeclared training variant")
    return report


def declared_mixture_summary(
    probabilities: dict[str, float], source_groups: dict[str, str]
) -> dict:
    if set(probabilities) != set(source_groups):
        raise ValueError(
            "every negative source needs exactly one probability and group"
        )
    total = float(sum(probabilities.values()))
    if total <= 0:
        raise ValueError("negative source probabilities need positive mass")
    group_shares: dict[str, float] = {"canonical_positive": 0.5}
    for source_id, probability in probabilities.items():
        group = source_groups[source_id]
        group_shares[group] = group_shares.get(group, 0.0) + 0.5 * probability / total
    classes = {
        name: {"sampling_share": 0.5, "weighted_pressure_share": 0.5}
        for name in ("positive", "negative")
    }
    groups = {
        name: {"sampling_share": share, "weighted_pressure_share": share}
        for name, share in group_shares.items()
    }
    return {"classes": classes, "groups": groups}


def positive_source_balance_report(provenance: dict, guard: dict) -> dict:
    """Measure source diversity over unique parent recordings, not overlays."""
    split_guards = guard.get("splits")
    examples = provenance.get("examples")
    if not isinstance(split_guards, dict) or not isinstance(examples, list):
        raise TypeError("positive source guard and feature examples are required")
    reports = {}
    violations = []
    for split, limits in split_guards.items():
        if not isinstance(limits, dict):
            raise TypeError(f"positive source guard for {split} must be an object")
        parents = {}
        for row in examples:
            if row.get("split") != split:
                continue
            parent = str(row.get("parent_source_id", ""))
            family = row.get("provider") or row.get("source_group")
            if not parent or not family:
                violations.append(
                    {
                        "split": split,
                        "reason": "missing_parent_or_source_family",
                        "source_id": row.get("source_id"),
                    }
                )
                continue
            family = str(family)
            previous = parents.setdefault(parent, family)
            if previous != family:
                raise ValueError(
                    f"positive parent {parent} has conflicting source families"
                )
        counts = Counter(parents.values())
        total = sum(counts.values())
        shares = (
            {family: count / total for family, count in sorted(counts.items())}
            if total
            else {}
        )
        minimum_families = int(limits.get("minimum_families", 1))
        maximum_family_share = float(limits.get("maximum_family_share", 1.0))
        if len(counts) < minimum_families:
            violations.append(
                {
                    "split": split,
                    "reason": "too_few_source_families",
                    "actual": len(counts),
                    "minimum": minimum_families,
                }
            )
        if shares and max(shares.values()) > maximum_family_share + 1e-12:
            dominant = max(shares, key=lambda family: (shares[family], family))
            violations.append(
                {
                    "split": split,
                    "reason": "source_family_overrepresented",
                    "family": dominant,
                    "actual_share": shares[dominant],
                    "maximum_share": maximum_family_share,
                }
            )
        reports[split] = {
            "unique_parent_count": total,
            "family_counts": dict(sorted(counts.items())),
            "family_shares": shares,
            "limits": {
                "minimum_families": minimum_families,
                "maximum_family_share": maximum_family_share,
            },
        }
    return {
        "schema_version": 1,
        "qualified": not violations,
        "splits": reports,
        "violations": violations,
    }


def training_positive_source_families(provenance: dict) -> list[str]:
    """Return the actual source family for every materialized train row.

    The order is the feature-array order written by the materializer.  This is
    the data-contract bridge that turns inventory diversity into realized batch
    diversity instead of assuming uniform row sampling is representative.
    """
    examples = provenance.get("examples")
    if not isinstance(examples, list):
        raise TypeError("feature provenance examples are required")
    families = []
    for row in examples:
        if row.get("split") != "train":
            continue
        family = row.get("provider") or row.get("source_group")
        if not family:
            raise ValueError(
                f"positive feature row has no source family: {row.get('source_id')}"
            )
        families.append(str(family))
    return families


def validate_realized_positive_sampling(
    counts: Mapping[str, int], guard: dict
) -> dict:
    """Fail closed when training did not actually sample provider diversity."""
    minimum_families = int(guard.get("minimum_families", 1))
    minimum_share = float(guard.get("minimum_family_share", 0.0))
    maximum_share = float(guard.get("maximum_family_share", 1.0))
    total = int(sum(int(value) for value in counts.values()))
    violations = []
    shares = {
        str(family): int(count) / total
        for family, count in sorted(counts.items())
        if int(count) > 0 and total
    }
    if len(shares) < minimum_families:
        violations.append(
            {
                "reason": "too_few_realized_positive_families",
                "actual": len(shares),
                "minimum": minimum_families,
            }
        )
    for family, share in shares.items():
        if share < minimum_share - 1e-12:
            violations.append(
                {
                    "reason": "realized_positive_family_underrepresented",
                    "family": family,
                    "actual_share": share,
                    "minimum_share": minimum_share,
                }
            )
        if share > maximum_share + 1e-12:
            violations.append(
                {
                    "reason": "realized_positive_family_overrepresented",
                    "family": family,
                    "actual_share": share,
                    "maximum_share": maximum_share,
                }
            )
    return {
        "qualified": not violations,
        "mode": guard.get("mode"),
        "total_samples": total,
        "family_counts": dict(sorted((str(k), int(v)) for k, v in counts.items())),
        "family_shares": shares,
        "limits": {
            "minimum_families": minimum_families,
            "minimum_family_share": minimum_share,
            "maximum_family_share": maximum_share,
        },
        "violations": violations,
    }


def realized_mixture_ledger(
    sequence, guard: dict, source_groups: dict[str, str]
) -> dict:
    group_counts = {"canonical_positive": int(sequence.positive_sample_count)}
    for source_id, count in sequence.negative_source_sample_counts.items():
        group = source_groups[source_id]
        group_counts[group] = group_counts.get(group, 0) + int(count)
    positive = int(sequence.positive_sample_count)
    negative = int(sum(sequence.negative_source_sample_counts.values()))
    total = positive + negative
    if total <= 0:
        raise ValueError("training produced no realized samples")
    class_counts = {"positive": positive, "negative": negative}

    def rows(counts):
        return {
            name: {
                "samples": count,
                "share": count / total,
                "weighted_pressure_share": count / total,
            }
            for name, count in sorted(counts.items())
        }

    return {
        "schema_version": 1,
        "total_samples": total,
        "mixture_guard": guard,
        "realized_classes": rows(class_counts),
        "realized_groups": rows(group_counts),
        "negative_source_samples": dict(
            sorted(sequence.negative_source_sample_counts.items())
        ),
        "positive_source_samples": dict(
            sorted(getattr(sequence, "positive_source_sample_counts", {}).items())
        ),
    }


def scheduled_sequence_weight(
    step: int, *, weight: float, every: int, start_step: int
) -> float:
    """Return a stable sequence-loss schedule for a zero-based train step."""
    completed = step + 1
    if completed < start_step or completed % every:
        return 0.0
    return float(weight)


def _validation_windows(sources, limit: int, seed: int):
    if limit < 1:
        raise ValueError("validation negative limit must be positive")
    loaded = []
    for source in sources:
        if source.path.is_file() and source.path.suffix == ".npy":
            values = np.load(source.path, mmap_mode="r")
        else:
            values = open_feature_archive(source.path)
        if len(values) == 0:
            raise ValueError(f"validation negative source is empty: {source.path}")
        loaded.append((source, values))
    examples = []
    rng = np.random.default_rng(seed)
    for index in range(limit):
        source, values = loaded[index % len(loaded)]
        item = decode_frontend_features(values[int(rng.integers(0, len(values)))])
        if item.shape == (260, 40):
            examples.append(item)
            continue
        if item.ndim != 2 or item.shape[1] != 40:
            raise ValueError(
                f"validation source has invalid feature shape: {source.path}"
            )
        if len(item) <= 260:
            window = np.zeros((260, 40), dtype=np.float32)
            window[: len(item)] = item
        else:
            start = (len(item) - 260) // 2
            window = item[start : start + 260]
        examples.append(window)
    return np.asarray(examples, dtype=np.float32)


def _sequence_scores(
    model, features: np.ndarray, topology: OrderedStateTopology, batch_size: int
):
    values = []
    for start in range(0, len(features), batch_size):
        logits = model(features[start : start + batch_size], training=False)
        from microwakeword.ordered_state import ordered_state_sequence_score

        values.extend(
            np.asarray(
                ordered_state_sequence_score(logits, topology).numpy(), dtype=np.float64
            )
        )
    return np.asarray(values, dtype=np.float64)


def evaluate_validation_checkpoint(
    model,
    positive_features: Path,
    positive_targets: Path,
    negative_sources,
    *,
    topology: OrderedStateTopology,
    negative_limit: int,
    batch_size: int,
    seed: int,
    keyword_frame_weight: float = 1.0,
):
    positives = np.asarray(np.load(positive_features, mmap_mode="r"), dtype=np.float32)
    targets = np.asarray(np.load(positive_targets, mmap_mode="r"), dtype=np.int32)
    if positives.ndim != 3 or positives.shape[1:] != (260, 40):
        raise ValueError("validation positive features must have shape [N, 260, 40]")
    if targets.ndim != 2 or targets.shape[0] != len(positives):
        raise ValueError(
            "validation positive features and targets have different counts"
        )
    if np.any(targets < 0) or np.any(targets >= topology.state_count):
        raise ValueError("validation target labels exceed the selected topology")
    negatives = _validation_windows(negative_sources, negative_limit, seed)
    positive_scores = _sequence_scores(model, positives, topology, batch_size)
    negative_scores = _sequence_scores(model, negatives, topology, batch_size)
    labels = np.concatenate(
        [
            np.ones(len(positives), dtype=np.float32),
            np.zeros(len(negatives), dtype=np.float32),
        ]
    )
    all_features = np.concatenate([positives, negatives], axis=0)
    all_targets = np.concatenate(
        [targets, np.full((len(negatives), targets.shape[1]), 1, dtype=np.int32)],
        axis=0,
    )
    logits = model(all_features, training=False)
    validation_loss = float(
        teacher_loss(
            logits,
            all_targets,
            labels,
            keyword_frame_weight=keyword_frame_weight,
            topology=topology,
        ).numpy()
    )
    candidates = sorted(set(positive_scores.tolist() + negative_scores.tolist()))
    if len(negative_scores):
        candidates.append(float(np.nextafter(np.max(negative_scores), math.inf)))
    ledger = []
    for threshold in sorted(set(candidates), reverse=True):
        false_accepts = int(np.sum(negative_scores >= threshold))
        detected = int(np.sum(positive_scores >= threshold))
        recall = detected / len(positive_scores) if len(positive_scores) else 0.0
        separation = float(np.min(positive_scores) - np.max(negative_scores))
        ledger.append(
            {
                "threshold": float(threshold),
                "false_accepts": false_accepts,
                "opportunity_recall": recall,
                "separation": separation,
                "validation_loss": validation_loss,
                "zero_false_accepts": false_accepts == 0,
            }
        )
    selected = max(
        ledger,
        key=lambda item: (
            item["zero_false_accepts"],
            item["opportunity_recall"],
            item["separation"],
            -item["validation_loss"],
            item["threshold"],
        ),
    )
    return {
        "selected": selected,
        "ledger": ledger,
        "positive_count": len(positive_scores),
        "negative_count": len(negative_scores),
    }


def train(args: argparse.Namespace) -> dict:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    balance_report = validate_balance_manifest(
        args.balance_manifest,
        args.balance_contract,
    )
    balance_report_path = output / "balance-report.json"
    balance_report_path.write_text(
        json.dumps(balance_report, indent=2, sort_keys=True) + "\n"
    )
    if not balance_report["qualified"]:
        raise ValueError(
            f"source-balance contract rejected manifest; see {balance_report_path}"
        )
    tf.keras.utils.set_random_seed(args.seed)
    topology = resolve_topology(args)
    feature_provenance = validate_feature_provenance(
        args.feature_provenance,
        args.positive_features,
        args.positive_targets,
        topology,
    )
    import yaml

    mixture_recipe = yaml.safe_load(
        args.batch_mixture_recipe.read_text(encoding="utf-8")
    )
    mixture_guard = mixture_recipe.get("mixture_guard")
    if not isinstance(mixture_guard, dict):
        raise TypeError("batch mixture recipe must contain mixture_guard")
    positive_source_guard = mixture_recipe.get("positive_source_guard")
    if not isinstance(positive_source_guard, dict):
        raise TypeError("batch mixture recipe must contain positive_source_guard")
    positive_balance = positive_source_balance_report(
        feature_provenance, positive_source_guard
    )
    positive_balance_path = output / "positive-source-balance-report.json"
    positive_balance_path.write_text(
        json.dumps(positive_balance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not positive_balance["qualified"]:
        raise ValueError(
            "positive source-balance guard rejected feature materialization; see "
            f"{positive_balance_path}"
        )
    positive_sampling_guard = mixture_recipe.get("positive_sampling_guard")
    if not isinstance(positive_sampling_guard, dict):
        raise TypeError("batch mixture recipe must contain positive_sampling_guard")
    if positive_sampling_guard.get("mode") != "uniform_family":
        raise ValueError("positive sampling mode must be uniform_family")
    positive_families = training_positive_source_families(feature_provenance)
    if len(positive_families) != len(
        np.load(args.positive_features, mmap_mode="r")
    ):
        raise ValueError("positive provenance does not match feature-array order")
    declared_family_count = len(set(positive_families))
    if declared_family_count < int(
        positive_sampling_guard.get("minimum_families", 1)
    ):
        raise ValueError("too few positive families for uniform-family sampling")
    declared = declared_mixture_summary(
        args.negative_source_probabilities, args.negative_source_groups
    )
    validate_declared_mixture(declared, mixture_guard)

    target_frames = int(np.load(args.positive_targets, mmap_mode="r").shape[1])
    target_values = np.load(args.positive_targets, mmap_mode="r")
    if np.any(target_values < 0) or np.any(target_values >= topology.state_count):
        raise ValueError(
            "positive target labels exceed the selected topology state count"
        )
    validation_target_values = np.load(args.validation_positive_targets, mmap_mode="r")
    if (
        validation_target_values.ndim != 2
        or validation_target_values.shape[1] != target_frames
    ):
        raise ValueError(
            "validation positive targets must have the same timeline width as training targets"
        )
    model = build_teacher(
        hidden_size=args.hidden_size,
        recurrent_layers=args.recurrent_layers,
        output_frames=target_frames,
        topology=topology,
    )
    if args.initial_weights is not None:
        model.load_weights(args.initial_weights)
    sequence = TeacherBatchSequence(
        args.positive_features,
        args.positive_targets,
        args.negative_source,
        batch_size=args.batch_size,
        seed=args.seed,
        steps_per_epoch=args.steps,
        negative_state=args.negative_state,
        negative_source_weights=[
            args.negative_source_probabilities.get(source.source_id, 1.0)
            for source in args.negative_source
        ],
        positive_source_families=positive_families,
        topology=topology,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    losses = []
    selection_ledger = []
    best_selection = None
    for step in range(args.steps):
        features, batch = sequence[step]
        with tf.GradientTape() as tape:
            logits = model(features, training=True)
            loss = teacher_loss(
                logits,
                batch["states"],
                batch["label"],
                frame_weight=args.frame_weight,
                sequence_weight=scheduled_sequence_weight(
                    step,
                    weight=args.sequence_weight,
                    every=args.sequence_every,
                    start_step=args.sequence_start_step,
                ),
                keyword_frame_weight=args.keyword_frame_weight,
                topology=topology,
            )
        gradients = tape.gradient(loss, model.trainable_variables)
        update_gradients = []
        update_variables = []
        for gradient, variable in zip(gradients, model.trainable_variables):
            if gradient is not None:
                update_gradients.append(gradient)
                update_variables.append(variable)
        clipped, gradient_norm = tf.clip_by_global_norm(
            update_gradients, args.gradient_clip_norm
        )
        updates = list(zip(clipped, update_variables))
        optimizer.apply_gradients(updates)
        loss_value = float(loss.numpy())
        losses.append(loss_value)
        if (step + 1) % args.validation_interval == 0 or step + 1 == args.steps:
            validation = evaluate_validation_checkpoint(
                model,
                args.validation_positive_features,
                args.validation_positive_targets,
                args.validation_negative_source,
                topology=topology,
                negative_limit=args.validation_negative_limit,
                batch_size=args.batch_size,
                seed=args.seed,
                keyword_frame_weight=args.keyword_frame_weight,
            )
            selected = validation["selected"]
            entry = {"step": step + 1, **validation}
            selection_ledger.append(entry)
            model.save_weights(output / f"checkpoint-{step + 1:06d}.weights.h5")
            if best_selection is None or (
                selected["zero_false_accepts"],
                selected["opportunity_recall"],
                selected["separation"],
                -selected["validation_loss"],
                selected["threshold"],
            ) > (
                best_selection["selected"]["zero_false_accepts"],
                best_selection["selected"]["opportunity_recall"],
                best_selection["selected"]["separation"],
                -best_selection["selected"]["validation_loss"],
                best_selection["selected"]["threshold"],
            ):
                best_selection = entry
                shutil.copyfile(
                    output / f"checkpoint-{step + 1:06d}.weights.h5",
                    output / "best.weights.h5",
                )
        if (step + 1) % args.log_interval == 0 or step == 0:
            print(
                json.dumps(
                    {
                        "step": step + 1,
                        "loss": loss_value,
                        "gradient_global_norm": float(gradient_norm.numpy()),
                        "best_validation": None
                        if best_selection is None
                        else best_selection["selected"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    model.save_weights(output / "last.weights.h5")
    model.save(output / "teacher.keras")
    sampling_ledger = realized_mixture_ledger(
        sequence, mixture_guard, args.negative_source_groups
    )
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
    sampling_ledger_path = output / "batch-mixture-ledger.json"
    sampling_ledger_path.write_text(
        json.dumps(sampling_ledger, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = {
        "schema_version": 1,
        "model": "kizz_offline_teacher",
        "input_shape": [260, 40],
        "output_shape": [target_frames, topology.state_count],
        "topology": {
            "phones": list(topology.phones),
            "states_per_phone": topology.states_per_phone,
            "state_names": list(topology.state_names),
        },
        "hidden_size": args.hidden_size,
        "recurrent_layers": args.recurrent_layers,
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "frame_weight": args.frame_weight,
        "keyword_frame_weight": args.keyword_frame_weight,
        "sequence_weight": args.sequence_weight,
        "sequence_every": args.sequence_every,
        "sequence_start_step": args.sequence_start_step,
        "gradient_clip_norm": args.gradient_clip_norm,
        "initial_weights": (
            str(args.initial_weights.resolve())
            if args.initial_weights is not None
            else None
        ),
        "initial_weights_sha256": (
            sha256_file(args.initial_weights)
            if args.initial_weights is not None
            else None
        ),
        "negative_state": args.negative_state,
        "negative_source_probabilities": args.negative_source_probabilities,
        "positive_features": str(args.positive_features.resolve()),
        "positive_features_sha256": sha256_file(args.positive_features),
        "positive_targets": str(args.positive_targets.resolve()),
        "positive_targets_sha256": sha256_file(args.positive_targets),
        "feature_provenance": str(args.feature_provenance.resolve()),
        "feature_provenance_sha256": sha256_file(args.feature_provenance),
        "feature_provenance_recipe": feature_provenance["recipe"],
        "balance_manifest": str(args.balance_manifest.resolve()),
        "balance_manifest_sha256": sha256_file(args.balance_manifest),
        "balance_contract": str(args.balance_contract.resolve()),
        "balance_report": str(balance_report_path),
        "balance_report_sha256": sha256_file(balance_report_path),
        "negative_sources": [
            {
                "id": source.source_id,
                "path": str(source.path),
                "group": args.negative_source_groups[source.source_id],
            }
            for source in args.negative_source
        ],
        "batch_mixture_recipe": str(args.batch_mixture_recipe.resolve()),
        "batch_mixture_recipe_sha256": sha256_file(args.batch_mixture_recipe),
        "positive_source_balance_report": str(positive_balance_path),
        "positive_source_balance_report_sha256": sha256_file(positive_balance_path),
        "positive_sampling_guard": positive_sampling_guard,
        "batch_mixture_ledger": str(sampling_ledger_path),
        "batch_mixture_ledger_sha256": sha256_file(sampling_ledger_path),
        "checkpoint_selection": "validation_zero_fp_opportunity_recall_separation_loss",
        "validation_positive_features": str(
            args.validation_positive_features.resolve()
        ),
        "validation_positive_targets": str(args.validation_positive_targets.resolve()),
        "validation_negative_sources": [
            {"id": source.source_id, "path": str(source.path)}
            for source in args.validation_negative_source
        ],
        "validation_negative_limit": args.validation_negative_limit,
        "validation_interval": args.validation_interval,
        "checkpoint_selection_ledger": selection_ledger,
        "best_validation": None
        if best_selection is None
        else best_selection["selected"],
        "last_loss": losses[-1],
        "mean_last_100_loss": float(np.mean(losses[-100:])),
    }
    (output / "teacher-training.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    return config


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-features", type=Path, required=True)
    parser.add_argument("--positive-targets", type=Path, required=True)
    parser.add_argument("--feature-provenance", type=Path, required=True)
    parser.add_argument("--balance-manifest", type=Path, required=True)
    parser.add_argument("--balance-contract", type=Path, required=True)
    parser.add_argument(
        "--negative-source", type=parse_source, action="append", required=True
    )
    parser.add_argument(
        "--negative-source-group",
        type=parse_source_group,
        action="append",
        required=True,
    )
    parser.add_argument("--batch-mixture-recipe", type=Path, required=True)
    parser.add_argument("--validation-positive-features", type=Path, required=True)
    parser.add_argument("--validation-positive-targets", type=Path, required=True)
    parser.add_argument(
        "--validation-negative-source",
        type=parse_source,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--negative-source-probability",
        type=parse_probability,
        action="append",
        default=[],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--recurrent-layers", type=int, default=2)
    parser.add_argument(
        "--topology",
        choices=("legacy", "double", "single"),
        default="legacy",
        help="ordered states per declared phrase phone: three, two, or one",
    )
    parser.add_argument(
        "--phrase-id",
        choices=tuple(sorted(WAKE_PHRASES)),
        default=HI_FI_KIZZ.phrase_id,
    )
    parser.add_argument("--states-per-phone", type=int)
    parser.add_argument(
        "--output-frames",
        type=int,
        help="Teacher timeline length; defaults to the positive-target width.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--frame-weight", type=float, default=0.25)
    parser.add_argument("--keyword-frame-weight", type=float, default=1.0)
    parser.add_argument("--sequence-weight", type=float, default=0.75)
    parser.add_argument("--sequence-every", type=int, default=10)
    parser.add_argument(
        "--sequence-start-step",
        type=int,
        default=1,
        help="First completed step eligible for sequence loss (frame warm-up).",
    )
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--initial-weights", type=Path)
    parser.add_argument("--negative-state", type=int, choices=(0, 1), default=1)
    parser.add_argument("--seed", type=int, default=24103)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--validation-negative-limit", type=int, default=4096)
    args = parser.parse_args(argv)
    if (
        args.steps < 1
        or args.learning_rate <= 0
        or args.frame_weight < 0
        or args.keyword_frame_weight <= 0
        or args.sequence_weight < 0
        or args.sequence_every < 1
        or args.sequence_start_step < 1
        or args.gradient_clip_norm <= 0
        or args.validation_interval < 1
        or args.validation_negative_limit < 1
    ):
        parser.error("invalid training objective or schedule")
    args.negative_source_probabilities = dict(args.negative_source_probability)
    args.negative_source_groups = dict(args.negative_source_group)
    if args.output_frames is not None:
        target_frames = int(np.load(args.positive_targets, mmap_mode="r").shape[1])
        if args.output_frames != target_frames:
            parser.error("--output-frames must match positive-targets second dimension")
    unknown = set(args.negative_source_probabilities) - {
        source.source_id for source in args.negative_source
    }
    if unknown:
        parser.error(f"probabilities reference unknown sources: {sorted(unknown)}")
    source_ids = {source.source_id for source in args.negative_source}
    if set(args.negative_source_probabilities) != source_ids:
        parser.error("every negative source requires an explicit probability")
    if set(args.negative_source_groups) != source_ids:
        parser.error("every negative source requires an explicit group")
    try:
        resolve_topology(args)
    except ValueError as error:
        parser.error(str(error))
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
