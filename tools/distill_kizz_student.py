#!/usr/bin/env python3
"""Distill a compact causal Kizz student from a frozen teacher-logit cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tensorflow as tf

from microwakeword.distillation import distillation_loss
from microwakeword.ordered_state import OrderedStateTopology
from microwakeword.ordered_state_model import model as build_student


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_teacher_qualification(
    path: Path, teacher_weights: Path, continuous_path: Path
) -> tuple[dict, dict]:
    report = json.loads(path.read_text())
    if not report.get("qualified"):
        raise ValueError("teacher qualification report does not pass its hard gate")
    if report.get("gate_scope") != "teacher_clip_and_anchor_prequalification":
        raise ValueError("teacher report has the wrong qualification scope")
    expected = report.get("model_sha256")
    actual = sha256_file(teacher_weights)
    if expected != actual:
        raise ValueError(
            "teacher qualification report is for different weights: "
            f"expected {expected}, got {actual}"
        )
    if report.get("reasons"):
        raise ValueError("teacher qualification report retains failure reasons")
    continuous = json.loads(continuous_path.read_text())
    qualification = continuous.get("qualification", {})
    config = continuous.get("config", {})
    if (
        continuous.get("gate_scope") != "untouched_continuous_qualification"
        or continuous.get("qualified") is not True
        or qualification.get("qualified") is not True
    ):
        raise ValueError("continuous qualification report does not pass its hard gate")
    if continuous.get("model_sha256") != actual:
        raise ValueError("continuous qualification report is for different weights")
    if continuous.get("test_is_untouched") is not True:
        raise ValueError("continuous qualification test is not declared untouched")
    if float(config.get("min_negative_exposure_hours", 0)) < 100.0:
        raise ValueError("continuous qualification requires at least 100 hours")
    if float(config.get("max_faph_upper_95", float("inf"))) > 0.1:
        raise ValueError("continuous qualification FAPH guard is too permissive")
    if float(qualification.get("negative_exposure_seconds", 0)) < 100 * 3600:
        raise ValueError("continuous qualification measured less than 100 hours")
    if float(qualification.get("false_accepts_per_hour_upper_95", float("inf"))) > 0.1:
        raise ValueError("continuous qualification exceeds the FAPH upper bound")
    if int(qualification.get("locked_anchor_false_accepts", -1)) != 0:
        raise ValueError("continuous qualification accepted a locked anchor")
    if float(qualification.get("recall", 0)) < 0.9:
        raise ValueError("continuous qualification recall is below 90 percent")
    return report, continuous


def require_detector_teacher_gate(path: Path, teacher_weights: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("gate_scope")
        != "teacher_detector_synthetic_bootstrap_prequalification"
        or report.get("qualified") is not True
        or report.get("eligible_for_detector_distillation") is not True
    ):
        raise ValueError("detector teacher gate does not permit distillation")
    if (
        report.get("deployment_qualification") is not False
        or report.get("eligible_for_final_deployment") is not False
    ):
        raise ValueError("detector teacher gate must remain non-deployment evidence")
    checkpoint = report.get("selected_checkpoint", {})
    actual = sha256_file(teacher_weights)
    if checkpoint.get("best_weights_sha256") != actual:
        raise ValueError("detector teacher gate is for different weights")
    declared = checkpoint.get("best_weights_path")
    if (
        not isinstance(declared, str)
        or Path(declared).expanduser().resolve() != teacher_weights.resolve()
    ):
        raise ValueError("detector teacher gate weights path differs")
    selection = report.get("selection", {})
    if (
        selection.get("split") != "validation"
        or float(selection.get("minimum_recall", 0)) < 0.95
        or float(selection.get("opportunity_recall", 0)) < 0.95
    ):
        raise ValueError("detector teacher gate recall contract is too permissive")
    training = report.get("training_report", {})
    bindings = report.get("bindings", {})
    references = [training] + list(bindings.values()) if isinstance(bindings, dict) else []
    if not references:
        raise ValueError("detector teacher gate has no provenance bindings")
    for binding in references:
        if not isinstance(binding, dict):
            raise ValueError("detector teacher gate provenance binding is malformed")
        raw_path = binding.get("path")
        expected = binding.get("sha256")
        if not isinstance(raw_path, str) or not isinstance(expected, str):
            raise ValueError("detector teacher gate provenance binding is incomplete")
        reference = Path(raw_path).expanduser().resolve()
        if not reference.is_file() or sha256_file(reference) != expected:
            raise ValueError("detector teacher gate provenance hash drift")
    topology = report.get("topology")
    if not isinstance(topology, dict):
        raise ValueError("detector teacher gate has no topology")
    return report


def detector_cache_teacher(cache_metadata: dict) -> tuple[Path, dict]:
    if (
        cache_metadata.get("schema_version") != 2
        or cache_metadata.get("cache_role") != "detector_student_distillation"
        or cache_metadata.get("deployment_qualification") is not False
    ):
        raise ValueError("cache is not a non-deployment detector distillation cache")
    selected = cache_metadata.get("selected_teacher", {})
    binding = selected.get("best_weights", {}) if isinstance(selected, dict) else {}
    raw_path = binding.get("path")
    expected = binding.get("sha256")
    if not isinstance(raw_path, str) or not isinstance(expected, str):
        raise ValueError("detector cache has no selected teacher binding")
    weights = Path(raw_path).expanduser().resolve()
    if not weights.is_file() or sha256_file(weights) != expected:
        raise ValueError("detector cache selected teacher hash drift")
    outputs = cache_metadata.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("detector cache has no output bindings")
    for name in ("features", "targets", "labels", "teacher_logits"):
        output = outputs.get(name, {})
        raw_output = output.get("path") if isinstance(output, dict) else None
        output_hash = output.get("sha256") if isinstance(output, dict) else None
        if not isinstance(raw_output, str) or not isinstance(output_hash, str):
            raise ValueError(f"detector cache has no {name} binding")
        output_path = Path(raw_output).expanduser().resolve()
        if not output_path.is_file() or sha256_file(output_path) != output_hash:
            raise ValueError(f"detector cache {name} hash drift")
    return weights, binding


def student_flags(
    num_states: int = 23, architecture: str = "control_mixconv"
) -> SimpleNamespace:
    if architecture == "control_mixconv":
        first_conv_filters = 48
        pointwise_filters = "96,96,96,96"
    elif architecture == "control_mixconv_small":
        # Keep v5c's exact stride, temporal kernels, and decoder-facing output
        # geometry while reducing the channel dimensions that dominate every
        # always-on embedded invoke.
        first_conv_filters = 24
        pointwise_filters = "48,48,48,48"
    else:
        raise ValueError(f"unsupported detector student architecture: {architecture}")
    return SimpleNamespace(
        pointwise_filters=pointwise_filters,
        residual_connection="0,0,0,0",
        repeat_in_block="1,1,1,1",
        mixconv_kernel_sizes="[3], [5], [7], [9]",
        first_conv_filters=first_conv_filters,
        first_conv_kernel_size=5,
        stride=3,
        num_states=num_states,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-prefix", type=Path, required=True)
    parser.add_argument("--teacher-qualification", type=Path)
    parser.add_argument("--continuous-qualification", type=Path)
    parser.add_argument("--detector-teacher-gate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--hard-weight", type=float, default=0.5)
    parser.add_argument("--teacher-weight", type=float, default=0.5)
    parser.add_argument(
        "--negative-state",
        type=int,
        default=1,
        help="hard frame state used for negative windows (1=silence)",
    )
    parser.add_argument("--sequence-weight", type=float, default=0.0)
    parser.add_argument(
        "--sequence-teacher-weight",
        type=float,
        default=0.0,
        help="match the teacher's ordered-state completion margin directly",
    )
    parser.add_argument(
        "--sequence-every",
        type=int,
        default=10,
        help="apply the slow sequence objective every N batches",
    )
    parser.add_argument("--init-weights", type=Path)
    parser.add_argument(
        "--student-architecture",
        choices=("control_mixconv", "control_mixconv_small"),
        default="control_mixconv",
    )
    parser.add_argument("--seed", type=int, default=24105)
    parser.add_argument("--log-interval", type=int, default=100)
    args = parser.parse_args(argv)
    if (
        args.steps < 1
        or args.batch_size < 1
        or args.sequence_every < 1
        or args.negative_state not in (0, 1)
        or args.sequence_teacher_weight < 0
    ):
        parser.error("steps, batch-size, and sequence-every must be positive")

    prefix = args.cache_prefix
    cache_metadata = json.loads(prefix.with_suffix(".json").read_text())
    if args.detector_teacher_gate is not None:
        try:
            teacher_weights, cache_teacher_binding = detector_cache_teacher(
                cache_metadata
            )
        except ValueError as error:
            parser.error(str(error))
    else:
        teacher_weights = Path(cache_metadata["teacher_weights"])
        cache_teacher_binding = None
    if args.detector_teacher_gate is not None:
        if args.teacher_qualification is not None or args.continuous_qualification is not None:
            parser.error(
                "--detector-teacher-gate is mutually exclusive with single-stage qualification"
            )
        qualification = require_detector_teacher_gate(
            args.detector_teacher_gate, teacher_weights
        )
        if (
            qualification["selected_checkpoint"]["best_weights_sha256"]
            != cache_teacher_binding["sha256"]
            or qualification["training_report"]["sha256"]
            != cache_metadata.get("teacher_training", {}).get("sha256")
        ):
            parser.error("detector cache and teacher gate provenance differ")
        continuous_qualification = None
        student_role = "permissive_detector_candidate_generator"
    else:
        if args.teacher_qualification is None or args.continuous_qualification is None:
            parser.error(
                "provide --detector-teacher-gate or both teacher and continuous qualifications"
            )
        qualification, continuous_qualification = require_teacher_qualification(
            args.teacher_qualification,
            teacher_weights,
            args.continuous_qualification,
        )
        student_role = "single_stage_wake_word"
    features = np.load(prefix.with_name("features.npy"), mmap_mode="r")
    targets = np.load(prefix.with_name("targets.npy"), mmap_mode="r")
    labels = np.load(prefix.with_name("labels.npy"), mmap_mode="r")
    teacher_logits = np.load(prefix.with_name("teacher_logits.npy"), mmap_mode="r")
    topology_payload = qualification.get("topology", {})
    try:
        topology = OrderedStateTopology(
            tuple(str(phone) for phone in topology_payload["phones"]),
            int(topology_payload["states_per_phone"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        parser.error(f"teacher qualification has invalid topology: {error}")
    cache_topology = cache_metadata.get("topology")
    if cache_topology is not None and (
        cache_topology.get("state_count") != topology.state_count
        or tuple(cache_topology.get("phones", ())) != topology.phones
        or cache_topology.get("states_per_phone") != topology.states_per_phone
    ):
        parser.error("distillation cache topology differs from teacher qualification")
    if (
        features.shape[0] != targets.shape[0]
        or features.shape[0] != labels.shape[0]
        or features.shape[0] != teacher_logits.shape[0]
    ):
        parser.error("cache arrays must contain the same number of samples")
    tf.keras.utils.set_random_seed(args.seed)
    if teacher_logits.shape[-1] != topology.state_count:
        parser.error("teacher logits do not match the qualified topology")
    student = build_student(
        student_flags(topology.state_count, args.student_architecture),
        (260, 40),
        None,
    )
    if args.init_weights is not None:
        student.load_weights(args.init_weights)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    rng = np.random.default_rng(args.seed)
    best_loss = float("inf")
    losses = []

    def _train_batch(x, hard, soft, sequence_labels, use_sequence: bool):
        with tf.GradientTape() as tape:
            logits = student(x, training=True)
            loss = distillation_loss(
                logits,
                soft,
                hard,
                temperature=args.temperature,
                hard_weight=args.hard_weight,
                teacher_weight=args.teacher_weight,
                sequence_weight=args.sequence_weight if use_sequence else 0.0,
                sequence_teacher_weight=(
                    args.sequence_teacher_weight if use_sequence else 0.0
                ),
                sequence_labels=sequence_labels,
                topology=topology,
            )
        gradients = tape.gradient(loss, student.trainable_variables)
        optimizer.apply_gradients(zip(gradients, student.trainable_variables))
        return loss

    @tf.function
    def train_batch_frame(x, hard, soft, sequence_labels):
        return _train_batch(x, hard, soft, sequence_labels, False)

    @tf.function
    def train_batch_sequence(x, hard, soft, sequence_labels):
        return _train_batch(x, hard, soft, sequence_labels, True)

    args.output.mkdir(parents=True, exist_ok=True)
    for step in range(args.steps):
        indexes = rng.integers(0, len(features), size=args.batch_size)
        hard_targets = np.asarray(targets[indexes], dtype=np.int32).copy()
        negative = np.asarray(labels[indexes]) < 0.5
        hard_targets[negative, :] = args.negative_state
        batch_args = (
            tf.convert_to_tensor(np.asarray(features[indexes], dtype=np.float32)),
            tf.convert_to_tensor(hard_targets),
            tf.convert_to_tensor(np.asarray(teacher_logits[indexes], dtype=np.float32)),
            tf.convert_to_tensor(np.asarray(labels[indexes], dtype=np.float32)),
        )
        if args.sequence_weight and (step + 1) % args.sequence_every == 0:
            loss = train_batch_sequence(*batch_args)
        else:
            loss = train_batch_frame(*batch_args)
        value = float(loss.numpy())
        losses.append(value)
        if value < best_loss:
            best_loss = value
            student.save_weights(args.output / "best.weights.h5")
        if (step + 1) % args.log_interval == 0 or step == 0:
            print(
                json.dumps({"step": step + 1, "loss": value, "best_loss": best_loss}),
                flush=True,
            )

    student.save_weights(args.output / "last.weights.h5")
    student.save(args.output / "student.keras")
    metadata = {
        "schema_version": 1,
        "model": "ordered_state_causal_student_distilled",
        "student_role": student_role,
        "deployment_qualification": False,
        "teacher_gate_mode": (
            "detector_recall_prequalification"
            if args.detector_teacher_gate
            else "single_stage_teacher_prequalification"
        ),
        "input_shape": [260, 40],
        "output_shape": [66, topology.state_count],
        "student_architecture": args.student_architecture,
        "parameter_count": int(student.count_params()),
        "selected_weights": {
            "path": str((args.output / "best.weights.h5").resolve()),
            "sha256": sha256_file(args.output / "best.weights.h5"),
            "selection": "minimum_observed_training_batch_loss",
        },
        "last_weights": {
            "path": str((args.output / "last.weights.h5").resolve()),
            "sha256": sha256_file(args.output / "last.weights.h5"),
        },
        "topology": {
            "phrase_id": topology_payload.get("phrase_id")
            or cache_metadata.get("topology", {}).get("phrase_id"),
            "text": topology_payload.get("text")
            or cache_metadata.get("topology", {}).get("text"),
            "phones": list(topology.phones),
            "states_per_phone": topology.states_per_phone,
            "state_count": topology.state_count,
        },
        "cache_prefix": str(prefix.resolve()),
        "cache_files_sha256": {
            name: sha256_file(prefix.with_name(name))
            for name in (
                "features.npy",
                "targets.npy",
                "labels.npy",
                "teacher_logits.npy",
            )
        },
        "teacher_qualification": (
            str(args.teacher_qualification.resolve())
            if args.teacher_qualification is not None
            else None
        ),
        "teacher_qualification_sha256": (
            sha256_file(args.teacher_qualification)
            if args.teacher_qualification is not None
            else None
        ),
        "teacher_qualification_threshold": (
            qualification["selection"]["threshold"]
            if args.detector_teacher_gate is not None
            else qualification["validation"]["operating_point"]["threshold"]
        ),
        "detector_teacher_gate": (
            str(args.detector_teacher_gate.resolve())
            if args.detector_teacher_gate is not None
            else None
        ),
        "detector_teacher_gate_sha256": (
            sha256_file(args.detector_teacher_gate)
            if args.detector_teacher_gate is not None
            else None
        ),
        "continuous_qualification": (
            str(args.continuous_qualification.resolve())
            if args.continuous_qualification is not None
            else None
        ),
        "continuous_qualification_sha256": (
            sha256_file(args.continuous_qualification)
            if args.continuous_qualification is not None
            else None
        ),
        "continuous_qualification_threshold": (
            continuous_qualification["qualification"]["threshold"]
            if continuous_qualification is not None
            else None
        ),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "temperature": args.temperature,
        "hard_weight": args.hard_weight,
        "teacher_weight": args.teacher_weight,
        "negative_state": args.negative_state,
        "sequence_weight": args.sequence_weight,
        "sequence_teacher_weight": args.sequence_teacher_weight,
        "sequence_every": args.sequence_every,
        "seed": args.seed,
        "best_loss": best_loss,
        "last_loss": losses[-1],
        "mean_last_100_loss": float(np.mean(losses[-100:])),
    }
    (args.output / "distillation-training.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
