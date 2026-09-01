#!/usr/bin/env python3
"""Extend a frozen detector-student cache with qualified StackChan positives.

The base cache remains immutable.  Qualified train-only device captures are
materialized as deterministic gain/timing variants, scored by the exact frozen
teacher bound to the base cache, and appended to a new cache.  Independently
qualified device validation captures are emitted separately and appended to
the original clean validation arrays for validation-only checkpoint selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from microwakeword.kizz_teacher import build_teacher
from microwakeword.ordered_state import OrderedStateTopology
from microwakeword.ordered_state_data import example_from_mapping, frame_state_targets
from microwakeword.wake_phrase import KIZZ_CONTROL
from tools.build_kizz_aligned_teacher_features_v3 import (
    CONTEXT_SAMPLES,
    SAMPLE_RATE,
    TARGET_FRAME_TIMES,
    _translated_record,
    apply_gain_db,
    frontend,
    load_audio,
    place_phrase_context,
)
from tools.build_kizz_phoneme_distillation_corpus import (
    load_device_training_rows,
    load_device_validation_rows,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _device_example(
    row: dict[str, Any],
    topology: OrderedStateTopology,
    *,
    desired_phrase_center_s: float | None,
    gain_db: float,
    speed: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    audio = load_audio(Path(str(row["path"])))
    adjusted = dict(row)
    if not 0.75 <= speed <= 1.25:
        raise ValueError("speed must stay inside the qualified perturbation range")
    if abs(speed - 1.0) > 1e-9:
        denominator = max(1, round(1000 * speed))
        actual_speed = denominator / 1000.0
        audio = resample_poly(audio, 1000, denominator).astype(np.float32)
        adjusted["phrase_span"] = {
            key: float(value) / actual_speed
            for key, value in row["phrase_span"].items()
        }
        adjusted["phone_spans"] = [
            {
                **span,
                "start_s": float(span["start_s"]) / actual_speed,
                "end_s": float(span["end_s"]) / actual_speed,
            }
            for span in row["phone_spans"]
        ]
    phrase = adjusted["phrase_span"]
    phrase_span = (float(phrase["start_s"]), float(phrase["end_s"]))
    context, translation = place_phrase_context(
        audio,
        phrase_span,
        desired_phrase_center_s=desired_phrase_center_s,
    )
    context = apply_gain_db(context, gain_db)
    translated = _translated_record(
        adjusted,
        translation,
        str(row["source_id"]),
        KIZZ_CONTROL,
    )
    targets = frame_state_targets(
        example_from_mapping(translated, expected_phones=topology.phones),
        TARGET_FRAME_TIMES,
        states_per_phone=topology.states_per_phone,
    )
    if targets is None:
        raise ValueError(f"device row produced no ordered targets: {row['source_id']}")
    return frontend(context), np.asarray(targets, dtype=np.int8)


def _materialize_rows(
    rows: Sequence[dict[str, Any]],
    topology: OrderedStateTopology,
    *,
    replicas: int,
    seed: int,
    augment: bool,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if not rows:
        return (
            np.empty((0, 260, 40), dtype=np.float16),
            np.empty((0, len(TARGET_FRAME_TIMES)), dtype=np.int8),
            [],
        )
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    ledger: list[dict[str, Any]] = []
    context_duration = CONTEXT_SAMPLES / SAMPLE_RATE
    for row in rows:
        phrase = row["phrase_span"]
        phrase_duration = float(phrase["end_s"]) - float(phrase["start_s"])
        for replica in range(replicas):
            variant_seed = int.from_bytes(
                hashlib.sha256(
                    f"{seed}\0{row['source_id']}\0{replica}".encode("utf-8")
                ).digest()[:8],
                "little",
            )
            rng = np.random.default_rng(variant_seed)
            if augment:
                low = 0.35 + phrase_duration * 0.5
                high = context_duration - 0.25 - phrase_duration * 0.5
                center = float(rng.uniform(low, max(low + 1e-6, high)))
                gain_db = float(rng.uniform(-3.0, 12.0))
                speed = float(rng.uniform(0.82, 1.18))
            else:
                center = None
                gain_db = 0.0
                speed = 1.0
            feature, target = _device_example(
                row,
                topology,
                desired_phrase_center_s=center,
                gain_db=gain_db,
                speed=speed,
            )
            features.append(feature)
            targets.append(target)
            ledger.append(
                {
                    "source_id": row["source_id"],
                    "source_audio_sha256": row["audio_sha256"],
                    "provider": row.get("provider"),
                    "voice": row.get("voice"),
                    "split": row.get("split"),
                    "replica": replica,
                    "seed": variant_seed,
                    "desired_phrase_center_s": center,
                    "gain_db": gain_db,
                    "speed": speed,
                }
            )
    return (
        np.asarray(features, dtype=np.float16),
        np.asarray(targets, dtype=np.int8),
        ledger,
    )


def _load_base_array(metadata: dict[str, Any], name: str) -> np.ndarray:
    binding = metadata.get("outputs", {}).get(name, {})
    path = Path(str(binding.get("path", ""))).resolve()
    if not path.is_file() or binding.get("sha256") != sha256_file(path):
        raise ValueError(f"base cache {name} binding drifted")
    return np.load(path, mmap_mode="r")


def _promoted_device_rows(path: Path) -> list[dict[str, Any]]:
    report = json.loads(path.read_text())
    if (
        report.get("kind")
        != "kizz_control_ordered_state_target_device_replay_features"
        or report.get("gate_scope")
        != "locked_test_only_target_channel_positive_features"
        or report.get("training_eligible") is not False
    ):
        raise ValueError("promoted device provenance is not consumed test evidence")
    rows = []
    for result in report.get("results", []):
        audio = Path(str(result.get("path", ""))).resolve()
        audio_hash = str(result.get("audio_sha256", ""))
        if not audio.is_file() or not audio_hash or sha256_file(audio) != audio_hash:
            raise ValueError("promoted target-device audio binding drifted")
        if not result.get("phrase_span") or not result.get("phone_spans"):
            raise ValueError("promoted target-device row lacks aligned phone spans")
        rows.append(
            {
                "source_id": f"consumed-device-positive:{result['capture_id']}",
                "path": str(audio),
                "audio_sha256": audio_hash,
                "provider": result.get("provider"),
                "voice": result.get("voice"),
                "split": "train",
                "phrase_span": result["phrase_span"],
                "phone_spans": result["phone_spans"],
                "source_group": "consumed_target_channel_hard_positive",
            }
        )
    if not rows:
        raise ValueError("promoted target-device provenance contains no rows")
    return rows


def extend(
    base_cache: Path,
    device_training_quality: Path,
    device_validation_quality: Path,
    base_validation_features: Path,
    base_validation_targets: Path,
    output: Path,
    *,
    replicas_per_device: int = 128,
    seed: int = 36013,
    promoted_device_provenance: Path | None = None,
    promoted_only: bool = False,
) -> dict[str, Any]:
    if replicas_per_device < 1:
        raise ValueError("replicas_per_device must be positive")
    metadata_path = base_cache.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("schema_version") != 2
        or metadata.get("cache_role") != "detector_student_distillation"
        or metadata.get("deployment_qualification") is not False
    ):
        raise ValueError("base cache is not a detector-student cache")
    topology_payload = metadata.get("topology", {})
    topology = OrderedStateTopology(
        tuple(str(phone) for phone in topology_payload.get("phones", ())),
        int(topology_payload.get("states_per_phone", 0)),
    )
    if topology.phones != KIZZ_CONTROL.phones or topology.states_per_phone != 1:
        raise ValueError("base cache topology is not single-state Kizz Control")
    offset = int(metadata.get("alignment_offset_frames", -1))
    output_frames = int(metadata.get("student_output_frames", -1))
    if offset < 0 or output_frames < 1 or offset + output_frames > len(TARGET_FRAME_TIMES):
        raise ValueError("base cache alignment contract is invalid")

    if promoted_only and promoted_device_provenance is None:
        raise ValueError("promoted_only requires promoted_device_provenance")
    original_train_rows = (
        [] if promoted_only else load_device_training_rows(device_training_quality)
    )
    promoted_rows = (
        _promoted_device_rows(promoted_device_provenance)
        if promoted_device_provenance is not None
        else []
    )
    train_rows = original_train_rows + promoted_rows
    validation_rows = (
        [] if promoted_only else load_device_validation_rows(device_validation_quality)
    )
    train_features, train_targets_full, train_ledger = _materialize_rows(
        train_rows,
        topology,
        replicas=replicas_per_device,
        seed=seed,
        augment=True,
    )
    validation_features, validation_targets_full, validation_ledger = _materialize_rows(
        validation_rows,
        topology,
        replicas=1,
        seed=seed,
        augment=False,
    )
    train_targets = train_targets_full[:, offset : offset + output_frames]
    validation_targets = validation_targets_full[:, offset : offset + output_frames]

    teacher_training_binding = metadata.get("teacher_training", {})
    teacher_training_path = Path(
        str(teacher_training_binding.get("path", ""))
    ).resolve()
    if (
        not teacher_training_path.is_file()
        or teacher_training_binding.get("sha256")
        != sha256_file(teacher_training_path)
    ):
        raise ValueError("base cache teacher training binding drifted")
    teacher_training = json.loads(teacher_training_path.read_text())
    teacher_binding = metadata.get("selected_teacher", {}).get("best_weights", {})
    teacher_weights = Path(str(teacher_binding.get("path", ""))).resolve()
    if not teacher_weights.is_file() or teacher_binding.get("sha256") != sha256_file(
        teacher_weights
    ):
        raise ValueError("base cache teacher weights binding drifted")
    teacher = build_teacher(
        hidden_size=int(teacher_training["hidden_size"]),
        recurrent_layers=int(teacher_training["recurrent_layers"]),
        output_frames=offset + output_frames,
        topology=topology,
    )
    teacher.load_weights(teacher_weights)
    teacher_logits = np.asarray(
        teacher.predict(np.asarray(train_features, dtype=np.float32), batch_size=64, verbose=0),
        dtype=np.float16,
    )[:, offset : offset + output_frames]

    base_features = _load_base_array(metadata, "features")
    base_targets = _load_base_array(metadata, "targets")
    base_labels = _load_base_array(metadata, "labels")
    base_logits = _load_base_array(metadata, "teacher_logits")
    features = np.concatenate((base_features, train_features)).astype(np.float16)
    targets = np.concatenate((base_targets, train_targets)).astype(np.int8)
    labels = np.concatenate(
        (base_labels, np.ones(len(train_features), dtype=np.float16))
    ).astype(np.float16)
    logits = np.concatenate((base_logits, teacher_logits)).astype(np.float16)

    base_validation_features_array = np.load(base_validation_features, mmap_mode="r")
    base_validation_targets_array = np.load(base_validation_targets, mmap_mode="r")
    if len(base_validation_features_array) != len(base_validation_targets_array):
        raise ValueError("base validation feature/target counts differ")
    combined_validation_features = np.concatenate(
        (base_validation_features_array, validation_features)
    ).astype(np.float32)
    combined_validation_targets = np.concatenate(
        (base_validation_targets_array, validation_targets_full)
    ).astype(np.int32)

    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "features": output / "features.npy",
        "targets": output / "targets.npy",
        "labels": output / "labels.npy",
        "teacher_logits": output / "teacher_logits.npy",
        "device_validation_features": output / "device-validation-features.npy",
        "device_validation_targets": output / "device-validation-targets.npy",
        "combined_validation_features": output / "validation-positive-features.npy",
        "combined_validation_targets": output / "validation-positive-targets.npy",
        "evaluation_feature_provenance": output
        / "evaluation-feature-provenance.json",
    }
    for name, values in (
        ("features", features),
        ("targets", targets),
        ("labels", labels),
        ("teacher_logits", logits),
        ("device_validation_features", validation_features.astype(np.float32)),
        ("device_validation_targets", validation_targets_full.astype(np.int32)),
        ("combined_validation_features", combined_validation_features),
        ("combined_validation_targets", combined_validation_targets),
    ):
        np.save(paths[name], values)

    base_provenance_binding = metadata.get("feature_provenance", {})
    base_provenance_path = Path(
        str(base_provenance_binding.get("path", ""))
    ).resolve()
    if (
        not base_provenance_path.is_file()
        or base_provenance_binding.get("sha256") != sha256_file(base_provenance_path)
    ):
        raise ValueError("base cache feature provenance binding drifted")
    evaluation_provenance = json.loads(base_provenance_path.read_text())
    evaluation_provenance["positive_counts"]["validation"] = len(
        combined_validation_features
    )
    evaluation_provenance["examples"].extend(
        {
            "source_id": f"{row['source_id']}::clean",
            "parent_source_id": row["source_id"],
            "split": "validation",
            "variant": "clean",
            "augmentation": None,
            "provider": row.get("provider"),
            "source_group": "device_channel_validation_positive",
            "source_audio_sha256": row.get("audio_sha256"),
        }
        for row in validation_rows
    )
    evaluation_provenance["device_validation_extension"] = {
        "quality_report": _binding(device_validation_quality),
        "base_provenance": _binding(base_provenance_path),
        "source_count": len(validation_rows),
    }
    paths["evaluation_feature_provenance"].write_text(
        json.dumps(evaluation_provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = dict(metadata)
    result.update(
        {
            "sample_count": len(features),
            "feature_shape": list(features.shape),
            "target_shape": list(targets.shape),
            "teacher_logit_shape": list(logits.shape),
            "outputs": {
                name: _binding(paths[name])
                for name in ("features", "targets", "labels", "teacher_logits")
            },
            "device_replay_extension": {
                "recipe": "qualified_stackchan_positive_cache_extension_v1",
                "base_cache": _binding(metadata_path),
                "training_quality": _binding(device_training_quality),
                "validation_quality": _binding(device_validation_quality),
                "base_validation_features": _binding(base_validation_features),
                "base_validation_targets": _binding(base_validation_targets),
                "replicas_per_device": replicas_per_device,
                "promoted_only": promoted_only,
                "seed": seed,
                "training_source_count": len(train_rows),
                "original_training_source_count": len(original_train_rows),
                "promoted_consumed_source_count": len(promoted_rows),
                "training_materialized_count": len(train_features),
                "validation_source_count": len(validation_rows),
                "training_ledger": train_ledger,
                "validation_ledger": validation_ledger,
                "promoted_device_provenance": (
                    _binding(promoted_device_provenance)
                    if promoted_device_provenance is not None
                    else None
                ),
                "validation_outputs": {
                    name: _binding(paths[name])
                    for name in (
                        "device_validation_features",
                        "device_validation_targets",
                        "combined_validation_features",
                        "combined_validation_targets",
                        "evaluation_feature_provenance",
                    )
                },
            },
        }
    )
    result["cache_sha256"] = sha256_json(result["outputs"])
    result_path = output / "cache.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--device-training-quality", type=Path, required=True)
    parser.add_argument("--device-validation-quality", type=Path, required=True)
    parser.add_argument("--base-validation-features", type=Path, required=True)
    parser.add_argument("--base-validation-targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicas-per-device", type=int, default=128)
    parser.add_argument("--seed", type=int, default=36013)
    parser.add_argument("--promoted-device-provenance", type=Path)
    parser.add_argument("--promoted-only", action="store_true")
    args = parser.parse_args(argv)
    report = extend(
        args.base_cache,
        args.device_training_quality,
        args.device_validation_quality,
        args.base_validation_features,
        args.base_validation_targets,
        args.output,
        replicas_per_device=args.replicas_per_device,
        seed=args.seed,
        promoted_device_provenance=args.promoted_device_provenance,
        promoted_only=args.promoted_only,
    )
    print(
        json.dumps(
            {
                "output": str((args.output / "cache.json").resolve()),
                "sample_count": report["sample_count"],
                "device_replay_extension": {
                    key: report["device_replay_extension"][key]
                    for key in (
                        "training_source_count",
                        "promoted_consumed_source_count",
                        "training_materialized_count",
                        "validation_source_count",
                    )
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
