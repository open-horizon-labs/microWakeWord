#!/usr/bin/env python3
"""Train a detector-conditioned, fixed-window Kizz Control verifier.

The input is the immutable output of ``build_kizz_candidate_verifier_dataset``.
Only detector-triggered windows are accepted.  Checkpoint and threshold selection
use validation candidates exclusively; test candidates are scored once after the
winning checkpoint and threshold have been frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
INPUT_FILES = ("features.npy", "labels.npy", "detector_scores.npy")
SPLITS = ("train", "validation", "test")
DEFAULT_CONDITIONAL_RECALL_FLOOR = 0.98
DEFAULT_NEGATIVE_SAMPLING_SHARE = 0.75
DEFAULT_NEGATIVE_GROUP_SAMPLING = "uniform_group"
DEFAULT_PHYSICAL_HARD_NEGATIVE_SHARE = 0.0
NEGATIVE_GROUP_SAMPLING_MODES = ("proportional_example", "uniform_group")
MIN_NEGATIVE_SAMPLING_SHARE = 0.50
MAX_NEGATIVE_SAMPLING_SHARE = 0.75
INPUT_SHAPE = (260, 40)
DEFAULT_MODEL_VARIANT = "compact"
MODEL_VARIANT_CHANNELS: dict[str, tuple[int, int, int, int, int]] = {
    # This is the deployed/original topology. Keep these values and the default
    # stable so existing invocations continue to produce the same graph.
    "compact": (24, 32, 48, 64, 96),
    # Same deployed footprint as compact, with bounded activations so
    # post-training integer calibration cannot be dominated by rare ReLU
    # outliers. ReLU6 is a fused TFLite/ESP-NN-friendly activation.
    "compact_relu6": (24, 32, 48, 64, 96),
    # Wider pointwise capacity without adding operator types or changing the
    # fixed candidate-window tensor contract.
    "wide": (32, 48, 64, 80, 112),
    # Candidate-triggered verifier capacity: still the same ESP-NN-friendly
    # Conv2D/depthwise/pointwise operator set, but sized for harder phonetic
    # collisions.  At about 7.6M MACs it remains a verifier-only option rather
    # than a continuously evaluated detector.
    "xwide": (48, 64, 96, 128, 160),
    # Preserve brief consonant transitions for Kizz/Kiss/Kids discrimination.
    # This keeps the same ESP-NN-friendly operator family but reduces temporal
    # downsampling from 32x to 16x. It is candidate-triggered, never continuous.
    "temporal": (32, 48, 64, 80, 96),
}
MODEL_VARIANT_STRIDES: dict[str, tuple[tuple[int, int], ...]] = {
    "compact": ((2, 2), (2, 2), (2, 2), (2, 2), (2, 2)),
    "compact_relu6": ((2, 2), (2, 2), (2, 2), (2, 2), (2, 2)),
    "wide": ((2, 2), (2, 2), (2, 2), (2, 2), (2, 2)),
    "xwide": ((2, 2), (2, 2), (2, 2), (2, 2), (2, 2)),
    # TensorFlow/ESP depthwise kernels require symmetric strides. Preserve
    # time in the regular stem convolution, then use symmetric depthwise ops.
    "temporal": ((1, 2), (2, 2), (2, 2), (2, 2), (2, 2)),
}
MODEL_JSON_PROVENANCE_KEY = "kizz_candidate_verifier_provenance"
LABEL_SMOOTHING = 0.0
L2_WEIGHT_DECAY = 0.0
FEATURE_AUGMENTATION_PROFILES: dict[str, dict[str, float | int]] = {
    "none": {
        "level_offset": 0.0,
        "noise_stddev": 0.0,
        "max_time_shift_frames": 0,
        "max_time_mask_frames": 0,
        "max_frequency_mask_bins": 0,
    },
    "moderate": {
        "level_offset": 0.75,
        "noise_stddev": 0.15,
        "max_time_shift_frames": 4,
        "max_time_mask_frames": 6,
        "max_frequency_mask_bins": 2,
    },
    "strong": {
        "level_offset": 1.5,
        "noise_stddev": 0.35,
        "max_time_shift_frames": 8,
        "max_time_mask_frames": 12,
        "max_frequency_mask_bins": 4,
    },
}
DEVICE_ROBUSTNESS_PROFILES: dict[str, dict[str, float | int | bool]] = {
    "none": {
        "fake_quantize_activations": False,
        "activation_noise_lsb_stddev": 0.0,
        "activation_quantization_bits": 8,
        "activation_min": 0.0,
        "activation_max": 6.0,
    },
    # Board traces showed one-LSB differences at the first convolution that
    # compounded through the graph. Train through the same 8-bit bottleneck
    # while perturbing every bounded activation by the observed error scale.
    "int8_lsb1": {
        "fake_quantize_activations": True,
        "activation_noise_lsb_stddev": 1.0,
        "activation_quantization_bits": 8,
        "activation_min": 0.0,
        "activation_max": 6.0,
    },
    "int8_lsb2": {
        "fake_quantize_activations": True,
        "activation_noise_lsb_stddev": 2.0,
        "activation_quantization_bits": 8,
        "activation_min": 0.0,
        "activation_max": 6.0,
    },
}
IDENTITY_FIELDS = (
    "parent_source_id",
    "source_parent_source_id",
    "speaker_id",
    "session_id",
    "ancestry_id",
)
HASH_FIELDS = (
    "audio_sha256",
    "sha256",
    "source_audio_sha256",
    "parent_source_audio_sha256",
    "feature_sha256",
)


@dataclass(frozen=True)
class VerifiedDataset:
    root: Path
    corpus_path: Path
    corpus_sha256: str
    corpus: dict[str, Any]
    rows: tuple[dict[str, Any], ...]
    features: np.ndarray
    labels: np.ndarray
    detector_scores: np.ndarray
    array_bindings: dict[str, dict[str, Any]]
    transitive_bindings: tuple[dict[str, Any], ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def model_topology_sha256(serialized: str) -> str:
    """Hash only the Keras inference topology, excluding bound metadata."""
    try:
        payload = json.loads(serialized)
    except json.JSONDecodeError as error:
        raise ValueError(f"model architecture is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("model architecture must be a JSON object")
    payload.pop("compile_config", None)
    payload.pop(MODEL_JSON_PROVENANCE_KEY, None)
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def bind_model_json_provenance(
    serialized: str,
    *,
    model_variant: str,
    cost: Mapping[str, Any],
) -> str:
    """Embed deterministic variant/spec/cost metadata without changing graph hash."""
    variant = _model_variant(model_variant)
    try:
        payload = json.loads(serialized)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"backend model architecture is not valid JSON: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("backend model architecture must be a JSON object")
    topology_sha256 = model_topology_sha256(serialized)
    payload[MODEL_JSON_PROVENANCE_KEY] = {
        "schema_version": 1,
        "name": "fixed_window_dscnn",
        "variant": variant,
        "channel_plan": list(MODEL_VARIANT_CHANNELS[variant]),
        "input_shape": [*INPUT_SHAPE, 1],
        "output": "one_logit",
        "dscnn_spec": json.loads(json.dumps(dscnn_spec(variant))),
        "cost": json.loads(json.dumps(cost)),
        "topology_sha256": topology_sha256,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _require_sha256(value: object, label: str) -> str:
    digest = str(value or "")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return digest


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as target:
            target.write(data)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        with open(temporary, "wb") as target:
            np.save(target, values, allow_pickle=False)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(fd)
    try:
        shutil.copyfile(source, temporary)
        with open(temporary, "rb") as copied:
            os.fsync(copied.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _resolved_binding_path(raw: object, relative_to: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} binding requires a path")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def _verify_transitive_bindings(
    value: Any,
    *,
    relative_to: Path,
    label: str = "corpus",
) -> tuple[dict[str, Any], ...]:
    """Verify every nested object that declares both ``path`` and ``sha256``."""
    verified: list[dict[str, Any]] = []

    def visit(item: Any, item_label: str) -> None:
        if isinstance(item, Mapping):
            if "path" in item or "sha256" in item:
                if "path" not in item or "sha256" not in item:
                    raise ValueError(f"{item_label} has an incomplete file binding")
                path = _resolved_binding_path(item["path"], relative_to, item_label)
                expected = _require_sha256(item["sha256"], f"{item_label} hash")
                if not path.is_file():
                    raise FileNotFoundError(path)
                actual = sha256_file(path)
                if actual != expected:
                    raise ValueError(
                        f"{item_label} hash drift: expected {expected}, got {actual}"
                    )
                verified.append(
                    {
                        "label": item_label,
                        "path": str(path),
                        "sha256": actual,
                        "bytes": path.stat().st_size,
                    }
                )
            for key, child in item.items():
                visit(child, f"{item_label}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{item_label}[{index}]")

    visit(value, label)
    unique = {
        (entry["label"], entry["path"], entry["sha256"]): entry for entry in verified
    }
    return tuple(unique[key] for key in sorted(unique))


def _identity_values(row: Mapping[str, Any]) -> set[str]:
    values = {
        str(row[key]) for key in IDENTITY_FIELDS if row.get(key) not in (None, "")
    }
    for key in ("ancestry_ids", "parent_source_ids"):
        raw = row.get(key, [])
        if isinstance(raw, list):
            values.update(str(value) for value in raw if value not in (None, ""))
    return values


def _hash_values(row: Mapping[str, Any]) -> set[str]:
    values = {
        str(row[key])
        for key in HASH_FIELDS
        if isinstance(row.get(key), str) and row[key]
    }
    raw = row.get("ancestry_sha256", [])
    if isinstance(raw, list):
        values.update(str(value) for value in raw if value)
    return values


def _verify_split_disjointness(rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        if not _identity_values(row):
            raise ValueError(f"{row.get('candidate_id')}: source identity is required")
        if not _hash_values(row):
            raise ValueError(f"{row.get('candidate_id')}: source hash is required")
    for left_index, left in enumerate(SPLITS):
        left_rows = [row for row in rows if row["split"] == left]
        left_ids = set().union(*map(_identity_values, left_rows))
        left_hashes = set().union(*map(_hash_values, left_rows))
        for right in SPLITS[left_index + 1 :]:
            right_rows = [row for row in rows if row["split"] == right]
            right_ids = set().union(*map(_identity_values, right_rows))
            right_hashes = set().union(*map(_hash_values, right_rows))
            if left_ids & right_ids:
                raise ValueError(
                    f"{left}/{right} identity overlap: {sorted(left_ids & right_ids)[:3]}"
                )
            if left_hashes & right_hashes:
                raise ValueError(
                    f"{left}/{right} hash overlap: {sorted(left_hashes & right_hashes)[:3]}"
                )


def _positive_provider(row: Mapping[str, Any]) -> str:
    provider = row.get("provider") or row.get("source_group")
    if not isinstance(provider, str) or not provider:
        raise ValueError(
            f"positive candidate {row.get('candidate_id')} lacks provider/source_group"
        )
    return provider


def _negative_group(row: Mapping[str, Any]) -> str:
    group = row.get("source_group") or row.get("semantic_label")
    if not isinstance(group, str) or not group:
        raise ValueError(
            f"negative candidate {row.get('candidate_id')} lacks source_group"
        )
    return group


def _is_physical_hard_negative(row: Mapping[str, Any]) -> bool:
    return int(row.get("label", -1)) == 0 and str(row.get("capture_id", "")).startswith(
        "hardneg-"
    )


def _verify_hard_negative_policy(
    corpus: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> None:
    policy = corpus.get("hard_negative_selection")
    if not isinstance(policy, Mapping):
        raise ValueError("corpus requires hard_negative_selection")
    if policy.get("scope") != "train_only":
        raise ValueError("hard-negative filtering must be train_only")
    if policy.get("ranking") != "detector_score_descending_then_candidate_id":
        raise ValueError("unsupported hard-negative ranking policy")
    top_k = policy.get("top_k")
    group_by = policy.get("group_by")
    if not isinstance(top_k, int) or top_k < 1 or group_by not in {"source", "session"}:
        raise ValueError("invalid hard-negative top-K/group policy")
    train_negatives = [
        row for row in rows if row["split"] == "train" and int(row["label"]) == 0
    ]
    if policy.get("selected_training_count") != len(train_negatives):
        raise ValueError("selected training-negative count differs from corpus rows")
    raw_training = policy.get("raw_training_count")
    if not isinstance(raw_training, int) or raw_training < len(train_negatives):
        raise ValueError("raw training-negative count is invalid")
    grouped: Counter[str] = Counter()
    for row in train_negatives:
        if group_by == "source":
            key = row.get("parent_source_id")
        else:
            key = row.get("session_id")
        if key in (None, ""):
            raise ValueError(f"train negative lacks {group_by} hard-negative identity")
        grouped[str(key)] += 1
    if grouped and max(grouped.values()) > top_k:
        raise ValueError("training hard-negative group exceeds declared top-K")

    counts = corpus.get("counts", {}).get("by_split", {})
    heldout_total = 0
    for split in ("validation", "test"):
        split_counts = counts.get(split)
        if not isinstance(split_counts, Mapping):
            raise ValueError(f"missing held-out count evidence for {split}")
        selected = sum(
            1 for row in rows if row["split"] == split and int(row["label"]) == 0
        )
        raw = split_counts.get("raw_negative_candidates")
        declared_selected = split_counts.get("selected_negative_candidates")
        if raw != selected or declared_selected != selected:
            raise ValueError(f"{split} detector candidates were filtered")
        heldout_total += selected
    if policy.get("heldout_candidates_unfiltered") != heldout_total:
        raise ValueError("held-out unfiltered-candidate count differs")


def load_verified_dataset(
    root: Path, *, expected_corpus_sha256: str
) -> VerifiedDataset:
    root = root.expanduser().resolve()
    corpus_path = root / "corpus.json"
    expected = _require_sha256(expected_corpus_sha256, "corpus expected hash")
    actual = sha256_file(corpus_path)
    if actual != expected:
        raise ValueError(f"corpus hash drift: expected {expected}, got {actual}")
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    if not isinstance(corpus, dict) or corpus.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported candidate-verifier corpus schema")
    if corpus.get("recipe") != "kizz_control_candidate_conditioned_verifier_v1":
        raise ValueError("corpus is not the candidate-verifier recipe")
    if corpus.get("candidate_condition") != "frozen_detector_trigger_only":
        raise ValueError("corpus is not conditioned on frozen detector triggers")

    transitive = _verify_transitive_bindings(
        {"bindings": corpus.get("bindings"), "detector": corpus.get("detector")},
        relative_to=corpus_path.parent,
    )
    array_hashes = corpus.get("array_sha256")
    if not isinstance(array_hashes, Mapping):
        raise ValueError("corpus requires array_sha256 bindings")
    array_bindings: dict[str, dict[str, Any]] = {}
    for name, declared in sorted(array_hashes.items()):
        expected_array = _require_sha256(declared, f"{name} hash")
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != expected_array:
            raise ValueError(
                f"{name} hash drift: expected {expected_array}, got {observed}"
            )
        array_bindings[name] = {
            "path": str(path),
            "sha256": observed,
            "bytes": path.stat().st_size,
        }
    missing = set(INPUT_FILES) - set(array_bindings)
    if missing:
        raise ValueError(f"corpus lacks required array bindings: {sorted(missing)}")

    features = np.load(root / "features.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(root / "labels.npy", mmap_mode="r", allow_pickle=False)
    detector_scores = np.load(
        root / "detector_scores.npy", mmap_mode="r", allow_pickle=False
    )
    rows = corpus.get("examples")
    if (
        not isinstance(rows, list)
        or not rows
        or not all(isinstance(r, dict) for r in rows)
    ):
        raise ValueError("corpus requires nonempty examples")
    if features.ndim != 3 or tuple(features.shape[1:]) != INPUT_SHAPE:
        raise ValueError(
            f"features must have shape [N,{INPUT_SHAPE[0]},{INPUT_SHAPE[1]}]"
        )
    if labels.shape != (len(rows),) or detector_scores.shape != (len(rows),):
        raise ValueError("feature, label, detector-score, and corpus counts differ")
    if len(features) != len(rows):
        raise ValueError("feature and corpus counts differ")
    if not np.issubdtype(features.dtype, np.number) or not np.all(
        np.isfinite(features)
    ):
        raise ValueError("features must be finite numeric values")
    if not np.all(np.isfinite(detector_scores)):
        raise ValueError("detector scores must be finite")
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("labels must be binary")

    seen_candidates: set[str] = set()
    split_class_counts: Counter[tuple[str, int]] = Counter()
    for index, row in enumerate(rows):
        candidate_id = row.get("candidate_id")
        if (
            not isinstance(candidate_id, str)
            or not candidate_id
            or candidate_id in seen_candidates
        ):
            raise ValueError("candidate_id values must be unique nonempty strings")
        seen_candidates.add(candidate_id)
        if row.get("detector_conditioned") is not True:
            raise ValueError(f"{candidate_id}: detector_conditioned must be true")
        split = row.get("split")
        label = row.get("label")
        if split not in SPLITS or label not in (0, 1, False, True):
            raise ValueError(f"{candidate_id}: invalid split or label")
        if row.get("feature_index") != index:
            raise ValueError(f"{candidate_id}: feature_index/order drift")
        if int(labels[index]) != int(label):
            raise ValueError(f"{candidate_id}: labels.npy differs from corpus")
        if not math.isclose(
            float(detector_scores[index]),
            float(row.get("detector_score")),
            rel_tol=1e-6,
            abs_tol=1e-7,
        ):
            raise ValueError(f"{candidate_id}: detector_scores.npy differs from corpus")
        expected_feature = _require_sha256(
            row.get("candidate_feature_sha256"), f"{candidate_id} feature hash"
        )
        observed_feature = hashlib.sha256(
            np.ascontiguousarray(features[index]).tobytes()
        ).hexdigest()
        if observed_feature != expected_feature:
            raise ValueError(f"{candidate_id}: candidate feature hash drift")
        split_class_counts[(str(split), int(label))] += 1

    for split in SPLITS:
        for label in (0, 1):
            if split_class_counts[(split, label)] < 1:
                raise ValueError(f"{split} requires positive and negative candidates")
    _verify_split_disjointness(rows)
    _verify_hard_negative_policy(corpus, rows)
    for row in rows:
        if row["split"] == "train" and int(row["label"]) == 1:
            _positive_provider(row)
        if row["split"] == "train" and int(row["label"]) == 0:
            _negative_group(row)

    declared_counts = corpus.get("counts", {})
    if declared_counts.get("selected_candidates") != len(rows):
        raise ValueError("selected candidate total differs from corpus rows")
    if declared_counts.get("selected_positives") != int(np.sum(labels == 1)):
        raise ValueError("selected positive total differs from labels.npy")
    if declared_counts.get("selected_negatives") != int(np.sum(labels == 0)):
        raise ValueError("selected negative total differs from labels.npy")
    return VerifiedDataset(
        root=root,
        corpus_path=corpus_path,
        corpus_sha256=actual,
        corpus=corpus,
        rows=tuple(dict(row) for row in rows),
        features=features,
        labels=labels,
        detector_scores=detector_scores,
        array_bindings=array_bindings,
        transitive_bindings=transitive,
    )


def _batch_class_counts(
    batch_size: int, negative_sampling_share: float
) -> tuple[int, int]:
    if batch_size < 4:
        raise ValueError("batch_size must be at least four")
    if (
        not math.isfinite(negative_sampling_share)
        or not MIN_NEGATIVE_SAMPLING_SHARE
        <= negative_sampling_share
        <= MAX_NEGATIVE_SAMPLING_SHARE
    ):
        raise ValueError(
            "negative_sampling_share must be finite and within "
            f"[{MIN_NEGATIVE_SAMPLING_SHARE},{MAX_NEGATIVE_SAMPLING_SHARE}]"
        )
    negative_samples = batch_size * negative_sampling_share
    if not float(negative_samples).is_integer():
        raise ValueError("batch_size * negative_sampling_share must be an integer")
    negative_count = int(negative_samples)
    positive_count = batch_size - negative_count
    if positive_count < 1 or negative_count < 1:
        raise ValueError("each batch must contain both classes")
    return positive_count, negative_count


class BalancedCandidateBatcher:
    """Deterministic bounded class sampling, balanced within source families."""

    def __init__(
        self,
        dataset: VerifiedDataset,
        *,
        batch_size: int,
        seed: int,
        augmentation_profile: str = "none",
        negative_sampling_share: float = DEFAULT_NEGATIVE_SAMPLING_SHARE,
        negative_group_sampling: str = DEFAULT_NEGATIVE_GROUP_SAMPLING,
        physical_hard_negative_share: float = DEFAULT_PHYSICAL_HARD_NEGATIVE_SHARE,
    ):
        positive_count, negative_count = _batch_class_counts(
            batch_size, negative_sampling_share
        )
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.negative_sampling_share = float(negative_sampling_share)
        if negative_group_sampling not in NEGATIVE_GROUP_SAMPLING_MODES:
            raise ValueError(
                "unknown negative_group_sampling: "
                f"{negative_group_sampling!r}; expected one of "
                f"{list(NEGATIVE_GROUP_SAMPLING_MODES)}"
            )
        self.negative_group_sampling = negative_group_sampling
        if (
            not math.isfinite(physical_hard_negative_share)
            or not 0.0 <= physical_hard_negative_share <= 0.5
        ):
            raise ValueError("physical_hard_negative_share must be within [0,0.5]")
        if (
            physical_hard_negative_share
            and negative_group_sampling != "proportional_example"
        ):
            raise ValueError(
                "physical_hard_negative_share requires proportional_example sampling"
            )
        self.physical_hard_negative_share = float(physical_hard_negative_share)
        self.negative_samples_per_batch = negative_count
        self.positive_samples_per_batch = positive_count
        if augmentation_profile not in FEATURE_AUGMENTATION_PROFILES:
            raise ValueError(
                f"unknown feature augmentation profile: {augmentation_profile}"
            )
        self.augmentation_profile = augmentation_profile
        self.augmentation = FEATURE_AUGMENTATION_PROFILES[augmentation_profile]
        self.positive_groups: dict[str, np.ndarray] = {}
        self.negative_groups: dict[str, np.ndarray] = {}
        grouped_positive: dict[str, list[int]] = defaultdict(list)
        grouped_negative: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(dataset.rows):
            if row["split"] != "train":
                continue
            if int(row["label"]) == 1:
                grouped_positive[_positive_provider(row)].append(index)
            else:
                grouped_negative[_negative_group(row)].append(index)
        if not grouped_positive or not grouped_negative:
            raise ValueError("training requires positive and negative groups")
        self.positive_groups = {
            key: np.asarray(values, dtype=np.int64)
            for key, values in sorted(grouped_positive.items())
        }
        self.negative_groups = {
            key: np.asarray(values, dtype=np.int64)
            for key, values in sorted(grouped_negative.items())
        }
        self.negative_indexes = np.concatenate(
            [self.negative_groups[key] for key in sorted(self.negative_groups)]
        )
        self.physical_hard_negative_indexes = np.asarray(
            [
                int(index)
                for index in self.negative_indexes
                if _is_physical_hard_negative(dataset.rows[int(index)])
            ],
            dtype=np.int64,
        )
        self.background_negative_indexes = np.asarray(
            [
                int(index)
                for index in self.negative_indexes
                if not _is_physical_hard_negative(dataset.rows[int(index)])
            ],
            dtype=np.int64,
        )
        if self.physical_hard_negative_share and (
            not len(self.physical_hard_negative_indexes)
            or not len(self.background_negative_indexes)
        ):
            raise ValueError(
                "physical hard-negative emphasis requires both physical and background negatives"
            )
        self.realized_positive: Counter[str] = Counter()
        self.realized_negative: Counter[str] = Counter()

    @staticmethod
    def _balanced_indexes(
        groups: Mapping[str, np.ndarray], count: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, list[str]]:
        names = sorted(groups)
        order = [names[int(index)] for index in rng.permutation(len(names))]
        selected: list[int] = []
        selected_groups: list[str] = []
        for slot in range(count):
            group = order[slot % len(order)]
            pool = groups[group]
            selected.append(int(pool[int(rng.integers(0, len(pool)))]))
            selected_groups.append(group)
        return np.asarray(selected, dtype=np.int64), selected_groups

    def _augment(self, features: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.augmentation_profile == "none":
            return features
        output = np.asarray(features, dtype=np.float32).copy()
        level_offset = float(self.augmentation["level_offset"])
        noise_stddev = float(self.augmentation["noise_stddev"])
        max_shift = int(self.augmentation["max_time_shift_frames"])
        max_time_mask = int(self.augmentation["max_time_mask_frames"])
        max_frequency_mask = int(self.augmentation["max_frequency_mask_bins"])
        for sample in output:
            if level_offset:
                sample += np.float32(rng.uniform(-level_offset, level_offset))
            if noise_stddev:
                sample += rng.normal(0.0, noise_stddev, sample.shape).astype(np.float32)
            if max_shift:
                shift = int(rng.integers(-max_shift, max_shift + 1))
                if shift > 0:
                    sample[shift:] = sample[:-shift]
                    sample[:shift] = 0.0
                elif shift < 0:
                    amount = -shift
                    sample[:-amount] = sample[amount:]
                    sample[-amount:] = 0.0
            if max_time_mask and rng.random() < 0.5:
                width = int(rng.integers(1, max_time_mask + 1))
                start = int(rng.integers(0, sample.shape[0] - width + 1))
                sample[start : start + width] = 0.0
            if max_frequency_mask and rng.random() < 0.5:
                width = int(rng.integers(1, max_frequency_mask + 1))
                start = int(rng.integers(0, sample.shape[1] - width + 1))
                sample[:, start : start + width] = 0.0
        np.maximum(output, 0.0, out=output)
        return output

    def batch(self, step: int) -> tuple[np.ndarray, np.ndarray]:
        if step < 0:
            raise ValueError("step must be nonnegative")
        rng = np.random.default_rng(self.seed + step)
        positives, positive_groups = self._balanced_indexes(
            self.positive_groups, self.positive_samples_per_batch, rng
        )
        if self.negative_group_sampling == "uniform_group":
            negatives, negative_groups = self._balanced_indexes(
                self.negative_groups, self.negative_samples_per_batch, rng
            )
        else:
            physical_count = int(
                round(
                    self.negative_samples_per_batch * self.physical_hard_negative_share
                )
            )
            if self.physical_hard_negative_share and physical_count == 0:
                physical_count = 1
            background_count = self.negative_samples_per_batch - physical_count
            background_pool = (
                self.background_negative_indexes
                if physical_count
                else self.negative_indexes
            )
            background = background_pool[
                rng.integers(0, len(background_pool), size=background_count)
            ]
            physical = (
                self.physical_hard_negative_indexes[
                    rng.integers(
                        0, len(self.physical_hard_negative_indexes), size=physical_count
                    )
                ]
                if physical_count
                else np.empty(0, dtype=np.int64)
            )
            negatives = np.concatenate([background, physical])
            negative_groups = [
                _negative_group(self.dataset.rows[int(index)]) for index in negatives
            ]
        indexes = np.concatenate([positives, negatives])
        labels = np.concatenate(
            [
                np.ones(self.positive_samples_per_batch, dtype=np.float32),
                np.zeros(self.negative_samples_per_batch, dtype=np.float32),
            ]
        )
        order = rng.permutation(self.batch_size)
        self.realized_positive.update(positive_groups)
        self.realized_negative.update(negative_groups)
        features = np.asarray(self.dataset.features[indexes], dtype=np.float32)
        features = self._augment(features, rng)[..., None]
        return features[order], labels[order]

    def report(self) -> dict[str, Any]:
        positive_total = sum(self.realized_positive.values())
        negative_total = sum(self.realized_negative.values())

        def group_rows(counts: Counter[str], total: int) -> dict[str, Any]:
            return {
                key: {"samples": count, "share_within_class": count / total}
                for key, count in sorted(counts.items())
            }

        return {
            "mode": (
                "bounded_negative_emphasis_uniform_group_round_robin"
                if self.negative_group_sampling == "uniform_group"
                else "bounded_negative_emphasis_proportional_example"
            ),
            "negative_group_sampling": self.negative_group_sampling,
            "candidate_condition": "frozen_detector_trigger_only",
            "sampling_split": "train",
            "configured_negative_sampling_share": self.negative_sampling_share,
            "configured_physical_hard_negative_share_within_negatives": self.physical_hard_negative_share,
            "negative_sampling_share_bounds": {
                "minimum": MIN_NEGATIVE_SAMPLING_SHARE,
                "maximum": MAX_NEGATIVE_SAMPLING_SHARE,
            },
            "samples_per_batch": {
                "positive": self.positive_samples_per_batch,
                "negative": self.negative_samples_per_batch,
            },
            "feature_augmentation": {
                "profile": self.augmentation_profile,
                **self.augmentation,
                "training_only": True,
                "deterministic_seeded_by_step": True,
            },
            "class_samples": {
                "positive": positive_total,
                "negative": negative_total,
            },
            "realized_negative_sampling_share": negative_total
            / (positive_total + negative_total),
            "positive_provider_samples": group_rows(
                self.realized_positive, positive_total
            ),
            "negative_group_samples": group_rows(
                self.realized_negative, negative_total
            ),
        }


def _model_variant(value: str) -> str:
    variant = str(value)
    if variant not in MODEL_VARIANT_CHANNELS:
        raise ValueError(
            f"unknown model variant {variant!r}; expected one of "
            f"{sorted(MODEL_VARIANT_CHANNELS)}"
        )
    return variant


def dscnn_spec(
    model_variant: str = DEFAULT_MODEL_VARIANT,
) -> tuple[dict[str, Any], ...]:
    """Return a deterministic ESP32-friendly verifier topology."""
    stem, pointwise_1, pointwise_2, pointwise_3, pointwise_4 = MODEL_VARIANT_CHANNELS[
        _model_variant(model_variant)
    ]
    stem_stride, ds1_stride, ds2_stride, ds3_stride, ds4_stride = MODEL_VARIANT_STRIDES[
        model_variant
    ]
    activation = "relu6" if model_variant == "compact_relu6" else "relu"
    return (
        {
            "name": "stem",
            "op": "Conv2D",
            "filters": stem,
            "kernel": (5, 5),
            "strides": stem_stride,
            "activation": activation,
        },
        {
            "name": "ds1_depthwise",
            "op": "DepthwiseConv2D",
            "kernel": (3, 3),
            "strides": ds1_stride,
            "activation": activation,
        },
        {
            "name": "ds1_pointwise",
            "op": "Conv2D",
            "filters": pointwise_1,
            "kernel": (1, 1),
            "strides": (1, 1),
            "activation": activation,
        },
        {
            "name": "ds2_depthwise",
            "op": "DepthwiseConv2D",
            "kernel": (3, 3),
            "strides": ds2_stride,
            "activation": activation,
        },
        {
            "name": "ds2_pointwise",
            "op": "Conv2D",
            "filters": pointwise_2,
            "kernel": (1, 1),
            "strides": (1, 1),
            "activation": activation,
        },
        {
            "name": "ds3_depthwise",
            "op": "DepthwiseConv2D",
            "kernel": (3, 3),
            "strides": ds3_stride,
            "activation": activation,
        },
        {
            "name": "ds3_pointwise",
            "op": "Conv2D",
            "filters": pointwise_3,
            "kernel": (1, 1),
            "strides": (1, 1),
            "activation": activation,
        },
        {
            "name": "ds4_depthwise",
            "op": "DepthwiseConv2D",
            "kernel": (3, 3),
            "strides": ds4_stride,
            "activation": activation,
        },
        {
            "name": "ds4_pointwise",
            "op": "Conv2D",
            "filters": pointwise_4,
            "kernel": (1, 1),
            "strides": (1, 1),
            "activation": activation,
        },
        # Candidate windows are causally aligned: the detector event is always
        # the final frame. Preserve coarse temporal position instead of turning
        # the verifier into a bag-of-phonemes classifier via global averaging.
        {"name": "temporal_flatten", "op": "Flatten"},
        {"name": "verifier_logit", "op": "Dense", "units": 1},
    )


def estimate_dscnn_cost(
    input_shape: tuple[int, int] = INPUT_SHAPE,
    model_variant: str = DEFAULT_MODEL_VARIANT,
) -> dict[str, Any]:
    height, width = input_shape
    channels = 1
    params = 0
    macs = 0
    layers: list[dict[str, Any]] = []
    for layer in dscnn_spec(model_variant):
        row = dict(layer)
        operation = str(layer["op"])
        if operation in {"Conv2D", "DepthwiseConv2D"}:
            stride_h, stride_w = layer["strides"]
            out_h = math.ceil(height / stride_h)
            out_w = math.ceil(width / stride_w)
            kernel_h, kernel_w = layer["kernel"]
            if operation == "Conv2D":
                out_channels = int(layer["filters"])
                layer_params = (
                    kernel_h * kernel_w * channels * out_channels + out_channels
                )
                layer_macs = (
                    out_h * out_w * kernel_h * kernel_w * channels * out_channels
                )
            else:
                out_channels = channels
                layer_params = kernel_h * kernel_w * channels + channels
                layer_macs = out_h * out_w * kernel_h * kernel_w * channels
            height, width, channels = out_h, out_w, out_channels
        elif operation == "Flatten":
            channels = height * width * channels
            height = width = 1
            layer_params = 0
            layer_macs = 0
        elif operation == "Dense":
            out_channels = int(layer["units"])
            layer_params = channels * out_channels + out_channels
            layer_macs = channels * out_channels
            channels = out_channels
        else:
            layer_params = 0
            layer_macs = 0
        params += layer_params
        macs += layer_macs
        row.update({"parameters": layer_params, "macs": layer_macs})
        layers.append(row)
    return {
        "input_shape": list(input_shape),
        "parameter_estimate": params,
        "mac_estimate": macs,
        "mac_scope": "Conv2D, DepthwiseConv2D, pointwise Conv2D, and Dense",
        "layers": layers,
    }


@dataclass(frozen=True)
class TensorFlowVerifierModel:
    """Shared-weight training and clean deployment graphs."""

    training: Any
    deployment: Any


class TensorFlowVerifierBackend:
    """Lazy TensorFlow backend; importing this module remains lightweight."""

    def build_model(
        self,
        *,
        learning_rate: float,
        seed: int,
        model_variant: str = DEFAULT_MODEL_VARIANT,
        device_robustness_profile: str = "none",
    ) -> Any:
        import tensorflow as tf

        model_variant = _model_variant(model_variant)
        if device_robustness_profile not in DEVICE_ROBUSTNESS_PROFILES:
            raise ValueError(
                f"unknown device robustness profile: {device_robustness_profile}"
            )
        if device_robustness_profile != "none" and model_variant != "compact_relu6":
            raise ValueError(
                "device robustness training requires the bounded compact_relu6 variant"
            )
        robustness = DEVICE_ROBUSTNESS_PROFILES[device_robustness_profile]
        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except (AttributeError, RuntimeError):
            pass
        inputs = tf.keras.Input(shape=INPUT_SHAPE + (1,), name="log_mel_window")
        value = inputs
        regularizer = tf.keras.regularizers.L2(L2_WEIGHT_DECAY)
        for layer in dscnn_spec(model_variant):
            operation = layer["op"]
            if operation == "Conv2D":
                value = tf.keras.layers.Conv2D(
                    int(layer["filters"]),
                    tuple(layer["kernel"]),
                    strides=tuple(layer["strides"]),
                    padding="same",
                    activation=str(layer.get("activation", "relu")),
                    use_bias=True,
                    kernel_regularizer=regularizer,
                    name=str(layer["name"]),
                )(value)
            elif operation == "DepthwiseConv2D":
                value = tf.keras.layers.DepthwiseConv2D(
                    tuple(layer["kernel"]),
                    strides=tuple(layer["strides"]),
                    padding="same",
                    activation=str(layer.get("activation", "relu")),
                    use_bias=True,
                    depthwise_regularizer=regularizer,
                    name=str(layer["name"]),
                )(value)
            elif operation == "Flatten":
                value = tf.keras.layers.Flatten(name=str(layer["name"]))(value)
            elif operation == "Dense":
                value = tf.keras.layers.Dense(
                    int(layer["units"]),
                    activation=str(layer.get("activation", "linear")),
                    kernel_regularizer=regularizer,
                    name=str(layer["name"]),
                )(value)
            else:  # pragma: no cover - guarded by the fixed local specification.
                raise AssertionError(f"unsupported operation {operation}")
        model_name = (
            "kizz_candidate_verifier_dscnn"
            if model_variant == DEFAULT_MODEL_VARIANT
            else f"kizz_candidate_verifier_dscnn_{model_variant}"
        )
        model = tf.keras.Model(inputs, value, name=model_name)
        if model.output_shape != (None, 1):
            raise ValueError("verifier model must output exactly one logit")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(
                from_logits=True, label_smoothing=LABEL_SMOOTHING
            ),
        )
        if device_robustness_profile == "none":
            return model
        training_inputs = tf.keras.Input(
            shape=INPUT_SHAPE + (1,), name="robust_log_mel_window"
        )
        training_value = training_inputs
        spec_by_name = {
            str(layer["name"]): layer for layer in dscnn_spec(model_variant)
        }
        for layer_index, keras_layer in enumerate(model.layers[1:]):
            training_value = keras_layer(training_value)
            layer_spec = spec_by_name[keras_layer.name]
            if layer_spec["op"] not in {"Conv2D", "DepthwiseConv2D"}:
                continue
            minimum = float(robustness["activation_min"])
            maximum = float(robustness["activation_max"])
            bits = int(robustness["activation_quantization_bits"])
            training_value = tf.keras.layers.Lambda(
                lambda tensor, lo=minimum, hi=maximum, nbits=bits: (
                    tf.quantization.fake_quant_with_min_max_vars(
                        tensor, lo, hi, num_bits=nbits
                    )
                ),
                name=f"{keras_layer.name}_training_fake_quant",
            )(training_value)
            lsb = (maximum - minimum) / ((1 << bits) - 1)
            stddev = float(robustness["activation_noise_lsb_stddev"]) * lsb
            training_value = tf.keras.layers.GaussianNoise(
                stddev=stddev,
                seed=seed + layer_index + 1,
                name=f"{keras_layer.name}_training_lsb_noise",
            )(training_value)
        training_model = tf.keras.Model(
            training_inputs,
            training_value,
            name=f"{model_name}_{device_robustness_profile}_training",
        )
        training_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(
                from_logits=True, label_smoothing=LABEL_SMOOTHING
            ),
        )
        return TensorFlowVerifierModel(training=training_model, deployment=model)

    @staticmethod
    def _training_model(model: Any) -> Any:
        return model.training if isinstance(model, TensorFlowVerifierModel) else model

    @staticmethod
    def _deployment_model(model: Any) -> Any:
        return model.deployment if isinstance(model, TensorFlowVerifierModel) else model

    @staticmethod
    def train_batch(model: Any, features: np.ndarray, labels: np.ndarray) -> float:
        return float(
            TensorFlowVerifierBackend._training_model(model).train_on_batch(
                features, labels
            )
        )

    @staticmethod
    def score(
        model: Any, features: np.ndarray, *, batch_size: int, purpose: str
    ) -> np.ndarray:
        del purpose
        values = TensorFlowVerifierBackend._deployment_model(model).predict(
            features, batch_size=batch_size, verbose=0
        )
        return np.asarray(values, dtype=np.float64).reshape(-1)

    @staticmethod
    def save_weights(model: Any, path: Path) -> None:
        TensorFlowVerifierBackend._deployment_model(model).save_weights(path)

    @staticmethod
    def load_weights(model: Any, path: Path) -> None:
        TensorFlowVerifierBackend._deployment_model(model).load_weights(path)

    @staticmethod
    def count_params(model: Any) -> int:
        return int(TensorFlowVerifierBackend._deployment_model(model).count_params())

    @staticmethod
    def model_json(model: Any) -> str:
        return str(TensorFlowVerifierBackend._deployment_model(model).to_json())


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("verifier logits must be a finite one-dimensional vector")
    return np.where(
        values >= 0,
        1.0 / (1.0 + np.exp(-values)),
        np.exp(values) / (1.0 + np.exp(values)),
    )


def _freeze_probability_threshold(
    selected_threshold: float, validation_logit_safety_margin: float
) -> float:
    clipped = min(
        max(float(selected_threshold), np.finfo(np.float64).tiny),
        1.0 - np.finfo(np.float64).eps,
    )
    selected_logit = math.log(clipped / (1.0 - clipped))
    transformed = float(
        _sigmoid(np.asarray([selected_logit - validation_logit_safety_margin]))[0]
    )
    # A probability/logit round trip may move the threshold upward by one ULP.
    # Safety margins may only preserve or relax the validation boundary.
    return min(float(selected_threshold), transformed)


def _positive_unit(row: Mapping[str, Any]) -> str:
    return str(row.get("parent_source_id") or row.get("candidate_id"))


def _metrics_at_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    threshold: float,
    recall_floor: float,
) -> dict[str, Any]:
    accepted = probabilities >= threshold
    positive = labels == 1
    negative = labels == 0
    positive_units = {
        _positive_unit(row) for row, label in zip(rows, labels) if label == 1
    }
    accepted_units = {
        _positive_unit(row)
        for row, label, is_accepted in zip(rows, labels, accepted)
        if label == 1 and is_accepted
    }
    true_candidates = int(np.sum(accepted & positive))
    false_candidates = int(np.sum(accepted & negative))
    accepted_count = true_candidates + false_candidates
    conditional_recall = len(accepted_units) / len(positive_units)
    precision = true_candidates / accepted_count if accepted_count else 1.0
    return {
        "threshold": float(threshold),
        "conditional_recall": conditional_recall,
        "conditional_recall_numerator": len(accepted_units),
        "conditional_recall_denominator": len(positive_units),
        "precision": precision,
        "true_candidates": true_candidates,
        "false_candidates": false_candidates,
        "accepted_candidates": accepted_count,
        "positive_candidates": int(np.sum(positive)),
        "negative_candidates": int(np.sum(negative)),
        "meets_conditional_recall_floor": conditional_recall + 1e-12 >= recall_floor,
    }


def operating_point_selection_key(
    point: Mapping[str, Any], recall_floor: float
) -> tuple[Any, ...]:
    meets = float(point["conditional_recall"]) + 1e-12 >= recall_floor
    if meets:
        return (
            1,
            -int(point["false_candidates"]),
            float(point["precision"]),
            float(point["conditional_recall"]),
            float(point["threshold"]),
        )
    return (
        0,
        float(point["conditional_recall"]),
        -int(point["false_candidates"]),
        float(point["precision"]),
        float(point["threshold"]),
    )


def evaluate_operating_point(
    logits: np.ndarray,
    labels: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    *,
    recall_floor: float,
    threshold: float | None = None,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int8)
    if labels.shape != (len(rows),) or not np.all(np.isin(labels, [0, 1])):
        raise ValueError("evaluation labels/rows differ or are not binary")
    probabilities = _sigmoid(np.asarray(logits))
    if probabilities.shape != labels.shape:
        raise ValueError("evaluation score and label counts differ")
    if not np.any(labels == 1) or not np.any(labels == 0):
        raise ValueError("evaluation requires positive and negative candidates")
    if threshold is not None:
        if not math.isfinite(float(threshold)) or not 0 <= float(threshold) <= 1:
            raise ValueError("frozen threshold must be finite and within [0,1]")
        return {
            "selection_performed": False,
            "selected": _metrics_at_threshold(
                probabilities, labels, rows, float(threshold), recall_floor
            ),
        }
    thresholds = set(float(value) for value in probabilities)
    thresholds.add(float(np.nextafter(np.max(probabilities), math.inf)))
    thresholds.add(0.0)
    ledger = [
        _metrics_at_threshold(probabilities, labels, rows, value, recall_floor)
        for value in sorted(thresholds, reverse=True)
    ]
    selected = max(
        ledger, key=lambda point: operating_point_selection_key(point, recall_floor)
    )
    return {"selection_performed": True, "selected": selected, "ledger": ledger}


def checkpoint_selection_key(
    item: Mapping[str, Any], recall_floor: float
) -> tuple[Any, ...]:
    return (
        *operating_point_selection_key(item["operating_point"], recall_floor),
        -float(item["validation_loss"]),
        -int(item["step"]),
    )


def _detector_score_edges(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("detector stratification scores must be finite")
    quantiles = np.quantile(values, [0.25, 0.5, 0.75]).tolist()
    return [-math.inf, *sorted(set(float(value) for value in quantiles)), math.inf]


def detector_score_stratification(
    logits: np.ndarray,
    labels: np.ndarray,
    detector_scores: np.ndarray,
    *,
    threshold: float,
    edges: Sequence[float],
) -> dict[str, Any]:
    probabilities = _sigmoid(np.asarray(logits))
    labels = np.asarray(labels, dtype=np.int8)
    detector_scores = np.asarray(detector_scores, dtype=np.float64)
    if probabilities.shape != labels.shape or labels.shape != detector_scores.shape:
        raise ValueError("stratification arrays differ")
    bands: list[dict[str, Any]] = []
    for lower, upper in zip(edges, edges[1:]):
        mask = (detector_scores >= lower) & (detector_scores < upper)
        accepted = mask & (probabilities >= threshold)
        true_count = int(np.sum(accepted & (labels == 1)))
        false_count = int(np.sum(accepted & (labels == 0)))
        accepted_count = true_count + false_count
        bands.append(
            {
                "lower_inclusive": None if math.isinf(lower) else float(lower),
                "upper_exclusive": None if math.isinf(upper) else float(upper),
                "candidates": int(np.sum(mask)),
                "positive_candidates": int(np.sum(mask & (labels == 1))),
                "negative_candidates": int(np.sum(mask & (labels == 0))),
                "accepted_candidates": accepted_count,
                "false_candidates": false_count,
                "precision": true_count / accepted_count if accepted_count else 1.0,
            }
        )
    return {"edge_source": "validation_detector_score_quartiles", "bands": bands}


def _indexes(dataset: VerifiedDataset, split: str) -> np.ndarray:
    return np.asarray(
        [index for index, row in enumerate(dataset.rows) if row["split"] == split],
        dtype=np.int64,
    )


def _save_weights_atomic(backend: Any, model: Any, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".weights.h5", dir=path.parent
    )
    os.close(fd)
    temporary = Path(raw)
    try:
        temporary.unlink()
        backend.save_weights(model, temporary)
        if not temporary.is_file():
            raise ValueError("backend did not write checkpoint")
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _verify_output_bindings(bindings: Mapping[str, Mapping[str, Any]]) -> None:
    for name, binding in bindings.items():
        path = Path(str(binding.get("path", ""))).resolve()
        expected = _require_sha256(binding.get("sha256"), f"output {name} hash")
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"output binding drift: {name}")


def _build_backend_model(
    backend: Any,
    *,
    learning_rate: float,
    seed: int,
    model_variant: str,
    device_robustness_profile: str,
) -> Any:
    """Build while preserving compact-only compatibility for legacy backends."""
    parameters = tuple(inspect.signature(backend.build_model).parameters.values())
    accepts_variant = any(
        parameter.name == "model_variant"
        or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    accepts_robustness = any(
        parameter.name == "device_robustness_profile"
        or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    if device_robustness_profile != "none" and not accepts_robustness:
        raise ValueError("training backend does not support device robustness profiles")
    if accepts_variant:
        kwargs: dict[str, Any] = {
            "learning_rate": learning_rate,
            "seed": seed,
            "model_variant": model_variant,
        }
        if accepts_robustness:
            kwargs["device_robustness_profile"] = device_robustness_profile
        return backend.build_model(**kwargs)
    if model_variant != DEFAULT_MODEL_VARIANT:
        raise ValueError(
            "training backend does not support the requested wide model variant"
        )
    return backend.build_model(learning_rate=learning_rate, seed=seed)


def train_candidate_verifier(
    dataset_root: Path,
    output: Path,
    *,
    expected_corpus_sha256: str,
    steps: int = 3000,
    batch_size: int = 64,
    learning_rate: float = 0.0005,
    eval_every: int = 100,
    conditional_recall_floor: float = DEFAULT_CONDITIONAL_RECALL_FLOOR,
    validation_logit_safety_margin: float = 0.0,
    augmentation_profile: str = "none",
    negative_sampling_share: float = DEFAULT_NEGATIVE_SAMPLING_SHARE,
    negative_group_sampling: str = DEFAULT_NEGATIVE_GROUP_SAMPLING,
    physical_hard_negative_share: float = DEFAULT_PHYSICAL_HARD_NEGATIVE_SHARE,
    model_variant: str = DEFAULT_MODEL_VARIANT,
    device_robustness_profile: str = "none",
    seed: int = 248,
    backend: Any | None = None,
    evaluator: Callable[..., dict[str, Any]] = evaluate_operating_point,
) -> dict[str, Any]:
    model_variant = _model_variant(model_variant)
    if device_robustness_profile not in DEVICE_ROBUSTNESS_PROFILES:
        raise ValueError(
            f"unknown device robustness profile: {device_robustness_profile}"
        )
    if device_robustness_profile != "none" and model_variant != "compact_relu6":
        raise ValueError(
            "device robustness training requires model_variant=compact_relu6"
        )
    if steps < 1 or eval_every < 1:
        raise ValueError("steps and eval_every must be positive")
    _batch_class_counts(batch_size, negative_sampling_share)
    if learning_rate <= 0 or not math.isfinite(learning_rate):
        raise ValueError("learning_rate must be finite and positive")
    if not 0 < conditional_recall_floor <= 1:
        raise ValueError("conditional_recall_floor must be within (0,1]")
    if validation_logit_safety_margin < 0 or not math.isfinite(
        validation_logit_safety_margin
    ):
        raise ValueError(
            "validation_logit_safety_margin must be finite and nonnegative"
        )
    if augmentation_profile not in FEATURE_AUGMENTATION_PROFILES:
        raise ValueError(
            f"unknown feature augmentation profile: {augmentation_profile}"
        )
    dataset = load_verified_dataset(
        dataset_root, expected_corpus_sha256=expected_corpus_sha256
    )
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse verifier output directory: {output}")
    output.mkdir(parents=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir()

    backend = backend or TensorFlowVerifierBackend()
    model = _build_backend_model(
        backend,
        learning_rate=learning_rate,
        seed=seed,
        model_variant=model_variant,
        device_robustness_profile=device_robustness_profile,
    )
    cost = estimate_dscnn_cost(model_variant=model_variant)
    parameter_count = int(backend.count_params(model))
    if parameter_count != int(cost["parameter_estimate"]):
        raise ValueError(
            f"model parameter count {parameter_count} differs from DS-CNN estimate "
            f"{cost['parameter_estimate']}"
        )
    model_path = output / "model.json"
    raw_model_json = backend.model_json(model)
    if not isinstance(raw_model_json, str) or not raw_model_json:
        raise ValueError("backend returned no model architecture")
    model_json = bind_model_json_provenance(
        raw_model_json,
        model_variant=model_variant,
        cost=cost,
    )
    topology_sha256 = model_topology_sha256(model_json)
    _atomic_bytes(model_path, model_json.encode("utf-8"))

    batcher = BalancedCandidateBatcher(
        dataset,
        batch_size=batch_size,
        seed=seed,
        augmentation_profile=augmentation_profile,
        negative_sampling_share=negative_sampling_share,
        negative_group_sampling=negative_group_sampling,
        physical_hard_negative_share=physical_hard_negative_share,
    )
    validation_indexes = _indexes(dataset, "validation")
    test_indexes = _indexes(dataset, "test")
    validation_features = np.asarray(
        dataset.features[validation_indexes], dtype=np.float32
    )[..., None]
    test_features = np.asarray(dataset.features[test_indexes], dtype=np.float32)[
        ..., None
    ]
    validation_labels = np.asarray(dataset.labels[validation_indexes], dtype=np.int8)
    test_labels = np.asarray(dataset.labels[test_indexes], dtype=np.int8)
    validation_rows = [dataset.rows[int(index)] for index in validation_indexes]
    test_rows = [dataset.rows[int(index)] for index in test_indexes]

    losses: list[float] = []
    ledger: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_validation_logits: np.ndarray | None = None
    for step in range(steps):
        train_features, train_labels = batcher.batch(step)
        loss = float(backend.train_batch(model, train_features, train_labels))
        if not math.isfinite(loss):
            raise ValueError(f"non-finite training loss at step {step + 1}")
        losses.append(loss)
        if step == 0 or (step + 1) % eval_every == 0 or step + 1 == steps:
            checkpoint_path = checkpoints / f"step-{step + 1:06d}.weights.h5"
            checkpoint = _save_weights_atomic(backend, model, checkpoint_path)
            validation_logits = np.asarray(
                backend.score(
                    model,
                    validation_features,
                    batch_size=batch_size,
                    purpose=f"validation-checkpoint-{step + 1}",
                ),
                dtype=np.float64,
            ).reshape(-1)
            validation = evaluator(
                validation_logits,
                validation_labels,
                validation_rows,
                recall_floor=conditional_recall_floor,
                threshold=None,
            )
            if validation.get("selection_performed") is not True:
                raise ValueError(
                    "validation evaluator did not perform threshold selection"
                )
            item = {
                "step": step + 1,
                "validation_loss": loss,
                "operating_point": validation["selected"],
                "threshold_ledger": validation["ledger"],
                "checkpoint": checkpoint,
            }
            ledger.append(item)
            if best is None or checkpoint_selection_key(
                item, conditional_recall_floor
            ) > checkpoint_selection_key(best, conditional_recall_floor):
                best = item
                best_validation_logits = validation_logits.copy()
    if best is None or best_validation_logits is None:
        raise AssertionError("training produced no validation checkpoint")

    last_path = output / "last.weights.h5"
    last_binding = _save_weights_atomic(backend, model, last_path)
    winner_checkpoint = Path(best["checkpoint"]["path"])
    best_path = output / "best.weights.h5"
    _atomic_copy(winner_checkpoint, best_path)
    best_binding = {
        "path": str(best_path),
        "sha256": sha256_file(best_path),
        "bytes": best_path.stat().st_size,
    }
    if best_binding["sha256"] != best["checkpoint"]["sha256"]:
        raise ValueError("best.weights.h5 is not byte-identical to winning checkpoint")

    selected_threshold = float(best["operating_point"]["threshold"])
    frozen_threshold = _freeze_probability_threshold(
        selected_threshold, validation_logit_safety_margin
    )
    validation_evaluation = evaluator(
        best_validation_logits,
        validation_labels,
        validation_rows,
        recall_floor=conditional_recall_floor,
        threshold=frozen_threshold,
    )
    backend.load_weights(model, best_path)
    test_logits = np.asarray(
        backend.score(
            model,
            test_features,
            batch_size=batch_size,
            purpose="test-once-after-frozen-selection",
        ),
        dtype=np.float64,
    ).reshape(-1)
    test_evaluation = evaluator(
        test_logits,
        test_labels,
        test_rows,
        recall_floor=conditional_recall_floor,
        threshold=frozen_threshold,
    )
    if test_evaluation.get("selection_performed") is not False:
        raise ValueError("test evaluator attempted threshold selection")

    validation_logits_path = output / "validation-winner-logits.npy"
    test_logits_path = output / "test-frozen-logits.npy"
    _atomic_npy(validation_logits_path, best_validation_logits)
    _atomic_npy(test_logits_path, test_logits)
    validation_detector_scores = np.asarray(
        dataset.detector_scores[validation_indexes], dtype=np.float64
    )
    test_detector_scores = np.asarray(
        dataset.detector_scores[test_indexes], dtype=np.float64
    )
    score_edges = _detector_score_edges(validation_detector_scores)

    output_bindings: dict[str, dict[str, Any]] = {
        "model_architecture": {
            "path": str(model_path),
            "sha256": sha256_file(model_path),
            "bytes": model_path.stat().st_size,
        },
        "best_weights": best_binding,
        "last_weights": last_binding,
        "validation_winner_logits": {
            "path": str(validation_logits_path),
            "sha256": sha256_file(validation_logits_path),
            "bytes": validation_logits_path.stat().st_size,
        },
        "test_frozen_logits": {
            "path": str(test_logits_path),
            "sha256": sha256_file(test_logits_path),
            "bytes": test_logits_path.stat().st_size,
        },
    }
    for item in ledger:
        output_bindings[f"checkpoint_step_{item['step']}"] = item["checkpoint"]
    _verify_output_bindings(output_bindings)

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "recipe": "kizz_control_candidate_conditioned_dscnn_verifier_v1",
        "candidate_conditioned": True,
        "deployment_qualification": False,
        "deployment_qualification_reason": (
            "synthetic candidate training and held-out scoring are not StackChan "
            "hardware qualification"
        ),
        "input_bindings": {
            "corpus": {
                "path": str(dataset.corpus_path),
                "sha256": dataset.corpus_sha256,
                "bytes": dataset.corpus_path.stat().st_size,
            },
            "arrays": dataset.array_bindings,
            "transitive": list(dataset.transitive_bindings),
        },
        "architecture": {
            "name": "fixed_window_dscnn",
            "variant": model_variant,
            "channel_plan": list(MODEL_VARIANT_CHANNELS[model_variant]),
            "dscnn_spec": json.loads(json.dumps(dscnn_spec(model_variant))),
            "int8_friendly_core_ops": [
                "Conv2D",
                "DepthwiseConv2D",
                "pointwise_Conv2D",
                "Flatten",
                "Dense",
            ],
            "parameter_count": parameter_count,
            "topology_sha256": topology_sha256,
            **cost,
            "input_shape": [*INPUT_SHAPE, 1],
            "output": "one_logit",
        },
        "training": {
            "steps": steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
            "loss": "binary_crossentropy_from_logits_label_smoothing",
            "label_smoothing": LABEL_SMOOTHING,
            "l2_weight_decay": L2_WEIGHT_DECAY,
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "sampling": batcher.report(),
            "hard_negative_policy": dataset.corpus["hard_negative_selection"],
            "device_robustness": {
                "profile": device_robustness_profile,
                **DEVICE_ROBUSTNESS_PROFILES[device_robustness_profile],
                "training_graph_only": True,
                "deployment_graph_contains_training_perturbation_ops": False,
            },
        },
        "selection_contract": {
            "selection_split": "validation",
            "objective": (
                "minimize_false_candidates_subject_to_conditional_recall_floor"
            ),
            "validation_selected_threshold_before_safety_margin": selected_threshold,
            "validation_logit_safety_margin": validation_logit_safety_margin,
            "conditional_recall_floor": conditional_recall_floor,
            "test_used_for_selection": False,
            "test_score_passes": 1,
        },
        "checkpoint_ledger": ledger,
        "winner": {
            "step": int(best["step"]),
            "frozen_threshold": frozen_threshold,
            "checkpoint": best["checkpoint"],
            "best_weights": best_binding,
        },
        "validation": validation_evaluation["selected"],
        "test": test_evaluation["selected"],
        "detector_score_stratification": {
            "validation": detector_score_stratification(
                best_validation_logits,
                validation_labels,
                validation_detector_scores,
                threshold=frozen_threshold,
                edges=score_edges,
            ),
            "test": detector_score_stratification(
                test_logits,
                test_labels,
                test_detector_scores,
                threshold=frozen_threshold,
                edges=score_edges,
            ),
        },
        "output_bindings": output_bindings,
    }
    report_path = output / "training-report.json"
    _atomic_bytes(report_path, _canonical_bytes(report))
    manifest_bindings = {
        **output_bindings,
        "training_report": {
            "path": str(report_path),
            "sha256": sha256_file(report_path),
            "bytes": report_path.stat().st_size,
        },
    }
    _verify_output_bindings(manifest_bindings)
    artifact_manifest = {
        "schema_version": 1,
        "artifact_kind": "kizz_candidate_verifier_training_outputs",
        "deployment_qualification": False,
        "model": {
            "name": "fixed_window_dscnn",
            "variant": model_variant,
            "channel_plan": list(MODEL_VARIANT_CHANNELS[model_variant]),
            "input_shape": [*INPUT_SHAPE, 1],
            "output": "one_logit",
            "dscnn_spec": json.loads(json.dumps(dscnn_spec(model_variant))),
            "parameter_count": parameter_count,
            "mac_count": int(cost["mac_estimate"]),
            "topology_sha256": topology_sha256,
            "model_json": output_bindings["model_architecture"],
        },
        "bindings": manifest_bindings,
    }
    _atomic_bytes(
        output / "artifact-manifest.json", _canonical_bytes(artifact_manifest)
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--corpus-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument(
        "--conditional-recall-floor",
        type=float,
        default=DEFAULT_CONDITIONAL_RECALL_FLOOR,
    )
    parser.add_argument("--seed", type=int, default=248)
    parser.add_argument(
        "--model-variant",
        choices=sorted(MODEL_VARIANT_CHANNELS),
        default=DEFAULT_MODEL_VARIANT,
        help="ESP32-friendly candidate-verifier capacity/topology variant",
    )
    parser.add_argument("--validation-logit-safety-margin", type=float, default=0.0)
    parser.add_argument(
        "--augmentation-profile",
        choices=sorted(FEATURE_AUGMENTATION_PROFILES),
        default="none",
    )
    parser.add_argument(
        "--device-robustness-profile",
        choices=sorted(DEVICE_ROBUSTNESS_PROFILES),
        default="none",
        help=(
            "training-only activation fake quantization/noise profile; non-none "
            "profiles require --model-variant compact_relu6"
        ),
    )
    parser.add_argument(
        "--negative-sampling-share",
        type=float,
        default=DEFAULT_NEGATIVE_SAMPLING_SHARE,
        help=(
            "Training-batch share reserved for detector-triggered negatives "
            f"({MIN_NEGATIVE_SAMPLING_SHARE} to {MAX_NEGATIVE_SAMPLING_SHARE})"
        ),
    )
    parser.add_argument(
        "--negative-group-sampling",
        choices=NEGATIVE_GROUP_SAMPLING_MODES,
        default=DEFAULT_NEGATIVE_GROUP_SAMPLING,
        help=(
            "Choose negative source groups uniformly (legacy behavior) or sample "
            "individual detector candidates in proportion to the observed corpus"
        ),
    )
    parser.add_argument(
        "--physical-hard-negative-share",
        type=float,
        default=DEFAULT_PHYSICAL_HARD_NEGATIVE_SHARE,
        help=(
            "Reserve this fraction of negative batch slots for train-only "
            "StackChan captures whose capture_id starts with hardneg-; requires "
            "proportional_example sampling (0 to 0.5)"
        ),
    )
    args = parser.parse_args(argv)
    report = train_candidate_verifier(
        args.dataset,
        args.output,
        expected_corpus_sha256=args.corpus_sha256,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_every=args.eval_every,
        conditional_recall_floor=args.conditional_recall_floor,
        validation_logit_safety_margin=args.validation_logit_safety_margin,
        augmentation_profile=args.augmentation_profile,
        negative_sampling_share=args.negative_sampling_share,
        negative_group_sampling=args.negative_group_sampling,
        physical_hard_negative_share=args.physical_hard_negative_share,
        model_variant=args.model_variant,
        device_robustness_profile=args.device_robustness_profile,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "winner_step": report["winner"]["step"],
                "validation": report["validation"],
                "test": report["test"],
                "deployment_qualification": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
