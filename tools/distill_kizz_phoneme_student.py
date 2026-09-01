#!/usr/bin/env python3
"""Distill the qualified generic phoneme teacher into one causal ESP32 model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from microwakeword.kizz_feature_archive import (
    decode_frontend_features,
    open_feature_archive,
)
from microwakeword.ctc_occupancy import ctc_state_occupation_log_probs
from microwakeword.kizz_phoneme_teacher import choose_validation_threshold
from microwakeword.kizz_viterbi_decoder import exhaustive_suffix_score
from microwakeword.ordered_state import OrderedStateTopology
from microwakeword.ordered_state_model import (
    SqueezeFrequency,
    model as build_student,
    parse,
)
from microwakeword.phoneme_student import (
    compact_phone_contract,
    resample_log_posteriors,
    student_output_times_seconds,
)
from microwakeword.wake_phrase import KIZZ_CONTROL
from tools.cache_kizz_phoneme_teacher_posteriors import load_cache
from tools.cache_kizz_teacher_representations import load_representation_cache
from tools.build_kizz_phoneme_distillation_corpus import (
    _quantized_context,
    frontend,
    load_audio,
    load_device_training_rows,
    load_device_validation_rows,
    place_phrase_context,
)
from tools.distill_kizz_student import student_flags

INPUT_SHAPE = (260, 40)
OUTPUT_FRAMES = 66
WINDOW_LENGTHS_SECONDS = (0.56, 0.68, 0.80, 0.96, 1.16, 1.40, 1.60)
OUTPUT_STEP_SECONDS = 0.030
WINDOW_LENGTHS_FRAMES = (19, 23, 27, 32, 39, 47, 54)
APPROVED_PROVIDERS = ("assemblyai", "deepgram", "elevenlabs", "kokoro")
POSITIVE_VARIANTS = ("clean", "overlay", "device")
DEVICE_POSITIVE_SOURCE_GROUP = "device_channel_positive"
COLLISION_TEXT_TO_PATH = {
    "Kids Control": "kidskontrol",
    "Kiss Control": "kiskontrol",
    "This control": "thiskontrol",
    # The compact vocabulary has no /w/ token. Keep Quiz Control under the
    # generic all-collision margin until its OTHER-backed path is separately
    # qualified; silently pretending it is another path would corrupt labels.
    "Quiz Control": None,
    "Kizz controller": "kizkontroller",
    "Kizz controlled": "kizkontrold",
    "Kizz patrol": "kizpatrol",
    "His control": "hiskontrol",
    "This controller is missing": "thiskontrol",
    "The kids control the television": "kidskontrol",
    "Kids can troll": "kidskantrol",
    "The kitchen controls are broken": "kitchenkontrol",
}


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def student_decoder_contract(
    contract: dict, algorithm: str = "max_add_ctc_viterbi"
) -> dict:
    """Return the exact portable decoder semantics used for student selection."""
    if algorithm not in ("max_add_ctc_viterbi", "forward_sum_ctc"):
        raise ValueError("unsupported student decoder algorithm")
    return {
        "type": "kizz_ctc_phone_decoder",
        "implementation": (
            "microwakeword.kizz_viterbi_decoder.exhaustive_suffix_score"
            if algorithm == "max_add_ctc_viterbi"
            else "microwakeword.ctc_forward.exhaustive_sliding_forward_score"
        ),
        "algorithm": algorithm,
        "score": (
            "maximum_ctc_path_log_probability_divided_by_path_token_count"
            if algorithm == "max_add_ctc_viterbi"
            else "summed_ctc_alignment_log_probability_divided_by_path_token_count"
        ),
        # These are deployment constants, not values re-derived with Python's
        # banker rounding. In particular, the 1.60 s window is 54 frames in
        # the evaluator and firmware, not round(53.333...) == 53.
        "window_lengths_frames": list(WINDOW_LENGTHS_FRAMES),
        "beta": 0.0,
        "selection": "maximum_canonical_fit_then_collision_margin",
        "compact_phone_contract_sha256": _canonical_hash(contract),
    }


def student_decoder_contract_hash(
    contract: dict, algorithm: str = "max_add_ctc_viterbi"
) -> str:
    return _canonical_hash(student_decoder_contract(contract, algorithm))


def student_flags_for_architecture(architecture: str, output_count: int):
    if architecture == "control_mixconv":
        return student_flags(output_count)
    if architecture == "control_mixconv_small":
        return student_flags(output_count, architecture)
    if architecture == "temporal_residual":
        return SimpleNamespace(
            pointwise_filters="80,96,96,96,96",
            residual_connection="1,1,1,1,1",
            repeat_in_block="1,1,1,1,1",
            mixconv_kernel_sizes="[3], [3], [5], [5], [9]",
            first_conv_filters=48,
            first_conv_kernel_size=5,
            stride=3,
            num_states=output_count,
        )
    if architecture == "dilated_temporal_memory":
        return SimpleNamespace(
            pointwise_filters="80,96,96,96,96",
            residual_connection="1,1,1,1,1",
            repeat_in_block="1,1,1,1,1",
            mixconv_kernel_sizes="[3], [3], [3], [3], [3]",
            temporal_dilations="1,2,4,8,16",
            causal_memory=True,
            warmup_output_drop=20,
            first_conv_filters=48,
            first_conv_kernel_size=5,
            stride=3,
            num_states=output_count,
        )
    if architecture == "dilated_temporal_memory_wide":
        return SimpleNamespace(
            pointwise_filters="112,128,128,128,128",
            residual_connection="1,1,1,1,1",
            repeat_in_block="1,1,1,1,1",
            mixconv_kernel_sizes="[3], [3], [3], [3], [3]",
            temporal_dilations="1,2,4,8,16",
            causal_memory=True,
            warmup_output_drop=20,
            first_conv_filters=64,
            first_conv_kernel_size=5,
            stride=3,
            num_states=output_count,
        )
    raise ValueError(f"unsupported student architecture: {architecture}")


def student_architecture_contract(
    contract: dict, architecture: str = "control_mixconv"
) -> dict:
    """Return the exact architecture expected by the portable converter."""
    flags = student_flags_for_architecture(architecture, len(contract["tokens"]))
    result = {
        "input_shape": list(INPUT_SHAPE),
        "output_frames": OUTPUT_FRAMES,
        "output_count": len(contract["tokens"]),
        "pointwise_filters": list(parse(flags.pointwise_filters)),
        "residual_connection": list(parse(flags.residual_connection)),
        "repeat_in_block": list(parse(flags.repeat_in_block)),
        "mixconv_kernel_sizes": list(parse(flags.mixconv_kernel_sizes)),
        "first_conv_filters": flags.first_conv_filters,
        "first_conv_kernel_size": flags.first_conv_kernel_size,
        "stride": flags.stride,
    }
    if architecture != "control_mixconv":
        result["architecture_id"] = architecture
    if getattr(flags, "causal_memory", False):
        result.update(
            {
                "causal_memory": True,
                "temporal_dilations": list(parse(flags.temporal_dilations)),
                "warmup_output_drop": int(flags.warmup_output_drop),
            }
        )
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    """Hash a file or a directory's complete, ordered contents."""
    path = path.resolve()
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise ValueError(f"provenance path does not exist: {path}")
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"provenance directory is empty: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(sha256_file(item).encode())
    return digest.hexdigest()


def provenance_ref(path: Path) -> dict:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": sha256_path(resolved)}


def expected_schedule_counts(
    steps: int, batch_size: int
) -> tuple[Counter[str], Counter[str], Counter[tuple[str, str]]]:
    if steps < 1 or batch_size < 8 or batch_size % 8:
        raise ValueError(
            "steps must be positive and batch size must be a positive multiple of eight"
        )
    half = batch_size // 2
    positive_variant_providers = Counter(
        (
            POSITIVE_VARIANTS[global_slot % len(POSITIVE_VARIANTS)],
            APPROVED_PROVIDERS[global_slot % len(APPROVED_PROVIDERS)],
        )
        for global_slot in range(steps * half)
    )
    providers = Counter()
    for (_, provider), count in positive_variant_providers.items():
        providers[provider] += count
    groups = (
        "public_speech",
        "kizz_control_phonetic_collision",
        "device_collision",
        "no_speech",
    )
    negatives = Counter(
        groups[(step * half + offset) % len(groups)]
        for step in range(steps)
        for offset in range(half)
    )
    return providers, negatives, positive_variant_providers


def nested_variant_provider_counts(
    counts: Counter[tuple[str, str]],
) -> dict[str, dict[str, int]]:
    return {
        variant: {
            provider: int(counts[(variant, provider)])
            for provider in APPROVED_PROVIDERS
        }
        for variant in POSITIVE_VARIANTS
    }


def positive_indices_by_provider(
    rows: list[dict], variant: str
) -> dict[str, np.ndarray]:
    """Partition training positives without allowing device rows into clean."""
    if variant not in ("clean", "device"):
        raise ValueError(f"unsupported corpus positive variant: {variant}")
    train = [index for index, row in enumerate(rows) if row["split"] == "train"]
    if variant == "device":
        selected = lambda row: row.get("source_group") == DEVICE_POSITIVE_SOURCE_GROUP
    else:
        selected = lambda row: row.get("source_group") != DEVICE_POSITIVE_SOURCE_GROUP
    result = {
        provider: np.asarray(
            [
                i
                for i in train
                if rows[i]["label"] == 1
                and rows[i].get("provider") == provider
                and selected(rows[i])
            ],
            dtype=np.int64,
        )
        for provider in APPROVED_PROVIDERS
    }
    device_providers = {
        row.get("provider")
        for row in rows
        if row["split"] == "train"
        and row["label"] == 1
        and row.get("source_group") == DEVICE_POSITIVE_SOURCE_GROUP
    }
    unknown = device_providers.difference(APPROVED_PROVIDERS)
    if unknown:
        raise ValueError(
            f"device positives use unapproved providers: {sorted(unknown)}"
        )
    return result


def collision_path_supervision(
    rows: list[dict], source_manifest: Path, contract: dict
) -> tuple[np.ndarray, dict]:
    """Bind each synthetic collision row to its declared decoder path."""
    payload = json.loads(source_manifest.read_text())
    if isinstance(payload, list):
        source_rows = payload
    else:
        source_rows = payload.get("examples", payload.get("rows", []))
    by_id = {str(row.get("source_id")): row for row in source_rows}
    path_names = list(contract["collision_paths"])
    path_indexes = {name: index for index, name in enumerate(path_names)}
    result = np.full(len(rows), -1, dtype=np.int32)
    counts: Counter[str] = Counter()
    generic_texts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        if row.get("source_group") != "kizz_control_phonetic_collision":
            continue
        source = by_id.get(str(row.get("parent_source_id")))
        if source is None or source.get("semantic_label") != "phonetic_collision":
            raise ValueError(
                "phonetic collision row is not bound to its source manifest"
            )
        text = str(source.get("render_text", ""))
        if text not in COLLISION_TEXT_TO_PATH:
            raise ValueError(
                f"phonetic collision has no supervision contract: {text!r}"
            )
        path_name = COLLISION_TEXT_TO_PATH[text]
        if path_name is None:
            generic_texts[text] += 1
            continue
        if path_name not in path_indexes:
            raise ValueError(
                f"collision supervision path is not in decoder: {path_name}"
            )
        result[index] = path_indexes[path_name]
        counts[path_name] += 1
    return result, {
        "algorithm": "source_render_text_to_declared_collision_path_v1",
        "path_counts": dict(sorted(counts.items())),
        "generic_text_counts": dict(sorted(generic_texts.items())),
        "path_order": path_names,
    }


def checkpoint_binding(
    output: Path, best_step: int, best_key: tuple[float, ...]
) -> dict:
    """Return the immutable binding for both emitted checkpoints."""
    if best_step < 1:
        raise ValueError("best checkpoint step must be positive")
    best = (output / "best.weights.h5").resolve()
    last = (output / "last.weights.h5").resolve()
    if not best.is_file() or not last.is_file():
        raise ValueError("distillation must emit both best and last checkpoints")
    best_hash = sha256_file(best)
    last_hash = sha256_file(last)
    return {
        "selected_checkpoint": "best",
        "weights": str(best),
        "weights_sha256": best_hash,
        "best_step": best_step,
        "best_key": list(best_key),
        "best_weights": {"path": str(best), "sha256": best_hash},
        "last_weights": {"path": str(last), "sha256": last_hash},
    }


def checkpoint_selection_key(
    point: dict, zero_false_accept_recall: float, separation: float | None
) -> tuple[float, ...]:
    """Rank validation checkpoints using only deployment-equivalent evidence."""

    finite_floor = -np.finfo(np.float64).max
    false_accepts = point.get("false_accepts_at_recall_floor")
    return (
        float(bool(point.get("qualified"))),
        float(zero_false_accept_recall),
        -float(false_accepts) if false_accepts is not None else finite_floor,
        float(point.get("recall", 0.0)),
        float(separation) if separation is not None else finite_floor,
    )


def multichannel_checkpoint_selection_key(
    point: dict,
    clean_zero_false_accept_recall: float,
    device_zero_false_accept_recall: float,
    device_accepted: int,
    device_required: int,
    separation: float | None,
) -> tuple[float, ...]:
    """Rank only checkpoints that satisfy both clean and target-channel gates."""

    base = checkpoint_selection_key(point, clean_zero_false_accept_recall, separation)
    return (
        float(bool(point.get("qualified")) and device_accepted >= device_required),
        min(
            float(clean_zero_false_accept_recall),
            float(device_zero_false_accept_recall),
        ),
        float(device_accepted),
        *base,
    )


def device_validation_features(quality_report: Path) -> tuple[np.ndarray, list[dict]]:
    """Materialize exact frontend tensors for held-out qualified device clips."""

    rows = load_device_validation_rows(quality_report)
    values = []
    for row in rows:
        audio = load_audio(Path(row["path"]))
        phrase = row["phrase_span"]
        context, _ = place_phrase_context(
            audio, (float(phrase["start_s"]), float(phrase["end_s"]))
        )
        _, exact_float = _quantized_context(context)
        values.append(frontend(exact_float))
    return np.asarray(values, dtype=np.float32), rows


def device_parent_feature_pairs(
    corpus: dict, rows: list[dict]
) -> dict[int, np.ndarray]:
    """Bind every device-training row to its exact aligned clean source view."""

    binding = corpus.get("manifests", {}).get("device_quality") or {}
    quality_path = Path(str(binding.get("path", ""))).resolve()
    if not quality_path.is_file() or binding.get("sha256") != sha256_file(quality_path):
        raise ValueError("distillation corpus device-quality binding drifted")
    qualified = load_device_training_rows(quality_path)
    qualified_by_hash = {row["audio_sha256"]: row for row in qualified}
    quality = json.loads(quality_path.read_text())
    selection_path = Path(str(quality.get("inputs", {}).get("selection", ""))).resolve()
    if not selection_path.is_file() or quality["inputs"].get(
        "selection_sha256"
    ) != sha256_file(selection_path):
        raise ValueError("device parent selection binding drifted")
    selected = json.loads(selection_path.read_text()).get("selected_examples", [])
    sources = {str(row.get("audio_sha256", "")): row for row in selected}
    pairs: dict[int, np.ndarray] = {}
    for index, row in enumerate(rows):
        if (
            row.get("source_group") != DEVICE_POSITIVE_SOURCE_GROUP
            or row.get("split") != "train"
        ):
            continue
        source = sources.get(str(row.get("parent_source_audio_sha256", "")))
        qualified_row = qualified_by_hash.get(str(row.get("source_audio_sha256", "")))
        if source is None or qualified_row is None:
            raise ValueError("device training row lacks its qualified clean parent")
        audio = load_audio(Path(source["path"]))
        phrase = source["phrase_span"]
        context, _ = place_phrase_context(
            audio, (float(phrase["start_s"]), float(phrase["end_s"]))
        )
        _, exact_float = _quantized_context(context)
        pairs[index] = frontend(exact_float)
    if len(pairs) != len(qualified):
        raise ValueError("not every qualified device training row has a clean pair")
    return pairs


def load_temporal_representation_cache(prefix: Path) -> tuple[dict, np.ndarray]:
    metadata = json.loads(prefix.with_suffix(".json").read_text())
    matrix = np.load(prefix.with_suffix(".npy"), mmap_mode="r")
    if (
        metadata.get("schema_version") != 1
        or metadata.get("representation")
        != "qualified_teacher_last_hidden_frame_aligned_train_pca"
        or list(matrix.shape) != metadata.get("shape")
        or matrix.ndim != 3
        or str(matrix.dtype) != metadata.get("dtype")
    ):
        raise ValueError("teacher temporal representation cache contract differs")
    unsigned = {key: value for key, value in metadata.items() if key != "cache_sha256"}
    digest = hashlib.sha256()
    digest.update(np.asarray(matrix).tobytes(order="C"))
    digest.update(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode())
    if digest.hexdigest() != metadata.get("cache_sha256"):
        raise ValueError("teacher temporal representation cache is stale or corrupt")
    return metadata, matrix


def require_cache_binding(
    cache_meta: dict, teacher_qualification: Path, teacher_manifest_sha256: str
) -> None:
    """Reject a cache made for another teacher model or corpus manifest.

    Posterior bytes depend on the teacher weights and audio corpus, not on the
    detector threshold.  A qualification report may therefore be monotonically
    tightened after the cache is produced, provided it explicitly binds the
    original report and preserves the exact teacher model identity.
    """
    embedded = cache_meta.get("provenance", {}).get("teacher_qualification", {})
    active = json.loads(teacher_qualification.read_text())
    active_hash = sha256_file(teacher_qualification)
    source_hash = (
        active.get("operating_point_rebinding", {})
        .get("source_teacher_qualification", {})
        .get("sha256")
    )
    if embedded.get("sha256") not in (active_hash, source_hash):
        raise ValueError(
            "teacher posterior cache is not bound to the active qualification lineage"
        )
    cache_model = cache_meta.get("model", {})
    active_model = active.get("model", {})
    for field in ("revision", "weights_sha256"):
        if cache_model.get(field) != active_model.get(field):
            raise ValueError(f"teacher posterior cache uses different {field}")
    if cache_meta.get("manifest_sha256") != teacher_manifest_sha256:
        raise ValueError(
            "teacher posterior cache is for a different distillation corpus"
        )


def load_teacher_sequence_cache(
    cache_prefix: Path,
    *,
    corpus_json: Path,
    posterior_cache_prefix: Path,
    contract: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Load original-resolution teacher decisions with fail-closed bindings."""
    prefix = cache_prefix.with_suffix("")
    metadata = json.loads(prefix.with_suffix(".json").read_text())
    if metadata.get("schema_version") != 1:
        raise ValueError("unsupported teacher sequence-score cache schema")
    if (
        metadata.get("representation")
        != "qualified_teacher_original_resolution_clip_decisions"
    ):
        raise ValueError("teacher sequence cache has the wrong representation")
    if metadata.get("corpus", {}).get("sha256") != sha256_file(corpus_json):
        raise ValueError("teacher sequence cache is for a different corpus")
    posterior_prefix = posterior_cache_prefix.with_suffix("")
    posterior = metadata.get("posterior_cache", {})
    if posterior.get("json_sha256") != sha256_file(
        posterior_prefix.with_suffix(".json")
    ) or posterior.get("npz_sha256") != sha256_file(
        posterior_prefix.with_suffix(".npz")
    ):
        raise ValueError("teacher sequence cache is for different posteriors")
    if metadata.get("compact_phone_contract_sha256") != _canonical_hash(contract):
        raise ValueError("teacher sequence cache compact vocabulary differs")
    scorer = metadata.get("scorer", {})
    if (
        scorer.get("algorithm") != "forward_sum_ctc"
        or scorer.get("window_selection")
        != "filter_margin_then_max_canonical_then_margin"
        or float(scorer.get("beta", math.nan)) != 0.0
    ):
        raise ValueError("teacher sequence cache scorer contract differs")
    with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as loaded:
        required = (
            "raw_canonical_fit",
            "raw_collision_margin",
            "deployment_canonical_fit",
            "deployment_collision_margin",
            "eligible",
            "decision_score",
        )
        if any(key not in loaded for key in required):
            raise ValueError("teacher sequence cache is incomplete")
        arrays = {key: np.asarray(loaded[key]) for key in required}
    count = int(metadata.get("counts", {}).get("examples", -1))
    if any(values.shape != (count,) for values in arrays.values()):
        raise ValueError("teacher sequence cache array length differs")
    if np.any(~np.isfinite(arrays["decision_score"])):
        raise ValueError("teacher decision scores must be finite")
    return metadata, arrays


def load_student_window_cache(
    cache_prefix: Path,
    *,
    corpus_json: Path,
    features_path: Path,
    contract: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Load hard streaming endpoints mined by a frozen compact student."""
    prefix = cache_prefix.with_suffix("")
    metadata = json.loads(prefix.with_suffix(".json").read_text())
    if (
        metadata.get("schema_version") != 1
        or metadata.get("representation") != "student_streaming_window_hard_mining"
    ):
        raise ValueError("unsupported student streaming-window cache")
    corpus = metadata.get("corpus", {})
    if corpus.get("sha256") != sha256_file(corpus_json) or corpus.get(
        "features_sha256"
    ) != sha256_file(features_path):
        raise ValueError("student window cache is for a different corpus")
    if metadata.get("compact_phone_contract_sha256") != _canonical_hash(contract):
        raise ValueError("student window cache compact vocabulary differs")
    scorer = metadata.get("scorer", {})
    if (
        scorer.get("algorithm") != "forward_sum_ctc"
        or scorer.get("window_lengths_frames") != list(WINDOW_LENGTHS_FRAMES)
        or int(scorer.get("hop_frames", -1)) != 1
        or float(scorer.get("beta", math.nan)) != 0.0
    ):
        raise ValueError("student window cache scorer contract differs")
    required = (
        "raw_canonical_fit",
        "raw_collision_margin",
        "raw_end_frame",
        "deployment_canonical_fit",
        "deployment_collision_margin",
        "deployment_end_frame",
        "decision_score",
        "eligible",
    )
    with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as loaded:
        if any(key not in loaded for key in required):
            raise ValueError("student window cache is incomplete")
        arrays = {key: np.asarray(loaded[key]) for key in required}
    count = int(metadata.get("counts", {}).get("examples", -1))
    if any(values.shape != (count,) for values in arrays.values()):
        raise ValueError("student window cache array length differs")
    if np.any(~np.isfinite(arrays["decision_score"])):
        raise ValueError("student window decision scores must be finite")
    return metadata, arrays


def load_causal_decision_cache(
    cache_prefix: Path,
    *,
    representation: str,
    corpus_json: Path,
    contract: dict,
    expected_examples: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Load endpoint-level decision targets with exact corpus bindings."""

    prefix = cache_prefix.with_suffix("")
    metadata = json.loads(prefix.with_suffix(".json").read_text())
    if (
        metadata.get("schema_version") != 1
        or metadata.get("representation") != representation
        or metadata.get("corpus", {}).get("sha256") != sha256_file(corpus_json)
        or metadata.get("compact_phone_contract_sha256") != _canonical_hash(contract)
    ):
        raise ValueError("causal decision cache contract differs")
    required = (
        "raw_canonical_fit",
        "raw_collision_margin",
        "decision_score",
        "eligible",
    )
    with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as loaded:
        if any(key not in loaded for key in required):
            raise ValueError("causal decision cache is incomplete")
        arrays = {key: np.asarray(loaded[key]) for key in required}
    expected_shape = (expected_examples, OUTPUT_FRAMES)
    if any(values.shape != expected_shape for values in arrays.values()):
        raise ValueError("causal decision cache tensor geometry differs")
    valid = np.isfinite(arrays["decision_score"])
    if representation.startswith("qualified_teacher"):
        if not np.all(valid):
            raise ValueError("teacher causal decision targets must be complete")
    elif np.any(valid[:, : min(WINDOW_LENGTHS_FRAMES) - 1]):
        raise ValueError("student cache exposes prefixes shorter than deployment")
    return metadata, arrays


def _rank_percentiles(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Return deterministic global percentile ranks for valid matrix entries."""

    matrix = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if matrix.shape != mask.shape or not np.any(mask):
        raise ValueError("rank-percentile inputs differ or are empty")
    flat = matrix[mask]
    order = np.argsort(flat, kind="mergesort")
    ranks = np.empty(len(flat), dtype=np.float64)
    ranks[order] = np.arange(len(flat), dtype=np.float64)
    ranks /= max(1, len(flat) - 1)
    result = np.full(matrix.shape, np.nan, dtype=np.float64)
    result[mask] = ranks
    return result


def deployable_causal_mask(decision_scores: np.ndarray) -> np.ndarray:
    """Mask teacher endpoints the student suffix decoder can actually score."""

    values = np.asarray(decision_scores)
    if values.ndim != 2 or values.shape[1] != OUTPUT_FRAMES:
        raise ValueError("causal decision scores differ from student output")
    valid = np.isfinite(values)
    valid[:, : min(WINDOW_LENGTHS_FRAMES) - 1] = False
    return valid


def validate_causal_loss_contract(
    *,
    teacher_causal_window_cache: Path | None,
    ranking_weight: float,
    tail_ranking_weight: float,
) -> None:
    """Reject clip-label ranking losses on randomly sampled causal prefixes."""

    if teacher_causal_window_cache and (ranking_weight or tail_ranking_weight):
        raise ValueError("causal-window transfer cannot use clip-label ranking losses")


def validate_reference_causal_contract(
    metadata: dict, *, architecture: dict, features_sha256: str
) -> None:
    """Bind disagreement mining to the current student timeline and features."""

    if (
        metadata.get("source_student", {}).get("architecture") != architecture
        or metadata.get("corpus", {}).get("features_sha256") != features_sha256
    ):
        raise ValueError("reference student causal cache uses different inputs")


def load_expanded_public_negatives(
    root: Path,
    *,
    source_manifest: Path,
    continuous_lock: Path,
) -> tuple[dict, np.ndarray]:
    """Load the complete speaker-split training-speech feature inventory."""
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if (
        metadata.get("schema_version") != 1
        or metadata.get("representation")
        != "expanded_public_speech_fixed_context_features"
        or metadata.get("selection", {}).get("split") != "train"
        or metadata.get("selection", {}).get("source_group") != "public_speech"
    ):
        raise ValueError("expanded public-negative cache has the wrong contract")
    if metadata.get("source_manifest", {}).get("sha256") != sha256_file(
        source_manifest
    ) or metadata.get("continuous_lock", {}).get("sha256") != sha256_file(
        continuous_lock
    ):
        raise ValueError("expanded public-negative cache provenance differs")
    feature_path = Path(metadata.get("features", {}).get("path", ""))
    if not feature_path.is_file() or metadata.get("features", {}).get(
        "sha256"
    ) != sha256_file(feature_path):
        raise ValueError("expanded public-negative feature bytes drifted")
    features = np.load(feature_path, mmap_mode="r")
    expected_shape = tuple(metadata.get("features", {}).get("shape", ()))
    if features.shape != expected_shape or features.shape[1:] != INPUT_SHAPE:
        raise ValueError("expanded public-negative feature shape differs")
    if len(features) != int(metadata.get("count", -1)):
        raise ValueError("expanded public-negative count differs")
    return metadata, features


def require_teacher_gates(
    clip_report: Path, continuous_report: Path
) -> tuple[dict, dict]:
    clip = json.loads(clip_report.read_text())
    continuous = json.loads(continuous_report.read_text())
    if (
        clip.get("gate_scope") != "teacher_clip_and_anchor_prequalification"
        or clip.get("qualified") is not True
    ):
        raise ValueError("phoneme teacher clip/device qualification did not pass")
    if clip.get("phones", {}).get("phrase_id") != "kizz-control":
        raise ValueError("teacher qualification is for the wrong phrase")
    if int(clip.get("counts", {}).get("natural_positive", 0)) < 20:
        raise ValueError("teacher qualification lacks target-channel positives")
    if int(clip.get("counts", {}).get("false_wake_accepted", -1)) != 0:
        raise ValueError("teacher accepted a locked household false wake")
    if float(clip.get("validation_operating_point", {}).get("recall", 0)) < 0.9:
        raise ValueError("teacher validation recall is below 90 percent")
    if (
        continuous.get("gate_scope") != "untouched_continuous_qualification"
        or continuous.get("qualified") is not True
    ):
        raise ValueError("teacher continuous qualification did not pass")
    bound_qualification_sha = continuous.get(
        "teacher_qualification_sha256"
    ) or continuous.get("teacher_qualification", {}).get("report_sha256")
    if bound_qualification_sha != sha256_file(clip_report):
        raise ValueError("continuous report is not bound to the clip qualification")
    counts = continuous.get("counts", {})
    if float(counts.get("exposure_hours", 0)) < 100.0:
        raise ValueError("continuous qualification measured less than 100 hours")
    if float(counts.get("faph_upper_95", math.inf)) > 0.1:
        raise ValueError("continuous qualification exceeds the FAPH confidence bound")
    for artifact_hash in ("weights_sha256", "config_sha256", "tokenizer_vocab_sha256"):
        if continuous.get("model", {}).get(artifact_hash) != clip.get("model", {}).get(
            artifact_hash
        ):
            raise ValueError(
                f"continuous and clip reports bind different teacher {artifact_hash}"
            )
    return clip, continuous


def map_ordered_targets(
    targets: np.ndarray, provenance: dict, contract: dict
) -> np.ndarray:
    values = np.asarray(targets, dtype=np.int32)
    expected_contract = compact_phone_contract()
    if contract != expected_contract:
        raise ValueError("overlay targets use a different compact phone contract")
    expected_topology = OrderedStateTopology(KIZZ_CONTROL.phones, states_per_phone=2)
    phrase = provenance.get("wake_phrase")
    if (
        not isinstance(phrase, dict)
        or phrase.get("phrase_id") != KIZZ_CONTROL.phrase_id
    ):
        raise ValueError("overlay targets do not use the Kizz Control phrase")
    if tuple(phrase.get("phones", ())) != expected_topology.phones:
        raise ValueError("overlay target phone topology differs from Kizz Control")
    if provenance.get("states_per_phone") != expected_topology.states_per_phone:
        raise ValueError(
            "overlay targets do not use the Kizz Control double-state contract"
        )
    if provenance.get("state_count") != expected_topology.state_count:
        raise ValueError("overlay target state_count differs from its topology")
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("overlay targets must have shape [examples, frames]")
    source_times = np.asarray(
        provenance.get("target_frame_times_seconds"), dtype=np.float64
    )
    if source_times.ndim != 1 or len(source_times) != values.shape[1]:
        raise ValueError("overlay target timeline does not match target frames")
    if np.any(~np.isfinite(source_times)) or np.any(np.diff(source_times) <= 0):
        raise ValueError(
            "overlay target timeline must be finite and strictly increasing"
        )
    raw_values = np.asarray(targets)
    if not np.issubdtype(raw_values.dtype, np.integer) and np.any(raw_values != values):
        raise ValueError("overlay targets contain non-integer state IDs")
    if np.any(values < 0) or np.any(values >= expected_topology.state_count):
        raise ValueError("overlay targets contain unexpected ordered-state IDs")
    target_times = student_output_times_seconds(
        student_flags(len(contract["tokens"])), OUTPUT_FRAMES
    )
    indexes = np.abs(source_times[:, None] - target_times[None, :]).argmin(axis=0)
    selected = values[:, indexes]
    state_to_token = {
        expected_topology.background_index: int(contract["blank_id"]),
        expected_topology.silence_index: int(contract["blank_id"]),
    }
    for phone_index, token_id in enumerate(contract["canonical_path"]):
        for state_index in range(expected_topology.states_per_phone):
            state_to_token[
                expected_topology.phone_state_index(phone_index, state_index)
            ] = int(token_id)
    result = np.empty(selected.shape, dtype=np.int16)
    for state_id, token_id in state_to_token.items():
        result[selected == state_id] = token_id
    return result


def require_overlay_parent_binding(
    corpus_rows: list[dict], overlay_provenance: dict
) -> dict:
    """Bind every overlay parent to the exact active clean train inventory."""
    clean = {
        (str(row.get("source_audio_sha256")), str(row.get("provider")))
        for row in corpus_rows
        if row.get("split") == "train"
        and int(row.get("label", 0)) == 1
        and row.get("source_group") != DEVICE_POSITIVE_SOURCE_GROUP
    }
    overlay_rows = [
        row
        for row in overlay_provenance.get("examples", [])
        if row.get("split") == "train" and row.get("variant") != "clean"
    ]
    overlay = {
        (str(row.get("source_audio_sha256")), str(row.get("provider")))
        for row in overlay_rows
    }
    if not clean or not overlay or clean != overlay:
        raise ValueError("overlay parents differ from the active clean train inventory")
    expected_variants = {"overlay-0", "overlay-1", "overlay-2", "overlay-3"}
    variants_by_parent: dict[tuple[str, str], set[str]] = {
        parent: set() for parent in overlay
    }
    for row in overlay_rows:
        parent = (str(row.get("source_audio_sha256")), str(row.get("provider")))
        variants_by_parent.setdefault(parent, set()).add(str(row.get("variant")))
    if any(variants != expected_variants for variants in variants_by_parent.values()):
        raise ValueError(
            "overlay parents do not each realize the four-variant contract"
        )
    ordered = sorted(clean)
    return {
        "algorithm": "exact_audio_sha256_provider_set_and_four_variants_v1",
        "parents": len(ordered),
        "binding_sha256": _canonical_hash(ordered),
    }


class DistillationBatcher:
    def __init__(
        self,
        features: np.ndarray,
        hard_targets: np.ndarray,
        teacher_targets: np.ndarray,
        teacher_occupation_targets: np.ndarray,
        teacher_representations: np.ndarray,
        teacher_temporal_representations: np.ndarray,
        teacher_sequence_targets: np.ndarray,
        teacher_sequence_supervision_mask: np.ndarray,
        collision_path_indexes: np.ndarray,
        streaming_window_targets: dict[str, np.ndarray],
        teacher_causal_targets: dict[str, np.ndarray] | None,
        reference_student_causal_targets: dict[str, np.ndarray] | None,
        expanded_public_negative_features: np.ndarray,
        rows: list[dict],
        overlay_features: np.ndarray,
        overlay_targets: np.ndarray,
        overlay_providers: list[str],
        overlay_teacher_targets: np.ndarray | None,
        overlay_teacher_occupation_targets: np.ndarray | None,
        overlay_teacher_sequence_targets: np.ndarray | None,
        overlay_teacher_sequence_supervision_mask: np.ndarray | None,
        device_parent_features: dict[int, np.ndarray],
        noise_sources: list[tuple[str, Path]],
        *,
        batch_size: int,
        seed: int,
        blank_id: int,
    ) -> None:
        if batch_size < 8 or batch_size % 8:
            raise ValueError("batch size must be a positive multiple of eight")
        self.features = features
        self.hard_targets = hard_targets
        self.teacher_targets = teacher_targets
        self.teacher_occupation_targets = teacher_occupation_targets
        self.teacher_representations = teacher_representations
        self.teacher_temporal_representations = teacher_temporal_representations
        self.teacher_sequence_targets = teacher_sequence_targets
        self.teacher_sequence_supervision_mask = teacher_sequence_supervision_mask
        self.collision_path_indexes = collision_path_indexes
        self.streaming_window_targets = streaming_window_targets
        self.teacher_causal_targets = teacher_causal_targets
        self.reference_student_causal_targets = reference_student_causal_targets
        self.expanded_public_negative_features = expanded_public_negative_features
        self.rows = rows
        self.overlay_features = overlay_features
        self.overlay_targets = overlay_targets
        self.overlay_providers = overlay_providers
        self.overlay_teacher_targets = overlay_teacher_targets
        self.overlay_teacher_occupation_targets = overlay_teacher_occupation_targets
        self.overlay_teacher_sequence_targets = overlay_teacher_sequence_targets
        self.overlay_teacher_sequence_supervision_mask = (
            overlay_teacher_sequence_supervision_mask
        )
        self.device_parent_features = device_parent_features
        self.noise = [
            (name, open_feature_archive(path)) for name, path in noise_sources
        ]
        if any(len(values) == 0 for _, values in self.noise):
            raise ValueError("noise feature archives must not be empty")
        self.batch_size = batch_size
        self.seed = seed
        self.blank_id = blank_id
        self.provider_counts: Counter[str] = Counter()
        self.positive_variant_provider_counts: Counter[tuple[str, str]] = Counter()
        self.negative_group_counts: Counter[str] = Counter()
        self.hard_negative_counts: Counter[str] = Counter()
        self.hard_positive_counts: Counter[str] = Counter()
        self.noise_source_counts: Counter[str] = Counter()
        self.expanded_public_negative_count = 0
        self.expanded_public_order = np.random.default_rng(seed + 238).permutation(
            len(expanded_public_negative_features)
        )
        if not len(self.expanded_public_order):
            raise ValueError("expanded public-negative cache must not be empty")
        self.clean_positive = positive_indices_by_provider(rows, "clean")
        self.device_positive = positive_indices_by_provider(rows, "device")
        train = [index for index, row in enumerate(rows) if row["split"] == "train"]
        self.overlay_positive = {
            provider: np.asarray(
                [i for i, value in enumerate(overlay_providers) if value == provider],
                dtype=np.int64,
            )
            for provider in APPROVED_PROVIDERS
        }
        if len(overlay_features) != len(overlay_targets) or len(
            overlay_features
        ) != len(overlay_providers):
            raise ValueError("overlay feature, target, and provider counts differ")
        if (overlay_teacher_targets is None) != (
            overlay_teacher_occupation_targets is None
        ):
            raise ValueError(
                "overlay teacher posterior and occupation caches must pair"
            )
        if overlay_teacher_targets is not None and (
            overlay_teacher_targets.shape
            != (len(overlay_features), OUTPUT_FRAMES, teacher_targets.shape[2])
            or overlay_teacher_occupation_targets.shape != overlay_teacher_targets.shape
        ):
            raise ValueError("overlay teacher targets must match overlay features")
        if overlay_teacher_sequence_targets is not None and (
            overlay_teacher_sequence_targets.shape != (len(overlay_features), 2)
            or not np.isfinite(overlay_teacher_sequence_targets).all()
        ):
            raise ValueError(
                "overlay teacher sequence targets must match overlay features"
            )
        if (overlay_teacher_sequence_targets is None) != (
            overlay_teacher_sequence_supervision_mask is None
        ) or (
            overlay_teacher_sequence_supervision_mask is not None
            and overlay_teacher_sequence_supervision_mask.shape
            != (len(overlay_features),)
        ):
            raise ValueError(
                "overlay teacher sequence mask must match overlay features"
            )
        if self.teacher_sequence_targets.shape != (len(rows), 2):
            raise ValueError(
                "teacher sequence targets must be [corpus, canonical/margin]"
            )
        if self.teacher_sequence_supervision_mask.shape != (len(rows),):
            raise ValueError("teacher sequence supervision mask must match corpus")
        if self.teacher_occupation_targets.shape != self.teacher_targets.shape:
            raise ValueError("teacher occupation targets must match posterior targets")
        if self.teacher_representations.ndim != 2 or len(
            self.teacher_representations
        ) != len(rows):
            raise ValueError("teacher representations must be [corpus, dimension]")
        if self.teacher_temporal_representations.shape[:2] != (
            len(rows),
            OUTPUT_FRAMES,
        ):
            raise ValueError(
                "teacher temporal representations must match corpus frames"
            )
        if self.collision_path_indexes.shape != (len(rows),):
            raise ValueError("collision path supervision must match the corpus")
        if any(
            values.shape != (len(rows),) for values in streaming_window_targets.values()
        ):
            raise ValueError("streaming-window targets must match the corpus")
        self.causal_valid = None
        self.causal_disagreement = None
        if teacher_causal_targets is not None:
            expected = (len(rows), OUTPUT_FRAMES)
            if any(
                values.shape != expected for values in teacher_causal_targets.values()
            ):
                raise ValueError("teacher causal targets must match corpus endpoints")
            complete = np.isfinite(teacher_causal_targets["decision_score"])
            if not np.all(complete):
                raise ValueError("teacher causal targets must cover every endpoint")
            self.causal_valid = deployable_causal_mask(
                teacher_causal_targets["decision_score"]
            )
            if reference_student_causal_targets is not None:
                if any(
                    values.shape != expected
                    for values in reference_student_causal_targets.values()
                ):
                    raise ValueError("reference student causal targets differ")
                train_rows = np.asarray(
                    [row.get("split") == "train" for row in rows], dtype=bool
                )[:, None]
                reference_valid = np.isfinite(
                    reference_student_causal_targets["decision_score"]
                )
                valid = self.causal_valid & reference_valid & train_rows
                teacher_rank = _rank_percentiles(
                    teacher_causal_targets["decision_score"], valid
                )
                student_rank = _rank_percentiles(
                    reference_student_causal_targets["decision_score"], valid
                )
                self.causal_disagreement = np.abs(teacher_rank - student_rank)
        if any(provider not in APPROVED_PROVIDERS for provider in overlay_providers):
            raise ValueError("overlay provenance contains an unapproved provider")
        if any(
            not len(self.clean_positive[p])
            or not len(self.overlay_positive[p])
            or not len(self.device_positive[p])
            for p in APPROVED_PROVIDERS
        ):
            raise ValueError(
                "every approved provider needs clean, overlaid, and device positives"
            )
        self.negative = {
            group: np.asarray(
                [
                    i
                    for i in train
                    if rows[i]["label"] == 0 and rows[i].get("source_group") == group
                ],
                dtype=np.int64,
            )
            for group in (
                "public_speech",
                "kizz_control_phonetic_collision",
                "device_collision",
            )
        }
        if any(not len(value) for value in self.negative.values()):
            raise ValueError("paired negative groups must not be empty")
        self.hard_clean_positive = {}
        for provider, indexes in self.clean_positive.items():
            order = sorted(
                indexes.tolist(),
                key=lambda index: (
                    float(self.streaming_window_targets["decision_score"][index]),
                    str(self.rows[index].get("source_id", "")),
                ),
            )
            hard_count = max(1, math.ceil(len(order) / 2))
            self.hard_clean_positive[provider] = np.asarray(
                order[:hard_count], dtype=np.int64
            )
        self.hard_negative = {}
        for group, indexes in self.negative.items():
            order = sorted(
                indexes.tolist(),
                key=lambda index: (
                    float(self.streaming_window_targets["decision_score"][index]),
                    str(self.rows[index].get("source_id", "")),
                ),
                reverse=True,
            )
            hard_count = max(1, math.ceil(len(order) / 2))
            self.hard_negative[group] = np.asarray(order[:hard_count], dtype=np.int64)

    def _causal_endpoint(self, index: int, global_slot: int, rng) -> int:
        """Choose random, disagreement/hard, and terminal endpoints in rotation."""

        if self.teacher_causal_targets is None:
            deployed_end = int(
                self.streaming_window_targets["deployment_end_frame"][index]
            )
            raw_end = int(self.streaming_window_targets["raw_end_frame"][index])
            return (
                deployed_end if deployed_end >= min(WINDOW_LENGTHS_FRAMES) else raw_end
            )
        valid = np.flatnonzero(self.causal_valid[index])
        mode = global_slot % 3
        if mode == 0:
            endpoint_index = int(rng.choice(valid))
        elif mode == 1 and self.causal_disagreement is not None:
            values = self.causal_disagreement[index, valid]
            endpoint_index = int(valid[int(np.nanargmax(values))])
        elif mode == 1:
            values = self.teacher_causal_targets["decision_score"][index, valid]
            endpoint_index = int(valid[int(np.argmax(values))])
        else:
            endpoint_index = int(valid[-1])
        return endpoint_index + 1

    def _causal_teacher_target(self, index: int, endpoint: int) -> np.ndarray:
        if self.teacher_causal_targets is None:
            return self.teacher_sequence_targets[index]
        column = int(endpoint) - 1
        return np.asarray(
            [
                self.teacher_causal_targets["decision_score"][index, column],
                self.teacher_causal_targets["raw_collision_margin"][index, column],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _window(values, rng):
        decoded = decode_frontend_features(values)
        if decoded.shape == INPUT_SHAPE:
            return np.asarray(decoded, dtype=np.float32)
        start = int(rng.integers(0, max(1, len(decoded) - INPUT_SHAPE[0] + 1)))
        window = np.asarray(decoded[start : start + INPUT_SHAPE[0]], dtype=np.float32)
        if len(window) == INPUT_SHAPE[0]:
            return window
        padded = np.zeros(INPUT_SHAPE, dtype=np.float32)
        padded[: len(window)] = window
        return padded

    def batch(self, step: int):
        rng = np.random.default_rng(self.seed + step)
        vocabulary = self.teacher_targets.shape[-1]
        x = np.empty((self.batch_size,) + INPUT_SHAPE, dtype=np.float32)
        hard = np.full((self.batch_size, OUTPUT_FRAMES), -1, dtype=np.int32)
        soft = np.zeros((self.batch_size, OUTPUT_FRAMES, vocabulary), dtype=np.float32)
        soft_mask = np.zeros(self.batch_size, dtype=np.float32)
        occupation = np.zeros_like(soft)
        occupation_mask = np.zeros(self.batch_size, dtype=np.float32)
        representation = np.zeros(
            (self.batch_size, self.teacher_representations.shape[1]), dtype=np.float32
        )
        representation_mask = np.zeros(self.batch_size, dtype=np.float32)
        temporal_representation = np.zeros(
            (
                self.batch_size,
                OUTPUT_FRAMES,
                self.teacher_temporal_representations.shape[2],
            ),
            dtype=np.float32,
        )
        temporal_representation_mask = np.zeros(self.batch_size, dtype=np.float32)
        teacher_sequence = np.zeros((self.batch_size, 2), dtype=np.float32)
        sequence_mask = np.zeros(self.batch_size, dtype=np.float32)
        collision_negative_mask = np.zeros(self.batch_size, dtype=np.float32)
        named_collision_path = np.full(self.batch_size, -1, dtype=np.int32)
        scoring_endpoints = np.full(self.batch_size, OUTPUT_FRAMES, dtype=np.int32)
        labels = np.zeros(self.batch_size, dtype=np.float32)
        paired_clean = np.zeros((self.batch_size,) + INPUT_SHAPE, dtype=np.float32)
        paired_clean_mask = np.zeros(self.batch_size, dtype=np.float32)
        half = self.batch_size // 2
        for slot in range(half):
            global_slot = step * half + slot
            variant = POSITIVE_VARIANTS[global_slot % len(POSITIVE_VARIANTS)]
            provider = APPROVED_PROVIDERS[global_slot % len(APPROVED_PROVIDERS)]
            self.provider_counts[provider] += 1
            self.positive_variant_provider_counts[(variant, provider)] += 1
            labels[slot] = 1
            if variant == "overlay":
                index = int(rng.choice(self.overlay_positive[provider]))
                x[slot] = np.asarray(self.overlay_features[index], dtype=np.float32)
                hard[slot] = self.overlay_targets[index]
                if self.overlay_teacher_targets is not None:
                    soft[slot] = self.overlay_teacher_targets[index]
                    soft_mask[slot] = 1
                    occupation[slot] = self.overlay_teacher_occupation_targets[index]
                    occupation_mask[slot] = 1
                if self.overlay_teacher_sequence_targets is not None:
                    teacher_sequence[slot] = self.overlay_teacher_sequence_targets[
                        index
                    ]
                    sequence_mask[slot] = (
                        self.overlay_teacher_sequence_supervision_mask[index]
                    )
                active = np.flatnonzero(hard[slot] != self.blank_id)
                if len(active):
                    scoring_endpoints[slot] = max(
                        min(WINDOW_LENGTHS_FRAMES), int(active[-1]) + 1
                    )
            else:
                source = (
                    self.device_positive if variant == "device" else self.clean_positive
                )
                full_schedule = len(POSITIVE_VARIANTS) * len(APPROVED_PROVIDERS)
                use_hard_pool = (
                    variant == "clean" and (global_slot // full_schedule) % 2 == 0
                )
                pool = (
                    self.hard_clean_positive[provider]
                    if use_hard_pool
                    else source[provider]
                )
                index = int(rng.choice(pool))
                if use_hard_pool:
                    self.hard_positive_counts[variant] += 1
                x[slot] = np.asarray(self.features[index], dtype=np.float32)
                if variant == "device" and self.device_parent_features:
                    paired_clean[slot] = self.device_parent_features[index]
                    paired_clean_mask[slot] = 1
                hard[slot] = self.hard_targets[index]
                soft[slot] = self.teacher_targets[index]
                soft_mask[slot] = 1
                occupation[slot] = self.teacher_occupation_targets[index]
                occupation_mask[slot] = 1
                representation[slot] = self.teacher_representations[index]
                representation_mask[slot] = 1
                temporal_representation[slot] = self.teacher_temporal_representations[
                    index
                ]
                temporal_representation_mask[slot] = 1
                scoring_endpoints[slot] = self._causal_endpoint(index, global_slot, rng)
                teacher_sequence[slot] = self._causal_teacher_target(
                    index, scoring_endpoints[slot]
                )
                sequence_mask[slot] = self.teacher_sequence_supervision_mask[index]
        groups = (
            "public_speech",
            "kizz_control_phonetic_collision",
            "device_collision",
            "no_speech",
        )
        for offset in range(half):
            slot = half + offset
            group = groups[(step * half + offset) % len(groups)]
            self.negative_group_counts[group] += 1
            if group == "no_speech":
                name, source = self.noise[(step + offset) % len(self.noise)]
                self.noise_source_counts[name] += 1
                item = source[int(rng.integers(0, len(source)))]
                x[slot] = self._window(item, rng)
                hard[slot] = self.blank_id
                scoring_endpoints[slot] = min(WINDOW_LENGTHS_FRAMES) + (
                    (step * half + offset)
                    % (OUTPUT_FRAMES - min(WINDOW_LENGTHS_FRAMES) + 1)
                )
            else:
                if group == "public_speech":
                    order_index = self.expanded_public_negative_count % len(
                        self.expanded_public_order
                    )
                    index = int(self.expanded_public_order[order_index])
                    self.expanded_public_negative_count += 1
                    x[slot] = np.asarray(
                        self.expanded_public_negative_features[index],
                        dtype=np.float32,
                    )
                    hard[slot] = self.blank_id
                    scoring_endpoints[slot] = min(WINDOW_LENGTHS_FRAMES) + (
                        (step * half + offset)
                        % (OUTPUT_FRAMES - min(WINDOW_LENGTHS_FRAMES) + 1)
                    )
                    continue
                group_slot = (step * half + offset) // len(groups)
                use_hard_pool = group_slot % 2 == 0
                pool = (
                    self.hard_negative[group] if use_hard_pool else self.negative[group]
                )
                index = int(rng.choice(pool))
                if use_hard_pool:
                    self.hard_negative_counts[group] += 1
                x[slot] = np.asarray(self.features[index], dtype=np.float32)
                hard[slot] = self.hard_targets[index]
                soft[slot] = self.teacher_targets[index]
                soft_mask[slot] = 1
                representation[slot] = self.teacher_representations[index]
                representation_mask[slot] = 1
                temporal_representation[slot] = self.teacher_temporal_representations[
                    index
                ]
                temporal_representation_mask[slot] = 1
                scoring_endpoints[slot] = self._causal_endpoint(
                    index, step * half + offset, rng
                )
                teacher_sequence[slot] = self._causal_teacher_target(
                    index, scoring_endpoints[slot]
                )
                sequence_mask[slot] = self.teacher_sequence_supervision_mask[index]
                if group == "kizz_control_phonetic_collision":
                    collision_negative_mask[slot] = 1
                    named_collision_path[slot] = self.collision_path_indexes[index]
        if np.any(scoring_endpoints < min(WINDOW_LENGTHS_FRAMES)) or np.any(
            scoring_endpoints > OUTPUT_FRAMES
        ):
            raise ValueError("mined streaming endpoint is outside student output")
        order = rng.permutation(self.batch_size)
        return (
            x[order],
            hard[order],
            soft[order],
            soft_mask[order],
            occupation[order],
            occupation_mask[order],
            representation[order],
            representation_mask[order],
            temporal_representation[order],
            temporal_representation_mask[order],
            labels[order],
            teacher_sequence[order],
            sequence_mask[order],
            collision_negative_mask[order],
            named_collision_path[order],
            scoring_endpoints[order],
            paired_clean[order],
            paired_clean_mask[order],
        )


def deployment_path_scores(
    logits,
    contract,
    *,
    algorithm: str = "max_add_ctc_viterbi",
    endpoints=None,
    return_selected_paths: bool = False,
):
    """Return differentiable canonical fit and collision margin.

    The max-add mode is firmware-exact before the hard beta gate. Forward-sum
    uses the same bounded recurrence with log-add-exp and is the differentiable
    contract for the forward-sum firmware decoder. Both deliberately return the
    strongest raw canonical suffix so rejected examples retain a finite signal;
    qualification applies the hard beta gate.
    """
    values = tf.convert_to_tensor(logits)
    if values.shape.rank != 3:
        raise ValueError("student logits must have shape [batch, time, token]")
    if algorithm not in ("max_add_ctc_viterbi", "forward_sum_ctc"):
        raise ValueError("unsupported student path-scoring algorithm")
    # Joint students append one causal binary decision channel after the
    # compact CTC vocabulary.  The decoder must never normalize that channel
    # into the phonetic posterior distribution.
    values = values[:, :, : len(contract["tokens"])]
    paths = [contract["canonical_path"], *contract["collision_paths"].values()]
    max_path_length = max(len(path) for path in paths)
    path_lengths = tf.constant([len(path) for path in paths], dtype=tf.int32)
    blank_id = int(contract["blank_id"])
    state_count = 2 * max_path_length + 1
    expanded = []
    for path in paths:
        states = [blank_id]
        for token in path:
            states.extend((int(token), blank_id))
        states.extend([blank_id] * (state_count - len(states)))
        expanded.append(states)
    expanded_tokens = tf.constant(expanded, dtype=tf.int32)
    state_indexes = tf.range(state_count, dtype=tf.int32)[None, :]
    valid_states = state_indexes < (2 * path_lengths[:, None] + 1)
    skip_allowed = tf.logical_and(
        state_indexes >= 2,
        tf.logical_and(
            expanded_tokens != blank_id,
            expanded_tokens
            != tf.concat(
                [
                    tf.fill([len(paths), 2], blank_id),
                    expanded_tokens[:, :-2],
                ],
                axis=1,
            ),
        ),
    )

    batch = tf.shape(values)[0]
    frame_count = tf.shape(values)[1]
    maximum_window = max(WINDOW_LENGTHS_FRAMES)
    if endpoints is None:
        selected_endpoints = tf.fill([batch], frame_count)
    else:
        selected_endpoints = tf.cast(tf.convert_to_tensor(endpoints), tf.int32)
        if selected_endpoints.shape.rank != 1:
            raise ValueError("student scoring endpoints must have shape [batch]")
        tf.debugging.assert_equal(
            tf.shape(selected_endpoints)[0],
            batch,
            message="student scoring endpoint batch differs",
        )
    tf.debugging.assert_greater_equal(
        selected_endpoints,
        min(WINDOW_LENGTHS_FRAMES),
        message="student scoring endpoint is too early",
    )
    tf.debugging.assert_less_equal(
        selected_endpoints,
        frame_count,
        message="student scoring endpoint exceeds model output",
    )
    positions = (
        selected_endpoints[:, None]
        - maximum_window
        + tf.range(maximum_window, dtype=tf.int32)[None, :]
    )
    valid_positions = positions >= 0
    safe_positions = tf.maximum(positions, 0)
    selected_logits = tf.gather(values, safe_positions, axis=1, batch_dims=1)
    log_probs = tf.nn.log_softmax(selected_logits, axis=-1)
    padded = tf.one_hot(
        int(contract["blank_id"]),
        len(contract["tokens"]),
        on_value=tf.cast(0.0, values.dtype),
        off_value=tf.cast(-1.0e9, values.dtype),
    )
    log_probs = tf.where(valid_positions[:, :, None], log_probs, padded)
    window_count = len(WINDOW_LENGTHS_FRAMES)
    negative = tf.cast(-1.0e9, values.dtype)
    scores = tf.fill([batch, window_count, len(paths), state_count], negative)
    starts = tf.constant(
        [maximum_window - length for length in WINDOW_LENGTHS_FRAMES],
        dtype=tf.int32,
    )
    initial_state_mask = tf.logical_and(valid_states, state_indexes < 2)
    for frame_index in range(maximum_window):
        emissions = tf.gather(log_probs[:, frame_index, :], expanded_tokens, axis=1)
        initial = tf.where(initial_state_mask[None, :, :], emissions, negative)[
            :, None, :, :
        ]
        from_one = tf.concat(
            [tf.fill(tf.shape(scores[..., :1]), negative), scores[..., :-1]],
            axis=-1,
        )
        from_two = tf.concat(
            [tf.fill(tf.shape(scores[..., :2]), negative), scores[..., :-2]],
            axis=-1,
        )
        from_two = tf.where(skip_allowed[None, None, :, :], from_two, negative)
        if algorithm == "max_add_ctc_viterbi":
            advanced = tf.maximum(scores, tf.maximum(from_one, from_two))
        else:
            advanced = tf.reduce_logsumexp(
                tf.stack((scores, from_one, from_two), axis=0), axis=0
            )
        advanced = advanced + emissions[:, None, :, :]
        advanced = tf.where(valid_states[None, None, :, :], advanced, negative)
        at_start = tf.equal(starts, frame_index)[None, :, None, None]
        after_start = tf.less(starts, frame_index)[None, :, None, None]
        scores = tf.where(
            at_start,
            initial,
            tf.where(after_start, advanced, scores),
        )

    final_blank = 2 * path_lengths
    final_token = final_blank - 1
    blank_scores = tf.reduce_sum(
        scores
        * tf.one_hot(final_blank, state_count, dtype=scores.dtype)[None, None, :, :],
        axis=-1,
    )
    token_scores = tf.reduce_sum(
        scores
        * tf.one_hot(final_token, state_count, dtype=scores.dtype)[None, None, :, :],
        axis=-1,
    )
    if algorithm == "max_add_ctc_viterbi":
        completed = tf.maximum(blank_scores, token_scores)
    else:
        completed = tf.reduce_logsumexp(
            tf.stack((blank_scores, token_scores), axis=0), axis=0
        )
    path_scores = completed / tf.cast(path_lengths[None, None, :], scores.dtype)
    canonical_by_window = path_scores[:, :, 0]
    collision_margin_by_window = canonical_by_window - tf.reduce_max(
        path_scores[:, :, 1:], axis=2
    )
    valid_windows = (
        selected_endpoints[:, None]
        >= tf.constant(WINDOW_LENGTHS_FRAMES, dtype=tf.int32)[None, :]
    )
    canonical_by_window = tf.where(valid_windows, canonical_by_window, negative)
    collision_margin_by_window = tf.where(
        valid_windows, collision_margin_by_window, negative
    )
    maximum_canonical = tf.reduce_max(canonical_by_window, axis=1, keepdims=True)
    tied_margin = tf.where(
        canonical_by_window == maximum_canonical,
        collision_margin_by_window,
        negative,
    )
    selected_window = tf.argmax(tied_margin, axis=1, output_type=tf.int32)
    selected_paths = tf.gather(path_scores, selected_window, axis=1, batch_dims=1)
    canonical_fit = selected_paths[:, 0]
    collision_margin = canonical_fit - tf.reduce_max(selected_paths[:, 1:], axis=1)
    if return_selected_paths:
        return canonical_fit, collision_margin, selected_paths
    return canonical_fit, collision_margin


def teacher_sequence_score_targets(
    teacher_log_probs: np.ndarray, contract: dict
) -> np.ndarray:
    """Measure the teacher's deployed canonical and collision decisions."""
    values = np.asarray(teacher_log_probs, dtype=np.float32)
    result = np.empty((len(values), 2), dtype=np.float32)
    decoder = student_decoder_contract(contract)
    for index, sequence in enumerate(values):
        score = exhaustive_suffix_score(
            sequence,
            contract,
            window_lengths=decoder["window_lengths_frames"],
            # Sequence transfer needs the raw canonical/margin pair even when
            # the teacher rejects the clip at the deployment beta.
            beta=-1.0e9,
        )
        result[index] = (score.canonical_fit, score.collision_margin)
    if np.any(~np.isfinite(result)):
        raise ValueError("teacher sequence targets must be finite")
    return result


def _masked_mean(values, mask):
    weights = tf.cast(mask, values.dtype)
    return tf.math.divide_no_nan(
        tf.reduce_sum(values * weights), tf.reduce_sum(weights)
    )


def _huber(values, delta: float = 1.0):
    absolute = tf.abs(values)
    return tf.where(
        absolute <= delta,
        0.5 * tf.square(absolute),
        delta * (absolute - 0.5 * delta),
    )


def strict_collision_negative_loss(
    collision_margin,
    collision_negative_mask,
    *,
    required_margin: float,
):
    """Require an explicit collision negative to favor a collision path.

    These rows are not generic background audio: their corpus contract says
    they contain a declared phonetic neighbor of the wake phrase.  A weak
    canonical score alone must therefore not make the loss disappear.  The
    deployment decoder rejects the row deterministically when a collision path
    outranks the canonical path, so train that exact decision boundary.
    """
    violation = tf.nn.relu(collision_margin + float(required_margin))
    return _masked_mean(violation, collision_negative_mask)


def delayed_occupation_loss(
    student_logits,
    teacher_log_occupations,
    mask,
    *,
    max_delay_frames: int,
):
    """Match sequence-conditioned CTC targets with bounded causal delay.

    One delay is selected per example, not independently per frame.  This is
    the Temporal Alignment Buffer contract: a causal student may emit later
    than its non-streaming teacher, but cannot move evidence arbitrarily.
    """

    if max_delay_frames < 0:
        raise ValueError("maximum occupation delay must be non-negative")
    frame_count = student_logits.shape[1]
    if frame_count is None or max_delay_frames >= int(frame_count):
        raise ValueError("maximum occupation delay is outside the output sequence")
    student_log_probs = tf.nn.log_softmax(student_logits, axis=-1)
    teacher_probs = tf.exp(teacher_log_occupations)
    delayed = []
    for delay in range(max_delay_frames + 1):
        usable = int(frame_count) - delay
        cross_entropy = -tf.reduce_mean(
            tf.reduce_sum(
                teacher_probs[:, :usable]
                * student_log_probs[:, delay : delay + usable],
                axis=-1,
            ),
            axis=-1,
        )
        delayed.append(cross_entropy)
    return _masked_mean(tf.reduce_min(tf.stack(delayed, axis=1), axis=1), mask)


def utterance_representation_loss(student_vectors, teacher_vectors, mask):
    """Distill normalized projected representations; the adapter is training-only."""

    student = tf.convert_to_tensor(student_vectors)
    if student.shape.rank != 2 or student.shape[-1] != teacher_vectors.shape[-1]:
        raise ValueError("student and teacher representation dimensions differ")
    student = tf.math.l2_normalize(student, axis=-1)
    teacher = tf.math.l2_normalize(teacher_vectors, axis=-1)
    cosine = 1.0 - tf.reduce_sum(student * teacher, axis=-1)
    distance = tf.reduce_mean(tf.abs(student - teacher), axis=-1)
    return _masked_mean(cosine + distance, mask)


def channel_consistency_loss(device_logits, clean_logits, mask):
    """Make aligned clean/device views preserve the same phone posterior path."""

    device = tf.nn.softmax(device_logits, axis=-1)
    clean = tf.stop_gradient(tf.nn.softmax(clean_logits, axis=-1))
    per_example = tf.reduce_mean(tf.square(device - clean), axis=(1, 2))
    return _masked_mean(per_example, mask)


def paired_path_consistency_loss(
    device_logits, clean_logits, endpoints, mask, contract, *, algorithm
):
    """Anchor device views to clean views in the deployed CTC decision space.

    This is deliberately distinct from posterior consistency.  Both views
    receive gradients from the agreement term, while the device
    view still receives the hard CTC/path and collision losses in the main
    objective.  Matching both canonical fit and collision margin keeps
    the invariance constraint attached to the actual deployment decision.
    """
    device_fit, device_margin = deployment_path_scores(
        device_logits, contract, algorithm=algorithm, endpoints=endpoints
    )
    clean_fit, clean_margin = deployment_path_scores(
        clean_logits, contract, algorithm=algorithm, endpoints=endpoints
    )
    target = tf.stack((clean_fit, clean_margin), axis=-1)
    observed = tf.stack((device_fit, device_margin), axis=-1)
    per_example = tf.reduce_mean(tf.square(observed - target), axis=-1)
    return _masked_mean(per_example, mask)


def paired_clean_path_supervision_loss(
    clean_logits, endpoints, mask, contract, *, algorithm, margin=0.10
):
    """Supervise the clean member of every aligned device/clean pair."""
    canonical, collision_margin = deployment_path_scores(
        clean_logits, contract, algorithm=algorithm, endpoints=endpoints
    )
    per_example = -canonical + tf.nn.relu(
        tf.cast(margin, canonical.dtype) - collision_margin
    )
    return _masked_mean(per_example, mask)


def temporal_representation_loss(student_frames, teacher_frames, mask):
    """Transfer time-resolved normalized teacher acoustics to student frames."""

    student = tf.math.l2_normalize(student_frames, axis=-1)
    teacher = tf.math.l2_normalize(teacher_frames, axis=-1)
    cosine = 1.0 - tf.reduce_sum(student * teacher, axis=-1)
    distance = tf.reduce_mean(tf.abs(student - teacher), axis=-1)
    return _masked_mean(tf.reduce_mean(cosine + distance, axis=-1), mask)


def teacher_sequence_ranking_loss(student_scores, teacher_scores, mask):
    """Transfer teacher ordering across positive and negative examples."""

    teacher_delta = teacher_scores[:, None] - teacher_scores[None, :]
    student_delta = student_scores[:, None] - student_scores[None, :]
    comparable = tf.logical_and(mask[:, None] > 0.5, mask[None, :] > 0.5)
    comparable = tf.logical_and(comparable, tf.abs(teacher_delta) >= 0.05)
    weight = tf.minimum(tf.abs(teacher_delta), 1.0)
    return _masked_mean(
        tf.nn.softplus(-tf.sign(teacher_delta) * student_delta) * weight,
        comparable,
    )


def teacher_sequence_listwise_loss(
    student_scores, teacher_scores, mask, *, temperature: float
):
    """Transfer the teacher's complete ordering over sampled causal windows."""

    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("listwise temperature must be positive and finite")
    valid = tf.cast(mask > 0.5, student_scores.dtype)

    def standardize(values):
        count = tf.reduce_sum(valid)
        mean = tf.math.divide_no_nan(tf.reduce_sum(values * valid), count)
        variance = tf.math.divide_no_nan(
            tf.reduce_sum(tf.square(values - mean) * valid), count
        )
        return (values - mean) * tf.math.rsqrt(variance + 1e-4)

    teacher = standardize(tf.cast(teacher_scores, student_scores.dtype))
    student = standardize(student_scores)
    floor = tf.cast(-1e4, student_scores.dtype)
    teacher_logits = tf.where(
        valid > 0,
        teacher / tf.cast(temperature, student_scores.dtype),
        floor,
    )
    student_logits = tf.where(
        valid > 0,
        student / tf.cast(temperature, student_scores.dtype),
        floor,
    )
    target = tf.stop_gradient(tf.nn.softmax(teacher_logits))
    return -tf.reduce_sum(target * tf.nn.log_softmax(student_logits))


def distillation_loss(
    logits,
    hard_targets,
    teacher_log_probs,
    teacher_mask,
    labels,
    contract,
    *,
    teacher_occupation_targets=None,
    teacher_occupation_mask=None,
    student_hidden=None,
    teacher_representations=None,
    teacher_representation_mask=None,
    representation_weight: float = 0.0,
    teacher_temperature: float = 1.0,
    occupation_weight: float = 0.0,
    occupation_max_delay_frames: int = 0,
    scoring_endpoints=None,
    hard_weight: float,
    teacher_weight: float,
    ctc_weight: float,
    collision_weight: float,
    negative_weight: float,
    negative_score_target: float,
    teacher_sequence_targets=None,
    sequence_teacher_mask=None,
    collision_negative_mask=None,
    named_collision_path=None,
    decoder_algorithm: str = "max_add_ctc_viterbi",
    sequence_teacher_weight: float = 0.0,
    sequence_listwise_weight: float = 0.0,
    sequence_listwise_temperature: float = 1.0,
    ranking_weight: float = 0.0,
    tail_ranking_weight: float = 0.0,
    ranking_margin: float = 0.5,
    negative_collision_weight: float = 0.0,
    negative_collision_margin: float = 0.10,
):
    if not math.isfinite(float(teacher_temperature)) or teacher_temperature <= 0:
        raise ValueError("teacher temperature must be positive and finite")
    temperature = tf.cast(teacher_temperature, logits.dtype)
    student_log_probs = tf.nn.log_softmax(logits / temperature, axis=-1)
    normalized_teacher = tf.nn.log_softmax(teacher_log_probs / temperature, axis=-1)
    teacher_probs = tf.exp(normalized_teacher)
    kl = tf.reduce_mean(
        tf.reduce_sum(
            teacher_probs * (normalized_teacher - student_log_probs), axis=-1
        ),
        axis=-1,
    ) * tf.square(temperature)
    teacher_loss = _masked_mean(kl, teacher_mask)
    if teacher_occupation_targets is None:
        teacher_occupation_targets = tf.zeros_like(logits)
    if teacher_occupation_mask is None:
        teacher_occupation_mask = tf.zeros_like(labels)
    occupation_loss = delayed_occupation_loss(
        logits,
        teacher_occupation_targets,
        teacher_occupation_mask,
        max_delay_frames=occupation_max_delay_frames,
    )
    if student_hidden is None:
        representation_loss = tf.cast(0.0, logits.dtype)
    else:
        representation_loss = utterance_representation_loss(
            student_hidden, teacher_representations, teacher_representation_mask
        )
    valid = hard_targets >= 0
    safe = tf.where(valid, hard_targets, 0)
    hard = tf.keras.losses.sparse_categorical_crossentropy(
        safe, logits, from_logits=True
    )
    hard_loss = tf.math.divide_no_nan(
        tf.reduce_sum(hard * tf.cast(valid, hard.dtype)),
        tf.reduce_sum(tf.cast(valid, hard.dtype)),
    )
    canonical, collision_margin, selected_paths = deployment_path_scores(
        logits,
        contract,
        algorithm=decoder_algorithm,
        endpoints=scoring_endpoints,
        return_selected_paths=True,
    )
    positive = labels > 0.5
    negative = tf.logical_not(positive)
    positive_scores = tf.boolean_mask(canonical, positive)
    negative_scores = tf.boolean_mask(canonical, negative)
    positive_ctc = -tf.reduce_mean(positive_scores)
    collision_loss = tf.reduce_mean(
        tf.nn.relu(0.10 - tf.boolean_mask(collision_margin, positive))
    )
    negative_hinge = tf.nn.softplus(canonical - float(negative_score_target))
    negative_loss = tf.reduce_mean(tf.boolean_mask(negative_hinge, negative))

    pairwise_ranking_loss = tf.reduce_mean(
        tf.nn.softplus(
            negative_scores[:, None] - positive_scores[None, :] + float(ranking_margin)
        )
    )
    positive_count = tf.shape(positive_scores)[0]
    floor_index = tf.maximum(
        0,
        tf.cast(
            tf.math.ceil(0.10 * tf.cast(positive_count, tf.float32)),
            tf.int32,
        )
        - 1,
    )
    positive_floor = tf.sort(positive_scores)[floor_index]
    tail_count = tf.maximum(
        1,
        tf.cast(tf.math.ceil(0.25 * tf.cast(positive_count, tf.float32)), tf.int32),
    )
    positive_tail = tf.sort(positive_scores)[:tail_count]
    negative_tail = tf.sort(negative_scores, direction="DESCENDING")[:tail_count]
    tail_ranking_loss = tf.reduce_mean(
        tf.nn.softplus(
            negative_tail[:, None] - positive_tail[None, :] + float(ranking_margin)
        )
    )

    if collision_negative_mask is None:
        collision_negative_mask = tf.zeros_like(labels)
    if named_collision_path is None:
        named_collision_path = tf.fill(tf.shape(labels), -1)
    named_collision_path = tf.cast(named_collision_path, tf.int32)
    safe_named_path = tf.maximum(named_collision_path, 0)
    named_collision_score = tf.gather(
        selected_paths[:, 1:], safe_named_path, axis=1, batch_dims=1
    )
    supervised_collision_margin = tf.where(
        named_collision_path >= 0,
        canonical - named_collision_score,
        collision_margin,
    )
    negative_collision_loss = strict_collision_negative_loss(
        supervised_collision_margin,
        collision_negative_mask,
        required_margin=negative_collision_margin,
    )

    if teacher_sequence_targets is None:
        teacher_sequence_targets = tf.zeros(
            [tf.shape(logits)[0], 2], dtype=logits.dtype
        )
    if sequence_teacher_mask is None:
        sequence_teacher_mask = tf.zeros_like(labels)
    teacher_decision = tf.cast(teacher_sequence_targets[:, 0], canonical.dtype)
    sequence_teacher_loss = teacher_sequence_ranking_loss(
        canonical, teacher_decision, sequence_teacher_mask
    )
    sequence_listwise_loss = teacher_sequence_listwise_loss(
        canonical,
        teacher_decision,
        sequence_teacher_mask,
        temperature=sequence_listwise_temperature,
    )

    total = (
        hard_weight * hard_loss
        + teacher_weight * teacher_loss
        + occupation_weight * occupation_loss
        + representation_weight * representation_loss
        + ctc_weight * positive_ctc
        + collision_weight * collision_loss
        + negative_weight * negative_loss
        + sequence_teacher_weight * sequence_teacher_loss
        + sequence_listwise_weight * sequence_listwise_loss
        + ranking_weight * pairwise_ranking_loss
        + tail_ranking_weight * tail_ranking_loss
        + negative_collision_weight * negative_collision_loss
    )
    return total, (
        hard_loss,
        teacher_loss,
        occupation_loss,
        representation_loss,
        positive_ctc,
        collision_loss,
        negative_loss,
        sequence_teacher_loss,
        sequence_listwise_loss,
        pairwise_ranking_loss,
        tail_ranking_loss,
        negative_collision_loss,
    )


def _student_scores(
    model,
    features: np.ndarray,
    contract: dict,
    batch_size: int,
    *,
    decoder_algorithm: str = "max_add_ctc_viterbi",
) -> np.ndarray:
    decoder = student_decoder_contract(contract, decoder_algorithm)
    lengths = tuple(decoder["window_lengths_frames"])
    if decoder_algorithm == "forward_sum_ctc":
        from microwakeword.ctc_forward_accelerated import suffix_forward_sum_scores

        logits = []
        for start in range(0, len(features), batch_size):
            logits.append(
                np.asarray(
                    model(
                        np.asarray(
                            features[start : start + batch_size],
                            dtype=np.float32,
                        ),
                        training=False,
                    ),
                    dtype=np.float32,
                )
            )
        if not logits:
            return np.empty(0, dtype=np.float64)
        return suffix_forward_sum_scores(
            np.concatenate(logits, axis=0),
            contract,
            window_lengths=lengths,
            beta=float(decoder["beta"]),
        )
    scores = []
    for start in range(0, len(features), batch_size):
        logits = np.asarray(
            model(
                np.asarray(features[start : start + batch_size], dtype=np.float32),
                training=False,
            )
        )
        # TensorFlow 2.21 may expose EagerTensor storage as a read-only NumPy
        # view. Normalize out of place so checkpoint evaluation is independent
        # of that implementation detail.
        logits = logits - np.max(logits, axis=-1, keepdims=True)
        log_probs = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
        for sequence in log_probs:
            scored = exhaustive_suffix_score(
                sequence,
                contract,
                window_lengths=lengths,
                beta=float(decoder["beta"]),
            )
            scores.append(scored.canonical_fit if scored.eligible else -math.inf)
    return np.asarray(scores, dtype=np.float64)


def rolling_mean_scores(logits, frames: int = 2):
    """Return the strongest adjacent-frame mean from causal logits."""
    values = tf.convert_to_tensor(logits)
    if values.shape.rank == 3 and values.shape[-1] == 1:
        values = tf.squeeze(values, axis=-1)
    if values.shape.rank != 2 or frames < 1:
        raise ValueError("logits must be [batch,time] and frames must be positive")
    if values.shape[1] is not None and frames > int(values.shape[1]):
        raise ValueError("rolling window exceeds output frames")
    windows = tf.signal.frame(values, frames, 1, axis=1)
    return tf.reduce_max(tf.reduce_mean(windows, axis=-1), axis=1)


def _joint_student_scores(
    model,
    features: np.ndarray,
    contract: dict,
    batch_size: int,
    *,
    decoder_algorithm: str,
    path_weight: float,
) -> np.ndarray:
    """Score a joint CTC/binary student using both deployed signals."""
    scores = []
    for start in range(0, len(features), batch_size):
        logits = model(
            np.asarray(features[start : start + batch_size], np.float32),
            training=False,
        )
        values = tf.convert_to_tensor(logits)
        if values.shape[-1] != len(contract["tokens"]) + 4:
            raise ValueError("joint student must append four decision channels")
        path, _ = deployment_path_scores(
            values[:, :, : len(contract["tokens"])],
            contract,
            algorithm=decoder_algorithm,
        )
        decision_frame = values[:, :, -4]
        decision_aux = values[:, :, -3:]
        decision_frame = decision_frame + 0.5 * (
            decision_aux[:, :, 0]
            - tf.reduce_max(decision_aux[:, :, 1:], axis=-1)
        )
        decision = rolling_mean_scores(decision_frame)
        scores.extend((decision + float(path_weight) * path).numpy().tolist())
    return np.asarray(scores, dtype=np.float64)


def initialize_joint_from_decision_model(model, decision_weights: Path):
    """Copy E4's encoder and four-channel deployed decision head."""
    flags = student_flags_for_architecture("dilated_temporal_memory", 1)
    flags.allow_scalar_output = True
    source_base = build_student(flags, INPUT_SHAPE, None)
    auxiliary = tf.keras.layers.Conv2D(
        3, 1, padding="same", name="phonetic_rejection_logits"
    )(source_base.get_layer("encoder_hidden").output)
    auxiliary = SqueezeFrequency(name="phonetic_rejection_sequence")(auxiliary)
    source = tf.keras.Model(
        source_base.input,
        tf.keras.layers.Concatenate(axis=-1)([source_base.output, auxiliary]),
    )
    source.load_weights(decision_weights)
    source_state_layer = next(
        layer
        for layer in reversed(source_base.layers)
        if layer.weights and layer.name.startswith("state_logits")
    )
    source_layers = [
        layer for layer in source_base.layers if layer.weights and layer is not source_state_layer
    ]
    target_state_layer = next(
        layer
        for layer in reversed(model.layers)
        if layer.weights and layer.name.startswith("state_logits")
    )
    target_layers = [
        layer for layer in model.layers if layer.weights and layer is not target_state_layer
    ]
    if len(source_layers) != len(target_layers):
        raise ValueError("joint decision initialization encoder topology differs")
    for source_layer, target_layer in zip(source_layers, target_layers):
        source_weights = source_layer.get_weights()
        target_weights = target_layer.get_weights()
        if [value.shape for value in source_weights] != [
            value.shape for value in target_weights
        ]:
            raise ValueError("joint decision initialization encoder tensors differ")
        target_layer.set_weights(source_weights)
    source_kernel, source_bias = source_state_layer.get_weights()
    source_aux_kernel, source_aux_bias = source.get_layer(
        "phonetic_rejection_logits"
    ).get_weights()
    target_layer = target_state_layer
    target_kernel, target_bias = target_layer.get_weights()
    target_kernel[:, :, :, -3:] = source_aux_kernel
    target_kernel[:, :, :, -4:-3] = source_kernel
    target_bias[-4:-1] = source_aux_bias
    target_bias[-4:-3] = source_bias
    target_layer.set_weights((target_kernel, target_bias))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--posterior-cache", type=Path, required=True)
    parser.add_argument("--teacher-sequence-cache", type=Path, required=True)
    parser.add_argument("--teacher-representation-cache", type=Path)
    parser.add_argument("--teacher-temporal-representation-cache", type=Path)
    parser.add_argument("--streaming-window-cache", type=Path, required=True)
    parser.add_argument("--teacher-causal-window-cache", type=Path)
    parser.add_argument("--reference-student-causal-window-cache", type=Path)
    parser.add_argument("--expanded-public-negatives", type=Path, required=True)
    parser.add_argument("--teacher-qualification", type=Path, required=True)
    parser.add_argument("--continuous-qualification", type=Path, required=True)
    parser.add_argument("--device-validation-quality-report", type=Path)
    parser.add_argument("--overlay-features", type=Path, required=True)
    parser.add_argument("--overlay-targets", type=Path, required=True)
    parser.add_argument("--overlay-provenance", type=Path, required=True)
    parser.add_argument("--overlay-posterior-cache", type=Path)
    parser.add_argument("--overlay-sequence-cache", type=Path)
    parser.add_argument(
        "--teacher-agreement-gate",
        action="store_true",
        help="Mask sequence KD where the frozen teacher disagrees with ground truth.",
    )
    parser.add_argument(
        "--noise-source", action="append", required=True, help="ID=RaggedMmap directory"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init-weights", type=Path)
    parser.add_argument(
        "--init-decision-weights",
        type=Path,
        help="Seed a joint student from a scalar ranked-decision checkpoint.",
    )
    parser.add_argument(
        "--init-encoder-weights",
        type=Path,
        help="Load matching encoder tensors while allowing an output-head mismatch.",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Train only the output head; intended for staged head initialization.",
    )
    parser.add_argument(
        "--recipe-id", default="kizz_control_compact_ctc_distillation_v7"
    )
    parser.add_argument(
        "--student-architecture",
        choices=(
            "control_mixconv",
            "control_mixconv_small",
            "temporal_residual",
            "dilated_temporal_memory",
            "dilated_temporal_memory_wide",
        ),
        default="control_mixconv",
    )
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hard-weight", type=float, default=0.25)
    parser.add_argument("--teacher-weight", type=float, default=0.5)
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--occupation-weight", type=float, default=0.0)
    parser.add_argument("--occupation-max-delay-frames", type=int, default=0)
    parser.add_argument("--representation-weight", type=float, default=0.0)
    parser.add_argument("--temporal-representation-weight", type=float, default=0.0)
    parser.add_argument("--channel-consistency-weight", type=float, default=0.0)
    parser.add_argument(
        "--paired-path-consistency-weight",
        type=float,
        default=0.0,
        help="Match aligned clean/device canonical fit and collision margin.",
    )
    parser.add_argument(
        "--paired-clean-supervision-weight",
        type=float,
        default=0.0,
        help="Supervise canonical fit and collision margin on aligned clean views.",
    )
    parser.add_argument(
        "--joint-decision-weight",
        type=float,
        default=0.0,
        help="Train four appended causal decision channels with the CTC path.",
    )
    parser.add_argument(
        "--joint-decision-path-weight",
        type=float,
        default=0.0,
        help="Weight the CTC canonical fit when selecting joint-student scores.",
    )
    parser.add_argument("--joint-decision-teacher-weight", type=float, default=0.0)
    parser.add_argument("--joint-decision-negative-frame-weight", type=float, default=0.0)
    parser.add_argument("--ctc-weight", type=float, default=0.25)
    parser.add_argument("--positive-collision-weight", type=float, default=0.25)
    parser.add_argument("--legacy-negative-weight", type=float, default=0.0)
    parser.add_argument("--sequence-teacher-weight", type=float, default=0.25)
    parser.add_argument("--sequence-listwise-weight", type=float, default=0.0)
    parser.add_argument("--sequence-listwise-temperature", type=float, default=1.0)
    parser.add_argument("--ranking-weight", type=float, default=0.75)
    parser.add_argument("--tail-ranking-weight", type=float, default=0.75)
    parser.add_argument("--ranking-margin", type=float, default=0.35)
    parser.add_argument("--negative-collision-weight", type=float, default=0.25)
    parser.add_argument("--negative-collision-margin", type=float, default=0.10)
    parser.add_argument(
        "--decoder-algorithm",
        choices=("max_add_ctc_viterbi", "forward_sum_ctc"),
        default="forward_sum_ctc",
    )
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=24106)
    parser.add_argument("--allow-partial-expanded-public-coverage", action="store_true")
    args = parser.parse_args()
    if sum(bool(value) for value in (args.init_weights, args.init_encoder_weights, args.init_decision_weights)) > 1:
        parser.error("student initialization modes are mutually exclusive")
    if args.steps < 1 or args.eval_every < 1:
        parser.error("--steps and --eval-every must be positive")
    loss_values = (
        args.hard_weight,
        args.teacher_weight,
        args.occupation_weight,
        args.representation_weight,
        args.temporal_representation_weight,
        args.channel_consistency_weight,
        args.paired_path_consistency_weight,
        args.paired_clean_supervision_weight,
        args.joint_decision_weight,
        args.joint_decision_path_weight,
        args.joint_decision_teacher_weight,
        args.joint_decision_negative_frame_weight,
        args.ctc_weight,
        args.positive_collision_weight,
        args.legacy_negative_weight,
        args.sequence_teacher_weight,
        args.sequence_listwise_weight,
        args.ranking_weight,
        args.tail_ranking_weight,
        args.ranking_margin,
        args.negative_collision_weight,
        args.negative_collision_margin,
    )
    if any(not np.isfinite(value) or value < 0 for value in loss_values):
        parser.error("loss weights and margins must be finite and non-negative")
    if not np.isfinite(args.teacher_temperature) or args.teacher_temperature <= 0:
        parser.error("--teacher-temperature must be positive and finite")
    if (
        not np.isfinite(args.sequence_listwise_temperature)
        or args.sequence_listwise_temperature <= 0
    ):
        parser.error("--sequence-listwise-temperature must be positive and finite")
    if args.sequence_listwise_weight and not args.teacher_causal_window_cache:
        parser.error(
            "--sequence-listwise-weight requires --teacher-causal-window-cache"
        )
    if (
        args.reference_student_causal_window_cache
        and not args.teacher_causal_window_cache
    ):
        parser.error(
            "--reference-student-causal-window-cache requires teacher causal targets"
        )
    try:
        validate_causal_loss_contract(
            teacher_causal_window_cache=args.teacher_causal_window_cache,
            ranking_weight=args.ranking_weight,
            tail_ranking_weight=args.tail_ranking_weight,
        )
    except ValueError as error:
        parser.error(str(error))
    if not 0 <= args.occupation_max_delay_frames < OUTPUT_FRAMES:
        parser.error("--occupation-max-delay-frames is outside student output")
    clip, continuous = require_teacher_gates(
        args.teacher_qualification, args.continuous_qualification
    )
    corpus = json.loads((args.corpus / "corpus.json").read_text())
    rows = corpus["examples"]
    contract = compact_phone_contract()
    if corpus.get("compact_phone_contract") != contract:
        raise ValueError("distillation corpus compact vocabulary differs")
    cache_meta, cache_arrays = load_cache(
        args.posterior_cache,
        expected_model_revision=clip["model"]["revision"],
        expected_weights_sha256=clip["model"]["weights_sha256"],
    )
    if cache_meta.get("vocabulary", {}).get("tokens") != contract["tokens"]:
        raise ValueError("teacher cache compact vocabulary differs from student")
    require_cache_binding(
        cache_meta,
        args.teacher_qualification,
        corpus["manifests"]["teacher"]["sha256"],
    )
    features = np.load(args.corpus / "features.npy", mmap_mode="r")
    hard_targets = np.load(args.corpus / "hard_targets.npy", mmap_mode="r")
    labels = np.load(args.corpus / "labels.npy", mmap_mode="r")
    if not (len(rows) == len(features) == len(hard_targets) == len(labels)):
        raise ValueError("distillation corpus metadata and tensor lengths differ")
    output_count = len(contract["tokens"]) + (4 if args.joint_decision_weight else 0)
    architecture_flags = student_flags_for_architecture(
        args.student_architecture, output_count
    )
    student_times = student_output_times_seconds(architecture_flags, OUTPUT_FRAMES)
    teacher_targets = np.empty(
        (len(rows), OUTPUT_FRAMES, len(contract["tokens"])), dtype=np.float32
    )
    teacher_occupation_targets = np.zeros_like(teacher_targets)
    matrix, offsets = cache_arrays["log_posteriors"], cache_arrays["offsets"]
    timing = cache_meta["timing"]
    for index in range(len(rows)):
        teacher_frames = matrix[offsets[index] : offsets[index + 1]]
        teacher_targets[index] = resample_log_posteriors(
            teacher_frames,
            teacher_frame_center_seconds=float(timing["frame_center_seconds"]),
            teacher_frame_stride_seconds=float(timing["frame_stride_seconds"]),
            student_times_seconds=student_times,
        )
        if int(rows[index]["label"]) == 1:
            occupation = ctc_state_occupation_log_probs(
                teacher_frames,
                contract["canonical_path"],
                int(contract["blank_id"]),
            )
            teacher_occupation_targets[index] = resample_log_posteriors(
                occupation,
                teacher_frame_center_seconds=float(timing["frame_center_seconds"]),
                teacher_frame_stride_seconds=float(timing["frame_stride_seconds"]),
                student_times_seconds=student_times,
            )
    sequence_cache_meta, sequence_cache = load_teacher_sequence_cache(
        args.teacher_sequence_cache,
        corpus_json=args.corpus / "corpus.json",
        posterior_cache_prefix=args.posterior_cache,
        contract=contract,
    )
    if int(sequence_cache_meta.get("counts", {}).get("examples", -1)) != len(rows):
        raise ValueError("teacher sequence cache corpus length differs")
    teacher_sequence_targets = np.stack(
        (
            sequence_cache["decision_score"],
            sequence_cache["raw_collision_margin"],
        ),
        axis=1,
    ).astype(np.float32)
    qualified_teacher_threshold = float(clip["validation_operating_point"]["threshold"])
    teacher_binary_decisions = (
        teacher_sequence_targets[:, 0] >= qualified_teacher_threshold
    )
    teacher_sequence_supervision_mask = (
        (teacher_binary_decisions == (np.asarray(labels) == 1)).astype(np.float32)
        if args.teacher_agreement_gate
        else np.ones(len(rows), dtype=np.float32)
    )
    representation_metadata = None
    if args.teacher_representation_cache:
        representation_metadata, cached_representations = load_representation_cache(
            args.teacher_representation_cache
        )
        if (
            representation_metadata.get("manifest_sha256")
            != corpus["manifests"]["teacher"]["sha256"]
        ):
            raise ValueError("teacher representation cache is for a different corpus")
        if representation_metadata.get("teacher_qualification", {}).get(
            "sha256"
        ) != sha256_file(args.teacher_qualification):
            raise ValueError("teacher representation cache is for a different teacher")
        teacher_representations = np.asarray(cached_representations, dtype=np.float32)
        if teacher_representations.shape != (len(rows), 96):
            raise ValueError(
                "teacher representation cache must match the 96-channel student hidden state"
            )
    else:
        teacher_representations = np.zeros((len(rows), 96), dtype=np.float32)
    if args.representation_weight and representation_metadata is None:
        parser.error("--representation-weight requires --teacher-representation-cache")
    temporal_representation_metadata = None
    if args.teacher_temporal_representation_cache:
        (
            temporal_representation_metadata,
            cached_temporal_representations,
        ) = load_temporal_representation_cache(
            args.teacher_temporal_representation_cache
        )
        if (
            temporal_representation_metadata.get("manifest_sha256")
            != corpus["manifests"]["teacher"]["sha256"]
            or temporal_representation_metadata.get("teacher_qualification", {}).get(
                "sha256"
            )
            != sha256_file(args.teacher_qualification)
            or temporal_representation_metadata.get("student_architecture")
            != args.student_architecture
        ):
            raise ValueError("teacher temporal representation cache binding differs")
        teacher_temporal_representations = np.asarray(
            cached_temporal_representations, dtype=np.float32
        )
    else:
        teacher_temporal_representations = np.zeros(
            (len(rows), OUTPUT_FRAMES, 96), dtype=np.float32
        )
    if args.temporal_representation_weight and temporal_representation_metadata is None:
        parser.error(
            "--temporal-representation-weight requires --teacher-temporal-representation-cache"
        )
    source_manifest_path = Path(corpus["manifests"]["source"]["path"])
    collision_path_indexes, collision_supervision = collision_path_supervision(
        rows, source_manifest_path, contract
    )
    window_cache_meta, streaming_window_targets = load_student_window_cache(
        args.streaming_window_cache,
        corpus_json=args.corpus / "corpus.json",
        features_path=args.corpus / "features.npy",
        contract=contract,
    )
    if int(window_cache_meta.get("counts", {}).get("examples", -1)) != len(rows):
        raise ValueError("student window cache corpus length differs")
    teacher_causal_metadata = None
    teacher_causal_targets = None
    if args.teacher_causal_window_cache:
        teacher_causal_metadata, teacher_causal_targets = load_causal_decision_cache(
            args.teacher_causal_window_cache,
            representation="qualified_teacher_causal_student_endpoint_decisions",
            corpus_json=args.corpus / "corpus.json",
            contract=contract,
            expected_examples=len(rows),
        )
        if teacher_causal_metadata.get("student_timeline", {}).get(
            "architecture"
        ) != student_architecture_contract(contract, args.student_architecture):
            raise ValueError("teacher causal cache uses a different student timeline")
        posterior = teacher_causal_metadata.get("posterior_cache", {})
        posterior_prefix = args.posterior_cache.with_suffix("")
        if posterior.get("json_sha256") != sha256_file(
            posterior_prefix.with_suffix(".json")
        ) or posterior.get("npz_sha256") != sha256_file(
            posterior_prefix.with_suffix(".npz")
        ):
            raise ValueError("teacher causal cache uses different posteriors")
    reference_causal_metadata = None
    reference_student_causal_targets = None
    if args.reference_student_causal_window_cache:
        (
            reference_causal_metadata,
            reference_student_causal_targets,
        ) = load_causal_decision_cache(
            args.reference_student_causal_window_cache,
            representation="frozen_student_causal_endpoint_decisions",
            corpus_json=args.corpus / "corpus.json",
            contract=contract,
            expected_examples=len(rows),
        )
        validate_reference_causal_contract(
            reference_causal_metadata,
            architecture=student_architecture_contract(
                contract, args.student_architecture
            ),
            features_sha256=sha256_file(args.corpus / "features.npy"),
        )
    expanded_meta, expanded_public_negative_features = load_expanded_public_negatives(
        args.expanded_public_negatives,
        source_manifest=Path(corpus["manifests"]["source"]["path"]),
        continuous_lock=Path(corpus["manifests"]["continuous_lock"]["path"]),
    )
    overlay_features = np.load(args.overlay_features, mmap_mode="r")
    raw_overlay_targets = np.load(args.overlay_targets, mmap_mode="r")
    overlay_provenance = json.loads(args.overlay_provenance.read_text())
    mapped_overlay_targets = map_ordered_targets(
        raw_overlay_targets, overlay_provenance, contract
    )
    overlay_parent_binding = require_overlay_parent_binding(rows, overlay_provenance)
    train_ledger = [
        row for row in overlay_provenance["examples"] if row["split"] == "train"
    ]
    if len(train_ledger) != len(overlay_features) or len(train_ledger) != len(
        raw_overlay_targets
    ):
        raise ValueError("overlay feature order differs from provenance")
    overlay_indexes = [
        i for i, row in enumerate(train_ledger) if row.get("variant") != "clean"
    ]
    overlay_teacher_targets = None
    overlay_teacher_occupation_targets = None
    overlay_teacher_sequence_targets = None
    overlay_teacher_sequence_supervision_mask = None
    overlay_posterior_metadata = None
    if args.overlay_posterior_cache:
        overlay_posterior_metadata, overlay_posterior_arrays = load_cache(
            args.overlay_posterior_cache,
            expected_model_revision=clip["model"]["revision"],
            expected_weights_sha256=clip["model"]["weights_sha256"],
        )
        if (
            overlay_posterior_metadata.get("manifest_sha256")
            != sha256_file(args.overlay_provenance)
            or overlay_posterior_metadata.get("vocabulary", {}).get("tokens")
            != contract["tokens"]
        ):
            raise ValueError("overlay teacher cache binding differs")
        cache_examples = overlay_posterior_metadata.get("examples", [])
        cache_by_source_id = {
            str(row.get("source_id")): index for index, row in enumerate(cache_examples)
        }
        if len(cache_by_source_id) != len(cache_examples):
            raise ValueError("overlay teacher cache has duplicate source IDs")
        overlay_teacher_targets = np.empty(
            (len(overlay_indexes), OUTPUT_FRAMES, len(contract["tokens"])),
            dtype=np.float32,
        )
        overlay_teacher_occupation_targets = np.empty_like(overlay_teacher_targets)
        overlay_matrix = overlay_posterior_arrays["log_posteriors"]
        overlay_offsets = overlay_posterior_arrays["offsets"]
        overlay_timing = overlay_posterior_metadata["timing"]
        for output_index, ledger_index in enumerate(overlay_indexes):
            row = train_ledger[ledger_index]
            cache_index = cache_by_source_id.get(str(row.get("source_id")))
            if cache_index is None:
                raise ValueError("overlay teacher cache is missing an active view")
            teacher_frames = overlay_matrix[
                overlay_offsets[cache_index] : overlay_offsets[cache_index + 1]
            ]
            overlay_teacher_targets[output_index] = resample_log_posteriors(
                teacher_frames,
                teacher_frame_center_seconds=float(
                    overlay_timing["frame_center_seconds"]
                ),
                teacher_frame_stride_seconds=float(
                    overlay_timing["frame_stride_seconds"]
                ),
                student_times_seconds=student_times,
            )
            occupation_frames = ctc_state_occupation_log_probs(
                teacher_frames,
                contract["canonical_path"],
                int(contract["blank_id"]),
            )
            overlay_teacher_occupation_targets[output_index] = resample_log_posteriors(
                occupation_frames,
                teacher_frame_center_seconds=float(
                    overlay_timing["frame_center_seconds"]
                ),
                teacher_frame_stride_seconds=float(
                    overlay_timing["frame_stride_seconds"]
                ),
                student_times_seconds=student_times,
            )
    overlay_sequence_metadata = None
    if args.overlay_sequence_cache:
        prefix = args.overlay_sequence_cache.with_suffix("")
        overlay_sequence_metadata = json.loads(prefix.with_suffix(".json").read_text())
        if (
            overlay_sequence_metadata.get("schema_version") != 1
            or overlay_sequence_metadata.get("representation")
            != "qualified_teacher_original_resolution_clip_decisions"
            or overlay_sequence_metadata.get("manifest", {}).get("sha256")
            != sha256_file(args.overlay_provenance)
        ):
            raise ValueError("overlay sequence cache binding differs")
        if not args.overlay_posterior_cache:
            parser.error("--overlay-sequence-cache requires --overlay-posterior-cache")
        posterior_ref = overlay_sequence_metadata.get("posterior_cache", {})
        posterior_prefix = args.overlay_posterior_cache.with_suffix("")
        if posterior_ref.get("json_sha256") != sha256_file(
            posterior_prefix.with_suffix(".json")
        ) or posterior_ref.get("npz_sha256") != sha256_file(
            posterior_prefix.with_suffix(".npz")
        ):
            raise ValueError("overlay sequence cache uses different posteriors")
        with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as loaded:
            all_overlay_sequence = np.stack(
                (loaded["decision_score"], loaded["raw_collision_margin"]), axis=1
            ).astype(np.float32)
        cache_source_ids = [
            str(row.get("source_id"))
            for row in overlay_posterior_metadata.get("examples", [])
        ]
        sequence_by_source_id = {
            source_id: all_overlay_sequence[index]
            for index, source_id in enumerate(cache_source_ids)
        }
        overlay_teacher_sequence_targets = np.stack(
            [
                sequence_by_source_id[str(train_ledger[index].get("source_id"))]
                for index in overlay_indexes
            ]
        )
        overlay_teacher_sequence_supervision_mask = (
            (
                overlay_teacher_sequence_targets[:, 0] >= qualified_teacher_threshold
            ).astype(np.float32)
            if args.teacher_agreement_gate
            else np.ones(len(overlay_teacher_sequence_targets), dtype=np.float32)
        )
    overlay_features = np.asarray(overlay_features[overlay_indexes])
    mapped_overlay_targets = mapped_overlay_targets[overlay_indexes]
    overlay_providers = [str(train_ledger[i]["provider"]) for i in overlay_indexes]
    device_parent_features = (
        device_parent_feature_pairs(corpus, rows)
        if (
            args.channel_consistency_weight
            or args.paired_path_consistency_weight
            or args.paired_clean_supervision_weight
        )
        else {}
    )
    noise_sources = []
    noise_names = set()
    for value in args.noise_source:
        name, separator, path = value.partition("=")
        if not separator or not name or not Path(path).is_dir():
            parser.error("--noise-source must be ID=existing-directory")
        if name in noise_names:
            parser.error("--noise-source IDs must be unique")
        noise_names.add(name)
        noise_sources.append((name, Path(path)))
    noise_sources.sort(key=lambda item: item[0])
    batcher = DistillationBatcher(
        features,
        hard_targets,
        teacher_targets,
        teacher_occupation_targets,
        teacher_representations,
        teacher_temporal_representations,
        teacher_sequence_targets,
        teacher_sequence_supervision_mask,
        collision_path_indexes,
        streaming_window_targets,
        teacher_causal_targets,
        reference_student_causal_targets,
        expanded_public_negative_features,
        rows,
        overlay_features,
        mapped_overlay_targets,
        overlay_providers,
        overlay_teacher_targets,
        overlay_teacher_occupation_targets,
        overlay_teacher_sequence_targets,
        overlay_teacher_sequence_supervision_mask,
        device_parent_features,
        noise_sources,
        batch_size=args.batch_size,
        seed=args.seed,
        blank_id=int(contract["blank_id"]),
    )
    tf.keras.utils.set_random_seed(args.seed)
    model = build_student(architecture_flags, INPUT_SHAPE, None)
    if args.init_weights:
        model.load_weights(args.init_weights)
    elif args.init_decision_weights:
        if output_count != len(contract["tokens"]) + 4:
            parser.error("--init-decision-weights requires --joint-decision-weight")
        initialize_joint_from_decision_model(model, args.init_decision_weights)
    elif args.init_encoder_weights:
        model.load_weights(args.init_encoder_weights, skip_mismatch=True)
    if args.freeze_encoder:
        for layer in model.layers:
            layer.trainable = layer.name == "state_logits"
    pooled_hidden = tf.keras.layers.GlobalAveragePooling2D(name="representation_pool")(
        model.get_layer("encoder_hidden").output
    )
    projected_hidden = (
        tf.keras.layers.Dense(
            96, use_bias=True, name="training_only_representation_projection"
        )(pooled_hidden)
        if args.representation_weight
        else pooled_hidden
    )
    temporal_hidden = tf.keras.layers.Reshape(
        (OUTPUT_FRAMES, int(model.get_layer("encoder_hidden").output.shape[-1])),
        name="training_only_temporal_squeeze",
    )(model.get_layer("encoder_hidden").output)
    if args.temporal_representation_weight and temporal_hidden.shape[-1] != 96:
        temporal_hidden = tf.keras.layers.Dense(
            96, use_bias=True, name="training_only_temporal_projection"
        )(temporal_hidden)
    training_model = tf.keras.Model(
        model.input,
        [model.output, projected_hidden, temporal_hidden],
        name="phoneme_student_distillation_training",
    )
    optimizer = tf.keras.optimizers.Adam(args.learning_rate)

    @tf.function
    def train_batch(
        x,
        hard,
        soft,
        soft_mask,
        occupation,
        occupation_mask,
        representation,
        representation_mask,
        temporal_representation,
        temporal_representation_mask,
        label,
        teacher_sequence,
        sequence_mask,
        collision_negative_mask,
        named_collision_path,
        scoring_endpoints,
        paired_clean,
        paired_clean_mask,
    ):
        with tf.GradientTape() as tape:
            logits, student_hidden, student_temporal = training_model(x, training=True)
            ctc_logits = logits[:, :, : len(contract["tokens"])]
            loss, parts = distillation_loss(
                ctc_logits,
                hard,
                soft,
                soft_mask,
                label,
                contract,
                teacher_occupation_targets=occupation,
                teacher_occupation_mask=occupation_mask,
                teacher_temperature=args.teacher_temperature,
                occupation_weight=args.occupation_weight,
                occupation_max_delay_frames=args.occupation_max_delay_frames,
                student_hidden=(student_hidden if args.representation_weight else None),
                teacher_representations=representation,
                teacher_representation_mask=representation_mask,
                representation_weight=args.representation_weight,
                scoring_endpoints=scoring_endpoints,
                hard_weight=args.hard_weight,
                teacher_weight=args.teacher_weight,
                ctc_weight=args.ctc_weight,
                collision_weight=args.positive_collision_weight,
                negative_weight=args.legacy_negative_weight,
                negative_score_target=float(
                    clip["validation_operating_point"]["threshold"]
                ),
                teacher_sequence_targets=teacher_sequence,
                sequence_teacher_mask=sequence_mask,
                collision_negative_mask=collision_negative_mask,
                named_collision_path=named_collision_path,
                decoder_algorithm=args.decoder_algorithm,
                sequence_teacher_weight=args.sequence_teacher_weight,
                sequence_listwise_weight=args.sequence_listwise_weight,
                sequence_listwise_temperature=args.sequence_listwise_temperature,
                ranking_weight=args.ranking_weight,
                tail_ranking_weight=args.tail_ranking_weight,
                ranking_margin=args.ranking_margin,
                negative_collision_weight=args.negative_collision_weight,
                negative_collision_margin=args.negative_collision_margin,
            )
            if args.temporal_representation_weight:
                temporal_loss = temporal_representation_loss(
                    student_temporal,
                    temporal_representation,
                    temporal_representation_mask,
                )
                loss = loss + args.temporal_representation_weight * temporal_loss
                parts = (*parts, temporal_loss)
            else:
                parts = (*parts, tf.cast(0.0, loss.dtype))
            if args.joint_decision_weight:
                decision_frame = logits[:, :, -4]
                decision_aux = logits[:, :, -3:]
                decision_frame = decision_frame + 0.5 * (
                    decision_aux[:, :, 0]
                    - tf.reduce_max(decision_aux[:, :, 1:], axis=-1)
                )
                decision_score = rolling_mean_scores(decision_frame)
                decision_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=label, logits=decision_score
                    )
                )
                loss = loss + args.joint_decision_weight * decision_loss
                parts = (*parts, decision_loss)
                if args.joint_decision_teacher_weight:
                    decision_teacher_loss = teacher_sequence_ranking_loss(
                        decision_score,
                        teacher_sequence[:, 0],
                        sequence_mask,
                    )
                    loss = (
                        loss
                        + args.joint_decision_teacher_weight
                        * decision_teacher_loss
                    )
                    parts = (*parts, decision_teacher_loss)
                else:
                    parts = (*parts, tf.cast(0.0, loss.dtype))
                if args.joint_decision_negative_frame_weight:
                    negative_frames = tf.boolean_mask(
                        logits[:, :, -1], label < 0.5
                    )
                    decision_negative_loss = tf.reduce_mean(
                        tf.nn.softplus(negative_frames)
                    )
                    loss = (
                        loss
                        + args.joint_decision_negative_frame_weight
                        * decision_negative_loss
                    )
                    parts = (*parts, decision_negative_loss)
                else:
                    parts = (*parts, tf.cast(0.0, loss.dtype))
            else:
                parts = (*parts, tf.cast(0.0, loss.dtype))
                parts = (*parts, tf.cast(0.0, loss.dtype))
                parts = (*parts, tf.cast(0.0, loss.dtype))
            paired_logits = None
            if (
                args.channel_consistency_weight
                or args.paired_path_consistency_weight
                or args.paired_clean_supervision_weight
            ):
                paired_logits, _, _ = training_model(paired_clean, training=True)
            if args.channel_consistency_weight:
                channel_loss = channel_consistency_loss(
                    logits, paired_logits, paired_clean_mask
                )
                loss = loss + args.channel_consistency_weight * channel_loss
                parts = (*parts, channel_loss)
            else:
                parts = (*parts, tf.cast(0.0, loss.dtype))
            if args.paired_clean_supervision_weight:
                clean_supervision_loss = paired_clean_path_supervision_loss(
                    paired_logits,
                    scoring_endpoints,
                    paired_clean_mask,
                    contract,
                    algorithm=args.decoder_algorithm,
                )
                loss = (
                    loss
                    + args.paired_clean_supervision_weight
                    * clean_supervision_loss
                )
                parts = (*parts, clean_supervision_loss)
            else:
                parts = (*parts, tf.cast(0.0, loss.dtype))
            if args.paired_path_consistency_weight:
                path_loss = paired_path_consistency_loss(
                    logits,
                    paired_logits,
                    scoring_endpoints,
                    paired_clean_mask,
                    contract,
                    algorithm=args.decoder_algorithm,
                )
                loss = loss + args.paired_path_consistency_weight * path_loss
                parts = (*parts, path_loss)
            else:
                parts = (*parts, tf.cast(0.0, loss.dtype))
        gradients = tape.gradient(loss, training_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, training_model.trainable_variables))
        return loss, parts

    validation_positive = np.asarray(
        features[
            [
                i
                for i, row in enumerate(rows)
                if row["split"] == "validation" and row["label"] == 1
            ]
        ],
        dtype=np.float32,
    )
    validation_negative_indexes = [
        i
        for i, row in enumerate(rows)
        if row["split"] == "validation" and row["label"] == 0
    ]
    validation_negative = np.asarray(
        features[validation_negative_indexes], dtype=np.float32
    )
    validation_seconds = sum(
        float(rows[i]["duration_seconds"]) for i in validation_negative_indexes
    )
    device_validation = None
    device_validation_rows = []
    device_validation_required = 0
    if args.device_validation_quality_report:
        device_validation, device_validation_rows = device_validation_features(
            args.device_validation_quality_report
        )
        device_validation_required = math.ceil(0.90 * len(device_validation_rows))
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_directory = args.output / "checkpoints"
    checkpoint_directory.mkdir(exist_ok=True)
    ledger = []
    finite_floor = -np.finfo(np.float64).max
    best_key = (
        (-1.0, -1.0, -1.0, -1.0, -1.0, finite_floor, -1.0, finite_floor)
        if device_validation is not None
        else (-1.0, -1.0, finite_floor, -1.0, finite_floor)
    )
    best_step = None
    for step in range(args.steps):
        batch = batcher.batch(step)
        loss, parts = train_batch(*(tf.convert_to_tensor(value) for value in batch))
        if (step + 1) % args.eval_every == 0 or step == 0:
            score_function = (
                _joint_student_scores if args.joint_decision_weight else _student_scores
            )
            score_kwargs = (
                {
                    "path_weight": args.joint_decision_path_weight,
                }
                if args.joint_decision_weight
                else {}
            )
            positive_scores = score_function(
                model,
                validation_positive,
                contract,
                args.batch_size,
                decoder_algorithm=args.decoder_algorithm,
                **score_kwargs,
            )
            negative_scores = score_function(
                model,
                validation_negative,
                contract,
                args.batch_size,
                decoder_algorithm=args.decoder_algorithm,
                **score_kwargs,
            )
            point = choose_validation_threshold(
                positive_scores,
                negative_scores,
                negative_exposure_seconds=validation_seconds,
                min_recall=0.90,
                max_faph=0.10,
            )
            zero_fp_recall = float(np.mean(positive_scores > np.max(negative_scores)))
            positive_floor = float(np.min(positive_scores))
            negative_ceiling = float(np.max(negative_scores))
            separation = (
                positive_floor - negative_ceiling
                if np.isfinite(positive_floor) and np.isfinite(negative_ceiling)
                else None
            )
            device_report = None
            if device_validation is not None:
                device_scores = score_function(
                    model,
                    device_validation,
                    contract,
                    args.batch_size,
                    decoder_algorithm=args.decoder_algorithm,
                    **score_kwargs,
                )
                threshold = point.get("threshold")
                device_accepted = (
                    int(np.sum(device_scores > float(threshold)))
                    if threshold is not None
                    else 0
                )
                device_zero_fp_accepted = int(np.sum(device_scores > negative_ceiling))
                device_report = {
                    "accepted_at_clean_operating_point": device_accepted,
                    "required_at_clean_operating_point": device_validation_required,
                    "recall_at_clean_operating_point": device_accepted
                    / len(device_scores),
                    "zero_false_accept_accepted": device_zero_fp_accepted,
                    "zero_false_accept_recall": device_zero_fp_accepted
                    / len(device_scores),
                    "total": len(device_scores),
                }
            item = {
                "step": step + 1,
                "loss": float(loss),
                "parts": [float(value) for value in parts],
                "operating_point": point,
                "zero_false_accept_recall": zero_fp_recall,
                "separation": separation,
                "device_validation": device_report,
            }
            step_checkpoint = checkpoint_directory / f"step-{step + 1:04d}.weights.h5"
            model.save_weights(step_checkpoint)
            item["checkpoint"] = {
                "path": str(step_checkpoint.resolve()),
                "sha256": sha256_file(step_checkpoint),
            }
            ledger.append(item)
            key = (
                multichannel_checkpoint_selection_key(
                    point,
                    zero_fp_recall,
                    float(device_report["zero_false_accept_recall"]),
                    int(device_report["accepted_at_clean_operating_point"]),
                    device_validation_required,
                    separation,
                )
                if device_report is not None
                else checkpoint_selection_key(point, zero_fp_recall, separation)
            )
            if key > best_key:
                best_key = key
                best_step = step + 1
                model.save_weights(args.output / "best.weights.h5")
            print(json.dumps(item), flush=True)
    model.save_weights(args.output / "last.weights.h5")
    if best_step is None:
        raise RuntimeError("distillation produced no best checkpoint")
    checkpoint = checkpoint_binding(args.output, best_step, best_key)
    decoder = student_decoder_contract(contract, args.decoder_algorithm)
    (
        expected_provider_counts,
        expected_negative_counts,
        expected_positive_variant_provider_counts,
    ) = expected_schedule_counts(args.steps, args.batch_size)
    if not args.allow_partial_expanded_public_coverage and expected_negative_counts[
        "public_speech"
    ] < len(expanded_public_negative_features):
        raise ValueError("configured run cannot cover every expanded public negative")
    if batcher.provider_counts != expected_provider_counts:
        raise ValueError(
            f"realized positive provider schedule drifted: {batcher.provider_counts}"
        )
    if batcher.negative_group_counts != expected_negative_counts:
        raise ValueError(
            f"realized negative group schedule drifted: {batcher.negative_group_counts}"
        )
    if (
        batcher.expanded_public_negative_count
        != expected_negative_counts["public_speech"]
    ):
        raise ValueError("expanded public-negative schedule drifted")
    if (
        batcher.positive_variant_provider_counts
        != expected_positive_variant_provider_counts
    ):
        raise ValueError(
            "realized positive variant/provider schedule drifted: "
            f"{batcher.positive_variant_provider_counts}"
        )
    provider_total = sum(batcher.provider_counts.values())
    provider_shares = {
        key: value / provider_total
        for key, value in sorted(batcher.provider_counts.items())
    }
    if set(provider_shares) != set(APPROVED_PROVIDERS) or any(
        not 0.24 <= share <= 0.26 for share in provider_shares.values()
    ):
        raise ValueError(
            f"realized positive provider balance failed: {provider_shares}"
        )
    architecture_metadata = student_architecture_contract(
        contract, args.student_architecture
    )
    architecture_metadata["output_count"] = output_count
    metadata = {
        "schema_version": 1,
        "recipe": args.recipe_id,
        "teacher_qualification": {
            "path": str(args.teacher_qualification.resolve()),
            "sha256": sha256_file(args.teacher_qualification),
        },
        "continuous_qualification": {
            "path": str(args.continuous_qualification.resolve()),
            "sha256": sha256_file(args.continuous_qualification),
        },
        "device_validation_quality": (
            {
                "path": str(args.device_validation_quality_report.resolve()),
                "sha256": sha256_file(args.device_validation_quality_report),
                "accepted_rows": len(device_validation_rows),
                "required_at_clean_operating_point": device_validation_required,
                "selection_role": "checkpoint_selection_only",
                "training_eligible": False,
            }
            if args.device_validation_quality_report
            else None
        ),
        "teacher_model": clip["model"],
        "initialization": (
            provenance_ref(args.init_weights)
            if args.init_weights
            else (
                provenance_ref(args.init_decision_weights)
                if args.init_decision_weights
                else (
                    {
                        "mode": "matching_encoder_tensors_skip_output_head",
                        **provenance_ref(args.init_encoder_weights),
                    }
                    if args.init_encoder_weights
                    else None
                )
            )
        ),
        "encoder_trainable": not args.freeze_encoder,
        "posterior_cache": {
            "prefix": str(args.posterior_cache.resolve()),
            "cache_sha256": cache_meta["cache_sha256"],
            "json": provenance_ref(args.posterior_cache.with_suffix(".json")),
            "npz": provenance_ref(args.posterior_cache.with_suffix(".npz")),
        },
        "teacher_representation_cache": (
            {
                "prefix": str(args.teacher_representation_cache.resolve()),
                "json": provenance_ref(
                    args.teacher_representation_cache.with_suffix(".json")
                ),
                "npy": provenance_ref(
                    args.teacher_representation_cache.with_suffix(".npy")
                ),
                "contract": representation_metadata,
            }
            if args.teacher_representation_cache
            else None
        ),
        "teacher_temporal_representation_cache": (
            {
                "prefix": str(args.teacher_temporal_representation_cache.resolve()),
                "json": provenance_ref(
                    args.teacher_temporal_representation_cache.with_suffix(".json")
                ),
                "npy": provenance_ref(
                    args.teacher_temporal_representation_cache.with_suffix(".npy")
                ),
                "contract": temporal_representation_metadata,
            }
            if args.teacher_temporal_representation_cache
            else None
        ),
        "teacher_sequence_cache": {
            "prefix": str(args.teacher_sequence_cache.resolve()),
            "json": provenance_ref(args.teacher_sequence_cache.with_suffix(".json")),
            "npz": provenance_ref(args.teacher_sequence_cache.with_suffix(".npz")),
            "scorer": sequence_cache_meta["scorer"],
            "split_reports": sequence_cache_meta["split_reports"],
        },
        "streaming_window_cache": {
            "prefix": str(args.streaming_window_cache.resolve()),
            "json": provenance_ref(args.streaming_window_cache.with_suffix(".json")),
            "npz": provenance_ref(args.streaming_window_cache.with_suffix(".npz")),
            "source_student": window_cache_meta["source_student"],
            "scorer": window_cache_meta["scorer"],
            "split_reports": window_cache_meta["split_reports"],
        },
        "teacher_causal_window_cache": (
            {
                "prefix": str(args.teacher_causal_window_cache.resolve()),
                "json": provenance_ref(
                    args.teacher_causal_window_cache.with_suffix(".json")
                ),
                "npz": provenance_ref(
                    args.teacher_causal_window_cache.with_suffix(".npz")
                ),
                "contract": teacher_causal_metadata,
            }
            if args.teacher_causal_window_cache
            else None
        ),
        "reference_student_causal_window_cache": (
            {
                "prefix": str(args.reference_student_causal_window_cache.resolve()),
                "json": provenance_ref(
                    args.reference_student_causal_window_cache.with_suffix(".json")
                ),
                "npz": provenance_ref(
                    args.reference_student_causal_window_cache.with_suffix(".npz")
                ),
                "contract": reference_causal_metadata,
            }
            if args.reference_student_causal_window_cache
            else None
        ),
        "expanded_public_negatives": {
            "root": provenance_ref(args.expanded_public_negatives),
            "metadata": provenance_ref(
                args.expanded_public_negatives / "metadata.json"
            ),
            "features": expanded_meta["features"],
            "count": expanded_meta["count"],
            "selection": expanded_meta["selection"],
            "realized_samples": batcher.expanded_public_negative_count,
            "complete_inventory_coverage": (
                batcher.expanded_public_negative_count >= expanded_meta["count"]
            ),
        },
        "corpus": {
            "root": provenance_ref(args.corpus),
            "corpus_json": provenance_ref(args.corpus / "corpus.json"),
            "features": provenance_ref(args.corpus / "features.npy"),
            "hard_targets": provenance_ref(args.corpus / "hard_targets.npy"),
            "labels": provenance_ref(args.corpus / "labels.npy"),
        },
        "overlay": {
            "features": provenance_ref(args.overlay_features),
            "targets": provenance_ref(args.overlay_targets),
            "provenance": provenance_ref(args.overlay_provenance),
            "parent_binding": overlay_parent_binding,
            "teacher_posterior_cache": (
                {
                    "prefix": str(args.overlay_posterior_cache.resolve()),
                    "json": provenance_ref(
                        args.overlay_posterior_cache.with_suffix(".json")
                    ),
                    "npz": provenance_ref(
                        args.overlay_posterior_cache.with_suffix(".npz")
                    ),
                    "contract": overlay_posterior_metadata,
                }
                if args.overlay_posterior_cache
                else None
            ),
            "teacher_sequence_cache": (
                {
                    "prefix": str(args.overlay_sequence_cache.resolve()),
                    "json": provenance_ref(
                        args.overlay_sequence_cache.with_suffix(".json")
                    ),
                    "npz": provenance_ref(
                        args.overlay_sequence_cache.with_suffix(".npz")
                    ),
                    "contract": overlay_sequence_metadata,
                }
                if args.overlay_sequence_cache
                else None
            ),
        },
        "noise_sources": [
            {"id": name, "artifact": provenance_ref(path)}
            for name, path in noise_sources
        ],
        "compact_phone_contract": contract,
        "architecture": architecture_metadata,
        "student_output_frames": OUTPUT_FRAMES,
        "decoder": {
            "contract": decoder,
            "contract_sha256": student_decoder_contract_hash(
                contract, args.decoder_algorithm
            ),
        },
        "student_output_times_seconds": student_times.tolist(),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "loss_contract": {
            "hard_weight": args.hard_weight,
            "teacher_weight": args.teacher_weight,
            "teacher_temperature": args.teacher_temperature,
            "sequence_occupation_weight": args.occupation_weight,
            "sequence_occupation_max_delay_frames": args.occupation_max_delay_frames,
            "sequence_occupation_target": "teacher_ctc_forward_backward_conditioned_on_canonical_path",
            "utterance_representation_weight": args.representation_weight,
            "temporal_representation_weight": args.temporal_representation_weight,
            "temporal_representation_target": (
                "qualified_teacher_hidden_sequence_train_pca_aligned_to_student_timeline"
                if args.temporal_representation_weight
                else None
            ),
            "temporal_representation_adapter": (
                "training_only_per_frame_dense_96_discarded_before_runtime"
                if args.temporal_representation_weight
                else None
            ),
            "paired_clean_device_channel_consistency_weight": args.channel_consistency_weight,
            "paired_clean_device_path_consistency_weight": args.paired_path_consistency_weight,
            "paired_clean_path_supervision_weight": args.paired_clean_supervision_weight,
            "paired_clean_path_supervision_target": (
                "canonical_fit_plus_collision_margin"
                if args.paired_clean_supervision_weight
                else None
            ),
            "joint_decision_weight": args.joint_decision_weight,
            "joint_decision_path_weight": args.joint_decision_path_weight,
            "joint_decision_teacher_weight": args.joint_decision_teacher_weight,
            "joint_decision_negative_frame_weight": args.joint_decision_negative_frame_weight,
            "joint_decision_output": (
                "appended_four_channel_causal_decision_head"
                if args.joint_decision_weight
                else None
            ),
            "paired_path_consistency_target": (
                "symmetric_canonical_fit_and_collision_margin"
                if args.paired_path_consistency_weight
                else None
            ),
            "utterance_representation_target": "qualified_teacher_last_hidden_time_mean_train_pca",
            "utterance_representation_adapter": (
                "training_only_global_mean_dense_96_discarded_before_runtime"
                if args.representation_weight
                else None
            ),
            "ctc_weight": args.ctc_weight,
            "positive_collision_weight": args.positive_collision_weight,
            "legacy_negative_weight": args.legacy_negative_weight,
            "legacy_negative_score_target": float(
                clip["validation_operating_point"]["threshold"]
            ),
            "sequence_teacher_weight": args.sequence_teacher_weight,
            "sequence_listwise_weight": args.sequence_listwise_weight,
            "sequence_listwise_temperature": args.sequence_listwise_temperature,
            "ranking_weight": args.ranking_weight,
            "tail_ranking_weight": args.tail_ranking_weight,
            "ranking_margin": args.ranking_margin,
            "negative_collision_weight": args.negative_collision_weight,
            "negative_collision_margin": args.negative_collision_margin,
            "path_scoring": args.decoder_algorithm,
            "path_training_geometry": "mined_streaming_endpoint_suffix_windows_pre_beta",
            "teacher_sequence_targets": [
                "original_resolution_decision_score",
                "raw_collision_margin",
            ],
            "teacher_sequence_transfer": "cross_label_pairwise_rank",
            "teacher_sequence_agreement_guardrail": {
                "enabled": args.teacher_agreement_gate,
                "rule": "transfer_only_when_frozen_teacher_binary_decision_matches_ground_truth",
                "threshold": qualified_teacher_threshold,
                "corpus_supervised": int(teacher_sequence_supervision_mask.sum()),
                "corpus_total": len(teacher_sequence_supervision_mask),
                "overlay_supervised": (
                    int(overlay_teacher_sequence_supervision_mask.sum())
                    if overlay_teacher_sequence_supervision_mask is not None
                    else 0
                ),
                "overlay_total": (
                    len(overlay_teacher_sequence_supervision_mask)
                    if overlay_teacher_sequence_supervision_mask is not None
                    else 0
                ),
            },
            "teacher_causal_window_transfer": (
                "random_disagreement_or_teacher_hard_terminal_endpoints"
                if args.teacher_causal_window_cache
                else None
            ),
            "ranking": "all_pairs_plus_top_quartile_negative_vs_bottom_quartile_positive",
            "collision_negative_rejection": "strict_collision_path_margin",
            "collision_negative_scope": "kizz_control_phonetic_collision",
            "collision_sequence_teacher_transfer": "disabled_for_explicit_collision_rows",
        },
        "collision_supervision": collision_supervision,
        "realized_positive_provider_counts": dict(
            sorted(batcher.provider_counts.items())
        ),
        "realized_positive_provider_shares": provider_shares,
        "realized_positive_variant_provider_counts": nested_variant_provider_counts(
            batcher.positive_variant_provider_counts
        ),
        "realized_negative_group_counts": dict(
            sorted(batcher.negative_group_counts.items())
        ),
        "realized_hard_negative_counts": dict(
            sorted(batcher.hard_negative_counts.items())
        ),
        "realized_hard_positive_counts": dict(
            sorted(batcher.hard_positive_counts.items())
        ),
        "realized_noise_source_counts": dict(
            sorted(batcher.noise_source_counts.items())
        ),
        "sampling_contract": {
            "seed": args.seed,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "positive_provider_order": list(APPROVED_PROVIDERS),
            "positive_variant_order": list(POSITIVE_VARIANTS),
            "positive_pair_schedule": {
                "variant_index": "global_positive_slot mod 3",
                "provider_index": "global_positive_slot mod 4",
            },
            "negative_group_order": [
                "public_speech",
                "kizz_control_phonetic_collision",
                "device_collision",
                "no_speech",
            ],
            "noise_source_order": [name for name, _ in noise_sources],
            "expected_positive_provider_counts": dict(
                sorted(expected_provider_counts.items())
            ),
            "expected_positive_variant_provider_counts": nested_variant_provider_counts(
                expected_positive_variant_provider_counts
            ),
            "expected_negative_group_counts": dict(
                sorted(expected_negative_counts.items())
            ),
        },
        "best_checkpoint_contract": {
            "selection_key": [
                "qualified",
                "zero_false_accept_recall",
                "negative_false_accepts_at_recall_floor",
                "recall",
                "separation",
            ],
            "tie_break": "earliest_evaluation_step",
            "evaluation_interval_steps": args.eval_every,
            "evaluation_geometry": "deployment_suffix_ending_at_latest_frame",
            "score_normalization": "log_softmax_before_ctc_forward_sum",
        },
        "validation_ledger": ledger,
        "best_key": list(best_key),
        "best_step": best_step,
        "student": checkpoint,
    }
    (args.output / "distillation.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
