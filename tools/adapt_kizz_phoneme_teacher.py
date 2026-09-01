#!/usr/bin/env python3
"""Adapt the pinned Kizz IPA/CTC teacher without weakening its base behavior.

This is deliberately a single-model adaptation recipe.  The frozen copy is used
for framewise KL anchoring while only ``lm_head`` and the last N wav2vec2
encoder layers are trainable.  The manifest is an immutable, SHA-bound input;
test/held-out rows are rejected rather than silently ignored.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from microwakeword.kizz_phoneme_teacher import (
    MODEL_ID,
    MODEL_REVISION,
    TARGET_SAMPLE_RATE,
    choose_validation_threshold,
    load_hf_teacher,
    resolve_hf_weights_path,
    resolve_phone_ids,
    sha256_file,
)
from tools.qualify_kizz_phoneme_teacher import (
    _finite_scores,
    _score_row,
)
from microwakeword.wake_phrase import KIZZ_CONTROL, WAKE_PHRASES, get_wake_phrase

APPROVED_PROVIDERS = ("assemblyai", "deepgram", "elevenlabs", "kokoro")
NEGATIVE_GROUPS = (
    "kizz_control_phonetic_collision",
    "device_collision",
    "public_speech",
)
REQUIRED_GROUPS = ("device_channel_positive",) + NEGATIVE_GROUPS


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_hash(value: Any) -> str:
    return _sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("examples", payload.get("rows"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("adaptation manifest must contain a non-empty examples list")
    return [dict(row) for row in rows]


def load_adaptation_manifest(
    path: Path,
    *,
    expected_sha256: str | None = None,
    model_id: str = MODEL_ID,
    revision: str = MODEL_REVISION,
    phrase_id: str = KIZZ_CONTROL.phrase_id,
) -> tuple[dict[str, Any], str]:
    """Load and validate a manifest before any model or audio is opened.

    The caller must provide the expected file hash (or the manifest must carry
    one in ``manifest_sha256``).  The declared teacher identity is mandatory.
    This prevents adapting a different checkpoint or an edited manifest by
    accident.
    """
    path = Path(path).resolve()
    actual = sha256_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("adaptation manifest must be a JSON object")
    declared_hash = payload.get("manifest_sha256") or payload.get("sha256")
    expected = expected_sha256 or declared_hash
    if not expected or expected != actual:
        raise ValueError(f"adaptation manifest SHA-256 mismatch: expected {expected}, got {actual}")
    identity = payload.get("base_teacher") or payload.get("teacher") or payload.get("inputs", {})
    declared_id = identity.get("model_id") or identity.get("base_model_id")
    declared_revision = identity.get("revision") or identity.get("base_revision")
    if declared_id != model_id or declared_revision != revision:
        raise ValueError("manifest is not bound to the pinned Kizz teacher")
    phrase = payload.get("wake_phrase", {})
    expected_phrase = get_wake_phrase(phrase_id)
    expected_collisions = {
        name: list(phones)
        for name, phones in zip(
            expected_phrase.collision_transcripts,
            expected_phrase.collision_phones,
            strict=True,
        )
    }
    if (
        phrase.get("phrase_id") != expected_phrase.phrase_id
        or phrase.get("phones") != list(expected_phrase.phones)
        or phrase.get("collision_paths") != expected_collisions
    ):
        raise ValueError("manifest wake-phrase phone contract differs from requested phrase")
    rows = _rows(payload)
    groups = {str(row.get("source_group")) for row in rows}
    missing = sorted(set(REQUIRED_GROUPS) - groups)
    if missing:
        raise ValueError(f"adaptation manifest missing required groups: {missing}")
    providers = {
        str(row.get("provider"))
        for row in rows
        if int(row.get("label", 0)) == 1
        and row.get("source_group") != "device_channel_positive"
    }
    missing_providers = sorted(set(APPROVED_PROVIDERS) - providers)
    if missing_providers:
        raise ValueError(f"clean positives missing providers: {missing_providers}")
    if {str(row.get("split")) for row in rows} != {"train", "validation"}:
        raise ValueError("adaptation manifest must contain only train and validation rows")
    train_rows = [row for row in rows if row.get("split") == "train"]
    validation_rows = [row for row in rows if row.get("split") == "validation"]
    for split_rows, split_name in ((train_rows, "train"), (validation_rows, "validation")):
        if not any(int(row.get("label", 0)) == 1 for row in split_rows):
            raise ValueError(f"adaptation {split_name} split lacks positives")
        if not any(int(row.get("label", 0)) == 0 for row in split_rows):
            raise ValueError(f"adaptation {split_name} split lacks negatives")
    validation_device_providers = {
        str(row.get("provider"))
        for row in validation_rows
        if int(row.get("label", 0)) == 1
        and row.get("source_group") == "device_channel_positive"
    }
    if missing := sorted(set(APPROVED_PROVIDERS) - validation_device_providers):
        raise ValueError(
            f"adaptation validation split missing device-channel providers: {missing}"
        )
    train_groups = {str(row.get("source_group")) for row in train_rows}
    missing_train_groups = sorted(set(REQUIRED_GROUPS) - train_groups)
    if missing_train_groups:
        raise ValueError(
            f"adaptation train split missing required groups: {missing_train_groups}"
        )
    train_providers = {
        str(row.get("provider"))
        for row in train_rows
        if int(row.get("label", 0)) == 1
        and row.get("source_group") != "device_channel_positive"
    }
    if missing := sorted(set(APPROVED_PROVIDERS) - train_providers):
        raise ValueError(f"adaptation train split missing providers: {missing}")
    return payload, actual


class DeterministicBatchMixture:
    """Exact, seed-independent row schedule with balanced provider positives.

    Every batch contains equal clean-positive and device-channel-positive rows,
    followed by negative rows.  Negative groups rotate in a fixed cycle; clean
    providers rotate in a fixed cycle.  RNG is used only for selecting among
    rows within a bucket, so adding rows cannot change earlier steps.
    """

    def __init__(self, rows: Sequence[Mapping[str, Any]], *, batch_size: int = 4, seed: int = 231):
        if batch_size < 4 or batch_size % 2:
            raise ValueError("batch_size must be an even number >= 4")
        self.rows = [dict(row) for row in rows if row.get("split", "train") == "train"]
        self.batch_size, self.seed = batch_size, int(seed)
        self.clean = {
            provider: [i for i, row in enumerate(self.rows)
                       if int(row.get("label", 0)) == 1
                       and row.get("source_group") != "device_channel_positive"
                       and row.get("provider") == provider]
            for provider in APPROVED_PROVIDERS
        }
        self.device = {
            provider: [
                i
                for i, row in enumerate(self.rows)
                if int(row.get("label", 0)) == 1
                and row.get("source_group") == "device_channel_positive"
                and row.get("provider") == provider
            ]
            for provider in APPROVED_PROVIDERS
        }
        self.negative = {
            group: [i for i, row in enumerate(self.rows)
                    if int(row.get("label", 0)) == 0 and row.get("source_group") == group]
            for group in NEGATIVE_GROUPS
        }
        if any(not value for value in self.clean.values()) or any(
            not value for value in self.device.values()
        ):
            raise ValueError("every provider and device_channel_positive need train rows")
        if any(not value for value in self.negative.values()):
            raise ValueError("every required negative group needs train rows")

    def batch(self, step: int) -> list[dict[str, Any]]:
        rng = np.random.default_rng(self.seed + int(step))
        result: list[dict[str, Any]] = []
        positive_slots = self.batch_size // 2
        for slot in range(positive_slots):
            provider_offset = slot // 2 + (2 if slot % 2 else 0)
            provider = APPROVED_PROVIDERS[
                (step + provider_offset) % len(APPROVED_PROVIDERS)
            ]
            bucket = self.clean[provider] if slot % 2 == 0 else self.device[provider]
            index = int(bucket[int(rng.integers(0, len(bucket)))])
            result.append(dict(self.rows[index]))
        for slot in range(self.batch_size - positive_slots):
            group = NEGATIVE_GROUPS[(step * (self.batch_size - positive_slots) + slot) % len(NEGATIVE_GROUPS)]
            index = int(self.negative[group][int(rng.integers(0, len(self.negative[group])))])
            result.append(dict(self.rows[index]))
        return result


def augment_positive_waveform(
    waveform: np.ndarray,
    rng: np.random.Generator,
    *,
    background: np.ndarray | None = None,
    sample_rate: int = TARGET_SAMPLE_RATE,
    preserve_probability: float = 0.25,
) -> np.ndarray:
    """Apply bounded, deterministic channel variation to a positive only."""
    values = np.asarray(waveform, dtype=np.float32).reshape(-1).copy()
    if not len(values):
        raise ValueError("positive waveform is empty")
    if rng.random() < preserve_probability:
        return np.clip(values, -1.0, 1.0).astype(np.float32)
    gain = float(10.0 ** rng.uniform(-3.0, 3.0) / 20.0)
    values *= gain
    shift = int(rng.integers(-round(sample_rate * 0.025), round(sample_rate * 0.025) + 1))
    if shift > 0:
        values = np.pad(values, (shift, 0))[: len(waveform)]
    elif shift < 0:
        values = np.pad(values[-shift:], (0, -shift))[: len(waveform)]
    # Small causal room impulse; normalization prevents runaway amplitudes.
    taps = np.array([1.0, 0.25, 0.10, 0.04], dtype=np.float32)
    taps *= float(rng.uniform(0.85, 1.0))
    values = np.convolve(values, taps, mode="full")[: len(waveform)]
    if background is not None and len(background):
        noise = np.resize(np.asarray(background, dtype=np.float32), len(values))
        signal_rms = max(float(np.sqrt(np.mean(values * values))), 1e-5)
        noise_rms = max(float(np.sqrt(np.mean(noise * noise))), 1e-5)
        snr_db = float(rng.uniform(12.0, 24.0))
        values += noise * (signal_rms / noise_rms) * (10.0 ** (-snr_db / 20.0))
    return np.clip(values, -1.0, 1.0).astype(np.float32)


def _ctc_fit(logits, paths, lengths, *, blank_id: int):
    import torch
    log_probs = torch.log_softmax(logits, dim=-1)
    fits = []
    for path in paths:
        target = torch.tensor(path, dtype=torch.long, device=logits.device).repeat(logits.shape[0], 1)
        target_lengths = torch.full((logits.shape[0],), len(path), dtype=torch.long, device=logits.device)
        loss = torch.nn.functional.ctc_loss(
            log_probs.transpose(0, 1), target, lengths, target_lengths,
            blank=blank_id, reduction="none", zero_infinity=True,
        )
        fits.append(-loss / max(1, len(path)))
    return torch.stack(fits, dim=1)


def adaptation_loss(
    logits,
    input_lengths,
    labels,
    base_log_probs,
    frame_mask,
    *,
    canonical_path: Sequence[int],
    collision_paths: Mapping[str, Sequence[int]],
    blank_id: int,
    negative_target: float = -4.0,
    collision_margin: float = 0.20,
    positive_weight: float = 1.0,
    negative_weight: float = 1.0,
    collision_weight: float = 1.0,
    kl_weight: float = 0.25,
    collision_mask=None,
) -> tuple[Any, dict[str, float]]:
    """Return loss and detached components; signs are intentionally explicit."""
    import torch
    canonical = _ctc_fit(logits, [canonical_path], input_lengths, blank_id=blank_id)[:, 0]
    positive = labels > 0.5
    negative = ~positive
    pos_loss = -canonical[positive].mean() if positive.any() else logits.sum() * 0
    neg_loss = torch.nn.functional.softplus(canonical[negative] - negative_target).mean() if negative.any() else logits.sum() * 0
    if collision_mask is None:
        collision_mask = torch.zeros(len(labels), dtype=torch.bool, device=logits.device)
    else:
        collision_mask = torch.as_tensor(collision_mask, dtype=torch.bool, device=logits.device)
    if not positive.any() and not collision_mask.any():
        positive_collision_loss = logits.sum() * 0
        negative_collision_loss = logits.sum() * 0
    else:
        paths = list(collision_paths.values())
        best_collision = _ctc_fit(logits, paths, input_lengths, blank_id=blank_id).max(dim=1).values
        positive_collision_loss = (
            torch.relu(
                best_collision[positive] + collision_margin - canonical[positive]
            ).mean()
            if positive.any()
            else logits.sum() * 0
        )
        negative_collision_loss = (
            torch.relu(
                canonical[collision_mask] + collision_margin
                - best_collision[collision_mask]
            ).mean()
            if collision_mask.any()
            else logits.sum() * 0
        )
    collision_loss = positive_collision_loss + negative_collision_loss
    valid = frame_mask.to(logits.dtype)
    student_log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    kl = torch.nn.functional.kl_div(student_log_probs, base_log_probs.exp(), reduction="none").sum(dim=-1)
    kl_loss = (kl * valid).sum() / valid.sum().clamp_min(1.0)
    total = positive_weight * pos_loss + negative_weight * neg_loss + collision_weight * collision_loss + kl_weight * kl_loss
    return total, {
        "positive_ctc": float(pos_loss.detach()),
        "negative_suppression": float(neg_loss.detach()),
        "positive_collision_margin": float(positive_collision_loss.detach()),
        "negative_collision_margin": float(negative_collision_loss.detach()),
        "collision_margin": float(collision_loss.detach()),
        "base_kl": float(kl_loss.detach()),
        "total": float(total.detach()),
    }


def _set_trainable(
    model,
    last_n: int,
    *,
    train_feature_projection: bool = False,
) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if not hasattr(model, "lm_head"):
        raise ValueError("teacher has no lm_head")
    for parameter in model.lm_head.parameters():
        parameter.requires_grad = True
    layers = getattr(getattr(getattr(model, "wav2vec2", None), "encoder", None), "layers", None)
    if layers is None or last_n < 0 or last_n > len(layers):
        raise ValueError("last_n_encoder_layers is outside the teacher encoder")
    for layer in list(layers)[-last_n:] if last_n else []:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    if train_feature_projection:
        projection = getattr(getattr(model, "wav2vec2", None), "feature_projection", None)
        if projection is None:
            raise ValueError("teacher has no wav2vec2 feature projection")
        for parameter in projection.parameters():
            parameter.requires_grad = True


def make_adaptation_models(
    base_model,
    *,
    last_n_encoder_layers: int,
    train_feature_projection: bool = False,
    gradient_checkpointing: bool = False,
):
    """Return (trainable adaptation copy, immutable frozen base copy)."""
    adapted = copy.deepcopy(base_model)
    frozen = base_model
    _set_trainable(
        adapted,
        int(last_n_encoder_layers),
        train_feature_projection=bool(train_feature_projection),
    )
    if gradient_checkpointing:
        enable = getattr(adapted, "gradient_checkpointing_enable", None)
        if enable is None:
            raise ValueError("teacher does not support gradient checkpointing")
        enable()
    frozen.eval()
    for parameter in frozen.parameters():
        parameter.requires_grad = False
    return adapted, frozen


def _valid_output_lengths(model, attention_mask, output_frames):
    """Map input attention lengths to model-frame validity without padding leaks."""
    import torch
    input_lengths = attention_mask.to(dtype=torch.long).sum(dim=1)
    helper = getattr(model, "_get_feat_extract_output_lengths", None)
    if helper is not None:
        lengths = helper(input_lengths)
    else:  # useful for tiny test doubles and unusual HF wrappers
        lengths = torch.floor(input_lengths.float() * output_frames / attention_mask.shape[1]).long()
    lengths = lengths.clamp(min=1, max=output_frames)
    indexes = torch.arange(output_frames, device=attention_mask.device)[None, :]
    return lengths, indexes < lengths[:, None]


def _weights_hash(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _jsonable_config(args: argparse.Namespace) -> dict[str, Any]:
    return {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}


def _encode_rows(rows, processor, *, sample_rate=TARGET_SAMPLE_RATE):
    import soundfile as sf
    import torch
    waves, masks = [], []
    for row in rows:
        values, rate = sf.read(Path(row["path"]), dtype="float32", always_2d=False)
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if int(rate) != sample_rate:
            raise ValueError(f"audio must be {sample_rate} Hz: {row['path']}")
        encoded = processor(values, sampling_rate=sample_rate, return_tensors="pt", return_attention_mask=True)
        waves.append(encoded.input_values[0]); masks.append(encoded.attention_mask[0])
    return torch.nn.utils.rnn.pad_sequence(waves, batch_first=True), torch.nn.utils.rnn.pad_sequence(masks, batch_first=True)


def evaluate_validation(model, frozen, processor, rows, *, device, canonical_path, collision_paths, blank_id, negative_target, collision_margin, batch_size, max_per_bucket, loss_weights):
    """Return a family-balanced adaptation-dev objective.

    Public speech has far more rows than any provider.  A raw row mean would
    therefore pick checkpoints almost entirely on that one source.  Instead,
    clean positive providers and negative families each receive one equal vote.
    """
    import torch
    validation = [dict(row) for row in rows if row.get("split") == "validation"]
    if not validation or not any(int(row.get("label", 0)) == 1 for row in validation) or not any(int(row.get("label", 0)) == 0 for row in validation):
        raise ValueError("adaptation manifest needs validation positives and negatives")
    model.eval(); frozen.eval()
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in validation:
        if int(row.get("label", 0)) == 1:
            family = (
                "device_positive"
                if row.get("source_group") == "device_channel_positive"
                else "clean_positive"
            )
            key = f"{family}:{row.get('provider')}"
        else:
            key = f"negative:{row.get('source_group')}"
        buckets.setdefault(key, []).append(row)
    required_buckets = {
        *(f"clean_positive:{provider}" for provider in APPROVED_PROVIDERS),
        *(f"device_positive:{provider}" for provider in APPROVED_PROVIDERS),
        *(f"negative:{group}" for group in NEGATIVE_GROUPS),
    }
    if missing := sorted(required_buckets - set(buckets)):
        raise ValueError(f"adaptation validation lacks balanced buckets: {missing}")
    buckets = {
        key: sorted(value, key=lambda row: str(row.get("source_id", "")))[
            :max_per_bucket
        ]
        for key, value in buckets.items()
    }
    ledger = []
    bucket_losses: dict[str, float] = {}
    with torch.no_grad():
        for bucket in sorted(buckets):
            weighted_total = 0.0
            row_count = 0
            for start in range(0, len(buckets[bucket]), batch_size):
                chunk = buckets[bucket][start : start + batch_size]
                inputs, attention = _encode_rows(chunk, processor)
                inputs, attention = inputs.to(device), attention.to(device)
                base = frozen(input_values=inputs, attention_mask=attention)
                out = model(input_values=inputs, attention_mask=attention)
                lengths, frame_mask = _valid_output_lengths(model, attention, out.logits.shape[1])
                labels = torch.tensor([int(row.get("label", 0)) for row in chunk], dtype=torch.float32, device=device)
                collisions = [row.get("source_group") == "kizz_control_phonetic_collision" for row in chunk]
                loss, parts = adaptation_loss(
                    out.logits, lengths, labels,
                    torch.log_softmax(base.logits, dim=-1), frame_mask,
                    canonical_path=canonical_path,
                    collision_paths=collision_paths,
                    blank_id=blank_id,
                    negative_target=negative_target,
                    collision_margin=collision_margin,
                    collision_mask=collisions,
                    **loss_weights,
                )
                weighted_total += float(loss) * len(chunk)
                row_count += len(chunk)
                ledger.append({"bucket": bucket, "start": start, "count": len(chunk), **parts})
            bucket_losses[bucket] = weighted_total / row_count
    model.train()
    return {
        "loss": float(np.mean(list(bucket_losses.values()))),
        "bucket_losses": bucket_losses,
        "batches": ledger,
        "rows": len(validation),
    }


def _detector_selection_metrics(
    model,
    processor,
    rows: Sequence[Mapping[str, Any]],
    *,
    device,
    token_ids: Mapping[str, Any],
    blank_id: int,
    window_lengths: Sequence[float],
    hop: float,
    beta: float,
    min_recall: float,
    max_faph: float,
) -> dict[str, Any]:
    """Evaluate the held-out detector contract for checkpoint selection.

    Only clean provider positives and validation negatives choose the operating
    point.  Device-channel positives are reported against that fixed point and
    never influence the threshold.
    """
    validation = [dict(row) for row in rows if row.get("split") == "validation"]
    clean = [
        row for row in validation
        if int(row.get("label", 0)) == 1
        and row.get("source_group") != "device_channel_positive"
    ]
    device_rows = [
        row for row in validation
        if int(row.get("label", 0)) == 1
        and row.get("source_group") == "device_channel_positive"
    ]
    negatives = [row for row in validation if int(row.get("label", 0)) == 0]
    if not clean or not device_rows or not negatives:
        raise ValueError("detector selection requires clean positives, device positives, and negatives")

    def score(group: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [
            _score_row(
                dict(row),
                model=model,
                processor=processor,
                token_ids=token_ids,
                blank_id=blank_id,
                device=device,
                window_lengths=window_lengths,
                hop=hop,
                beta=beta,
            )
            for row in group
        ]

    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    try:
        scored_clean = score(clean)
        scored_device = score(device_rows)
        scored_negative = score(negatives)
    finally:
        if was_training and hasattr(model, "train"):
            model.train()
    negative_seconds = sum(float(row.get("duration_seconds") or 0.0) for row in scored_negative)
    point = choose_validation_threshold(
        _finite_scores(scored_clean),
        _finite_scores(scored_negative),
        negative_exposure_seconds=negative_seconds,
        min_recall=min_recall,
        max_faph=max_faph,
    )
    threshold = point.get("threshold")

    def accepted(item: Mapping[str, Any]) -> bool:
        return bool(
            threshold is not None
            and item.get("score") is not None
            and item.get("collision_margin", -math.inf) >= beta
            and float(item["score"]) >= float(threshold)
        )

    clean_accepts = sum(accepted(item) for item in scored_clean)
    device_accepts = sum(accepted(item) for item in scored_device)
    negative_accepts = sum(accepted(item) for item in scored_negative)
    for item in (*scored_clean, *scored_device, *scored_negative):
        item["accepted"] = accepted(item)
    clean_recall = clean_accepts / max(1, len(scored_clean))
    device_recall = device_accepts / max(1, len(scored_device))
    exposure_hours = max(negative_seconds / 3600.0, 1e-12)
    metrics = {
        "qualified_clean_operating_point": bool(point.get("qualified")),
        "threshold": threshold,
        "threshold_selection": point,
        "clean": {
            "accepted": clean_accepts,
            "total": len(scored_clean),
            "recall": clean_recall,
        },
        "device_channel": {
            "accepted": device_accepts,
            "total": len(scored_device),
            "recall": device_recall,
        },
        "negative": {
            "accepted": negative_accepts,
            "total": len(scored_negative),
            "exposure_seconds": negative_seconds,
            "faph": negative_accepts / exposure_hours,
        },
        "contract": {
            "window_lengths_seconds": list(window_lengths),
            "hop_seconds": hop,
            "collision_margin_beta": beta,
            "min_recall": min_recall,
            "max_faph": max_faph,
        },
        "rows": {
            "clean_positive": scored_clean,
            "device_channel_positive": scored_device,
            "validation_negative": scored_negative,
        },
    }
    return metrics


def _checkpoint_rank(selection: Mapping[str, Any], validation_loss: float) -> tuple:
    """Deterministic detector-first checkpoint ordering."""
    return (
        bool(selection.get("qualified_clean_operating_point", False)),
        float(selection["device_channel"]["recall"]),
        float(selection["clean"]["recall"]),
        -int(selection["negative"]["accepted"]),
        -float(selection["negative"]["faph"]),
        -float(validation_loss),
    )


def _selection_progress(selection: Mapping[str, Any]) -> dict[str, Any]:
    """Return compact diagnostics even when no deployable threshold exists."""
    point = selection["threshold_selection"]
    rows = selection["rows"]["clean_positive"]
    eligible = sum(item.get("score") is not None for item in rows)
    return {
        "qualified_clean_operating_point": selection["qualified_clean_operating_point"],
        "clean_recall": selection["clean"]["recall"],
        "clean_eligible_recall": eligible / max(1, len(rows)),
        "recall_at_floor": point.get("recall"),
        "false_accepts_at_recall_floor": point.get("false_accepts_at_recall_floor"),
        "faph_at_selected_threshold": selection["negative"]["faph"],
        "device_channel_recall": selection["device_channel"]["recall"],
    }


def _save_checkpoint(model, processor, directory: Path) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory, safe_serialization=True, max_shard_size="10GB")
    processor.save_pretrained(directory)
    weights = resolve_hf_weights_path(
        str(directory), revision="local", local_files_only=True
    )
    return {
        "path": str(weights.resolve()),
        "file_sha256": sha256_file(weights),
        "state_sha256": _weights_hash(model),
    }


def _evaluate_checkpoint(
    model,
    frozen,
    processor,
    rows,
    *,
    step: int,
    checkpoint_directory: Path,
    device,
    canonical_path,
    collision_paths,
    blank_id: int,
    args: argparse.Namespace,
    loss_weights: Mapping[str, float],
) -> dict[str, Any]:
    """Evaluate and persist one comparable pre/post-optimization checkpoint."""
    validation = evaluate_validation(
        model,
        frozen,
        processor,
        rows,
        device=device,
        canonical_path=canonical_path,
        collision_paths=collision_paths,
        blank_id=blank_id,
        negative_target=args.negative_target,
        collision_margin=args.collision_margin,
        batch_size=args.validation_batch_size,
        max_per_bucket=args.validation_max_per_bucket,
        loss_weights=loss_weights,
    )
    selection = _detector_selection_metrics(
        model,
        processor,
        rows,
        device=device,
        token_ids={
            "canonical": canonical_path,
            "collisions": tuple(collision_paths.values()),
        },
        blank_id=blank_id,
        window_lengths=args.window_length,
        hop=args.hop,
        beta=args.beta,
        min_recall=args.min_recall,
        max_faph=args.max_faph,
    )
    checkpoint = _save_checkpoint(model, processor, checkpoint_directory)
    rank = _checkpoint_rank(selection, validation["loss"])
    return {
        "step": int(step),
        "validation": validation,
        "detector_selection": {
            "rank": list(rank),
            "checkpoint": checkpoint,
            "metrics": selection,
        },
    }


def _promote_checkpoint(checkpoint: Mapping[str, str], best_directory: Path) -> dict[str, str]:
    """Copy an immutable evaluated checkpoint to the stable downstream path."""
    source_directory = Path(checkpoint["path"]).resolve().parent
    if best_directory.exists():
        shutil.rmtree(best_directory)
    shutil.copytree(source_directory, best_directory)
    weights = resolve_hf_weights_path(
        str(best_directory), revision="local", local_files_only=True
    )
    return {
        "path": str(weights.resolve()),
        "file_sha256": sha256_file(weights),
        "state_sha256": str(checkpoint["state_sha256"]),
    }


def _discard_evaluated_checkpoint(
    checkpoint: dict[str, Any], checkpoints_directory: Path
) -> None:
    """Remove a hashed candidate after any winning copy is safely promoted."""
    weights = Path(str(checkpoint["path"])).resolve()
    directory = weights.parent
    root = checkpoints_directory.resolve()
    if directory.parent != root or not directory.name.startswith("step-"):
        raise ValueError("refusing to discard a checkpoint outside the run directory")
    if directory.exists():
        shutil.rmtree(directory)
    checkpoint["retained"] = False


def _checkpoint_record_rank(record: Mapping[str, Any]) -> tuple:
    return tuple(record["detector_selection"]["rank"])


def _checkpoint_selection_metadata(
    validation_ledger: Sequence[Mapping[str, Any]],
    best_record: Mapping[str, Any],
    best_checkpoint: Mapping[str, str],
) -> dict[str, Any]:
    if not validation_ledger or int(validation_ledger[0].get("step", -1)) != 0:
        raise ValueError("checkpoint selection ledger must begin with step 0")
    baseline = validation_ledger[0]
    return {
        "criterion": [
            "qualified_clean_operating_point descending",
            "device_channel_positive recall descending",
            "clean validation recall descending",
            "validation negative false accepts ascending",
            "validation negative FAPH ascending",
            "adaptation validation loss ascending",
        ],
        "threshold_source": "clean validation positives + validation negatives only",
        "device_channel_source": "validation only; measured at fixed clean threshold",
        "baseline": {
            "step": 0,
            **dict(baseline["detector_selection"]),
        },
        "evaluated_steps": [int(record["step"]) for record in validation_ledger],
        "selected_step": int(best_record["step"]),
        "selected_checkpoint": dict(best_checkpoint),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import soundfile as sf

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    phrase_spec = get_wake_phrase(args.phrase_id)
    payload, manifest_hash = load_adaptation_manifest(
        Path(args.manifest),
        expected_sha256=args.manifest_sha256,
        phrase_id=phrase_spec.phrase_id,
    )
    device_name = args.device
    if device_name == "mps" and not torch.backends.mps.is_available():
        device_name = "cpu"
    base_weights_path = resolve_hf_weights_path(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=args.local_files_only,
    )
    model, processor, tokenizer, _ = load_hf_teacher(
        MODEL_ID,
        revision=MODEL_REVISION,
        device="cpu",
        local_files_only=args.local_files_only,
    )
    adapted, frozen = make_adaptation_models(
        model,
        last_n_encoder_layers=args.last_n_encoder_layers,
        train_feature_projection=args.train_feature_projection,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    del model
    gc.collect()
    device = torch.device(device_name)
    adapted.train().to(device); frozen.eval().to(device)
    optimizer = torch.optim.AdamW([p for p in adapted.parameters() if p.requires_grad], lr=args.learning_rate)
    rows = _rows(payload)
    mixture = DeterministicBatchMixture(rows, batch_size=args.batch_size, seed=args.seed)
    token_ids = resolve_phone_ids(tokenizer, phrase_spec.phones)
    collision_phones = {
        name: phones
        for name, phones in zip(
            phrase_spec.collision_transcripts,
            phrase_spec.collision_phones,
            strict=True,
        )
    }
    collision_ids = {
        name: resolve_phone_ids(tokenizer, phones)
        for name, phones in collision_phones.items()
    }
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    training_ledger = []
    validation_ledger = []
    realized_groups = Counter(); realized_providers = Counter(); realized_clean_providers = Counter(); realized_device_providers = Counter()
    loss_weights = {
        "positive_weight": args.positive_weight,
        "negative_weight": args.negative_weight,
        "collision_weight": args.collision_weight,
        "kl_weight": args.kl_weight,
    }
    blank_id = int(tokenizer.pad_token_id or 0)
    baseline_record = _evaluate_checkpoint(
        adapted,
        frozen,
        processor,
        rows,
        step=0,
        checkpoint_directory=output / "checkpoints" / "step-000000",
        device=device,
        canonical_path=token_ids,
        collision_paths=collision_ids,
        blank_id=blank_id,
        args=args,
        loss_weights=loss_weights,
    )
    validation_ledger.append(baseline_record)
    best_record = baseline_record
    best_rank = _checkpoint_record_rank(best_record)
    best_checkpoint = _promote_checkpoint(
        baseline_record["detector_selection"]["checkpoint"], output / "best"
    )
    baseline_metrics = baseline_record["detector_selection"]["metrics"]
    print(
        json.dumps(
            {
                "step": 0,
                "validation_loss": baseline_record["validation"]["loss"],
                "validation_bucket_losses": baseline_record["validation"]["bucket_losses"],
                **_selection_progress(baseline_metrics),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for step in range(args.steps):
        batch_rows = mixture.batch(step)
        raw_values = []
        for row in batch_rows:
            values, rate = sf.read(Path(row["path"]), dtype="float32", always_2d=False)
            values = np.asarray(values, dtype=np.float32)
            if values.ndim == 2:
                values = values.mean(axis=1, dtype=np.float32)
            values = values.reshape(-1)
            if int(rate) != TARGET_SAMPLE_RATE:
                raise ValueError(f"audio must be {TARGET_SAMPLE_RATE} Hz: {row['path']}")
            raw_values.append(values)
        negative_backgrounds = [
            values
            for row, values in zip(batch_rows, raw_values, strict=True)
            if int(row.get("label", 0)) == 0
        ]
        waves, masks, labels = [], [], []
        collision_mask = []
        for row_index, (row, values) in enumerate(
            zip(batch_rows, raw_values, strict=True)
        ):
            if int(row.get("label", 0)) == 1 and row.get("source_group") != "device_channel_positive":
                rng = np.random.default_rng(
                    args.seed + step * args.batch_size + row_index
                )
                background = (
                    negative_backgrounds[row_index % len(negative_backgrounds)]
                    if negative_backgrounds
                    and rng.random() < args.background_overlay_probability
                    else None
                )
                values = augment_positive_waveform(
                    values,
                    rng,
                    background=background,
                    preserve_probability=args.unaugmented_fraction,
                )
            encoded = processor(values, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt", return_attention_mask=True)
            waves.append(encoded.input_values[0]); masks.append(encoded.attention_mask[0]); labels.append(int(row.get("label", 0)))
            collision_mask.append(row.get("source_group") == "kizz_control_phonetic_collision")
            realized_groups[str(row.get("source_group"))] += 1; realized_providers[str(row.get("provider", "none"))] += 1
            if int(row.get("label", 0)) == 1:
                target = (
                    realized_device_providers
                    if row.get("source_group") == "device_channel_positive"
                    else realized_clean_providers
                )
                target[str(row.get("provider", "none"))] += 1
        inputs = torch.nn.utils.rnn.pad_sequence(waves, batch_first=True).to(device)
        attention = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True).to(device)
        y = torch.tensor(labels, dtype=torch.float32, device=device)
        with torch.no_grad():
            base_out = frozen(input_values=inputs, attention_mask=attention)
            base_log_probs = torch.log_softmax(base_out.logits, dim=-1)
        out = adapted(input_values=inputs, attention_mask=attention)
        lengths, frame_mask = _valid_output_lengths(adapted, attention, out.logits.shape[1])
        # A tensor subclass is unnecessary in production; the explicit mask is
        # supplied by this closure-compatible carrier for adaptation_loss.
        loss, parts = adaptation_loss(
            out.logits, lengths, y, base_log_probs, frame_mask,
            canonical_path=token_ids,
            collision_paths=collision_ids,
            blank_id=blank_id,
            negative_target=args.negative_target,
            collision_margin=args.collision_margin,
            collision_mask=collision_mask,
            **loss_weights,
        )
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(adapted.parameters(), args.max_grad_norm); optimizer.step()
        completed_step = step + 1
        training_ledger.append({"step": completed_step, **parts})
        if completed_step % args.progress_interval == 0:
            print(
                json.dumps({"step": completed_step, "steps": args.steps, **parts}),
                flush=True,
            )
        if completed_step % args.validation_interval == 0 or completed_step == args.steps:
            record = _evaluate_checkpoint(
                adapted,
                frozen,
                processor,
                rows,
                step=completed_step,
                checkpoint_directory=output / "checkpoints" / f"step-{completed_step:06d}",
                device=device,
                canonical_path=token_ids,
                collision_paths=collision_ids,
                blank_id=blank_id,
                args=args,
                loss_weights=loss_weights,
            )
            validation_ledger.append(record)
            validation = record["validation"]
            selection = record["detector_selection"]["metrics"]
            rank = _checkpoint_record_rank(record)
            print(
                json.dumps(
                    {
                        "step": completed_step,
                        "validation_loss": validation["loss"],
                        "validation_bucket_losses": validation["bucket_losses"],
                        **_selection_progress(selection),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if rank > best_rank:
                best_rank = rank
                best_record = record
                best_checkpoint = _promote_checkpoint(
                    record["detector_selection"]["checkpoint"], output / "best"
                )
            if not args.retain_evaluated_checkpoints:
                _discard_evaluated_checkpoint(
                    record["detector_selection"]["checkpoint"],
                    output / "checkpoints",
                )
    last_checkpoint = _save_checkpoint(adapted, processor, output / "last")
    metadata = {
        "schema_version": 1,
        "kind": "kizz_phoneme_teacher_adaptation",
        "base_model": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "weights_path": str(base_weights_path.resolve()),
            "weights_sha256": sha256_file(base_weights_path),
        },
        "wake_phrase": {
            "phrase_id": phrase_spec.phrase_id,
            "phones": list(phrase_spec.phones),
            "collision_paths": {
                key: list(value) for key, value in collision_phones.items()
            },
        },
        "manifest": {
            "path": str(Path(args.manifest).resolve()),
            "sha256": manifest_hash,
        },
        "config": _jsonable_config(args),
        "realized_counts": {
            "groups": dict(realized_groups),
            "providers": dict(realized_providers),
            "clean_positive_providers": dict(realized_clean_providers),
            "device_positive_providers": dict(realized_device_providers),
        },
        "best_validation_loss": best_record["validation"]["loss"],
        "best_selection_rank": list(best_rank),
        "checkpoint_selection": _checkpoint_selection_metadata(
            validation_ledger, best_record, best_checkpoint
        ),
        "training_ledger": training_ledger,
        "validation_ledger": validation_ledger,
        "checkpoints": {
            "base": baseline_record["detector_selection"]["checkpoint"],
            "best": best_checkpoint,
            "last": last_checkpoint,
        },
    }
    (output / "training.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--validation-interval", type=int, default=50)
    parser.add_argument("--validation-batch-size", type=int, default=4)
    parser.add_argument("--validation-max-per-bucket", type=int, default=32)
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--last-n-encoder-layers", type=int, default=2)
    parser.add_argument("--train-feature-projection", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--retain-evaluated-checkpoints", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--negative-target", type=float, default=-4.0)
    parser.add_argument("--collision-margin", type=float, default=0.2)
    parser.add_argument("--window-length", type=float, action="append")
    parser.add_argument("--hop", type=float, default=0.06)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--min-recall", type=float, default=0.90)
    parser.add_argument("--max-faph", type=float, default=0.10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--unaugmented-fraction", type=float, default=0.25)
    parser.add_argument("--background-overlay-probability", type=float, default=0.50)
    parser.add_argument("--positive-weight", type=float, default=1.0)
    parser.add_argument("--negative-weight", type=float, default=1.0)
    parser.add_argument("--collision-weight", type=float, default=1.0)
    parser.add_argument("--kl-weight", type=float, default=0.25)
    parser.add_argument(
        "--phrase-id", choices=tuple(sorted(WAKE_PHRASES)), default=KIZZ_CONTROL.phrase_id
    )
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    args.window_length = args.window_length or [
        0.56, 0.68, 0.80, 0.96, 1.16, 1.40, 1.60
    ]
    if (
        args.steps < 1
        or args.validation_interval < 1
        or args.validation_batch_size < 1
        or args.validation_max_per_bucket < 1
        or args.progress_interval < 1
        or args.learning_rate <= 0
        or args.max_grad_norm <= 0
        or args.hop <= 0
        or args.beta < 0
        or not 0 < args.min_recall <= 1
        or args.max_faph < 0
        or any(length <= 0 for length in args.window_length)
        or not 0 <= args.unaugmented_fraction <= 1
        or not 0 <= args.background_overlay_probability <= 1
        or min(
            args.positive_weight,
            args.negative_weight,
            args.collision_weight,
            args.kl_weight,
        )
        < 0
    ):
        parser.error("invalid adaptation optimization settings")
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
