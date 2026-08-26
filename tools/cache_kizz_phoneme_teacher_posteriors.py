#!/usr/bin/env python3
"""Cache compact CTC posteriors from the pinned generic phoneme teacher.

The cache is deliberately a data artifact, not a training implementation.  It
contains log probabilities for a compact vocabulary consisting of blank, every
phone used by the selected wake phrase and its declared collision paths, and an
OTHER bucket.  OTHER is computed with log-sum-exp over *all* teacher classes
not represented explicitly, so no probability mass is discarded.

Timing is derived from the teacher's convolution configuration.  In
particular, this tool never applies an empirical alignment offset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from microwakeword.kizz_phoneme_teacher import (
    MODEL_ID,
    MODEL_REVISION,
    TARGET_SAMPLE_RATE,
    load_hf_teacher,
    resolve_hf_weights_path,
    resolve_phone_ids,
    sha256_file,
)
from microwakeword.wake_phrase import HI_FI_KIZZ, WAKE_PHRASES, get_wake_phrase


SCHEMA_VERSION = 1
OTHER = "OTHER"


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    if isinstance(payload, dict):
        for key in ("examples", "records", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(row) for row in value]
    raise ValueError(f"manifest has no examples/records/items list: {path}")


def _audio_sha256(path: Path) -> str:
    return sha256_file(path)


def processor_vocab_hash(tokenizer: Any) -> str:
    """Hash every tokenizer value relevant to resolving teacher classes."""
    vocab = tokenizer.get_vocab()
    payload = {
        "vocab": {str(key): int(value) for key, value in sorted(vocab.items())},
        "unk_token_id": getattr(tokenizer, "unk_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
    }
    return _sha256_bytes(_canonical_json(payload))


def compact_vocabulary(tokenizer: Any, *, phrase_id: str = HI_FI_KIZZ.phrase_id) -> dict[str, Any]:
    """Build a deterministic compact vocabulary and ordered phone paths.

    Phone symbols are deduplicated only in the vocabulary.  The paths retain
    their original order and repeated occurrences, e.g. Kizz Control contains
    ``k`` at positions 0 and 3.
    """
    phrase = get_wake_phrase(phrase_id)
    required_phones = tuple(dict.fromkeys(phrase.phones + tuple(phone for path in phrase.collision_phones for phone in path)))
    phone_ids = resolve_phone_ids(tokenizer, required_phones)
    blank_id = getattr(tokenizer, "pad_token_id", None)
    if blank_id is None:
        raise ValueError("tokenizer must expose pad_token_id for the CTC blank")
    blank_id = int(blank_id)
    if blank_id in phone_ids:
        raise ValueError("CTC blank collides with a required phone token")
    if len(set(phone_ids)) != len(phone_ids):
        raise ValueError("required phone symbols map to duplicate teacher token IDs")

    compact_tokens = ("<blank>",) + required_phones + (OTHER,)
    compact_ids = {token: index for index, token in enumerate(compact_tokens)}
    return {
        "tokens": list(compact_tokens),
        "phone_tokens": list(required_phones),
        "teacher_token_ids": {phone: token_id for phone, token_id in zip(required_phones, phone_ids)},
        "teacher_blank_id": blank_id,
        "other_compact_id": compact_ids[OTHER],
        "canonical_path": [compact_ids[phone] for phone in phrase.phones],
        "collision_paths": {
            transcript: [compact_ids[phone] for phone in path]
            for transcript, path in zip(phrase.collision_transcripts, phrase.collision_phones)
        },
        "phrase_id": phrase.phrase_id,
        "phrase_text": phrase.text,
        "canonical_phones": list(phrase.phones),
        "collision_phones": {
            transcript: list(path)
            for transcript, path in zip(phrase.collision_transcripts, phrase.collision_phones)
        },
    }


def teacher_timing_metadata(model: Any, *, sample_rate: int = TARGET_SAMPLE_RATE) -> dict[str, Any]:
    """Derive frame timing from Wav2Vec2's actual convolution geometry."""
    config = model.config
    kernels = tuple(int(value) for value in getattr(config, "conv_kernel", ()))
    strides = tuple(int(value) for value in getattr(config, "conv_stride", ()))
    if not kernels or len(kernels) != len(strides) or any(value <= 0 for value in kernels + strides):
        raise ValueError("teacher config must expose positive conv_kernel/conv_stride")
    receptive_field = 1
    jump = 1
    for kernel, stride in zip(kernels, strides):
        receptive_field += (kernel - 1) * jump
        jump *= stride
    ratio = getattr(config, "inputs_to_logits_ratio", None)
    if ratio is not None and int(ratio) != jump:
        raise ValueError("teacher timing config disagrees with convolution strides")
    return {
        "sample_rate": int(sample_rate),
        "frame_stride_samples": jump,
        "frame_stride_seconds": jump / sample_rate,
        "receptive_field_samples": receptive_field,
        "receptive_field_seconds": receptive_field / sample_rate,
        "frame_start_seconds": 0.0,
        "frame_center_seconds": (receptive_field - 1) / (2 * sample_rate),
        "timing_basis": "model.config.conv_kernel_and_conv_stride",
        "arbitrary_offset_applied": False,
    }


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    return (maximum + np.log(np.exp(values - maximum).sum(axis=axis, keepdims=True))).squeeze(axis)


def compact_log_posteriors(logits: np.ndarray, vocabulary: Mapping[str, Any]) -> np.ndarray:
    """Map teacher logits to compact log probabilities with exact mass closure."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or not values.shape[0]:
        raise ValueError("teacher logits must be a non-empty [frames, classes] array")
    selected = {int(v) for v in vocabulary["teacher_token_ids"].values()}
    blank = int(vocabulary["teacher_blank_id"])
    if blank in selected or max(selected | {blank}) >= values.shape[1]:
        raise ValueError("teacher token ID is outside the logits vocabulary")
    log_probs = values - _logsumexp(values, axis=1)[:, None]
    output = np.empty((len(values), len(vocabulary["tokens"])), dtype=np.float32)
    output[:, 0] = log_probs[:, blank]
    for compact_id, phone in enumerate(vocabulary["phone_tokens"], start=1):
        output[:, compact_id] = log_probs[:, int(vocabulary["teacher_token_ids"][phone])]
    other_ids = [index for index in range(values.shape[1]) if index not in selected and index != blank]
    output[:, int(vocabulary["other_compact_id"])] = _logsumexp(log_probs[:, other_ids], axis=1) if other_ids else -np.inf
    return output


def _model_logits(model: Any, processor: Any, waveform: np.ndarray, device: Any) -> np.ndarray:
    inputs = processor(waveform, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")
    input_items = inputs.items() if hasattr(inputs, "items") else vars(inputs).items()
    kwargs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in input_items
        if not key.startswith("_")
    }
    try:
        import torch
    except ImportError:  # pragma: no cover - the real CLI requires torch
        torch = None
    if torch is None:
        output = model(**kwargs)
    else:
        with torch.inference_mode():
            output = model(**kwargs)
    logits = output.logits if hasattr(output, "logits") else output[0]
    if hasattr(logits, "detach"):
        return logits[0].detach().cpu().numpy()
    return np.asarray(logits[0] if np.asarray(logits).ndim == 3 else logits)


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("soundfile is required to cache teacher posteriors") from error
    waveform, sample_rate = sf.read(path, always_2d=False, dtype="float32")
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1, dtype=np.float32)
    if waveform.ndim != 1 or not len(waveform):
        raise ValueError(f"audio is empty or not mono/stereo: {path}")
    return waveform, int(sample_rate)


def cache_manifest(
    manifest_path: Path,
    output_prefix: Path,
    *,
    model: Any,
    processor: Any,
    tokenizer: Any,
    device: Any,
    phrase_id: str = HI_FI_KIZZ.phrase_id,
    model_identity: Mapping[str, Any] | None = None,
    teacher_qualification: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rows = _rows(manifest_path)
    vocabulary = compact_vocabulary(tokenizer, phrase_id=phrase_id)
    timing = teacher_timing_metadata(model)
    arrays: list[np.ndarray] = []
    offsets = [0]
    examples = []
    for index, row in enumerate(rows):
        path = Path(row["path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        source_sha = _audio_sha256(path)
        declared_sha = row.get("audio_sha256") or row.get("source_audio_sha256")
        if declared_sha and declared_sha != source_sha:
            raise ValueError(f"manifest audio hash mismatch: {path}")
        waveform, sample_rate = _load_audio(path)
        if sample_rate != TARGET_SAMPLE_RATE:
            raise ValueError(f"audio must be {TARGET_SAMPLE_RATE} Hz: {path} ({sample_rate})")
        compact = compact_log_posteriors(_model_logits(model, processor, waveform, device), vocabulary)
        arrays.append(compact)
        offsets.append(offsets[-1] + len(compact))
        examples.append({
            "source_id": row.get("source_id"),
            "path": str(path),
            "audio_sha256": source_sha,
            "sample_rate": sample_rate,
            "source_frame_count": int(len(waveform)),
            "teacher_frame_count": int(len(compact)),
            "offset": offsets[-2],
            "end_offset": offsets[-1],
            "duration_seconds": len(waveform) / sample_rate,
            "label": row.get("label"),
            "split": row.get("split"),
        })
        if (index + 1) % 50 == 0 or index + 1 == len(rows):
            print(
                json.dumps({"cached": index + 1, "total": len(rows)}),
                flush=True,
            )
    matrix = np.concatenate(arrays, axis=0) if arrays else np.empty((0, len(vocabulary["tokens"])), dtype=np.float32)
    offsets_array = np.asarray(offsets, dtype=np.int64)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "representation": "compact_log_posteriors",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "example_count": len(examples),
        "posterior_shape": list(matrix.shape),
        "offsets_shape": list(offsets_array.shape),
        "model": dict(model_identity or {"id": MODEL_ID, "revision": MODEL_REVISION}),
        "processor_vocab_sha256": processor_vocab_hash(tokenizer),
        "sample_rate": TARGET_SAMPLE_RATE,
        "timing": timing,
        "vocabulary": vocabulary,
        "examples": examples,
        "provenance": {
            "source_audio_hash": "sha256 of exact manifest-referenced file bytes",
            "teacher_revision_pinned": True,
            "teacher_qualification": dict(teacher_qualification or {}),
            "probability_mass": "OTHER is log-sum-exp of every non-blank, non-explicit teacher class",
        },
    }
    data_hash = _sha256_bytes(matrix.tobytes(order="C") + offsets_array.tobytes(order="C") + _canonical_json(metadata))
    metadata["cache_sha256"] = data_hash
    output_prefix = output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=output_prefix.name, suffix=".npz", dir=output_prefix.parent, delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        np.savez_compressed(temp_path, log_posteriors=matrix, offsets=offsets_array)
        os.replace(temp_path, output_prefix.with_suffix(".npz"))
        output_prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    finally:
        temp_path.unlink(missing_ok=True)
    return metadata


def load_cache(
    cache_prefix: Path,
    *,
    expected_model_revision: str = MODEL_REVISION,
    expected_processor_vocab_sha256: str | None = None,
    expected_source_audio_sha256: Sequence[str] | None = None,
    expected_weights_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Load and validate a cache, rejecting stale provenance or changed bytes."""
    prefix = cache_prefix.with_suffix("")
    metadata = json.loads(prefix.with_suffix(".json").read_text())
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported teacher posterior cache schema")
    if metadata.get("model", {}).get("revision") != expected_model_revision:
        raise ValueError("stale cache: teacher revision differs")
    if (
        expected_weights_sha256
        and metadata.get("model", {}).get("weights_sha256")
        != expected_weights_sha256
    ):
        raise ValueError("stale cache: teacher weights differ")
    if expected_processor_vocab_sha256 and metadata.get("processor_vocab_sha256") != expected_processor_vocab_sha256:
        raise ValueError("stale cache: processor vocabulary differs")
    with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as loaded:
        arrays = {"log_posteriors": np.asarray(loaded["log_posteriors"]), "offsets": np.asarray(loaded["offsets"])}
    expected_hash = _sha256_bytes(arrays["log_posteriors"].tobytes(order="C") + arrays["offsets"].tobytes(order="C") + _canonical_json({key: value for key, value in metadata.items() if key != "cache_sha256"}))
    if metadata.get("cache_sha256") != expected_hash:
        raise ValueError("stale or corrupted teacher posterior cache")
    if expected_source_audio_sha256 is not None:
        actual = [row.get("audio_sha256") for row in metadata.get("examples", [])]
        if list(expected_source_audio_sha256) != actual:
            raise ValueError("stale cache: source audio set differs")
    return metadata, arrays


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="cache prefix; writes .npz and .json")
    parser.add_argument("--phrase-id", choices=tuple(sorted(WAKE_PHRASES)), default=HI_FI_KIZZ.phrase_id)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--teacher-qualification", type=Path, required=True)
    args = parser.parse_args(argv)
    qualification = json.loads(args.teacher_qualification.read_text())
    if qualification.get("qualified") is not True:
        parser.error("--teacher-qualification must be a qualified report")
    teacher_model = qualification.get("model", {})
    if not all(teacher_model.get(key) for key in ("id", "revision", "weights_sha256")):
        parser.error("teacher qualification lacks exact model identity")
    weights_path = resolve_hf_weights_path(
        teacher_model["id"],
        revision=teacher_model["revision"],
        local_files_only=True,
    )
    if sha256_file(weights_path) != teacher_model["weights_sha256"]:
        parser.error("qualified teacher weights changed before posterior caching")
    model, processor, tokenizer, device = load_hf_teacher(
        teacher_model["id"],
        revision=teacher_model["revision"],
        device=args.device,
        local_files_only=True,
    )
    metadata = cache_manifest(
        args.manifest,
        args.output,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        device=device,
        phrase_id=args.phrase_id,
        model_identity={
            "id": teacher_model["id"],
            "revision": teacher_model["revision"],
            "weights_sha256": teacher_model["weights_sha256"],
        },
        teacher_qualification={
            "path": str(args.teacher_qualification.resolve()),
            "sha256": sha256_file(args.teacher_qualification),
        },
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
