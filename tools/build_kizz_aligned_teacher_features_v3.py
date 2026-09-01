#!/usr/bin/env python3
"""Build acoustically-qualified, phone-aligned Kizz teacher features.

This builder is intentionally stricter than the archived clean-slate-v1 path:

* only canonical positives accepted by the pinned acoustic aligner are read;
* measured phone spans are translated through the exact waveform crop/pad;
* one ordered state is emitted for each of the seven phones (nine outputs with
  background and silence), avoiding the 630 ms minimum imposed by 21 states;
* augmentation descendants remain in their parent's split and are created only
  for training; and
* validation and test positives remain clean, voice-disjoint opportunities.

The tool writes fixed ``[N, 260, 40]`` frontend arrays and ``[N, 87]`` state
targets.  It can also materialize selected canonical-v3 collision groups as
fixed negative sources.  Locked deployment anchors are never accepted here as
training examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

from microwakeword.audio.audio_utils import MicroFrontend
from microwakeword.ordered_state_data import example_from_mapping, frame_state_targets
from microwakeword.wake_phrase import (
    HI_FI_KIZZ,
    WAKE_PHRASES,
    WakePhraseSpec,
    get_wake_phrase,
)

SAMPLE_RATE = 16_000
INPUT_FRAMES = 260
FEATURE_BINS = 40
OUTPUT_FRAMES = 87
CONTEXT_SAMPLES = 41_920
SAMPLES_PER_CALL = 160
TARGET_FRAME_TIMES = 0.015 + 0.030 * np.arange(OUTPUT_FRAMES)
ALLOWED_ALIGNMENT_METHODS = frozenset(
    (
        "ctc_forced_alignment",
        "inherited_ctc_forced_alignment",
        "wav2vec2_ipa_ctc_forced_alignment",
    )
)
DEFAULT_OVERLAY_SNR_DB = (5.0, 10.0, 15.0, 20.0)
DEFAULT_GAIN_DB_RANGE = (-6.0, 3.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("examples", payload.get("records"))
    else:
        rows = None
    if not isinstance(rows, list):
        raise TypeError(f"{path}: manifest must contain examples or records")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: every manifest row must be an object")
    return rows


def load_pronunciation_acceptances(audit_path: Path, source_manifest: Path) -> set[str]:
    """Load an all-split pronunciation allowlist bound to its source manifest."""
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    scope = payload.get("scope", {})
    if (
        payload.get("gate_scope") != "independent_source_pronunciation_qc"
        or payload.get("source_manifest_sha256") != sha256_file(source_manifest)
        or payload.get("qualified") is not True
        or scope.get("gate_mode") not in {"all", "training_eligible"}
        or set(scope.get("splits", [])) != {"train", "validation", "test"}
    ):
        raise ValueError(
            "source pronunciation audit is not the qualified bound all-split gate"
        )
    results = payload.get("results", [])
    identities = [str(row.get("source_id", "")) for row in results]
    if (
        not identities
        or any(not identity for identity in identities)
        or len(set(identities)) != len(identities)
    ):
        raise ValueError("source pronunciation audit has missing/duplicate identities")
    return {str(row["source_id"]) for row in results if row.get("accepted") is True}


def _span_seconds(value: Mapping[str, Any], name: str) -> tuple[float, float]:
    if value.get("start_s") is not None and value.get("end_s") is not None:
        start, end = float(value["start_s"]), float(value["end_s"])
    elif value.get("start_ms") is not None and value.get("end_ms") is not None:
        start, end = float(value["start_ms"]) / 1000.0, float(value["end_ms"]) / 1000.0
    else:
        raise ValueError(f"{name} requires seconds or milliseconds")
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise ValueError(f"{name} is invalid")
    return start, end


def validate_aligned_positive(
    row: Mapping[str, Any], phrase_spec: WakePhraseSpec = HI_FI_KIZZ
) -> None:
    """Fail closed unless a row is an accepted canonical phone alignment."""
    if int(row.get("label", -1)) != 1:
        raise ValueError("aligned positive must have label 1")
    if row.get("training_eligible") is not True:
        raise ValueError("aligned positive must be explicitly training eligible")
    if row.get("locked_deployment_anchor"):
        raise ValueError("locked deployment anchors may not enter training")
    if row.get("semantic_label") != "canonical_exact":
        raise ValueError("aligned positive must be canonical_exact")
    if tuple(row.get("target_phones", ())) != tuple(phrase_spec.phones):
        raise ValueError("aligned positive must declare the canonical phrase phones")
    if row.get("split") not in ("train", "validation", "test"):
        raise ValueError("aligned positive must declare a supported split")
    alignment = row.get("alignment")
    if not isinstance(alignment, Mapping):
        raise TypeError("aligned positive requires alignment metadata")
    if alignment.get("method") not in ALLOWED_ALIGNMENT_METHODS:
        raise ValueError("aligned positive requires a measured CTC alignment")
    decision = alignment.get("pronunciation_decision")
    if not isinstance(decision, Mapping) or decision.get("accepted") is not True:
        raise ValueError("aligned positive failed acoustic pronunciation qualification")
    phrase = row.get("phrase_span")
    phones = row.get("phone_spans")
    if not isinstance(phrase, Mapping) or not isinstance(phones, list):
        raise TypeError("aligned positive requires phrase and phone spans")
    phrase_start, phrase_end = _span_seconds(phrase, "phrase_span")
    if len(phones) != len(phrase_spec.phones):
        raise ValueError(
            f"aligned positive requires exactly {len(phrase_spec.phones)} phone spans"
        )
    previous = phrase_start
    for expected, span in zip(phrase_spec.phones, phones):
        if span.get("phone") != expected:
            raise ValueError("phone spans are not the canonical ordered sequence")
        start, end = _span_seconds(span, f"phone {expected}")
        if start + 1e-6 < previous or start < phrase_start or end > phrase_end + 1e-6:
            raise ValueError("phone spans must be ordered inside the phrase")
        previous = end


def load_audio(path: Path) -> np.ndarray:
    samples, rate = sf.read(path, dtype="float32", always_2d=False)
    values = np.asarray(samples, dtype=np.float32)
    if values.ndim == 2:
        values = np.mean(values, axis=1)
    if values.ndim != 1 or not len(values) or np.any(~np.isfinite(values)):
        raise ValueError(f"{path}: invalid audio")
    if rate != SAMPLE_RATE:
        values = resample_poly(values, SAMPLE_RATE, rate).astype(np.float32)
    return np.clip(values, -1.0, 1.0)


def place_phrase_context(
    samples: np.ndarray,
    phrase_span_s: tuple[float, float],
    *,
    desired_phrase_center_s: float | None = None,
) -> tuple[np.ndarray, float]:
    """Return a fixed context and the source-to-context time translation.

    ``translation_s`` is added to every measured source span.  Cropping and
    padding are chosen so the complete phrase remains inside the context.
    """
    values = np.asarray(samples, dtype=np.float32)
    start_s, end_s = phrase_span_s
    source_duration = len(values) / SAMPLE_RATE
    context_duration = CONTEXT_SAMPLES / SAMPLE_RATE
    if end_s > source_duration + 1e-6:
        raise ValueError("phrase span exceeds source waveform")
    phrase_center = (start_s + end_s) * 0.5
    desired = (
        context_duration * 0.5
        if desired_phrase_center_s is None
        else float(desired_phrase_center_s)
    )
    if not math.isfinite(desired) or not 0 < desired < context_duration:
        raise ValueError("desired phrase center must be inside the context")

    if len(values) >= CONTEXT_SAMPLES:
        ideal_start_s = phrase_center - desired
        minimum_start_s = max(0.0, end_s - context_duration)
        maximum_start_s = min(start_s, source_duration - context_duration)
        if maximum_start_s + 1e-9 < minimum_start_s:
            raise ValueError("phrase cannot fit inside teacher context")
        crop_start_s = min(max(ideal_start_s, minimum_start_s), maximum_start_s)
        crop_start = round(crop_start_s * SAMPLE_RATE)
        crop_start = min(max(crop_start, 0), len(values) - CONTEXT_SAMPLES)
        return (
            values[crop_start : crop_start + CONTEXT_SAMPLES],
            -crop_start / SAMPLE_RATE,
        )

    ideal_left = round((desired - phrase_center) * SAMPLE_RATE)
    left = min(max(ideal_left, 0), CONTEXT_SAMPLES - len(values))
    right = CONTEXT_SAMPLES - len(values) - left
    return np.pad(values, (left, right)), left / SAMPLE_RATE


def frontend(samples: np.ndarray) -> np.ndarray:
    """Run the exact product microfrontend on one fixed PCM context."""
    values = np.asarray(samples, dtype=np.float32)
    if values.shape != (CONTEXT_SAMPLES,):
        raise ValueError("frontend requires one fixed waveform context")
    pcm = np.rint(np.clip(values, -1.0, 1.0) * 32767.0).astype("<i2")
    processor = MicroFrontend()
    process = getattr(processor, "process_samples", None) or processor.ProcessSamples
    rows = []
    raw = pcm.tobytes()
    offset = 0
    while offset + SAMPLES_PER_CALL * 2 <= len(raw):
        result = process(raw[offset : offset + SAMPLES_PER_CALL * 2])
        feature = np.asarray(result.features, dtype=np.float32)
        if feature.ndim == 2:
            feature = feature[0]
        if feature.shape == (FEATURE_BINS,):
            rows.append(feature)
        read = int(getattr(result, "samples_read", SAMPLES_PER_CALL))
        if read <= 0:
            raise ValueError("microfrontend made no progress")
        offset += read * 2
    result = np.stack(rows)
    if result.shape != (INPUT_FRAMES, FEATURE_BINS):
        raise ValueError(
            f"frontend emitted {result.shape}, expected {(INPUT_FRAMES, FEATURE_BINS)}"
        )
    return result


def _translated_record(
    row: Mapping[str, Any],
    translation_s: float,
    source_id: str,
    phrase_spec: WakePhraseSpec,
) -> dict[str, Any]:
    phrase_start, phrase_end = _span_seconds(row["phrase_span"], "phrase_span")
    phones = []
    for span in row["phone_spans"]:
        start, end = _span_seconds(span, "phone_span")
        phones.append(
            {
                "phone": span["phone"],
                "start_s": start + translation_s,
                "end_s": end + translation_s,
            }
        )
    record = {
        "source_id": source_id,
        "truth": True,
        "duration_s": CONTEXT_SAMPLES / SAMPLE_RATE,
        "phrase_span": {
            "start_s": phrase_start + translation_s,
            "end_s": phrase_end + translation_s,
        },
        "phone_spans": phones,
    }
    # Re-parse to enforce the ordered-state timing contract after translation.
    example_from_mapping(record, expected_phones=phrase_spec.phones)
    return record


def _background_context(
    samples: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, int, bool]:
    values = np.asarray(samples, dtype=np.float32)
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("background audio must be finite and non-empty")
    if len(values) >= CONTEXT_SAMPLES:
        start = int(rng.integers(0, len(values) - CONTEXT_SAMPLES + 1))
        return values[start : start + CONTEXT_SAMPLES], start, False
    repeats = math.ceil(CONTEXT_SAMPLES / len(values))
    return np.tile(values, repeats)[:CONTEXT_SAMPLES], 0, repeats > 1


def apply_gain_db(samples: np.ndarray, gain_db: float) -> np.ndarray:
    if not math.isfinite(gain_db):
        raise ValueError("gain must be finite")
    return np.clip(
        np.asarray(samples, dtype=np.float32) * (10.0 ** (gain_db / 20.0)),
        -1.0,
        1.0,
    ).astype(np.float32)


def apply_room_impulse_response(
    samples: np.ndarray, impulse_response: np.ndarray
) -> tuple[np.ndarray, int]:
    """Apply a finite-energy RIR after trimming its pre-arrival silence."""
    signal = np.asarray(samples, dtype=np.float32)
    rir = np.asarray(impulse_response, dtype=np.float32)
    if not len(rir) or not np.all(np.isfinite(rir)):
        raise ValueError("room impulse response must be finite and non-empty")
    peak = float(np.max(np.abs(rir)))
    if peak <= 1e-8:
        raise ValueError("room impulse response has no usable impulse")
    arrivals = np.flatnonzero(np.abs(rir) >= peak * 0.05)
    arrival_sample = int(arrivals[0])
    rir = rir[arrival_sample : arrival_sample + SAMPLE_RATE * 2]
    energy = float(np.sqrt(np.sum(np.square(rir, dtype=np.float64))))
    if not math.isfinite(energy) or energy <= 1e-8:
        raise ValueError("room impulse response has no finite energy")
    rir = rir / energy
    convolved = fftconvolve(signal, rir, mode="full")[: len(signal)]
    peak_after = float(np.max(np.abs(convolved)))
    if peak_after > 1.0:
        convolved = convolved / peak_after
    return convolved.astype(np.float32), arrival_sample


def mix_at_snr(
    foreground: np.ndarray,
    background: np.ndarray,
    phrase_span_s: tuple[float, float],
    snr_db: float,
) -> np.ndarray:
    """Mix a background at measured phrase-active SNR without peak normalization."""
    first = max(0, math.floor(phrase_span_s[0] * SAMPLE_RATE))
    last = min(len(foreground), math.ceil(phrase_span_s[1] * SAMPLE_RATE))
    signal = foreground[first:last]
    signal_rms = float(np.sqrt(np.mean(np.square(signal, dtype=np.float64)) + 1e-12))
    noise_rms = float(np.sqrt(np.mean(np.square(background, dtype=np.float64)) + 1e-12))
    scale = signal_rms / (10.0 ** (float(snr_db) / 20.0) * noise_rms)
    return np.clip(foreground + background * scale, -1.0, 1.0).astype(np.float32)


def _eligible_audio_rows(
    path: Path | None,
    *,
    allowed_source_groups: frozenset[str] | None = None,
    kind: str,
) -> list[dict[str, Any]]:
    if path is None:
        return []
    result = []
    for raw in _rows(path):
        if (
            raw.get("split") != "train"
            or raw.get("training_eligible") is not True
            or raw.get("locked_deployment_anchor") is True
        ):
            continue
        if allowed_source_groups is not None and raw.get("source_group") not in allowed_source_groups:
            continue
        row = dict(raw)
        audio = Path(str(row.get("path", ""))).resolve()
        expected = str(row.get("audio_sha256", row.get("sha256", "")))
        if not audio.is_file() or not expected or sha256_file(audio) != expected:
            raise ValueError(f"{kind} audio hash drift: {audio}")
        row["path"] = str(audio)
        row["audio_sha256"] = expected
        result.append(row)
    identities = [str(row.get("source_id", "")) for row in result]
    if any(not identity for identity in identities) or len(identities) != len(set(identities)):
        raise ValueError(f"{kind} rows require unique source_id values")
    return sorted(result, key=lambda row: str(row["source_id"]))


def _background_rows(path: Path | None) -> list[dict[str, Any]]:
    return _eligible_audio_rows(
        path,
        allowed_source_groups=frozenset(("public_speech", "music", "background_noise")),
        kind="background",
    )


def _rir_rows(path: Path | None) -> list[dict[str, Any]]:
    return _eligible_audio_rows(path, kind="RIR")


def build(
    positive_manifests: Sequence[Path],
    output_dir: Path,
    *,
    source_pronunciation_audit: Path | None = None,
    source_manifest: Path | None = None,
    background_manifest: Path | None = None,
    rir_manifest: Path | None = None,
    negative_manifest: Path | None = None,
    overlay_snr_db: Sequence[float] = DEFAULT_OVERLAY_SNR_DB,
    gain_db_range: tuple[float, float] = DEFAULT_GAIN_DB_RANGE,
    negative_groups: Sequence[str] = (),
    include_inherited_alignments: bool = False,
    states_per_phone: int = 1,
    seed: int = 24103,
    phrase_spec: WakePhraseSpec = HI_FI_KIZZ,
    waveform_output_dir: Path | None = None,
) -> dict[str, Any]:
    if not positive_manifests:
        raise ValueError("at least one aligned positive manifest is required")
    if states_per_phone not in (1, 2, 3):
        raise ValueError("states_per_phone must be 1, 2, or 3")
    state_count = 2 + len(phrase_spec.phones) * states_per_phone
    if (source_pronunciation_audit is None) != (source_manifest is None):
        raise ValueError(
            "source pronunciation audit and source manifest must be provided together"
        )
    pronunciation_accepted = (
        load_pronunciation_acceptances(source_pronunciation_audit, source_manifest)
        if source_pronunciation_audit is not None and source_manifest is not None
        else None
    )
    direct = []
    pronunciation_excluded = []
    seen = set()
    for manifest in positive_manifests:
        for row in _rows(manifest):
            if (
                not include_inherited_alignments
                and isinstance(row.get("alignment"), Mapping)
                and row["alignment"].get("method") == "inherited_ctc_forced_alignment"
            ):
                continue
            validate_aligned_positive(row, phrase_spec)
            source_id = str(row["source_id"])
            if (
                pronunciation_accepted is not None
                and source_id not in pronunciation_accepted
            ):
                pronunciation_excluded.append(source_id)
                continue
            if source_id in seen:
                raise ValueError(f"duplicate aligned source_id: {source_id}")
            seen.add(source_id)
            direct.append(dict(row))
    backgrounds = _background_rows(background_manifest)
    rirs = _rir_rows(rir_manifest)
    snrs = tuple(float(value) for value in overlay_snr_db)
    if any(not math.isfinite(value) for value in snrs):
        raise ValueError("overlay SNRs must be finite")
    if snrs and not backgrounds:
        raise ValueError("training overlays require a background manifest")
    gain_min, gain_max = (float(value) for value in gain_db_range)
    if not math.isfinite(gain_min) or not math.isfinite(gain_max) or gain_min > gain_max:
        raise ValueError("gain range must be finite and ordered")

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_rows: dict[str, list[np.ndarray]] = {
        split: [] for split in ("train", "validation", "test")
    }
    target_rows: dict[str, list[np.ndarray]] = {
        split: [] for split in ("train", "validation", "test")
    }
    ledger: list[dict[str, Any]] = []
    for row in sorted(direct, key=lambda item: (item["split"], item["source_id"])):
        split = str(row["split"])
        source_audio = load_audio(Path(row["path"]))
        expected_source_hash = str(row.get("audio_sha256", ""))
        if not expected_source_hash or sha256_file(Path(row["path"])) != expected_source_hash:
            raise ValueError(f"positive audio hash drift: {row['path']}")
        source_phrase = _span_seconds(row["phrase_span"], "phrase_span")
        variants: list[tuple[str, np.ndarray, float, dict[str, Any] | None]] = []
        clean, translation = place_phrase_context(source_audio, source_phrase)
        variants.append(("clean", clean, translation, None))
        if split == "train":
            for index, snr in enumerate(snrs):
                variant_seed = int.from_bytes(
                    hashlib.sha256(
                        f"{seed}\0{row['source_id']}\0{index}".encode()
                    ).digest()[:8],
                    "little",
                )
                rng = np.random.default_rng(variant_seed)
                # Move only training descendants; timing labels are translated exactly.
                phrase_duration = source_phrase[1] - source_phrase[0]
                low = 0.35 + phrase_duration * 0.5
                high = CONTEXT_SAMPLES / SAMPLE_RATE - 0.25 - phrase_duration * 0.5
                desired = float(rng.uniform(low, max(low + 1e-6, high)))
                foreground, variant_translation = place_phrase_context(
                    source_audio, source_phrase, desired_phrase_center_s=desired
                )
                background_row = backgrounds[int(rng.integers(0, len(backgrounds)))]
                background, background_crop_start, background_tiled = _background_context(
                    load_audio(Path(background_row["path"])), rng
                )
                shifted_phrase = (
                    source_phrase[0] + variant_translation,
                    source_phrase[1] + variant_translation,
                )
                rir_row = rirs[int(rng.integers(0, len(rirs)))] if rirs else None
                rir_arrival_sample = None
                if rir_row is not None:
                    foreground, rir_arrival_sample = apply_room_impulse_response(
                        foreground, load_audio(Path(rir_row["path"]))
                    )
                gain_db = float(rng.uniform(gain_min, gain_max))
                foreground = apply_gain_db(foreground, gain_db)
                mixed = mix_at_snr(foreground, background, shifted_phrase, snr)
                variants.append(
                    (
                        f"overlay-{index}",
                        mixed,
                        variant_translation,
                        {
                            "snr_db": snr,
                            "background_source_id": background_row["source_id"],
                            "background_audio_sha256": background_row.get(
                                "audio_sha256"
                            ),
                            "background_source_group": background_row.get("source_group"),
                            "background_crop_start_sample": background_crop_start,
                            "background_tiled": background_tiled,
                            "foreground_gain_db": gain_db,
                            "rir_source_id": rir_row.get("source_id") if rir_row else None,
                            "rir_audio_sha256": rir_row.get("audio_sha256") if rir_row else None,
                            "rir_stratum": rir_row.get("stratum") if rir_row else None,
                            "rir_arrival_trim_samples": rir_arrival_sample,
                            "seed": variant_seed,
                        },
                    )
                )
        for variant, waveform, shift, augmentation in variants:
            variant_id = f"{row['source_id']}::{variant}"
            translated = _translated_record(row, shift, variant_id, phrase_spec)
            targets = frame_state_targets(
                example_from_mapping(translated, expected_phones=phrase_spec.phones),
                TARGET_FRAME_TIMES,
                states_per_phone=states_per_phone,
            )
            if targets is None or np.min(targets) < 0 or np.max(targets) >= state_count:
                raise ValueError(f"{variant_id}: invalid ordered-state target grid")
            feature_rows[split].append(frontend(waveform))
            target_rows[split].append(targets)
            waveform_path = None
            waveform_sha256 = None
            if waveform_output_dir is not None:
                waveform_path = (
                    waveform_output_dir
                    / split
                    / f"{hashlib.sha256(variant_id.encode()).hexdigest()[:24]}.wav"
                )
                waveform_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(waveform_path, waveform, SAMPLE_RATE, subtype="FLOAT")
                waveform_sha256 = sha256_file(waveform_path)
            ledger.append(
                {
                    "source_id": variant_id,
                    "parent_source_id": row["source_id"],
                    "split": split,
                    "variant": variant,
                    "translation_seconds": shift,
                    "phrase_span": translated["phrase_span"],
                    "phone_spans": translated["phone_spans"],
                    "augmentation": augmentation,
                    "source_audio_sha256": row.get("audio_sha256"),
                    "source_group": row.get("source_group"),
                    "provider": row.get("provider"),
                    "speaker_id": row.get("speaker_id"),
                    "voice_id": row.get("voice_id"),
                    "ancestry_id": row.get("ancestry_id"),
                    **(
                        {
                            "path": str(waveform_path.resolve()),
                            "audio_sha256": waveform_sha256,
                        }
                        if waveform_path
                        else {}
                    ),
                }
            )

    split_counts = {}
    for split in ("train", "validation", "test"):
        if not feature_rows[split]:
            raise ValueError(f"aligned positive split is empty: {split}")
        features = np.stack(feature_rows[split]).astype(np.float32, copy=False)
        targets = np.stack(target_rows[split]).astype(np.int32, copy=False)
        np.save(output_dir / f"positive_features-{split}.npy", features)
        np.save(output_dir / f"positive_targets-{split}.npy", targets)
        split_counts[split] = len(features)

    negative_counts = {}
    if negative_groups:
        materialization_manifest = negative_manifest or background_manifest
        if materialization_manifest is None:
            raise ValueError("negative groups require a negative manifest")
        allowed = set(negative_groups)
        grouped: dict[tuple[str, str], list[np.ndarray]] = {
            (split, group): []
            for split in ("train", "validation", "test")
            for group in allowed
        }
        for row in _rows(materialization_manifest):
            group = str(row.get("source_group", ""))
            split = str(row.get("split", ""))
            if (
                split not in ("train", "validation", "test")
                or int(row.get("label", -1)) != 0
                or group not in allowed
            ):
                continue
            audio = load_audio(Path(row["path"]))
            if len(audio) >= CONTEXT_SAMPLES:
                start = (len(audio) - CONTEXT_SAMPLES) // 2
                context = audio[start : start + CONTEXT_SAMPLES]
            else:
                left = (CONTEXT_SAMPLES - len(audio)) // 2
                context = np.pad(audio, (left, CONTEXT_SAMPLES - len(audio) - left))
            grouped[(split, group)].append(frontend(context))
        observed_groups = {
            group for (_, group), values in grouped.items() if values
        }
        if observed_groups != allowed:
            raise ValueError(
                f"negative groups are absent from every split: {sorted(allowed-observed_groups)}"
            )
        for split in ("train", "validation", "test"):
            negative_counts[split] = {}
            for group in sorted(allowed):
                values_for_group = grouped[(split, group)]
                if not values_for_group:
                    continue
                values = np.stack(values_for_group).astype(np.float32, copy=False)
                np.save(output_dir / f"negative-{split}-{group}.npy", values)
                negative_counts[split][group] = len(values)
            if not negative_counts[split]:
                raise ValueError(f"negative split is empty: {split}")

    report = {
        "schema_version": 3,
        "recipe": "kizz_aligned_teacher_features_v3",
        "positive_manifests": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in positive_manifests
        ],
        "source_pronunciation_audit": (
            {
                "path": str(source_pronunciation_audit.resolve()),
                "sha256": sha256_file(source_pronunciation_audit),
                "source_manifest": str(source_manifest.resolve()),
                "source_manifest_sha256": sha256_file(source_manifest),
                "accepted_aligned_count": len(direct),
                "excluded_aligned_count": len(pronunciation_excluded),
            }
            if source_pronunciation_audit is not None and source_manifest is not None
            else None
        ),
        "background_manifest": (
            {
                "path": str(background_manifest.resolve()),
                "sha256": sha256_file(background_manifest),
            }
            if background_manifest is not None
            else None
        ),
        "rir_manifest": (
            {
                "path": str(rir_manifest.resolve()),
                "sha256": sha256_file(rir_manifest),
                "eligible_count": len(rirs),
            }
            if rir_manifest is not None
            else None
        ),
        "negative_manifest": (
            {
                "path": str(negative_manifest.resolve()),
                "sha256": sha256_file(negative_manifest),
            }
            if negative_manifest is not None
            else None
        ),
        "input_shape": [INPUT_FRAMES, FEATURE_BINS],
        "target_shape": [OUTPUT_FRAMES],
        "states_per_phone": states_per_phone,
        "state_count": state_count,
        "wake_phrase": {
            "phrase_id": phrase_spec.phrase_id,
            "text": phrase_spec.text,
            "phones": list(phrase_spec.phones),
        },
        "target_frame_times_seconds": TARGET_FRAME_TIMES.tolist(),
        "overlay_snr_db": list(snrs),
        "foreground_gain_db_range": [gain_min, gain_max],
        "include_inherited_alignments": include_inherited_alignments,
        "seed": seed,
        **(
            {"waveform_output_dir": str(waveform_output_dir.resolve())}
            if waveform_output_dir
            else {}
        ),
        "positive_counts": split_counts,
        "negative_counts": negative_counts,
        "examples": ledger,
    }
    (output_dir / "feature-provenance.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aligned-positive-manifest", type=Path, action="append", required=True
    )
    parser.add_argument("--source-pronunciation-audit", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--background-manifest", type=Path)
    parser.add_argument("--rir-manifest", type=Path)
    parser.add_argument(
        "--negative-manifest",
        type=Path,
        help=(
            "Split-aware source for materialized negative groups. Defaults to "
            "--background-manifest for backward compatibility."
        ),
    )
    parser.add_argument("--overlay-snr-db", type=float, action="append")
    parser.add_argument("--gain-db-min", type=float, default=DEFAULT_GAIN_DB_RANGE[0])
    parser.add_argument("--gain-db-max", type=float, default=DEFAULT_GAIN_DB_RANGE[1])
    parser.add_argument("--negative-group", action="append", default=[])
    parser.add_argument(
        "--include-inherited-alignments",
        action="store_true",
        help="Include pre-existing augmented descendants; disabled for clean-slate runs.",
    )
    parser.add_argument("--states-per-phone", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument(
        "--phrase-id",
        choices=tuple(sorted(WAKE_PHRASES)),
        default=HI_FI_KIZZ.phrase_id,
    )
    parser.add_argument("--seed", type=int, default=24103)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--waveform-output-dir",
        type=Path,
        help="Optionally persist the exact generated views for teacher scoring.",
    )
    args = parser.parse_args(argv)
    report = build(
        args.aligned_positive_manifest,
        args.output,
        source_pronunciation_audit=args.source_pronunciation_audit,
        source_manifest=args.source_manifest,
        background_manifest=args.background_manifest,
        rir_manifest=args.rir_manifest,
        negative_manifest=args.negative_manifest,
        overlay_snr_db=args.overlay_snr_db or DEFAULT_OVERLAY_SNR_DB,
        gain_db_range=(args.gain_db_min, args.gain_db_max),
        negative_groups=args.negative_group,
        include_inherited_alignments=args.include_inherited_alignments,
        states_per_phone=args.states_per_phone,
        seed=args.seed,
        phrase_spec=get_wake_phrase(args.phrase_id),
        waveform_output_dir=args.waveform_output_dir,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("positive_counts", "negative_counts", "state_count")
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
