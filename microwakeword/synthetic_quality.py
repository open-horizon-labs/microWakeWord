"""Measure and filter generated speech before it becomes training evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
import webrtcvad


@dataclass(frozen=True)
class QualityBounds:
    minimum_speech_ms: int
    maximum_speech_ms: int
    maximum_source_ms: int
    maximum_clipped_fraction: float = 0.001
    minimum_rms_dbfs: float = -50.0

    def to_dict(self) -> dict:
        return asdict(self)


def reference_bounds(
    spans_ms: list[float],
    clip_duration_ms: int,
    maximum_jitter_ms: int,
    minimum_span_ratio: float = 0.75,
    maximum_span_ratio: float = 1.25,
) -> QualityBounds:
    """Derive human-anchored bounds without allowing source truncation."""
    if len(spans_ms) < 3:
        raise ValueError("at least three recorded phrase spans are required")
    if any(span <= 0 for span in spans_ms):
        raise ValueError("recorded phrase spans must be positive")
    if not 0 < minimum_span_ratio <= 1:
        raise ValueError("minimum span ratio must be between zero and one")
    if maximum_span_ratio < 1:
        raise ValueError("maximum span ratio must be at least one")
    maximum_source_ms = clip_duration_ms - maximum_jitter_ms
    if maximum_source_ms <= 0:
        raise ValueError("maximum jitter must be shorter than the training clip")
    low, high = np.quantile(spans_ms, [0.05, 0.95])
    return QualityBounds(
        minimum_speech_ms=max(100, math.floor(low * minimum_span_ratio)),
        maximum_speech_ms=min(maximum_source_ms, math.ceil(high * maximum_span_ratio)),
        maximum_source_ms=maximum_source_ms,
    )


def _pcm16_mono(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, pcm = wavfile.read(path)
    if pcm.dtype != np.int16:
        raise ValueError(f"{path} must be signed-16 PCM")
    if pcm.ndim != 1:
        raise ValueError(f"{path} must be mono")
    return int(sample_rate), pcm


def audio_metrics(path: Path, vad_mode: int = 0, frame_ms: int = 20) -> dict:
    """Return objective source and speech-span measurements for one WAV."""
    sample_rate, source_pcm = _pcm16_mono(path)
    duration_ms = source_pcm.size * 1000 / sample_rate
    normalized = source_pcm.astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(np.square(normalized)))) if normalized.size else 0.0
    rms_dbfs = 20 * math.log10(max(rms, 1e-12))
    clipped_fraction = (
        float(np.mean(np.abs(source_pcm.astype(np.int32)) >= 32760))
        if source_pcm.size
        else 0.0
    )

    if sample_rate != 16000:
        divisor = math.gcd(sample_rate, 16000)
        pcm = resample_poly(normalized, 16000 // divisor, sample_rate // divisor)
        pcm = np.clip(pcm * 32768, -32768, 32767).astype(np.int16)
    else:
        pcm = source_pcm

    frame_samples = 16000 * frame_ms // 1000
    vad = webrtcvad.Vad(vad_mode)
    voiced = [
        index
        for index, start in enumerate(
            range(0, pcm.size - frame_samples + 1, frame_samples)
        )
        if vad.is_speech(pcm[start : start + frame_samples].tobytes(), 16000)
    ]
    if voiced:
        speech_start_ms = voiced[0] * frame_ms
        speech_end_ms = (voiced[-1] + 1) * frame_ms
        speech_span_ms = speech_end_ms - speech_start_ms
    else:
        speech_start_ms = None
        speech_end_ms = None
        speech_span_ms = 0

    return {
        "duration_ms": round(duration_ms, 3),
        "speech_start_ms": speech_start_ms,
        "speech_end_ms": speech_end_ms,
        "speech_span_ms": speech_span_ms,
        "rms_dbfs": round(rms_dbfs, 3),
        "clipped_fraction": round(clipped_fraction, 8),
    }


def quality_reasons(metrics: dict, truth: str, bounds: QualityBounds) -> list[str]:
    """Explain why a generated clip must not enter the feature corpus."""
    reasons = []
    if metrics["duration_ms"] > bounds.maximum_source_ms:
        reasons.append("source_would_be_truncated")
    if metrics["clipped_fraction"] > bounds.maximum_clipped_fraction:
        reasons.append("clipped_audio")
    if metrics["rms_dbfs"] < bounds.minimum_rms_dbfs:
        reasons.append("audio_too_quiet")
    if metrics["speech_span_ms"] <= 0:
        reasons.append("no_speech_detected")
    elif truth == "positive":
        if metrics["speech_span_ms"] < bounds.minimum_speech_ms:
            reasons.append("speech_span_too_short")
        if metrics["speech_span_ms"] > bounds.maximum_speech_ms:
            reasons.append("speech_span_too_long")
    return reasons


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_quality_mask(
    mask_path: Path, recipe_path: Path, generation_manifest_path: Path
) -> dict:
    """Load a mask only when it belongs to this recipe and generated corpus."""
    mask = json.loads(mask_path.read_text())
    if mask.get("schema_version") != 1:
        raise ValueError("unsupported quality-mask schema")
    if mask.get("recipe_sha256") != sha256(recipe_path):
        raise ValueError("quality-mask recipe hash does not match")
    if mask.get("generation_manifest_sha256") != sha256(generation_manifest_path):
        raise ValueError("quality-mask generation manifest hash does not match")
    rejected = mask.get("rejected")
    if not isinstance(rejected, dict):
        raise ValueError("quality mask requires a rejected-path map")
    return mask
