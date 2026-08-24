# coding=utf-8
# Copyright 2026 Open Horizon Labs.
#
# Licensed under the Apache License, Version 2.0.

"""A waveform teacher backed by a pretrained speech representation model.

This is the D experiment.  It is deliberately separate from the existing
micro-speech/state teacher: the backbone sees 16 kHz waveform samples and the
head learns one stream-level Kizz event score.  The model is offline-only;
student/firmware code must never import this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


TARGET_SAMPLE_RATE = 16_000
CONTEXT_SAMPLES = 32_000


@dataclass(frozen=True)
class WaveformExample:
    """One deterministic audio source and its binary event label."""

    path: Path
    label: int
    source_id: str


def list_audio_files(root: Path) -> tuple[Path, ...]:
    """Return stable, non-empty WAV/FLAC paths below ``root``."""
    if not root.is_dir():
        raise ValueError(f"audio root does not exist: {root}")
    paths = tuple(sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".wav", ".flac"}
    ))
    if not paths:
        raise ValueError(f"audio root contains no WAV/FLAC files: {root}")
    return paths


def load_waveform(path: Path, *, sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Read mono float audio and resample with linear interpolation."""
    try:
        import soundfile as sf
    except ImportError as error:  # pragma: no cover - exercised by CLI setup
        raise RuntimeError(
            "D teacher audio loading requires soundfile; install requirements-kizz-teacher.txt"
        ) from error
    values, source_rate = sf.read(path, always_2d=False, dtype="float32")
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 2:
        values = np.mean(values, axis=1, dtype=np.float32)
    if values.ndim != 1 or not len(values):
        raise ValueError(f"audio must be a non-empty mono/stereo waveform: {path}")
    if int(source_rate) == int(sample_rate):
        return values
    target_length = max(1, round(len(values) * sample_rate / source_rate))
    source_x = np.linspace(0.0, 1.0, num=len(values), endpoint=False)
    target_x = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
    return np.interp(target_x, source_x, values).astype(np.float32)


def fit_context(
    waveform: np.ndarray,
    *,
    context_samples: int = CONTEXT_SAMPLES,
    start: int | None = None,
) -> np.ndarray:
    """Crop or zero-pad an audio item to the fixed teacher context."""
    values = np.asarray(waveform, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("waveform must be one-dimensional")
    if len(values) >= context_samples:
        offset = 0 if start is None else max(0, min(int(start), len(values) - context_samples))
        return values[offset : offset + context_samples].copy()
    result = np.zeros(context_samples, dtype=np.float32)
    offset = 0 if start is None else max(0, min(int(start), context_samples - len(values)))
    result[offset : offset + len(values)] = values
    return result


def mix_positive_context(
    positive: np.ndarray,
    background: np.ndarray,
    *,
    rng: np.random.Generator,
    context_samples: int = CONTEXT_SAMPLES,
) -> np.ndarray:
    """Place a positive clip into a same-length background context.

    This prevents the teacher from using zero-padding or clip duration as the
    positive shortcut.  The positive is scaled conservatively to preserve its
    acoustic identity while allowing realistic masking.
    """
    output = fit_context(background, context_samples=context_samples)
    positive = np.asarray(positive, dtype=np.float32)
    if len(positive) > context_samples:
        start = int(rng.integers(0, len(positive) - context_samples + 1))
        positive = positive[start : start + context_samples]
    max_start = context_samples - len(positive)
    start = int(rng.integers(0, max_start + 1)) if max_start else 0
    gain = float(rng.uniform(0.75, 1.0))
    output[start : start + len(positive)] += gain * positive
    peak = float(np.max(np.abs(output)))
    if peak > 0.99:
        output *= 0.99 / peak
    return output.astype(np.float32)


def build_model(backbone_name: str = "microsoft/wavlm-base-plus", *, unfreeze_last_n: int = 0):
    """Build the offline teacher and return ``(model, hidden_size)``.

    Imports are lazy so the base microWakeWord package remains TensorFlow-only.
    """
    try:
        import torch
        from torch import nn
        from transformers import AutoModel
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "D teacher requires torch, transformers, and soundfile; "
            "install requirements-kizz-teacher.txt"
        ) from error

    backbone = AutoModel.from_pretrained(backbone_name)
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    if unfreeze_last_n:
        layers = getattr(backbone.encoder, "layers", None)
        if layers is None:
            raise ValueError("selected backbone has no encoder.layers to unfreeze")
        for layer in list(layers)[-unfreeze_last_n:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    hidden_size = int(backbone.config.hidden_size)

    class KizzPretrainedTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.representation_norm = nn.LayerNorm(hidden_size)
            self.temporal = nn.Sequential(
                nn.Conv1d(hidden_size, 192, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv1d(192, 96, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv1d(96, 1, kernel_size=1),
            )

        def forward(self, input_values):
            hidden = self.backbone(input_values).last_hidden_state
            hidden = self.representation_norm(hidden)
            frame_logits = self.temporal(hidden.transpose(1, 2)).squeeze(1)
            # Multiple-instance learning: the phrase may occur anywhere in
            # the context, while every frame in a negative must reject it.
            score = torch.logsumexp(frame_logits, dim=1) - np.log(frame_logits.shape[1])
            return score, frame_logits

    return KizzPretrainedTeacher(), hidden_size


__all__ = [
    "CONTEXT_SAMPLES",
    "TARGET_SAMPLE_RATE",
    "WaveformExample",
    "build_model",
    "fit_context",
    "list_audio_files",
    "load_waveform",
    "mix_positive_context",
]
