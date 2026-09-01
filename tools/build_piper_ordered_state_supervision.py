#!/usr/bin/env python3
"""Build measured Piper frame supervision for the Kizz ordered-state model.

Piper's duration predictor exposes one measured sample count per model-input
token.  This tool aggregates that trace into the fixed canonical phone sequence,
places the phrase deterministically inside the model's emitted two-second
timeline, extracts the product microfrontend features, and delegates final
alignment validation to ``build_ordered_state_frame_supervision``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from microwakeword.audio.audio_utils import generate_features_for_clip
from microwakeword.ordered_state import KIZZ_PHONES

try:
    from tools.build_ordered_state_frame_supervision import build_frame_supervision
except ModuleNotFoundError:  # pragma: no cover - direct script invocation
    from build_ordered_state_frame_supervision import build_frame_supervision

SAMPLE_RATE = 16000
FEATURE_STEP_SECONDS = 0.01
FRONTEND_WINDOW_SECONDS = 0.03
TARGET_STEP_SECONDS = 0.03
RECEPTIVE_FIELD_SECONDS = 0.67
FIRST_TARGET_CENTER_SECONDS = RECEPTIVE_FIELD_SECONDS - (FRONTEND_WINDOW_SECONDS / 2)
DEFAULT_FEATURE_FRAMES = 260
DEFAULT_TARGET_FRAMES = 66
ATOMIC_KIZZ_PHONES = ("h", "a", "ɪ", "f", "a", "ɪ", "k", "ɪ", "z")
PHONE_GROUPS = ((0,), (1, 2), (3,), (4, 5), (6,), (7,), (8,))


class PhraseDoesNotFitError(ValueError):
    """A measured synthesis is longer than the fixed emitted-state timeline."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("Piper metadata is empty")
    return records


def _finite(value: Any, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _stress_start(tokens: list[Mapping[str, Any]], token_index: int) -> float | None:
    """Return a directly preceding standalone stress-token start, if present."""
    for index in range(token_index - 1, -1, -1):
        token = tokens[index]
        if token.get("kind") == "phoneme":
            base = str(token.get("phoneme_base") or "")
            if token.get("stress") and not base:
                return _finite(token["start_s"], "stress start")
            if base.strip():
                return None
    return None


def canonical_phone_spans(timing: Mapping[str, Any]) -> list[dict[str, float | str]]:
    """Aggregate Piper's measured token spans into canonical Kizz phones.

    The trace must resolve exactly to ``h a ɪ f a ɪ k ɪ z`` after explicit
    stress, separator, boundary, and whitespace tokens are removed.  Diphthongs
    are grouped from their measured component spans.  Inter-phone gaps are split
    at the midpoint of adjacent measured spans so frame targets cover one
    continuous phrase without inventing equal phone durations.
    """
    tokens = timing.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        raise ValueError("Piper timing requires a non-empty token trace")
    atomic = []
    for token_index, token in enumerate(tokens):
        if not isinstance(token, Mapping) or token.get("kind") != "phoneme":
            continue
        base = str(token.get("phoneme_base") or "")
        if not base or base.isspace():
            continue
        start = _finite(token.get("start_s"), "phone start")
        end = _finite(token.get("end_s"), "phone end")
        if end <= start:
            raise ValueError("Piper phone timing must have positive duration")
        atomic.append((token_index, base, start, end))
    if tuple(item[1] for item in atomic) != ATOMIC_KIZZ_PHONES:
        raise ValueError("Piper trace does not exactly match canonical Hi-Fi Kizz")

    measured_groups = []
    for group in PHONE_GROUPS:
        first = atomic[group[0]]
        last = atomic[group[-1]]
        start = _stress_start(tokens, first[0])
        measured_groups.append(
            {
                "start_s": first[2] if start is None else start,
                "end_s": last[3],
            }
        )
    boundaries = [
        (measured_groups[index]["end_s"] + measured_groups[index + 1]["start_s"]) / 2.0
        for index in range(len(measured_groups) - 1)
    ]
    spans = []
    for index, phone in enumerate(KIZZ_PHONES):
        start = measured_groups[0]["start_s"] if index == 0 else boundaries[index - 1]
        end = (
            measured_groups[-1]["end_s"]
            if index == len(KIZZ_PHONES) - 1
            else boundaries[index]
        )
        if end <= start:
            raise ValueError("aggregated phone timing must be ordered")
        spans.append({"phone": phone, "start_s": start, "end_s": end})
    return spans


def read_resampled_wav(path: Path) -> np.ndarray:
    sample_rate, samples = wavfile.read(path)
    if samples.ndim != 1 or samples.dtype != np.int16:
        raise ValueError(f"{path}: expected mono 16-bit PCM")
    if sample_rate <= 0:
        raise ValueError(f"{path}: invalid sample rate")
    if sample_rate != SAMPLE_RATE:
        divisor = math.gcd(int(sample_rate), SAMPLE_RATE)
        samples = resample_poly(
            samples.astype(np.float32),
            SAMPLE_RATE // divisor,
            int(sample_rate) // divisor,
        )
        samples = np.clip(np.rint(samples), -32768, 32767).astype(np.int16)
    return samples


def deterministic_offset(
    source_id: str,
    seed: int,
    source_duration: float,
    phrase_start: float,
    phrase_end: float,
    feature_frames: int,
    target_frames: int,
) -> float:
    """Place the entire measured phrase inside the emitted-state timeline."""
    audio_duration = FRONTEND_WINDOW_SECONDS + feature_frames * FEATURE_STEP_SECONDS
    last_target = (
        FIRST_TARGET_CENTER_SECONDS + (target_frames - 1) * TARGET_STEP_SECONDS
    )
    minimum = max(0.0, FIRST_TARGET_CENTER_SECONDS - phrase_start)
    maximum = min(audio_duration - source_duration, last_target - phrase_end)
    if maximum < minimum:
        raise PhraseDoesNotFitError(
            "synthesized phrase cannot fit the ordered-state timeline"
        )
    digest = hashlib.sha256(f"{seed}:{source_id}".encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    return minimum + fraction * (maximum - minimum)


def prepare_record(
    metadata: Mapping[str, Any],
    metadata_path: Path,
    audio_root: Path,
    examples_dir: Path,
    split: str,
    seed: int,
    feature_frames: int,
    target_frames: int,
) -> dict[str, Any]:
    relative_file = metadata.get("file")
    if not isinstance(relative_file, str) or not relative_file:
        raise ValueError("Piper metadata record requires file")
    relative_path = Path(relative_file)
    if relative_path.is_absolute():
        raise ValueError("Piper metadata file must be relative to audio_root")
    resolved_audio_root = audio_root.resolve()
    source_path = (resolved_audio_root / relative_path).resolve()
    try:
        source_path.relative_to(resolved_audio_root)
    except ValueError as error:
        raise ValueError("Piper metadata file escapes audio_root") from error
    if not source_path.is_file():
        raise ValueError(f"missing Piper WAV: {source_path}")
    timing = metadata.get("phoneme_timing")
    if not isinstance(timing, Mapping):
        raise ValueError("Piper record does not contain measured phoneme_timing")
    spans = canonical_phone_spans(timing)
    samples = read_resampled_wav(source_path)
    source_duration = len(samples) / SAMPLE_RATE
    source_id = f"piper-{split}:{relative_file}"
    offset = deterministic_offset(
        source_id,
        seed,
        source_duration,
        float(spans[0]["start_s"]),
        float(spans[-1]["end_s"]),
        feature_frames,
        target_frames,
    )
    output_samples = int(
        round(
            (FRONTEND_WINDOW_SECONDS + feature_frames * FEATURE_STEP_SECONDS)
            * SAMPLE_RATE
        )
    )
    start_sample = int(round(offset * SAMPLE_RATE))
    if start_sample + len(samples) > output_samples:
        raise ValueError("rounded Piper placement exceeds the output window")
    placed = np.zeros(output_samples, dtype=np.int16)
    placed[start_sample : start_sample + len(samples)] = samples
    features = generate_features_for_clip(placed, step_ms=10, use_c=True)
    if features.shape != (feature_frames, 40):
        raise ValueError(
            f"microfrontend produced {features.shape}, expected {(feature_frames, 40)}"
        )
    examples_dir.mkdir(parents=True, exist_ok=True)
    relative_digest = hashlib.sha256(relative_path.as_posix().encode()).hexdigest()[:16]
    feature_path = examples_dir / f"{relative_path.stem}-{relative_digest}.npy"
    if feature_path.exists():
        raise ValueError(f"duplicate Piper feature destination: {feature_path}")
    np.save(feature_path, features.astype(np.float32, copy=False))
    shifted_spans = [
        {
            "phone": span["phone"],
            "start_s": float(span["start_s"]) + start_sample / SAMPLE_RATE,
            "end_s": float(span["end_s"]) + start_sample / SAMPLE_RATE,
        }
        for span in spans
    ]
    speakers = (int(metadata["speaker_1"]), int(metadata["speaker_2"]))
    target_times = [
        FIRST_TARGET_CENTER_SECONDS + index * TARGET_STEP_SECONDS
        for index in range(target_frames)
    ]
    return {
        "source_id": source_id,
        "source_group": f"piper:{speakers[0]}+{speakers[1]}",
        "split": split,
        "truth": True,
        "text": metadata.get("text"),
        "duration_s": output_samples / SAMPLE_RATE,
        "features_path": str(feature_path),
        "feature_frame_step_seconds": FEATURE_STEP_SECONDS,
        "target_frame_times_s": target_times,
        "alignment": {
            "method": "synthesizer",
            "timing_source": str(metadata_path),
            "timing_record": {
                "file": relative_file,
                "source_wav_sha256": sha256_file(source_path),
                "measured_token_samples": True,
                "placement_start_sample": start_sample,
            },
        },
        "phrase_span": {
            "start_s": shifted_spans[0]["start_s"],
            "end_s": shifted_spans[-1]["end_s"],
        },
        "phone_spans": shifted_spans,
    }


def build_piper_supervision(
    metadata_path: Path,
    audio_root: Path,
    output: Path,
    *,
    split: str = "train",
    seed: int = 241,
    feature_frames: int = DEFAULT_FEATURE_FRAMES,
    target_frames: int = DEFAULT_TARGET_FRAMES,
) -> dict[str, Any]:
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    if feature_frames < 1 or target_frames < 1:
        raise ValueError("frame counts must be positive")
    records = []
    rejected = []
    for metadata in load_jsonl(metadata_path):
        try:
            records.append(
                prepare_record(
                    metadata,
                    metadata_path,
                    audio_root,
                    output / "examples",
                    split,
                    seed,
                    feature_frames,
                    target_frames,
                )
            )
        except PhraseDoesNotFitError as error:
            rejected.append(
                {
                    "file": metadata.get("file"),
                    "reason": str(error),
                    "source_wav_sha256": sha256_file(audio_root / metadata["file"]),
                }
            )
    if not records:
        raise ValueError("no Piper examples fit the ordered-state timeline")
    manifest = {
        "schema_version": 1,
        "source_metadata": str(metadata_path),
        "source_metadata_sha256": sha256_file(metadata_path),
        "seed": seed,
        "rejected": rejected,
        "records": records,
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "frame-supervision-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = build_frame_supervision(
        records,
        output,
        output / "arrays",
        expected_feature_frames=feature_frames,
        expected_target_frames=target_frames,
        allow_measured_synthesizer_timing=True,
    )
    summary.update(
        {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "arrays": str(output / "arrays"),
            "rejected_examples": len(rejected),
        }
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="train"
    )
    parser.add_argument("--seed", type=int, default=241)
    parser.add_argument("--feature-frames", type=int, default=DEFAULT_FEATURE_FRAMES)
    parser.add_argument("--target-frames", type=int, default=DEFAULT_TARGET_FRAMES)
    args = parser.parse_args()
    summary = build_piper_supervision(
        args.metadata,
        args.audio_root,
        args.output,
        split=args.split,
        seed=args.seed,
        feature_frames=args.feature_frames,
        target_frames=args.target_frames,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
