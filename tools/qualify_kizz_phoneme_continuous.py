#!/usr/bin/env python3
"""Qualify the pinned Kizz Control phoneme teacher on raw negative audio.

Archives are read directly from gzip'd tar members; extracted locked manifests
are read with bounded soundfile blocks.
Each WAV is treated as one continuous negative stream.  Model inference happens
once per overlapping chunk, while every chunk's windows are scored with the
same CTC primitives used by ``qualify_kizz_phoneme_teacher.py``.  The report is
bound to the archive, teacher revision, frozen threshold, and collision margin.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import tarfile
import wave
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from microwakeword.kizz_continuous_evaluation import detect_events, poisson_upper_95
from microwakeword.kizz_phoneme_teacher import (
    MODEL_ID,
    MODEL_REVISION,
    TARGET_SAMPLE_RATE,
    WindowScore,
    ctc_log_probability_batch,
    load_hf_teacher,
    resolve_hf_weights_path,
    resolve_phone_ids,
    score_window,
    sha256_file,
    sha256_text,
)
from microwakeword.wake_phrase import KIZZ_CONTROL, get_wake_phrase

DEFAULT_WINDOW_LENGTHS = (0.56, 0.68, 0.80, 0.96, 1.16, 1.40, 1.60)
DEFAULT_HOP_SECONDS = 0.06
DEFAULT_CHUNK_SECONDS = 30.0
DEFAULT_REFRACTORY_SECONDS = 1.0
DEFAULT_MIN_EXPOSURE_HOURS = 100.0
DEFAULT_MAX_FAPH_UPPER_95 = 0.10
REQUIRED_PHRASE_ID = "kizz-control"
REQUIRED_TEACHER_HASHES = ("weights_sha256", "config_sha256", "tokenizer_vocab_sha256")
SUPPORTED_SOURCE_SAMPLE_RATES = (16_000, 44_100)


def _load_teacher_qualification(path: Path) -> tuple[dict, str]:
    """Load and validate the immutable teacher operating-point contract."""
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or payload.get("qualified") is not True:
        raise ValueError("teacher qualification must be a qualified report")
    phrase_id = payload.get("phones", {}).get("phrase_id") or payload.get("phrase", {}).get("phrase_id")
    if phrase_id != REQUIRED_PHRASE_ID:
        raise ValueError(f"teacher qualification phrase must be {REQUIRED_PHRASE_ID!r}")
    scoring = payload.get("scoring", {})
    threshold = scoring.get("threshold")
    beta = scoring.get("collision_margin_beta")
    if not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)):
        raise ValueError("teacher qualification has no finite frozen threshold")
    if not isinstance(beta, (int, float)) or not math.isfinite(float(beta)):
        raise ValueError("teacher qualification has no finite frozen collision margin")
    model = payload.get("model")
    if not isinstance(model, dict):
        raise ValueError("teacher qualification has no model contract")
    missing = [key for key in REQUIRED_TEACHER_HASHES if not model.get(key)]
    if missing:
        raise ValueError(f"teacher qualification is missing exact teacher hashes: {missing}")
    if not model.get("id") or not model.get("revision"):
        raise ValueError("teacher qualification is missing model id or revision")
    return payload, sha256_file(path)


def _normalize_category(value: str) -> str:
    if value == "connected_speech":
        return "speech"
    if value in {"speech", "music", "noise"}:
        return value
    raise ValueError(f"unsupported negative-audio category: {value!r}")


def _category(member_name: str) -> str:
    parts = Path(member_name).parts
    for value in ("speech", "music", "noise"):
        if value in parts:
            return value
    raise ValueError(f"cannot classify MUSAN member by speech/music/noise path: {member_name}")


def _manifest_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = next(
            (payload[key] for key in ("examples", "files", "entries", "records") if isinstance(payload.get(key), list)),
            None,
        )
    else:
        rows = None
    if rows is None or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("locked manifest must contain a non-empty list of file records")
    required = {"path", "sha256", "duration_seconds", "category"}
    if any(not required.issubset(row) for row in rows):
        raise ValueError("locked manifest records require path, sha256, duration_seconds, and category")
    return [dict(row) for row in rows]


def _audio_duration(path: Path) -> float:
    try:
        import soundfile as sf
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("soundfile is required for extracted WAV/FLAC manifests") from error
    info = sf.info(str(path))
    if info.samplerate not in SUPPORTED_SOURCE_SAMPLE_RATES:
        raise ValueError(f"{path}: unsupported sample rate {info.samplerate} Hz")
    if info.channels != 1:
        raise ValueError(f"{path}: audio must be mono")
    return info.frames / info.samplerate


def _validate_locked_manifest(path: Path) -> list[dict]:
    rows = _manifest_rows(path)
    seen: set[Path] = set()
    validated = []
    for row in rows:
        audio_path = Path(str(row["path"]))
        if not audio_path.is_absolute():
            raise ValueError(f"locked manifest path is not absolute: {audio_path}")
        audio_path = audio_path.resolve()
        if audio_path in seen:
            raise ValueError(f"locked manifest contains duplicate path: {audio_path}")
        seen.add(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        expected_hash = str(row["sha256"]).lower()
        actual_hash = sha256_file(audio_path)
        if actual_hash != expected_hash:
            raise ValueError(f"manifest hash drift for {audio_path}: expected {expected_hash}, got {actual_hash}")
        expected_duration = float(row["duration_seconds"])
        actual_duration = _audio_duration(audio_path)
        if not math.isfinite(expected_duration) or abs(actual_duration - expected_duration) > 1e-6:
            raise ValueError(
                f"manifest duration drift for {audio_path}: expected {expected_duration}, got {actual_duration}"
            )
        validated.append(
            {
                "path": audio_path,
                "sha256": actual_hash,
                "duration_seconds": actual_duration,
                "category": _normalize_category(str(row["category"])),
            }
        )
    return validated


def _read_wav_chunks(fileobj, *, chunk_frames: int) -> Iterable[tuple[np.ndarray, int]]:
    """Yield float32 mono chunks and their exact frame counts from a WAV stream."""
    with wave.open(fileobj, "rb") as wav:
        if wav.getframerate() != TARGET_SAMPLE_RATE:
            raise ValueError(f"WAV sample rate must be {TARGET_SAMPLE_RATE} Hz")
        if wav.getnchannels() != 1:
            raise ValueError("WAV must be mono")
        if wav.getsampwidth() != 2:
            raise ValueError("WAV must be signed 16-bit PCM")
        while True:
            raw = wav.readframes(chunk_frames)
            if not raw:
                break
            values = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            yield values, len(values)


def _rolling_chunks(read_frames: Callable[[int], np.ndarray], *, chunk_frames: int, overlap_frames: int):
    next_start = 0
    retained = np.empty(0, dtype=np.float32)
    while True:
        incoming = np.asarray(read_frames(chunk_frames - len(retained)), dtype=np.float32)
        if not len(incoming):
            break
        chunk = np.concatenate((retained, incoming))
        start = next_start
        yield start, chunk
        consumed = max(0, len(chunk) - overlap_frames)
        next_start = start + consumed
        retained = chunk[consumed:]


def _overlap_chunks(audio_file, *, chunk_frames: int, overlap_frames: int, soundfile_mode: bool = False):
    """Yield ``(offset, waveform)`` while retaining exact frame offsets."""
    if soundfile_mode:
        try:
            import soundfile as sf
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("soundfile is required for extracted WAV/FLAC manifests") from error
        with sf.SoundFile(audio_file) as audio:
            if audio.samplerate not in SUPPORTED_SOURCE_SAMPLE_RATES:
                raise ValueError(f"unsupported audio sample rate {audio.samplerate} Hz")
            if audio.channels != 1:
                raise ValueError("audio must be mono")
            if audio.samplerate == TARGET_SAMPLE_RATE:
                read_frames = lambda count: audio.read(count, dtype="float32", always_2d=False)
            else:
                try:
                    from scipy.signal import resample_poly
                except ImportError as error:  # pragma: no cover - environment dependent
                    raise RuntimeError("scipy is required to resample 44.1 kHz manifest audio") from error

                def read_frames(count):
                    source_count = max(1, round(count * audio.samplerate / TARGET_SAMPLE_RATE))
                    values = audio.read(source_count, dtype="float32", always_2d=False)
                    if not len(values):
                        return values
                    converted = resample_poly(values, TARGET_SAMPLE_RATE, audio.samplerate)
                    expected = max(1, round(len(values) * TARGET_SAMPLE_RATE / audio.samplerate))
                    return np.asarray(converted[:expected], dtype=np.float32)

            yield from _rolling_chunks(
                read_frames,
                chunk_frames=chunk_frames,
                overlap_frames=overlap_frames,
            )
        return

    with wave.open(audio_file, "rb") as wav:
        if wav.getframerate() != TARGET_SAMPLE_RATE:
            raise ValueError(f"WAV sample rate must be {TARGET_SAMPLE_RATE} Hz")
        if wav.getnchannels() != 1:
            raise ValueError("WAV must be mono")
        if wav.getsampwidth() != 2:
            raise ValueError("WAV must be signed 16-bit PCM")
        yield from _rolling_chunks(
            lambda count: np.frombuffer(wav.readframes(count), dtype="<i2").astype(np.float32) / 32768.0,
            chunk_frames=chunk_frames,
            overlap_frames=overlap_frames,
        )


def _infer_chunk(model, processor, waveform: np.ndarray, device):
    import torch

    inputs = processor(waveform, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")
    with torch.inference_mode():
        return model(input_values=inputs.input_values.to(device)).logits[0].detach().cpu().numpy()


def _scan_chunk(
    log_probs: np.ndarray,
    *,
    canonical_tokens: Sequence[int],
    collision_tokens: Sequence[Sequence[int]],
    blank_id: int,
    window_lengths: Sequence[int],
    hop: int,
    beta: float,
    minimum_score: float,
) -> dict[int, float]:
    """Return one margin-gated score per frame start, reusing CTC semantics.

    Canonical fits are evaluated for every window.  Collision paths are only
    evaluated where the canonical fit can cross the already-frozen validation
    threshold.  Below-threshold windows remain explicit ``-inf`` sentinels so
    event boundaries are identical to an exhaustive scan.
    """
    values = np.asarray(log_probs, dtype=np.float64)
    if values.ndim != 2 or not len(values):
        raise ValueError("model logits must be a non-empty [frames, vocabulary] array")
    if not math.isfinite(minimum_score):
        raise ValueError("minimum_score must be the finite frozen threshold")
    canonical = tuple(int(value) for value in canonical_tokens)
    collisions = tuple(tuple(int(value) for value in path) for path in collision_tokens)
    required_ids = tuple(sorted({int(blank_id), *canonical, *(value for path in collisions for value in path)}))
    if not required_ids or required_ids[0] < 0 or required_ids[-1] >= values.shape[1]:
        raise ValueError("CTC token ID is outside the model vocabulary")
    remap = {token_id: compact_id for compact_id, token_id in enumerate(required_ids)}
    compact = values[:, required_ids]
    compact_blank = remap[int(blank_id)]
    compact_canonical = tuple(remap[value] for value in canonical)
    compact_collisions = tuple(tuple(remap[value] for value in path) for path in collisions)
    scores: dict[int, float] = {}
    seen_lengths: set[int] = set()
    for requested_length in window_lengths:
        length = min(int(requested_length), len(values))
        if length <= 0 or length in seen_lengths:
            continue
        seen_lengths.add(length)
        starts = list(range(0, len(values) - length + 1, hop))
        tail = len(values) - length
        if not starts or starts[-1] != tail:
            starts.append(tail)
        start_indexes = np.asarray(starts, dtype=np.int64)
        windows = compact[
            start_indexes[:, None] + np.arange(length, dtype=np.int64)[None, :]
        ]
        canonical_fits = ctc_log_probability_batch(
            windows, compact_canonical, blank_id=compact_blank
        ) / max(1, len(compact_canonical))
        gated = np.full(len(starts), -math.inf, dtype=np.float64)
        candidates = np.flatnonzero(canonical_fits >= minimum_score)
        if len(candidates):
            if compact_collisions:
                collision_fits = np.stack(
                    [
                        ctc_log_probability_batch(
                            windows[candidates], path, blank_id=compact_blank
                        )
                        / max(1, len(path))
                        for path in compact_collisions
                    ],
                    axis=1,
                )
                margins = canonical_fits[candidates] - np.max(
                    collision_fits, axis=1
                )
            else:
                margins = np.full(len(candidates), math.inf)
            accepted = candidates[margins >= beta]
            gated[accepted] = canonical_fits[accepted]
        for start, score in zip(starts, gated, strict=True):
            scores[start] = max(scores.get(start, -math.inf), float(score))
    return scores


def _score_member(
    member_name: str,
    category: str,
    fileobj,
    *,
    model,
    processor,
    device,
    token_ids: dict[str, tuple[int, ...]],
    blank_id: int,
    window_lengths_seconds: Sequence[float],
    hop_seconds: float,
    beta: float,
    threshold: float,
    chunk_seconds: float,
    refractory_seconds: float,
    infer_logits: Callable | None = None,
    soundfile_mode: bool = False,
) -> dict:
    chunk_frames = max(1, round(chunk_seconds * TARGET_SAMPLE_RATE))
    max_window_seconds = max(window_lengths_seconds)
    # The overlap is measured in audio frames.  A chunk must still contain a
    # complete longest window after the retained overlap is accounted for.
    overlap_frames = min(
        chunk_frames - 1,
        max(1, round(max_window_seconds * TARGET_SAMPLE_RATE)),
    )
    # Microsecond keys keep chunk-local frame-rate rounding from changing the
    # global timestamp used for duplicate-window collapse.
    all_scores: dict[int, float] = {}
    total_frames = 0
    frame_rate: float | None = None
    infer = infer_logits or (lambda waveform: _infer_chunk(model, processor, waveform, device))
    for offset, waveform in _overlap_chunks(
        fileobj,
        chunk_frames=chunk_frames,
        overlap_frames=overlap_frames,
        soundfile_mode=soundfile_mode,
    ):
        total_frames = max(total_frames, offset + len(waveform))
        logits = np.asarray(infer(waveform), dtype=np.float64)
        if logits.ndim != 2 or not len(logits):
            raise ValueError("model inference must return [frames, vocabulary] logits")
        frame_rate = len(logits) / (len(waveform) / TARGET_SAMPLE_RATE)
        lengths = tuple(max(1, round(seconds * frame_rate)) for seconds in window_lengths_seconds)
        hop = max(1, round(hop_seconds * frame_rate))
        log_probs = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
        for start, score in _scan_chunk(
            log_probs,
            canonical_tokens=token_ids["canonical"],
            collision_tokens=token_ids["collisions"],
            blank_id=blank_id,
            window_lengths=lengths,
            hop=hop,
            beta=beta,
            minimum_score=threshold,
        ).items():
            timestamp = offset / TARGET_SAMPLE_RATE + start / frame_rate
            timestamp_key = round(timestamp * 1_000_000)
            all_scores[timestamp_key] = max(all_scores.get(timestamp_key, -math.inf), score)
    timestamps = tuple(key / 1_000_000.0 for key in sorted(all_scores))
    scores = tuple(all_scores[key] for key in sorted(all_scores))
    return {
        "member": member_name,
        "category": category,
        "duration_seconds": total_frames / TARGET_SAMPLE_RATE,
        "frame_rate": frame_rate,
        "timestamps": timestamps,
        "scores": scores,
    }


def _evaluate_member(scored: dict, *, threshold: float, refractory_seconds: float) -> dict:
    events = detect_events(
        scored["timestamps"], scored["scores"], threshold, refractory_seconds=refractory_seconds
    )
    return {
        "member": scored["member"],
        "category": scored["category"],
        "duration_seconds": scored["duration_seconds"],
        "events": [
            {
                "start_seconds": event.start_seconds,
                "end_seconds": event.end_seconds,
                "peak_score": event.peak_score,
                "peak_timestamp_seconds": event.peak_timestamp_seconds,
            }
            for event in events
        ],
    }


def qualify_archive(
    archive: Path | None,
    *,
    threshold: float,
    beta: float,
    window_lengths_seconds: Sequence[float] = DEFAULT_WINDOW_LENGTHS,
    hop_seconds: float = DEFAULT_HOP_SECONDS,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    refractory_seconds: float = DEFAULT_REFRACTORY_SECONDS,
    min_exposure_hours: float = DEFAULT_MIN_EXPOSURE_HOURS,
    max_faph_upper_95: float = DEFAULT_MAX_FAPH_UPPER_95,
    model=None,
    processor=None,
    device=None,
    token_ids: dict[str, tuple[int, ...]] | None = None,
    blank_id: int = 0,
    infer_logits: Callable | None = None,
    model_id: str = MODEL_ID,
    revision: str = MODEL_REVISION,
    phrase_id: str = KIZZ_CONTROL.phrase_id,
    manifest: Path | None = None,
    teacher_qualification: dict | None = None,
    teacher_qualification_sha256: str | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict:
    if teacher_qualification is not None:
        if teacher_qualification.get("qualified") is not True:
            raise ValueError("teacher qualification must be qualified")
        qualification_phrase = teacher_qualification.get("phones", {}).get("phrase_id") or teacher_qualification.get("phrase", {}).get("phrase_id")
        if qualification_phrase != REQUIRED_PHRASE_ID:
            raise ValueError(f"teacher qualification phrase must be {REQUIRED_PHRASE_ID!r}")
        qualification_model = teacher_qualification["model"]
        threshold = float(teacher_qualification["scoring"]["threshold"])
        beta = float(teacher_qualification["scoring"]["collision_margin_beta"])
        model_id = qualification_model["id"]
        revision = qualification_model["revision"]
        if not teacher_qualification_sha256:
            raise ValueError("teacher qualification SHA is required")
    if not math.isfinite(threshold) or not math.isfinite(beta):
        raise ValueError("threshold and beta must be finite")
    if not window_lengths_seconds or any(value <= 0 for value in window_lengths_seconds):
        raise ValueError("window lengths must be positive")
    if hop_seconds <= 0 or chunk_seconds <= 0 or refractory_seconds < 0:
        raise ValueError("invalid chunk, hop, or refractory setting")
    if chunk_seconds <= max(window_lengths_seconds):
        raise ValueError("chunk_seconds must exceed the longest scoring window")
    if (archive is None) == (manifest is None):
        raise ValueError("provide exactly one of archive or manifest")
    if archive is not None and not archive.is_file():
        raise FileNotFoundError(archive)
    locked_rows = _validate_locked_manifest(manifest) if manifest is not None else None
    if infer_logits is None and (model is None or processor is None or device is None):
        model, processor, _, device = load_hf_teacher(
            model_id, revision=revision, device=device or "cpu", local_files_only=True
        )
    if token_ids is None:
        raise ValueError("token_ids are required; resolve them from the pinned tokenizer")

    results: list[dict] = []
    category_counts = defaultdict(lambda: {"files": 0, "events": 0, "exposure_seconds": 0.0})

    def score_one(
        member_name: str,
        category: str,
        handle,
        *,
        soundfile_mode: bool = False,
        declared_duration: float | None = None,
    ) -> None:
        member_infer = (lambda waveform: infer_logits(waveform)) if infer_logits else None
        scored = _score_member(
            member_name,
            category,
            handle,
            model=model,
            processor=processor,
            device=device,
            token_ids=token_ids,
            blank_id=blank_id,
            window_lengths_seconds=window_lengths_seconds,
            hop_seconds=hop_seconds,
            beta=beta,
            threshold=threshold,
            chunk_seconds=chunk_seconds,
            refractory_seconds=refractory_seconds,
            infer_logits=member_infer,
            soundfile_mode=soundfile_mode,
        )
        result = _evaluate_member(scored, threshold=threshold, refractory_seconds=refractory_seconds)
        if declared_duration is not None:
            result["duration_seconds"] = declared_duration
        results.append(result)
        stats = category_counts[result["category"]]
        stats["files"] += 1
        stats["events"] += len(result["events"])
        stats["exposure_seconds"] += result["duration_seconds"]
        if progress is not None:
            progress(
                len(results),
                len(locked_rows) if locked_rows is not None else 0,
                result["member"],
            )

    if archive is not None:
        with tarfile.open(archive, mode="r:gz") as source:
            for member in source:
                if not member.isfile() or not member.name.lower().endswith(".wav"):
                    continue
                handle = source.extractfile(member)
                if handle is None:
                    raise ValueError(f"unable to read tar member: {member.name}")
                score_one(member.name, _category(member.name), handle)
    else:
        assert locked_rows is not None
        for row in locked_rows:
            with row["path"].open("rb") as handle:
                score_one(
                    str(row["path"]),
                    row["category"],
                    handle,
                    soundfile_mode=True,
                    declared_duration=row["duration_seconds"],
                )

    exposure_seconds = sum(item["duration_seconds"] for item in results)
    false_accepts = sum(len(item["events"]) for item in results)
    exposure_hours = exposure_seconds / 3600.0
    upper = poisson_upper_95(false_accepts, exposure_hours) if exposure_hours else math.inf
    reasons = []
    if exposure_hours < min_exposure_hours:
        reasons.append(f"negative exposure {exposure_hours:.4f}h is below {min_exposure_hours:.4f}h")
    if upper > max_faph_upper_95:
        reasons.append(f"FAPH upper bound {upper:.4f} exceeds {max_faph_upper_95:.4f}")
    return {
        "schema_version": 1,
        "gate_scope": "untouched_continuous_qualification",
        "qualified": not reasons,
        "source": (
            {
                "archive": str(archive.resolve()),
                "archive_sha256": sha256_file(archive),
                "format": "gzip tar; 16 kHz mono signed PCM WAV members",
            }
            if archive is not None
            else {
                "manifest": str(manifest.resolve()),
                "manifest_sha256": sha256_file(manifest),
                "format": "locked JSON; absolute mono WAV/FLAC paths, 16 kHz or 44.1 kHz resampled to 16 kHz",
                "accepted_source_sample_rates": list(SUPPORTED_SOURCE_SAMPLE_RATES),
                "model_input_sample_rate": TARGET_SAMPLE_RATE,
            }
        ),
        "model": {
            "id": model_id,
            "revision": revision,
            "id_sha256": sha256_text(model_id),
            "revision_sha256": sha256_text(revision),
            **(
                {
                    key: teacher_qualification["model"][key]
                    for key in REQUIRED_TEACHER_HASHES
                }
                if teacher_qualification is not None
                else {}
            ),
        },
        "teacher_qualification": (
            {
                "qualified": True,
                "report_sha256": teacher_qualification_sha256,
                "model_id": teacher_qualification["model"]["id"],
                "revision": teacher_qualification["model"]["revision"],
                "weights_sha256": teacher_qualification["model"]["weights_sha256"],
                "config_sha256": teacher_qualification["model"]["config_sha256"],
                "tokenizer_vocab_sha256": teacher_qualification["model"]["tokenizer_vocab_sha256"],
            }
            if teacher_qualification is not None
            else None
        ),
        "phrase": {"phrase_id": phrase_id},
        "scoring": {
            "window_lengths_seconds": list(window_lengths_seconds),
            "hop_seconds": hop_seconds,
            "collision_margin_beta": beta,
            "threshold": threshold,
            "refractory_seconds": refractory_seconds,
            "chunk_seconds": chunk_seconds,
        },
        "limits": {"min_exposure_hours": min_exposure_hours, "max_faph_upper_95": max_faph_upper_95},
        "counts": {
            "files": len(results),
            "false_accepts": false_accepts,
            "exposure_seconds": exposure_seconds,
            "exposure_hours": exposure_hours,
            "faph": false_accepts / exposure_hours if exposure_hours else None,
            "faph_upper_95": upper if exposure_hours else None,
        },
        "categories": dict(category_counts),
        "members": results,
        "runtime": {"platform": platform.platform(), "evaluator_sha256": sha256_file(Path(__file__)), "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("numpy",)
            if _installed(name)
        }},
        "failure_reasons": reasons,
    }


def _installed(name: str) -> bool:
    try:
        importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--archive", type=Path)
    source_group.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-qualification", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--window-length", type=float, action="append", default=None)
    parser.add_argument("--hop", type=float, default=DEFAULT_HOP_SECONDS)
    parser.add_argument("--chunk-seconds", type=float, default=DEFAULT_CHUNK_SECONDS)
    parser.add_argument("--refractory-seconds", type=float, default=DEFAULT_REFRACTORY_SECONDS)
    parser.add_argument("--min-exposure-hours", type=float, default=DEFAULT_MIN_EXPOSURE_HOURS)
    parser.add_argument("--max-faph-upper-95", type=float, default=DEFAULT_MAX_FAPH_UPPER_95)
    parser.add_argument("--progress-interval", type=int, default=25)
    args = parser.parse_args(argv)
    qualification, qualification_sha256 = _load_teacher_qualification(args.teacher_qualification)
    phrase = get_wake_phrase(REQUIRED_PHRASE_ID)
    qualification_model = qualification["model"]
    weights_path = resolve_hf_weights_path(
        qualification_model["id"],
        revision=qualification_model["revision"],
        local_files_only=True,
    )
    if sha256_file(weights_path) != qualification_model["weights_sha256"]:
        parser.error("qualified teacher weights changed before continuous scoring")
    model, processor, tokenizer, device = load_hf_teacher(
        qualification_model["id"],
        revision=qualification_model["revision"],
        device=args.device,
        local_files_only=True,
    )
    collisions = tuple(resolve_phone_ids(tokenizer, phones) for phones in phrase.collision_phones)
    token_ids = {"canonical": resolve_phone_ids(tokenizer, phrase.phones), "collisions": collisions}
    blank_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    if args.progress_interval < 1:
        parser.error("--progress-interval must be positive")

    def report_progress(completed: int, total: int, member: str) -> None:
        if completed % args.progress_interval == 0 or (total and completed == total):
            print(
                json.dumps(
                    {"continuous_files_scored": completed, "total": total or None, "member": member},
                    sort_keys=True,
                ),
                flush=True,
            )

    report = qualify_archive(
        args.archive,
        threshold=float(qualification["scoring"]["threshold"]),
        beta=float(qualification["scoring"]["collision_margin_beta"]),
        window_lengths_seconds=args.window_length or DEFAULT_WINDOW_LENGTHS,
        hop_seconds=args.hop,
        chunk_seconds=args.chunk_seconds,
        refractory_seconds=args.refractory_seconds,
        min_exposure_hours=args.min_exposure_hours,
        max_faph_upper_95=args.max_faph_upper_95,
        model=model,
        processor=processor,
        device=device,
        token_ids=token_ids,
        blank_id=blank_id,
        model_id=qualification_model["id"],
        revision=qualification_model["revision"],
        phrase_id=phrase.phrase_id,
        manifest=args.manifest,
        teacher_qualification=qualification,
        teacher_qualification_sha256=qualification_sha256,
        progress=report_progress,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"qualified": report["qualified"], "counts": report["counts"]}, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
