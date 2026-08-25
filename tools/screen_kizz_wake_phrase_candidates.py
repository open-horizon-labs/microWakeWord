#!/usr/bin/env python3
"""Pre-screen replacement Kizz wake phrases with a pinned phoneme CTC model.

This is deliberately a *pre-screen*, not a qualification or deployment gate.
It does not generate audio, train a model, or modify any manifest.  Candidate
thresholds are derived from that candidate's positive renders only; the locked
false wakes are evaluation evidence, never threshold-selection data.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from microwakeword.kizz_phoneme_teacher import (
    MODEL_ID,
    MODEL_REVISION,
    TARGET_SAMPLE_RATE,
    load_hf_teacher,
    resolve_phone_ids,
    sha256_file,
    sha256_text,
)


def parse_candidate(value: str) -> tuple[str, str]:
    """Parse the CLI's ``ID=TEXT`` form without silently accepting bad IDs."""
    candidate_id, separator, text = value.partition("=")
    candidate_id, text = candidate_id.strip(), text.strip()
    if not separator or not candidate_id or not text:
        raise ValueError("candidate must be non-empty ID=TEXT")
    if any(character.isspace() for character in candidate_id):
        raise ValueError("candidate ID cannot contain whitespace")
    return candidate_id, text


def parse_candidates(values: Iterable[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        candidate_id, text = parse_candidate(value)
        if candidate_id in parsed:
            raise ValueError(f"duplicate candidate ID: {candidate_id}")
        parsed[candidate_id] = text
    if not parsed:
        raise ValueError("at least one --candidate is required")
    return parsed


def token_edit_distance(left: Sequence[str], right: Sequence[str]) -> int:
    """Levenshtein distance over phone tokens, not characters."""
    previous = list(range(len(right) + 1))
    for index, left_token in enumerate(left, 1):
        current = [index]
        for right_index, right_token in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_token != right_token),
                )
            )
        previous = current
    return previous[-1]


def minimum_subsequence_edit_distance(
    target: Sequence[str], observed: Sequence[str]
) -> int:
    """Return edit distance from ``target`` to the best observed subsequence.

    False-wake clips contain up to nine seconds of context. Prefix and suffix
    phones outside the best local match therefore carry no edit penalty.
    """
    previous = [0] * (len(observed) + 1)
    for target_index, target_token in enumerate(target, 1):
        current = [target_index]
        for observed_index, observed_token in enumerate(observed, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[observed_index] + 1,
                    previous[observed_index - 1] + (target_token != observed_token),
                )
            )
        previous = current
    return min(previous)


def positive_threshold(scores: Sequence[float], target_recall: float) -> dict:
    """Return the lowest score retaining the requested positive recall.

    This function intentionally has no negative-score argument.  A threshold
    selected here says only how the candidate behaves on its renders; it says
    nothing about qualification or false-accept rate.
    """
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0, 1]")
    finite = np.asarray([float(value) for value in scores if math.isfinite(value)])
    required = math.ceil(target_recall * len(scores))
    if not len(finite) or required <= 0 or required > len(finite):
        return {
            "threshold": None,
            "target_recall": target_recall,
            "required_count": required,
            "finite_positive_count": len(finite),
            "achieved_recall": 0.0,
        }
    ordered = np.sort(finite)
    threshold = float(ordered[-required])
    return {
        "threshold": threshold,
        "target_recall": target_recall,
        "required_count": required,
        "finite_positive_count": len(finite),
        "achieved_recall": float(np.sum(finite >= threshold) / len(scores)),
    }


def discover_positive_audio(
    directory: Path, candidate_ids: Iterable[str]
) -> dict[str, list[Path]]:
    """Find ``ID--*.aiff``/``ID--*.wav`` renders, deterministically."""
    all_audio = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".aiff", ".wav"}
    )
    result = {}
    for candidate_id in candidate_ids:
        prefix = candidate_id + "--"
        result[candidate_id] = [
            path for path in all_audio if path.name.startswith(prefix)
        ]
    return result


def positive_render_dimensions(
    paths: Sequence[Path], candidate_id: str
) -> dict[str, list[str]]:
    """Extract voice and rate labels from ``ID--VOICE--RATE`` filenames."""
    voices = set()
    rates = set()
    pairs = set()
    prefix = candidate_id + "--"
    for path in paths:
        remainder = path.stem.removeprefix(prefix)
        parts = remainder.split("--")
        if len(parts) != 2 or not all(parts):
            raise ValueError(
                f"positive render must be named {candidate_id}--VOICE--RATE: {path}"
            )
        voice, rate = parts
        if (voice, rate) in pairs:
            raise ValueError(
                f"positive render matrix repeats {candidate_id}--{voice}--{rate}"
            )
        pairs.add((voice, rate))
        voices.add(voice)
        rates.add(rate)
    expected = {(voice, rate) for voice in voices for rate in rates}
    if pairs != expected:
        missing = sorted(expected - pairs)
        raise ValueError(
            f"positive render matrix is incomplete for {candidate_id}: {missing}"
        )
    return {"voices": sorted(voices), "rates": sorted(rates)}


def _manifest_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    for key in ("examples", "records", "anchors", "items", "observations"):
        rows = payload.get(key) if isinstance(payload, dict) else None
        if isinstance(rows, list):
            return [dict(row) for row in rows]
    raise ValueError(f"manifest has no row list: {path}")


def _row_path(row: dict, manifest: Path) -> Path:
    value = row.get("path") or row.get("audio_path") or row.get("recording")
    if not value:
        raise ValueError(f"false-wake row has no audio path: {row!r}")
    path = Path(value)
    return path if path.is_absolute() else (manifest.parent / path)


def _load_audio(path: Path) -> np.ndarray:
    try:
        import soundfile as sf
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("soundfile is required for audio screening") from error
    values, sample_rate = sf.read(path, always_2d=False, dtype="float32")
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 2:
        values = values.mean(axis=1, dtype=np.float32)
    if values.ndim != 1 or not len(values):
        raise ValueError(f"audio is empty or not mono/stereo: {path}")
    if int(sample_rate) != TARGET_SAMPLE_RATE:
        target_length = max(
            1, round(len(values) * TARGET_SAMPLE_RATE / int(sample_rate))
        )
        source_x = np.linspace(0.0, 1.0, len(values), endpoint=False)
        target_x = np.linspace(0.0, 1.0, target_length, endpoint=False)
        values = np.interp(target_x, source_x, values).astype(np.float32)
    return values


def _windows(log_probs, lengths: Sequence[int], hop: int):
    # CTC kernels are not consistently implemented on MPS; inference remains
    # on the requested device, while this small batched reduction is portable.
    values = log_probs.detach().cpu()
    chunks = []
    starts = []
    for requested in lengths:
        length = min(int(requested), len(values))
        if length <= 0:
            continue
        positions = list(range(0, len(values) - length + 1, hop))
        if not positions or positions[-1] != len(values) - length:
            positions.append(len(values) - length)
        for start in positions:
            chunks.append(values[start : start + length])
            starts.append((start, length))
    return chunks, starts


def batched_candidate_scores(
    log_probs,
    candidate_tokens: dict[str, Sequence[int]],
    *,
    blank_id: int,
    lengths: Sequence[int],
    hop: int,
) -> dict[str, dict]:
    """Score all candidate windows from one model emission tensor.

    Variable window lengths are padded into one CTC batch. The model has
    already run once for this audio; only inexpensive CTC DP remains. This
    pre-screen deliberately uses each candidate's direct CTC fit. Reusing the
    current phrase's hand-written collision set here would make unrelated
    replacement phrases appear artificially clean.
    """
    import torch

    values = log_probs.detach()
    chunks, locations = _windows(values, lengths, hop)
    if not chunks:
        return {
            candidate_id: {"score": None, "start_frame": None, "end_frame": None}
            for candidate_id in candidate_tokens
        }
    max_length = max(len(chunk) for chunk in chunks)
    batch = torch.full(
        (len(chunks), max_length, values.shape[-1]),
        -1e9,
        dtype=values.dtype,
        device=values.device,
    )
    for index, chunk in enumerate(chunks):
        batch[index, : len(chunk)] = chunk
    emissions = batch.transpose(0, 1)
    input_lengths = torch.tensor(
        [len(chunk) for chunk in chunks], dtype=torch.long, device=values.device
    )

    output = {}
    for candidate_id, target in candidate_tokens.items():
        targets = torch.tensor(target, dtype=torch.long, device=values.device).repeat(
            len(chunks)
        )
        target_lengths = torch.full(
            (len(chunks),), len(target), dtype=torch.long, device=values.device
        )
        losses = torch.nn.functional.ctc_loss(
            emissions,
            targets,
            input_lengths,
            target_lengths,
            blank=blank_id,
            reduction="none",
            zero_infinity=False,
        )
        fits = -losses / max(1, len(target))
        index = int(torch.argmax(fits).item())
        start, length = locations[index]
        score = float(fits[index].cpu())
        output[candidate_id] = {
            "score": score if math.isfinite(score) else None,
            "start_frame": start,
            "end_frame": start + length,
        }
    return output


def greedy_phone_tokens(logits, tokenizer) -> list[str]:
    """CTC-collapse a model output into phone tokens for edit-distance evidence."""
    ids = logits.argmax(dim=-1).detach().cpu().tolist()
    blank = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    result = []
    previous = None
    for token_id in ids:
        if token_id == blank:
            previous = None
            continue
        if token_id != previous:
            token = tokenizer.convert_ids_to_tokens(int(token_id))
            if token not in {tokenizer.unk_token, tokenizer.pad_token}:
                result.append(token)
        previous = token_id
    return result


def derive_ipa(tokenizer, text: str) -> tuple[str, ...]:
    """Derive IPA using the pinned tokenizer's configured espeak backend."""
    phonemes = tokenizer.phonemize(text)
    return tuple(phone for phone in phonemes.split() if phone)


def _sha_audio(path: Path) -> str:
    return sha256_file(path)


def _weights_hash(
    model_id: str, revision: str, local_files_only: bool
) -> tuple[str, str]:
    from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
    from transformers.utils.hub import cached_file

    for filename in (WEIGHTS_NAME, SAFE_WEIGHTS_NAME):
        try:
            cached = cached_file(
                model_id,
                filename,
                revision=revision,
                local_files_only=local_files_only,
            )
        except OSError:
            continue
        if cached:
            path = Path(cached)
            return str(path.resolve()), sha256_file(path)
    raise FileNotFoundError(
        f"could not resolve pinned model weights for {model_id}@{revision}"
    )


def build_report(args: argparse.Namespace) -> dict:
    candidates = parse_candidates(args.candidate)
    positive_paths = discover_positive_audio(args.positive_audio_dir, candidates)
    missing = [
        candidate_id for candidate_id, paths in positive_paths.items() if not paths
    ]
    if missing:
        raise ValueError("no positive renders for candidate(s): " + ", ".join(missing))
    positive_dimensions = {
        candidate_id: positive_render_dimensions(paths, candidate_id)
        for candidate_id, paths in positive_paths.items()
    }
    insufficient = []
    for candidate_id, paths in positive_paths.items():
        dimensions = positive_dimensions[candidate_id]
        if (
            len(paths) < args.minimum_positive_renders
            or len(dimensions["voices"]) < args.minimum_positive_voices
            or len(dimensions["rates"]) < args.minimum_positive_rates
        ):
            insufficient.append(
                f"{candidate_id}={len(paths)} renders/"
                f"{len(dimensions['voices'])} voices/{len(dimensions['rates'])} rates"
            )
    if insufficient:
        raise ValueError("insufficient positive diversity: " + ", ".join(insufficient))
    positive_input_hashes = {
        str(path.resolve()): _sha_audio(path)
        for paths in positive_paths.values()
        for path in paths
    }
    if len(set(positive_input_hashes.values())) != len(positive_input_hashes):
        raise ValueError("positive render matrix contains duplicate audio recordings")
    false_rows = _manifest_rows(args.false_wake_manifest)
    if len(false_rows) != 62:
        raise ValueError(
            f"expected all 62 false wakes, manifest contains {len(false_rows)} rows"
        )
    false_inputs = []
    for index, row in enumerate(false_rows):
        false_wake_id = row.get("source_id") or row.get("id")
        if not false_wake_id:
            raise ValueError(f"false-wake row {index} has no stable ID")
        path = _row_path(row, args.false_wake_manifest).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        false_inputs.append((str(false_wake_id), row, path, _sha_audio(path)))
    false_ids = [item[0] for item in false_inputs]
    false_audio_hashes = [item[3] for item in false_inputs]
    if len(set(false_ids)) != 62 or len(set(false_audio_hashes)) != 62:
        raise ValueError(
            "false-wake manifest must contain 62 unique IDs and recordings"
        )
    model, processor, tokenizer, device = load_hf_teacher(
        args.model_id,
        revision=args.revision,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    phone_sequences = {
        candidate_id: derive_ipa(tokenizer, text)
        for candidate_id, text in candidates.items()
    }
    token_ids = {
        candidate_id: resolve_phone_ids(tokenizer, phones)
        for candidate_id, phones in phone_sequences.items()
    }
    blank_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    positive_scores = {candidate_id: [] for candidate_id in candidates}
    positive_results = []
    false_results = []
    audio_hashes = dict(positive_input_hashes)
    import torch

    def score_audio(path: Path):
        waveform = _load_audio(path)
        inputs = processor(
            waveform, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt"
        )
        with torch.inference_mode():
            logits = model(input_values=inputs.input_values.to(device)).logits[0]
        log_probs = torch.log_softmax(logits, dim=-1)
        fps = len(log_probs) / (len(waveform) / TARGET_SAMPLE_RATE)
        scores = batched_candidate_scores(
            log_probs,
            token_ids,
            blank_id=blank_id,
            lengths=[max(1, round(seconds * fps)) for seconds in args.window_length],
            hop=max(1, round(args.hop * fps)),
        )
        return logits, scores, fps

    audio_cache = {}
    for candidate_id, paths in positive_paths.items():
        for path in paths:
            cache_key = str(path.resolve())
            if cache_key not in audio_cache:
                audio_cache[cache_key] = score_audio(path)
            logits, scores, fps = audio_cache[cache_key]
            item = {
                "candidate_id": candidate_id,
                "path": str(path.resolve()),
                "audio_sha256": audio_hashes[str(path.resolve())],
                "score": scores[candidate_id]["score"],
                "emission_frames_per_second": fps,
            }
            score = scores[candidate_id]["score"]
            positive_scores[candidate_id].append(
                score if score is not None else -math.inf
            )
            positive_results.append(item)
    thresholds = {
        candidate_id: positive_threshold(scores, args.target_recall)
        for candidate_id, scores in positive_scores.items()
    }
    for row in positive_results:
        threshold = thresholds[row["candidate_id"]]["threshold"]
        row["accepted_at_positive_threshold"] = bool(
            threshold is not None
            and row["score"] is not None
            and row["score"] >= threshold
        )

    for false_wake_id, row, path, audio_sha256 in false_inputs:
        cache_key = str(path.resolve())
        if cache_key not in audio_cache:
            audio_cache[cache_key] = score_audio(path)
        logits, scores, fps = audio_cache[cache_key]
        decoded = greedy_phone_tokens(logits, tokenizer)
        audio_hashes[str(path.resolve())] = audio_sha256
        by_candidate = {}
        for candidate_id, value in scores.items():
            threshold = thresholds[candidate_id]["threshold"]
            by_candidate[candidate_id] = {
                **value,
                "threshold": threshold,
                "greedy_phone_edit_distance": minimum_subsequence_edit_distance(
                    phone_sequences[candidate_id], decoded
                ),
                "accepted_at_positive_threshold": bool(
                    threshold is not None
                    and value["score"] is not None
                    and value["score"] >= threshold
                ),
            }
        false_results.append(
            {
                "false_wake_id": false_wake_id,
                "path": str(path.resolve()),
                "audio_sha256": audio_hashes[str(path.resolve())],
                "greedy_decoded_phones": decoded,
                "candidate_scores": by_candidate,
                "emission_frames_per_second": fps,
            }
        )
    candidate_summaries = {}
    for candidate_id, text in candidates.items():
        threshold = thresholds[candidate_id]["threshold"]
        accepted_false = [
            item
            for item in false_results
            if item["candidate_scores"][candidate_id]["accepted_at_positive_threshold"]
        ]
        distances = [
            item["candidate_scores"][candidate_id]["greedy_phone_edit_distance"]
            for item in false_results
        ]
        finite_positive = np.asarray(
            [value for value in positive_scores[candidate_id] if math.isfinite(value)],
            dtype=np.float64,
        )
        candidate_summaries[candidate_id] = {
            "text": text,
            "ipa": list(phone_sequences[candidate_id]),
            "token_ids": list(token_ids[candidate_id]),
            "positive_only_threshold": thresholds[candidate_id],
            "positive_render_count": len(positive_paths[candidate_id]),
            "positive_voices": positive_dimensions[candidate_id]["voices"],
            "positive_rates": positive_dimensions[candidate_id]["rates"],
            "positive_score_summary": {
                "minimum": float(np.min(finite_positive)),
                "median": float(np.median(finite_positive)),
                "maximum": float(np.max(finite_positive)),
            },
            "false_wake_accepts": len(accepted_false),
            "minimum_greedy_phone_edit_distance": min(distances),
            "accepted_false_wake_ids": [
                item["false_wake_id"] for item in accepted_false
            ],
            "threshold": threshold,
        }

    ranking = sorted(
        candidates,
        key=lambda candidate_id: (
            candidate_summaries[candidate_id]["false_wake_accepts"],
            -candidate_summaries[candidate_id]["minimum_greedy_phone_edit_distance"],
            candidate_id,
        ),
    )
    weight_path, weight_sha = _weights_hash(
        args.model_id, args.revision, args.local_files_only
    )
    versions = {
        name: _version(name) for name in ("torch", "transformers", "numpy", "soundfile")
    }
    config_json = model.config.to_json_string() if hasattr(model, "config") else ""
    vocab_json = json.dumps(tokenizer.get_vocab(), ensure_ascii=False, sort_keys=True)
    tool_path = Path(__file__).resolve()
    return {
        "schema_version": 1,
        "pre_screen_only": True,
        "non_qualification": True,
        "qualification_status": "not_a_qualification",
        "qualified": False,
        "model": {
            "id": args.model_id,
            "revision": args.revision,
            "weights_path": weight_path,
            "weights_sha256": weight_sha,
            "config_sha256": sha256_text(config_json),
            "tokenizer_vocab_sha256": sha256_text(vocab_json),
        },
        "runtime": {
            "device": str(device),
            "platform": platform.platform(),
            "dependencies": versions,
        },
        "candidates": candidate_summaries,
        "ranking": ranking,
        "scoring": {
            "window_length_seconds": args.window_length,
            "hop_seconds": args.hop,
            "threshold_source": "candidate_positive_renders_only",
            "candidate_score": "best_length_normalized_ctc_fit",
        },
        "provenance": {
            "false_wake_manifest": {
                "path": str(args.false_wake_manifest.resolve()),
                "sha256": sha256_file(args.false_wake_manifest),
            },
            "positive_generator": args.positive_generator,
            "audio_sha256": audio_hashes,
            "screen_tool": {
                "path": str(tool_path),
                "sha256": sha256_file(tool_path),
            },
        },
        "counts": {
            "candidates": len(candidates),
            "positive_renders": len(positive_results),
            "false_wakes": len(false_results),
            "false_wake_expected": 62,
        },
        "positive_results": positive_results,
        "false_wake_results": false_results,
        "limitations": [
            "pre_screen_only",
            "no qualification decision",
            "no deployment recommendation",
            "thresholds use positive renders only",
            "false-wake results are evidence, not training labels",
        ],
    }


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate", action="append", required=True, metavar="ID=TEXT"
    )
    parser.add_argument("--positive-audio-dir", type=Path, required=True)
    parser.add_argument("--positive-generator", required=True)
    parser.add_argument("--minimum-positive-renders", type=int, default=24)
    parser.add_argument("--minimum-positive-voices", type=int, default=12)
    parser.add_argument("--minimum-positive-rates", type=int, default=2)
    parser.add_argument("--false-wake-manifest", type=Path, required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument("--window-length", type=float, action="append", default=None)
    parser.add_argument("--hop", type=float, default=0.06)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.window_length = args.window_length or [
        0.56,
        0.68,
        0.80,
        0.96,
        1.16,
        1.40,
        1.60,
        2.00,
    ]
    if (
        not 0 < args.target_recall <= 1
        or args.hop <= 0
        or args.minimum_positive_renders <= 0
        or args.minimum_positive_voices <= 0
        or args.minimum_positive_rates <= 0
        or any(value <= 0 for value in args.window_length)
    ):
        parser.error(
            "target recall, hop, and window lengths must be positive; recall must be <= 1"
        )
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        + "\n"
    )
    print(
        json.dumps(
            {"pre_screen_only": True, "qualified": False, "counts": report["counts"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
