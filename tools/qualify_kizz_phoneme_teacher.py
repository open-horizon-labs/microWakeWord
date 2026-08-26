#!/usr/bin/env python3
"""Qualify the pinned pretrained IPA/CTC Kizz teacher.

The command is evaluation-only.  It reads raw audio referenced by manifests and
writes one JSON report; it never creates training data, checkpoints, or caches.
``AutoProcessor`` is deliberately not used because it is broken in the pinned
Transformers 4.55.4 environment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from microwakeword.kizz_evaluation_contract import (
    require_disjoint_groups,
    validate_audio_rows,
)
from microwakeword.kizz_phoneme_teacher import (
    MODEL_ID,
    MODEL_REVISION,
    TARGET_SAMPLE_RATE,
    WindowScore,
    choose_validation_threshold,
    load_hf_teacher,
    resolve_phone_ids,
    resolve_hf_weights_path,
    sha256_file,
    sha256_text,
)
from microwakeword.wake_phrase import HI_FI_KIZZ, WAKE_PHRASES, get_wake_phrase


def _payload(path: Path) -> object:
    return json.loads(path.read_text())


def _rows(path: Path) -> list[dict]:
    payload = _payload(path)
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    for key in ("examples", "records", "anchors", "items", "observations"):
        value = payload.get(key) if isinstance(payload, dict) else None
        if isinstance(value, list):
            return [dict(item) for item in value]
    raise ValueError(f"manifest has no examples/records/anchors list: {path}")


def _audio_sha(row: dict) -> str | None:
    return row.get("audio_sha256") or row.get("source_audio_sha256")


def dedupe_rows(rows: Iterable[dict]) -> tuple[list[dict], list[dict]]:
    """Dedupe by recorded audio SHA, retaining the first provenance row."""
    kept: list[dict] = []
    duplicates: list[dict] = []
    seen: dict[str, dict] = {}
    for row in rows:
        identity = _audio_sha(row)
        if identity is None:
            identity = "path:" + str(Path(row["path"]).resolve())
        if identity in seen:
            duplicates.append(
                {
                    "audio_sha256": identity,
                    "duplicate_of": seen[identity].get("source_id"),
                }
            )
            continue
        seen[identity] = row
        kept.append(row)
    return kept, duplicates


def _load_audio(path: Path) -> np.ndarray:
    try:
        import soundfile as sf
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("soundfile is required for the qualification CLI") from error
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


def _best_window_score_torch(
    log_probs,
    *,
    canonical_tokens: Sequence[int],
    collision_tokens: Sequence[Sequence[int]],
    blank_id: int,
    window_lengths: Sequence[int],
    hop: int,
    beta: float,
) -> WindowScore:
    """Vectorized equivalent of the pure CTC window scorer for the CLI."""
    import torch

    values = log_probs.detach().cpu()
    best: WindowScore | None = None
    seen_lengths = set()
    for requested_length in window_lengths:
        length = min(int(requested_length), len(values))
        if length <= 0 or length in seen_lengths:
            continue
        seen_lengths.add(length)
        starts = list(range(0, len(values) - length + 1, hop))
        tail = len(values) - length
        if not starts or starts[-1] != tail:
            starts.append(tail)
        windows = torch.stack([values[start : start + length] for start in starts])
        emissions = windows.transpose(0, 1)
        input_lengths = torch.full((len(starts),), length, dtype=torch.long)

        def fits(
            tokens: Sequence[int],
            *,
            _starts=starts,
            _emissions=emissions,
            _input_lengths=input_lengths,
        ):
            target = torch.tensor(tokens, dtype=torch.long).expand(len(_starts), -1)
            target_lengths = torch.full((len(_starts),), len(tokens), dtype=torch.long)
            losses = torch.nn.functional.ctc_loss(
                _emissions,
                target,
                _input_lengths,
                target_lengths,
                blank=blank_id,
                reduction="none",
                zero_infinity=False,
            )
            return -losses / len(tokens)

        canonical = fits(canonical_tokens)
        collision = (
            torch.stack([fits(tokens) for tokens in collision_tokens]).max(0).values
        )
        margins = canonical - collision
        for index, start in enumerate(starts):
            margin = float(margins[index])
            if margin < beta:
                continue
            candidate = WindowScore(
                start_frame=start,
                end_frame=start + length,
                canonical_fit=float(canonical[index]),
                collision_fit=float(collision[index]),
                collision_margin=margin,
            )
            if best is None or (
                candidate.canonical_fit,
                candidate.collision_margin,
            ) > (best.canonical_fit, best.collision_margin):
                best = candidate
    return best or WindowScore(0, 0, -math.inf, math.inf, -math.inf)


def _wake_context_metadata(row: dict, waveform_duration_seconds: float) -> dict:
    metadata_path_value = row.get("metadata_path")
    if not metadata_path_value:
        raise ValueError("locked false wake has no metadata_path")
    metadata_path = Path(metadata_path_value).resolve()
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("sha256") != _audio_sha(row):
        raise ValueError("false-wake metadata audio hash differs from manifest")
    expected_id = str(row.get("source_id", "")).removeprefix("false-wake:")
    if metadata.get("observation_id") != expected_id:
        raise ValueError("false-wake metadata observation ID differs from manifest")
    trigger_seconds = float(metadata["pre_wake_ms"]) / 1000.0
    if not 0 < trigger_seconds <= waveform_duration_seconds:
        raise ValueError("false-wake trigger offset is outside the recording")
    return {
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "firmware_sha": metadata.get("firmware_sha"),
        "device_profile": metadata.get("device_profile"),
        "wake_trigger_seconds": trigger_seconds,
    }


def _score_row(
    row: dict,
    *,
    model,
    processor,
    token_ids: dict[str, tuple[int, ...]],
    blank_id: int,
    device,
    window_lengths: Sequence[float],
    hop: float,
    beta: float,
    wake_context_seconds: float | None = None,
) -> dict:
    path = Path(row["path"])
    base = {
        "source_id": row.get("source_id"),
        "audio_sha256": _audio_sha(row),
        "path": str(path),
        "label": int(row.get("label", 0)),
        "split": row.get("split"),
        "duration_seconds": row.get("duration_seconds"),
    }
    if not path.is_file():
        base.update(
            score=None,
            collision_margin=None,
            accepted=False,
            failure_reasons=["missing_audio"],
        )
        return base
    try:
        waveform = _load_audio(path)
        import torch

        inputs = processor(
            waveform, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt"
        )
        with torch.inference_mode():
            logits = model(input_values=inputs.input_values.to(device)).logits[0]
        decoded_phones = processor.batch_decode(
            logits.argmax(dim=-1).detach().cpu().unsqueeze(0)
        )[0]
        log_probs = torch.log_softmax(logits, dim=-1)
        frames_per_second = len(log_probs) / (len(waveform) / TARGET_SAMPLE_RATE)
        frame_lengths = tuple(
            max(1, round(seconds * frames_per_second)) for seconds in window_lengths
        )
        best = _best_window_score_torch(
            log_probs,
            canonical_tokens=token_ids["canonical"],
            collision_tokens=token_ids["collisions"],
            blank_id=blank_id,
            window_lengths=frame_lengths,
            hop=max(1, round(hop * frames_per_second)),
            beta=beta,
        )
        full_best = best
        wake_context = None
        if wake_context_seconds is not None:
            wake_context = _wake_context_metadata(
                row, len(waveform) / TARGET_SAMPLE_RATE
            )
            trigger_seconds = wake_context["wake_trigger_seconds"]
            first_frame = max(
                0,
                math.floor(
                    (trigger_seconds - wake_context_seconds) * frames_per_second
                ),
            )
            last_frame = min(
                len(log_probs), math.ceil(trigger_seconds * frames_per_second)
            )
            context_best = _best_window_score_torch(
                log_probs[first_frame:last_frame],
                canonical_tokens=token_ids["canonical"],
                collision_tokens=token_ids["collisions"],
                blank_id=blank_id,
                window_lengths=frame_lengths,
                hop=max(1, round(hop * frames_per_second)),
                beta=beta,
            )
            best = WindowScore(
                start_frame=context_best.start_frame + first_frame,
                end_frame=context_best.end_frame + first_frame,
                canonical_fit=context_best.canonical_fit,
                collision_fit=context_best.collision_fit,
                collision_margin=context_best.collision_margin,
            )
            wake_context.update(
                context_seconds=wake_context_seconds,
                context_frame_bounds=[first_frame, last_frame],
            )
        score = best.canonical_fit if math.isfinite(best.canonical_fit) else None
        collision_fit = (
            best.collision_fit if math.isfinite(best.collision_fit) else None
        )
        margin = best.collision_margin if math.isfinite(best.collision_margin) else None
        best_window_seconds = [
            best.start_frame / frames_per_second,
            best.end_frame / frames_per_second,
        ]
        if wake_context is not None:
            wake_context["best_window_seconds"] = best_window_seconds
            wake_context["best_window_is_pre_wake"] = best_window_seconds[
                1
            ] <= wake_context["wake_trigger_seconds"] + (1.0 / frames_per_second)
        base.update(
            score=score,
            full_recording_score=(
                full_best.canonical_fit
                if math.isfinite(full_best.canonical_fit)
                else None
            ),
            collision_fit=collision_fit,
            collision_margin=margin,
            best_window_frames=[best.start_frame, best.end_frame],
            best_window_seconds=best_window_seconds,
            emission_frames=len(log_probs),
            emission_frames_per_second=frames_per_second,
            decoded_phones=decoded_phones,
            wake_context=wake_context,
            accepted=False,
            failure_reasons=[],
        )
        if not math.isfinite(best.canonical_fit):
            base["failure_reasons"].append("no_window_passed_collision_margin")
    # Report a row-level reason without hiding the rest of the locked evidence.
    except Exception as error:  # noqa: BLE001
        base.update(
            score=None,
            collision_margin=None,
            accepted=False,
            failure_reasons=[f"scoring_error:{type(error).__name__}:{error}"],
        )
    return base


def _finite_scores(items: Sequence[dict]) -> np.ndarray:
    return np.asarray(
        [
            item["score"] if item.get("score") is not None else -math.inf
            for item in items
        ],
        dtype=np.float64,
    )


def _sha_json(path: Path) -> str:
    return sha256_file(path)


def _validated_adaptation_metadata(
    report_path: Path,
    *,
    model_directory: Path,
    weights_path: Path,
    weights_sha256: str,
    phrase_id: str,
) -> dict:
    report = json.loads(report_path.read_text())
    if report.get("kind") != "kizz_phoneme_teacher_adaptation":
        raise ValueError("adaptation report kind is invalid")
    if report.get("wake_phrase", {}).get("phrase_id") != phrase_id:
        raise ValueError("adaptation report wake phrase differs from qualification")
    checkpoint = report.get("checkpoints", {}).get("best")
    if not isinstance(checkpoint, dict):
        raise ValueError("adaptation report has no best checkpoint")
    if (
        Path(str(checkpoint.get("path", ""))).resolve() != weights_path.resolve()
        or checkpoint.get("file_sha256") != weights_sha256
        or weights_path.parent.resolve() != model_directory.resolve()
    ):
        raise ValueError("adaptation report is not bound to the selected model weights")
    manifest = report.get("manifest", {})
    manifest_path = Path(str(manifest.get("path", "")))
    if (
        not manifest_path.is_file()
        or manifest.get("sha256") != sha256_file(manifest_path)
    ):
        raise ValueError("adaptation training manifest provenance drifted")
    return {
        "path": str(report_path.resolve()),
        "sha256": sha256_file(report_path),
        "training_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": manifest["sha256"],
        },
        "checkpoint": checkpoint,
    }


def build_report(args: argparse.Namespace) -> dict:
    phrase_spec = get_wake_phrase(args.phrase_id)
    collision_phones = {
        transcript: phones
        for transcript, phones in zip(
            phrase_spec.collision_transcripts,
            phrase_spec.collision_phones,
            strict=True,
        )
    }
    model, processor, tokenizer, device = load_hf_teacher(
        args.model_id,
        revision=args.revision,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    token_ids = {
        "canonical": resolve_phone_ids(tokenizer, phrase_spec.phones),
        "collisions": tuple(
            resolve_phone_ids(tokenizer, phones) for phones in collision_phones.values()
        ),
    }
    blank_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    aligned_rows = [
        row
        for path in args.aligned_manifest
        for row in _rows(path)
        if int(row.get("label", -1)) == 1 and row.get("split") in ("validation", "test")
    ]
    validation_negative = [
        row
        for row in _rows(args.validation_negative_manifest)
        if int(row.get("label", -1)) == 0 and row.get("split") == "validation"
    ]
    validation_positive_rows = [
        row for row in aligned_rows if row.get("split") == "validation"
    ]
    test_positive_rows = [row for row in aligned_rows if row.get("split") == "test"]
    natural_rows = [
        row
        for row in _rows(args.natural_positive_manifest)
        if int(row.get("label", -1)) == 1
    ]
    false_wake_rows = _rows(args.false_wake_manifest)
    validation_negative, validation_negative_duplicates = dedupe_rows(
        validation_negative
    )
    validation_negative = sorted(
        validation_negative,
        key=lambda row: hashlib.sha256(
            str(_audio_sha(row) or row.get("source_id") or row["path"]).encode()
        ).hexdigest(),
    )[: args.max_validation_negatives]
    evidence_contracts = {
        "validation_positive": validate_audio_rows(
            validation_positive_rows, group="validation_positive"
        ),
        "validation_negative": validate_audio_rows(
            validation_negative, group="validation_negative"
        ),
        "aligned_test_positive": validate_audio_rows(
            test_positive_rows, group="aligned_test_positive"
        ),
        "natural_positive": validate_audio_rows(natural_rows, group="natural_positive"),
        "false_wake_anchor": validate_audio_rows(
            false_wake_rows,
            group="false_wake_anchor",
            require_locked_anchor=True,
        ),
    }
    evidence_groups = {
        "validation_positive": validation_positive_rows,
        "validation_negative": validation_negative,
        "aligned_test_positive": test_positive_rows,
        "natural_positive": natural_rows,
        "false_wake_anchor": false_wake_rows,
    }
    require_disjoint_groups(evidence_groups)
    require_disjoint_groups(
        {
            "validation": validation_positive_rows + validation_negative,
            "heldout": test_positive_rows + natural_rows + false_wake_rows,
        },
        include_partition_identity=True,
    )
    validation_positive_rows, validation_positive_duplicates = dedupe_rows(
        validation_positive_rows
    )
    test_positive_rows, test_positive_duplicates = dedupe_rows(test_positive_rows)
    natural, natural_duplicates = dedupe_rows(natural_rows)
    false_wakes, false_duplicates = dedupe_rows(false_wake_rows)
    aligned = validation_positive_rows + test_positive_rows
    aligned_duplicates = validation_positive_duplicates + test_positive_duplicates

    def score_group(
        name: str, rows: Sequence[dict], *, wake_context_seconds: float | None = None
    ) -> list[dict]:
        scored = []
        for index, row in enumerate(rows):
            scored.append(
                _score_row(
                    row,
                    model=model,
                    processor=processor,
                    token_ids=token_ids,
                    blank_id=blank_id,
                    device=device,
                    window_lengths=args.window_length,
                    hop=args.hop,
                    beta=args.beta,
                    wake_context_seconds=wake_context_seconds,
                )
            )
            if (index + 1) % 50 == 0 or index + 1 == len(rows):
                print(
                    json.dumps(
                        {"group": name, "scored": index + 1, "total": len(rows)}
                    ),
                    flush=True,
                )
        return scored

    scored_aligned = score_group("aligned_positive", aligned)
    scored_validation_negative = score_group("validation_negative", validation_negative)
    validation_positive = [
        item
        for item in scored_aligned
        if item.get("split") == "validation" and item["label"] == 1
    ]
    test_positive = [
        item
        for item in scored_aligned
        if item.get("split") == "test" and item["label"] == 1
    ]
    validation_seconds = sum(
        float(item.get("duration_seconds") or 0) for item in scored_validation_negative
    )
    point = choose_validation_threshold(
        _finite_scores(validation_positive),
        _finite_scores(scored_validation_negative),
        negative_exposure_seconds=validation_seconds,
        min_recall=args.min_recall,
        max_faph=args.max_faph,
    )
    threshold = point["threshold"]
    scored_natural = score_group("natural_positive", natural)
    scored_false = score_group(
        "false_wake", false_wakes, wake_context_seconds=args.wake_context_seconds
    )
    for group in (
        scored_aligned,
        scored_validation_negative,
        scored_natural,
        scored_false,
    ):
        for item in group:
            if (
                threshold is not None
                and item.get("score") is not None
                and item.get("collision_margin", -math.inf) >= args.beta
            ):
                item["accepted"] = bool(item["score"] >= threshold)
                if not item["accepted"]:
                    item["failure_reasons"].append("below_validation_threshold")
            elif threshold is None:
                item["failure_reasons"].append("no_qualifying_validation_threshold")
    false_accepts = sum(1 for item in scored_false if item["accepted"])
    test_accepts = sum(1 for item in test_positive if item["accepted"])
    natural_accepts = sum(1 for item in scored_natural if item["accepted"])
    test_required = math.ceil(args.min_recall * len(test_positive))
    natural_required = math.ceil(args.min_recall * len(scored_natural))
    reasons = []
    scoring_failures = [
        item
        for group in (
            scored_aligned,
            scored_validation_negative,
            scored_natural,
            scored_false,
        )
        for item in group
        if any(
            reason == "missing_audio" or reason.startswith("scoring_error:")
            for reason in item.get("failure_reasons", [])
        )
    ]
    if not point["qualified"]:
        reasons.append("validation_operating_point_not_qualified")
    if scoring_failures:
        reasons.append("qualification_audio_scoring_failure")
    if test_accepts < test_required:
        reasons.append("aligned_test_recall_below_minimum")
    if natural_accepts < natural_required:
        reasons.append("natural_positive_recall_below_minimum")
    if len(scored_natural) < args.minimum_natural_positives:
        reasons.append("insufficient_natural_positive_evidence")
    if len(scored_false) != 62:
        reasons.append("false_wake_anchor_count_not_62")
    if any(
        not item.get("wake_context", {}).get("best_window_is_pre_wake", False)
        for item in scored_false
    ):
        reasons.append("false_wake_trigger_context_not_proven")
    if false_accepts:
        reasons.append("quarantined_false_wake_accepted")
    qualified = not reasons
    module_path = (
        Path(__file__).resolve().parents[1] / "microwakeword/kizz_phoneme_teacher.py"
    )
    qualifier_path = Path(__file__).resolve()
    versions = {}
    for name in ("torch", "transformers", "numpy", "soundfile"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    config_json = model.config.to_json_string() if hasattr(model, "config") else ""
    vocab_json = json.dumps(tokenizer.get_vocab(), ensure_ascii=False, sort_keys=True)
    weights_path = resolve_hf_weights_path(
        args.model_id,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    weights_sha256 = sha256_file(weights_path)
    adaptation = None
    if args.adaptation_report is not None:
        adaptation = _validated_adaptation_metadata(
            args.adaptation_report,
            model_directory=Path(args.model_id),
            weights_path=weights_path,
            weights_sha256=weights_sha256,
            phrase_id=phrase_spec.phrase_id,
        )
        if args.revision != weights_sha256:
            raise ValueError(
                "local adapted teacher revision must equal its weights SHA-256"
            )
    return {
        "schema_version": 1,
        "gate_scope": "teacher_clip_and_anchor_prequalification",
        "qualified": qualified,
        "model": {
            "id": args.model_id,
            "revision": args.revision,
            "local_files_only": args.local_files_only,
            "id_sha256": sha256_text(args.model_id),
            "revision_sha256": sha256_text(args.revision),
            "config_sha256": sha256_text(config_json),
            "tokenizer_vocab_sha256": sha256_text(vocab_json),
            "weights_path": str(weights_path.resolve()),
            "weights_sha256": weights_sha256,
            "adaptation_report": adaptation,
        },
        "dependencies": versions,
        "runtime": {"device": str(device), "platform": platform.platform()},
        "provenance": {
            "teacher_module_sha256": sha256_file(module_path),
            "qualifier_tool_sha256": sha256_file(qualifier_path),
            "manifests": {
                "aligned": [
                    {"path": str(path.resolve()), "sha256": _sha_json(path)}
                    for path in args.aligned_manifest
                ],
                "validation_negative": {
                    "path": str(args.validation_negative_manifest.resolve()),
                    "sha256": _sha_json(args.validation_negative_manifest),
                },
                "natural_positive": {
                    "path": str(args.natural_positive_manifest.resolve()),
                    "sha256": _sha_json(args.natural_positive_manifest),
                },
                "false_wake": {
                    "path": str(args.false_wake_manifest.resolve()),
                    "sha256": _sha_json(args.false_wake_manifest),
                },
            },
        },
        "phones": {
            "phrase_id": phrase_spec.phrase_id,
            "text": phrase_spec.text,
            "canonical": list(phrase_spec.phones),
            "collisions": {
                key: list(value) for key, value in collision_phones.items()
            },
            "token_ids": {
                "canonical": list(token_ids["canonical"]),
                "collisions": [list(value) for value in token_ids["collisions"]],
            },
            "blank_id": blank_id,
        },
        "scoring": {
            "window_lengths_seconds": args.window_length,
            "hop_seconds": args.hop,
            "collision_margin_beta": args.beta,
            # Freeze the validation-selected operating point next to the
            # decoder settings.  Continuous qualification consumes this
            # immutable value; it must never re-select a threshold on the
            # untouched 100-hour corpus.
            "threshold": threshold,
            "threshold_selection": "validation_only",
            "false_wake_context_seconds": args.wake_context_seconds,
        },
        "limits": {
            "min_recall": args.min_recall,
            "max_faph": args.max_faph,
            "minimum_natural_positives": args.minimum_natural_positives,
        },
        "validation_operating_point": point,
        "counts": {
            "aligned_validation_positive": len(validation_positive),
            "validation_negative": len(scored_validation_negative),
            "validation_negative_exposure_seconds": validation_seconds,
            "aligned_test_positive": len(test_positive),
            "aligned_test_accepted": test_accepts,
            "natural_positive": len(scored_natural),
            "natural_positive_accepted": natural_accepts,
            "false_wake_anchors": len(scored_false),
            "false_wake_accepted": false_accepts,
            "deduplicated": len(aligned_duplicates)
            + len(validation_negative_duplicates)
            + len(natural_duplicates)
            + len(false_duplicates),
        },
        "results": {
            "aligned": scored_aligned,
            "validation_negative": scored_validation_negative,
            "natural_positive": scored_natural,
            "false_wake_anchors": scored_false,
        },
        "deduplication": {
            "aligned": aligned_duplicates,
            "validation_negative": validation_negative_duplicates,
            "natural_positive": natural_duplicates,
            "false_wake": false_duplicates,
        },
        "evidence_contracts": evidence_contracts,
        "failure_reasons": reasons,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-manifest", type=Path, action="append", required=True)
    parser.add_argument("--validation-negative-manifest", type=Path, required=True)
    parser.add_argument("--natural-positive-manifest", type=Path, required=True)
    parser.add_argument("--false-wake-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--adaptation-report", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--window-length", type=float, action="append")
    parser.add_argument("--hop", type=float, default=0.06)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--wake-context-seconds", type=float, default=2.0)
    parser.add_argument("--max-validation-negatives", type=int, default=1024)
    parser.add_argument("--min-recall", type=float, default=0.90)
    parser.add_argument("--max-faph", type=float, default=0.10)
    parser.add_argument(
        "--phrase-id",
        choices=tuple(sorted(WAKE_PHRASES)),
        default=HI_FI_KIZZ.phrase_id,
    )
    parser.add_argument("--minimum-natural-positives", type=int, default=1)
    args = parser.parse_args(argv)
    if Path(args.model_id).is_dir() and args.adaptation_report is None:
        parser.error("a local adapted teacher requires --adaptation-report")
    if args.adaptation_report is not None and not args.adaptation_report.is_file():
        parser.error("--adaptation-report must be an existing file")
    args.window_length = args.window_length or [
        0.56,
        0.68,
        0.80,
        0.96,
        1.16,
        1.40,
        1.60,
    ]
    if (
        args.hop <= 0
        or args.wake_context_seconds <= 0
        or any(length <= 0 for length in args.window_length)
        or args.max_validation_negatives < 1
        or not 0 < args.min_recall <= 1
        or args.max_faph < 0
        or args.minimum_natural_positives < 0
    ):
        parser.error("invalid scoring or qualification limits")
    if args.device == "mps":
        try:
            import torch

            if not torch.backends.mps.is_available():
                parser.error("--device mps requested but MPS is unavailable")
        except ImportError:
            parser.error("PyTorch is required")
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "qualified": report["qualified"],
                "validation_operating_point": report["validation_operating_point"],
                "counts": report["counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
