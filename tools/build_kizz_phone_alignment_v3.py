#!/usr/bin/env python3
"""Acoustically qualify and CTC-align canonical-v3 Kizz positives.

The text sent to a synthesizer is not evidence that the waveform contains the
canonical pronunciation.  This tool compares the canonical CTC path against
explicit collision paths, rejects ambiguous or collision-like renderings, and
then derives seven ordered phone regions from the measured canonical token
alignment.  No rejected waveform or descendant overlay reaches the positive
training selection.

Run this tool in the pinned optional alignment environment::

    uv run --python 3.12 --with torch==2.8.0 --with torchaudio==2.8.0 \
      --with soundfile==0.14.0 python tools/build_kizz_phone_alignment_v3.py ...

TorchAudio 2.8 is pinned because its forced-alignment API was deprecated in
2.8 and removed in 2.9.  The model checkpoint hash is recorded in the output.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

SAMPLE_RATE = 16_000
CANONICAL_TRANSCRIPT = "hifikiz"
CANONICAL_PHONES = ("h", "aɪ", "f", "aɪ", "k", "ɪ", "z")
COLLISION_TRANSCRIPTS = (
    "hifikids",
    "hifikiss",
    "highfivekiz",
    "hiffykiz",
    "hippykiz",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def pronunciation_decision(
    candidate_nll: Mapping[str, float],
    *,
    minimum_margin_per_token: float,
    maximum_canonical_nll_per_token: float,
) -> dict[str, Any]:
    """Return a deterministic canonical-vs-collision decision.

    CTC losses are normalized by transcript length before comparison because
    the collision spellings have different character counts.  Acceptance
    requires both an absolute canonical-fit ceiling and positive separation
    from every declared collision.
    """

    required = {CANONICAL_TRANSCRIPT, *COLLISION_TRANSCRIPTS}
    missing = required - set(candidate_nll)
    if missing:
        raise ValueError(f"candidate losses are missing: {sorted(missing)}")
    minimum_margin_per_token = _finite(
        minimum_margin_per_token, "minimum_margin_per_token"
    )
    maximum_canonical_nll_per_token = _finite(
        maximum_canonical_nll_per_token,
        "maximum_canonical_nll_per_token",
    )
    if minimum_margin_per_token < 0 or maximum_canonical_nll_per_token <= 0:
        raise ValueError("pronunciation thresholds must be positive")
    normalized = {
        transcript: _finite(candidate_nll[transcript], transcript) / len(transcript)
        for transcript in sorted(required)
    }
    canonical = normalized[CANONICAL_TRANSCRIPT]
    closest_collision, collision = min(
        ((transcript, normalized[transcript]) for transcript in COLLISION_TRANSCRIPTS),
        key=lambda item: (item[1], item[0]),
    )
    margin = collision - canonical
    reasons = []
    if canonical > maximum_canonical_nll_per_token:
        reasons.append("canonical_fit_too_weak")
    if margin < minimum_margin_per_token:
        reasons.append("collision_not_separated")
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "canonical_nll_per_token": canonical,
        "closest_collision": closest_collision,
        "closest_collision_nll_per_token": collision,
        "canonical_margin_per_token": margin,
        "normalized_nll_per_token": normalized,
        "raw_nll": {
            transcript: float(candidate_nll[transcript])
            for transcript in sorted(required)
        },
    }


def phone_spans_from_token_spans(
    token_spans: Sequence[Mapping[str, float]],
    *,
    waveform_samples: int,
    emission_frames: int,
    crop_offset_seconds: float = 0.0,
) -> tuple[dict[str, float], list[dict[str, float | str]]]:
    """Expand seven measured CTC token centers into contiguous phone regions."""

    if len(token_spans) != len(CANONICAL_PHONES):
        raise ValueError("canonical alignment must contain exactly seven tokens")
    if waveform_samples < 1 or emission_frames < 1:
        raise ValueError("waveform and emission lengths must be positive")
    crop_offset_seconds = _finite(crop_offset_seconds, "crop_offset_seconds")
    if crop_offset_seconds < 0:
        raise ValueError("crop_offset_seconds must be non-negative")
    seconds_per_emission_frame = waveform_samples / SAMPLE_RATE / emission_frames
    centers = []
    for index, span in enumerate(token_spans):
        start = _finite(span["start"], f"token_spans[{index}].start")
        end = _finite(span["end"], f"token_spans[{index}].end")
        if start < 0 or end <= start or end > emission_frames:
            raise ValueError("token spans must be positive and inside emissions")
        centers.append((start + end) * 0.5 * seconds_per_emission_frame)
    if any(right <= left for left, right in itertools.pairwise(centers)):
        raise ValueError("canonical CTC token centers must be strictly ordered")

    boundaries = [
        max(0.0, centers[0] - (centers[1] - centers[0]) * 0.5),
        *[(left + right) * 0.5 for left, right in itertools.pairwise(centers)],
        min(
            waveform_samples / SAMPLE_RATE,
            centers[-1] + (centers[-1] - centers[-2]) * 0.5,
        ),
    ]
    if any(right <= left for left, right in itertools.pairwise(boundaries)):
        raise ValueError("derived phone regions must have positive duration")
    phone_spans = [
        {
            "phone": phone,
            "start_s": crop_offset_seconds + boundaries[index],
            "end_s": crop_offset_seconds + boundaries[index + 1],
        }
        for index, phone in enumerate(CANONICAL_PHONES)
    ]
    phrase_span = {
        "start_s": float(phone_spans[0]["start_s"]),
        "end_s": float(phone_spans[-1]["end_s"]),
    }
    return phrase_span, phone_spans


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
    except ImportError as error:  # pragma: no cover - optional runtime
        raise RuntimeError("install soundfile for Kizz alignment") from error
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    values = np.asarray(samples, dtype=np.float32)
    if values.ndim == 2:
        values = np.mean(values, axis=1)
    if values.ndim != 1 or not len(values) or np.any(~np.isfinite(values)):
        raise ValueError(f"{path}: invalid audio")
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz, got {sample_rate}")
    return values, sample_rate


def _crop_for_alignment(
    row: Mapping[str, Any], samples: np.ndarray
) -> tuple[np.ndarray, float]:
    duration = len(samples) / SAMPLE_RATE
    declared = row.get("phrase_span")
    if isinstance(declared, Mapping):
        if declared.get("start_s") is not None and declared.get("end_s") is not None:
            declared_start = _finite(declared.get("start_s"), "phrase_span.start_s")
            declared_end = _finite(declared.get("end_s"), "phrase_span.end_s")
        elif (
            declared.get("start_ms") is not None and declared.get("end_ms") is not None
        ):
            declared_start = (
                _finite(declared.get("start_ms"), "phrase_span.start_ms") / 1000.0
            )
            declared_end = (
                _finite(declared.get("end_ms"), "phrase_span.end_ms") / 1000.0
            )
        else:
            raise ValueError("declared phrase span needs seconds or milliseconds")
        start = max(0.0, declared_start - 0.2)
        end = min(
            duration,
            declared_end + 0.2,
        )
        if end <= start:
            raise ValueError("declared phrase span is invalid")
    elif str(row.get("source_group", "")).startswith("device_"):
        start, end = 0.0, min(duration, 2.0)
    else:
        start, end = 0.0, duration
    first = round(start * SAMPLE_RATE)
    last = round(end * SAMPLE_RATE)
    return samples[first:last], first / SAMPLE_RATE


class MmsAligner:
    """Bounded TorchAudio MMS CTC scorer and forced aligner."""

    def __init__(self, device: str = "cpu") -> None:
        try:
            import torch
            import torchaudio
        except ImportError as error:  # pragma: no cover - optional runtime
            raise RuntimeError(
                "install torch==2.8.0 and torchaudio==2.8.0 for alignment"
            ) from error
        if torchaudio.__version__.split("+")[0] != "2.8.0":
            raise RuntimeError("Kizz alignment is pinned to torchaudio 2.8.0")
        self.torch = torch
        self.torchaudio = torchaudio
        self.device = torch.device(device)
        self.bundle = torchaudio.pipelines.MMS_FA
        self.model = self.bundle.get_model(with_star=False).to(self.device).eval()
        self.dictionary = self.bundle.get_dict(star=None)
        self.transcripts = (CANONICAL_TRANSCRIPT, *COLLISION_TRANSCRIPTS)
        unknown = {
            character
            for transcript in self.transcripts
            for character in transcript
            if character not in self.dictionary
        }
        if unknown:
            raise RuntimeError(
                f"MMS dictionary is missing characters: {sorted(unknown)}"
            )
        checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / "model.pt"
        if not checkpoint.is_file():
            raise RuntimeError("MMS alignment checkpoint was not materialized")
        self.checkpoint = checkpoint.resolve()
        self.checkpoint_sha256 = sha256_file(self.checkpoint)

    def score_and_align(self, samples: np.ndarray) -> dict[str, Any]:
        torch = self.torch
        functional = self.torchaudio.functional
        waveform = torch.from_numpy(np.asarray(samples, dtype=np.float32)).unsqueeze(0)
        with torch.inference_mode():
            emissions, _ = self.model(waveform.to(self.device))
        emissions = emissions.cpu()
        log_probs = emissions.log_softmax(-1).transpose(0, 1)
        input_lengths = torch.tensor([emissions.shape[1]], dtype=torch.long)
        candidate_nll = {}
        for transcript in self.transcripts:
            target = torch.tensor(
                [self.dictionary[character] for character in transcript],
                dtype=torch.long,
            )
            target_lengths = torch.tensor([len(target)], dtype=torch.long)
            loss = torch.nn.functional.ctc_loss(
                log_probs,
                target,
                input_lengths,
                target_lengths,
                blank=0,
                reduction="sum",
                zero_infinity=False,
            )
            candidate_nll[transcript] = float(loss)

        target = torch.tensor(
            [[self.dictionary[character] for character in CANONICAL_TRANSCRIPT]],
            dtype=torch.int32,
        )
        alignments, scores = functional.forced_align(emissions, target, blank=0)
        merged = functional.merge_tokens(alignments[0], scores[0].exp())
        if len(merged) != len(CANONICAL_TRANSCRIPT):
            raise ValueError("MMS canonical alignment returned the wrong token count")
        return {
            "candidate_nll": candidate_nll,
            "emission_frames": int(emissions.shape[1]),
            "token_spans": [
                {
                    "token": CANONICAL_TRANSCRIPT[index],
                    "start": int(span.start),
                    "end": int(span.end),
                    "score": float(span.score),
                }
                for index, span in enumerate(merged)
            ],
        }


def _alignment_record(
    row: Mapping[str, Any],
    aligner: MmsAligner,
    *,
    minimum_margin_per_token: float,
    maximum_canonical_nll_per_token: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    samples, _ = _load_audio(Path(row["path"]))
    cropped, crop_offset = _crop_for_alignment(row, samples)
    measured = aligner.score_and_align(cropped)
    decision = pronunciation_decision(
        measured["candidate_nll"],
        minimum_margin_per_token=minimum_margin_per_token,
        maximum_canonical_nll_per_token=maximum_canonical_nll_per_token,
    )
    audit = {
        "source_id": row["source_id"],
        "path": row["path"],
        "audio_sha256": row["audio_sha256"],
        "split": row["split"],
        "source_group": row["source_group"],
        "render_text": row.get("render_text"),
        "crop_offset_seconds": crop_offset,
        "crop_duration_seconds": len(cropped) / SAMPLE_RATE,
        **decision,
    }
    if not decision["accepted"]:
        return None, audit
    phrase_span, phone_spans = phone_spans_from_token_spans(
        measured["token_spans"],
        waveform_samples=len(cropped),
        emission_frames=measured["emission_frames"],
        crop_offset_seconds=crop_offset,
    )
    selected = dict(row)
    selected.update(
        {
            "phrase_span": phrase_span,
            "phone_spans": phone_spans,
            "alignment": {
                "method": "ctc_forced_alignment",
                "timing_source": str(aligner.checkpoint),
                "model_sha256": aligner.checkpoint_sha256,
                "transcript": CANONICAL_TRANSCRIPT,
                "token_spans": measured["token_spans"],
                "pronunciation_decision": decision,
            },
        }
    )
    return selected, audit


def _inherit_overlay(
    row: Mapping[str, Any],
    parent: Mapping[str, Any] | None,
    *,
    model_sha256: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    audit = {
        "source_id": row["source_id"],
        "path": row["path"],
        "audio_sha256": row["audio_sha256"],
        "split": row["split"],
        "source_group": row["source_group"],
        "render_text": row.get("render_text"),
    }
    if parent is None:
        audit.update(
            {
                "accepted": False,
                "reasons": ["parent_not_acoustically_qualified"],
            }
        )
        return None, audit
    duration_delta = abs(
        float(row["duration_seconds"]) - float(parent["duration_seconds"])
    )
    if duration_delta > 0.03:
        audit.update(
            {
                "accepted": False,
                "reasons": ["overlay_duration_changed"],
                "duration_delta_seconds": duration_delta,
            }
        )
        return None, audit
    selected = dict(row)
    selected["phrase_span"] = parent["phrase_span"]
    selected["phone_spans"] = parent["phone_spans"]
    selected["alignment"] = {
        "method": "inherited_ctc_forced_alignment",
        "timing_source": parent["source_id"],
        "model_sha256": model_sha256,
        "parent_audio_sha256": parent["audio_sha256"],
        "pronunciation_decision": parent["alignment"]["pronunciation_decision"],
    }
    audit.update(
        {
            "accepted": True,
            "reasons": [],
            "inherited_from": parent["source_id"],
        }
    )
    return selected, audit


def build(
    manifest_path: Path,
    output_path: Path,
    *,
    device: str,
    minimum_margin_per_token: float,
    maximum_canonical_nll_per_token: float,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    positives = [
        row
        for row in manifest.get("examples", [])
        if int(row.get("label", -1)) == 1 and row.get("training_eligible") is True
    ]
    if not positives:
        raise ValueError("canonical manifest has no eligible positives")
    aligner = MmsAligner(device)
    direct = [row for row in positives if row["source_group"] != "noisy_overlay"]
    overlays = [row for row in positives if row["source_group"] == "noisy_overlay"]
    selected = []
    audit_rows = []
    accepted_by_provenance = {}
    for index, row in enumerate(direct):
        aligned, audit = _alignment_record(
            row,
            aligner,
            minimum_margin_per_token=minimum_margin_per_token,
            maximum_canonical_nll_per_token=maximum_canonical_nll_per_token,
        )
        audit_rows.append(audit)
        if aligned is not None:
            selected.append(aligned)
            accepted_by_provenance[aligned["provenance_id"]] = aligned
        print(
            json.dumps(
                {
                    "aligned": index + 1,
                    "direct_total": len(direct),
                    "accepted": aligned is not None,
                    "source_id": row["source_id"],
                    "margin": audit["canonical_margin_per_token"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    for row in overlays:
        parent = accepted_by_provenance.get(row["parent_id"])
        inherited, audit = _inherit_overlay(
            row, parent, model_sha256=aligner.checkpoint_sha256
        )
        audit_rows.append(audit)
        if inherited is not None:
            selected.append(inherited)

    counts = Counter((str(row["split"]), str(row["source_group"])) for row in selected)
    rejection_reasons = Counter(
        reason for row in audit_rows if not row["accepted"] for reason in row["reasons"]
    )
    result = {
        "schema_version": 1,
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": sha256_file(manifest_path),
        "target": {
            "transcript": CANONICAL_TRANSCRIPT,
            "phones": list(CANONICAL_PHONES),
        },
        "aligner": {
            "bundle": "torchaudio.pipelines.MMS_FA",
            "torchaudio_version": aligner.torchaudio.__version__,
            "checkpoint": str(aligner.checkpoint),
            "checkpoint_sha256": aligner.checkpoint_sha256,
            "device": str(aligner.device),
        },
        "thresholds": {
            "minimum_margin_per_token": minimum_margin_per_token,
            "maximum_canonical_nll_per_token": maximum_canonical_nll_per_token,
        },
        "counts": {
            "input_positives": len(positives),
            "direct_positives": len(direct),
            "overlay_positives": len(overlays),
            "selected": len(selected),
            "rejected": len(positives) - len(selected),
            "selected_by_split_and_source": [
                {"split": split, "source_group": source, "count": count}
                for (split, source), count in sorted(counts.items())
            ],
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
        },
        "examples": sorted(
            selected,
            key=lambda row: (row["split"], row["source_group"], row["source_id"]),
        ),
        "audit": sorted(audit_rows, key=lambda row: row["source_id"]),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--minimum-margin-per-token",
        type=float,
        default=0.0,
        help=(
            "Require the canonical CTC transcript to outrank every declared "
            "collision. Zero is the semantic decision boundary; positive "
            "arbitrary margins are optional stricter ablations."
        ),
    )
    parser.add_argument("--maximum-canonical-nll-per-token", type=float, default=3.50)
    args = parser.parse_args(argv)
    result = build(
        args.manifest,
        args.output,
        device=args.device,
        minimum_margin_per_token=args.minimum_margin_per_token,
        maximum_canonical_nll_per_token=args.maximum_canonical_nll_per_token,
    )
    print(json.dumps(result["counts"], indent=2, sort_keys=True))
    return 0 if result["examples"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
