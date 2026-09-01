#!/usr/bin/env python3
"""Acoustically qualify and CTC-align canonical-v3 Kizz positives.

The text sent to a synthesizer is not evidence that the waveform contains the
canonical pronunciation. This tool compares the canonical phoneme path against
explicit collision paths, rejects ambiguous or collision-like renderings, and
derives ordered phone regions from a measured token alignment. No rejected
waveform or descendant overlay reaches the positive training selection.

Run this tool in the pinned optional alignment environment::

    uv run --python 3.12 --with torch==2.8.0 --with torchaudio==2.8.0 \
      --with soundfile==0.14.0 python tools/build_kizz_phone_alignment_v3.py ...

TorchAudio 2.8 is pinned because its forced-alignment API was deprecated in
2.8 and removed in 2.9.  The model checkpoint hash is recorded in the output.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from microwakeword.wake_phrase import (
    HI_FI_KIZZ,
    WAKE_PHRASES,
    WakePhraseSpec,
    get_wake_phrase,
)

SAMPLE_RATE = 16_000
CANONICAL_TRANSCRIPT = HI_FI_KIZZ.ctc_transcript
CANONICAL_PHONES = HI_FI_KIZZ.phones
COLLISION_TRANSCRIPTS = HI_FI_KIZZ.collision_transcripts


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


def select_provider_balanced(
    rows: Sequence[Mapping[str, Any]],
    *,
    required_providers: Sequence[str],
    maximum_provider_share: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select the largest deterministic cohort satisfying provider contracts."""
    required = tuple(dict.fromkeys(str(value) for value in required_providers))
    if not required:
        copied = [dict(row) for row in rows]
        return copied, {
            "enabled": False,
            "qualified": True,
            "selected": len(copied),
            "excluded": 0,
        }
    if len(required) < 2:
        raise ValueError("provider balance requires at least two providers")
    maximum_provider_share = _finite(
        maximum_provider_share, "maximum_provider_share"
    )
    if not 1 / len(required) <= maximum_provider_share < 1:
        raise ValueError("maximum provider share is infeasible for required providers")

    eligible = [dict(row) for row in rows if str(row.get("provider")) in required]
    selected: list[dict[str, Any]] = []
    split_reports = {}
    violations = []
    splits = sorted({str(row.get("split")) for row in rows})
    for split in splits:
        groups = {
            provider: [
                row
                for row in eligible
                if row.get("split") == split and row.get("provider") == provider
            ]
            for provider in required
        }
        missing = [provider for provider, items in groups.items() if not items]
        if missing:
            violations.append(
                {
                    "split": split,
                    "reason": "required_provider_missing_after_acoustic_gate",
                    "providers": missing,
                }
            )
            continue
        retain = {provider: len(items) for provider, items in groups.items()}
        while True:
            total = sum(retain.values())
            shares = {provider: count / total for provider, count in retain.items()}
            dominant = max(required, key=lambda provider: (shares[provider], provider))
            if shares[dominant] <= maximum_provider_share:
                break
            if retain[dominant] <= 1:
                violations.append(
                    {
                        "split": split,
                        "reason": "provider_share_cannot_be_balanced",
                        "provider": dominant,
                    }
                )
                break
            retain[dominant] -= 1
        split_rows = []
        for provider, items in groups.items():
            ordered = sorted(
                items,
                key=lambda row: hashlib.sha256(
                    f"{seed}:{split}:{provider}:{row['source_id']}".encode()
                ).hexdigest(),
            )
            split_rows.extend(ordered[: retain[provider]])
        selected.extend(split_rows)
        total = len(split_rows)
        split_reports[split] = {
            "acoustically_accepted_counts": {
                provider: len(groups[provider]) for provider in required
            },
            "selected_counts": dict(
                sorted(Counter(row["provider"] for row in split_rows).items())
            ),
            "selected_shares": {
                provider: sum(row["provider"] == provider for row in split_rows) / total
                for provider in required
            },
        }

    selected.sort(
        key=lambda row: (row["split"], row["source_group"], row["source_id"])
    )
    return selected, {
        "enabled": True,
        "qualified": not violations,
        "required_providers": list(required),
        "maximum_provider_share": maximum_provider_share,
        "seed": seed,
        "acoustically_accepted": len(rows),
        "eligible_required_provider_rows": len(eligible),
        "selected": len(selected),
        "excluded": len(rows) - len(selected),
        "splits": split_reports,
        "violations": violations,
    }


def pronunciation_decision(
    candidate_nll: Mapping[str, float],
    *,
    minimum_margin_per_token: float,
    maximum_canonical_nll_per_token: float,
    phrase_spec: WakePhraseSpec = HI_FI_KIZZ,
    token_lengths: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Return a deterministic canonical-vs-collision decision.

    CTC losses are normalized by transcript length before comparison because
    the collision spellings have different character counts.  Acceptance
    requires both an absolute canonical-fit ceiling and positive separation
    from every declared collision.
    """

    canonical_transcript = phrase_spec.ctc_transcript
    collision_transcripts = phrase_spec.collision_transcripts
    required = {canonical_transcript, *collision_transcripts}
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
    lengths = token_lengths or {transcript: len(transcript) for transcript in required}
    if set(lengths) != required or any(int(lengths[key]) < 1 for key in required):
        raise ValueError("token_lengths must cover every candidate with positive values")
    normalized = {
        transcript: _finite(candidate_nll[transcript], transcript)
        / int(lengths[transcript])
        for transcript in sorted(required)
    }
    canonical = normalized[canonical_transcript]
    closest_collision, collision = min(
        ((transcript, normalized[transcript]) for transcript in collision_transcripts),
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
    phones: Sequence[str] = CANONICAL_PHONES,
) -> tuple[dict[str, float], list[dict[str, float | str]]]:
    """Expand measured CTC token centers into contiguous phone regions."""

    if len(token_spans) != len(phones):
        raise ValueError(
            f"canonical alignment must contain exactly {len(phones)} tokens"
        )
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
        for index, phone in enumerate(phones)
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

    def __init__(
        self, device: str = "cpu", phrase_spec: WakePhraseSpec = HI_FI_KIZZ
    ) -> None:
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
        self.phrase_spec = phrase_spec
        self.bundle = torchaudio.pipelines.MMS_FA
        self.model = self.bundle.get_model(with_star=False).to(self.device).eval()
        self.dictionary = self.bundle.get_dict(star=None)
        self.transcripts = (
            phrase_spec.ctc_transcript,
            *phrase_spec.collision_transcripts,
        )
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
        self.token_lengths = {
            transcript: len(transcript) for transcript in self.transcripts
        }
        self.metadata = {
            "backend": "mms_character_ctc",
            "bundle": "torchaudio.pipelines.MMS_FA",
            "torchaudio_version": torchaudio.__version__,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha256,
            "device": str(self.device),
        }

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

        canonical_transcript = self.phrase_spec.ctc_transcript
        target = torch.tensor(
            [[self.dictionary[character] for character in canonical_transcript]],
            dtype=torch.int32,
        )
        alignments, scores = functional.forced_align(emissions, target, blank=0)
        merged = functional.merge_tokens(alignments[0], scores[0].exp())
        if len(merged) != len(canonical_transcript):
            raise ValueError("MMS canonical alignment returned the wrong token count")
        return {
            "candidate_nll": candidate_nll,
            "emission_frames": int(emissions.shape[1]),
            "token_spans": [
                {
                    "token": canonical_transcript[index],
                    "start": int(span.start),
                    "end": int(span.end),
                    "score": float(span.score),
                }
                for index, span in enumerate(merged)
            ],
        }


class IpaCtcAligner:
    """Pinned open-weight IPA/CTC scorer and forced phone aligner."""

    MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
    MODEL_REVISION = "ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4"

    def __init__(
        self, device: str = "cpu", phrase_spec: WakePhraseSpec = HI_FI_KIZZ
    ) -> None:
        try:
            import torch
            import torchaudio
            from transformers.utils import WEIGHTS_NAME
            from transformers.utils.hub import cached_file
        except ImportError as error:  # pragma: no cover - optional runtime
            raise RuntimeError(
                "install torch==2.8.0, torchaudio==2.8.0, transformers==4.55.4, "
                "and phonemizer==3.3.0 for IPA alignment"
            ) from error
        from microwakeword.kizz_phoneme_teacher import (
            load_hf_teacher,
            resolve_phone_ids,
        )

        if torchaudio.__version__.split("+")[0] != "2.8.0":
            raise RuntimeError("Kizz IPA alignment is pinned to torchaudio 2.8.0")
        try:
            transformers_version = importlib.metadata.version("transformers")
        except importlib.metadata.PackageNotFoundError as error:  # pragma: no cover
            raise RuntimeError("transformers is required for IPA alignment") from error
        if transformers_version != "4.55.4":
            raise RuntimeError("Kizz IPA alignment is pinned to transformers 4.55.4")

        self.torch = torch
        self.torchaudio = torchaudio
        self.phrase_spec = phrase_spec
        self.model, self.processor, self.tokenizer, self.device = load_hf_teacher(
            self.MODEL_ID,
            revision=self.MODEL_REVISION,
            device=device,
            local_files_only=True,
        )
        self.blank_id = int(
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else 0
        )
        sequences = (phrase_spec.phones, *phrase_spec.collision_phones)
        self.transcripts = (
            phrase_spec.ctc_transcript,
            *phrase_spec.collision_transcripts,
        )
        self.token_ids = {
            transcript: resolve_phone_ids(self.tokenizer, phones)
            for transcript, phones in zip(self.transcripts, sequences, strict=True)
        }
        self.token_lengths = {
            transcript: len(tokens) for transcript, tokens in self.token_ids.items()
        }
        checkpoint = cached_file(
            self.MODEL_ID,
            WEIGHTS_NAME,
            revision=self.MODEL_REVISION,
            local_files_only=True,
        )
        if not checkpoint:
            raise RuntimeError("pinned IPA model checkpoint was not materialized")
        self.checkpoint = Path(checkpoint).resolve()
        self.checkpoint_sha256 = sha256_file(self.checkpoint)
        self.metadata = {
            "backend": "wav2vec2_ipa_ctc",
            "model_id": self.MODEL_ID,
            "revision": self.MODEL_REVISION,
            "transformers_version": transformers_version,
            "torchaudio_version": torchaudio.__version__,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha256,
            "device": str(self.device),
            "canonical_phones": list(phrase_spec.phones),
            "collision_phone_paths": {
                transcript: list(phones)
                for transcript, phones in zip(
                    phrase_spec.collision_transcripts,
                    phrase_spec.collision_phones,
                    strict=True,
                )
            },
        }

    def score_and_align(self, samples: np.ndarray) -> dict[str, Any]:
        torch = self.torch
        functional = self.torchaudio.functional
        inputs = self.processor(
            samples, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        )
        with torch.inference_mode():
            logits = self.model(
                input_values=inputs.input_values.to(self.device)
            ).logits[0].cpu()
        log_probs = torch.log_softmax(logits, dim=-1)
        input_lengths = torch.tensor([len(log_probs)], dtype=torch.long)
        candidate_nll = {}
        for transcript in self.transcripts:
            target = torch.tensor(self.token_ids[transcript], dtype=torch.long)
            target_lengths = torch.tensor([len(target)], dtype=torch.long)
            loss = torch.nn.functional.ctc_loss(
                log_probs.unsqueeze(1),
                target,
                input_lengths,
                target_lengths,
                blank=self.blank_id,
                reduction="sum",
                zero_infinity=False,
            )
            candidate_nll[transcript] = float(loss)

        canonical = torch.tensor(
            [self.token_ids[self.phrase_spec.ctc_transcript]], dtype=torch.int32
        )
        alignments, scores = functional.forced_align(
            log_probs.unsqueeze(0), canonical, blank=self.blank_id
        )
        merged = functional.merge_tokens(alignments[0], scores[0].exp())
        if len(merged) != len(self.phrase_spec.phones):
            raise ValueError("IPA canonical alignment returned the wrong token count")
        return {
            "candidate_nll": candidate_nll,
            "emission_frames": int(len(log_probs)),
            "token_spans": [
                {
                    "token": self.phrase_spec.phones[index],
                    "start": int(span.start),
                    "end": int(span.end),
                    "score": float(span.score),
                }
                for index, span in enumerate(merged)
            ],
        }


def _alignment_record(
    row: Mapping[str, Any],
    aligner: MmsAligner | IpaCtcAligner,
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
        phrase_spec=aligner.phrase_spec,
        token_lengths=aligner.token_lengths,
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
        phones=aligner.phrase_spec.phones,
    )
    selected = dict(row)
    selected.update(
        {
            "phrase_span": phrase_span,
            "phone_spans": phone_spans,
            "alignment": {
                "method": f"{aligner.metadata['backend']}_forced_alignment",
                "timing_source": str(aligner.checkpoint),
                "model_sha256": aligner.checkpoint_sha256,
                "transcript": aligner.phrase_spec.ctc_transcript,
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
    phrase_spec: WakePhraseSpec = HI_FI_KIZZ,
    alignment_backend: str = "auto",
    required_providers: Sequence[str] = (),
    maximum_provider_share: float = 0.35,
    provider_balance_seed: int = 231,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    positives = [
        row
        for row in manifest.get("examples", [])
        if int(row.get("label", -1)) == 1 and row.get("training_eligible") is True
    ]
    if not positives:
        raise ValueError("canonical manifest has no eligible positives")
    if alignment_backend == "auto":
        alignment_backend = (
            "wav2vec2_ipa_ctc"
            if phrase_spec.phrase_id == "kizz-control"
            else "mms_character_ctc"
        )
    if alignment_backend == "wav2vec2_ipa_ctc":
        aligner: MmsAligner | IpaCtcAligner = IpaCtcAligner(device, phrase_spec)
    elif alignment_backend == "mms_character_ctc":
        aligner = MmsAligner(device, phrase_spec)
    else:
        raise ValueError(f"unknown alignment backend: {alignment_backend}")
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

    acoustically_accepted_count = len(selected)
    selected, selection_contract = select_provider_balanced(
        selected,
        required_providers=required_providers,
        maximum_provider_share=maximum_provider_share,
        seed=provider_balance_seed,
    )
    if not selection_contract["qualified"]:
        raise ValueError(
            "post-alignment provider contract failed: "
            + json.dumps(selection_contract["violations"], sort_keys=True)
        )

    counts = Counter((str(row["split"]), str(row["source_group"])) for row in selected)
    rejection_reasons = Counter(
        reason for row in audit_rows if not row["accepted"] for reason in row["reasons"]
    )
    result = {
        "schema_version": 1,
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": sha256_file(manifest_path),
        "target": {
            "phrase_id": phrase_spec.phrase_id,
            "text": phrase_spec.text,
            "transcript": phrase_spec.ctc_transcript,
            "phones": list(phrase_spec.phones),
            "collision_transcripts": list(phrase_spec.collision_transcripts),
        },
        "aligner": aligner.metadata,
        "thresholds": {
            "minimum_margin_per_token": minimum_margin_per_token,
            "maximum_canonical_nll_per_token": maximum_canonical_nll_per_token,
        },
        "selection_contract": selection_contract,
        "counts": {
            "input_positives": len(positives),
            "direct_positives": len(direct),
            "overlay_positives": len(overlays),
            "acoustically_accepted": acoustically_accepted_count,
            "selected": len(selected),
            "rejected_acoustically": len(positives) - acoustically_accepted_count,
            "excluded_by_provider_contract": acoustically_accepted_count
            - len(selected),
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
        "--alignment-backend",
        choices=("auto", "mms_character_ctc", "wav2vec2_ipa_ctc"),
        default="auto",
    )
    parser.add_argument(
        "--required-provider",
        action="append",
        default=[],
        help=(
            "Require this provider after acoustic qualification and retain the "
            "largest cohort satisfying --maximum-provider-share; repeatable."
        ),
    )
    parser.add_argument("--maximum-provider-share", type=float, default=0.35)
    parser.add_argument("--provider-balance-seed", type=int, default=231)
    parser.add_argument(
        "--phrase-id",
        choices=tuple(sorted(WAKE_PHRASES)),
        default=HI_FI_KIZZ.phrase_id,
    )
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
        phrase_spec=get_wake_phrase(args.phrase_id),
        alignment_backend=args.alignment_backend,
        required_providers=args.required_provider,
        maximum_provider_share=args.maximum_provider_share,
        provider_balance_seed=args.provider_balance_seed,
    )
    print(json.dumps(result["counts"], indent=2, sort_keys=True))
    return 0 if result["examples"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
