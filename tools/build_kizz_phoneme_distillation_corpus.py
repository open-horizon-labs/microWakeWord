#!/usr/bin/env python3
"""Materialize exact audio/feature pairs for Kizz Control phoneme distillation.

The output contains acoustically-qualified clean positives, the separately
qualified *train-only* StackChan replay positives, and deterministic negative
windows.  It excludes qualification evidence and the pre-locked 100-hour
continuous corpus before writing any training row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from microwakeword.phoneme_student import compact_phone_contract, student_output_times_seconds
from microwakeword.wake_phrase import KIZZ_CONTROL
from tools.build_kizz_aligned_teacher_features_v3 import (
    CONTEXT_SAMPLES,
    SAMPLE_RATE,
    frontend,
    load_audio,
    place_phrase_context,
    validate_aligned_positive,
)
from tools.distill_kizz_student import student_flags


SCHEMA_VERSION = 1
DEFAULT_PUBLIC_PER_SPLIT = {"train": 1024, "validation": 1024, "test": 1024}
APPROVED_PROVIDERS = ("assemblyai", "deepgram", "elevenlabs", "kokoro")
STUDENT_TEST_EVIDENCE_FILENAME = "student-test-positives.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    for key in ("examples", "records"):
        value = payload.get(key) if isinstance(payload, dict) else None
        if isinstance(value, list):
            return [dict(row) for row in value]
    raise ValueError(f"manifest contains no rows: {path}")


def _locked_hashes(path: Path) -> set[str]:
    payload = json.loads(path.read_text())
    if (
        payload.get("gate_scope") != "locked_untouched_continuous_negative_corpus"
        or payload.get("locked_before_scoring") is not True
        or payload.get("training_eligible") is not False
        or float(payload.get("counts", {}).get("exposure_hours", 0)) < 100.0
    ):
        raise ValueError("continuous exclusion manifest is not a valid 100-hour lock")
    values = {str(row["sha256"]) for row in payload.get("examples", [])}
    if len(values) != len(payload.get("examples", [])):
        raise ValueError("continuous exclusion manifest has duplicate hashes")
    return values


def load_pronunciation_acceptances(
    audit_path: Path, source_manifest: Path
) -> set[str]:
    """Load the independent, all-split pronunciation allowlist."""
    payload = json.loads(audit_path.read_text())
    scope = payload.get("scope", {})
    if (
        payload.get("gate_scope") != "independent_source_pronunciation_qc"
        or payload.get("source_manifest_sha256") != sha256_file(source_manifest)
        or scope.get("gate_mode") != "all"
        or set(scope.get("splits", [])) != {"train", "validation", "test"}
    ):
        raise ValueError("source pronunciation audit is not the bound all-split gate")
    results = payload.get("results", [])
    identities = [str(row.get("source_id", "")) for row in results]
    if not identities or any(not value for value in identities) or len(set(identities)) != len(identities):
        raise ValueError("source pronunciation audit has missing/duplicate identities")
    return {
        str(row["source_id"])
        for row in results
        if row.get("accepted") is True
    }


def load_device_training_rows(quality_report_path: Path) -> list[dict]:
    """Resolve the exact 4x4 train-only target-device replay contract."""
    quality = json.loads(quality_report_path.read_text())
    if (
        quality.get("kind")
        != "kizz_control_teacher_adaptation_device_replay_quality"
        or quality.get("gate_scope")
        != "train_only_target_channel_positive_quality"
        or quality.get("qualified") is not True
        or quality.get("counts", {}).get("providers")
        != {provider: 4 for provider in APPROVED_PROVIDERS}
    ):
        raise ValueError("device replay quality report is not the qualified 4x4 contract")
    inputs = quality.get("inputs", {})
    resolved = {}
    for key in ("corpus", "selection", "qualification_evidence"):
        path = Path(str(inputs.get(key, ""))).resolve()
        if not path.is_file() or inputs.get(f"{key}_sha256") != sha256_file(path):
            raise ValueError(f"device replay quality input drifted: {key}")
        resolved[key] = path

    selection = json.loads(resolved["selection"].read_text())
    selected = selection.get("selected_examples", [])
    if (
        selection.get("kind")
        != "kizz_control_teacher_adaptation_device_replay_selection"
        or selection.get("locked_before_teacher_adaptation") is not True
        or selection.get("selected_count") != 16
        or len(selected) != 16
    ):
        raise ValueError("device replay selection is not the locked 16-row contract")
    sources = {}
    for row in selected:
        validate_aligned_positive(row, KIZZ_CONTROL)
        identity = str(row.get("audio_sha256", ""))
        if not identity or identity in sources:
            raise ValueError("device replay selection has missing/duplicate source audio")
        sources[identity] = dict(row)

    heldout_rows = _rows(resolved["qualification_evidence"])
    heldout_voices = set()
    heldout_hashes = set()
    for row in heldout_rows:
        heldout_path = Path(str(row.get("path", ""))).resolve()
        heldout_hash = str(row.get("audio_sha256") or row.get("sha256") or "")
        if (
            not heldout_path.is_file()
            or not heldout_hash
            or sha256_file(heldout_path) != heldout_hash
        ):
            raise ValueError("qualification evidence audio drifted")
        heldout_hashes.add(heldout_hash)
        if row.get("source_audio_sha256"):
            heldout_hashes.add(str(row["source_audio_sha256"]))
        heldout_voices.add(
            (
                str(
                    row.get("provider")
                    or row.get("conditions", {}).get("source_provider", "")
                ).lower(),
                str(
                    row.get("voice")
                    or row.get("conditions", {}).get("source_voice", "")
                ).lower(),
            )
        )
    corpus = json.loads(resolved["corpus"].read_text())
    captures = corpus.get("captures", [])
    results = {
        str(row.get("capture_id")): row for row in quality.get("results", [])
    }
    if len(captures) != 16 or len(results) != 16:
        raise ValueError("device replay corpus/quality result count is not exactly 16")

    corpus_root = resolved["corpus"].parent
    rows = []
    seen_audio = set()
    counts = Counter()
    voices = set()
    for capture in captures:
        conditions = capture.get("conditions", {})
        source_hash = str(conditions.get("source_audio_sha256", ""))
        source = sources.get(source_hash)
        capture_id = str(capture.get("capture_id", ""))
        result = results.get(capture_id)
        provider = str(conditions.get("source_provider", "")).lower()
        voice = str(conditions.get("source_voice", "")).lower()
        audio_hash = str(capture.get("sha256", ""))
        path = (corpus_root / str(capture.get("path", ""))).resolve()
        if (
            source is None
            or result is None
            or result.get("qualified") is not True
            or result.get("failure_reasons")
            or result.get("audio_sha256") != audio_hash
            or result.get("source_audio_sha256") != source_hash
            or capture.get("split") != "train"
            or capture.get("truth") != "positive"
            or capture.get("phrase") != "Kizz Control"
            or provider not in APPROVED_PROVIDERS
            or (provider, voice) in heldout_voices
            or source_hash in heldout_hashes
            or audio_hash in heldout_hashes
            or source.get("provider") != provider
            or str(source.get("voice", "")).lower() != voice
            or source.get("descriptor_sha256")
            != conditions.get("source_descriptor_sha256")
            or not path.is_file()
            or sha256_file(path) != audio_hash
            or audio_hash in seen_audio
        ):
            raise ValueError(f"invalid or drifted device replay row: {capture_id}")
        lag = float(result.get("playback_lag_seconds"))
        if not np.isfinite(lag) or lag <= 0:
            raise ValueError(f"device replay has invalid measured lag: {capture_id}")
        seen_audio.add(audio_hash)
        counts[provider] += 1
        voices.add((provider, voice))
        phrase_span = {
            "start_s": float(source["phrase_span"]["start_s"]) + lag,
            "end_s": float(source["phrase_span"]["end_s"]) + lag,
        }
        phone_spans = [
            {
                "phone": span["phone"],
                "start_s": float(span["start_s"]) + lag,
                "end_s": float(span["end_s"]) + lag,
            }
            for span in source["phone_spans"]
        ]
        rows.append(
            {
                "source_id": f"device-adaptation:{capture_id}",
                "parent_source_id": source["source_id"],
                "path": str(path),
                "audio_sha256": audio_hash,
                "parent_source_audio_sha256": source_hash,
                "label": 1,
                "split": "train",
                "source_group": "device_channel_positive",
                "provider": provider,
                "voice": voice,
                "speaker_id": capture.get("speaker_id"),
                "training_eligible": True,
                "locked_deployment_anchor": False,
                "phrase_span": phrase_span,
                "phone_spans": phone_spans,
                "playback_lag_seconds": lag,
                "quality_report_sha256": sha256_file(quality_report_path),
            }
        )
    if (
        counts != Counter({provider: 4 for provider in APPROVED_PROVIDERS})
        or len(voices) != 16
    ):
        raise ValueError("device replay provider/voice balance is not exactly 4x4")
    return sorted(
        rows, key=lambda row: (row["provider"], row["voice"], row["audio_sha256"])
    )


def deterministic_negative_context(samples: np.ndarray, audio_sha256: str) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float32)
    if len(values) >= CONTEXT_SAMPLES:
        extent = len(values) - CONTEXT_SAMPLES + 1
        start = int(audio_sha256[:16], 16) % extent
        return values[start : start + CONTEXT_SAMPLES]
    left = (CONTEXT_SAMPLES - len(values)) // 2
    return np.pad(values, (left, CONTEXT_SAMPLES - len(values) - left))


def hard_phone_targets(phone_spans: list[dict], translation: float) -> np.ndarray:
    contract = compact_phone_contract()
    token_ids = {token: index for index, token in enumerate(contract["tokens"])}
    times = student_output_times_seconds(student_flags(len(contract["tokens"])), 66)
    targets = np.full(len(times), int(contract["blank_id"]), dtype=np.int16)
    previous = -1
    required_frames = []
    for phone_index, span in enumerate(phone_spans):
        start = float(span["start_s"]) + translation
        end = float(span["end_s"]) + translation
        token_id = token_ids[str(span["phone"])]
        in_span = np.flatnonzero((times >= start) & (times < end))
        targets[in_span] = token_id
        # A measured phone can be shorter than the 30-ms student cadence.
        # Preserve its ordered hard target at the nearest still-available frame
        # instead of silently deleting it from supervision.
        minimum = previous + 1
        maximum = len(times) - (len(phone_spans) - phone_index)
        chosen = int(np.clip(np.argmin(np.abs(times - (start + end) * 0.5)), minimum, maximum))
        required_frames.append((chosen, token_id))
        previous = chosen
    for chosen, token_id in required_frames:
        targets[chosen] = token_id
    if not all(token in targets for token in contract["canonical_path"]):
        raise ValueError("positive context lost a canonical phone on the student timeline")
    return targets


def select_negative_rows(
    rows: list[dict], locked: set[str], *, public_per_split: dict[str, int]
) -> list[dict]:
    selected: list[dict] = []
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if int(row.get("label", -1)) != 0 or row.get("split") not in public_per_split:
            continue
        audio_hash = str(row.get("audio_sha256", ""))
        if not audio_hash or audio_hash in locked:
            continue
        if row.get("locked_deployment_anchor") or row.get("training_eligible") is False:
            continue
        grouped[(str(row["split"]), str(row.get("source_group")))].append(dict(row))
    for key in grouped:
        grouped[key].sort(key=lambda row: (row["audio_sha256"], row["source_id"]))
    for split, limit in public_per_split.items():
        selected.extend(grouped[(split, "public_speech")][: int(limit)])
        selected.extend(grouped[(split, "kizz_control_phonetic_collision")])
        selected.extend(grouped[(split, "device_collision")])
    if any(str(row["audio_sha256"]) in locked for row in selected):
        raise AssertionError("locked continuous audio entered distillation")
    return selected


def _quantized_context(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pcm = np.rint(np.clip(values, -1.0, 1.0) * 32767.0).astype("<i2")
    return pcm, pcm.astype(np.float32) / 32767.0


def student_test_positive_evidence(ledger: list[dict]) -> dict:
    """Freeze the materialized, training-ineligible aligned test positives."""
    rows = [
        dict(row)
        for row in ledger
        if row.get("split") == "test" and int(row.get("label", 0)) == 1
    ]
    if not rows:
        raise ValueError("distillation corpus has no aligned test positives")
    if any(row.get("training_eligible") is not False for row in rows):
        raise ValueError("student test positives must be training-ineligible")
    return {
        "schema_version": 1,
        "gate_scope": "student_aligned_test_positive_evidence",
        "locked_before_student_training": True,
        "training_eligible": False,
        "counts": {"positives": len(rows)},
        "examples": rows,
    }


def build(
    aligned_manifest: Path,
    source_manifest: Path,
    source_pronunciation_audit: Path,
    continuous_lock: Path,
    output: Path,
    *,
    device_quality_report: Path | None = None,
    public_per_split: dict[str, int] | None = None,
) -> dict:
    public_per_split = dict(public_per_split or DEFAULT_PUBLIC_PER_SPLIT)
    contract = compact_phone_contract()
    pronunciation_accepted = load_pronunciation_acceptances(
        source_pronunciation_audit, source_manifest
    )
    aligned_input_count = 0
    pronunciation_excluded = []
    positives = []
    for row in _rows(aligned_manifest):
        validate_aligned_positive(row, KIZZ_CONTROL)
        aligned_input_count += 1
        if str(row.get("source_id")) not in pronunciation_accepted:
            pronunciation_excluded.append(str(row.get("source_id")))
            continue
        positives.append(dict(row))
    if any(row.get("provider") not in APPROVED_PROVIDERS for row in positives):
        raise ValueError("post-acoustic positive set contains an unapproved provider")
    device_positives = (
        load_device_training_rows(device_quality_report)
        if device_quality_report is not None
        else []
    )
    locked = _locked_hashes(continuous_lock)
    if any(row["audio_sha256"] in locked for row in device_positives):
        raise ValueError("locked continuous audio entered device-positive training")
    negatives = select_negative_rows(
        _rows(source_manifest), locked, public_per_split=public_per_split
    )
    output.mkdir(parents=True, exist_ok=True)
    audio_root = output / "audio"
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    labels: list[int] = []
    ledger: list[dict] = []

    work = (
        [(row, True) for row in positives]
        + [(row, True) for row in device_positives]
        + [(row, False) for row in negatives]
    )
    for index, (row, positive) in enumerate(work):
        source_path = Path(row["path"]).resolve()
        if sha256_file(source_path) != row.get("audio_sha256"):
            raise ValueError(f"source audio hash drift: {source_path}")
        source_audio = load_audio(source_path)
        if positive:
            phrase = row["phrase_span"]
            context, translation = place_phrase_context(
                source_audio, (float(phrase["start_s"]), float(phrase["end_s"]))
            )
            hard = hard_phone_targets(row["phone_spans"], translation)
        else:
            context = deterministic_negative_context(source_audio, row["audio_sha256"])
            translation = None
            hard = np.full(66, -1, dtype=np.int16)
        pcm, exact_float = _quantized_context(context)
        identity = hashlib.sha256(
            (str(row["source_id"]) + "\0clean-context-v1").encode()
        ).hexdigest()
        path = audio_root / str(row["split"]) / ("positive" if positive else "negative") / f"{identity[:24]}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, pcm, SAMPLE_RATE, subtype="PCM_16")
        materialized_hash = sha256_file(path)
        features.append(frontend(exact_float))
        targets.append(hard)
        labels.append(1 if positive else 0)
        ledger.append(
            {
                "source_id": f"phoneme-distill:{identity}",
                "parent_source_id": row["source_id"],
                "path": str(path.resolve()),
                "audio_sha256": materialized_hash,
                "source_audio_sha256": row["audio_sha256"],
                "parent_source_audio_sha256": row.get(
                    "parent_source_audio_sha256"
                ),
                "label": 1 if positive else 0,
                "split": row["split"],
                "source_group": row.get("source_group"),
                "provider": row.get("provider"),
                "speaker_id": row.get("speaker_id"),
                "translation_seconds": translation,
                "duration_seconds": CONTEXT_SAMPLES / SAMPLE_RATE,
                "training_eligible": row["split"] == "train",
                "locked_deployment_anchor": False,
            }
        )
        if (index + 1) % 250 == 0 or index + 1 == len(work):
            print(json.dumps({"materialized": index + 1, "total": len(work)}), flush=True)

    np.save(output / "features.npy", np.asarray(features, dtype=np.float16))
    np.save(output / "hard_targets.npy", np.asarray(targets, dtype=np.int16))
    np.save(output / "labels.npy", np.asarray(labels, dtype=np.int8))
    teacher_manifest = {"schema_version": 1, "examples": ledger}
    (output / "teacher-manifest.json").write_text(
        json.dumps(teacher_manifest, indent=2, sort_keys=True) + "\n"
    )
    test_evidence = student_test_positive_evidence(ledger)
    (output / STUDENT_TEST_EVIDENCE_FILENAME).write_text(
        json.dumps(test_evidence, indent=2, sort_keys=True) + "\n"
    )
    split_counts = Counter((row["split"], "positive" if row["label"] else "negative") for row in ledger)
    provider_counts = Counter(row["provider"] for row in ledger if row["label"])
    report = {
        "schema_version": SCHEMA_VERSION,
        "recipe": "kizz_control_compact_phoneme_distillation_v3",
        "compact_phone_contract": contract,
        "input_shape": [260, 40],
        "student_output_frames": 66,
        "student_output_times_seconds": student_output_times_seconds(student_flags(len(contract["tokens"])), 66).tolist(),
        "manifests": {
            "aligned": {"path": str(aligned_manifest.resolve()), "sha256": sha256_file(aligned_manifest)},
            "source": {"path": str(source_manifest.resolve()), "sha256": sha256_file(source_manifest)},
            "source_pronunciation_audit": {
                "path": str(source_pronunciation_audit.resolve()),
                "sha256": sha256_file(source_pronunciation_audit),
            },
            "continuous_lock": {"path": str(continuous_lock.resolve()), "sha256": sha256_file(continuous_lock)},
            "device_quality": (
                {
                    "path": str(device_quality_report.resolve()),
                    "sha256": sha256_file(device_quality_report),
                }
                if device_quality_report is not None
                else None
            ),
            "teacher": {"path": str((output / "teacher-manifest.json").resolve()), "sha256": sha256_file(output / "teacher-manifest.json")},
            "student_test_positive": {
                "path": str((output / STUDENT_TEST_EVIDENCE_FILENAME).resolve()),
                "sha256": sha256_file(output / STUDENT_TEST_EVIDENCE_FILENAME),
            },
        },
        "continuous_exclusion_hash_count": len(locked),
        "counts": {
            "total": len(ledger),
            "splits": {f"{split}:{kind}": count for (split, kind), count in sorted(split_counts.items())},
            "positive_providers": dict(sorted(provider_counts.items())),
            "device_positive_providers": dict(
                sorted(
                    Counter(
                        row["provider"]
                        for row in ledger
                        if row["source_group"] == "device_channel_positive"
                    ).items()
                )
            ),
            "student_test_positives": len(test_evidence["examples"]),
            "aligned_positive_input": aligned_input_count,
            "pronunciation_accepted_positive": len(positives),
            "pronunciation_excluded_positive": len(pronunciation_excluded),
            "negative_groups": dict(sorted(Counter(row["source_group"] for row in ledger if not row["label"]).items())),
        },
        "examples": ledger,
        "array_sha256": {
            name: sha256_file(output / name)
            for name in ("features.npy", "hard_targets.npy", "labels.npy")
        },
    }
    (output / "corpus.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-pronunciation-audit", type=Path, required=True)
    parser.add_argument("--continuous-lock", type=Path, required=True)
    parser.add_argument("--device-quality-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--public-train", type=int, default=1024)
    parser.add_argument("--public-validation", type=int, default=1024)
    parser.add_argument("--public-test", type=int, default=1024)
    args = parser.parse_args()
    report = build(
        args.aligned_manifest,
        args.source_manifest,
        args.source_pronunciation_audit,
        args.continuous_lock,
        args.output,
        device_quality_report=args.device_quality_report,
        public_per_split={"train": args.public_train, "validation": args.public_validation, "test": args.public_test},
    )
    print(json.dumps(report["counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
