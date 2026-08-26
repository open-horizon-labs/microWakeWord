#!/usr/bin/env python3
"""Cache qualified-teacher decisions at every causal student endpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from microwakeword.ctc_forward_accelerated import suffix_forward_sum_details
from microwakeword.phoneme_student import (
    compact_phone_contract,
    student_output_times_seconds,
)
from tools.cache_kizz_phoneme_teacher_posteriors import load_cache
from tools.cache_kizz_teacher_sequence_scores import (
    WINDOW_LENGTHS_FRAMES as TEACHER_WINDOW_LENGTHS,
    _canonical_hash,
    _sha256_file,
)
from tools.distill_kizz_phoneme_student import (
    OUTPUT_FRAMES,
    student_architecture_contract,
    student_flags_for_architecture,
)

SCORE_KEYS = (
    "raw_canonical_fit",
    "raw_collision_margin",
    "deployment_canonical_fit",
    "deployment_collision_margin",
    "decision_score",
    "eligible",
)


def teacher_endpoint_frames(
    student_times: np.ndarray,
    *,
    teacher_frame_center_seconds: float,
    teacher_frame_stride_seconds: float,
    teacher_frame_count: int,
) -> np.ndarray:
    """Map each causal student output to teacher frames available by that time."""

    times = np.asarray(student_times, dtype=np.float64)
    if (
        times.ndim != 1
        or not len(times)
        or np.any(~np.isfinite(times))
        or teacher_frame_stride_seconds <= 0
        or teacher_frame_count < 1
    ):
        raise ValueError("invalid teacher/student endpoint geometry")
    ends = (
        np.floor(
            (times - float(teacher_frame_center_seconds))
            / float(teacher_frame_stride_seconds)
        ).astype(np.int32)
        + 1
    )
    return np.clip(ends, 1, int(teacher_frame_count))


def causal_suffix_score_grid(
    values: np.ndarray,
    contract: dict,
    *,
    end_frames: np.ndarray,
    window_lengths: tuple[int, ...],
    beta: float = 0.0,
    progress=None,
) -> dict[str, np.ndarray]:
    """Score every declared prefix with suffix-only deployment semantics."""

    logits = np.asarray(values, dtype=np.float32)
    ends = np.asarray(end_frames, dtype=np.int32)
    if (
        logits.ndim != 3
        or not len(logits)
        or ends.ndim != 1
        or not len(ends)
        or np.any(np.diff(ends) < 0)
        or np.any(ends < min(window_lengths))
        or np.any(ends > logits.shape[1])
    ):
        raise ValueError("invalid causal suffix score grid")
    result = {
        key: np.zeros(
            (len(logits), len(ends)), dtype=(bool if key == "eligible" else np.float32)
        )
        for key in SCORE_KEYS
    }
    for endpoint_index, end in enumerate(ends):
        scored = suffix_forward_sum_details(
            logits[:, : int(end)],
            contract,
            window_lengths=window_lengths,
            beta=beta,
        )
        for key in SCORE_KEYS:
            result[key][:, endpoint_index] = scored[key]
        if progress is not None:
            progress(endpoint_index + 1, len(ends))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--posterior-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--student-architecture",
        choices=(
            "control_mixconv",
            "temporal_residual",
            "dilated_temporal_memory",
            "dilated_temporal_memory_wide",
        ),
        default="temporal_residual",
    )
    args = parser.parse_args()

    corpus_path = args.corpus / "corpus.json"
    corpus = json.loads(corpus_path.read_text())
    rows = corpus["examples"]
    contract = compact_phone_contract()
    if corpus.get("compact_phone_contract") != contract:
        parser.error("corpus compact-phone contract differs")
    prefix = args.posterior_cache.with_suffix("")
    declared = json.loads(prefix.with_suffix(".json").read_text())
    metadata, arrays = load_cache(
        prefix,
        expected_model_revision=declared["model"]["revision"],
        expected_weights_sha256=declared["model"]["weights_sha256"],
    )
    if metadata.get("manifest_sha256") != corpus["manifests"]["teacher"]["sha256"]:
        parser.error("posterior cache is not bound to the active corpus")
    offsets = arrays["offsets"]
    lengths = np.diff(offsets)
    if len(lengths) != len(rows) or np.any(lengths != lengths[0]):
        parser.error("causal-window cache requires equal teacher frame counts")
    values = np.stack(
        [
            arrays["log_posteriors"][offsets[i] : offsets[i + 1]]
            for i in range(len(rows))
        ]
    )
    flags = student_flags_for_architecture(
        args.student_architecture, len(contract["tokens"])
    )
    student_times = student_output_times_seconds(flags, OUTPUT_FRAMES)
    timing = metadata["timing"]
    teacher_ends = teacher_endpoint_frames(
        student_times,
        teacher_frame_center_seconds=float(timing["frame_center_seconds"]),
        teacher_frame_stride_seconds=float(timing["frame_stride_seconds"]),
        teacher_frame_count=int(lengths[0]),
    )
    if np.any(teacher_ends < min(TEACHER_WINDOW_LENGTHS)):
        parser.error("student timeline begins before the shortest teacher window")
    scores = causal_suffix_score_grid(
        values,
        contract,
        end_frames=teacher_ends,
        window_lengths=TEACHER_WINDOW_LENGTHS,
        beta=0.0,
        progress=lambda completed, total: (
            print(
                json.dumps({"endpoints_cached": completed, "total": total}),
                flush=True,
            )
            if completed % 8 == 0 or completed == total
            else None
        ),
    )
    scores["teacher_end_frame"] = teacher_ends
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), **scores)
    cache_metadata = {
        "schema_version": 1,
        "representation": "qualified_teacher_causal_student_endpoint_decisions",
        "corpus": {
            "path": str(corpus_path.resolve()),
            "sha256": _sha256_file(corpus_path),
            "teacher_manifest_sha256": corpus["manifests"]["teacher"]["sha256"],
        },
        "posterior_cache": {
            "prefix": str(prefix.resolve()),
            "json_sha256": _sha256_file(prefix.with_suffix(".json")),
            "npz_sha256": _sha256_file(prefix.with_suffix(".npz")),
            "cache_sha256": metadata["cache_sha256"],
        },
        "teacher_model": metadata["model"],
        "compact_phone_contract_sha256": _canonical_hash(contract),
        "student_timeline": {
            "architecture": student_architecture_contract(
                contract, args.student_architecture
            ),
            "output_times_seconds": student_times.tolist(),
            "teacher_end_frames": teacher_ends.tolist(),
        },
        "scorer": {
            "algorithm": "forward_sum_ctc",
            "window_lengths_frames": list(TEACHER_WINDOW_LENGTHS),
            "beta": 0.0,
            "window_selection": "suffix_only_at_each_causal_student_endpoint",
            "decision_score": "raw_canonical_fit + min(raw_collision_margin, 0)",
        },
        "counts": {
            "examples": len(rows),
            "teacher_frames": int(lengths[0]),
            "student_endpoints": len(student_times),
        },
    }
    if any(
        np.any(~np.isfinite(scores[key]))
        for key in ("raw_canonical_fit", "raw_collision_margin", "decision_score")
    ):
        raise ValueError("teacher causal targets must be finite")
    args.output.with_suffix(".json").write_text(
        json.dumps(cache_metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "examples": len(rows),
                "endpoints": len(student_times),
                "architecture": args.student_architecture,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
