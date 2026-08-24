#!/usr/bin/env python3
"""Apply the same hard operating-point gate to a waveform D teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from microwakeword.kizz_pretrained_teacher import (
    CONTEXT_SAMPLES,
    TARGET_SAMPLE_RATE,
    build_model,
    fit_context,
    list_audio_files,
    load_waveform,
    mix_positive_context,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scores_for_paths(model, paths, *, batch_size, device, positive_background_paths=None):
    import torch

    values = []
    rng = np.random.default_rng(24111)
    for start in range(0, len(paths), batch_size):
        waveforms = []
        window_counts = []
        for path in paths[start : start + batch_size]:
            waveform = load_waveform(path, sample_rate=TARGET_SAMPLE_RATE)
            if positive_background_paths is not None:
                background_path = positive_background_paths[int(rng.integers(0, len(positive_background_paths)))]
                background = load_waveform(background_path, sample_rate=TARGET_SAMPLE_RATE)
                windows = [mix_positive_context(waveform, background, rng=rng)]
            elif len(waveform) > CONTEXT_SAMPLES:
                offsets = range(0, len(waveform) - CONTEXT_SAMPLES + 1, CONTEXT_SAMPLES // 2)
                windows = [fit_context(waveform, start=int(offset)) for offset in offsets]
            else:
                windows = [fit_context(waveform)]
            waveforms.extend(windows)
            window_counts.append(len(windows))
        with torch.no_grad():
            scores, _ = model(torch.from_numpy(np.asarray(waveforms)).to(device))
        scores = scores.detach().cpu().numpy().astype(np.float64)
        # A path is accepted if any deployment-like sliding window is accepted.
        cursor = 0
        for count in window_counts:
            values.append(float(np.max(scores[cursor : cursor + count])))
            cursor += count
    return np.asarray(values, dtype=np.float64)


def operating_point(positive, negative, negative_seconds, min_recall, max_faph):
    thresholds = np.unique(np.concatenate([positive, negative]))
    candidates = []
    for threshold in thresholds:
        recall = float(np.mean(positive >= threshold))
        false_accepts = int(np.sum(negative >= threshold))
        faph = false_accepts / max(negative_seconds / 3600.0, 1e-12)
        if recall >= min_recall and faph <= max_faph:
            candidates.append((recall, -faph, float(threshold), false_accepts))
    recall_threshold = float(np.quantile(positive, 1.0 - min_recall, method="lower"))
    recall_false_accepts = int(np.sum(negative >= recall_threshold))
    recall_faph = recall_false_accepts / max(negative_seconds / 3600.0, 1e-12)
    if not candidates:
        return {
            "qualified": False,
            "threshold": None,
            "false_accepts": None,
            "threshold_at_recall_floor": recall_threshold,
            "faph_at_recall_floor": recall_faph,
        }
    recall, neg_faph, threshold, false_accepts = max(candidates)
    return {
        "qualified": True,
        "threshold": threshold,
        "positive_recall": recall,
        "faph": -neg_faph,
        "false_accepts": false_accepts,
        "threshold_at_recall_floor": recall_threshold,
        "faph_at_recall_floor": recall_faph,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--positive-dir", type=Path, required=True)
    parser.add_argument("--positive-background-dir", type=Path, required=True)
    parser.add_argument("--negative-dir", type=Path, action="append", required=True)
    parser.add_argument("--heldout-false-wake-dir", type=Path)
    parser.add_argument("--heldout-false-wake-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-recall", type=float, default=0.90)
    parser.add_argument("--max-faph", type=float, default=0.10)
    parser.add_argument("--max-heldout-false-wake-accepts", type=int, default=0)
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    import torch

    if bool(args.heldout_false_wake_dir) == bool(args.heldout_false_wake_manifest):
        parser.error("provide exactly one held-out false-wake source")

    report = json.loads(args.training_report.read_text())
    device = torch.device(args.device or ("mps" if torch.backends.mps.is_available() else "cpu"))
    model, _ = build_model(report["backbone"], unfreeze_last_n=report["unfreeze_last_n"])
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.to(device).eval()
    positive_paths = list_audio_files(args.positive_dir)
    positive_background_paths = list_audio_files(args.positive_background_dir)
    negative_paths = [path for directory in args.negative_dir for path in list_audio_files(directory)]
    if args.heldout_false_wake_manifest:
        payload = json.loads(args.heldout_false_wake_manifest.read_text())
        heldout_paths = tuple(Path(item["path"]) for item in payload["examples"])
        if any(int(item["label"]) != 0 for item in payload["examples"]):
            parser.error("held-out false-wake manifest must contain only negative examples")
    else:
        heldout_paths = list_audio_files(args.heldout_false_wake_dir)
    positive_scores = scores_for_paths(model, positive_paths, batch_size=args.batch_size, device=device, positive_background_paths=positive_background_paths)
    negative_scores = scores_for_paths(model, negative_paths, batch_size=args.batch_size, device=device)
    heldout_scores = scores_for_paths(model, heldout_paths, batch_size=args.batch_size, device=device)
    negative_seconds = sum(len(load_waveform(path, sample_rate=TARGET_SAMPLE_RATE)) for path in negative_paths) / TARGET_SAMPLE_RATE
    point = operating_point(positive_scores, negative_scores, negative_seconds, args.min_recall, args.max_faph)
    threshold = point["threshold"]
    heldout_accepts = None if threshold is None else int(np.sum(heldout_scores >= threshold))
    qualified = bool(point["qualified"] and heldout_accepts <= args.max_heldout_false_wake_accepts)
    result = {
        "schema_version": 1,
        "model": str(args.model.resolve()),
        "model_sha256": sha256_file(args.model),
        "backbone": report["backbone"],
        "positive_count": len(positive_scores),
        "negative_count": len(negative_scores),
        "negative_exposure_seconds": negative_seconds,
        "heldout_false_wake_count": len(heldout_scores),
        "heldout_false_wake_accepts": heldout_accepts,
        "limits": {"min_recall": args.min_recall, "max_faph": args.max_faph, "max_heldout_false_wake_accepts": args.max_heldout_false_wake_accepts},
        "operating_point": point,
        "score_summary": {
            "positive": {
                "min": float(np.min(positive_scores)),
                "median": float(np.median(positive_scores)),
                "max": float(np.max(positive_scores)),
            },
            "negative": {
                "min": float(np.min(negative_scores)),
                "median": float(np.median(negative_scores)),
                "max": float(np.max(negative_scores)),
            },
            "heldout_false_wake": {
                "min": float(np.min(heldout_scores)),
                "median": float(np.median(heldout_scores)),
                "max": float(np.max(heldout_scores)),
            },
        },
        "qualified": qualified,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if qualified else 2


if __name__ == "__main__":
    raise SystemExit(main())
