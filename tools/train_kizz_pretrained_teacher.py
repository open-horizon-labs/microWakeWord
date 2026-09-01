#!/usr/bin/env python3
"""Train the offline D teacher on waveform event labels."""

from __future__ import annotations

import argparse
import json
import random
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
    mix_positive_context_with_mask,
)
from microwakeword.kizz_data_contract import sha256_file as balance_sha256_file
from microwakeword.kizz_data_contract import validate_balance_manifest

sha256_file = balance_sha256_file


def write_manifest(path: Path, examples: list[dict]) -> None:
    path.write_text(
        json.dumps({"schema_version": 2, "examples": examples}, indent=2) + "\n"
    )


def collect_examples(
    root: Path, label: int, source_id: str, limit: int | None
) -> list[dict]:
    paths = list_audio_files(root)
    if limit is not None:
        paths = paths[:limit]
    return [
        {
            "path": str(path.resolve()),
            "label": int(label),
            "source_group": source_id,
            "split": "train",
        }
        for path in paths
    ]


def load_manifest(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    examples = payload.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError(f"manifest has no examples: {path}")
    return examples


def _fit_context_with_mask(waveform):
    values = fit_context(waveform, context_samples=CONTEXT_SAMPLES)
    mask = np.zeros(CONTEXT_SAMPLES, dtype=np.int64)
    mask[: min(len(waveform), CONTEXT_SAMPLES)] = 1
    return values, mask


def _source_weights(specs):
    weights = {"public_speech": 3.0}
    for spec in specs:
        source, separator, value = spec.partition("=")
        if not separator or not source or float(value) <= 0:
            raise ValueError(f"invalid --negative-source-weight: {spec!r}")
        weights[source] = float(value)
    return weights


def batch_examples(
    examples,
    background_paths,
    *,
    batch_size,
    rng,
    negative_source_weights,
):
    import torch

    positives = [item for item in examples if int(item["label"]) == 1]
    negatives = [item for item in examples if int(item["label"]) == 0]
    positive_count = batch_size // 2
    negative_count = batch_size - positive_count
    probabilities = np.asarray(
        [
            negative_source_weights.get(item.get("source_group"), 1.0)
            for item in negatives
        ],
        dtype=np.float64,
    )
    probabilities /= probabilities.sum()
    selected = [
        positives[int(i)] for i in rng.integers(0, len(positives), size=positive_count)
    ]
    selected.extend(
        negatives[int(i)]
        for i in rng.choice(len(negatives), size=negative_count, p=probabilities)
    )
    rng.shuffle(selected)
    values = []
    masks = []
    labels = []
    for item in selected:
        waveform = load_waveform(Path(item["path"]), sample_rate=TARGET_SAMPLE_RATE)
        if int(item["label"]):
            background_path = background_paths[
                int(rng.integers(0, len(background_paths)))
            ]
            background = load_waveform(background_path, sample_rate=TARGET_SAMPLE_RATE)
            waveform, mask = mix_positive_context_with_mask(
                waveform, background, rng=rng
            )
        else:
            if len(waveform) > CONTEXT_SAMPLES:
                start = int(rng.integers(0, len(waveform) - CONTEXT_SAMPLES + 1))
                waveform = waveform[start : start + CONTEXT_SAMPLES]
                mask = np.ones(CONTEXT_SAMPLES, dtype=np.int64)
            else:
                waveform, mask = _fit_context_with_mask(waveform)
        values.append(waveform)
        masks.append(mask)
        labels.append(float(item["label"]))
    return (
        torch.from_numpy(np.asarray(values, dtype=np.float32)),
        torch.from_numpy(np.asarray(masks, dtype=np.int64)),
        torch.tensor(labels, dtype=torch.float32),
    )


def temporal_auxiliary_loss(frame_logits, labels):
    """Localize positives while requiring every negative frame to reject."""
    import torch
    import torch.nn.functional as F

    valid = torch.isfinite(frame_logits)
    losses = []
    positives = labels > 0.5
    if torch.any(positives):
        positive_peaks = torch.amax(
            frame_logits[positives].masked_fill(~valid[positives], -torch.inf), dim=1
        )
        losses.append(F.softplus(1.0 - positive_peaks).mean())
    if torch.any(~positives):
        negative_frames = frame_logits[~positives][valid[~positives]]
        if negative_frames.numel():
            losses.append(F.softplus(negative_frames).mean())
    if not losses:
        return frame_logits.new_zeros(())
    return torch.stack(losses).mean()


def validation_scores(model, examples, background_paths, *, device, seed):
    import torch

    rng = np.random.default_rng(seed)
    positive_scores = []
    negative_scores = []
    negative_seconds = 0.0
    model.eval()
    with torch.inference_mode():
        for item in examples:
            waveform = load_waveform(Path(item["path"]), sample_rate=TARGET_SAMPLE_RATE)
            if int(item["label"]):
                background = load_waveform(
                    background_paths[int(rng.integers(0, len(background_paths)))],
                    sample_rate=TARGET_SAMPLE_RATE,
                )
                waveform, mask = mix_positive_context_with_mask(
                    waveform, background, rng=rng
                )
            else:
                negative_seconds += len(waveform) / TARGET_SAMPLE_RATE
                waveform, mask = _fit_context_with_mask(waveform)
            inputs = torch.from_numpy(waveform[None]).to(device)
            attention_mask = torch.from_numpy(mask[None]).to(device)
            score, _ = model(inputs, attention_mask=attention_mask)
            target = positive_scores if int(item["label"]) else negative_scores
            target.append(float(score[0].detach().cpu()))
    return (
        np.asarray(positive_scores, dtype=np.float64),
        np.asarray(negative_scores, dtype=np.float64),
        negative_seconds,
    )


def validation_point(positive, negative, negative_seconds, *, min_recall, max_faph):
    thresholds = np.unique(np.concatenate([positive, negative]))
    candidates = []
    for threshold in thresholds:
        recall = float(np.mean(positive >= threshold))
        false_accepts = int(np.sum(negative >= threshold))
        faph = false_accepts / max(negative_seconds / 3600.0, 1e-12)
        if recall >= min_recall and faph <= max_faph:
            candidates.append((recall, -faph, float(threshold), false_accepts))
    if candidates:
        recall, neg_faph, threshold, false_accepts = max(candidates)
        return {
            "qualified": True,
            "recall": recall,
            "faph": -neg_faph,
            "threshold": threshold,
            "false_accepts": false_accepts,
        }
    threshold = float(np.quantile(positive, 1.0 - min_recall, method="lower"))
    false_accepts = int(np.sum(negative >= threshold))
    return {
        "qualified": False,
        "recall": float(np.mean(positive >= threshold)),
        "faph": false_accepts / max(negative_seconds / 3600.0, 1e-12),
        "threshold": threshold,
        "false_accepts": false_accepts,
    }


def train(args: argparse.Namespace) -> dict:
    import torch
    import torch.nn.functional as F

    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    balance_report = validate_balance_manifest(
        args.manifest,
        args.balance_contract,
    )
    balance_report_path = output / "balance-report.json"
    balance_report_path.write_text(
        json.dumps(balance_report, indent=2, sort_keys=True) + "\n"
    )
    if not balance_report["qualified"]:
        raise ValueError(
            "source-balance contract rejected manifest; see " f"{balance_report_path}"
        )
    model, hidden_size = build_model(
        args.backbone, unfreeze_last_n=args.unfreeze_last_n
    )
    model.to(device)
    model.train()
    # Frozen weights are not enough: dropout and layer-normalization behavior
    # must also be deterministic for a frozen representation.
    if args.unfreeze_last_n == 0:
        model.backbone.eval()
    all_examples = load_manifest(args.manifest)
    examples = [item for item in all_examples if item.get("split") == "train"]
    validation = [item for item in all_examples if item.get("split") == "validation"]
    if not examples:
        raise ValueError("manifest has no train examples for teacher fitting")
    if not validation:
        raise ValueError("manifest has no validation examples for checkpoint selection")
    background_paths = list_audio_files(args.background_dir)
    negative_source_weights = _source_weights(args.negative_source_weight)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1e-4)
    best_loss = float("inf")
    losses = []
    validation_history = []
    best_validation_key = None
    for step in range(args.steps):
        inputs, attention_mask, labels = batch_examples(
            examples,
            background_paths,
            batch_size=args.batch_size,
            rng=rng,
            negative_source_weights=negative_source_weights,
        )
        inputs = inputs.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        model.train()
        if args.unfreeze_last_n == 0:
            model.backbone.eval()
        scores, frame_logits = model(inputs, attention_mask=attention_mask)
        classification_loss = F.binary_cross_entropy_with_logits(scores, labels)
        temporal_loss = temporal_auxiliary_loss(frame_logits, labels)
        loss = classification_loss + args.temporal_weight * temporal_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        value = float(loss.detach().cpu())
        losses.append(value)
        if (step + 1) % args.validation_interval == 0 or step == 0:
            positive_scores, negative_scores, negative_seconds = validation_scores(
                model,
                validation,
                background_paths,
                device=device,
                seed=args.seed + step + 1,
            )
            point = validation_point(
                positive_scores,
                negative_scores,
                negative_seconds,
                min_recall=args.min_recall,
                max_faph=args.max_faph,
            )
            validation_history.append({"step": step + 1, **point})
            validation_key = (
                int(point["qualified"]),
                -point["faph"],
                point["recall"],
            )
            if best_validation_key is None or validation_key > best_validation_key:
                best_validation_key = validation_key
                best_loss = value
                torch.save(model.state_dict(), output / "best.pt")
        if step == 0 or (step + 1) % args.log_interval == 0:
            print(
                json.dumps({"step": step + 1, "loss": value, "best_loss": best_loss}),
                flush=True,
            )
    torch.save(model.state_dict(), output / "last.pt")
    report = {
        "schema_version": 1,
        "model": "kizz_pretrained_waveform_teacher",
        "backbone": args.backbone,
        "hidden_size": hidden_size,
        "sample_rate": TARGET_SAMPLE_RATE,
        "context_samples": CONTEXT_SAMPLES,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "balance_contract": str(args.balance_contract.resolve()),
        "balance_report": str(balance_report_path),
        "balance_report_sha256": sha256_file(balance_report_path),
        "background_dir": str(args.background_dir.resolve()),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "unfreeze_last_n": args.unfreeze_last_n,
        "seed": args.seed,
        "device": str(device),
        "best_loss": best_loss,
        "last_loss": losses[-1],
        "mean_last_100_loss": float(np.mean(losses[-100:])),
        "temporal_weight": args.temporal_weight,
        "negative_source_weights": negative_source_weights,
        "validation_interval": args.validation_interval,
        "min_recall": args.min_recall,
        "max_faph": args.max_faph,
        "validation_history": validation_history,
        "best_validation_key": best_validation_key,
    }
    (output / "teacher-training.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--balance-contract", type=Path, required=True)
    parser.add_argument("--background-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone", default="microsoft/wavlm-base-plus")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--unfreeze-last-n", type=int, default=0)
    parser.add_argument("--seed", type=int, default=24103)
    parser.add_argument("--device")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--validation-interval", type=int, default=50)
    parser.add_argument("--temporal-weight", type=float, default=0.25)
    parser.add_argument(
        "--negative-source-weight",
        action="append",
        default=[],
        help="Override negative source sampling weight, e.g. public_speech=3",
    )
    parser.add_argument("--min-recall", type=float, default=0.90)
    parser.add_argument("--max-faph", type=float, default=0.10)
    args = parser.parse_args(argv)
    if (
        args.steps < 1
        or args.batch_size < 4
        or args.learning_rate <= 0
        or args.validation_interval < 1
        or args.temporal_weight < 0
    ):
        parser.error("invalid training parameters")
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
