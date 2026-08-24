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
    list_audio_files,
    load_waveform,
    mix_positive_context,
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


def batch_examples(examples, background_paths, *, batch_size, rng):
    import torch

    selected = [
        examples[int(i)] for i in rng.integers(0, len(examples), size=batch_size)
    ]
    values = []
    labels = []
    for item in selected:
        waveform = load_waveform(Path(item["path"]), sample_rate=TARGET_SAMPLE_RATE)
        if int(item["label"]):
            background_path = background_paths[
                int(rng.integers(0, len(background_paths)))
            ]
            background = load_waveform(background_path, sample_rate=TARGET_SAMPLE_RATE)
            waveform = mix_positive_context(waveform, background, rng=rng)
        else:
            if len(waveform) > CONTEXT_SAMPLES:
                start = int(rng.integers(0, len(waveform) - CONTEXT_SAMPLES + 1))
                waveform = waveform[start : start + CONTEXT_SAMPLES]
            else:
                waveform = np.pad(waveform, (0, CONTEXT_SAMPLES - len(waveform)))
        values.append(waveform)
        labels.append(float(item["label"]))
    return torch.from_numpy(np.asarray(values, dtype=np.float32)), torch.tensor(
        labels, dtype=torch.float32
    )


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
    all_examples = load_manifest(args.manifest)
    examples = [item for item in all_examples if item.get("split") == "train"]
    if not examples:
        raise ValueError("manifest has no train examples for teacher fitting")
    background_paths = list_audio_files(args.background_dir)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1e-4)
    best_loss = float("inf")
    losses = []
    for step in range(args.steps):
        inputs, labels = batch_examples(
            examples, background_paths, batch_size=args.batch_size, rng=rng
        )
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        scores, _ = model(inputs)
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        value = float(loss.detach().cpu())
        losses.append(value)
        if value < best_loss:
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
    args = parser.parse_args(argv)
    if args.steps < 1 or args.batch_size < 2 or args.learning_rate <= 0:
        parser.error("invalid training parameters")
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
