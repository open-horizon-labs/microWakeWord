#!/usr/bin/env python3
"""CLI for deterministic, resumable Kizz hard-negative mining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from microwakeword.hard_negative_mining import inventory, merge_shards, mine


def main() -> int:
    parser = argparse.ArgumentParser(
        epilog=(
            "Deployed v19: --model "
            "v19=/private/tmp/kizz-training/checkpoint-candidates/"
            "v19-live-preserved-from-git.tflite (SHA256 "
            "76250d0cef49f893df4724ea6cce0e87b8a8d0d63cf10fbe23c0e624298871ff)."
        )
    )
    parser.add_argument("--root", action="append", type=Path)
    parser.add_argument("--model", action="append", required=False, default=[])
    parser.add_argument(
        "--require-model-sha",
        action="append",
        default=[],
        metavar="NAME=SHA256",
        help="fail unless a configured model has this exact lowercase SHA-256",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoff", type=float, default=0.5)
    parser.add_argument("--context-frames", type=int, default=200)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--nms-frames", type=int, default=220)
    parser.add_argument("--per-source-quota", type=int, default=128)
    parser.add_argument("--per-item-quota", type=int, default=4)
    parser.add_argument(
        "--score-band-quota",
        type=int,
        help="per-band cap; default allocates the source quota across active bands",
    )
    parser.add_argument("--reserve-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--merge-shards", action="store_true")
    parser.add_argument("--allow-incomplete-merge", action="store_true")
    args = parser.parse_args()
    if args.merge_shards:
        print(
            json.dumps(
                merge_shards(
                    args.output,
                    args.shard_count,
                    allow_incomplete=args.allow_incomplete_merge,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.inventory:
        if not args.root:
            parser.error("--root is required with --inventory")
        print(json.dumps(inventory(args.root), indent=2, sort_keys=True))
        return 0
    if not args.root or not args.model:
        parser.error("--root and --model NAME=PATH are required for mining")
    required_model_shas = {}
    for requirement in args.require_model_sha:
        if "=" not in requirement:
            parser.error("--require-model-sha must be NAME=SHA256")
        name, digest = requirement.split("=", 1)
        if not name or name in required_model_shas:
            parser.error("--require-model-sha names must be non-empty and unique")
        required_model_shas[name] = digest
    result = mine(
        args.root,
        args.model,
        args.output,
        cutoff=args.cutoff,
        context_frames=args.context_frames,
        stride=args.stride,
        nms_frames=args.nms_frames,
        per_source_quota=args.per_source_quota,
        per_item_quota=args.per_item_quota,
        score_band_quota=args.score_band_quota,
        reserve_fraction=args.reserve_fraction,
        seed=args.seed,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        max_items=args.max_items,
        checkpoint=args.checkpoint,
        checkpoint_interval=args.checkpoint_interval,
        required_model_shas=required_model_shas,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
