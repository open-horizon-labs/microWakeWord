#!/usr/bin/env python3
"""Attach reviewed phrase spans to a device corpus manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from microwakeword.device_corpus import MANIFEST_NAME, validate_device_corpus


def apply_phrase_spans(manifest: dict, spans: dict) -> int:
    captures = {item["capture_id"]: item for item in manifest["captures"]}
    unknown = sorted(set(spans) - set(captures))
    if unknown:
        raise ValueError(f"unknown capture ids: {unknown}")
    for capture_id, phrase_span in spans.items():
        captures[capture_id]["phrase_span"] = phrase_span
    return len(spans)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument(
        "--spans",
        type=Path,
        required=True,
        help="JSON object mapping capture_id to {start_ms, end_ms}",
    )
    args = parser.parse_args()
    manifest_path = args.corpus / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    spans = json.loads(args.spans.read_text())
    if not isinstance(spans, dict):
        raise ValueError("spans file must contain an object")
    updated = apply_phrase_spans(manifest, spans)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    prior = manifest_path.read_bytes()
    temporary.replace(manifest_path)
    try:
        validate_device_corpus(args.corpus)
    except Exception:
        manifest_path.write_bytes(prior)
        raise
    print(json.dumps({"updated_phrase_spans": updated}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
