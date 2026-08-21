#!/usr/bin/env python3
"""Validate provenance, audio format, and split isolation for a device corpus."""

import argparse
import json
from pathlib import Path

from microwakeword.device_corpus import validate_device_corpus


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    args = parser.parse_args()
    manifest = validate_device_corpus(args.corpus)
    print(
        json.dumps(
            {
                "corpus_id": manifest["corpus_id"],
                "captures": len(manifest["captures"]),
                "detected": sum(item["detected"] for item in manifest["captures"]),
                "missed": sum(not item["detected"] for item in manifest["captures"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
