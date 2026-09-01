#!/usr/bin/env python3
"""Explicitly promote one reviewed false-wake observation to a hard negative."""

import argparse
import json
from pathlib import Path

from microwakeword.false_wake import promote_false_wake


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--observation", required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--speaker-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    entry = promote_false_wake(
        args.corpus,
        args.observation,
        reviewer=args.reviewer,
        split=args.split,
        speaker_id=args.speaker_id,
        session_id=args.session_id,
        reason=args.reason,
    )
    print(json.dumps(entry, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
