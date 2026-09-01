#!/usr/bin/env python3
"""CLI for the bounded promoted-audio feature builder."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from microwakeword.promoted_audio import main


if __name__ == "__main__":
    raise SystemExit(main())
