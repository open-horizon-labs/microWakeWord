#!/usr/bin/env python3
"""Run the standalone device enrollment endpoint."""

import argparse
from pathlib import Path

from aiohttp import web

from microwakeword.enrollment import EnrollmentService


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()
    web.run_app(
        EnrollmentService(args.corpus).application(), host=args.host, port=args.port
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
