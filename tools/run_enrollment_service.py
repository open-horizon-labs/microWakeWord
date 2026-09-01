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
    parser.add_argument(
        "--public-base-url",
        help=(
            "LAN-reachable HTTP base URL advertised to devices for audio "
            "uploads, for example http://192.168.1.10:8091"
        ),
    )
    args = parser.parse_args()
    web.run_app(
        EnrollmentService(args.corpus, args.public_base_url).application(),
        host=args.host,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
