#!/usr/bin/env python3
"""Simulate a microphone device against any explicitly configured trainer URL."""

from __future__ import annotations

import argparse
import asyncio
import json

from aiohttp import ClientSession, WSMsgType


async def simulate(
    endpoint: str, device_id: str, device_profile: str, detected: bool
) -> None:
    async with ClientSession() as session, session.ws_connect(endpoint) as websocket:
        await websocket.send_json(
            {
                "type": "hello",
                "device_id": device_id,
                "device_profile": device_profile,
                "firmware_sha": "simulated",
                "audio": {
                    "sample_rate": 16000,
                    "channels": 1,
                    "sample_format": "s16le",
                    "frontend": "m5unified_mic",
                    "gain_profile": "default",
                    "preprocessing": {},
                },
            }
        )
        async for message in websocket:
            if message.type != WSMsgType.TEXT:
                continue
            event = json.loads(message.data)
            print(json.dumps(event), flush=True)
            if event.get("type") == "training_capture":
                byte_count = event["duration_ms"] * 16000 * 2 // 1000
                await websocket.send_json(
                    {
                        "type": "training_sample",
                        "capture_id": event["capture_id"],
                        "bytes": byte_count,
                        "detected": detected,
                    }
                )
                await websocket.send_bytes(b"\0" * byte_count)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint",
        required=True,
        help="Full trainer WebSocket URL; never derived from a voice URL",
    )
    parser.add_argument("--device-id", default="simulated-device-1")
    parser.add_argument("--device-profile", default="m5stack_stackchan_v1")
    parser.add_argument(
        "--detected", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()
    asyncio.run(
        simulate(args.endpoint, args.device_id, args.device_profile, args.detected)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
