#!/usr/bin/env python3
"""Run a provenance-bound Mac-speaker to StackChan-microphone replay schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_load(line: str) -> dict[str, int] | None:
    if "KIZZ_PERF load " not in line:
        return None
    values: dict[str, int] = {}
    for token in line.split():
        key, separator, value = token.partition("=")
        if separator and re.fullmatch(r"[a-z][a-z0-9_]*", key) and value.isdigit():
            values[key] = int(value)
    return values


def classify(line: str) -> str | None:
    if "Kizz compact verifier:" in line:
        return "compact"
    if "Kizz ordered verifier:" in line:
        return "ordered"
    # Count the platform-level handoff, which is the end-to-end acceptance
    # event. The lower-level "detected locally" diagnostic can be omitted when
    # false-wake evidence transport is busy even though the wake was accepted.
    if "Kizz wake detected on-device:" in line:
        return "accepted"
    if "Kizz detector candidate rejected by layered cascade" in line:
        return "rejected"
    if "KIZZ_PERF load " in line:
        return "perf_load"
    if "KIZZ_PERF timing " in line:
        return "perf_timing"
    if "KIZZ_PERF memory " in line:
        return "perf_memory"
    if line.startswith("ESP-ROM:"):
        return "boot"
    if "rst:" in line:
        return "reset_reason"
    if "Guru Meditation" in line or "abort() was called" in line:
        return "crash"
    return None


def _schedule(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("schedule must be a JSON object")
    rows = value.get("segments", value.get("schedule"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("schedule segments are required")
    normalized = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ValueError(f"segment[{index}] must be an object")
        sources = raw.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"segment[{index}] sources are required")
        paths = [Path(str(item)).expanduser().resolve() for item in sources]
        if any(not item.is_file() for item in paths):
            raise ValueError(f"segment[{index}] source is missing")
        normalized.append(
            {
                "label": str(raw.get("label") or f"segment-{index + 1}"),
                "duration_seconds": float(raw.get("duration_seconds", 0)),
                "gap_seconds": float(raw.get("gap_seconds", 0)),
                "volume": float(raw.get("volume", 0.42)),
                "loop": bool(raw.get("loop", True)),
                "sources": paths,
            }
        )
        if normalized[-1]["duration_seconds"] <= 0:
            raise ValueError(f"segment[{index}] duration must be positive")
    return value, normalized


def _play(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    started = time.monotonic()
    deadline = started + float(row["duration_seconds"])
    sources = list(row["sources"])
    played: list[dict[str, Any]] = []
    index = 0
    while time.monotonic() < deadline and (bool(row["loop"]) or index < len(sources)):
        source = sources[index % len(sources)]
        index += 1
        remaining = max(0.05, deadline - time.monotonic())
        process = subprocess.Popen(
            ["afplay", "-v", str(row["volume"]), str(source)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        status = None
        try:
            status = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            status = -15
        played.append(
            {
                "path": str(source),
                "sha256": sha256_file(source),
                "status": status,
            }
        )
        if status not in {0, -15}:
            raise RuntimeError(f"afplay failed for {source}: {status}")
    remaining = deadline - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)
    return played


def run(
    serial_port: str,
    schedule_path: Path,
    output_dir: Path,
    *,
    ready_timeout: float,
    post_roll: float,
) -> dict[str, Any]:
    try:
        import serial
    except ImportError as error:  # pragma: no cover - environment diagnostic
        raise RuntimeError(
            "pyserial is required: python -m pip install pyserial"
        ) from error
    schedule_path = schedule_path.expanduser().resolve()
    schedule, rows = _schedule(schedule_path)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_path = output_dir / "serial-raw.log"
    events_path = output_dir / "events.jsonl"
    opened_at = time.monotonic()
    active: dict[str, Any] = {"label": None, "test_started": None}
    lock = threading.Lock()
    stop = threading.Event()
    ready = threading.Event()
    events: list[dict[str, Any]] = []
    load_rows: list[dict[str, Any]] = []

    connection = serial.Serial(serial_port, 115200, timeout=0.2)
    connection.dtr = False
    connection.rts = False

    def reader() -> None:
        with (
            raw_path.open("w", encoding="utf-8") as raw_output,
            events_path.open("w", encoding="utf-8") as event_output,
        ):
            while not stop.is_set():
                data = connection.readline()
                if not data:
                    continue
                line = data.decode("utf-8", errors="replace").strip()
                elapsed = time.monotonic() - opened_at
                raw_output.write(f"{elapsed:10.3f} {line}\n")
                raw_output.flush()
                if "Kizz cascade ready:" in line:
                    ready.set()
                kind = classify(line)
                if kind is None:
                    continue
                with lock:
                    label = active["label"]
                    test_started = active["test_started"]
                event = {
                    "wall_time": datetime.now(timezone.utc).isoformat(),
                    "elapsed_open_seconds": round(elapsed, 3),
                    "elapsed_test_seconds": (
                        round(time.monotonic() - test_started, 3)
                        if test_started is not None
                        else None
                    ),
                    "segment": label,
                    "kind": kind,
                    "line": line,
                }
                load = parse_load(line)
                if load is not None:
                    event["counters"] = load
                    load_rows.append(event)
                events.append(event)
                event_output.write(json.dumps(event, sort_keys=True) + "\n")
                event_output.flush()

    thread = threading.Thread(target=reader, name="stackchan-serial", daemon=True)
    thread.start()
    try:
        if not ready.wait(ready_timeout):
            raise TimeoutError("StackChan did not report a ready Kizz cascade")
        time.sleep(2)
        test_started = time.monotonic()
        with lock:
            active["test_started"] = test_started
        played_segments = []
        for row in rows:
            with lock:
                active["label"] = row["label"]
            played = _play(row)
            played_segments.append(
                {
                    **{
                        key: row[key]
                        for key in (
                            "label",
                            "duration_seconds",
                            "gap_seconds",
                            "volume",
                            "loop",
                        )
                    },
                    "sources": played,
                }
            )
            with lock:
                active["label"] = None
            if row["gap_seconds"] > 0:
                time.sleep(row["gap_seconds"])
        time.sleep(post_roll)
    finally:
        stop.set()
        thread.join(timeout=2)
        connection.close()

    active_events = [
        event for event in events if event["elapsed_test_seconds"] is not None
    ]
    counts = Counter(event["kind"] for event in active_events)
    by_segment: dict[str, Counter[str]] = {}
    for event in active_events:
        if event["segment"] is not None:
            by_segment.setdefault(event["segment"], Counter())[event["kind"]] += 1
    first_load = next(
        (
            event["counters"]
            for event in load_rows
            if event["elapsed_test_seconds"] is not None
        ),
        None,
    )
    last_load = load_rows[-1]["counters"] if load_rows else None
    deltas = {}
    if first_load and last_load:
        for key in sorted(set(first_load) & set(last_load)):
            if last_load[key] >= first_load[key]:
                deltas[key] = last_load[key] - first_load[key]
    summary = {
        "schema_version": 1,
        "kind": "kizz_control_physical_playback_result",
        "schedule": {"path": str(schedule_path), "sha256": sha256_file(schedule_path)},
        "schedule_metadata": {
            key: value
            for key, value in schedule.items()
            if key not in {"schedule", "segments"}
        },
        "serial_port": serial_port,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(time.monotonic() - test_started, 3),
        "counts": dict(sorted(counts.items())),
        "by_segment": {
            key: dict(sorted(value.items()))
            for key, value in sorted(by_segment.items())
        },
        "counter_deltas": deltas,
        "segments": played_segments,
        "artifacts": {"serial_raw": str(raw_path), "events": str(events_path)},
        "passed_no_crash": counts["boot"] == 0 and counts["crash"] == 0,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial-port", required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ready-timeout", type=float, default=30)
    parser.add_argument("--post-roll", type=float, default=12)
    args = parser.parse_args(argv)
    report = run(
        args.serial_port,
        args.schedule,
        args.output_dir,
        ready_timeout=args.ready_timeout,
        post_roll=args.post_roll,
    )
    return 0 if report["passed_no_crash"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
