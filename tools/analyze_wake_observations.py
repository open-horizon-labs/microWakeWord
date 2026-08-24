#!/usr/bin/env python3
"""Correlate quarantined wake observations with UHC STT telemetry.

The result is weak labeling evidence for verifier experiments, not corpus truth.
Human review remains required before promoting any observation to a hard negative.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any
from urllib.request import urlopen


COMMAND_PATTERNS = (
    r"\b(play|pause|stop|next|previous|skip|resume|mute|unmute)\b",
    r"\b(turn|set|raise|lower|increase|decrease)\b.*\b(volume|loud|quiet)\b",
    r"\bwhat(?:'s| is) playing\b",
    r"\b(switch|change)\b.*\b(zone|room)\b",
    r"\b(play|queue)\b.+\b(by|from|album|artist)\b",
)


def load_reliability(url: str | None, path: Path | None) -> dict[str, Any]:
    if path is not None:
        return json.loads(path.read_text())
    if url is None:
        return {"recent": []}
    with urlopen(url, timeout=5) as response:
        return json.load(response)


def transcript_from_detail(detail: object) -> str:
    if not isinstance(detail, str):
        return ""
    return detail.split(" (confidence=", 1)[0].strip()


def normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def token_similarity(left: str, right: str) -> float:
    left_tokens = set(normalized(left).split())
    right_tokens = set(normalized(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def observation_files(corpus: Path) -> list[Path]:
    directories = (
        corpus / "observations" / "false-wakes",
        corpus / "observations" / "wakes",
    )
    return sorted(
        path
        for directory in directories
        if directory.exists()
        for path in directory.glob("*.json")
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weak_label(
    observation: dict[str, Any], matched: list[dict[str, Any]]
) -> tuple[str, str]:
    if observation.get("outcome") == "no_command":
        return "false_wake_no_command", "device VAD timed out without command speech"
    transcripts = [
        transcript_from_detail(event.get("detail"))
        for event in matched
        if event.get("outcome") in {"completed", "endpoint_hint"}
    ]
    transcripts = [text for text in transcripts if text]
    if not transcripts:
        return "speech_unconfirmed", "no usable STT transcript in correlation window"
    command_like = any(
        re.search(pattern, text, re.IGNORECASE)
        for pattern in COMMAND_PATTERNS
        for text in transcripts
    )
    if command_like:
        return "stt_command_candidate", "transcript matches a UHC command heuristic"
    if len(transcripts) >= 2:
        agreement = max(
            token_similarity(left, right)
            for index, left in enumerate(transcripts)
            for right in transcripts[index + 1 :]
        )
        if agreement >= 0.5:
            return "stt_speech_candidate", "providers agree on non-command speech"
    return "review", "speech evidence exists but automatic label is uncertain"


def analyze(
    corpus: Path, reliability: dict[str, Any], window_seconds: float
) -> dict[str, Any]:
    events = reliability.get("recent", [])
    observations = []
    for path in observation_files(corpus):
        observation = json.loads(path.read_text())
        relative_metadata_path = path.relative_to(corpus)
        audio_relative_path = Path(str(observation.get("path", "")))
        audio_path = corpus / audio_relative_path
        if not audio_relative_path.parts or not audio_path.is_file():
            raise ValueError(
                f"observation {path} references missing audio {audio_path}"
            )
        audio_sha256 = sha256_file(audio_path)
        recorded_sha256 = observation.get("sha256")
        if recorded_sha256 and recorded_sha256 != audio_sha256:
            raise ValueError(
                f"observation {path} audio hash mismatch: "
                f"metadata={recorded_sha256} actual={audio_sha256}"
            )
        timestamp = float(observation.get("received_at", 0))
        by_turn: dict[int, list[dict[str, Any]]] = {}
        for event in events:
            event_timestamp = float(event.get("timestamp", 0))
            if abs(event_timestamp - timestamp) <= window_seconds:
                by_turn.setdefault(int(event.get("turn_id", -1)), []).append(event)
        turn_id, matched = min(
            by_turn.items(),
            key=lambda item: abs(float(item[1][0].get("timestamp", 0)) - timestamp),
            default=(-1, []),
        )
        label, basis = weak_label(observation, matched)
        transcripts = [
            {
                "provider": event.get("provider"),
                "outcome": event.get("outcome"),
                "text": transcript_from_detail(event.get("detail")),
                "latency_ms": event.get("latency_ms"),
            }
            for event in matched
            if event.get("outcome") in {"first_partial", "completed", "endpoint_hint"}
        ]
        observations.append(
            {
                "observation_id": observation.get("observation_id"),
                "metadata_path": str(relative_metadata_path),
                "metadata_sha256": sha256_file(path),
                "path": str(audio_relative_path),
                "audio_sha256": audio_sha256,
                "audio_bytes": audio_path.stat().st_size,
                "device_outcome": observation.get("outcome"),
                "wake_probability": observation.get("wake_probability"),
                "c_rms_dbfs": observation.get("c_rms_dbfs"),
                "pre_wake_ms": observation.get("pre_wake_ms", 0),
                "pre_wake_samples": observation.get("pre_wake_samples"),
                "post_wake_samples": observation.get("post_wake_samples"),
                "matched_turn_id": turn_id if matched else None,
                "stt": transcripts,
                "weak_label": label,
                "label_basis": basis,
                "review": observation.get("review"),
            }
        )
    return {
        "schema_version": 1,
        "source": "quarantined_wake_observations_plus_uhc_reliability",
        "source_corpus": str(corpus.resolve()),
        "human_review_required": True,
        "training_eligible": False,
        "observation_count": len(observations),
        "label_counts": dict(Counter(item["weak_label"] for item in observations)),
        "observations": observations,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument(
        "--reliability-url", default="http://127.0.0.1:8088/voice/reliability"
    )
    parser.add_argument("--reliability-file", type=Path)
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.reliability_file is not None:
        reliability = load_reliability(None, args.reliability_file)
    else:
        reliability = load_reliability(args.reliability_url, None)
    report = analyze(args.corpus, reliability, args.window_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {key: report[key] for key in ("observation_count", "label_counts")},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
