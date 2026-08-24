#!/usr/bin/env python3
"""Score continuous WAV sources with the deployed ordered-state path.

The manifest is a JSON object containing ``sources``.  Each source has ``id``,
``path``, optional ``session_id`` and ``label``, and optional positive
``occurrences`` (objects with ``id``, ``start_seconds`` and ``end_seconds``).
The frontend and decoder are reset only between sources.  Output is JSONL:
one header, score records, event records, positive occurrence records, and a
source summary for each source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from microwakeword.audio.audio_utils import MicroFrontend
from microwakeword.ordered_state import KIZZ_TOPOLOGY, OrderedStateDecoder
from microwakeword.ordered_state_tflite import tflite_output_logits
from tools.evaluate_ordered_state import apply_decoder_contract

SAMPLE_RATE = 16000
SAMPLES_PER_FRONTEND_CALL = 160


def event_matches_occurrence(
    event: Mapping[str, Any],
    start_seconds: float,
    end_seconds: float,
    tolerance_seconds: float = 0.0,
) -> bool:
    """Return whether the complete decoded event fits the annotated phrase span."""
    return (
        float(event["start_timestamp"]) >= start_seconds - tolerance_seconds
        and float(event["end_timestamp"]) <= end_seconds + tolerance_seconds
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    sources = payload.get("sources") if isinstance(payload, Mapping) else payload
    if not isinstance(sources, list) or not sources:
        raise ValueError("manifest must contain a non-empty sources list")
    result = []
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping) or not source.get("path"):
            raise ValueError(f"source {index} needs a path")
        item = dict(source)
        item["id"] = str(item.get("id", Path(item["path"]).stem))
        audio_path = Path(item["path"])
        if not audio_path.is_absolute():
            item["path"] = str((path.parent / audio_path).resolve())
        result.append(item)
    return result


def read_wav(path: Path) -> tuple[np.ndarray, float]:
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1:
            raise ValueError(f"{path}: expected mono WAV")
        if source.getsampwidth() != 2 or source.getcomptype() != "NONE":
            raise ValueError(f"{path}: expected uncompressed 16-bit PCM WAV")
        sample_rate = source.getframerate()
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz, got {sample_rate}")
        samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
        return samples, len(samples) / sample_rate


def frontend_features(samples: np.ndarray) -> Iterable[np.ndarray]:
    """Yield the repository C microfrontend's persistent 10 ms outputs."""
    frontend = MicroFrontend()
    process = getattr(frontend, "process_samples", None) or frontend.ProcessSamples
    raw = samples.astype("<i2", copy=False).tobytes()
    offset = 0
    while offset + SAMPLES_PER_FRONTEND_CALL * 2 <= len(raw):
        result = process(raw[offset : offset + SAMPLES_PER_FRONTEND_CALL * 2])
        if result.features is not None and len(result.features) > 0:
            yield np.asarray(result.features, dtype=np.float32)
        samples_read = int(getattr(result, "samples_read", SAMPLES_PER_FRONTEND_CALL))
        if samples_read <= 0:
            raise ValueError("microfrontend returned no samples_read progress")
        offset += samples_read * 2


class QuantizedStreamingModel:
    """One-frame adapter for the repository's internal-state TFLite model."""

    def __init__(self, model_path: Path, interpreter_factory=None):
        self.model_path = model_path
        if interpreter_factory is None:
            try:
                import tensorflow as tf

                interpreter_factory = tf.lite.Interpreter
            except (ImportError, AttributeError):
                from tflite_runtime.interpreter import Interpreter

                interpreter_factory = Interpreter
        self.interpreter_factory = interpreter_factory
        interpreter = interpreter_factory(model_path=str(model_path))
        self._bind_interpreter(interpreter)

    def _bind_interpreter(self, interpreter) -> None:
        interpreter.allocate_tensors()
        self.interpreter = interpreter
        inputs = interpreter.get_input_details()
        outputs = interpreter.get_output_details()
        if len(inputs) != 1:
            raise ValueError("ordered-state streaming artifact must have one input")
        if not outputs:
            raise ValueError("ordered-state streaming artifact has no output")
        self.input = inputs[0]
        self.output = outputs[0]
        input_shape = tuple(int(value) for value in self.input["shape"])
        if len(input_shape) != 3 or input_shape[0] != 1 or input_shape[-1] != 40:
            raise ValueError(
                f"ordered-state streaming input must be [1, stride, 40], got {input_shape}"
            )
        if int(np.prod(self.output["shape"])) != KIZZ_TOPOLOGY.state_count:
            raise ValueError("artifact output must contain exactly 23 state logits")

    def reset(self) -> None:
        """Reset internal TFLite streaming variables at a source boundary."""
        self._bind_interpreter(
            self.interpreter_factory(model_path=str(self.model_path))
        )

    def _input_tensor(self, feature: np.ndarray) -> np.ndarray:
        shape = tuple(int(value) for value in self.input["shape"])
        if int(np.prod(shape)) != feature.size:
            raise ValueError(
                f"artifact input shape {shape} does not match {feature.size} frontend bins"
            )
        values = feature.reshape(shape).astype(np.float32)
        dtype = np.dtype(self.input["dtype"])
        if np.issubdtype(dtype, np.integer):
            scale, zero_point = self.input.get("quantization", (0.0, 0))
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("quantized input needs a positive scale")
            values = np.rint(values / float(scale) + float(zero_point))
            info = np.iinfo(dtype)
            values = np.clip(values, info.min, info.max)
        return values.astype(dtype)

    def step(self, feature: np.ndarray) -> np.ndarray:
        self.interpreter.set_tensor(self.input["index"], self._input_tensor(feature))
        self.interpreter.invoke()
        return tflite_output_logits(
            self.interpreter.get_tensor(self.output["index"]), self.output
        )


def _event_json(event: Any, frame_step: float) -> dict[str, Any]:
    result = dict(vars(event)) if hasattr(event, "__dict__") else dict(event)
    result["start_timestamp"] = result["start_frame"] * frame_step
    # Frames name 30 ms emission intervals. The end is exclusive, so a
    # 21-state path from frame N through N+20 has its full 630 ms extent.
    result["end_timestamp"] = (result["end_frame"] + 1) * frame_step
    result["duration_seconds"] = result["end_timestamp"] - result["start_timestamp"]
    return {key: _json_value(value) for key, value in result.items()}


def _json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        return _json_value(value.item())
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


@dataclass
class _OnlineScoreStats:
    count: int = 0
    finite_count: int = 0
    negative_infinity_count: int = 0
    positive_infinity_count: int = 0
    finite_minimum: float = math.inf
    finite_maximum: float = -math.inf
    finite_sum: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        if math.isfinite(value):
            self.finite_count += 1
            self.finite_minimum = min(self.finite_minimum, value)
            self.finite_maximum = max(self.finite_maximum, value)
            self.finite_sum += value
        elif value == -math.inf:
            self.negative_infinity_count += 1
        elif value == math.inf:
            self.positive_infinity_count += 1

    def as_dict(self) -> dict[str, Any]:
        finite = self.finite_count > 0
        saturated = self.negative_infinity_count + self.positive_infinity_count
        return {
            "count": self.count,
            "finite_count": self.finite_count,
            "negative_infinity_count": self.negative_infinity_count,
            "positive_infinity_count": self.positive_infinity_count,
            "saturated_fraction": saturated / self.count if self.count else 0.0,
            "finite_minimum": self.finite_minimum if finite else None,
            "finite_maximum": self.finite_maximum if finite else None,
            "finite_mean": self.finite_sum / self.finite_count if finite else None,
        }


def score_source(
    source: Mapping[str, Any],
    model: QuantizedStreamingModel,
    decoder_args: Mapping[str, Any],
    frame_step_seconds: float,
    model_hash: str,
    decoder_hash: str,
    completion_margin: float = 0.0,
    cooldown_frames: int = 0,
    full_frame_logits: bool = False,
    occurrence_tolerance_seconds: float = 0.0,
) -> list[dict[str, Any]]:
    path = Path(source["path"])
    samples, exposure = read_wav(path)
    model.reset()
    decoder = OrderedStateDecoder(
        **decoder_args,
        completion_margin=completion_margin,
        cooldown_frames=cooldown_frames,
    )
    decoder.reset()
    records: list[dict[str, Any]] = []
    source_id = str(source["id"])
    session_id = source.get("session_id")
    frame_index = 0
    completion_stats = _OnlineScoreStats()
    event_count = 0
    event_records: list[dict[str, Any]] = []
    feature_batch: list[np.ndarray] = []
    input_shape = tuple(int(value) for value in model.input["shape"])
    stride = input_shape[1]
    if stride <= 0:
        raise ValueError("streaming artifact input stride must be positive")
    for feature in frontend_features(samples):
        feature_batch.append(feature)
        if len(feature_batch) < stride:
            continue
        logits = model.step(np.asarray(feature_batch, dtype=np.float32))
        feature_batch.clear()
        event = decoder.step(logits, frame_index=frame_index)
        completion_stats.update(
            float(event.score)
            if event is not None
            else decoder.current_completion_score
        )
        if full_frame_logits:
            records.append(
                {
                    "type": "score",
                    "source_id": source_id,
                    "session_id": session_id,
                    "frame_index": frame_index,
                    "timestamp": frame_index * frame_step_seconds,
                    "score": logits.tolist(),
                    "exposure_seconds": exposure,
                    "model_sha256": model_hash,
                    "decoder_contract_sha256": decoder_hash,
                }
            )
        if event is not None:
            event_count += 1
            event_record = {
                "type": "event",
                "source_id": source_id,
                "session_id": session_id,
                "exposure_seconds": exposure,
                "model_sha256": model_hash,
                "decoder_contract_sha256": decoder_hash,
                **_event_json(event, frame_step_seconds),
            }
            records.append(event_record)
            event_records.append(event_record)
        frame_index += 1

    if (
        not math.isfinite(occurrence_tolerance_seconds)
        or occurrence_tolerance_seconds < 0
    ):
        raise ValueError("occurrence_tolerance_seconds must be finite and non-negative")
    detected_occurrence_ids = []
    for occurrence_index, occurrence in enumerate(source.get("occurrences", [])):
        if str(source.get("label", "")).lower() not in {
            "positive",
            "wake",
            "target",
            "true",
            "1",
        }:
            raise ValueError("positive occurrences require a positive source label")
        start = float(occurrence["start_seconds"])
        end = float(occurrence["end_seconds"])
        if not (0 <= start < end <= exposure) or not math.isfinite(start + end):
            raise ValueError(f"invalid positive occurrence in {source_id}")
        occurrence_id = str(occurrence.get("id", occurrence_index))
        matches = [
            event
            for event in event_records
            if event_matches_occurrence(event, start, end, occurrence_tolerance_seconds)
        ]
        if matches:
            detected_occurrence_ids.append(occurrence_id)
        records.append(
            {
                "type": "positive_occurrence",
                "source_id": source_id,
                "session_id": session_id,
                "occurrence_id": occurrence_id,
                "start_timestamp": start,
                "end_timestamp": end,
                "label": "positive",
                "exposure_seconds": exposure,
                "model_sha256": model_hash,
                "decoder_contract_sha256": decoder_hash,
                "detected": bool(matches),
                "matching_event_count": len(matches),
                "matching_criterion": "event_contained_in_occurrence",
            }
        )
    occurrence_count = len(source.get("occurrences", []))
    records.append(
        {
            "type": "source_summary",
            "source_id": source_id,
            "session_id": session_id,
            "label": str(source.get("label", "negative")),
            "exposure_seconds": exposure,
            "frame_count": frame_index,
            "event_count": event_count,
            "positive_occurrence_count": occurrence_count,
            "positive_occurrence_ids": [
                str(item.get("id", index))
                for index, item in enumerate(source.get("occurrences", []))
            ],
            "detected_positive_occurrence_count": len(detected_occurrence_ids),
            "detected_positive_occurrence_ids": detected_occurrence_ids,
            "positive_occurrence_recall": (
                len(detected_occurrence_ids) / occurrence_count
                if occurrence_count
                else None
            ),
            "occurrence_tolerance_seconds": occurrence_tolerance_seconds,
            "completion_score_stats": completion_stats.as_dict(),
            "completion_margin": completion_margin,
            "cooldown_frames": cooldown_frames,
            "model_sha256": model_hash,
            "decoder_contract_sha256": decoder_hash,
        }
    )
    return records


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--decoder-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--completion-margin", type=float, default=0.0)
    parser.add_argument("--cooldown-frames", type=int, default=0)
    parser.add_argument("--occurrence-tolerance-seconds", type=float, default=0.0)
    parser.add_argument(
        "--full-frame-logits",
        action="store_true",
        help="Emit one full 23-logit score record per frame; short diagnostics only",
    )
    args = parser.parse_args(argv)
    if not args.model.is_file() or not args.decoder_contract.is_file():
        parser.error("model and decoder contract must be files")
    if not math.isfinite(args.completion_margin):
        parser.error("--completion-margin must be finite")
    if args.cooldown_frames < 0:
        parser.error("--cooldown-frames must be non-negative")
    if (
        not math.isfinite(args.occurrence_tolerance_seconds)
        or args.occurrence_tolerance_seconds < 0
    ):
        parser.error("--occurrence-tolerance-seconds must be finite and non-negative")
    contract = json.loads(args.decoder_contract.read_text())
    decoder_args, frame_step = apply_decoder_contract({}, contract)
    if not math.isclose(frame_step, 0.03, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "ordered-state streaming scorer requires the 30 ms artifact cadence"
        )
    model_hash = sha256_file(args.model)
    decoder_hash = sha256_file(args.decoder_contract)
    model = QuantizedStreamingModel(args.model)
    output = [
        {
            "type": "header",
            "schema_version": 1,
            "frame_step_seconds": frame_step,
            "state_count": KIZZ_TOPOLOGY.state_count,
            "model": str(args.model.resolve()),
            "model_sha256": model_hash,
            "decoder_contract": str(args.decoder_contract.resolve()),
            "decoder_contract_sha256": decoder_hash,
            "completion_margin": args.completion_margin,
            "cooldown_frames": args.cooldown_frames,
            "full_frame_logits": args.full_frame_logits,
        }
    ]
    for source in load_manifest(args.manifest):
        output.extend(
            score_source(
                source,
                model,
                decoder_args,
                frame_step,
                model_hash,
                decoder_hash,
                completion_margin=args.completion_margin,
                cooldown_frames=args.cooldown_frames,
                full_frame_logits=args.full_frame_logits,
                occurrence_tolerance_seconds=args.occurrence_tolerance_seconds,
            )
        )
    args.output.write_text(
        "".join(json.dumps(line, sort_keys=True) + "\n" for line in output)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
