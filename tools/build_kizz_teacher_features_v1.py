#!/usr/bin/env python3
"""Materialize clean-slate fixed frontend windows for teacher C.

This first pass deliberately uses the sequence objective only.  It writes a
zero placeholder target array with an explicit ``frame_supervision`` marker;
the trainer must be invoked with ``--frame-weight 0``.  No phone boundaries are
invented from clip duration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from microwakeword.audio.audio_utils import MicroFrontend

INPUT_FRAMES = 260
FEATURE_BINS = 40
OUTPUT_FRAMES = 66
SAMPLES_PER_CALL = 160


def frontend(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    samples = np.asarray(samples)
    if samples.ndim == 2:
        samples = np.mean(samples, axis=1)
    if sample_rate != 16_000:
        samples = resample_poly(samples, 16_000, sample_rate).astype(np.float32)
    # A few licensed noise clips are shorter than one 10 ms frontend hop.
    # Treat them as short silence-plus-noise examples rather than dropping
    # them or failing the entire corpus materialization.
    minimum_samples = 3_200  # 200 ms lets the frontend warm up reliably.
    if len(samples) < minimum_samples:
        samples = np.pad(samples, (0, minimum_samples - len(samples)))
    samples = np.clip(samples, -1.0, 1.0)
    pcm = np.rint(samples * 32767.0).astype("<i2", copy=False)
    processor = MicroFrontend()
    process = getattr(processor, "process_samples", None) or processor.ProcessSamples
    rows = []
    raw = pcm.tobytes()
    offset = 0
    while offset + SAMPLES_PER_CALL * 2 <= len(raw):
        result = process(raw[offset : offset + SAMPLES_PER_CALL * 2])
        values = np.asarray(result.features, dtype=np.float32)
        if values.ndim == 2:
            values = values[0]
        if values.shape == (FEATURE_BINS,):
            rows.append(values)
        samples_read = int(getattr(result, "samples_read", SAMPLES_PER_CALL))
        if samples_read <= 0:
            raise ValueError("microfrontend made no progress")
        offset += samples_read * 2
    if not rows:
        raise ValueError("audio produced no frontend frames")
    return np.stack(rows)


def fixed_window(path: Path) -> np.ndarray:
    samples, rate = sf.read(path, dtype="float32", always_2d=False)
    values = frontend(samples, rate)
    if len(values) >= INPUT_FRAMES:
        start = (len(values) - INPUT_FRAMES) // 2
        return values[start : start + INPUT_FRAMES]
    result = np.zeros((INPUT_FRAMES, FEATURE_BINS), dtype=np.float32)
    result[: len(values)] = values
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    split_reports = {}
    for split in ("train", "validation", "test"):
        rows = [item for item in payload["examples"] if item.get("split") == split]
        positives = [item for item in rows if int(item["label"]) == 1]
        negatives: dict[str, list[np.ndarray]] = {}
        for item in rows:
            if int(item["label"]) == 0:
                negatives.setdefault(item["source_group"], []).append(
                    fixed_window(Path(item["path"]))
                )
        if positives:
            np.save(
                args.output / f"positive_features-{split}.npy",
                np.stack([fixed_window(Path(item["path"])) for item in positives]),
            )
            np.save(
                args.output / f"positive_targets-{split}.npy",
                np.zeros((len(positives), OUTPUT_FRAMES), dtype=np.int32),
            )
        negative_paths = {}
        for source, values in sorted(negatives.items()):
            path = args.output / f"negative-{split}-{source}.npy"
            np.save(path, np.stack(values).astype(np.float32, copy=False))
            negative_paths[source] = str(path.resolve())
        split_reports[split] = {
            "positive_count": len(positives),
            "negative_counts": {
                source: len(values) for source, values in sorted(negatives.items())
            },
            "negative_sources": negative_paths,
        }
    train = [item for item in payload["examples"] if item.get("split") == "train"]
    positives = [item for item in train if int(item["label"]) == 1]
    negatives = {}
    for item in train:
        if int(item["label"]) == 0:
            negatives.setdefault(item["source_group"], []).append(item)
    report = {
        "schema_version": 1,
        "manifest": str(args.manifest.resolve()),
        "positive_count": len(positives),
        "negative_counts": {
            source: len(values) for source, values in sorted(negatives.items())
        },
        "splits": split_reports,
        "frame_supervision": "none_sequence_only",
        "input_shape": [INPUT_FRAMES, FEATURE_BINS],
    }
    (args.output / "feature-provenance.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
