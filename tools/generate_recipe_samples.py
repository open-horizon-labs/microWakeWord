#!/usr/bin/env python3
"""Generate collision-free positive and hard-negative TTS corpora from a recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import yaml


def slug(text: str) -> str:
    readable = "_".join(
        "".join(c.lower() if c.isalnum() else " " for c in text).split()
    )
    return f"{readable}-{hashlib.sha256(text.encode()).hexdigest()[:8]}"


def speaker_cohorts(generation: dict) -> dict[str, dict]:
    cohorts = generation.get("speaker_cohorts")
    if not isinstance(cohorts, dict) or set(cohorts) != {
        "train",
        "validation",
        "test",
    }:
        raise ValueError("generation requires train/validation/test speaker_cohorts")
    occupied: set[int] = set()
    fraction = 0.0
    for split, cohort in cohorts.items():
        start = cohort.get("speaker_start")
        end = cohort.get("speaker_end")
        weight = cohort.get("sample_fraction")
        if not isinstance(start, int) or not isinstance(end, int) or end <= start:
            raise ValueError(f"{split} speaker cohort requires a valid half-open range")
        if end > int(generation["max_speakers"]):
            raise ValueError(f"{split} speaker cohort exceeds max_speakers")
        speakers = set(range(start, end))
        if occupied & speakers:
            raise ValueError("synthetic speaker cohorts overlap")
        occupied.update(speakers)
        if not isinstance(weight, (int, float)) or weight <= 0:
            raise ValueError(f"{split} speaker cohort requires sample_fraction > 0")
        fraction += float(weight)
    if not math.isclose(fraction, 1.0):
        raise ValueError("synthetic speaker cohort fractions must sum to 1")
    return cohorts


def split_sample_counts(total: int, cohorts: dict[str, dict]) -> dict[str, int]:
    """Allocate an exact total with deterministic largest-remainder rounding."""
    raw = {
        split: total * float(cohort["sample_fraction"])
        for split, cohort in cohorts.items()
    }
    counts = {split: math.floor(value) for split, value in raw.items()}
    remaining = total - sum(counts.values())
    ranked = sorted(
        raw, key=lambda split: (raw[split] - counts[split], split), reverse=True
    )
    for split in ranked[:remaining]:
        counts[split] += 1
    return counts


def generator_command(
    phrase: dict,
    generation: dict,
    model: Path,
    output_dir: Path,
    batch_size: int,
    cohort: dict,
    samples: int,
    random_seed: int,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "piper_sample_generator",
        phrase["text"],
        "--model",
        str(model),
        "--max-samples",
        str(samples),
        "--batch-size",
        str(batch_size),
        "--output-dir",
        str(output_dir),
    ]
    for option, key in (
        ("--length-scales", "length_scales"),
        ("--noise-scales", "noise_scales"),
        ("--noise-scale-ws", "noise_scale_ws"),
        ("--slerp-weights", "slerp_weights"),
    ):
        command.extend([option, *(str(value) for value in generation[key])])
    command.extend(["--max-speakers", str(generation["max_speakers"])])
    command.extend(
        [
            "--speaker-range",
            str(cohort["speaker_start"]),
            str(cohort["speaker_end"]),
            "--random-seed",
            str(random_seed),
            "--metadata-file",
            str(output_dir / "synthesis-metadata.jsonl"),
        ]
    )
    return command


def generation_signature(command: list[str]) -> list[str]:
    signature = ["<python>", *command[1:]]
    for option, replacement in (
        ("--output-dir", "<output>"),
        ("--metadata-file", "<metadata>"),
    ):
        if option in signature:
            index = signature.index(option)
            signature[index + 1] = replacement
    return signature


def normalized_text(text: str) -> str:
    """Return the identity form used to detect sentence leakage."""
    return " ".join(text.strip().split()).casefold()


def read_connected_text_source(path: Path) -> dict:
    """Read and validate one sentence source without Piper's blank-line filtering."""
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"connected sentence source is not UTF-8: {path}") from error
    lines = []
    seen = set()
    for line_number, line in enumerate(text.splitlines(), 1):
        sentence = " ".join(line.strip().split())
        if not sentence:
            raise ValueError(
                f"connected sentence source has a blank line: {path}:{line_number}"
            )
        identity = normalized_text(sentence)
        if identity in seen:
            raise ValueError(
                f"connected sentence source has a duplicate: {path}:{line_number}"
            )
        seen.add(identity)
        lines.append(
            {
                "id": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
                "text": sentence,
            }
        )
    if not lines:
        raise ValueError(f"connected sentence source is empty: {path}")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "lines": lines,
        "line_count": len(lines),
    }


def validate_connected_sentence_sources(
    entries: list[dict], recipe_dir: Path | None = None
) -> list[dict]:
    """Validate source files and enforce sentence-disjoint train/validation/test splits."""
    if not isinstance(entries, list):
        raise ValueError("connected_sentence_sources must be a list")
    validated = []
    all_ids: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("name"):
            raise ValueError("each connected sentence source requires a name")
        samples_per_text = entry.get("samples_per_text")
        if not isinstance(samples_per_text, int) or samples_per_text <= 0:
            raise ValueError("connected sentence source requires samples_per_text > 0")
        sources = {}
        for split in ("train", "validation", "test"):
            source = entry.get(split)
            if not source:
                raise ValueError(
                    f"connected sentence source {entry['name']} requires {split}, validation, and test files"
                )
            path = Path(source)
            if recipe_dir and not path.is_absolute():
                path = recipe_dir / path
            sources[split] = read_connected_text_source(path)
            for line in sources[split]["lines"]:
                prior = all_ids.get(line["id"])
                if prior is not None:
                    raise ValueError(
                        "connected sentence identity overlaps between "
                        f"{prior} and {entry['name']}:{split}"
                    )
                all_ids[line["id"]] = f"{entry['name']}:{split}"
        validated.append(
            {
                "name": entry["name"],
                "samples_per_text": samples_per_text,
                "sources": sources,
            }
        )
    return validated


def verify_connected_output(
    output_dir: Path,
    source: dict,
    samples_per_text: int,
    speaker_start: int | None = None,
    speaker_end: int | None = None,
) -> dict:
    """Verify every generated WAV is represented by the Piper provenance JSONL."""
    metadata_path = output_dir / "synthesis-metadata.jsonl"
    wavs = sorted(output_dir.glob("*.wav"))
    if not metadata_path.exists():
        raise ValueError(f"missing Piper provenance: {metadata_path}")
    records = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected = {normalized_text(line["text"]): line["id"] for line in source["lines"]}
    actual_files = []
    actual_ids = []
    realized_speakers = set()
    if (speaker_start is None) != (speaker_end is None):
        raise ValueError("speaker_start and speaker_end must be provided together")
    for record in records:
        filename = record.get("file")
        text = record.get("text")
        if not isinstance(filename, str) or not isinstance(text, str):
            raise ValueError(f"invalid Piper provenance record in {metadata_path}")
        identity = normalized_text(text)
        if identity not in expected:
            raise ValueError(f"Piper provenance text is not in source: {text!r}")
        wav = output_dir / filename
        if wav not in wavs:
            raise ValueError(f"Piper provenance references missing WAV: {wav}")
        actual_files.append(wav)
        actual_ids.append(expected[identity])
        if speaker_start is not None and speaker_end is not None:
            for field in ("speaker_1", "speaker_2"):
                speaker = record.get(field)
                if (
                    not isinstance(speaker, int)
                    or not speaker_start <= speaker < speaker_end
                ):
                    raise ValueError(
                        f"Piper {field} is outside the declared split cohort: {speaker}"
                    )
                realized_speakers.add(speaker)
    if len(records) != len(wavs) or len(set(actual_files)) != len(wavs):
        raise ValueError(f"Piper provenance/WAV count mismatch in {output_dir}")
    counts = Counter(actual_ids)
    if set(counts) != set(expected.values()) or any(
        count != samples_per_text for count in counts.values()
    ):
        raise ValueError(f"Piper did not allocate samples evenly in {output_dir}")
    return {
        "metadata": str(metadata_path),
        "wav_count": len(wavs),
        "line_ids": sorted(counts),
        "realized_speaker_ids": sorted(realized_speakers),
    }


def validate_realized_speaker_isolation(items: list[dict]) -> None:
    """Reject any realized Piper base speaker shared across connected splits."""
    by_split: dict[str, set[int]] = {}
    for item in items:
        if item.get("text_source"):
            by_split.setdefault(item["split"], set()).update(
                item.get("realized_speaker_ids", [])
            )
    for split, speakers in by_split.items():
        for other_split, other_speakers in by_split.items():
            if split >= other_split:
                continue
            overlap = speakers & other_speakers
            if overlap:
                raise ValueError(
                    f"realized Piper speakers overlap between {split} and "
                    f"{other_split}: {sorted(overlap)}"
                )


def connected_generation_plan(
    sources: list[dict],
    generation: dict,
    model: Path,
    output: Path,
    batch_size: int,
    random_seed: int,
) -> list[dict]:
    """Build one deterministic Piper command per connected source and split."""
    plan = []
    cohorts = speaker_cohorts(generation)
    for source in sources:
        for split, cohort in cohorts.items():
            source_info = source["sources"][split]
            total = source_info["line_count"] * source["samples_per_text"]
            if total <= 0:
                raise ValueError(
                    f"connected source {source['name']} has zero allocation"
                )
            output_dir = output / "hard_negative" / source["name"] / split
            command = generator_command(
                {"text": source_info["path"]},
                generation,
                model,
                output_dir,
                batch_size,
                cohort,
                total,
                random_seed + len(plan),
            )
            plan.append(
                {
                    "class": "hard_negative",
                    "text": None,
                    "text_source": source["name"],
                    "source_path": source_info["path"],
                    "source_sha256": source_info["sha256"],
                    "normalized_line_ids": [
                        line["id"] for line in source_info["lines"]
                    ],
                    "line_count": source_info["line_count"],
                    "samples_per_text": source["samples_per_text"],
                    "samples": total,
                    "split": split,
                    "speaker_start": cohort["speaker_start"],
                    "speaker_end": cohort["speaker_end"],
                    "age_group": cohort["age_group"],
                    "seed": random_seed + len(plan),
                    "output": str(output_dir),
                    "command": command,
                }
            )
    return plan


def reusable_phrase_source(
    manifests: list[dict],
    text: str,
    samples: int,
    model_sha256: str | None,
    expected_command: list[str],
) -> Path | None:
    """Find audio generated by the same model, even if its label changed."""
    for manifest in manifests:
        if manifest.get("generator_model_sha256") != model_sha256:
            continue
        for item in manifest.get("plan", []):
            if (
                item.get("text") == text
                and item.get("samples") == samples
                and generation_signature(item.get("command", []))
                == generation_signature(expected_command)
            ):
                source = Path(item["output"])
                if len(list(source.glob("*.wav"))) == samples:
                    return source
    return None


def hardlink_phrase_corpus(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for wav in sorted(source.glob("*.wav")):
        target = destination / wav.name
        if not target.exists():
            os.link(wav, target)
    metadata = source / "synthesis-metadata.jsonl"
    if metadata.exists():
        shutil.copy2(metadata, destination / metadata.name)


def reusable_connected_source(
    manifests: list[dict], item: dict, model_sha256: str | None
) -> Path | None:
    """Find a complete connected-source directory with identical provenance."""
    for manifest in manifests:
        if manifest.get("generator_model_sha256") != model_sha256:
            continue
        for prior in manifest.get("plan", []):
            same_identity = (
                prior.get("text_source") == item.get("text_source")
                and prior.get("source_sha256") == item.get("source_sha256")
                and prior.get("source_path") == item.get("source_path")
                and prior.get("split") == item.get("split")
                and prior.get("samples_per_text") == item.get("samples_per_text")
                and prior.get("normalized_line_ids") == item.get("normalized_line_ids")
            )
            if not same_identity:
                continue
            if generation_signature(prior.get("command", [])) != generation_signature(
                item["command"]
            ):
                continue
            source = Path(prior["output"])
            try:
                verify_connected_output(
                    source,
                    {
                        "lines": [
                            {"id": line_id, "text": text}
                            for line_id, text in zip(
                                item["normalized_line_ids"],
                                item["normalized_texts"],
                            )
                        ]
                    },
                    item["samples_per_text"],
                    item["speaker_start"],
                    item["speaker_end"],
                )
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            return source
    return None


def hardlink_corpus(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for wav in sorted(source.glob("*.wav")):
        target = destination / wav.name
        if not target.exists():
            os.link(wav, target)
    metadata = source / "synthesis-metadata.jsonl"
    if metadata.exists():
        shutil.copy2(metadata, destination / metadata.name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--generator-source",
        type=Path,
        help="Source checkout to add to PYTHONPATH (needed by current Piper source)",
    )
    parser.add_argument(
        "--reuse-generated",
        type=Path,
        action="append",
        default=[],
        help="Hardlink phrase audio from compatible prior generated corpora",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    recipe_bytes = args.recipe.read_bytes()
    recipe = yaml.safe_load(recipe_bytes)
    cohorts = speaker_cohorts(recipe["generation"])
    model_sha256 = (
        hashlib.sha256(args.model.read_bytes()).hexdigest()
        if args.model.exists()
        else None
    )
    reuse_manifests = []
    for reusable in args.reuse_generated:
        manifest_path = reusable / "generation-manifest.json"
        if not manifest_path.exists():
            raise ValueError(f"missing reusable generation manifest: {manifest_path}")
        reuse_manifests.append(json.loads(manifest_path.read_text()))
    plan = []
    for class_name, key in (
        ("positive", "positive_phrases"),
        ("hard_negative", "hard_negative_phrases"),
    ):
        for phrase in recipe[key]:
            phrase_slug = slug(phrase["text"])
            counts = split_sample_counts(phrase["samples"], cohorts)
            for split, cohort in cohorts.items():
                phrase_dir = args.output / class_name / phrase_slug / split
                seed = int(recipe["random_seed"]) + len(plan)
                command = generator_command(
                    phrase,
                    recipe["generation"],
                    args.model,
                    phrase_dir,
                    args.batch_size,
                    cohort,
                    counts[split],
                    seed,
                )
                plan.append(
                    {
                        "class": class_name,
                        "text": phrase["text"],
                        "group": phrase_slug,
                        "split": split,
                        "samples": counts[split],
                        "speaker_start": cohort["speaker_start"],
                        "speaker_end": cohort["speaker_end"],
                        "age_group": cohort["age_group"],
                        "output": str(phrase_dir),
                        "command": command,
                    }
                )
                if args.dry_run:
                    continue
                reusable = reusable_phrase_source(
                    reuse_manifests,
                    phrase["text"],
                    counts[split],
                    model_sha256,
                    command,
                )
                if reusable is not None and not phrase_dir.exists():
                    hardlink_phrase_corpus(reusable, phrase_dir)
                    plan[-1]["reused_from"] = str(reusable)
                metadata_path = phrase_dir / "synthesis-metadata.jsonl"
                if phrase_dir.exists():
                    existing = len(list(phrase_dir.glob("*.wav")))
                    metadata_lines = (
                        len(metadata_path.read_text().splitlines())
                        if metadata_path.exists()
                        else 0
                    )
                    if existing == counts[split] and metadata_lines == counts[split]:
                        continue
                    if existing > counts[split]:
                        raise ValueError(
                            f"{phrase_dir} has {existing} WAVs but recipe requests "
                            f"{counts[split]}; move it aside before shrinking"
                        )
                phrase_dir.mkdir(parents=True, exist_ok=True)
                environment = os.environ.copy()
                if args.generator_source:
                    prior = environment.get("PYTHONPATH")
                    environment["PYTHONPATH"] = str(args.generator_source)
                    if prior:
                        environment["PYTHONPATH"] += os.pathsep + prior
                completed = subprocess.run(command, check=False, env=environment)
                generated = len(list(phrase_dir.glob("*.wav")))
                if completed.returncode != 0 and generated != counts[split]:
                    completed.check_returncode()

    connected_entries = validate_connected_sentence_sources(
        recipe.get("connected_sentence_sources", []), args.recipe.parent
    )
    connected_plan = connected_generation_plan(
        connected_entries,
        recipe["generation"],
        args.model,
        args.output,
        args.batch_size,
        int(recipe["random_seed"]) + len(plan),
    )
    for item in connected_plan:
        source = next(
            source["sources"][item["split"]]
            for source in connected_entries
            if source["name"] == item["text_source"]
        )
        item["normalized_texts"] = [line["text"] for line in source["lines"]]
        output_dir = Path(item["output"])
        if args.dry_run:
            plan.append(item)
            continue
        reusable = reusable_connected_source(reuse_manifests, item, model_sha256)
        if reusable is not None and not output_dir.exists():
            hardlink_corpus(reusable, output_dir)
            item["reused_from"] = str(reusable)
        if output_dir.exists():
            wav_count = len(list(output_dir.glob("*.wav")))
            metadata_path = output_dir / "synthesis-metadata.jsonl"
            metadata_count = (
                len(metadata_path.read_text(encoding="utf-8").splitlines())
                if metadata_path.exists()
                else 0
            )
            if wav_count == item["samples"] and metadata_count == item["samples"]:
                verified = verify_connected_output(
                    output_dir,
                    source,
                    item["samples_per_text"],
                    item["speaker_start"],
                    item["speaker_end"],
                )
                item.update(verified)
                plan.append(item)
                continue
            if wav_count > item["samples"]:
                raise ValueError(
                    f"{output_dir} has {wav_count} WAVs but recipe requests {item['samples']}"
                )
        output_dir.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        if args.generator_source:
            prior = environment.get("PYTHONPATH")
            environment["PYTHONPATH"] = str(args.generator_source)
            if prior:
                environment["PYTHONPATH"] += os.pathsep + prior
        completed = subprocess.run(item["command"], check=False, env=environment)
        generated = len(list(output_dir.glob("*.wav")))
        if completed.returncode != 0 and generated != item["samples"]:
            completed.check_returncode()
        verified = verify_connected_output(
            output_dir,
            source,
            item["samples_per_text"],
            item["speaker_start"],
            item["speaker_end"],
        )
        item.update(verified)
        plan.append(item)

    validate_realized_speaker_isolation(plan)

    manifest = {
        "schema_version": 2,
        "recipe": str(args.recipe),
        "recipe_sha256": hashlib.sha256(recipe_bytes).hexdigest(),
        "generator_model": str(args.model),
        "generator_model_sha256": model_sha256,
        "generator_source": (
            str(args.generator_source) if args.generator_source else None
        ),
        "plan": plan,
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "generation-manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
