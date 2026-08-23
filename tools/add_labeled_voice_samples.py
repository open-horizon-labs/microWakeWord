#!/usr/bin/env python3
"""Add age-labeled TTS voices to a generated recipe corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Callable
from urllib import parse, request
import wave

import yaml

SPLITS = {"train", "validation", "test"}
AGE_GROUPS = {"adult", "child"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def slug(text: str) -> str:
    readable = "_".join(
        "".join(c.lower() if c.isalnum() else " " for c in text).split()
    )
    return f"{readable}-{hashlib.sha256(text.encode()).hexdigest()[:8]}"


def load_catalog(path: Path) -> dict:
    catalog = yaml.safe_load(path.read_text())
    if catalog.get("schema_version") != 1:
        raise ValueError("voice catalog requires schema_version 1")
    if catalog.get("provider") != "elevenlabs":
        raise ValueError("unsupported voice provider")
    if not isinstance(catalog.get("voices"), list) or not catalog["voices"]:
        raise ValueError("voice catalog requires at least one voice")
    identities: dict[str, str] = {}
    names: set[str] = set()
    for voice in catalog["voices"]:
        required = {"name", "voice_id", "split", "age_group"}
        if not required.issubset(voice):
            raise ValueError(f"voice requires {sorted(required)}")
        if voice["name"] in names:
            raise ValueError(f"duplicate voice name: {voice['name']}")
        names.add(voice["name"])
        if voice["split"] not in SPLITS:
            raise ValueError(f"invalid voice split: {voice['split']}")
        if voice["age_group"] not in AGE_GROUPS:
            raise ValueError(f"invalid voice age_group: {voice['age_group']}")
        identity = f"elevenlabs:{voice['voice_id']}"
        prior = identities.setdefault(identity, voice["split"])
        if prior != voice["split"]:
            raise ValueError(f"voice identity crosses {prior} and {voice['split']}")
        for field in (
            "samples_per_phrase",
            "positive_samples_per_phrase",
            "hard_negative_samples_per_phrase",
        ):
            samples = int(voice.get(field, 1))
            if samples < 1:
                raise ValueError(f"{field} must be positive")
    return catalog


def samples_for_class(voice: dict, class_name: str) -> int:
    """Resolve a class-specific sample count with legacy catalog fallback."""
    return int(
        voice.get(
            f"{class_name}_samples_per_phrase",
            voice.get("samples_per_phrase", 1),
        )
    )


def elevenlabs_pcm(
    api_key: str,
    voice_id: str,
    text: str,
    model_id: str,
    seed: int,
    voice_settings: dict,
) -> bytes:
    endpoint = (
        "https://api.elevenlabs.io/v1/text-to-speech/"
        + parse.quote(voice_id, safe="")
        + "?output_format=pcm_16000"
    )
    body = json.dumps(
        {
            "text": text,
            "model_id": model_id,
            "language_code": "en",
            "seed": seed,
            "voice_settings": voice_settings,
        }
    ).encode()
    call = request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/pcm",
        },
    )
    with request.urlopen(call, timeout=60) as response:
        return response.read()


def write_pcm_wav(path: Path, pcm: bytes) -> None:
    if not pcm or len(pcm) % 2:
        raise ValueError("TTS provider returned invalid signed-16 PCM")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".wav.partial")
    with wave.open(str(temporary), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(pcm)
    temporary.replace(path)


def _variant(catalog: dict, sample_index: int) -> dict:
    variants = catalog.get("voice_settings") or [
        {"stability": 0.35, "similarity_boost": 0.75, "speed": 0.9},
        {"stability": 0.5, "similarity_boost": 0.8, "speed": 1.0},
        {"stability": 0.7, "similarity_boost": 0.85, "speed": 1.1},
    ]
    return variants[sample_index % len(variants)]


def add_samples(
    recipe_path: Path,
    generated: Path,
    catalog_path: Path,
    api_key: str,
    synthesize: Callable[..., bytes] = elevenlabs_pcm,
    allow_catalog_update: bool = False,
    initialize_manifest: bool = False,
) -> dict:
    recipe = yaml.safe_load(recipe_path.read_text())
    manifest_path = generated / "generation-manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    elif initialize_manifest:
        generated.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": 2,
            "recipe": str(recipe_path),
            "recipe_sha256": sha256(recipe_path),
            "generator_model": None,
            "generator_model_sha256": None,
            "generator_source": "labeled-voices-only",
            "plan": [],
        }
    else:
        raise ValueError(f"missing generation manifest: {manifest_path}")
    if manifest.get("schema_version") != 2:
        raise ValueError("generated corpus requires schema_version 2")
    if manifest.get("recipe_sha256") != sha256(recipe_path):
        raise ValueError("generated corpus does not match the recipe")
    catalog = load_catalog(catalog_path)
    catalog_hash = sha256(catalog_path)
    prior_catalog = manifest.get("labeled_voice_catalog_sha256")
    if prior_catalog and prior_catalog != catalog_hash and not allow_catalog_update:
        raise ValueError("generated corpus already uses a different voice catalog")

    base_plan = [item for item in manifest["plan"] if not item.get("speaker_id")]
    labeled_plan = []
    model_id = catalog.get("model_id", "eleven_multilingual_v2")
    random_seed = int(recipe["random_seed"])
    for class_name, key in (
        ("positive", "positive_phrases"),
        ("hard_negative", "hard_negative_phrases"),
    ):
        for phrase_index, phrase in enumerate(recipe[key]):
            group = slug(phrase["text"])
            for voice_index, voice in enumerate(catalog["voices"]):
                samples = samples_for_class(voice, class_name)
                output = (
                    generated
                    / class_name
                    / group
                    / voice["split"]
                    / f"labeled-{slug(voice['name'])}"
                )
                plan_item = {
                    "class": class_name,
                    "text": phrase["text"],
                    "group": group,
                    "split": voice["split"],
                    "samples": samples,
                    "speaker_id": voice["voice_id"],
                    "speaker_name": voice["name"],
                    "provider": "elevenlabs",
                    "age_group": voice["age_group"],
                    "output": str(output),
                }
                labeled_plan.append(plan_item)

                output.mkdir(parents=True, exist_ok=True)
                metadata_path = output / "synthesis-metadata.jsonl"
                existing_metadata = {}
                if metadata_path.exists():
                    existing_metadata = {
                        record["file"]: record
                        for record in (
                            json.loads(line)
                            for line in metadata_path.read_text().splitlines()
                            if line.strip()
                        )
                    }
                for sample_index in range(samples):
                    seed = (
                        random_seed
                        + phrase_index * 100_000
                        + voice_index * 1_000
                        + sample_index
                    )
                    filename = f"{sample_index:04d}-{seed}.wav"
                    target = output / filename
                    if target.exists() and filename in existing_metadata:
                        continue
                    settings = _variant(catalog, sample_index)
                    pcm = synthesize(
                        api_key,
                        voice["voice_id"],
                        phrase["text"],
                        model_id,
                        seed,
                        settings,
                    )
                    write_pcm_wav(target, pcm)
                    existing_metadata[filename] = {
                        "file": filename,
                        "text": phrase["text"],
                        "speaker_id": voice["voice_id"],
                        "speaker_name": voice["name"],
                        "provider": "elevenlabs",
                        "age_group": voice["age_group"],
                        "model_id": model_id,
                        "seed": seed,
                        "voice_settings": settings,
                    }
                    metadata_path.write_text(
                        "".join(
                            json.dumps(record, sort_keys=True) + "\n"
                            for _, record in sorted(existing_metadata.items())
                        )
                    )

    manifest["plan"] = [*base_plan, *labeled_plan]
    manifest["labeled_voice_catalog"] = str(catalog_path)
    manifest["labeled_voice_catalog_sha256"] = catalog_hash
    with tempfile.NamedTemporaryFile(
        "w", dir=generated, prefix="generation-manifest.", delete=False
    ) as temporary:
        json.dump(manifest, temporary, indent=2)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--voice-catalog", type=Path, required=True)
    parser.add_argument("--api-key-env", default="ELEVENLABS_API_KEY")
    parser.add_argument(
        "--update-catalog",
        action="store_true",
        help="Allow a revised catalog to extend an existing generated corpus",
    )
    parser.add_argument(
        "--initialize",
        action="store_true",
        help=(
            "Initialize a labeled-voices-only corpus when no generation "
            "manifest exists"
        ),
    )
    args = parser.parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        parser.error(f"{args.api_key_env} is not set")
    add_samples(
        args.recipe,
        args.generated,
        args.voice_catalog,
        api_key,
        allow_catalog_update=args.update_catalog,
        initialize_manifest=args.initialize,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
