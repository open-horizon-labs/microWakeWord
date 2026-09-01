#!/usr/bin/env python3
"""Design split-bound adult and child voices for a labeled TTS catalog."""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Callable
from urllib import error, request

import yaml

try:
    from tools.add_labeled_voice_samples import AGE_GROUPS, SPLITS
except ModuleNotFoundError:  # Direct execution from the tools directory.
    from add_labeled_voice_samples import AGE_GROUPS, SPLITS


DESIGN_ENDPOINT = "https://api.elevenlabs.io/v1/text-to-voice/design"
CREATE_ENDPOINT = "https://api.elevenlabs.io/v1/text-to-voice"
API_KEY_ENVIRONMENTS = ("ELEVENLABS_API_KEY", "ELEVEN_LABS_API_KEY")


def post_json(endpoint: str, api_key: str, body: dict) -> dict:
    call = request.Request(
        endpoint,
        data=json.dumps(body).encode(),
        method="POST",
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with request.urlopen(call, timeout=120) as response:
            return json.loads(response.read())
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"ElevenLabs request failed with HTTP {exc.code}: {detail}"
        ) from exc


def resolve_api_key(preferred_environment: str | None = None) -> str | None:
    """Return the first configured key without requiring one spelling."""
    environments = (
        (preferred_environment,) if preferred_environment else API_KEY_ENVIRONMENTS
    )
    return next((os.environ.get(name) for name in environments if os.environ.get(name)), None)


def validate_designs(spec: dict) -> None:
    if spec.get("schema_version") != 1:
        raise ValueError("voice design spec requires schema_version 1")
    names = set()
    for voice in spec.get("voices", []):
        required = {"name", "split", "age_group", "description"}
        if not required.issubset(voice):
            raise ValueError(f"voice design requires {sorted(required)}")
        if voice["name"] in names:
            raise ValueError(f"duplicate voice design name: {voice['name']}")
        names.add(voice["name"])
        if voice["split"] not in SPLITS:
            raise ValueError(f"invalid voice split: {voice['split']}")
        if voice["age_group"] not in AGE_GROUPS:
            raise ValueError(f"invalid voice age_group: {voice['age_group']}")


def design_catalog(
    spec_path: Path,
    output: Path,
    preview_dir: Path,
    api_key: str,
    post: Callable[[str, str, dict], dict] = post_json,
) -> dict:
    spec = yaml.safe_load(spec_path.read_text())
    validate_designs(spec)
    existing = yaml.safe_load(output.read_text()) if output.exists() else {}
    resolved = {
        voice["name"]: voice
        for voice in existing.get("voices", [])
        if voice.get("voice_id")
    }
    preview_dir.mkdir(parents=True, exist_ok=True)
    voices = []
    for index, voice in enumerate(spec["voices"]):
        if voice["name"] in resolved:
            voices.append(resolved[voice["name"]])
            continue
        seed = int(spec.get("random_seed", 231)) + index
        designed = post(
            DESIGN_ENDPOINT,
            api_key,
            {
                "voice_description": voice["description"],
                "model_id": spec.get("design_model_id", "eleven_ttv_v3"),
                "text": spec["preview_text"],
                "seed": seed,
                "guidance_scale": float(spec.get("guidance_scale", 5)),
                "should_enhance": False,
            },
        )
        previews = designed.get("previews", [])
        if not previews:
            raise ValueError(f"provider returned no previews for {voice['name']}")
        for preview_index, preview in enumerate(previews):
            (preview_dir / f"{voice['name']}-{preview_index}.mp3").write_bytes(
                base64.b64decode(preview["audio_base_64"])
            )
        selected = int(voice.get("preview_index", 0))
        if selected >= len(previews):
            raise ValueError(f"invalid preview_index for {voice['name']}")
        created = post(
            CREATE_ENDPOINT,
            api_key,
            {
                "voice_name": voice["name"],
                "voice_description": voice["description"],
                "generated_voice_id": previews[selected]["generated_voice_id"],
            },
        )
        voices.append(
            {
                "name": voice["name"],
                "voice_id": created["voice_id"],
                "split": voice["split"],
                "age_group": voice["age_group"],
                "samples_per_phrase": int(voice.get("samples_per_phrase", 1)),
                "design": {
                    "description": voice["description"],
                    "seed": seed,
                    "preview_index": selected,
                },
            }
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "provider": "elevenlabs",
                    "model_id": spec.get("tts_model_id", "eleven_multilingual_v2"),
                    "voices": voices,
                },
                sort_keys=False,
            )
        )
    return yaml.safe_load(output.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview-dir", type=Path, required=True)
    parser.add_argument(
        "--api-key-env",
        help=(
            "Read the key from this environment variable. By default the tool "
            "accepts ELEVENLABS_API_KEY or ELEVEN_LABS_API_KEY."
        ),
    )
    args = parser.parse_args()
    api_key = resolve_api_key(args.api_key_env)
    if not api_key:
        expected = args.api_key_env or " or ".join(API_KEY_ENVIRONMENTS)
        parser.error(f"{expected} is not set")
    design_catalog(args.spec, args.output, args.preview_dir, api_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
