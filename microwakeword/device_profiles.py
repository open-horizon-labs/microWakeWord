"""Validated inventory of product targets and their acoustic profiles."""

from __future__ import annotations

import json
import re
from pathlib import Path

FIRMWARE_STATUSES = {"implemented", "not_implemented", "not_applicable"}
CORPUS_STATUSES = {"collected", "not_collected", "not_applicable"}
PROFILE_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*_v[1-9][0-9]*$")


def load_device_profiles(path: Path) -> dict:
    catalog = json.loads(path.read_text())
    if catalog.get("schema_version") != 1:
        raise ValueError("device profile catalog schema_version must be 1")
    targets = catalog.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("device profile catalog requires targets")

    target_ids: set[str] = set()
    profile_ids: set[str] = set()
    for target in targets:
        target_id = target.get("target_id")
        if not isinstance(target_id, str) or not target_id:
            raise ValueError("each target requires target_id")
        if target_id in target_ids:
            raise ValueError(f"duplicate target_id: {target_id}")
        target_ids.add(target_id)
        if not isinstance(target.get("exact_hardware"), str):
            raise ValueError(f"target {target_id} requires exact_hardware")

        microphone = target.get("microphone")
        enrollment = target.get("enrollment")
        if not isinstance(microphone, dict) or not isinstance(enrollment, dict):
            raise ValueError(f"target {target_id} requires microphone and enrollment")
        present = microphone.get("present")
        if not isinstance(present, bool):
            raise ValueError(f"target {target_id} requires boolean microphone.present")
        evidence = microphone.get("evidence")
        if not isinstance(evidence, str) or not evidence.startswith("https://"):
            raise ValueError(f"target {target_id} requires microphone evidence")

        firmware_status = enrollment.get("firmware_status")
        corpus_status = enrollment.get("corpus_status")
        if firmware_status not in FIRMWARE_STATUSES:
            raise ValueError(f"target {target_id} has invalid firmware_status")
        if corpus_status not in CORPUS_STATUSES:
            raise ValueError(f"target {target_id} has invalid corpus_status")

        profile_id = enrollment.get("device_profile")
        if present:
            if not isinstance(microphone.get("hardware"), str):
                raise ValueError(f"microphone target {target_id} requires hardware")
            if not isinstance(profile_id, str) or not PROFILE_PATTERN.fullmatch(profile_id):
                raise ValueError(f"microphone target {target_id} requires versioned profile")
            if profile_id in profile_ids:
                raise ValueError(f"duplicate device_profile: {profile_id}")
            profile_ids.add(profile_id)
            if firmware_status == "not_applicable" or corpus_status == "not_applicable":
                raise ValueError(f"microphone target {target_id} cannot be not_applicable")
        elif (
            microphone.get("hardware") is not None
            or profile_id is not None
            or firmware_status != "not_applicable"
            or corpus_status != "not_applicable"
        ):
            raise ValueError(f"non-microphone target {target_id} cannot enroll")
    return catalog


def microphone_targets(catalog: dict) -> list[dict]:
    return [target for target in catalog["targets"] if target["microphone"]["present"]]
