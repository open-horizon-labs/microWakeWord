"""Human-reviewed promotion of quarantined false-wake observations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

from microwakeword.device_corpus import MANIFEST_NAME, validate_device_corpus


def promote_false_wake(
    corpus: Path,
    observation_id: str,
    *,
    reviewer: str,
    split: str,
    speaker_id: str,
    session_id: str,
    reason: str,
) -> dict:
    """Promote one reviewed observation into the device corpus.

    This is intentionally a separate, explicit operation. The source evidence
    remains in observations/false-wakes after promotion.
    """
    if not reviewer.strip() or not reason.strip():
        raise ValueError("reviewer and reason are required")
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    if not speaker_id or not session_id:
        raise ValueError("speaker_id and session_id are required")

    observation_dir = corpus / "observations" / "false-wakes"
    metadata_path = observation_dir / f"{observation_id}.json"
    source_path = observation_dir / f"{observation_id}.wav"
    if not metadata_path.is_file() or not source_path.is_file():
        raise ValueError(f"quarantined observation is missing: {observation_id}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("kind") != "false_wake_no_command":
        raise ValueError("observation is not a false_wake_no_command")
    if metadata.get("promoted_capture_id"):
        raise ValueError("observation has already been promoted")

    manifest_path = corpus / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    validate_device_corpus(corpus)
    speaker = manifest.get("speakers", {}).get(speaker_id)
    if speaker is None:
        raise ValueError(f"speaker is not registered: {speaker_id}")
    if speaker.get("split") != split:
        raise ValueError("split differs from registered speaker split")
    for capture in manifest["captures"]:
        if capture["session_id"] == session_id and capture["split"] != split:
            raise ValueError("session would cross corpus splits")
        if capture["device_id"] == metadata["device_id"] and capture["device_profile"] != metadata["device_profile"]:
            raise ValueError("device would cross device profiles")

    capture_id = f"hard-negative-{observation_id}"
    if any(item["capture_id"] == capture_id for item in manifest["captures"]):
        raise ValueError(f"capture already exists: {capture_id}")
    destination = corpus / "audio" / f"{capture_id}.wav"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, destination)
    entry = {
        "capture_id": capture_id,
        "path": str(destination.relative_to(corpus)),
        "truth": "hard_negative",
        "source": "ambient",
        "phrase": "unlabeled ambient false wake",
        "speaker_id": speaker_id,
        "session_id": session_id,
        "split": split,
        "detected": False,
        "samples": metadata["samples"],
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "device_id": metadata["device_id"],
        "device_profile": metadata["device_profile"],
        "firmware_sha": metadata.get("firmware_sha"),
        "conditions": {"reviewed_false_wake": observation_id},
        "review": {"reviewer": reviewer, "reason": reason},
    }
    manifest["captures"].append(entry)
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    prior_manifest = manifest_path.read_bytes()
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary_manifest.replace(manifest_path)
    try:
        validate_device_corpus(corpus)
    except Exception:
        manifest_path.write_bytes(prior_manifest)
        destination.unlink(missing_ok=True)
        raise

    metadata["review"] = {"reviewer": reviewer, "reason": reason}
    metadata["promoted_capture_id"] = capture_id
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return entry
