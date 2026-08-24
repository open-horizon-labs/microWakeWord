"""Content hashes for reproducible training and evaluation inputs."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    """Hash a file or directory, following directory symlinks safely.

    Directory hashes frame each file with its lexical path beneath ``path``.
    Symlinked dataset views are followed so a mutation of the consumed target
    changes the hash. Repeated directory targets and cycles are rejected.
    """
    path = Path(path)
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise ValueError(f"training input is not a file or directory: {path}")

    digest = hashlib.sha256()
    visited_directories: dict[tuple[int, int], Path] = {}
    file_count = 0
    for current, directory_names, file_names in os.walk(path, followlinks=True):
        current_path = Path(current)
        identity_stat = current_path.stat()
        identity = (identity_stat.st_dev, identity_stat.st_ino)
        if identity in visited_directories:
            raise ValueError(
                "training input contains a repeated or cyclic directory target: "
                f"{current_path} and {visited_directories[identity]}"
            )
        visited_directories[identity] = current_path
        directory_names.sort()
        file_names.sort()
        for name in file_names:
            child = current_path / name
            if not child.is_file():
                raise ValueError(f"training input contains an unreadable file: {child}")
            relative = child.relative_to(path).as_posix().encode()
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            with child.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            file_count += 1
    if not file_count:
        raise ValueError(f"training input directory is empty: {path}")
    return digest.hexdigest()


__all__ = ["sha256_file", "sha256_path"]
