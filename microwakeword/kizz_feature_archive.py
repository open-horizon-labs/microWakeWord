"""Read-only access to current and legacy microWakeWord feature archives.

The public negative archives used by the Kizz experiments predate the
``mmap_ninja`` metadata sidecars added in 0.9.  Their data, offsets, and shapes
are complete, but constructing ``RaggedMmap`` silently reports length zero.
This module recognizes that exact legacy layout, validates every boundary and
shape against the data file, and exposes it without modifying the archive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from mmap_ninja.ragged import RaggedMmap

FEATURE_BINS = 40
UINT16_FRONTEND_SCALE = 0.0390625


def decode_frontend_features(value: np.ndarray) -> np.ndarray:
    """Return product-scale float features from either archive representation.

    Historical mmap archives store the microfrontend's quantized uint16 bins;
    directly casting those bins to float makes them 25.6 times larger than
    freshly generated product features.  That train/eval channel mismatch was
    present in the archived teacher recipe, so decoding is centralized and
    mandatory here.
    """

    values = np.asarray(value)
    if values.ndim != 2 or values.shape[1] != FEATURE_BINS:
        raise ValueError("frontend features must have shape [frames, 40]")
    if values.dtype == np.uint16:
        decoded = values.astype(np.float32) * UINT16_FRONTEND_SCALE
    elif np.issubdtype(values.dtype, np.floating):
        decoded = values.astype(np.float32, copy=False)
    else:
        raise ValueError(f"unsupported frontend feature dtype: {values.dtype}")
    if np.any(~np.isfinite(decoded)):
        raise ValueError("frontend features contain non-finite values")
    return decoded


@runtime_checkable
class FeatureArchive(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> np.ndarray: ...


class LegacyRaggedFeatureArchive:
    """Validated reader for the metadata-light historical archive format."""

    def __init__(self, path: Path, *, feature_bins: int = FEATURE_BINS) -> None:
        self.path = Path(path).resolve()
        if feature_bins < 1:
            raise ValueError("feature_bins must be positive")
        required = {
            "data": self.path / "data.ninja",
            "starts": self.path / "starts" / "data.ninja",
            "ends": self.path / "ends" / "data.ninja",
            "shapes": self.path / "shapes" / "data.ninja",
        }
        missing = [name for name, item in required.items() if not item.is_file()]
        if missing:
            raise ValueError(
                f"legacy feature archive is missing {sorted(missing)}: {self.path}"
            )
        self._starts = np.memmap(required["starts"], dtype="<i8", mode="r")
        self._ends = np.memmap(required["ends"], dtype="<i8", mode="r")
        flat_shapes = np.memmap(required["shapes"], dtype="<i8", mode="r")
        if not len(self._starts) or len(self._starts) != len(self._ends):
            raise ValueError("legacy feature archive offsets are empty or mismatched")
        if len(flat_shapes) != 2 * len(self._starts):
            raise ValueError("legacy feature archive shapes must be [N, 2]")
        self._shapes = np.asarray(flat_shapes).reshape(-1, 2)
        if (
            int(self._starts[0]) != 0
            or np.any(self._ends <= self._starts)
            or np.any(self._starts[1:] != self._ends[:-1])
            or np.any(self._shapes[:, 0] < 1)
            or np.any(self._shapes[:, 1] != feature_bins)
        ):
            raise ValueError("legacy feature archive has invalid offsets or shapes")
        flattened = self._shapes[:, 0] * self._shapes[:, 1]
        if np.any(flattened != self._ends - self._starts):
            raise ValueError("legacy feature archive shapes disagree with offsets")
        element_count = int(self._ends[-1])
        data_bytes = required["data"].stat().st_size
        if element_count < 1 or data_bytes % element_count:
            raise ValueError("legacy feature archive data size is inconsistent")
        width = data_bytes // element_count
        if width == 2:
            # Historical microfrontend archives stored their non-negative
            # integer feature bins directly as little-endian uint16.
            dtype = "<u2"
        elif width == 4:
            dtype = "<f4"
        else:
            raise ValueError("legacy feature archive data size is inconsistent")
        self._data = np.memmap(required["data"], dtype=dtype, mode="r")

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, index: int) -> np.ndarray:
        if not isinstance(index, (int, np.integer)):
            raise TypeError("legacy feature archive indexes must be integers")
        normalized = int(index)
        if normalized < 0:
            normalized += len(self)
        if normalized < 0 or normalized >= len(self):
            raise IndexError(index)
        start = int(self._starts[normalized])
        end = int(self._ends[normalized])
        shape = tuple(int(value) for value in self._shapes[normalized])
        return np.asarray(self._data[start:end]).reshape(shape)


def open_feature_archive(path: Path) -> FeatureArchive:
    """Open a feature archive and fail if either layout is incomplete."""

    resolved = Path(path).resolve()
    if not resolved.is_dir():
        raise ValueError(f"feature archive does not exist: {resolved}")
    if (resolved / "shapes_are_flat.ninja").is_file():
        archive = RaggedMmap(resolved)
        if len(archive) < 1:
            raise ValueError(f"feature archive is empty: {resolved}")
        return archive
    return LegacyRaggedFeatureArchive(resolved)


__all__ = [
    "FEATURE_BINS",
    "UINT16_FRONTEND_SCALE",
    "FeatureArchive",
    "LegacyRaggedFeatureArchive",
    "decode_frontend_features",
    "open_feature_archive",
]
