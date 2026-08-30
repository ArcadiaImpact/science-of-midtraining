"""Disk-backed flat vectors for full-P scoring intermediates (scimt-only).

The streaming SOURCE path holds several [P]-sized vectors (metric diagonals,
basis transitions, per-stage scales). At 12B full coverage each is 43–86 GB;
holding them anonymous killed the phase repeatedly (pod run 20260819T095144Z:
anon RSS 492 GB at SIGKILL with all of them resident), and mmap-backing large
actively-walked arrays pinned page cache to the cgroup limit instead (the
PR #532/#533 factor-loading lesson). These vectors are therefore written once
chunk-wise to plain files and consumed through positional reads into small
reusable buffers: no mapping, page cache stays unmapped-clean (reliably
reclaimable), anonymous residency is O(chunk).

Numerics: values are written exactly as computed (same dtype, same
elementwise operations chunk-by-chunk — IEEE-754 elementwise ops are
independent per element, so chunked evaluation equals whole-array evaluation
bitwise) and reads return the same bytes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

# 64M elements: 512 MB fp64 / 256 MB fp32 per chunk buffer.
DISK_VECTOR_CHUNK = 64 * 1024 * 1024


@dataclass(frozen=True)
class DiskVector:
    """A flat on-disk vector with positional chunk reads (no mmap)."""

    path: Path
    length: int
    dtype: np.dtype

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "dtype", np.dtype(self.dtype))
        if self.length < 0:
            raise ValueError("DiskVector length must be nonnegative")

    def read(self, window: slice) -> np.ndarray:
        start, stop, step = window.indices(self.length)
        if step != 1:
            raise ValueError("DiskVector reads must be contiguous")
        count = max(0, stop - start)
        itemsize = self.dtype.itemsize
        with open(self.path, "rb") as handle:
            data = os.pread(handle.fileno(), count * itemsize, start * itemsize)
        if len(data) != count * itemsize:
            raise ValueError(
                f"DiskVector short read from {self.path}: expected "
                f"{count * itemsize} bytes at offset {start * itemsize}, "
                f"got {len(data)}"
            )
        array = np.frombuffer(data, dtype=self.dtype)
        # np.frombuffer returns a read-only view over `data`; callers may
        # compute in place on their own buffers, never on ours.
        return array

    def delete(self) -> None:
        self.path.unlink(missing_ok=True)


def write_disk_vector(
    path: str | Path,
    length: int,
    dtype: Any,
    chunk_fn: Callable[[slice], np.ndarray],
    *,
    chunk: int = DISK_VECTOR_CHUNK,
    validate: Callable[[slice, np.ndarray], None] | None = None,
) -> DiskVector:
    """Write ``chunk_fn(window)`` sequentially to ``path`` and return a reader.

    ``validate`` runs per chunk BEFORE the write (raise to abort — the
    partial file is removed), replacing whole-array validation walks with
    equal-semantics chunked ones.
    """

    path = Path(path)
    dtype = np.dtype(dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "wb") as handle:
            for start in range(0, length, chunk):
                window = slice(start, min(start + chunk, length))
                values = np.asarray(chunk_fn(window), dtype=dtype).reshape(-1)
                expected = window.stop - window.start
                if values.shape[0] != expected:
                    raise ValueError(
                        f"disk vector chunk for {window} returned "
                        f"{values.shape[0]} elements, expected {expected}"
                    )
                if validate is not None:
                    validate(window, values)
                handle.write(values.tobytes())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return DiskVector(path, length, dtype)


def as_chunk_reader(vector: Any) -> Callable[[slice], np.ndarray]:
    """A uniform ``window -> ndarray`` reader over DiskVector or ndarray."""

    if isinstance(vector, DiskVector):
        return vector.read
    array = np.asarray(vector)

    def _read(window: slice) -> np.ndarray:
        return array[window]

    return _read
