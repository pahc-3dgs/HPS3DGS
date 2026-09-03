"""Evaluation metrics and compression statistics."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Iterable

from .codec import estimate_payload_bytes


def file_bytes(paths: str | Path | Iterable[str | Path]):
    """Sum real on-disk sizes of files/directories (actual bitstream bytes)."""

    total = 0
    if isinstance(paths, (str, Path, os.PathLike)):
        paths = [paths]
    for item in paths:
        item = Path(item)
        if item.is_dir():
            total += file_bytes(sorted(item.rglob("*")))
        elif item.is_file():
            total += item.stat().st_size
    return int(total)


def sha256_file(path: str | Path, chunk_size: int = 1 << 20):
    """SHA-256 of a file, hex encoded."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compression_report(
    original_count: int,
    final_count: int,
    original_bytes: int | None = None,
    final_payload: Any | None = None,
    final_bytes: int | None = None,
    bitstream_paths: str | Path | Iterable[str | Path] | None = None,
    estimated_tensor_bytes: int | None = None,
):
    """Build a JSON-friendly compression report.

    ``bitstream_bytes`` counts real files on disk and is the number that should
    be reported as the compressed size. ``estimated_tensor_bytes`` (from
    :func:`pahc.codec.estimate_payload_bytes`) is an in-memory estimate of the
    serialised payload and is reported separately - it must not be presented as
    an actual bitstream size.
    """

    if original_bytes is None:
        original_bytes = original_count * (3 + 3 + 45 + 1 + 3 + 4) * 4
    if estimated_tensor_bytes is None and final_payload is not None:
        estimated_tensor_bytes = estimate_payload_bytes(final_payload)
    actual_bytes = None
    if bitstream_paths is not None:
        actual_bytes = file_bytes(bitstream_paths)
    if final_bytes is not None:
        # Backwards compatible alias for the actual size.
        actual_bytes = int(final_bytes)
    if actual_bytes is None:
        if estimated_tensor_bytes is None:
            estimated_tensor_bytes = final_count * (3 + 3 + 45 + 1 + 3 + 4) * 4
        actual_bytes = estimated_tensor_bytes

    report = {
        "original_count": int(original_count),
        "final_count": int(final_count),
        "original_bytes": int(original_bytes),
        "estimated_tensor_bytes": int(estimated_tensor_bytes) if estimated_tensor_bytes is not None else None,
        "bitstream_bytes": int(actual_bytes),
        "bitstream_is_actual": bitstream_paths is not None or final_bytes is not None,
        "CRp": float(original_count / max(final_count, 1)),
        "CRs": float(original_bytes / max(float(actual_bytes), 1.0)),
        "original_mb": original_bytes / (1024 * 1024),
        "final_mb": actual_bytes / (1024 * 1024),
    }
    return report
