"""Evaluation metrics and compression statistics."""

from __future__ import annotations

from typing import Any

from .codec import estimate_payload_bytes


def compression_report(
    original_count: int,
    final_count: int,
    original_bytes: int | None = None,
    final_payload: Any | None = None,
    final_bytes: int | None = None,
):
    """Build a compact JSON-friendly compression report."""

    if original_bytes is None:
        original_bytes = original_count * (3 + 3 + 45 + 1 + 3 + 4) * 4
    if final_bytes is None:
        if final_payload is not None:
            final_bytes = estimate_payload_bytes(final_payload)
        else:
            final_bytes = final_count * (3 + 3 + 45 + 1 + 3 + 4) * 4
    return {
        "original_count": int(original_count),
        "final_count": int(final_count),
        "original_bytes": int(original_bytes),
        "final_bytes": int(final_bytes),
        "CRp": float(original_count / max(final_count, 1)),
        "CRs": float(original_bytes / max(float(final_bytes), 1.0)),
        "original_mb": original_bytes / (1024 * 1024),
        "final_mb": final_bytes / (1024 * 1024),
    }
