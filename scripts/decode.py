#!/usr/bin/env python3
"""Decode a HPS3DGS compact scene."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from src.codec import decode_compact_scene, estimate_payload_bytes, load_compact_scene
from src.metrics import compression_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-residuals", action="store_true",
                        help="fail instead of silently zeroing a missing residual")
    args = parser.parse_args()

    scene = load_compact_scene(args.compact_scene)
    decoded = decode_compact_scene(scene, require_residuals=args.require_residuals)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(decoded, args.output)
    report = compression_report(
        original_count=int(scene.metadata.get("original_count", decoded["xyz"].shape[0])),
        final_count=int(decoded["xyz"].shape[0]),
        bitstream_paths=[args.compact_scene],
    )
    report["estimated_tensor_bytes_decoded"] = estimate_payload_bytes(decoded)
    report["sh_mode"] = scene.sh_mode
    report["lossless_attributes"] = scene.metadata.get("lossless_attributes")
    print(json.dumps(report, indent=2))
    print(f"Saved decoded tensors to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
