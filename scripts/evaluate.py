#!/usr/bin/env python3
"""Compute HPS3DGS compression statistics."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.codec import load_compact_scene
from src.metrics import compression_report, file_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-scene", type=Path, required=True)
    parser.add_argument("--original-count", type=int, required=True)
    parser.add_argument("--final-count", type=int, default=None)
    parser.add_argument("--bitstream", type=Path, action="append", default=None,
                        help="extra bitstream files/dirs (e.g. a HAC++ bundle); repeatable")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    scene = load_compact_scene(args.compact_scene)
    final_count = args.final_count
    if final_count is None:
        xyz = scene.static_gaussians.get("xyz")
        if xyz is None:
            xyz = scene.templates.get("xyz")
        final_count = int(xyz.shape[0]) if xyz is not None else 0

    bitstream_paths = [args.compact_scene] + list(args.bitstream or [])
    report = compression_report(
        args.original_count,
        final_count,
        final_payload=scene.to_payload(),
        bitstream_paths=bitstream_paths,
    )
    report["bitstream_files"] = {str(path): file_bytes(path) for path in bitstream_paths}
    report["sh_mode"] = scene.sh_mode
    report["lossless_attributes"] = scene.metadata.get("lossless_attributes")
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
