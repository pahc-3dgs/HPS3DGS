#!/usr/bin/env python3
"""Decode a PAHC-3DGS compact scene."""


import argparse
from pathlib import Path

import torch

from src.codec import decode_compact_scene, load_compact_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scene = load_compact_scene(args.compact_scene)
    decoded = decode_compact_scene(scene)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(decoded, args.output)
    print(f"Saved decoded tensors to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
