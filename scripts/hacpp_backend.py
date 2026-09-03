#!/usr/bin/env python3
"""Phase-0 HAC++ backend entry points.

Subcommands
-----------
export  Export a PAHC CompactScene (or a raw basis .pt) as an HAC++ init cloud
        (``hacpp_init.ply``/``.npz`` + ``owners.npz`` + ``init_config.json``).
inspect Dependency report for the external HAC++ checkout (never modified).

The exporter is a data adapter: HAC++ anchors/offsets/features/MLPs/hash are a
different parameterisation, so a SAGA/3DGS checkpoint cannot be restored into a
HAC++ ``GaussianModel``. The exported cloud initialises a HAC++ *student* that
must be trained with HAC++ ``train.py --init_ply`` before encode/decode.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from src.hacpp.bridge import HacppBridge  # noqa: E402
from src.hacpp.export import export_hacpp_init, load_init  # noqa: E402


def _load_basis(path: Path):
    """Load a PAHC CompactScene (.pahc.pt) or a raw tensor dict (.pt)."""

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "static_gaussians" in payload:
        from src.codec import CompactScene

        scene = CompactScene.from_payload(payload)
        xyz = scene.static_gaussians["xyz"]
        features_dc = None
        codebooks = scene.codebooks.get("features_dc")
        if codebooks is not None:
            values = codebooks["centers"][codebooks["indices"].long()]
            features_dc = values.reshape(values.shape[0], *codebooks.get("value_shape", values.shape[1:]))
        else:
            features_dc = scene.static_gaussians.get("features_dc")
        template_rows = {}
        for template_id, value in scene.templates.items():
            if isinstance(value, dict) and "indices" in value:
                template_rows[int(template_id)] = value["indices"].long()
        return xyz, features_dc, template_rows, "compact_scene"
    if isinstance(payload, dict) and "xyz" in payload:
        return payload["xyz"], payload.get("features_dc"), {}, "raw_tensors"
    raise ValueError("%s is neither a CompactScene payload nor a basis tensor dict" % path)


def cmd_export(args):
    xyz, features_dc, template_rows, kind = _load_basis(args.basis)
    config = export_hacpp_init(
        out_dir=args.out_dir,
        xyz=xyz,
        features_dc=features_dc,
        template_rows=template_rows or None,
        voxel_size=args.voxel_size,
        n_offsets=args.n_offsets,
        feat_dim=args.feat_dim,
        log2_hashmap_size=args.log2_hashmap_size,
        source=str(args.basis),
    )
    files = {
        path.name: path.stat().st_size
        for path in sorted(Path(args.out_dir).iterdir())
        if path.is_file()
    }
    print(json.dumps({"kind": kind, "config": config, "files": files}, indent=2))
    return 0


def cmd_verify(args):
    data = load_init(args.out_dir)
    print(
        json.dumps(
            {
                "points": int(data["points"].shape[0]),
                "owners": len(data["template_ids"]),
                "ordering": data["ordering"],
                "config": data["config"],
            },
            indent=2,
        )
    )
    return 0


def cmd_inspect(args):
    bridge = HacppBridge(hacpp_root=args.hacpp_root, python_exe=args.python)
    print(json.dumps(bridge.inspect(), indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hacpp-root", default=None, help="external HAC++ checkout (read-only)")
    parser.add_argument("--python", default=None, help="python executable for the HAC++ subprocess")
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export", help="export a PAHC basis as an HAC++ init cloud")
    export.add_argument("--basis", type=Path, required=True)
    export.add_argument("--out-dir", type=Path, required=True)
    export.add_argument("--voxel-size", type=float, default=None)
    export.add_argument("--n-offsets", type=int, default=10)
    export.add_argument("--feat-dim", type=int, default=50)
    export.add_argument("--log2-hashmap-size", type=int, default=19)

    verify = sub.add_parser("verify", help="re-read an exported init dir")
    verify.add_argument("--out-dir", type=Path, required=True)

    inspect = sub.add_parser("inspect", help="HAC++ dependency report")
    inspect.set_defaults(hacpp_root=None)

    args = parser.parse_args()
    if args.command == "export":
        return cmd_export(args)
    if args.command == "verify":
        return cmd_verify(args)
    if args.command == "inspect":
        from src.hacpp.bridge import DEFAULT_HACPP_ROOT

        args.hacpp_root = args.hacpp_root or DEFAULT_HACPP_ROOT
        return cmd_inspect(args)
    parser.error("unknown command")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
