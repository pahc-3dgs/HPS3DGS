"""Export a PAHC basis as an HAC++ *initialisation* cloud + owner sidecar.

This is a data-format adapter, not a weight converter: HAC++ anchors, offsets,
anchor features, its MLPs and its hash grid are a different parameterisation of
a 3DGS scene and cannot be restored from a SAGA/3DGS checkpoint
(:mod:`pahc.hacpp.bridge` refuses that explicitly). What we can do honestly is

1. write the canonical basis points as the HAC++ init point cloud so a
   ``GaussianModel.create_from_pcd`` student can be trained from them, and
2. ship the owner sidecar so the trained shared anchors can be mapped back to
   component templates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from .owners import build_owner_index, save_owners


#: Band-0 SH normalisation constant used by every 3DGS codebase
#: (``SH2RGB(sh) = C0 * sh + 0.5``).
SH_C0 = 0.28209479177387814


def basis_to_init_points(
    xyz: torch.Tensor,
    features_dc: Optional[torch.Tensor] = None,
    normals: Optional[torch.Tensor] = None,
):
    """Convert basis tensors into HAC++ ``BasicPointCloud``-shaped arrays."""

    points = xyz.detach().cpu().to(torch.float32).numpy()
    if features_dc is not None:
        # SH DC term -> RGB through the standard band-0 coefficient:
        # rgb = C0 * sh_dc + 0.5, matching SH2RGB in 3DGS/HAC++/SAGA.
        dc = features_dc.detach().cpu().to(torch.float32).reshape(points.shape[0], 3, -1)[..., 0]
        colors = ((SH_C0 * dc + 0.5).clamp(0.0, 1.0) * 255.0).to(torch.uint8).numpy()
    else:
        colors = np.full_like(points[:, :3], 128, dtype=np.uint8)
    if normals is None:
        normals = np.zeros_like(points)
    else:
        normals = normals.detach().cpu().to(torch.float32).numpy()
    return points, colors, normals


def save_init_ply(path, points, colors, normals):
    """Write the init cloud as a PLY HAC++ ``Scene(ply_path=...)`` can read."""

    from plyfile import PlyData, PlyElement

    vertices = np.empty(
        points.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertices["x"], vertices["y"], vertices["z"] = points[:, 0], points[:, 1], points[:, 2]
    vertices["nx"], vertices["ny"], vertices["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
    vertices["red"], vertices["green"], vertices["blue"] = (
        colors[:, 0],
        colors[:, 1],
        colors[:, 2],
    )
    element = PlyElement.describe(vertices, "vertex")
    PlyData([element], text=False).write(str(path))


def export_hacpp_init(
    out_dir: str | Path,
    xyz: torch.Tensor,
    features_dc: Optional[torch.Tensor] = None,
    normals: Optional[torch.Tensor] = None,
    template_rows: Optional[Dict[int, torch.Tensor]] = None,
    voxel_size: Optional[float] = None,
    n_offsets: int = 10,
    feat_dim: int = 50,
    log2_hashmap_size: int = 19,
    source: str = "",
):
    """Write ``hacpp_init.npz``, ``owners.npz`` and ``init_config.json``.

    ``template_rows`` maps template id -> rows of ``xyz``. Anchors that belong
    to no template keep owner id ``-1`` and are encoded by the shared HAC++
    model exactly like everything else.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    points, colors, normals_out = basis_to_init_points(xyz, features_dc, normals)

    counts: Dict[int, int] = {}
    if template_rows:
        owner_id, row_in_owner, counts = build_owner_index(template_rows, points.shape[0])
        save_owners(out_dir / "owners.npz", owner_id, row_in_owner, sorted(template_rows))
    else:
        owner_id = torch.full((points.shape[0],), -1, dtype=torch.long)
        row_in_owner = torch.full((points.shape[0],), -1, dtype=torch.long)
        save_owners(out_dir / "owners.npz", owner_id, row_in_owner, [])

    np.savez_compressed(
        out_dir / "hacpp_init.npz",
        points=points,
        colors=colors,
        normals=normals_out,
    )
    save_init_ply(out_dir / "hacpp_init.ply", points, colors, normals_out)
    config = {
        "num_points": int(points.shape[0]),
        "num_owners": int(len(template_rows or {})),
        "owner_counts": {str(key): int(value) for key, value in counts.items()},
        "voxel_size_hint": voxel_size,
        "n_offsets": int(n_offsets),
        "feat_dim": int(feat_dim),
        "log2_hashmap_size": int(log2_hashmap_size),
        "source": source,
        "note": (
            "initialisation only: the HAC++ student must be trained from this "
            "cloud; SAGA/3DGS Gaussian attributes cannot be restored into a "
            "HAC++ GaussianModel"
        ),
    }
    (out_dir / "init_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config


def load_init(out_dir: str | Path):
    """Read back what :func:`export_hacpp_init` wrote (validation helper)."""

    out_dir = Path(out_dir)
    data = np.load(out_dir / "hacpp_init.npz", allow_pickle=False)
    config = json.loads((out_dir / "init_config.json").read_text(encoding="utf-8"))
    owner_id, row_in_owner, template_ids, ordering = None, None, [], "morton"
    if (out_dir / "owners.npz").is_file():
        from .owners import load_owners

        owner_id, row_in_owner, template_ids, ordering = load_owners(out_dir / "owners.npz")
    return {
        "points": data["points"],
        "colors": data["colors"],
        "normals": data["normals"],
        "config": config,
        "owner_id": owner_id,
        "row_in_owner": row_in_owner,
        "template_ids": template_ids,
        "ordering": ordering,
    }
