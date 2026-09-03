"""Assemble a PAHC + shared-HAC++ bitstream bundle from a CompactScene.

This is where the "shared" contract becomes concrete: the bundle holds one
``hacpp/`` stream directory and one shared decoder; every component instance
contributes only ``template_id + quat + t + scale (+ SH residual)``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch

from ..codec import CompactScene
from .bundle import finalize_bundle
from .manifest import HacppManifest, InstanceSection
from .owners import build_owner_index, owner_blocks_are_contiguous, save_owners


def _pack_shapes(shapes: list[list[int]]):
    """Rectangle-ify ragged shape lists (numpy cannot store ragged rows)."""

    if not shapes:
        return np.zeros((0, 1), dtype=np.int64)
    ndim = max(max((len(shape) for shape in shapes), default=1), 1)
    packed = np.ones((len(shapes), ndim), dtype=np.int64)
    for row, shape in enumerate(shapes):
        if shape:
            packed[row, : len(shape)] = np.asarray(shape, dtype=np.int64)
    return packed


def save_instances(path, scene: CompactScene):
    """Serialise instance poses, row maps and residuals to a single npz stream."""

    n = len(scene.instances)
    quat = torch.zeros((n, 4))
    trans = torch.zeros((n, 3))
    scale = torch.ones((n, 1))
    center = torch.zeros((n, 3))
    template_ids = np.zeros((n,), dtype=np.int32)
    instance_ids = np.zeros((n,), dtype=np.int32)
    residual_keys: list[str] = []
    residual_lengths: list[int] = []
    residual_shapes: list[list[int]] = []
    residual_ndims: list[int] = []
    residual_values: list[torch.Tensor] = []
    # Per-instance windows into the flattened residual entry list.
    residual_entry_starts = np.zeros((n,), dtype=np.int64)
    residual_entry_counts = np.zeros((n,), dtype=np.int64)
    row_map_offsets = np.zeros((n + 1,), dtype=np.int64)
    row_map_values: list[torch.Tensor] = []
    for row, instance in enumerate(scene.instances):
        rotation = instance.rotation.detach().cpu().reshape(-1)
        if rotation.numel() == 9:
            from ..math_utils import rotmat_to_quat

            rotation = rotmat_to_quat(rotation)
        quat[row] = rotation[:4]
        trans[row] = instance.translation.detach().cpu().reshape(3)
        if instance.scale is not None:
            scale[row] = torch.abs(instance.scale.detach().cpu()).reshape(1)
        if instance.center is not None:
            center[row] = instance.center.detach().cpu().reshape(3)
        template_ids[row] = int(instance.template_id)
        instance_ids[row] = int(instance.instance_id)
        if instance.row_map is not None:
            row_map_values.append(instance.row_map.detach().cpu().reshape(-1).to(torch.int32))
        row_map_offsets[row + 1] = row_map_offsets[row] + (
            0 if instance.row_map is None else int(instance.row_map.numel())
        )
        residual_entry_starts[row] = len(residual_keys)
        for key, value in sorted(instance.residuals.items()):
            value = value.detach().cpu()
            residual_keys.append(key)
            residual_lengths.append(int(value.numel()))
            residual_shapes.append([int(size) for size in value.shape])
            residual_ndims.append(int(value.dim()))
            residual_values.append(value.reshape(-1))
        residual_entry_counts[row] = len(residual_keys) - int(residual_entry_starts[row])
    payload = {
        "template_ids": template_ids,
        "instance_ids": instance_ids,
        "quat": quat.numpy(),
        "translation": trans.numpy(),
        "scale": scale.numpy(),
        "center": center.numpy(),
        "sh_mode": np.asarray(scene.sh_mode),
        "residual_attributes": np.asarray(scene.residual_attributes),
        "row_map_offsets": row_map_offsets,
        "residual_keys": np.asarray(residual_keys),
        "residual_lengths": np.asarray(residual_lengths, dtype=np.int64),
        "residual_shapes": _pack_shapes(residual_shapes),
        "residual_ndims": np.asarray(residual_ndims, dtype=np.int64),
        "residual_entry_starts": residual_entry_starts,
        "residual_entry_counts": residual_entry_counts,
    }
    if residual_values:
        payload["residual_values"] = torch.cat(residual_values).numpy()
    if row_map_values:
        payload["row_map_values"] = torch.cat(row_map_values).numpy()
    np.savez_compressed(path, **payload)
    return payload


def load_instances(path) -> list[dict[str, Any]]:
    data = np.load(path, allow_pickle=False)
    n = data["template_ids"].shape[0]
    residual_keys = [str(item) for item in data["residual_keys"].tolist()]
    residual_lengths = data["residual_lengths"].tolist()
    residual_shapes = data["residual_shapes"].tolist() if "residual_shapes" in data else []
    residual_ndims = data["residual_ndims"].tolist() if "residual_ndims" in data else []
    entry_starts = data["residual_entry_starts"].tolist() if "residual_entry_starts" in data else [0] * n
    entry_counts = data["residual_entry_counts"].tolist() if "residual_entry_counts" in data else [0] * n
    values = data["residual_values"] if "residual_values" in data else np.zeros((0,))
    row_map_values = (
        torch.from_numpy(data["row_map_values"].astype(np.int64))
        if "row_map_values" in data
        else torch.zeros((0,), dtype=torch.long)
    )
    row_map_offsets = data["row_map_offsets"].tolist() if "row_map_offsets" in data else [0] * (n + 1)
    instances = []
    flat_cursor = 0
    for row in range(n):
        residuals: Dict[str, torch.Tensor] = {}
        for entry in range(int(entry_starts[row]), int(entry_starts[row]) + int(entry_counts[row])):
            key = residual_keys[entry]
            length = int(residual_lengths[entry])
            ndim = int(residual_ndims[entry]) if entry < len(residual_ndims) else 1
            shape = [int(size) for size in residual_shapes[entry][:ndim]] or [length]
            chunk = values[flat_cursor : flat_cursor + length]
            flat_cursor += length
            residuals[key] = torch.from_numpy(chunk.astype(np.float32)).reshape(shape)
        lo, hi = int(row_map_offsets[row]), int(row_map_offsets[row + 1])
        row_map = row_map_values[lo:hi].clone() if hi > lo else None
        instances.append(
            {
                "template_id": int(data["template_ids"][row]),
                "instance_id": int(data["instance_ids"][row]),
                "quat": torch.from_numpy(data["quat"][row].astype(np.float32)),
                "translation": torch.from_numpy(data["translation"][row].astype(np.float32)),
                "scale": torch.from_numpy(data["scale"][row].astype(np.float32)),
                "center": torch.from_numpy(data["center"][row].astype(np.float32)),
                "residuals": residuals,
                "row_map": row_map,
            }
        )
    return instances


def build_bundle(
    out_root: str | Path,
    scene: CompactScene,
    hacpp_stream_dir: Optional[str | Path] = None,
    shared_decoder_weights: Optional[str | Path] = None,
    template_rows: Optional[Dict[int, torch.Tensor]] = None,
    backend: Optional[Dict[str, Any]] = None,
    stream_globs: Iterable[str] = ("*.b", "*.npz"),
    scene_id: str = "",
    notes: str = "",
    owner_phase: str = "init_only",
    owner_provenance: str = "",
):
    """Write ``<out_root>`` as a manifest-described bundle.

    Parameters
    ----------
    hacpp_stream_dir:
        Directory produced by ``HacppBridge.encode`` (``xyz_gpcc.npz``,
        ``*.b`` streams, ``hash.b``, ``masks.b``). Files are copied into
        ``<out_root>/hacpp/``.
    shared_decoder_weights:
        The single ``mlp.pt`` written next to those streams. It is copied once
        and referenced once by ``shared_decoder``; hash/mask streams stay in
        ``streams``.
    template_rows:
        ``{template_id: rows into the basis}``; required when the scene has
        indexed templates so the owner sidecar can be built.
    """

    from .manifest import MANIFEST_NAME, SharedDecoder  # noqa: F401 (MANIFEST_NAME used below)

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "hacpp").mkdir(exist_ok=True)

    streams: Dict[str, list[str]] = {}
    if hacpp_stream_dir is not None:
        hacpp_dir = Path(hacpp_stream_dir)
        for pattern in stream_globs:
            for src in sorted(hacpp_dir.glob(pattern)):
                if src.name == "mlp.pt":
                    continue
                dst = out_root / "hacpp" / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)
                key = {
                    "xyz_gpcc.npz": "anchor",
                    "hash.b": "hash",
                    "masks.b": "masks",
                }.get(src.name, "feature_offsets")
                streams.setdefault(key, [])
                rel = dst.relative_to(out_root).as_posix()
                if rel not in streams[key]:
                    streams[key].append(rel)

    shared: Dict[str, list[str]] = []
    if shared_decoder_weights is not None:
        from .manifest import verify_shared_decoder_file

        offending = verify_shared_decoder_file(shared_decoder_weights)
        if offending:
            raise ValueError(
                "shared decoder file %s carries hash/encoding state %s; the hash "
                "grid must stay in hash.b and be billed once"
                % (shared_decoder_weights, offending)
            )
        dst = out_root / "hacpp" / Path(shared_decoder_weights).name
        shutil.copy2(shared_decoder_weights, dst)
        shared.append(dst.relative_to(out_root).as_posix())

    save_instances(out_root / "instances.npz", scene)
    if scene.static_gaussians:
        torch.save(
            {key: value.cpu() for key, value in scene.static_gaussians.items()},
            out_root / "basis.pt",
        )
        streams.setdefault("basis", ["basis.pt"])

    num_owners = 0
    if template_rows is None and scene.templates:
        first = next(iter(scene.templates.values()))
        if isinstance(first, dict) and "indices" in first:
            template_rows = {
                int(tid): value["indices"].long() for tid, value in scene.templates.items()
            }
    if template_rows:
        num_basis = int(scene.static_gaussians["xyz"].shape[0])
        owner_id, row_in_owner, counts = build_owner_index(template_rows, num_basis)
        if not owner_blocks_are_contiguous(owner_id):
            raise ValueError(
                "template anchors are not contiguous in basis order; HAC++ "
                "Morton-sorts anchors, so reorder the basis per owner before export"
            )
        save_owners(out_root / "owners.npz", owner_id, row_in_owner, sorted(template_rows))
        num_owners = len(template_rows)

    if not shared:
        raise ValueError(
            "build_bundle requires the single shared HAC++ decoder weights file "
            "(the 'shared_mlp.pt' written by the driver's encode command). A "
            "bundle without it would silently claim a shared decoder that does "
            "not exist; write an explicit init-only manifest instead."
        )

    manifest = HacppManifest(
        scene_id=scene_id,
        shared_decoder=SharedDecoder(
            weights=shared[0],
            auxiliary=shared[1:],
            feat_dim=int(backend.get("feat_dim", 50)) if backend else 50,
            n_offsets=int(backend.get("n_offsets", 10)) if backend else 10,
            log2_hashmap_size=int(backend.get("log2_hashmap_size", 19)) if backend else 19,
        ),
        streams=streams,
        instances=InstanceSection(
            file="instances.npz",
            count=len(scene.instances),
            sh_mode=scene.sh_mode,
            scaling_domain=scene.domain_of("scaling"),
        ),
        backend=backend or {},
        notes=notes
        or (
            "shared HAC++ decoder: exactly one per scene; owner ids are "
            "initialisation labels only (Phase 0)"
        ),
    )
    manifest.owners.num_owners = num_owners
    manifest.owners.num_anchors = (
        int(scene.static_gaussians["xyz"].shape[0]) if scene.static_gaussians else 0
    )
    manifest.owners.phase = owner_phase
    manifest.owners.provenance = owner_provenance
    manifest.validate(root=None)  # structure check before hashing files
    finalize_bundle(out_root, manifest, exclude=(MANIFEST_NAME,))
    manifest.validate(root=out_root)  # on-disk check after hashing
    return manifest
