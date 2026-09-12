"""Assemble a HPS3DGS + shared-HAC++ bitstream bundle from a CompactScene.

This is where the "shared" contract becomes concrete: the bundle holds one
``hacpp/`` stream directory and one shared decoder; every component instance
contributes only ``template_id + quat + t + scale (+ SH residual)``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

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


#: Exact stream filenames HAC++'s ``conduct_encoding`` writes per scene.
#: These are copied *by name*, never via a broad glob: a broad ``*.b``/``*.npz``
#: glob is exactly how x_bound_min.pkl / decoder_config.json got silently
#: dropped from earlier bundles.
EXACT_STREAM_NAMES = {
    "xyz_gpcc.npz": "anchor",
    "hash.b": "hash",
    "masks.b": "masks",
    "x_bound_min.pkl": "x_bound_min",
    "x_bound_max.pkl": "x_bound_max",
}

#: Arithmetic-coded stream families, batch/channel-indexed by HAC++:
#: ``feat_<step>_<cc>.b``, ``scaling_<step>.b``, ``offsets_<step>.b``.
CODED_STREAM_PREFIXES = ("feat_", "scaling_", "offsets_")

DECODER_CONFIG_NAME = "decoder_config.json"

#: Shared-decoder weight files the driver may have written next to the streams.
SHARED_DECODER_NAMES = ("shared_mlp.pt", "mlp.pt")


def classify_stream_dir(hacpp_dir: str | Path):
    """Classify *every* file in an encoder output dir; refuse the unknown.

    Returns ``(streams, decoder_config, shared_decoder_path)`` where
    ``streams`` maps manifest stream keys to source paths. Raises
    :class:`ValueError` if any file is unrecognised (so nothing is silently
    left behind) or if a file required for a self-contained decode is missing.
    """

    hacpp_dir = Path(hacpp_dir)
    if not hacpp_dir.is_dir():
        raise ValueError("HAC++ stream dir not found: %s" % hacpp_dir)

    streams: Dict[str, list[Path]] = {}
    decoder_config: Dict[str, Any] = {}
    shared_decoder: Optional[Path] = None
    unrecognised: list[str] = []
    for src in sorted(hacpp_dir.iterdir()):
        if not src.is_file():
            raise ValueError("unexpected non-file entry in stream dir: %s" % src)
        name = src.name
        if name == DECODER_CONFIG_NAME:
            import json

            decoder_config = json.loads(src.read_text(encoding="utf-8"))
            continue
        if name in SHARED_DECODER_NAMES:
            if shared_decoder is not None:
                raise ValueError(
                    "stream dir carries two shared-decoder files: %s and %s"
                    % (shared_decoder.name, name)
                )
            shared_decoder = src
            continue
        if name in EXACT_STREAM_NAMES:
            streams.setdefault(EXACT_STREAM_NAMES[name], []).append(src)
            continue
        if name.endswith(".b") and name.startswith(CODED_STREAM_PREFIXES):
            family = name.split("_", 1)[0]
            streams.setdefault(family, []).append(src)
            continue
        unrecognised.append(name)

    if unrecognised:
        raise ValueError(
            "stream dir %s contains files that are not part of the HAC++ codec "
            "contract; refusing to pack them silently: %s. Add them to "
            "EXACT_STREAM_NAMES/CODED_STREAM_PREFIXES with an explicit role, or "
            "remove them from the encoder output." % (hacpp_dir, sorted(unrecognised))
        )
    missing = [
        name for name, key in EXACT_STREAM_NAMES.items() if not streams.get(key)
    ]
    for family in ("feat", "scaling", "offsets"):
        if not streams.get(family):
            missing.append("%s_<...>.b (at least one)" % family)
    if missing:
        raise ValueError(
            "stream dir %s is not a complete HAC++ encoder output; missing %s"
            % (hacpp_dir, ", ".join(sorted(missing)))
        )
    if not decoder_config:
        raise ValueError(
            "stream dir %s has no %s; without it the bundle cannot record the "
            "architecture needed to instantiate a HAC++ GaussianModel at "
            "decode time." % (hacpp_dir, DECODER_CONFIG_NAME)
        )
    return streams, decoder_config, shared_decoder


def build_bundle(
    out_root: str | Path,
    scene: Optional[CompactScene] = None,
    hacpp_stream_dir: Optional[str | Path] = None,
    shared_decoder_weights: Optional[str | Path] = None,
    template_rows: Optional[Dict[int, torch.Tensor]] = None,
    backend: Optional[Dict[str, Any]] = None,
    decoder_config: Optional[Dict[str, Any]] = None,
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
        ``x_bound_{min,max}.pkl``, ``hash.b``, ``masks.b``, the
        ``feat_/scaling_/offsets_*.b`` streams and ``decoder_config.json``).
        Every file is classified explicitly and copied; unknown files abort
        the packing instead of being skipped.
    shared_decoder_weights:
        The single ``shared_mlp.pt`` written next to those streams. When
        omitted, the copy inside ``hacpp_stream_dir`` is used. It is copied
        once and referenced once by ``shared_decoder``; hash/mask streams stay
        in ``streams``.
    scene:
        HPS3DGS ``CompactScene`` contributing ``instances.npz``/``owners.npz``.
        ``None`` packs a single-scene HAC++ codec bundle (``codec_only``).
    decoder_config:
        Optional explicit architecture config; overrides the values recorded
        in ``<hacpp_stream_dir>/decoder_config.json``.
    template_rows:
        ``{template_id: rows into the basis}``; required when the scene has
        indexed templates so the owner sidecar can be built.
    """

    from .manifest import MANIFEST_NAME, SharedDecoder

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "hacpp").mkdir(exist_ok=True)

    streams: Dict[str, list[str]] = {}
    packed_decoder_config: Dict[str, Any] = {}
    bundled_decoder: Optional[Path] = None
    if hacpp_stream_dir is None:
        raise ValueError("build_bundle requires a complete HAC++ encoder output directory")
    if hacpp_stream_dir is not None:
        classified, packed_decoder_config, bundled_decoder = classify_stream_dir(hacpp_stream_dir)
        for key, sources in classified.items():
            for src in sources:
                dst = out_root / "hacpp" / src.name
                shutil.copy2(src, dst)
                rel = dst.relative_to(out_root).as_posix()
                if rel not in streams.setdefault(key, []):
                    streams[key].append(rel)

    shared: list[str] = []
    if shared_decoder_weights is not None:
        from .manifest import verify_shared_decoder_file

        offending = verify_shared_decoder_file(shared_decoder_weights)
        if offending:
            raise ValueError(
                "shared decoder file %s carries hash/encoding state %s; the hash "
                "grid must stay in hash.b and be billed once"
                % (shared_decoder_weights, offending)
            )
        dst = out_root / "hacpp" / "shared_mlp.pt"
        shutil.copy2(shared_decoder_weights, dst)
        shared.append(dst.relative_to(out_root).as_posix())
    elif bundled_decoder is not None:
        from .manifest import verify_shared_decoder_file

        offending = verify_shared_decoder_file(bundled_decoder)
        if offending:
            raise ValueError(
                "shared decoder file %s carries hash/encoding state %s; the hash "
                "grid must stay in hash.b and be billed once"
                % (bundled_decoder, offending)
            )
        dst = out_root / "hacpp" / "shared_mlp.pt"
        shutil.copy2(bundled_decoder, dst)
        shared.append(dst.relative_to(out_root).as_posix())

    num_owners = 0
    instances = InstanceSection(file="instances.npz", count=0)
    if scene is not None:
        save_instances(out_root / "instances.npz", scene)
        if scene.static_gaussians:
            torch.save(
                {key: value.cpu() for key, value in scene.static_gaussians.items()},
                out_root / "basis.pt",
            )
            streams.setdefault("basis", []).append("basis.pt")
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
        instances = InstanceSection(
            file="instances.npz",
            count=len(scene.instances),
            sh_mode=scene.sh_mode,
            scaling_domain=scene.domain_of("scaling"),
        )

    if not shared:
        raise ValueError(
            "build_bundle requires the single shared HAC++ decoder weights file "
            "(the 'shared_mlp.pt' written by the driver's encode command). A "
            "bundle without it would silently claim a shared decoder that does "
            "not exist; write an explicit init-only manifest instead."
        )

    merged_decoder_config: Dict[str, Any] = dict(packed_decoder_config)
    if decoder_config:
        merged_decoder_config.update(decoder_config)
    from .manifest import validate_shared_decoder_file

    measured_parameter_bytes = validate_shared_decoder_file(
        out_root / "hacpp" / "shared_mlp.pt",
        use_feat_bank=bool(merged_decoder_config.get("use_feat_bank", False)),
    )
    declared_parameter_bytes = int(
        merged_decoder_config.get("shared_decoder_parameter_bytes", -1)
    )
    if declared_parameter_bytes != measured_parameter_bytes:
        raise ValueError(
            "shared decoder parameter-byte mismatch: decoder_config=%s measured=%s"
            % (declared_parameter_bytes, measured_parameter_bytes)
        )
    (out_root / "hacpp" / DECODER_CONFIG_NAME).write_text(
        json.dumps(merged_decoder_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = HacppManifest(
        scene_id=scene_id,
        shared_decoder=SharedDecoder(
            weights=shared[0],
            auxiliary=shared[1:],
            feat_dim=int(merged_decoder_config.get("feat_dim", 50)),
            n_offsets=int(merged_decoder_config.get("n_offsets", 10)),
            log2_hashmap_size=int(merged_decoder_config.get("log2_hashmap_size", 13)),
            log2_hashmap_size_2D=int(merged_decoder_config.get("log2_hashmap_size_2D", 15)),
        ),
        streams=streams,
        instances=instances,
        backend=backend or {},
        decoder_config=merged_decoder_config,
        codec_only=scene is None,
        notes=notes
        or (
            "self-contained HAC++ codec bundle: x_bound + shared_mlp.pt + all "
            "codec streams + decoder_config, decodable without the original "
            "model directory"
            if scene is None
            else
            "shared HAC++ decoder: exactly one per scene; owner ids are "
            "initialisation labels only (Phase 0)"
        ),
    )
    if scene is not None:
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
