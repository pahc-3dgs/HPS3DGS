"""Writer/reader for the HPS3DGS + shared-HAC++ bitstream bundle.

Sizes recorded in the manifest are **real on-disk bytes** (``st_size``) plus a
SHA-256, never in-memory tensor estimates. ``storage`` additionally separates
the two things that must never be conflated:

* ``codec_stream_bytes/_mib`` - physical bytes occupied by the copied HAC++
  streams, including the serialized ``x_bound_*.pkl`` files;
* ``paper_total_bytes/_mib`` - the HAC++ paper/log convention: encoded streams
  excluding serialized bounds + 24 raw bound bytes + raw float32 MLP weights;
* ``artifact_bytes/_mib`` - the *whole* bundle directory, shared decoder and
  manifest included, i.e. what actually has to be stored/transferred.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from ..metrics import file_bytes, sha256_file
from .manifest import HacppManifest, ManifestError

MIB = 1024 * 1024


def _mib(num_bytes: int) -> float:
    return round(num_bytes / MIB, 6)


def _codec_stream_rels(manifest: HacppManifest) -> List[str]:
    """Every declared stream that belongs to the per-scene codec payload.

    The x_bound pkls are part of the HAC++-reported payload (the reference
    adds 32*3*2 bits for the bounds); the shared decoder is not.
    """

    shared_weights = manifest.shared_decoder.weights
    rels = []
    for rels_for_key in manifest.streams.values():
        for rel in rels_for_key:
            if rel == shared_weights or not rel.startswith("hacpp/"):
                continue
            rels.append(rel)
    return rels


def _write_manifest(root: Path, manifest: HacppManifest):
    (root / "manifest.json").write_text(manifest.to_json() + "\n", encoding="utf-8")


def finalize_bundle(root: str | Path, manifest: HacppManifest, exclude: Iterable[str] = ()):
    """Hash and size every file under ``root`` and write ``manifest.json``.

    Called after all streams are on disk. ``exclude`` names relative paths
    (typically ``manifest.json`` itself) that must not be accounted.

    ``manifest.storage`` is filled here from the *measured* files:
    Physical stream bytes, paper-accounted bytes, and complete artifact bytes
    are separate fields. The manifest is rewritten until the recorded artifact
    size reaches a fixed point (its own size feeds the total).
    """

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    from .manifest import MANIFEST_NAME

    exclude = set(exclude) | {MANIFEST_NAME}
    files = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel in exclude:
            continue
        files[rel] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest.files = files

    codec_rels = [rel for rel in _codec_stream_rels(manifest) if rel in files]
    shared_rels = [rel for rel in (manifest.shared_decoder.weights,) if rel in files]
    storage = dict(manifest.storage)
    storage["codec_stream_bytes"] = sum(files[rel]["bytes"] for rel in codec_rels)
    storage["shared_decoder_bytes"] = sum(files[rel]["bytes"] for rel in shared_rels)
    bound_rels = {
        "hacpp/x_bound_min.pkl",
        "hacpp/x_bound_max.pkl",
    }
    encoded_rels = [rel for rel in codec_rels if rel not in bound_rels]
    mlp_parameter_bytes = int(
        manifest.decoder_config.get("shared_decoder_parameter_bytes", 0)
    )
    storage["paper_bound_bytes"] = 24
    storage["shared_decoder_parameter_bytes"] = mlp_parameter_bytes
    storage["paper_total_bytes"] = (
        sum(files[rel]["bytes"] for rel in encoded_rels)
        + storage["paper_bound_bytes"]
        + mlp_parameter_bytes
    )
    storage["definitions"] = {
        "codec_stream": "physical encoder-output files, including torch-serialized bounds; shared decoder excluded",
        "paper_total": "anchor/feat/scaling/offsets/hash/masks file bytes + 24 raw bound bytes + raw MLP parameter bytes",
        "shared_decoder": "physical torch serialization of the single hacpp/shared_mlp.pt",
        "artifact": "entire bundle directory, shared decoder and manifest included",
    }
    manifest.storage = storage

    manifest.storage["codec_stream_files"] = sorted(set(codec_rels))

    # artifact_bytes includes manifest.json, whose size depends on the recorded
    # number itself. Keep artifact_mib fixed-width so its textual representation
    # cannot make the manifest oscillate by one byte at a rounding boundary.
    manifest.storage["artifact_mib"] = "0000000000.000000"
    for _ in range(10):
        _write_manifest(root, manifest)
        artifact_bytes = sum(info["bytes"] for info in files.values()) + (root / MANIFEST_NAME).stat().st_size
        previous = manifest.storage.get("artifact_bytes")
        manifest.storage["artifact_bytes"] = artifact_bytes
        manifest.storage["codec_stream_mib"] = _mib(manifest.storage["codec_stream_bytes"])
        manifest.storage["paper_total_mib"] = _mib(manifest.storage["paper_total_bytes"])
        manifest.storage["shared_decoder_mib"] = _mib(manifest.storage["shared_decoder_bytes"])
        if previous == artifact_bytes:
            break
    else:
        raise ManifestError("manifest artifact size did not reach a fixed point")
    manifest.storage["artifact_mib"] = "%017.6f" % (artifact_bytes / MIB)
    _write_manifest(root, manifest)
    return manifest


def read_bundle(root: str | Path, verify: bool = False):
    """Load and validate a bundle manifest (optionally verifying checksums)."""

    from .manifest import MANIFEST_NAME

    root = Path(root)
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ManifestError("not a bundle: %s is missing" % manifest_path)
    manifest = HacppManifest.from_json(manifest_path.read_text(encoding="utf-8"))
    manifest.validate(root=root, verify_checksums=verify)
    return manifest


def bundle_bytes(root: str | Path):
    """Actual total on-disk size of a bundle, manifest included."""

    return file_bytes(Path(root).rglob("*"))


def bundle_artifact_bytes(root: str | Path):
    """Actual on-disk size of a bundle, measured independently of the manifest."""

    root = Path(root)
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def bundle_section_bytes(root: str | Path, manifest: HacppManifest):
    """Per-section actual byte counts for reporting."""

    root = Path(root)
    sections = {}
    shared = [manifest.shared_decoder.weights, *manifest.shared_decoder.auxiliary]
    sections["shared_decoder"] = sum((root / rel).stat().st_size for rel in shared)
    stream_rels = [rel for rels in manifest.streams.values() for rel in rels]
    if not manifest.codec_only:
        stream_rels += [manifest.owners.file, manifest.instances.file]
    for rel in stream_rels:
        top = rel.split("/")[0]
        sections.setdefault(top, 0)
        sections[top] += (root / rel).stat().st_size
    sections["manifest"] = (root / "manifest.json").stat().st_size
    return sections
