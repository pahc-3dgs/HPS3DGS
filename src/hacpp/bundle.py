"""Writer/reader for the PAHC + shared-HAC++ bitstream bundle.

Sizes recorded in the manifest are **real on-disk bytes** (``st_size``) plus a
SHA-256, never in-memory tensor estimates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from ..metrics import file_bytes, sha256_file
from .manifest import HacppManifest, ManifestError


def finalize_bundle(root: str | Path, manifest: HacppManifest, exclude: Iterable[str] = ()):
    """Hash and size every file under ``root`` and write ``manifest.json``.

    Called after all streams are on disk. ``exclude`` names relative paths
    (typically ``manifest.json`` itself) that must not be accounted.
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
    (root / MANIFEST_NAME).write_text(manifest.to_json() + "\n", encoding="utf-8")
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


def bundle_section_bytes(root: str | Path, manifest: HacppManifest):
    """Per-section actual byte counts for reporting."""

    root = Path(root)
    sections = {}
    shared = [manifest.shared_decoder.weights, *manifest.shared_decoder.auxiliary]
    sections["shared_decoder"] = sum((root / rel).stat().st_size for rel in shared)
    stream_rels = [rel for rels in manifest.streams.values() for rel in rels]
    stream_rels += [manifest.owners.file, manifest.instances.file]
    for rel in stream_rels:
        top = rel.split("/")[0]
        sections.setdefault(top, 0)
        sections[top] += (root / rel).stat().st_size
    sections["manifest"] = (root / "manifest.json").stat().st_size
    return sections
