"""Bitstream-bundle manifest for the PAHC + shared-HAC++ layout.

Design invariant enforced here: a scene carries **exactly one** shared HAC++
decoder (MLPs + entropy models). The hash grid is *not* part of the shared
decoder section because HAC++ emits it separately as ``hash.b``; listing it in
both places would double-bill the same bytes.

Layout::

    <bundle>/
      manifest.json
      hacpp/                     # raw HAC++ encoder output
        xyz_gpcc.npz             # anchors (GPCC, Morton order)
        feat_0_*.b ...           # anchor feature bitstreams
        scaling_*.b, offsets_*.b
        hash.b                   # binary hash-grid embeddings
        masks.b
        shared_mlp.pt            # shared decoder weights; excludes hash grid
      owners.npz                 # anchor -> (owner/template id, row index)
      instances.npz              # template_id + quat + t + scale (+ residuals)
      render_config.json
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

MANIFEST_VERSION = "0.2"
MANIFEST_NAME = "manifest.json"

#: Relative paths HAC++ writes for a single scene.
HACPP_STREAM_FILES = {
    "anchor": "hacpp/xyz_gpcc.npz",
    "hash": "hacpp/hash.b",
    "masks": "hacpp/masks.b",
}


class ManifestError(ValueError):
    """Raised when a bundle violates the shared-decoder contract."""


@dataclass
class SharedDecoder:
    """The one-and-only shared HAC++ decoder of a scene."""

    kind: str = "hacpp_shared_v1"
    weights: str = "hacpp/shared_mlp.pt"
    #: Files carrying decoder state. ``hash.b`` is deliberately excluded: it is
    #: an HAC++ *scene* stream, not shared decoder state, and it is already
    #: counted under ``streams``.
    auxiliary: List[str] = field(default_factory=list)
    excludes_hash: bool = True
    feat_dim: int = 50
    n_offsets: int = 10
    log2_hashmap_size: int = 19
    note: str = "one shared decoder per scene; templates must not instantiate MLPs"

    def validate(self):
        if self.kind != "hacpp_shared_v1":
            raise ManifestError("Unknown shared decoder kind %r" % self.kind)
        if not self.excludes_hash:
            raise ManifestError("shared decoder must exclude the hash grid (hash.b)")
        if any("hash.b" in str(item) for item in [self.weights, *self.auxiliary]):
            raise ManifestError("hash.b must not be listed in the shared decoder")
        if any("encoding_xyz" in str(item) for item in [self.weights, *self.auxiliary]):
            raise ManifestError("shared decoder must not embed the hash grid (encoding_xyz)")
        return self


@dataclass
class OwnerSection:
    file: str = "owners.npz"
    num_anchors: int = 0
    num_owners: int = 0
    ordering: str = "morton"
    layout: str = "owner_id:int32, row_in_owner:int32"
    #: ``init_only``: owner ids come from PAHC clustering and are *not*
    #: propagated through HAC++ anchor densification/pruning, so they are valid
    #: only for the initialisation cloud. Claiming ``stable`` requires
    #: ``provenance`` describing how ownership was tracked during training -
    #: without it the manifest refuses the stronger claim.
    phase: str = "init_only"
    provenance: str = ""

    def validate(self):
        if self.phase not in ("init_only", "stable"):
            raise ManifestError("owner phase must be 'init_only' or 'stable'")
        if self.phase == "stable" and not self.provenance:
            raise ManifestError(
                "owner phase 'stable' requires provenance (how ownership was "
                "tracked through HAC++ densification/pruning)"
            )
        return self


def _walk_state_keys(node, prefix=""):
    """Yield every ``module.param`` path inside a (possibly nested) state dict."""

    if isinstance(node, dict):
        for key, value in node.items():
            name = "%s.%s" % (prefix, key) if prefix else str(key)
            yield name
            yield from _walk_state_keys(value, name)
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            name = "%s.%d" % (prefix, index) if prefix else str(index)
            yield from _walk_state_keys(value, name)


def verify_shared_decoder_file(path):
    """Structural check: the shared decoder state must not embed the hash grid.

    ``encoding_xyz`` (the hash grid) nested anywhere - top level or inside a
    sub-module state dict - is rejected. Returns the offending key paths
    (empty list means the file is compliant).
    """

    import torch

    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.0 has no weights_only
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise ManifestError("shared decoder file %s is not a state dict" % path)
    return [
        key for key in _walk_state_keys(state) if "encoding" in key or "hash" in key
    ]


@dataclass
class InstanceSection:
    file: str = "instances.npz"
    count: int = 0
    layout: str = "template_id:int32, quat:float32[N,4], t:float32[N,3], scale:float32[N,1], center:float32[N,3]"
    sh_mode: str = "residual"
    scaling_domain: str = "log"


@dataclass
class HacppManifest:
    version: str = MANIFEST_VERSION
    scene_id: str = ""
    shared_decoder: SharedDecoder = field(default_factory=SharedDecoder)
    streams: Dict[str, List[str]] = field(default_factory=dict)
    owners: OwnerSection = field(default_factory=OwnerSection)
    instances: InstanceSection = field(default_factory=InstanceSection)
    files: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    backend: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self):
        return asdict(self)

    def to_json(self, indent: int = 2):
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        shared = SharedDecoder(**data.get("shared_decoder", {}))
        owners = OwnerSection(**data.get("owners", {}))
        instances = InstanceSection(**data.get("instances", {}))
        return cls(
            version=data.get("version", MANIFEST_VERSION),
            scene_id=data.get("scene_id", ""),
            shared_decoder=shared,
            streams=data.get("streams", {}),
            owners=owners,
            instances=instances,
            files=data.get("files", {}),
            backend=data.get("backend", {}),
            notes=data.get("notes", ""),
        )

    @classmethod
    def from_json(cls, text: str):
        return cls.from_dict(json.loads(text))

    def validate(self, root: str | Path | None = None, verify_checksums: bool = False):
        """Enforce the single-shared-decoder contract and file presence.

        With ``root=None`` only the structural contract is checked (used before
        files are hashed); with a ``root`` every declared file must exist with
        the recorded size.
        """

        self.shared_decoder.validate()
        self.owners.validate()
        if self.owners.num_owners <= 0:
            raise ManifestError("bundle must declare at least one owner/template")
        if self.instances.count > 0 and self.owners.num_owners <= 0:
            raise ManifestError("instances require owners")

        expected: Dict[str, str] = {MANIFEST_NAME: ""}
        for key, rel in HACPP_STREAM_FILES.items():
            expected[rel] = key
        expected[self.shared_decoder.weights] = "shared_decoder"
        expected[self.owners.file] = "owners"
        expected[self.instances.file] = "instances"
        for rels in self.streams.values():
            for rel in rels:
                expected[rel] = "stream"

        if root is None:
            return self

        root = Path(root)
        for rel in expected:
            if rel not in self.files and rel != MANIFEST_NAME:
                raise ManifestError("manifest does not account for %s" % rel)
        for rel, info in self.files.items():
            if root is None:
                continue
            path = root / rel
            if not path.is_file():
                raise ManifestError("missing bundle file %s" % rel)
            if int(info.get("bytes", -1)) != path.stat().st_size:
                raise ManifestError(
                    "size mismatch for %s: manifest=%s disk=%s"
                    % (rel, info.get("bytes"), path.stat().st_size)
                )
            if verify_checksums and "sha256" in info:
                from ..metrics import sha256_file

                if sha256_file(path) != info["sha256"]:
                    raise ManifestError("checksum mismatch for %s" % rel)
        return self
