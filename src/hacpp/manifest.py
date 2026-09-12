"""Bitstream-bundle manifest for the HPS3DGS + shared-HAC++ layout.

Design invariant enforced here: a scene carries **exactly one** shared HAC++
decoder (MLPs + entropy models). The hash grid is *not* part of the shared
decoder section because HAC++ emits it separately as ``hash.b``; listing it in
both places would double-bill the same bytes.

Layout::

    <bundle>/
      manifest.json
      hacpp/                     # raw HAC++ encoder output
        xyz_gpcc.npz             # anchors (GPCC, Morton order)
        x_bound_min.pkl          # training-time hash-grid normalisation bounds
        x_bound_max.pkl          #   (consumed by conduct_decoding)
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

MANIFEST_VERSION = "0.3"
MANIFEST_NAME = "manifest.json"
DECODER_CONFIG_FILE = "hacpp/decoder_config.json"
SUPPORTED_MANIFEST_VERSIONS = ("0.3",)

#: Relative paths HAC++ writes for a single scene.
HACPP_STREAM_FILES = {
    "anchor": "hacpp/xyz_gpcc.npz",
    "hash": "hacpp/hash.b",
    "masks": "hacpp/masks.b",
    "x_bound_min": "hacpp/x_bound_min.pkl",
    "x_bound_max": "hacpp/x_bound_max.pkl",
}

#: A self-contained (``codec_only``) bundle must carry *all* of these, plus at
#: least one arithmetic-coded stream of each family (feat/scaling/offsets).
#: ``conduct_decoding`` reads exactly this set from ``pre_path_name``; a bundle
#: missing any of them cannot be decoded without going back to the original
#: model directory.
CODEC_REQUIRED_FILES = {
    **HACPP_STREAM_FILES,
    "shared_decoder": "hacpp/shared_mlp.pt",
}

#: Keys that must be present in ``decoder_config`` for a reader to instantiate
#: a ``GaussianModel`` without access to the original model directory
#: (architecture/hyper-parameters; cfg_args alone is not enough because the
#: hash-grid flags are not recorded there).
DECODER_CONFIG_REQUIRED_KEYS = (
    "config_version",
    "feat_dim",
    "n_offsets",
    "voxel_size",
    "update_depth",
    "update_init_factor",
    "update_hierachy_factor",
    "use_feat_bank",
    "n_features_per_level",
    "log2_hashmap_size",
    "log2_hashmap_size_2D",
    "resolutions_list",
    "resolutions_list_2D",
    "use_2D",
    "ste_binary",
    "ste_multistep",
    "add_noise",
    "decoded_version",
    "white_background",
    "is_synthetic_nerf",
    "eval",
    "all_views_train_test",
    "Q",
    "dtype",
    "shared_decoder_parameter_bytes",
)


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
    # Defaults mirror the driver's ENCODING_CONFIG (HAC++ train.py flags), not
    # the GaussianModel constructor defaults, which are different (19/17).
    log2_hashmap_size: int = 13
    log2_hashmap_size_2D: int = 15
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
    #: ``init_only``: owner ids come from HPS3DGS clustering and are *not*
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


def validate_shared_decoder_file(path, use_feat_bank: bool = False):
    """Validate module coverage and return raw tensor bytes for accounting."""

    import torch

    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise ManifestError("shared decoder file %s is not a state dict" % path)
    offending = [
        key for key in _walk_state_keys(state) if "encoding" in key or "hash" in key
    ]
    if offending:
        raise ManifestError(
            "shared decoder file %s embeds forbidden hash/encoding state: %s"
            % (path, offending)
        )
    required = {
        "opacity_mlp", "cov_mlp", "color_mlp", "grid_mlp", "deform_mlp"
    }
    if use_feat_bank:
        required.add("mlp_feature_bank")
    missing = sorted(required - set(state))
    extra = sorted(set(state) - required)
    if missing or extra:
        raise ManifestError(
            "shared decoder modules mismatch: missing=%s extra=%s"
            % (missing, extra)
        )

    tensor_bytes = 0
    for module_name, module_state in state.items():
        if not isinstance(module_state, dict):
            raise ManifestError("shared decoder module %s is not a state dict" % module_name)
        for value in module_state.values():
            if not torch.is_tensor(value):
                raise ManifestError("shared decoder module %s contains a non-tensor value" % module_name)
            tensor_bytes += value.numel() * value.element_size()
    return int(tensor_bytes)


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
    #: Architecture needed to rebuild the HAC++ ``GaussianModel`` at decode
    #: time without the original model directory (see
    #: ``DECODER_CONFIG_REQUIRED_KEYS``).
    decoder_config: Dict[str, Any] = field(default_factory=dict)
    #: Actual on-disk accounting (never estimates):
    #: ``codec_stream_bytes/_mib`` - physical HAC++ encoder-output files,
    #: ``paper_total_bytes/_mib`` - encoded streams + 24 raw bound bytes + raw
    #: float32 MLP parameters (the HAC++ paper/log convention),
    #: ``shared_decoder_bytes/_mib`` - physical ``shared_mlp.pt`` serialization,
    #: ``artifact_bytes/_mib`` - the whole bundle directory, manifest included.
    storage: Dict[str, Any] = field(default_factory=dict)
    #: True for a single-scene HAC++ codec bundle that carries no HPS3DGS
    #: owners/instances sidecars; the owner contract is then not applicable.
    codec_only: bool = False
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
            decoder_config=data.get("decoder_config", {}),
            storage=data.get("storage", {}),
            codec_only=bool(data.get("codec_only", False)),
            notes=data.get("notes", ""),
        )

    @classmethod
    def from_json(cls, text: str):
        return cls.from_dict(json.loads(text))

    def validate(self, root: str | Path | None = None, verify_checksums: bool = False):
        """Enforce the single-shared-decoder contract and file presence.

        With ``root=None`` only the structural contract is checked (used before
        files are hashed); with a ``root`` every declared file must exist with
        the recorded size (and checksum, when ``verify_checksums``).

        Version contract: only manifests written by this code (``0.3``) are
        accepted; older bundles lack ``decoder_config`` and the x_bound streams
        and cannot be decoded self-contained, so they fail loudly instead of
        half-working.
        """

        if self.version not in SUPPORTED_MANIFEST_VERSIONS:
            raise ManifestError(
                "unsupported manifest version %r (supported: %s); re-encode the "
                "bundle - older manifests are not self-contained"
                % (self.version, ", ".join(SUPPORTED_MANIFEST_VERSIONS))
            )
        self.shared_decoder.validate()
        self._validate_codec_contract()
        if not self.codec_only:
            self.owners.validate()
            if self.owners.num_owners <= 0:
                raise ManifestError("bundle must declare at least one owner/template")
            if self.instances.count > 0 and self.owners.num_owners <= 0:
                raise ManifestError("instances require owners")

        expected: Dict[str, str] = {MANIFEST_NAME: ""}
        expected[DECODER_CONFIG_FILE] = "decoder_config"
        for key, rel in HACPP_STREAM_FILES.items():
            expected[rel] = key
        expected[self.shared_decoder.weights] = "shared_decoder"
        if not self.codec_only:
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
        if "artifact_bytes" in self.storage:
            actual = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
            if int(self.storage["artifact_bytes"]) != actual:
                raise ManifestError(
                    "artifact size mismatch: manifest=%s disk=%s"
                    % (self.storage["artifact_bytes"], actual)
                )
        measured_parameter_bytes = validate_shared_decoder_file(
            root / self.shared_decoder.weights,
            use_feat_bank=bool(self.decoder_config["use_feat_bank"]),
        )
        if measured_parameter_bytes != int(
            self.decoder_config["shared_decoder_parameter_bytes"]
        ):
            raise ManifestError(
                "shared decoder parameter-byte mismatch: decoder_config=%s measured=%s"
                % (
                    self.decoder_config["shared_decoder_parameter_bytes"],
                    measured_parameter_bytes,
                )
            )
        return self

    def _validate_codec_contract(self):
        """Self-contained (``codec_only``) bundle: everything decode needs."""

        missing = [key for key in DECODER_CONFIG_REQUIRED_KEYS if key not in self.decoder_config]
        if missing:
            raise ManifestError(
                "decoder_config is missing keys required to instantiate the "
                "HAC++ GaussianModel: %s" % ", ".join(missing)
            )
        if int(self.decoder_config["config_version"]) != 1:
            raise ManifestError("unsupported decoder_config version %r" % self.decoder_config["config_version"])
        if self.decoder_config["dtype"] != "float32":
            raise ManifestError("only float32 shared decoder weights are currently supported")
        if int(self.decoder_config["shared_decoder_parameter_bytes"]) <= 0:
            raise ManifestError("shared_decoder_parameter_bytes must be measured and positive")
        declared = {rel for rels in self.streams.values() for rel in rels}
        declared.add(self.shared_decoder.weights)
        for key, rel in CODEC_REQUIRED_FILES.items():
            if rel not in declared:
                raise ManifestError(
                    "self-contained bundle must declare %s under %r (got streams=%s)"
                    % (rel, key, sorted(declared))
                )
        for family in ("feat", "scaling", "offsets"):
            if not any(
                rel.startswith("hacpp/%s_" % family) and rel.endswith(".b") for rel in declared
            ):
                raise ManifestError(
                    "self-contained bundle must include at least one '%s_*' "
                    "arithmetic-coded stream" % family
                )
        return self
