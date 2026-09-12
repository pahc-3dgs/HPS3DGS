"""Compact HPS3DGS representation and storage helpers.

Format version 0.2 fixes two correctness problems of the 0.1 writer:

* instance scaling is applied in the *parameter domain* of each attribute.
  Standard 3DGS stores ``_scaling`` in log space (``get_scaling = exp``), so a
  uniform instance scale ``s`` must add ``log(s)`` instead of multiplying.
  HAC++ stores its anchor ``_scaling`` in the same log domain, so the same rule
  applies there; only tensors that already hold physical scales use the linear
  rule.
* per-instance appearance is never silently dropped. ``sh_mode`` selects the
  policy and the writer must record it in ``metadata``:

  - ``residual`` (default): the instance stores
    ``attr_target - attr_posed_template`` for every attribute listed in
    ``residual_attributes``. Decoding adds the residual back, so the
    reconstruction reproduces the *target* instance appearance exactly for
    those attributes - provided a row correspondence between the target
    cluster and the template rows exists. That correspondence is stored in
    ``TemplateInstance.row_map`` (deterministic nearest-neighbour in
    ``scripts/compress.py``); when it cannot be built the writer must fail
    instead of claiming losslessness.
  - ``full``: store the instance attribute tensor verbatim (lossless, no
    sharing benefit for that attribute).
  - ``zero`` (``dc_only``): drop instance SH. Metadata carries
    ``sh_mode: zero`` and ``lossless: false``; this is a lossy reconstruction
    and must not be reported as equivalent.

  A claim of losslessness is only made when every attribute of the instance is
  either stored verbatim or covered by a residual plus a proven row map.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import torch

from .math_utils import quat_multiply, quat_to_rotmat, rotmat_to_quat


FORMAT_VERSION = "0.2"

#: Attribute names that hold scales in log domain for both standard 3DGS and
#: HAC++ checkpoints. Instance scaling must add ``log(s)`` for these.
LOG_DOMAIN_SCALING_KEYS = ("scaling",)

SH_MODES = ("residual", "full", "zero")


class ScalingDomain(str, Enum):
    """Parameter domain of a scale tensor."""

    LOG = "log"
    LINEAR = "linear"


@dataclass
class TemplateInstance:
    """Pose and template relation for one reconstructed instance.

    ``rotation`` is a world-from-canonical quaternion (w, x, y, z) or a 3x3
    matrix. ``translation`` is expressed in world units and is added *after*
    the rotation about ``center``. ``scale`` is a scalar uniform scale.

    ``row_map`` is the deterministic correspondence between this instance's
    Gaussians and the canonical template rows: ``row_map[j]`` is the template
    row that instance Gaussian ``j`` is derived from. ``None`` means identity
    (the instance uses every template row, in order), which is only valid when
    the target cluster has exactly the template's row count.
    """

    template_id: int
    instance_id: int
    rotation: torch.Tensor
    translation: torch.Tensor
    scale: torch.Tensor | None = None
    center: torch.Tensor | None = None
    # Optional per-instance appearance residual keyed by attribute name.
    residuals: dict[str, torch.Tensor] = field(default_factory=dict)
    row_map: torch.Tensor | None = None

    def to_payload(self):
        return {
            "template_id": self.template_id,
            "instance_id": self.instance_id,
            "rotation": self.rotation,
            "translation": self.translation,
            "scale": self.scale,
            "center": self.center,
            "residuals": self.residuals,
            "row_map": self.row_map,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]):
        return cls(
            template_id=payload["template_id"],
            instance_id=payload["instance_id"],
            rotation=payload["rotation"],
            translation=payload["translation"],
            scale=payload.get("scale"),
            center=payload.get("center"),
            residuals=payload.get("residuals", {}) or {},
            row_map=payload.get("row_map"),
        )


def apply_instance_scaling(scaling: torch.Tensor, scale_factor: torch.Tensor, domain: str | ScalingDomain):
    """Apply a uniform instance scale in the parameter domain of ``scaling``.

    ``domain='log'`` adds ``log(s)`` (standard 3DGS ``_scaling`` and HAC++
    anchor ``_scaling``); ``domain='linear'`` multiplies by ``s``.
    """

    domain = ScalingDomain(domain)
    s = torch.abs(scale_factor).reshape(-1)
    if s.numel() == 1:
        s = s.reshape(1).expand(scaling.shape[0] if scaling.dim() > 0 else 1)
    if domain is ScalingDomain.LOG:
        return scaling + torch.log(s).reshape([-1] + [1] * (scaling.dim() - 1))
    return scaling * s.reshape([-1] + [1] * (scaling.dim() - 1))


def instance_scale_log_factor(scale_factor: torch.Tensor):
    """Return ``log(|s|)`` as a float scalar (testing helper)."""

    return float(torch.log(torch.abs(scale_factor).reshape(-1)[0]).item())


@dataclass
class CompactScene:
    """Hierarchical compact representation used by HPS3DGS.

    ``templates`` maps ``str(template_id)`` -> ``{"indices": LongTensor[N]}``
    pointing into ``static_gaussians`` (the *canonical* template rows), or, for
    backward compatibility with format 0.1, directly to a dict of tensors.

    ``static_gaussians`` holds the basis Gaussians (xyz plus every attribute
    that is not carried by the HAC++/codebook layer).
    """

    templates: dict[str, Any]
    static_gaussians: dict[str, torch.Tensor] = field(default_factory=dict)
    instances: list[TemplateInstance] = field(default_factory=list)
    codebooks: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = FORMAT_VERSION
    #: Per-attribute parameter domain, e.g. {"scaling": "log"}. Defaults to log
    #: for the standard 3DGS/HAC++ scaling tensor.
    scaling_domains: dict[str, str] = field(default_factory=dict)
    #: How per-instance SH is encoded: residual | full | zero.
    sh_mode: str = "residual"
    #: Attributes carried as per-instance residuals (or verbatim / zeroed
    #: according to ``sh_mode`` for ``features_rest``).
    residual_attributes: list[str] = field(default_factory=lambda: ["features_dc", "features_rest"])

    def domain_of(self, key: str) -> str:
        if key in self.scaling_domains:
            return self.scaling_domains[key]
        return ScalingDomain.LOG.value if key in LOG_DOMAIN_SCALING_KEYS else ScalingDomain.LINEAR.value

    def to_payload(self):
        return {
            "version": self.version,
            "templates": self.templates,
            "static_gaussians": self.static_gaussians,
            "instances": [item.to_payload() for item in self.instances],
            "codebooks": self.codebooks,
            "metadata": self.metadata,
            "scaling_domains": self.scaling_domains,
            "sh_mode": self.sh_mode,
            "residual_attributes": self.residual_attributes,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]):
        version = payload.get("version", "unknown")
        scaling_domains = payload.get("scaling_domains")
        if scaling_domains is None:
            # Format 0.1 stored raw `_scaling` (log domain) with no marker.
            scaling_domains = {"scaling": ScalingDomain.LOG.value} if version.startswith("0.1") else {}
        sh_mode = payload.get("sh_mode")
        if sh_mode is None:
            # Format 0.1 zeroed instance SH and (incorrectly) reported it as
            # equivalent; mark the payload honestly instead.
            sh_mode = "zero"
        return cls(
            residual_attributes=payload.get(
                "residual_attributes", ["features_dc", "features_rest"]
            ),
            version=version,
            templates=payload.get("templates", {}),
            static_gaussians=payload.get("static_gaussians", {}),
            instances=[
                TemplateInstance.from_payload(item) for item in payload.get("instances", [])
            ],
            codebooks=payload.get("codebooks", {}),
            metadata=payload.get("metadata", {}),
            scaling_domains=scaling_domains,
            sh_mode=sh_mode,
        )


def save_compact_scene(scene: CompactScene, path: str | Path):
    """Save a compact scene as a torch payload."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(scene.to_payload(), path)


def load_compact_scene(path: str | Path, map_location: str | torch.device = "cpu"):
    """Load a compact scene saved by :func:`save_compact_scene`."""

    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # torch 1.x has no weights_only argument
        payload = torch.load(path, map_location=map_location)
    return CompactScene.from_payload(payload)


def decode_compact_basis(scene: CompactScene):
    """Decode only the stored basis Gaussians (no instance expansion)."""

    base = {key: value.clone() for key, value in scene.static_gaussians.items()}
    for name, codebook in scene.codebooks.items():
        values = codebook["centers"][codebook["indices"].long()]
        value_shape = tuple(codebook.get("value_shape", values.shape[1:]))
        base[name] = values.reshape(values.shape[0], *value_shape)
    return base


def resolve_template_rows(scene: CompactScene, base: dict[str, torch.Tensor]):
    """Return ``{template_id: LongTensor rows into base}``.

    Indexed templates (format >= 0.2) reference rows of ``static_gaussians``
    which is what makes templates *shared*: one canonical set of Gaussians,
    many instance poses.
    """

    if not scene.templates:
        return {}
    indexed = all(isinstance(value, dict) and "indices" in value for value in scene.templates.values())
    if indexed:
        return {
            int(template_id): value["indices"].long()
            for template_id, value in scene.templates.items()
        }
    # Format 0.1 stored one full tensor dict per template under key "0"; those
    # rows are not part of static_gaussians, so they cannot be addressed by
    # index. They are reported as absent so callers can fall back to the
    # explicit template tensors.
    return {}


def template_tensors(scene: CompactScene, template_id: int, base: dict[str, torch.Tensor]):
    """Materialise the canonical tensors of one template."""

    rows = resolve_template_rows(scene, base)
    if template_id in rows:
        return {key: value[rows[template_id]] for key, value in base.items()}
    value = scene.templates.get(str(template_id), scene.templates.get(template_id))
    if isinstance(value, dict) and "indices" in value:
        return {key: value_[value["indices"].long()] for key, value_ in base.items()}
    if isinstance(value, dict):
        return {key: value_.clone() for key, value_ in value.items()}
    raise KeyError("Unknown template id %r" % template_id)


def transform_instance_tensors(
    output: dict[str, torch.Tensor],
    rotation: torch.Tensor,
    translation: torch.Tensor,
    scale: torch.Tensor | None,
    center: torch.Tensor | None,
    scaling_domains: dict[str, str] | None = None,
):
    """Apply one rigid instance transform to canonical tensors, in place.

    Positions follow ``x_w = c + t + R @ (s * (x_c - c))``; quaternions compose
    as ``q_w = q_i (x) q_c``; scales are updated in the parameter domain of
    each tensor (log for ``scaling`` unless overridden).
    """

    scaling_domains = scaling_domains or {}
    reference = output["xyz"]
    if center is None:
        center = output["xyz"].mean(dim=0)
    else:
        center = center.to(reference)
    if rotation.shape == (4,):
        rotation_matrix = quat_to_rotmat(rotation)
        global_quat = rotation
    else:
        rotation_matrix = rotation
        global_quat = rotmat_to_quat(rotation)
    translation = translation.to(reference)
    if scale is None:
        scale = reference.new_ones(1)
    else:
        scale = torch.abs(scale.to(reference)).reshape(1)

    if "xyz" in output:
        output["xyz"] = (output["xyz"] - center) * scale
        output["xyz"] = output["xyz"].matmul(rotation_matrix.t()) + center + translation
    if "rotation" in output:
        output["rotation"] = quat_multiply(global_quat.unsqueeze(0), output["rotation"])
    for key in output:
        if key == "scaling" or key.endswith("scaling") or key.endswith("_scaling"):
            domain = scaling_domains.get(key, ScalingDomain.LOG.value)
            output[key] = apply_instance_scaling(output[key], scale, domain)
    return output


def expand_compact_basis(
    scene: CompactScene,
    base: dict[str, torch.Tensor] | None = None,
    require_residuals: bool = False,
):
    """Expand the basis into all template instances.

    ``base`` may be pre-decoded (e.g. quantised) tensors; when omitted the
    stored basis is decoded. Per-instance appearance follows ``sh_mode`` /
    ``residual_attributes`` and is never silently discarded.

    ``require_residuals=True`` is the mode a *lossless* claim must decode in:
    every attribute in ``residual_attributes`` then needs a stored residual,
    and a missing one raises instead of decoding to a silent zero.
    """

    if base is None:
        base = decode_compact_basis(scene)
    else:
        base = {key: value for key, value in base.items()}

    chunks: list[dict[str, torch.Tensor]] = []
    if base:
        chunks.append({key: value.clone() for key, value in base.items()})

    for instance in scene.instances:
        canonical = template_tensors(scene, instance.template_id, base)
        if instance.row_map is not None:
            rows = instance.row_map.to(torch.long).reshape(-1)
            if rows.numel() == 0:
                raise ValueError("Instance %d has an empty row_map" % instance.instance_id)
            if int(rows.min()) < 0 or int(rows.max()) >= canonical["xyz"].shape[0]:
                raise ValueError(
                    "Instance %d row_map references rows outside template %d "
                    "(%d rows)" % (instance.instance_id, instance.template_id, canonical["xyz"].shape[0])
                )
            canonical = {key: value[rows] for key, value in canonical.items()}
        elif require_residuals and len(scene.instances) > 0:
            raise ValueError(
                "Instance %d has no row_map; a lossless expansion needs the "
                "deterministic target->template correspondence" % instance.instance_id
            )
        output = {key: value.clone() for key, value in canonical.items()}
        transform_instance_tensors(
            output,
            instance.rotation,
            instance.translation,
            instance.scale,
            instance.center,
            scaling_domains=scene.scaling_domains,
        )
        # Per-instance appearance. Every attribute in ``residual_attributes``
        # is either reconstructed exactly from a stored residual (residual /
        # full) or explicitly zeroed (dc_only -> ``features_rest``); the choice
        # is recorded in metadata by the writer, never assumed.
        for key in list(output.keys()):
            if key not in scene.residual_attributes:
                continue
            if scene.sh_mode == "zero" and key == "features_rest":
                output[key] = torch.zeros_like(output[key])
            elif scene.sh_mode == "full":
                if key not in instance.residuals:
                    raise ValueError(
                        "sh_mode='full' requires a stored %r for instance %d"
                        % (key, instance.instance_id)
                    )
                output[key] = instance.residuals[key].to(output[key])
            else:  # residual
                residual = instance.residuals.get(key)
                if residual is None:
                    if require_residuals:
                        raise ValueError(
                            "Instance %d is missing the stored %r residual that "
                            "its lossless claim requires" % (instance.instance_id, key)
                        )
                    continue
                output[key] = output[key] + residual.to(output[key])
        chunks.append(output)

    if not chunks:
        return {}
    keys = set(chunks[0].keys())
    for chunk in chunks[1:]:
        keys &= set(chunk.keys())
    return {key: torch.cat([chunk[key] for chunk in chunks], dim=0) for key in sorted(keys)}


def decode_compact_scene(scene: CompactScene, require_residuals: bool = False):
    """Decode static templates and instances into Gaussian tensor dictionaries."""

    return expand_compact_basis(
        scene, decode_compact_basis(scene), require_residuals=require_residuals
    )


def estimate_payload_bytes(payload: Any):
    """Estimated *in-memory* tensor bytes. Not a bitstream size."""

    if isinstance(payload, torch.Tensor):
        return int(payload.numel() * payload.element_size())
    if isinstance(payload, dict):
        return sum(estimate_payload_bytes(value) for value in payload.values())
    if isinstance(payload, (list, tuple)):
        return sum(estimate_payload_bytes(value) for value in payload)
    return 0


def build_instance_residuals(
    template_tensors_by_id: dict[int, dict[str, torch.Tensor]],
    instances: list[TemplateInstance],
    attribute: str = "features_rest",
    instance_tensors: dict[int, dict[str, torch.Tensor]] | None = None,
):
    """Compute per-instance SH residuals in the canonical parameter frame.

    ``instance_tensors[instance_id][attribute]`` holds the world-frame SH of the
    instance's Gaussians (matched row-by-row to the canonical template). The
    residual is ``instance - template``; decoding adds the (unrotated) template
    SH back, so the reconstruction is exact by construction.
    """

    instance_tensors = instance_tensors or {}
    for instance in instances:
        canonical = template_tensors_by_id.get(instance.template_id)
        world = instance_tensors.get(instance.instance_id)
        if canonical is None or world is None or attribute not in canonical:
            continue
        template = canonical[attribute]
        values = world[attribute] if attribute in world else world["features_rest"]
        if values.shape != template.shape:
            raise ValueError(
                "Instance %d %s shape %s does not match template %s"
                % (instance.instance_id, attribute, tuple(values.shape), tuple(template.shape))
            )
        instance.residuals[attribute] = (values - template).detach().cpu()
    return instances


def nearest_row_map(posed_template_xyz: torch.Tensor, target_xyz: torch.Tensor):
    """Deterministic target->template-row correspondence.

    Row ``j`` of the target cluster is assigned the pose-transformed template
    row with the smallest Euclidean distance. ``torch.cdist`` + ``argmin``
    (first minimum wins) keeps the mapping deterministic for a given input
    ordering, which is what makes a lossless residual claim checkable.
    """

    if posed_template_xyz.shape[0] == 0 or target_xyz.shape[0] == 0:
        raise ValueError("Cannot build a row correspondence for an empty cluster")
    dist = torch.cdist(
        target_xyz.to(posed_template_xyz).unsqueeze(0),
        posed_template_xyz.to(target_xyz).unsqueeze(0),
    )[0]
    return torch.argmin(dist, dim=1).to(torch.long)


def compute_instance_residuals(
    scene: CompactScene,
    instance_payloads: list[dict[str, torch.Tensor]],
    attributes: list[str] | None = None,
    base: dict[str, torch.Tensor] | None = None,
):
    """Fill ``instance.residuals`` so decoding reproduces the target payloads.

    ``instance_payloads[i]`` holds the world-frame tensors of instance ``i`` as
    they should come out of the decoder (typically the *target* cluster's
    attributes). Every instance must already carry a ``row_map`` whose length
    equals its target row count - the posed template chunk is then exactly the
    gathered rows, and ``decode_compact_scene(scene)`` plus the residual equals
    the payload for the listed attributes. Anything else raises instead of
    producing a silent lossless claim.
    """

    attributes = attributes or list(scene.residual_attributes)
    if len(instance_payloads) != len(scene.instances):
        raise ValueError(
            "instance payload count %d does not match instance count %d"
            % (len(instance_payloads), len(scene.instances))
        )
    for index, instance in enumerate(scene.instances):
        if instance.row_map is None:
            raise ValueError(
                "Instance %d has no row_map; build the deterministic "
                "target->template correspondence (nearest_row_map) before "
                "computing residuals" % instance.instance_id
            )
        count = int(instance_payloads[index]["xyz"].shape[0])
        if int(instance.row_map.numel()) != count:
            raise ValueError(
                "Instance %d row_map has %d rows but the target payload has %d"
                % (instance.instance_id, int(instance.row_map.numel()), count)
            )
    decoded_base = decode_compact_basis(scene) if base is None else base
    for index, instance in enumerate(scene.instances):
        payload = instance_payloads[index]
        canonical = template_tensors(scene, instance.template_id, decoded_base)
        rows = instance.row_map.to(torch.long).reshape(-1)
        canonical = {key: value[rows] for key, value in canonical.items()}
        posed_tensors = {key: value.clone() for key, value in canonical.items()}
        transform_instance_tensors(
            posed_tensors,
            instance.rotation,
            instance.translation,
            instance.scale,
            instance.center,
            scaling_domains=scene.scaling_domains,
        )
        for attribute in attributes:
            if attribute not in payload:
                continue
            target = payload[attribute].detach()
            posed = posed_tensors[attribute]
            if posed.shape != target.shape:
                raise ValueError(
                    "Instance %d: posed template %s != target %s for %r; cannot "
                    "claim a lossless residual without a valid row_map"
                    % (instance.instance_id, tuple(posed.shape), tuple(target.shape), attribute)
                )
            if scene.sh_mode == "full":
                instance.residuals[attribute] = target.detach().cpu()
            else:
                instance.residuals[attribute] = (target - posed).detach().cpu()
    return scene


def lossless_attributes(scene: CompactScene):
    """Attributes reproduced exactly on decode, given the scene's policy."""

    if scene.sh_mode == "full":
        return list(scene.residual_attributes)
    if scene.sh_mode == "zero":
        return [name for name in scene.residual_attributes if name != "features_rest"]
    return [
        name
        for name in scene.residual_attributes
        if all(name in instance.residuals for instance in scene.instances)
    ]
