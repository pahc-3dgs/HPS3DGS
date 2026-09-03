"""Shared-HAC++ instance math (CPU, torch only).

Everything here is pure tensor algebra so it can be unit tested without CUDA
and without importing the external HAC++ repository.

Conventions
-----------
A component instance maps canonical (template) space to world space with an
affine map::

    x_w = c + t + R @ (s * (x_c - c))
        = A x_c + b,   A = s R,   b = c + t - s R c

where ``c`` is the template pivot (instance ``center``), ``R`` the instance
rotation, ``t`` its translation and ``s`` a uniform scale. Note the pivot stays
fixed: the instance is scaled/rotated *about* ``c`` and then translated.

For generated Gaussians (HAC++ anchor + neural offsets):

* anchor position:  ``anchor_w = A anchor_c + b``
* offset direction: a canonical direction ``v`` maps to ``s R v``, so the world
  offset of an anchor is ``s R (raw_offset * voxel_scaling_c)``
* Gaussian scale:   HAC++ ``get_scaling`` is already physical (``exp``), so the
  world scale is ``s * scaling_c`` (linear domain). Standard 3DGS ``_scaling``
  is log-domain, so there the instance scale must be *added* as ``log(s)``.
* Gaussian rotation:``q_w = q_i (x) q_gen``
* MLP conditioning: the HAC++ MLPs take the *local* view direction, i.e. the
  world view vector must be mapped back into the template frame
  (``R^T d_w``, and divided by ``s`` for the camera position).
"""

from __future__ import annotations

import torch

from ..math_utils import quat_multiply, quat_to_rotmat


def affine_from_instance(
    center: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    scale: torch.Tensor | float | None = None,
):
    """Return ``(A, b)`` with ``x_w = A x_c + b`` for one instance pose."""

    if rotation.shape == (4,):
        R = quat_to_rotmat(rotation)
    else:
        R = rotation
    R = R.reshape(3, 3)
    center = center.reshape(3)
    translation = translation.reshape(3)
    if scale is None:
        s = torch.ones((), dtype=R.dtype, device=R.device)
    else:
        s = torch.abs(torch.as_tensor(scale, dtype=R.dtype, device=R.device)).reshape(())
    A = s * R
    b = center + translation - s * (R @ center)
    return A, b


def to_local_point(A: torch.Tensor, b: torch.Tensor, x_w: torch.Tensor):
    """Map world points back into canonical space: ``x_c = R^T (x_w - b) / s``."""

    R = A / torch.linalg.norm(A[:, 0]).clamp_min(1e-12)
    s = torch.linalg.norm(A[:, 0])
    return (x_w - b) @ R * (1.0 / s)


def transform_positions(x_c, center, rotation, translation, scale=None):
    """``c + t + R (s (x_c - c))`` for one instance pose over N points."""

    A, b = affine_from_instance(center, rotation, translation, scale)
    return x_c @ A.t() + b


def compose_rotation(local_quat: torch.Tensor, instance_quat: torch.Tensor):
    """``q_w = q_i (x) q_local`` (world-from-canonical composed with canonical)."""

    return quat_multiply(instance_quat.unsqueeze(0), local_quat)


def transform_scaling(scaling: torch.Tensor, scale, domain: str = "log"):
    """Apply a uniform instance scale in the parameter domain of ``scaling``.

    ``domain='log'`` for the stored ``_scaling`` of standard 3DGS and HAC++
    anchors (adds ``log(s)``); ``domain='linear'`` for physical scales.
    """

    s = torch.abs(torch.as_tensor(scale, dtype=scaling.dtype, device=scaling.device)).reshape(-1)
    if s.numel() == 1:
        s = s.reshape(1).expand(scaling.shape[0])
    if domain == "log":
        return scaling + torch.log(s).reshape(-1, *([1] * (scaling.dim() - 1)))
    if domain == "linear":
        return scaling * s.reshape(-1, *([1] * (scaling.dim() - 1)))
    raise ValueError("Unknown scaling domain %r" % domain)


def world_view_to_local(view_dir_w: torch.Tensor, rotation: torch.Tensor):
    """Rotate world view directions into the canonical frame: ``R^T d_w``."""

    if rotation.shape == (4,):
        R = quat_to_rotmat(rotation)
    else:
        R = rotation
    return view_dir_w @ R


def local_camera_center(
    center: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    scale,
    camera_center_w: torch.Tensor,
):
    """Camera center expressed in canonical space.

    Defined so that ``anchor_w - cam_w == s R (anchor_c - cam_c)`` for every
    anchor, which is exactly what the view-adaptive MLP conditioning expects.
    """

    A, b = affine_from_instance(center, rotation, translation, scale)
    R = A / torch.linalg.norm(A[:, 0]).clamp_min(1e-12)
    s = torch.linalg.norm(A[:, 0])
    return (camera_center_w.reshape(1, 3) - b.reshape(1, 3)) @ R * (1.0 / s)


def expand_instance_anchors(
    anchor_c: torch.Tensor,
    raw_offsets: torch.Tensor,
    voxel_scaling_c: torch.Tensor,
    center: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    scale=None,
):
    """World-space Gaussian centers for one HAC++ instance.

    Parameters
    ----------
    anchor_c:
        ``[N, 3]`` canonical anchors.
    raw_offsets:
        ``[N, K, 3]`` decoded HAC++ offsets (before the voxel-scale multiply).
    voxel_scaling_c:
        ``[N, 3]`` physical anchor scaling (``exp(_scaling[:, :3])``) used by
        HAC++ to scale raw offsets in :func:`generate_neural_gaussians`.
    """

    if raw_offsets.dim() == 2:
        raw_offsets = raw_offsets.unsqueeze(1)
    if raw_offsets.dim() != 3 or raw_offsets.shape[-1] != 3:
        raise ValueError("raw_offsets must have shape [N, K, 3] or [N, 3]")
    if voxel_scaling_c.dim() != 2 or voxel_scaling_c.shape[-1] != 3:
        raise ValueError("voxel_scaling_c must have shape [N, 3]")
    if raw_offsets.shape[0] != voxel_scaling_c.shape[0]:
        raise ValueError("raw_offsets and voxel_scaling_c must share N")
    offsets_c = raw_offsets * voxel_scaling_c.unsqueeze(1)  # [N, K, 3]
    A, b = affine_from_instance(center, rotation, translation, scale)
    anchor_w = anchor_c @ A.t() + b  # [N, 3]
    offsets_w = torch.einsum("ij,nkj->nki", A, offsets_c)  # s R v
    return anchor_w.unsqueeze(1) + offsets_w  # [N, K, 3]


def expand_instance_gaussian_rotation(gen_quat_c: torch.Tensor, instance_quat: torch.Tensor):
    """World rotation of MLP-generated Gaussians: ``q_i (x) q_gen``."""

    return compose_rotation(gen_quat_c, instance_quat)


def expand_instance_gaussian_scaling(
    gen_scaling_c: torch.Tensor,
    scale,
    domain: str = "linear",
):
    """World scale of MLP-generated Gaussians.

    HAC++ generates physical scales (``sigmoid(mlp) * anchor_scaling``), so the
    instance scale multiplies in the linear domain by default.
    """

    return transform_scaling(gen_scaling_c, scale, domain=domain)
