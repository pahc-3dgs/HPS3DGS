"""Physics-aware geometry compression migrated from geo32.py."""

from __future__ import annotations

import hdbscan
import open3d as o3d
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import DBSCAN, KMeans
from tqdm import tqdm

from .math_utils import quat_multiply, quat_to_rotmat, rotmat_to_quat


@dataclass
class ClusterConfig:
    """Parameters for semantic and spatial clustering."""

    hdbscan_min_cluster_size: int = 500
    hdbscan_epsilon: float = 0.01
    core_set_min_samples: int = 200_000
    sample_keep_probability: float = 0.99
    target_block_size: int = 100_000
    min_block_size: int = 100
    dbscan_eps: float = 0.25
    dbscan_min_samples: int = 10


@dataclass
class MatchingConfig:
    """Parameters for geometry-aware component matching."""

    min_cluster_size: int = 1000
    wg: float = 0.4
    ws: float = 0.6
    tau_p: float = 5.0
    extent_ratio_min: float = 0.33
    extent_ratio_max: float = 3.0


@dataclass
class RefinementConfig:
    """Parameters for differentiable pose and distribution refinement."""

    tau_r: float = 0.005
    initial_pose_iters: int = 1000
    refit_pose_iters: int = 150
    refine_cycles: int = 7
    distribution_iters: int = 5000
    final_distribution_iters: int = 10000
    pose_lr_t: float = 0.005
    pose_lr_q: float = 0.001
    refit_lr_t: float = 0.0002
    refit_lr_q: float = 0.0001


@dataclass
class PAHCConfig:
    """Top-level geometry compression configuration."""

    feature_dim: int = 32
    feature_iteration: int = 10000
    clustering: ClusterConfig = field(default_factory=ClusterConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)


def micro_clustering(xyz: torch.Tensor, features: torch.Tensor, config: ClusterConfig | None = None):
    """Cluster semantic features with the core-set HDBSCAN strategy from geo32.py."""

    del xyz
    config = config or ClusterConfig()
    normed_features = F.normalize(features, dim=-1, p=2)
    total_points = normed_features.shape[0]
    if total_points == 0:
        raise ValueError("Cannot cluster an empty feature tensor.")

    sample_mask = torch.rand(total_points, device=features.device) < config.sample_keep_probability
    if sample_mask.sum() < config.core_set_min_samples and total_points > config.core_set_min_samples:
        indices = torch.randperm(total_points, device=features.device)[: config.core_set_min_samples]
        sampled_features = normed_features[indices]
    else:
        sampled_features = normed_features[sample_mask]

    n_sampled = int(sampled_features.shape[0])
    print(f"[micro_clustering] Running HDBSCAN on {n_sampled:,}/{total_points:,} sampled points "
          f"(min_cluster_size={config.hdbscan_min_cluster_size}, eps={config.hdbscan_epsilon})...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config.hdbscan_min_cluster_size,
        cluster_selection_epsilon=config.hdbscan_epsilon,
    )
    sampled_labels = clusterer.fit_predict(sampled_features.detach().cpu().numpy())

    valid_labels = [label for label in np.unique(sampled_labels) if label != -1]
    n_noise = int(np.sum(sampled_labels == -1))
    print(f"[micro_clustering] Found {len(valid_labels)} clusters, {n_noise} noise points "
          f"(from {n_sampled:,} sampled)")
    if not valid_labels:
        raise RuntimeError("HDBSCAN found no valid semantic clusters.")

    centers = []
    sampled_features_cpu = sampled_features.detach().cpu()
    for label in valid_labels:
        mask = sampled_labels == label
        center = F.normalize(sampled_features_cpu[mask].mean(dim=0), dim=-1)
        centers.append(center)
    cluster_centers = torch.stack(centers).to(features.device)
    scores = torch.matmul(normed_features, cluster_centers.t())
    return scores.argmax(dim=-1)


def split_label_spatially(
    xyz: torch.Tensor,
    labels: torch.Tensor,
    target_block_size: int = 20_000,
    min_block_size: int = 50,
    dbscan_eps: float = 0.25,
    dbscan_min_samples: int = 10,
):
    """Split semantic labels into spatially connected blocks using DBSCAN and KMeans."""

    xyz_np = xyz.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    new_labels = torch.full_like(labels, -1, dtype=torch.long)
    global_offset = 0

    unique_labels = np.unique(labels_np)
    for label in tqdm(unique_labels, desc="Spatial splitting", unit="label"):
        label_mask = labels_np == label
        pts = xyz_np[label_mask]
        if len(pts) < dbscan_min_samples:
            continue

        db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, n_jobs=-1).fit(pts)
        sub_labels = db.labels_
        original_indices = np.where(label_mask)[0]
        for sub_label in np.unique(sub_labels):
            if sub_label == -1:
                continue
            sub_mask = sub_labels == sub_label
            count = int(np.sum(sub_mask))
            if count < min_block_size:
                continue
            curr_indices = original_indices[sub_mask]
            if count > target_block_size * 1.5:
                k = max(2, round(count / target_block_size))
                km_labels = KMeans(n_clusters=k, n_init=3, random_state=42).fit_predict(xyz_np[curr_indices])
                for k_id in range(k):
                    final_indices = curr_indices[km_labels == k_id]
                    new_labels[final_indices] = global_offset
                    global_offset += 1
            else:
                new_labels[curr_indices] = global_offset
                global_offset += 1

    noise_mask = new_labels == -1
    noise_count = int(noise_mask.sum().item())
    if noise_count > 0:
        new_labels[noise_mask] = torch.arange(
            global_offset,
            global_offset + noise_count,
            device=labels.device,
            dtype=torch.long,
        )
    return new_labels


def compute_cluster_properties(
    xyz: torch.Tensor,
    features: torch.Tensor,
    scaling: torch.Tensor | None,
    labels: torch.Tensor,
    min_size: int = 50,
):
    """Compute covariance spectra and mean semantic features for component blocks."""

    del scaling
    props = []
    for label in tqdm(torch.unique(labels).tolist(), desc="Computing cluster props", unit="cluster"):
        mask = labels == label
        count = int(mask.sum().item())
        if count < min_size:
            continue
        pts = xyz[mask]
        feats = features[mask]
        if feats.dim() == 3:
            feats = feats.squeeze(1)
        mean_feat = F.normalize(feats.mean(dim=0), dim=0).detach().cpu()
        if count < 4:
            eigenvalues = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float32)
        else:
            centered = pts - pts.mean(dim=0)
            cov = centered.t().matmul(centered) / max(count - 1, 1)
            eigenvalues = torch.clamp(torch.linalg.eigvalsh(cov), min=1e-8).detach().cpu()
        props.append(
            {
                "label": int(label),
                "count": count,
                "mean_feat": mean_feat,
                "eigenvalues": eigenvalues,
            }
        )
    return props


def spectral_divergence(eigenvalues_i: torch.Tensor, eigenvalues_j: torch.Tensor):
    """Rotation-invariant spectral divergence from the paper geometry layer."""

    lambda_i, _ = torch.sort(eigenvalues_i, dim=-1, descending=True)
    lambda_j, _ = torch.sort(eigenvalues_j, dim=-1, descending=True)
    term1 = lambda_i / (lambda_j + 1e-6)
    term2 = lambda_j / (lambda_i + 1e-6)
    return 0.5 * torch.sum(term1 + term2 - 2, dim=-1)


def combined_distance(props_i, props_j, w_g: float = 0.4, w_s: float = 0.6):
    """Combine spectral divergence and semantic cosine distance."""

    d_s = spectral_divergence(props_i["eigenvalues"], props_j["eigenvalues"])
    d_geo = torch.tanh(d_s / 3.0)
    feat_i = props_i["mean_feat"]
    feat_j = props_j["mean_feat"]
    if not torch.isclose(torch.norm(feat_i), torch.tensor(1.0), atol=1e-3):
        feat_i = F.normalize(feat_i, dim=0)
        feat_j = F.normalize(feat_j, dim=0)
    d_sem = 1.0 - torch.dot(feat_i, feat_j)
    return w_g * d_geo + w_s * d_sem


def find_matches_physics_aware(
    cluster_props,
    threshold: float = 0.85,
    w_g: float = 0.4,
    w_s: float = 0.6,
):
    """Select global template-instance matches with best-first gating."""

    possible = []
    num_props = len(cluster_props)
    print(f"[find_matches] Checking {num_props * (num_props - 1) // 2} candidate pairs from {num_props} clusters...")
    for i in range(num_props):
        for j in range(i + 1, len(cluster_props)):
            p1 = cluster_props[i]
            p2 = cluster_props[j]
            dist = float(combined_distance(p1, p2, w_g=w_g, w_s=w_s).item())
            if dist < threshold:
                if p1["count"] >= p2["count"]:
                    src, tgt = p1["label"], p2["label"]
                else:
                    src, tgt = p2["label"], p1["label"]
                possible.append({"src": src, "tgt": tgt, "dist": dist})
    possible.sort(key=lambda item: item["dist"])

    matches = []
    templates: set[int] = set()
    replaced_instances: set[int] = set()
    for match in possible:
        src = match["src"]
        tgt = match["tgt"]
        if tgt not in replaced_instances and tgt not in templates and src not in replaced_instances:
            matches.append((src, tgt))
            templates.add(src)
            replaced_instances.add(tgt)
    print(f"[find_matches] Selected {len(matches)} template-instance pairs from "
          f"{len(possible)} candidates below threshold")
    return matches


def preprocess_point_cloud(pcd: Any, voxel_size: float):
    """Downsample a point cloud and compute FPFH features."""

    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd_down, pcd_fpfh


def get_ransac_alignment(src_pts: torch.Tensor, tgt_pts: torch.Tensor):
    """Estimate coarse rigid alignment with Open3D FPFH and RANSAC."""

    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(src_pts.detach().cpu().numpy())
    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(tgt_pts.detach().cpu().numpy())
    max_extent = max(tgt_pcd.get_axis_aligned_bounding_box().get_max_extent(), 1e-6)
    voxel_size = max_extent / 100.0
    src_down, src_fpfh = preprocess_point_cloud(src_pcd, voxel_size)
    tgt_down, tgt_fpfh = preprocess_point_cloud(tgt_pcd, voxel_size)
    distance_threshold = voxel_size * 2
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down,
        tgt_down,
        src_fpfh,
        tgt_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(True),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    transform = torch.from_numpy(result.transformation.copy()).float().to(src_pts.device)
    return transform[:3, :3], transform[:3, 3]


class TransformModule(nn.Module):
    """Differentiable transform module kept close to the geo32 implementation."""

    def __init__(self, init_R: torch.Tensor, init_T: torch.Tensor, src_center: torch.Tensor, init_s: float = 1.0):
        super().__init__()
        self.register_buffer("src_center", src_center)
        self.translation = nn.Parameter(init_T.clone())
        self.rotation_q = nn.Parameter(rotmat_to_quat(init_R))
        self.scale_factor = nn.Parameter(torch.tensor([init_s], dtype=torch.float32, device=init_R.device))

    def forward(self, points: torch.Tensor, local_quats: torch.Tensor):
        global_q = F.normalize(self.rotation_q, dim=0)
        R = quat_to_rotmat(global_q)
        points_centered = points - self.src_center
        iso_scale = torch.abs(self.scale_factor).expand(3)
        rotated_points = torch.matmul(points_centered * iso_scale.unsqueeze(0), R.t())
        transformed_points = rotated_points + self.src_center + self.translation
        transformed_quats = quat_multiply(global_q.unsqueeze(0), local_quats)
        return transformed_points, transformed_quats


def create_sub_gaussian_model(original_gaussians: Any, mask: torch.Tensor, gaussian_cls: Any | None = None):
    """Create a masked Gaussian model with the fields used by refinement."""

    cls = gaussian_cls or original_gaussians.__class__
    sub_gm = cls(original_gaussians.active_sh_degree)
    idx = torch.where(mask)[0]
    sub_gm._xyz = nn.Parameter(original_gaussians._xyz[idx].clone())
    sub_gm._features_dc = nn.Parameter(original_gaussians._features_dc[idx].clone())
    sub_gm._features_rest = torch.zeros_like(original_gaussians._features_rest[: len(idx)])
    sub_gm._scaling = nn.Parameter(original_gaussians._scaling[idx].clone())
    sub_gm._rotation = nn.Parameter(original_gaussians._rotation[idx].clone())
    sub_gm._opacity = nn.Parameter(original_gaussians._opacity[idx].clone())
    device = original_gaussians._xyz.device
    sub_gm.max_radii2D = torch.zeros((len(idx)), device=device)
    sub_gm.xyz_gradient_accum = torch.zeros((len(idx), 1), device=device)
    sub_gm.denom = torch.zeros((len(idx), 1), device=device)
    sub_gm.active_sh_degree = original_gaussians.active_sh_degree
    return sub_gm


def optimize_pose_simple(
    scene: Any,
    orig_gaussians: Any,
    pipe: Any,
    src_mask: torch.Tensor,
    tgt_mask: torch.Tensor,
    prev_transform: TransformModule,
    render_with_depth: Any,
    l1_loss: Any,
    iters: int = 150,
    lr_t: float = 0.005,
    lr_q: float = 0.001,
):
    """Optimize a candidate pose with rendered RGB and depth losses."""

    cameras = scene.getTrainCameras()
    device = orig_gaussians._xyz.device
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    src_gm = create_sub_gaussian_model(orig_gaussians, src_mask)
    tgt_gm = create_sub_gaussian_model(orig_gaussians, tgt_mask)
    src_orig_xyz = src_gm._xyz.detach().clone()
    src_orig_rot = src_gm._rotation.detach().clone()
    transform_net = TransformModule(
        quat_to_rotmat(F.normalize(prev_transform.rotation_q.detach(), dim=0)),
        prev_transform.translation.detach(),
        prev_transform.src_center.detach().clone(),
        init_s=float(prev_transform.scale_factor.item()),
    ).to(device)
    optimizer = torch.optim.Adam(
        [
            {"params": [transform_net.translation], "lr": lr_t},
            {"params": [transform_net.rotation_q], "lr": lr_q},
        ]
    )
    tgt_ones = torch.ones((tgt_gm._xyz.shape[0], 1), dtype=torch.float32, device=device)
    src_ones = torch.ones((src_gm._xyz.shape[0], 1), dtype=torch.float32, device=device)
    loss_history = {"iter": [], "rgb": [], "depth": [], "total": []}
    for i in tqdm(range(iters), desc="  Pose optimization", unit="iter", leave=False):
        view_cam = random.choice(cameras)
        with torch.no_grad():
            gt_pkg = render_with_depth(view_cam, tgt_gm, pipe, bg_color, override_mask=tgt_ones)
            gt_image = gt_pkg["render"]
            gt_depth = gt_pkg["depth"]
        src_gm._xyz, src_gm._rotation = transform_net(src_orig_xyz, src_orig_rot)
        fg_pkg = render_with_depth(view_cam, src_gm, pipe, bg_color, override_mask=src_ones)
        loss_rgb = l1_loss(fg_pkg["render"], gt_image)
        loss_depth = l1_loss(fg_pkg["depth"], gt_depth)
        loss = loss_rgb + 0.1 * loss_depth
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history["iter"].append(i)
        loss_history["rgb"].append(float(loss_rgb.item()))
        loss_history["depth"].append(float(loss_depth.item()))
        loss_history["total"].append(float(loss.item()))
    final_metric = loss_history["total"][-1] if loss_history["total"] else 0.0
    return transform_net, final_metric, loss_history


def global_distribution_alignment(
    scene: Any,
    gaussians: Any,
    pipeline_args: Any,
    labels: torch.Tensor,
    accurate_alignments,
    render: Any,
    l1_loss: Any,
    ssim: Any,
    iters: int = 300,
):
    """Refine Gaussian parameters using render and local distribution losses."""

    if not accurate_alignments:
        return {"loss_render": [], "loss_dist": [], "loss_xyz": [], "loss_col": [], "loss_scale": [], "total_loss": []}

    optimizer = torch.optim.Adam(
        [
            {"params": [gaussians._xyz], "lr": 0.0001, "name": "xyz"},
            {"params": [gaussians._features_dc], "lr": 0.002, "name": "f_dc"},
            {"params": [gaussians._scaling], "lr": 0.001, "name": "scaling"},
            {"params": [gaussians._rotation], "lr": 0.001, "name": "rotation"},
        ]
    )
    cameras = scene.getTrainCameras()
    device = gaussians._xyz.device
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    loss_dict = {"loss_render": [], "loss_dist": [], "loss_xyz": [], "loss_col": [], "loss_scale": [], "total_loss": []}

    for _step in tqdm(range(iters), desc="  Distribution alignment", unit="iter", leave=False):
        optimizer.zero_grad()
        cam = random.choice(cameras)
        image = render(cam, gaussians, pipeline_args, bg_color)["render"]
        gt_image = cam.original_image.to(device)
        loss_render = 0.8 * l1_loss(image, gt_image) + 0.2 * (1.0 - ssim(image, gt_image))
        loss_dist: torch.Tensor | float = 0.0
        step_xyz = 0.0
        step_col = 0.0
        step_scale = 0.0
        for align in accurate_alignments:
            sm = align["src_mask"]
            tm = align["tgt_mask"]
            src_xyz = gaussians._xyz[sm]
            src_rot = gaussians._rotation[sm]
            src_scale = gaussians._scaling[sm]
            src_dc = gaussians._features_dc[sm]
            tgt_xyz = gaussians._xyz[tm]
            tgt_scale = gaussians._scaling[tm]
            tgt_dc = gaussians._features_dc[tm]
            transform_net = align["transform_net"]
            trans_xyz, _ = transform_net(src_xyz, src_rot)
            trans_scale = src_scale * torch.abs(transform_net.scale_factor).expand(3).unsqueeze(0)
            sample_n = min(800, src_xyz.shape[0])
            sample_m = min(800, tgt_xyz.shape[0])
            idx_s = torch.randperm(src_xyz.shape[0], device=device)[:sample_n]
            idx_t = torch.randperm(tgt_xyz.shape[0], device=device)[:sample_m]
            src_xyz_sub = trans_xyz[idx_s]
            tgt_xyz_sub = tgt_xyz[idx_t]
            dists = torch.cdist(src_xyz_sub, tgt_xyz_sub)
            min_dist_s, min_id_s = torch.min(dists, dim=1)
            min_dist_t, min_id_t = torch.min(dists, dim=0)
            loss_xyz = min_dist_s.mean() + min_dist_t.mean()
            src_dc_sub = src_dc[idx_s]
            tgt_dc_sub = tgt_dc[idx_t]
            loss_col = F.mse_loss(src_dc_sub, tgt_dc_sub[min_id_s]) + F.mse_loss(tgt_dc_sub, src_dc_sub[min_id_t])
            src_scale_sub = trans_scale[idx_s]
            tgt_scale_sub = tgt_scale[idx_t]
            loss_scale = F.mse_loss(src_scale_sub, tgt_scale_sub[min_id_s]) + F.mse_loss(tgt_scale_sub, src_scale_sub[min_id_t])
            step_xyz += float(loss_xyz.item())
            step_col += float(loss_col.item())
            step_scale += float(loss_scale.item())
            loss_dist = loss_dist + loss_xyz * 20.0 + loss_col * 2.0 + loss_scale * 5.0
        total_loss = loss_render * 10.0 + loss_dist
        total_loss.backward()
        optimizer.step()
        loss_dict["loss_render"].append(float(loss_render.item()))
        loss_dict["loss_dist"].append(float(loss_dist.item() if isinstance(loss_dist, torch.Tensor) else loss_dist))
        loss_dict["loss_xyz"].append(step_xyz)
        loss_dict["loss_col"].append(step_col)
        loss_dict["loss_scale"].append(step_scale)
        loss_dict["total_loss"].append(float(total_loss.item()))
    return loss_dict


def build_instance_payloads(gaussians: Any, alignments, keep_appearance: bool = True):
    """Build transformed instance payloads and the mask of replaced primitives.

    The payload holds the *world-frame* appearance of the instance, i.e. the
    source cluster's attributes after the refined pose. ``keep_appearance``
    controls whether ``features_rest`` is carried (lossless) or dropped
    (``dc_only``); dropping must be recorded in the payload metadata by the
    caller, it is not silently done here.
    """

    pts_to_remove_mask = torch.zeros(gaussians._xyz.shape[0], dtype=torch.bool, device=gaussians._xyz.device)
    payloads = []
    for align in alignments:
        src_mask = align["src_mask"]
        tgt_mask = align["tgt_mask"]
        transform_net = align["transform_net"]
        with torch.no_grad():
            fin_trans_xyz, fin_trans_rot = transform_net(gaussians._xyz[src_mask], gaussians._rotation[src_mask])
            iso_scale = torch.abs(transform_net.scale_factor).expand(3).unsqueeze(0)
            payloads.append(
                {
                    "xyz": fin_trans_xyz,
                    "rotation": fin_trans_rot,
                    "scaling": gaussians._scaling[src_mask] * iso_scale,
                    "opacity": gaussians._opacity[src_mask].clone(),
                    "features_dc": gaussians._features_dc[src_mask].clone(),
                    "features_rest": (
                        gaussians._features_rest[src_mask].clone()
                        if keep_appearance
                        else torch.zeros_like(gaussians._features_rest[src_mask])
                    ),
                }
            )
        pts_to_remove_mask = pts_to_remove_mask | tgt_mask
    return pts_to_remove_mask, payloads


def refine_matched_components(
    scene,
    gaussians,
    pipe,
    labels,
    matches,
    backend,
    config=None,
):
    config = config or PAHCConfig()
    refinement = config.refinement
    xyz = gaussians.get_xyz.detach()
    alignments = []

    print(f"[refine] Processing {len(matches)} matched pairs...")
    for src, tgt in tqdm(matches, desc="Refining matches", unit="pair"):
        src_mask = labels == src
        tgt_mask = labels == tgt
        src_pts = xyz[src_mask]
        tgt_pts = xyz[tgt_mask]
        if src_pts.shape[0] < 4 or tgt_pts.shape[0] < 4:
            continue

        src_extent = (src_pts.max(0)[0] - src_pts.min(0)[0]).norm()
        tgt_extent = (tgt_pts.max(0)[0] - tgt_pts.min(0)[0]).norm()
        if src_extent <= 1e-8:
            continue
        ratio = float((tgt_extent / src_extent).item())
        if ratio < config.matching.extent_ratio_min or ratio > config.matching.extent_ratio_max:
            continue

        init_R, init_T = get_ransac_alignment(src_pts, tgt_pts)
        src_center = src_pts.mean(dim=0)
        init_T_centered = init_T + init_R.matmul(src_center) - src_center
        transform = TransformModule(init_R, init_T_centered, src_center).to(xyz.device)
        transform, final_metric, loss_history = optimize_pose_simple(
            scene,
            gaussians,
            pipe,
            src_mask,
            tgt_mask,
            transform,
            backend.render_with_depth,
            backend.l1_loss,
            iters=refinement.initial_pose_iters,
            lr_t=refinement.pose_lr_t,
            lr_q=refinement.pose_lr_q,
        )
        if final_metric < refinement.tau_r:
            alignments.append(
                {
                    "src": src,
                    "tgt": tgt,
                    "src_mask": src_mask,
                    "tgt_mask": tgt_mask,
                    "transform_net": transform,
                    "loss_history": loss_history,
                }
            )

    for cycle in tqdm(range(refinement.refine_cycles), desc="Global distribution cycles", unit="cycle"):
        if not alignments:
            break
        iterations = refinement.distribution_iters
        if cycle == refinement.refine_cycles - 1:
            iterations = refinement.final_distribution_iters
        global_distribution_alignment(
            scene,
            gaussians,
            pipe,
            labels,
            alignments,
            backend.render,
            backend.l1_loss,
            backend.ssim,
            iters=iterations,
        )

        if cycle == refinement.refine_cycles - 1:
            continue
        for alignment in alignments:
            transform, final_metric, loss_history = optimize_pose_simple(
                scene,
                gaussians,
                pipe,
                alignment["src_mask"],
                alignment["tgt_mask"],
                alignment["transform_net"],
                backend.render_with_depth,
                backend.l1_loss,
                iters=refinement.refit_pose_iters,
                lr_t=refinement.refit_lr_t,
                lr_q=refinement.refit_lr_q,
            )
            alignment["transform_net"] = transform
            alignment["refit_metric"] = final_metric
            alignment["refit_loss_history"] = loss_history

    pts_to_remove_mask, payloads = build_instance_payloads(
        gaussians, alignments, keep_appearance=keep_appearance
    )
    return alignments, pts_to_remove_mask, payloads


def run_geo32_compression(
    scene,
    gaussians,
    features,
    pipe,
    backend,
    config=None,
    labels=None,
):
    config = config or PAHCConfig()
    discovery = run_geometry_compression(
        gaussians.get_xyz.detach(),
        features,
        gaussians._scaling.detach(),
        config,
        labels,
    )
    labels = discovery["labels"]
    alignments, pts_to_remove_mask, payloads = refine_matched_components(
        scene,
        gaussians,
        pipe,
        labels,
        discovery["matches"],
        backend,
        config,
    )
    return {
        "labels": labels,
        "cluster_props": discovery["cluster_props"],
        "matches": discovery["matches"],
        "alignments": alignments,
        "pts_to_remove_mask": pts_to_remove_mask,
        "instance_payloads": payloads,
    }


def run_geometry_compression(
    xyz: torch.Tensor,
    features: torch.Tensor,
    scaling: torch.Tensor | None = None,
    config: PAHCConfig | None = None,
    labels: torch.Tensor | None = None,
):
    """Run the non-rendering geometry discovery path used by smoke tests and dry runs."""

    config = config or PAHCConfig()
    if labels is None:
        labels = micro_clustering(xyz, features, config.clustering)
        labels = split_label_spatially(
            xyz,
            labels,
            target_block_size=config.clustering.target_block_size,
            min_block_size=config.clustering.min_block_size,
            dbscan_eps=config.clustering.dbscan_eps,
            dbscan_min_samples=config.clustering.dbscan_min_samples,
        )
    props = compute_cluster_properties(xyz, features, scaling, labels, min_size=config.matching.min_cluster_size)
    matches = find_matches_physics_aware(
        props,
        threshold=config.matching.tau_p,
        w_g=config.matching.wg,
        w_s=config.matching.ws,
    )
    return {
        "labels": labels,
        "cluster_props": props,
        "matches": matches,
        "alignments": [],
        "pts_to_remove_mask": None,
        "instance_payloads": [],
    }
