#!/usr/bin/env python3
"""Run HPS3DGS compression.

Closed-loop contract enforced here:

* every tensor the evaluation renders comes from the *saved* CompactScene -
  the scene is written to disk, reloaded, decoded and only then rendered. The
  unquantized reference is evaluated from the original scene tensors.
* instance payloads are built from each alignment's **target** cluster
  (``tgt_mask``), and a deterministic target->posed-template row map is built
  before residuals, so ``sh_mode=residual`` really reconstructs the target
  appearance.
* reported sizes distinguish the actual on-disk bitstream (the saved file's
  size) from the in-memory tensor estimate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from src.backend import load_trained_scene
from src.codec import (
    CompactScene,
    TemplateInstance,
    compute_instance_residuals,
    decode_compact_basis,
    decode_compact_scene,
    estimate_payload_bytes,
    load_compact_scene,
    lossless_attributes,
    nearest_row_map,
    save_compact_scene,
)
from src.compression import (
    ClusterConfig,
    MatchingConfig,
    HPS3DGSConfig,
    RefinementConfig,
    run_geo32_compression,
)
from src.metrics import compression_report
from src.quantization import quantize_appearance_attributes


def load_config(path):
    cfg_raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config = HPS3DGSConfig(
        feature_dim=int(cfg_raw.get("feature_dim", 32)),
        feature_iteration=int(cfg_raw.get("feature_iteration", 10000)),
        appearance_keep_sh=bool(cfg_raw.get("appearance", {}).get("keep_instance_sh", True)),
        clustering=ClusterConfig(**cfg_raw.get("clustering", {})),
        matching=MatchingConfig(**cfg_raw.get("matching", {})),
        refinement=RefinementConfig(**cfg_raw.get("refinement", {})),
    )
    return cfg_raw, config


def codebook_payload(codebooks):
    return {
        name: {
            "centers": codebook.centers.cpu(),
            "indices": codebook.indices.cpu(),
            "value_shape": codebook.value_shape,
        }
        for name, codebook in codebooks.items()
    }


def quantized_attribute_names(appearance_cfg):
    aliases = {
        "dc": "features_dc",
        "sh": "features_rest",
        "scale": "scaling",
        "rot": "rotation",
        "rotation": "rotation",
        "opacity": "opacity",
    }
    configured = appearance_cfg.get(
        "quantized_attributes",
        ["dc", "sh", "scale", "rotation", "opacity"],
    )
    return [aliases[name] for name in configured]


def make_gaussian_model(backend, tensors, sh_degree, active_sh_degree, device):
    model = backend.GaussianModel(sh_degree)
    model._xyz = nn.Parameter(tensors["xyz"].to(device))
    model._features_dc = nn.Parameter(tensors["features_dc"].to(device))
    model._features_rest = nn.Parameter(tensors["features_rest"].to(device))
    model._scaling = nn.Parameter(tensors["scaling"].to(device))
    model._rotation = nn.Parameter(tensors["rotation"].to(device))
    model._opacity = nn.Parameter(tensors["opacity"].to(device))
    model._mask = torch.ones((model._xyz.shape[0], 1), device=device)
    model.max_radii2D = torch.zeros(model._xyz.shape[0], device=device)
    model.xyz_gradient_accum = torch.zeros((model._xyz.shape[0], 1), device=device)
    model.denom = torch.zeros((model._xyz.shape[0], 1), device=device)
    model.active_sh_degree = active_sh_degree
    return model


def current_gaussian_tensors(gaussians):
    return {
        "xyz": gaussians._xyz.detach().clone(),
        "features_dc": gaussians._features_dc.detach().clone(),
        "features_rest": gaussians._features_rest.detach().clone(),
        "scaling": gaussians._scaling.detach().clone(),
        "rotation": gaussians._rotation.detach().clone(),
        "opacity": gaussians._opacity.detach().clone(),
    }


def build_compact_scene(gaussians, result, appearance_cfg, sh_mode="residual"):
    """Build the CompactScene, including target payloads, row maps, residuals."""

    original_count = int(gaussians._xyz.shape[0])
    remove_mask = result["pts_to_remove_mask"]
    if remove_mask is None:
        remove_mask = torch.zeros(original_count, dtype=torch.bool, device=gaussians._xyz.device)
    keep_mask = ~remove_mask
    kept_indices = torch.where(keep_mask)[0]
    old_to_base = torch.full((original_count,), -1, dtype=torch.long, device=gaussians._xyz.device)
    old_to_base[kept_indices] = torch.arange(kept_indices.shape[0], device=gaussians._xyz.device)

    templates = {}
    instances: list[TemplateInstance] = []
    template_ids = {}
    target_payloads: list[dict[str, torch.Tensor]] = []
    for alignment in result["alignments"]:
        src = int(alignment["src"])
        if src not in template_ids:
            template_id = len(template_ids)
            template_ids[src] = template_id
            source_indices = torch.where(alignment["src_mask"])[0]
            templates[str(template_id)] = {
                "indices": old_to_base[source_indices].cpu(),
                "source_label": src,
            }
        transform = alignment["transform_net"]
        tgt_mask = alignment["tgt_mask"]
        src_mask = alignment["src_mask"]
        # Deterministic target -> posed-template correspondence.
        posed_template_xyz, _ = transform(gaussians._xyz[src_mask], gaussians._rotation[src_mask])
        row_map = nearest_row_map(posed_template_xyz.detach(), gaussians._xyz[tgt_mask].detach())
        instances.append(
            TemplateInstance(
                template_id=template_ids[src],
                instance_id=int(alignment["tgt"]),
                rotation=torch.nn.functional.normalize(transform.rotation_q.detach(), dim=0).cpu(),
                translation=transform.translation.detach().cpu(),
                scale=torch.abs(transform.scale_factor.detach()).cpu(),
                center=transform.src_center.detach().cpu(),
                row_map=row_map.cpu(),
            )
        )
        # Target-cluster tensors: what the decoder must reproduce.
        target_payloads.append(
            {
                "xyz": gaussians._xyz[tgt_mask].detach(),
                "rotation": gaussians._rotation[tgt_mask].detach(),
                # Target tensors are already stored in the world-frame 3DGS
                # parameter domain. Instance log-scale is applied to the posed
                # template, not to the target a second time.
                "scaling": gaussians._scaling[tgt_mask].detach(),
                "opacity": gaussians._opacity[tgt_mask].detach(),
                "features_dc": gaussians._features_dc[tgt_mask].detach(),
                "features_rest": gaussians._features_rest[tgt_mask].detach(),
            }
        )

    gaussians.remove_gaussians(remove_mask)
    attributes = {
        "features_dc": gaussians._features_dc.detach(),
        "features_rest": gaussians._features_rest.detach(),
        "scaling": gaussians._scaling.detach(),
        "rotation": gaussians._rotation.detach(),
        "opacity": gaussians._opacity.detach(),
    }
    quantized_names = set(quantized_attribute_names(appearance_cfg))
    quantized_attributes = {
        name: values for name, values in attributes.items() if name in quantized_names
    }
    configured_sizes = appearance_cfg.get("codebook_size", {})
    codebook_sizes = {
        "features_dc": configured_sizes.get("dc", 4096),
        "features_rest": configured_sizes.get("sh", 4096),
        "scaling": configured_sizes.get("scale", 4096),
        "rotation": configured_sizes.get("rotation", 4096),
        "opacity": configured_sizes.get("opacity", 4096),
    }
    with torch.no_grad():
        codebooks = quantize_appearance_attributes(
            quantized_attributes,
            codebook_sizes=codebook_sizes,
            num_iters=int(appearance_cfg.get("num_iters", 10)),
        )

    residual_attributes = [name for name in ("features_dc", "features_rest") if sh_mode != "zero" or name != "features_rest"]
    if sh_mode == "zero":
        residual_attributes = ["features_dc"]
    static_gaussians = {"xyz": gaussians._xyz.detach().cpu()}
    for name, values in attributes.items():
        if name not in quantized_names:
            static_gaussians[name] = values.detach().cpu()
    compact = CompactScene(
        templates=templates,
        static_gaussians=static_gaussians,
        instances=instances,
        codebooks=codebook_payload(codebooks),
        metadata={},
        scaling_domains={"scaling": "log"},
        sh_mode=sh_mode,
        residual_attributes=residual_attributes,
    )
    # Residuals against the *quantised* basis, so decode(basis+residual) == target.
    compute_instance_residuals(compact, target_payloads, base=decode_compact_basis(compact))

    compact.metadata = {
        "original_count": original_count,
        "basis_count": int(gaussians._xyz.shape[0]),
        "removed_count": int(remove_mask.sum().item()),
        "num_clusters": len(result["cluster_props"]),
        "num_matches": len(result["matches"]),
        "num_alignments": len(result["alignments"]),
        "num_templates": len(templates),
        "num_instances": len(instances),
        "quantized_attributes": sorted(quantized_names),
        "sh_mode": sh_mode,
        "residual_attributes": list(residual_attributes),
        "lossless_attributes": lossless_attributes(compact),
        "lossless_appearance": sh_mode in ("residual", "full"),
        "row_map_rows": [int(instance.row_map.numel()) for instance in instances],
        "residual_tensors_bytes": int(
            sum(
                tensor.numel() * tensor.element_size()
                for instance in instances
                for tensor in instance.residuals.values()
            )
        ),
        "scaling_domain": "log",
    }
    return compact


def evaluate_rendering(scene, gaussians, pipe, backend, white_background):
    """Render the held-out split when available, else the train split (stated)."""

    device = gaussians._xyz.device
    background = torch.tensor(
        [1, 1, 1] if white_background else [0, 0, 0],
        dtype=torch.float32,
        device=device,
    )
    test_cameras = scene.getTestCameras() if hasattr(scene, "getTestCameras") else []
    if test_cameras:
        cameras, split = test_cameras, "test"
    else:
        cameras, split = scene.getTrainCameras(), "train_fallback"
    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0
    render_times = []
    for camera in tqdm(cameras, desc="  Rendering %s views" % split, unit="view"):
        with torch.no_grad():
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.time()
            image = backend.render(camera, gaussians, pipe, background)["render"].clamp(0, 1)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            render_times.append(time.time() - start)
            target = camera.original_image.to(device).clamp(0, 1)
            psnr_sum += float(backend.psnr(image, target).mean().item())
            ssim_sum += float(backend.ssim(image, target).mean().item())
            lpips_sum += float(backend.lpips(image, target, net_type="vgg").mean().item())
    count = len(cameras)
    average_seconds = sum(render_times) / max(count, 1)
    return {
        "split": split,
        "test_camera_count": len(test_cameras),
        "fallback_reason": None if test_cameras else "scene has no held-out test cameras",
        "views": count,
        "psnr": psnr_sum / max(count, 1),
        "ssim": ssim_sum / max(count, 1),
        "lpips": lpips_sum / max(count, 1),
        "render_ms": average_seconds * 1000,
        "fps": 1.0 / max(average_seconds, 1e-12),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/hps-3dgs.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/scene.hps-3dgs.pt"))
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--source-path", type=Path, default=None)
    parser.add_argument("--saga-root", type=Path, default=None)
    parser.add_argument("--feature-iteration", type=int, default=None)
    parser.add_argument("--labels-path", type=Path, default=None)
    parser.add_argument("--recluster", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument(
        "--sh-mode",
        choices=["residual", "full", "zero"],
        default="residual",
        help="per-instance SH policy: residual/full are exact, zero is lossy dc_only",
    )
    args = parser.parse_args()

    cfg_raw, config = load_config(args.config)
    appearance_cfg = cfg_raw.get("appearance", {})
    if args.model_path is None:
        parser.error("--model-path is required")
    feature_iteration = args.feature_iteration or config.feature_iteration
    print("=== Stage 1/4: Loading trained scene ===")
    loaded = load_trained_scene(
        args.model_path,
        source_path=args.source_path,
        feature_iteration=feature_iteration,
        feature_dim=config.feature_dim,
        saga_root=args.saga_root,
    )
    print(f"  Loaded {loaded['gaussians']._xyz.shape[0]} Gaussians")

    legacy_labels_path = args.model_path / "micro_cluster_labels.pt"
    if args.labels_path:
        labels_path = args.labels_path
    elif legacy_labels_path.exists() and not args.recluster:
        labels_path = legacy_labels_path
    else:
        labels_path = args.output.with_suffix(".labels.pt")
    labels = None
    if labels_path.exists() and not args.recluster:
        print(f"  Loading cached labels from {labels_path}")
        labels = torch.load(labels_path, map_location=loaded["features"].device)
    print("=== Stage 2/4: Geometry compression ===")
    result = run_geo32_compression(
        loaded["scene"],
        loaded["gaussians"],
        loaded["features"],
        loaded["pipe"],
        loaded["backend"],
        config=config,
        labels=labels,
    )
    if labels is None:
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result["labels"].cpu(), labels_path)
        print(f"  Saved labels to {labels_path}")

    print("=== Stage 3/4: Building compact scene + appearance quantization ===")
    reference_tensors = current_gaussian_tensors(loaded["gaussians"])
    compact = build_compact_scene(
        loaded["gaussians"],
        result,
        appearance_cfg,
        sh_mode=args.sh_mode,
    )
    compact.metadata["model_path"] = str(args.model_path.resolve())
    compact.metadata["feature_iteration"] = feature_iteration
    compact.metadata["sh_mode"] = args.sh_mode

    # ---- close the loop: save, reload from disk, decode the reloaded object
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_compact_scene(compact, args.output)
    reloaded = load_compact_scene(args.output)
    decoded_tensors = decode_compact_scene(reloaded, require_residuals=True)
    compact.metadata["roundtrip"] = {
        "reloaded_from": str(args.output.resolve()),
        "decoded_count": int(decoded_tensors["xyz"].shape[0]),
        "decoded_attributes": sorted(decoded_tensors.keys()),
    }

    if not args.skip_evaluation:
        print("=== Stage 4/4: Rendering evaluation (decoded from disk) ===")
        device = reference_tensors["xyz"].device
        decoded_model = make_gaussian_model(
            loaded["backend"],
            decoded_tensors,
            loaded["dataset"].sh_degree,
            loaded["gaussians"].active_sh_degree,
            device,
        )
        compact.metadata["decoded_metrics"] = evaluate_rendering(
            loaded["scene"],
            decoded_model,
            loaded["pipe"],
            loaded["backend"],
            loaded["dataset"].white_background,
        )
        print("  Evaluating unquantized reference model...")
        reference_model = make_gaussian_model(
            loaded["backend"],
            reference_tensors,
            loaded["dataset"].sh_degree,
            loaded["gaussians"].active_sh_degree,
            device,
        )
        compact.metadata["reference_metrics"] = evaluate_rendering(
            loaded["scene"],
            reference_model,
            loaded["pipe"],
            loaded["backend"],
            loaded["dataset"].white_background,
        )
        save_compact_scene(compact, args.output)

    # ---- sizes: actual on-disk bitstream vs in-memory tensor estimate
    payload = reloaded.to_payload()
    report = compression_report(
        original_count=int(compact.metadata["original_count"]),
        final_count=int(decoded_tensors["xyz"].shape[0]),
        final_payload=payload,
        bitstream_paths=[args.output],
    )
    report["estimated_tensor_bytes_note"] = (
        "in-memory tensor estimate; the compressed size is bitstream_bytes"
    )
    # Do not embed a file-size report into the file it measures: doing so makes
    # the reported byte count stale immediately. Keep it in a sidecar instead.
    metrics_path = Path(str(args.output) + ".metrics.json")
    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"compression_report": report}, indent=2))
    print(f"Saved compact scene to {args.output}")
    print(f"Saved size report to {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
