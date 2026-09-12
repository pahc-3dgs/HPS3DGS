"""SegAnyGaussians backend used by HPS3DGS."""

import argparse
import importlib
import os
import sys
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn.functional as F


def ensure_saga_on_path(saga_root=None):
    root = saga_root or os.environ.get("HPS_3DGS_SAGA_ROOT")
    if root is None:
        root = Path(__file__).resolve().parents[1] / "third_party" / "SegAnyGAussians"
    root = Path(root).resolve()

    if not (root / "scene").exists() or not (root / "gaussian_renderer").exists():
        raise RuntimeError(
            "SegAnyGaussians is not available. Initialize third_party/SegAnyGAussians "
            "or set HPS_3DGS_SAGA_ROOT."
        )

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


class SagaBackend:
    def __init__(self, saga_root=None):
        self.saga_root = ensure_saga_on_path(saga_root)
        self.arguments = importlib.import_module("arguments")
        scene = importlib.import_module("scene")
        renderer = importlib.import_module("gaussian_renderer")
        loss_utils = importlib.import_module("utils.loss_utils")
        image_utils = importlib.import_module("utils.image_utils")
        lpips_module = importlib.import_module("lpipsPyTorch")

        self.GaussianModel = scene.GaussianModel
        self.FeatureGaussianModel = scene.FeatureGaussianModel
        self.Scene = scene.Scene
        self.render = renderer.render
        self.render_with_depth = renderer.render_with_depth
        self.l1_loss = loss_utils.l1_loss
        self.ssim = loss_utils.ssim
        self.psnr = image_utils.psnr
        self.lpips = lpips_module.lpips


def load_trained_scene(
    model_path,
    source_path=None,
    feature_iteration=10000,
    feature_dim=32,
    saga_root=None,
):
    model_path = Path(model_path).resolve()
    config_path = model_path / "cfg_args"
    if not config_path.exists():
        raise FileNotFoundError("Missing SegAnyGaussians config: %s" % config_path)
    dataset = eval(config_path.read_text(encoding="utf-8"), {"Namespace": Namespace})
    dataset.model_path = str(model_path)
    if source_path:
        dataset.source_path = str(Path(source_path).resolve())
    else:
        dataset.source_path = str(Path(dataset.source_path).resolve())
    dataset.need_features = False
    dataset.need_masks = False
    dataset.feature_dim = feature_dim
    if not hasattr(dataset, "allow_principle_point_shift"):
        dataset.allow_principle_point_shift = False
    if not hasattr(dataset, "init_from_3dgs_pcd"):
        dataset.init_from_3dgs_pcd = False

    backend = SagaBackend(saga_root)
    scene_gaussians = backend.GaussianModel(dataset.sh_degree)
    feature_gaussians = backend.FeatureGaussianModel(feature_dim)
    scene = backend.Scene(
        dataset,
        scene_gaussians,
        feature_gaussians,
        load_iteration=-1,
        feature_load_iteration=feature_iteration,
        shuffle=False,
        mode="eval",
        target="contrastive_feature",
    )

    parser = argparse.ArgumentParser(add_help=False)
    pipeline_params = backend.arguments.PipelineParams(parser)
    pipe = pipeline_params.extract(parser.parse_args([]))

    features = feature_gaussians.get_point_features.detach()
    scale_gate_path = model_path / "point_cloud" / ("iteration_%d" % feature_iteration) / "scale_gate.pt"
    if scale_gate_path.exists():
        scale_gate = torch.nn.Sequential(
            torch.nn.Linear(1, feature_dim),
            torch.nn.Sigmoid(),
        ).to(features.device)
        scale_gate.load_state_dict(torch.load(scale_gate_path, map_location=features.device, weights_only=True))
        features = features * scale_gate(torch.tensor([0.0], device=features.device)).unsqueeze(0)
    features = F.normalize(features, dim=-1, p=2)

    if scene_gaussians.get_xyz.shape[0] != features.shape[0]:
        raise RuntimeError(
            "Scene Gaussian count (%d) does not match feature count (%d)."
            % (scene_gaussians.get_xyz.shape[0], features.shape[0])
        )

    return {
        "backend": backend,
        "dataset": dataset,
        "scene": scene,
        "gaussians": scene_gaussians,
        "feature_gaussians": feature_gaussians,
        "features": features,
        "pipe": pipe,
        "scale_gate_path": scale_gate_path,
    }
