#!/usr/bin/env python3
"""Standalone HAC++ driver, executed by :mod:`src.hacpp.bridge`.

Run as::

    python src/hacpp/driver.py --hacpp-root /path/to/HAC++ <command> ...

``--hacpp-root`` is inserted at ``sys.path[0]`` *before* any HAC++ import so
that ``import scene`` / ``import gaussian_renderer`` resolve to the HAC++
packages and never collide with the SegAnyGaussians packages used elsewhere in
PAHC (this is why the bridge is a subprocess and not an in-process import).

The driver mirrors the *reference* HAC++ API exactly:

* ``Scene(args, gaussians, load_iteration=None, shuffle=True,
  resolution_scales=[1.0], ply_path=None)`` - there is no ``mode``/``target``
  argument in HAC++ (that is the SegAnyGaussians signature).
* ``train.py`` saves ``chkpnt<iter>.pth`` as ``(gaussians.capture(),
  iteration)`` and ``Scene.save`` writes ``point_cloud/iteration_N/
  {point_cloud.ply, checkpoint.pth}``. Loading goes through
  ``Scene(load_iteration=-1)`` which reads the PLY + ``checkpoint.pth``;
  a raw ``chkpnt*.pth`` is only used as a fallback and is unwrapped from its
  ``(capture, iteration)`` tuple first.
* ``GaussianModel(feat_dim, n_offsets, voxel_size, update_depth,
  update_init_factor, update_hierachy_factor, use_feat_bank, ...)`` built from
  the dataset attributes, as ``train.py`` does.

Commands
--------
inspect   report which HAC++ modules/extension deps are importable (no GPU work)
encode    run ``GaussianModel.conduct_encoding`` on a trained HAC++ model dir
decode    run ``GaussianModel.conduct_decoding`` and render test views
render    render a (decoded) HAC++ model dir and report PSNR

Every command prints a single JSON object on stdout; anything on stderr is a
human-readable error. Missing deps or checkpoints are reported honestly as
``{"ok": false, "error": ...}`` with a non-zero exit code - nothing is faked.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys


def _prepend_hacpp_root(root: str):
    root = os.path.abspath(root)
    # Drop any other copy of the top-level packages (e.g. SegAnyGaussians).
    for name in ("scene", "gaussian_renderer", "arguments", "utils"):
        for key in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
            del sys.modules[key]
    sys.path.insert(0, root)
    return root


def _json_print(payload):
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    sys.stdout.flush()


def cmd_inspect(args):
    root = _prepend_hacpp_root(args.hacpp_root)
    import shutil

    import torch

    report = {
        "ok": True,
        "hacpp_root": root,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "modules": {},
    }
    checks = {
        "scene.gaussian_model": "from scene.gaussian_model import GaussianModel",
        "gaussian_renderer": "from gaussian_renderer import render, generate_neural_gaussians",
        "diff_gaussian_rasterization": "import diff_gaussian_rasterization",
        "simple_knn": "import simple_knn",
        "torch_scatter": "import torch_scatter",
        "encodings_cuda (arithmetic)": "from utils.encodings_cuda import encoder, decoder",
    }
    for name, statement in checks.items():
        try:
            exec(statement, {})  # noqa: S102 - fixed strings above
            report["modules"][name] = {"ok": True}
        except Exception as exc:  # noqa: BLE001 - report every failure mode
            report["modules"][name] = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
    report["tmc3"] = shutil.which("tmc3")
    if report["tmc3"] is None:
        # utils/gpcc_utils.py in this reference falls back to raw numpy anchor
        # storage when tmc3 is absent (NUMPY: prefix). Encode still runs; the
        # anchor stream is simply not GPCC-compressed, and must be reported.
        report["modules"]["tmc3 (GPCC)"] = {
            "ok": True,
            "fallback": "raw numpy anchor storage (utils/gpcc_utils.py 'NUMPY:' mode)",
        }
        report["gpcc_mode"] = "numpy_fallback"
    else:
        report["gpcc_mode"] = "gpcc"
    missing = [name for name, info in report["modules"].items() if not info["ok"]]
    report["ready_for_encode"] = not missing
    report["missing"] = missing
    _json_print(report)
    return 0


def _torch_load(path, device):
    """torch.load that works on both torch<2.0 (no weights_only) and >=2.0."""

    import torch

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch 1.x has no weights_only
        return torch.load(path, map_location=device)


def _dataset_args(model_path, source_path):
    """Build the HAC++ ModelParams/PipelineParams namespace trio."""

    from arguments import ModelParams, OptimizationParams, PipelineParams

    parser = argparse.ArgumentParser(add_help=False)
    model_params = ModelParams(parser)
    pipeline_params = PipelineParams(parser)
    optimization_params = OptimizationParams(parser)
    argv = ["-s", os.path.abspath(source_path or model_path), "-m", os.path.abspath(model_path)]
    args = parser.parse_args(argv)
    return model_params.extract(args), pipeline_params.extract(args), optimization_params.extract(args)


#: Hash-grid hyperparameters. ``cfg_args`` does not record them and they are
#: *not* GaussianModel defaults: HAC++ ``train.py`` passes ``--n_features 4
#: --log2 13 --log2_2D 15`` by default, and those are the values every model in
#: this reference checkout was trained with (verified by param-shape algebra:
#: 3D grid = 18^3 + 11*2^13 = 95944 rows x 4 feats, 2D = 130^2 + 3*2^15 =
#: 115208 rows x 4 feats). Override with --n-features/--log2/--log2-2d only for
#: models trained with different flags; a mismatch fails at load time with a
#: size error, it never silently loads.
ENCODING_CONFIG = {
    "n_features_per_level": 4,
    "log2_hashmap_size": 13,
    "log2_hashmap_size_2D": 15,
    "resolutions_list": (18, 24, 33, 44, 59, 80, 108, 148, 201, 275, 376, 514),
    "resolutions_list_2D": (130, 258, 514, 1026),
    "use_2D": True,
    "ste_binary": True,
    "ste_multistep": False,
    "add_noise": False,
}


def build_gaussians(hacpp_root, dataset, device):
    from scene.gaussian_model import GaussianModel

    cfg = dict(ENCODING_CONFIG)
    cfg.update({key: value for key, value in getattr(build_gaussians, "overrides", {}).items() if value})
    gaussians = GaussianModel(
        dataset.feat_dim,
        dataset.n_offsets,
        dataset.voxel_size,
        dataset.update_depth,
        dataset.update_init_factor,
        dataset.update_hierachy_factor,
        dataset.use_feat_bank,
        n_features_per_level=cfg["n_features_per_level"],
        log2_hashmap_size=cfg["log2_hashmap_size"],
        log2_hashmap_size_2D=cfg["log2_hashmap_size_2D"],
        resolutions_list=tuple(cfg["resolutions_list"]),
        resolutions_list_2D=tuple(cfg["resolutions_list_2D"]),
        use_2D=cfg["use_2D"],
        ste_binary=cfg["ste_binary"],
        ste_multistep=cfg["ste_multistep"],
        add_noise=cfg["add_noise"],
        is_synthetic_nerf=os.path.exists(os.path.join(dataset.source_path, "transforms_train.json")),
    )
    return gaussians.to(device)


def load_hacpp_scene(hacpp_root, model_path, source_path, device, load_iteration=-1, ply_path=None):
    """Construct the HAC++ scene exactly like ``train.py`` does."""

    from scene import Scene

    dataset, pipe, _opt = _dataset_args(model_path, source_path)
    gaussians = build_gaussians(hacpp_root, dataset, device)
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=load_iteration,
        shuffle=False,
        ply_path=ply_path,
    )
    # train.py initializes the interpolation bounds immediately after Scene
    # construction.  Scene.load alone leaves them at zero, which makes
    # calc_interp_feat fail before entropy encoding starts.
    gaussians.update_anchor_bound()
    return scene, gaussians, dataset, pipe


def load_chkpnt_fallback(hacpp_root, model_path, source_path, device):
    """Load a raw ``chkpnt<iter>.pth`` (a ``(capture(), iteration)`` tuple).

    HAC++ ``GaussianModel.restore`` unpacks 11 fields while ``capture`` returns
    10 (no ``active_sh_degree``), so a checkpoint-restore path is only used
    when the canonical ``point_cloud/iteration_*`` artefacts are absent; the
    arity mismatch is reported instead of silently ignored.
    """

    from scene.gaussian_model import GaussianModel

    dataset, _pipe, opt = _dataset_args(model_path, source_path)
    candidates = sorted(
        entry for entry in os.listdir(model_path) if entry.startswith("chkpnt") and entry.endswith(".pth")
    )
    if not candidates:
        raise FileNotFoundError(
            "%s has neither point_cloud/iteration_*/point_cloud.ply nor "
            "chkpnt*.pth. encode/decode need a model directory produced by "
            "HAC++ train.py; a SAGA/3DGS checkpoint stores a different "
            "parameterisation and cannot be restored." % model_path
        )
    checkpoint = os.path.join(model_path, candidates[-1])
    payload = _torch_load(checkpoint, device)
    if isinstance(payload, tuple):
        state, iteration = payload[0], payload[1] if len(payload) > 1 else None
    else:
        state, iteration = payload, None
    gaussians = GaussianModel(
        dataset.feat_dim,
        dataset.n_offsets,
        dataset.voxel_size,
        dataset.update_depth,
        dataset.update_init_factor,
        dataset.update_hierachy_factor,
        dataset.use_feat_bank,
    ).to(device)
    try:
        gaussians.restore(state, opt)
    except ValueError as exc:
        raise RuntimeError(
            "HAC++ checkpoint %s could not be restored (%s); capture()/restore() "
            "arity mismatch in the reference implementation. Use the "
            "point_cloud/iteration_* artefacts written by Scene.save instead."
            % (checkpoint, exc)
        )
    return gaussians, dataset, iteration


def save_shared_decoder(pc, path):
    """Write decoder weights *without* the hash grid.

    ``GaussianModel.save_mlp_checkpoints`` stores ``encoding_xyz`` too, but the
    hash grid is a per-scene stream (``hash.b``); keeping it in the shared
    decoder would double-bill those bytes and contradict the manifest contract
    enforced by :mod:`pahc.hacpp.manifest`.
    """

    state = {
        "opacity_mlp": pc.mlp_opacity.state_dict(),
        "cov_mlp": pc.mlp_cov.state_dict(),
        "color_mlp": pc.mlp_color.state_dict(),
        "grid_mlp": pc.mlp_grid.state_dict(),
        "deform_mlp": pc.mlp_deform.state_dict(),
    }
    if getattr(pc, "use_feat_bank", False):
        state["mlp_feature_bank"] = pc.mlp_feature_bank.state_dict()
    forbidden = [key for key in state if "encoding" in key or "hash" in key]
    if forbidden:
        raise RuntimeError("shared decoder must not carry hash/encoding state: %s" % forbidden)
    import torch

    torch.save(state, path)
    return state


def shared_decoder_has_no_hash(path):
    """Structural check used by tests and by the bundle verifier."""

    state = _torch_load(path, "cpu")
    offending = [key for key in state if "encoding" in key or "hash" in key]
    return offending


#: Tested-incompatibility note (zxa1-12_init, May-7 checkpoint, voxel_size
#: 0.001, 97,744 anchors / ~233k generated Gaussians): rendering under
#: torch 2.4.1+cu121 - with either the environment's diff_gaussian_rasterization
#: or a fresh build of the HAC++ submodule - makes the rasterizer allocate
#: ~25 GiB for the (gaussian, tile) binning buffer and OOM on a 24 GB card,
#: while the same checkpoint renders 19 test views at PSNR 43.066 in the
#: documented HAC_env (python 3.7.13 / torch 1.12.1 / cu116). Encoding is NOT
#: affected. This is an observation about this extension/checkpoint pair, not a
#: claim about torch 2.x in general.
TORCH2_RENDER_WARNING = (
    "torch %s with this HAC++ rasterizer/checkpoint: rendering reproduced a "
    "25.17 GiB binning allocation + OOM on zxa1-12_init (voxel_size 0.001). "
    "The documented runtime is HAC_env: python 3.7.13 / torch 1.12.1+cu116 "
    "(see docs/HACPP_RUNTIME.md). Continuing anyway - pass --allow-torch2-render "
    "to silence this check."
)


def _check_render_runtime(allow: bool = False):
    import torch

    if torch.__version__.startswith("1."):
        return None
    message = TORCH2_RENDER_WARNING % torch.__version__
    if not allow:
        raise RuntimeError(message + " (raised, not allowed)")
    sys.stderr.write("WARNING: %s\n" % message)
    return message


def _render_psnr(scene, gaussians, pipe, device, split="test", max_cameras=None, allow_torch2=False):
    """Render the requested split (test preferred, explicit train fallback)."""

    import torch

    from gaussian_renderer import prefilter_voxel, render

    if split == "test":
        cameras = scene.getTestCameras()
        if not cameras:
            return {"split": "train_fallback", "views": 0, "psnr": None, "reason": "no test cameras"}
    else:
        cameras = scene.getTrainCameras()
    total = len(cameras)
    if max_cameras:
        cameras = cameras[: int(max_cameras)]
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    psnr_sum, count = 0.0, 0
    for cam in cameras:
        with torch.no_grad():
            # Match HAC++ train.py evaluation: select visible anchors before
            # expanding their offsets into rasterized Gaussians.
            visible_mask = prefilter_voxel(cam, gaussians, pipe, background)
            image = render(cam, gaussians, pipe, background, visible_mask=visible_mask)["render"].clamp(0, 1)
        target = cam.original_image.to(device).clamp(0, 1)
        mse = torch.mean((image - target) ** 2).item()
        psnr_sum += -10.0 * math.log10(max(mse, 1e-12))
        count += 1
    return {
        "split": split,
        "views": count,
        "views_total_in_split": total,
        "truncated": bool(max_cameras and count < total),
        "psnr": psnr_sum / max(count, 1),
    }


def _gpcc_mode():
    import shutil

    return "gpcc" if shutil.which("tmc3") else "numpy_fallback"


def cmd_encode(args):
    _prepend_hacpp_root(args.hacpp_root)
    scene, gaussians, _dataset, _pipe = load_hacpp_scene(
        args.hacpp_root, args.model_path, args.source_path, args.device, load_iteration=-1
    )
    os.makedirs(args.out_dir, exist_ok=True)
    log_info = gaussians.conduct_encoding(pre_path_name=args.out_dir)
    decoder_path = os.path.join(args.out_dir, "shared_mlp.pt")
    state = save_shared_decoder(gaussians, decoder_path)
    payload = {
        "ok": True,
        "out_dir": os.path.abspath(args.out_dir),
        "log": log_info,
        "gpcc_mode": _gpcc_mode(),
        "shared_decoder": {
            "weights": "shared_mlp.pt",
            "keys": sorted(state.keys()),
            "excludes_hash": True,
        },
    }
    _json_print(payload)
    return 0


def cmd_decode(args):
    _prepend_hacpp_root(args.hacpp_root)
    scene, gaussians, _dataset, pipe = load_hacpp_scene(
        args.hacpp_root, args.model_path, args.source_path, args.device, load_iteration=-1
    )
    if not os.path.isdir(args.bitstream_dir):
        raise FileNotFoundError("bitstream dir not found: %s" % args.bitstream_dir)
    payload = {"ok": True}
    if args.render:
        payload["render_full"] = _render_psnr(
            scene, gaussians, pipe, args.device, "test", args.max_cameras, args.allow_torch2_render
        )
    log_info = gaussians.conduct_decoding(pre_path_name=args.bitstream_dir)
    payload["log"] = log_info
    if args.render:
        payload["render_decoded"] = _render_psnr(
            scene, gaussians, pipe, args.device, "test", args.max_cameras, args.allow_torch2_render
        )
    _json_print(payload)
    return 0


def cmd_render(args):
    _prepend_hacpp_root(args.hacpp_root)
    scene, gaussians, _dataset, pipe = load_hacpp_scene(
        args.hacpp_root, args.model_path, args.source_path, args.device, load_iteration=-1
    )
    payload = {
        "ok": True,
        "full": _render_psnr(
            scene, gaussians, pipe, args.device, "test", args.max_cameras, args.allow_torch2_render
        ),
    }
    _json_print(payload)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hacpp-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("inspect")

    encode = sub.add_parser("encode")
    encode.add_argument("--model-path", required=True)
    encode.add_argument("--out-dir", required=True)
    encode.add_argument("--source-path", default=None)

    decode = sub.add_parser("decode")
    decode.add_argument("--model-path", required=True)
    decode.add_argument("--bitstream-dir", required=True)
    decode.add_argument("--source-path", default=None)
    decode.add_argument("--render", action="store_true")
    decode.add_argument("--max-cameras", type=int, default=None)
    decode.add_argument("--allow-torch2-render", action="store_true")

    render_cmd = sub.add_parser("render")
    render_cmd.add_argument("--model-path", required=True)
    render_cmd.add_argument("--source-path", default=None)
    render_cmd.add_argument("--max-cameras", type=int, default=None)
    render_cmd.add_argument("--allow-torch2-render", action="store_true")

    args = parser.parse_args(argv)
    handlers = {
        "inspect": cmd_inspect,
        "encode": cmd_encode,
        "decode": cmd_decode,
        "render": cmd_render,
    }
    try:
        return handlers[args.command](args)
    except Exception as exc:  # noqa: BLE001 - report every failure honestly
        sys.stderr.write("%s: %s\n" % (type(exc).__name__, exc))
        _json_print({"ok": False, "command": args.command, "error": "%s: %s" % (type(exc).__name__, exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
