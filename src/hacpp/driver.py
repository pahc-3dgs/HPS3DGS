#!/usr/bin/env python3
"""Standalone HAC++ driver, executed by :mod:`src.hacpp.bridge`.

Run as::

    python src/hacpp/driver.py --hacpp-root /path/to/HAC++ <command> ...

``--hacpp-root`` is inserted at ``sys.path[0]`` *before* any HAC++ import so
that ``import scene`` / ``import gaussian_renderer`` resolve to the HAC++
packages and never collide with the SegAnyGaussians packages used elsewhere in
HPS3DGS (this is why the bridge is a subprocess and not an in-process import).

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

Every command prints a single JSON object on stdout wrapped in a fixed
sentinel pair (see :data:`RESULT_BEGIN` / :data:`RESULT_END`). HAC++ itself
prints human-readable progress lines (and tmc3/GPCC writes its own banner) on
stdout *before* the JSON, so the sentinel is the only reliable frame; the
bridge skips everything outside the pair and keeps it as diagnostics. Missing
deps or checkpoints are reported honestly as ``{"ok": false, "error": ...}``
with a non-zero exit code - nothing is faked.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

#: stdout framing for the driver's final JSON payload. The JSON is written on
#: its own single line right after ``RESULT_BEGIN`` (json.dumps escapes inner
#: newlines), then ``RESULT_END`` closes the block. The bridge scans backwards
#: from the end of stdout so that HAC++ log lines that merely *mention* the
#: sentinel cannot fake a result.
RESULT_BEGIN = "@@HACPP_RESULT@@"
RESULT_END = "@@HACPP_RESULT_END@@"


def _prepend_hacpp_root(root: str):
    root = os.path.abspath(root)
    # Drop any other copy of the top-level packages (e.g. SegAnyGaussians).
    for name in ("scene", "gaussian_renderer", "arguments", "utils"):
        for key in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
            del sys.modules[key]
    sys.path.insert(0, root)
    return root


def _json_print(payload):
    """Emit the final JSON payload inside the sentinel frame (single lines)."""

    line = json.dumps(payload, sort_keys=True)
    sys.stdout.write("%s %s\n%s\n" % (RESULT_BEGIN, line, RESULT_END))
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
    """Build the HAC++ ModelParams/PipelineParams namespace trio.

    ``ModelParams`` hard-codes ``voxel_size=0.08``, but a model trained with
    ``--voxel_size 0.005`` records the true value in the ``cfg_args`` file that
    ``train.py`` writes next to the checkpoint (and ``get_combined_args`` reads
    it back).  Without that step the driver quantizes every anchor to an 8 cm
    grid, which collapses the compressed anchor stream and destroys the decoded
    representation.  Mirror ``get_combined_args`` here instead of re-parsing
    only ``-s/-m``.
    """

    from arguments import ModelParams, OptimizationParams, PipelineParams

    parser = argparse.ArgumentParser(add_help=False)
    model_params = ModelParams(parser)
    pipeline_params = PipelineParams(parser)
    optimization_params = OptimizationParams(parser)
    argv = ["-s", os.path.abspath(source_path or model_path), "-m", os.path.abspath(model_path)]
    args = parser.parse_args(argv)
    cfg_path = os.path.join(os.path.abspath(model_path), "cfg_args")
    try:
        with open(cfg_path) as cfg_file:
            cfg = eval(cfg_file.read(), {"Namespace": argparse.Namespace})  # noqa: S307 - trusted model dir
        for key, value in vars(cfg).items():
            if hasattr(args, key):
                setattr(args, key, value)
    except (OSError, NameError, SyntaxError, TypeError):
        # No/unsupported cfg_args: fall back to the defaults above.
        pass
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

DECODER_CONFIG_VERSION = 1


def make_decoder_config(dataset, model_path, source_path, shared_decoder_parameter_bytes=0):
    """Return every value needed to instantiate the entropy decoder.

    This deliberately stores values, not paths. Dataset images/cameras are
    required only for rendering and remain outside the codec bundle.
    """

    source = os.path.abspath(source_path or getattr(dataset, "source_path", model_path))
    config = {
        "config_version": DECODER_CONFIG_VERSION,
        "feat_dim": int(dataset.feat_dim),
        "n_offsets": int(dataset.n_offsets),
        "voxel_size": float(dataset.voxel_size),
        "update_depth": int(dataset.update_depth),
        "update_init_factor": int(dataset.update_init_factor),
        "update_hierachy_factor": int(dataset.update_hierachy_factor),
        "use_feat_bank": bool(dataset.use_feat_bank),
        "decoded_version": True,
        "is_synthetic_nerf": os.path.exists(os.path.join(source, "transforms_train.json")),
        "white_background": bool(getattr(dataset, "white_background", False)),
        "eval": bool(getattr(dataset, "eval", True)),
        "all_views_train_test": bool(getattr(dataset, "all_views_train_test", False)),
        "Q": 1,
        "dtype": "float32",
        "shared_decoder_parameter_bytes": int(shared_decoder_parameter_bytes),
    }
    for key, value in ENCODING_CONFIG.items():
        config[key] = list(value) if isinstance(value, tuple) else value
    return config


def build_gaussians_from_decoder_config(config, device):
    """Instantiate an empty HAC++ model using only portable bundle metadata."""

    from scene.gaussian_model import GaussianModel

    required = (
        "feat_dim", "n_offsets", "voxel_size", "update_depth",
        "update_init_factor", "update_hierachy_factor", "use_feat_bank",
        "n_features_per_level", "log2_hashmap_size", "log2_hashmap_size_2D",
        "resolutions_list", "resolutions_list_2D", "use_2D", "ste_binary",
        "ste_multistep", "add_noise", "decoded_version", "is_synthetic_nerf",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError("decoder_config missing required keys: %s" % ", ".join(missing))
    model = GaussianModel(
        int(config["feat_dim"]),
        int(config["n_offsets"]),
        float(config["voxel_size"]),
        int(config["update_depth"]),
        int(config["update_init_factor"]),
        int(config["update_hierachy_factor"]),
        bool(config["use_feat_bank"]),
        n_features_per_level=int(config["n_features_per_level"]),
        log2_hashmap_size=int(config["log2_hashmap_size"]),
        log2_hashmap_size_2D=int(config["log2_hashmap_size_2D"]),
        resolutions_list=tuple(config["resolutions_list"]),
        resolutions_list_2D=tuple(config["resolutions_list_2D"]),
        use_2D=bool(config["use_2D"]),
        ste_binary=bool(config["ste_binary"]),
        ste_multistep=bool(config["ste_multistep"]),
        add_noise=bool(config["add_noise"]),
        Q=config.get("Q", 1),
        decoded_version=bool(config["decoded_version"]),
        is_synthetic_nerf=bool(config["is_synthetic_nerf"]),
    )
    return model.to(device)


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
        # Match HAC++ train.py's render_sets/run_codec path: the checkpoint stores
        # _scaling/_anchor/_mask in their already-(de)compressed form, so
        # get_scaling/get_anchor/get_mask must NOT re-apply exp()/voxel rounding/
        # sigmoid STE. decoded_version=True is the reference rendering mode.
        decoded_version=True,
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
    # Match HAC++ train.py evaluation (render_sets calls gaussians.eval()).
    gaussians.eval()
    # The hash-grid normalization x_bound is a training-time constant: train.py
    # computes it with update_anchor_bound() at step 10000 and only persists it
    # into <model>/bitstreams/ when it encodes.  The anchors saved in the PLY can
    # densify/grow afterwards, so recomputing x_bound from them yields a slightly
    # different normalization that inflates the arithmetic-coded feat/scaling/
    # offsets streams (~1.7x).  Prefer the persisted training-time value.
    _restore_training_x_bound(gaussians, model_path)
    return scene, gaussians, dataset, pipe


def _restore_training_x_bound(gaussians, model_path):
    """Override the anchor-derived x_bound with train.py's persisted value.

    ``train.py`` writes ``<model_path>/bitstreams/{x_bound_min,x_bound_max}.pkl``
    during its run_codec evaluation.  Those tensors carry the normalization the
    hash grid was actually trained with.  Missing/unreadable files fall back to
    the freshly-computed bound.
    """

    import torch

    base = os.path.join(os.path.abspath(model_path), "bitstreams")
    for attr in ("x_bound_min", "x_bound_max"):
        path = os.path.join(base, attr + ".pkl")
        if not os.path.exists(path):
            continue
        try:
            value = torch.load(path, map_location=gaussians.get_anchor.device)
            if torch.is_tensor(value):
                setattr(gaussians, attr, value.to(gaussians.get_anchor.device))
        except Exception:  # noqa: BLE001 - fall back to the computed bound
            continue


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
    enforced by :mod:`src.hacpp.manifest`.
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


def shared_decoder_parameter_bytes(state):
    """Paper accounting: raw tensor payload, excluding torch.save headers."""

    return sum(value.numel() * value.element_size() for module in state.values() for value in module.values())


def load_shared_decoder(pc, path, device):
    """Restore the MLP-only state saved by :func:`save_shared_decoder`."""

    state = _torch_load(path, device)
    modules = {
        "opacity_mlp": pc.mlp_opacity,
        "cov_mlp": pc.mlp_cov,
        "color_mlp": pc.mlp_color,
        "grid_mlp": pc.mlp_grid,
        "deform_mlp": pc.mlp_deform,
    }
    if getattr(pc, "use_feat_bank", False):
        modules["mlp_feature_bank"] = pc.mlp_feature_bank
    missing = [key for key in modules if key not in state]
    extra = sorted(set(state) - set(modules))
    if missing or extra:
        raise ValueError("shared decoder keys mismatch: missing=%s extra=%s" % (missing, extra))
    for key, module in modules.items():
        module.load_state_dict(state[key])
    pc.eval()
    return sorted(state)


def shared_decoder_has_no_hash(path):
    """Structural check used by tests and by the bundle verifier."""

    state = _torch_load(path, "cpu")
    offending = [key for key in state if "encoding" in key or "hash" in key]
    return offending


#: Previously mis-diagnosed as a torch-2.x rasterizer incompatibility: rendering
#: zxa1-12_init reproduced a 25.17 GiB (gaussian, tile) binning allocation + OOM
#: on a 24 GB card.  The real root cause is NOT the torch version - the driver
#: constructed GaussianModel with decoded_version=False, so get_scaling applied
#: exp() to the checkpoint's already-linear _scaling (every Gaussian blew up to
#: ~1.0 scales).  Fixed by passing decoded_version=True (see build_gaussians);
#: render_full then matches train.py evaluation.  The torch-2.x path is retained
#: only as an un-re-verified warning.
TORCH2_RENDER_WARNING = (
    "torch %s: rendering this HAC++ checkpoint previously reproduced a 25.17 GiB "
    "binning allocation + OOM (zxa1-12_init). Root cause was decoded_version=False "
    "re-applying exp() to linear _scaling, not the torch version; decoded_version=True "
    "fixes render_full to match train.py. Torch 2.x has not been re-verified."
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


def _render_psnr(
    scene,
    gaussians,
    pipe,
    device,
    split="test",
    max_cameras=None,
    allow_torch2=False,
    white_background=False,
):
    """Render the requested split (test preferred, explicit train fallback)."""

    import torch

    from gaussian_renderer import prefilter_voxel, render
    from utils.image_utils import psnr
    from utils.loss_utils import ssim
    import lpips

    if split == "test":
        cameras = scene.getTestCameras()
        if not cameras:
            return {"split": "train_fallback", "views": 0, "psnr": None, "reason": "no test cameras"}
    else:
        cameras = scene.getTrainCameras()
    total = len(cameras)
    if max_cameras:
        cameras = cameras[: int(max_cameras)]
    background = torch.tensor(
        [1, 1, 1] if white_background else [0, 0, 0],
        dtype=torch.float32,
        device=device,
    )
    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
    psnr_sum, ssim_sum, lpips_sum, count = 0.0, 0.0, 0.0, 0
    render_times = []
    for cam in cameras:
        with torch.no_grad():
            # Match HAC++ train.py evaluation: select visible anchors before
            # expanding their offsets into rasterized Gaussians.
            visible_mask = prefilter_voxel(cam, gaussians, pipe, background)
            torch.cuda.synchronize()
            render_start = time.perf_counter()
            image = render(cam, gaussians, pipe, background, visible_mask=visible_mask)["render"].clamp(0, 1)
            torch.cuda.synchronize()
            render_times.append(time.perf_counter() - render_start)
        target = cam.original_image.to(device).clamp(0, 1)
        # HAC++'s official metric path saves PNGs and reads them back before
        # evaluation. Reproduce torchvision.save_image's uint8 conversion in
        # memory so these values are directly comparable without writing 300
        # temporary images for the full/decoded pair.
        image_metric = torch.floor(image * 255.0 + 0.5).clamp(0, 255) / 255.0
        target_metric = torch.floor(target * 255.0 + 0.5).clamp(0, 255) / 255.0
        image_batch = image_metric.unsqueeze(0)
        target_batch = target_metric.unsqueeze(0)
        psnr_sum += psnr(image_batch, target_batch).mean().item()
        ssim_sum += ssim(image_batch, target_batch).mean().item()
        with torch.no_grad():
            lpips_sum += lpips_fn(image_batch, target_batch, normalize=False).mean().item()
        count += 1
    timed = render_times[5:] if len(render_times) > 5 else render_times
    mean_render_seconds = sum(timed) / max(len(timed), 1)
    return {
        "split": split,
        "views": count,
        "views_total_in_split": total,
        "truncated": bool(max_cameras and count < total),
        "psnr": psnr_sum / max(count, 1),
        "ssim": ssim_sum / max(count, 1),
        "lpips": lpips_sum / max(count, 1),
        "render_fps": 1.0 / mean_render_seconds if mean_render_seconds else None,
        "fps_warmup_views_excluded": min(5, len(render_times)),
        "metric_domain": "png_uint8_equivalent",
    }


def _gpcc_mode():
    import shutil

    return "gpcc" if shutil.which("tmc3") else "numpy_fallback"


def cmd_encode(args):
    _prepend_hacpp_root(args.hacpp_root)
    scene, gaussians, dataset, _pipe = load_hacpp_scene(
        args.hacpp_root, args.model_path, args.source_path, args.device, load_iteration=-1
    )
    os.makedirs(args.out_dir, exist_ok=True)
    log_info = gaussians.conduct_encoding(pre_path_name=args.out_dir)
    decoder_path = os.path.join(args.out_dir, "shared_mlp.pt")
    state = save_shared_decoder(gaussians, decoder_path)
    config = make_decoder_config(
        dataset,
        args.model_path,
        args.source_path,
        shared_decoder_parameter_bytes(state),
    )
    config_path = os.path.join(args.out_dir, "decoder_config.json")
    with open(config_path, "w") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    payload = {
        "ok": True,
        "out_dir": os.path.abspath(args.out_dir),
        "log": log_info,
        "gpcc_mode": _gpcc_mode(),
        "shared_decoder": {
            "weights": "shared_mlp.pt",
            "keys": sorted(state.keys()),
            "excludes_hash": True,
            "parameter_bytes": config["shared_decoder_parameter_bytes"],
        },
        "decoder_config": "decoder_config.json",
    }
    _json_print(payload)
    return 0


def cmd_decode(args):
    _prepend_hacpp_root(args.hacpp_root)
    if not os.path.isdir(args.bitstream_dir):
        raise FileNotFoundError("bitstream dir not found: %s" % args.bitstream_dir)
    config_path = os.path.join(args.bitstream_dir, "decoder_config.json")
    decoder_path = os.path.join(args.bitstream_dir, "shared_mlp.pt")
    if not os.path.isfile(config_path):
        raise FileNotFoundError("portable decoder config not found: %s" % config_path)
    if not os.path.isfile(decoder_path):
        raise FileNotFoundError("portable shared decoder not found: %s" % decoder_path)
    with open(config_path) as handle:
        decoder_config = json.load(handle)

    scene = None
    pipe = None
    dataset = None
    if args.model_path is not None:
        scene, gaussians, dataset, pipe = load_hacpp_scene(
            args.hacpp_root, args.model_path, args.source_path, args.device, load_iteration=-1
        )
    else:
        if args.render:
            raise ValueError("--render requires --model-path for cameras and ground-truth images")
        gaussians = build_gaussians_from_decoder_config(decoder_config, args.device)

    payload = {"ok": True, "self_contained_decode": True}
    if args.render:
        payload["render_full"] = _render_psnr(
            scene, gaussians, pipe, args.device, "test", args.max_cameras,
            args.allow_torch2_render, bool(dataset.white_background)
        )
    payload["shared_decoder_keys"] = load_shared_decoder(gaussians, decoder_path, args.device)
    log_info = gaussians.conduct_decoding(pre_path_name=args.bitstream_dir)
    payload["log"] = log_info
    payload["decoded"] = {
        "anchors": int(gaussians.get_anchor.shape[0]),
        "feat_dim": int(gaussians.feat_dim),
        "n_offsets": int(gaussians.n_offsets),
        "voxel_size": float(gaussians.voxel_size),
    }
    if args.render:
        payload["render_decoded"] = _render_psnr(
            scene, gaussians, pipe, args.device, "test", args.max_cameras,
            args.allow_torch2_render, bool(dataset.white_background)
        )
    _json_print(payload)
    return 0


def cmd_render(args):
    _prepend_hacpp_root(args.hacpp_root)
    scene, gaussians, dataset, pipe = load_hacpp_scene(
        args.hacpp_root, args.model_path, args.source_path, args.device, load_iteration=-1
    )
    payload = {
        "ok": True,
        "full": _render_psnr(
            scene, gaussians, pipe, args.device, "test", args.max_cameras,
            args.allow_torch2_render, bool(dataset.white_background)
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
    decode.add_argument("--model-path", default=None)
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
