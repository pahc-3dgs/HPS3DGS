#!/usr/bin/env python3
"""In-process HAC++ driver, executed by :mod:`pahc.hacpp.bridge`.

Run as::

    python -m pahc.hacpp.driver --hacpp-root /path/to/HAC++ <command> ...

``--hacpp-root`` is inserted at ``sys.path[0]`` *before* any HAC++ import so
that ``import scene`` / ``import gaussian_renderer`` resolve to the HAC++
packages and never collide with the SegAnyGaussians packages used elsewhere in
PAHC (this is why the bridge is a subprocess and not an in-process import).

Commands
--------
inspect      report which HAC++ modules/extension deps are importable (no GPU work)
encode       run ``GaussianModel.conduct_encoding`` on a trained HAC++ model dir
decode       run ``GaussianModel.conduct_decoding`` and optional test render
render       render a (decoded) HAC++ model dir and report PSNR/SSIM

Every command prints a single JSON object on stdout; anything on stderr is a
human-readable error. Missing deps or checkpoints are reported honestly as
``{"ok": false, "error": ...}`` with a non-zero exit code.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _prepend_hacpp_root(root: str):
    root = os.path.abspath(root)
    # Remove any other copy of the top-level packages first (e.g. SegAnyGaussians).
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
    try:
        import shutil

        report["tmc3"] = shutil.which("tmc3")
    except Exception:  # noqa: BLE001
        report["tmc3"] = None
    missing = [name for name, info in report["modules"].items() if not info["ok"]]
    report["ready_for_encode"] = not missing
    report["missing"] = missing
    _json_print(report)
    return 0


def _load_model(hacpp_root, model_path, device, decoded=False):
    from argparse import Namespace

    import torch

    from scene.gaussian_model import GaussianModel

    model_path = os.path.abspath(model_path)
    cfg_path = os.path.join(model_path, "cfg_args")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(
            "%s does not contain cfg_args; this must be a *trained HAC++* model "
            "directory produced by HAC++ train.py, not a SAGA/3DGS checkpoint "
            "(those store a different parameterisation and cannot be restored)."
            % model_path
        )
    dataset = eval(open(cfg_path).read(), {"Namespace": Namespace})  # noqa: S307
    model_kwargs = dict(
        feat_dim=50,
        n_offsets=10,
        voxel_size=0.01,
        update_depth=3,
        update_init_factor=100,
        update_hierachy_factor=4,
        use_feat_bank=getattr(dataset, "use_feat_bank", False),
        n_features_per_level=2,
        log2_hashmap_size=getattr(dataset, "log2_hashmap_size", 19),
        log2_hashmap_size_2D=getattr(dataset, "log2_hashmap_size_2D", 17),
        resolutions_list=getattr(
            dataset,
            "resolutions_list",
            (18, 24, 33, 44, 59, 80, 108, 148, 201, 275, 376, 514),
        ),
        resolutions_list_2D=getattr(dataset, "resolutions_list_2D", (130, 258, 514, 1026)),
        ste_binary=getattr(dataset, "ste_binary", True),
        ste_multistep=getattr(dataset, "ste_multistep", False),
        add_noise=getattr(dataset, "add_noise", False),
        Q=1,
        use_2D=getattr(dataset, "use_2D", True),
        decoded_version=decoded,
        is_synthetic_nerf=getattr(dataset, "is_synthetic_nerf", False),
    )
    pc = GaussianModel(**model_kwargs)
    checkpoint = os.path.join(model_path, "chkpnt.pth")
    if not os.path.isfile(checkpoint):
        candidates = sorted(
            entry for entry in os.listdir(model_path) if entry.startswith("chkpnt")
        )
        if not candidates:
            raise FileNotFoundError(
                "no chkpnt*.pth in %s; HAC++ encode/decode needs a trained "
                "HAC++ student checkpoint" % model_path
            )
        checkpoint = os.path.join(model_path, candidates[-1])
    state = torch.load(checkpoint, map_location=device)
    pc.restore(state, Namespace(percent_dense=getattr(dataset, "percent_dense", 0.01)))
    if decoded:
        pc.decoded_version = True
    return pc, dataset


def cmd_encode(args):
    _prepend_hacpp_root(args.hacpp_root)
    import torch  # noqa: F401 - the model constructor assumes CUDA is set up

    device = args.device
    pc, dataset = _load_model(args.hacpp_root, args.model_path, device)
    os.makedirs(args.out_dir, exist_ok=True)
    log_info = pc.conduct_encoding(pre_path_name=args.out_dir)
    payload = {"ok": True, "out_dir": os.path.abspath(args.out_dir), "log": log_info}
    mlp_path = os.path.join(args.out_dir, "mlp.pt")
    pc.save_mlp_checkpoints(mlp_path)
    payload["shared_decoder"] = {"weights": os.path.relpath(mlp_path, args.out_dir)}
    _json_print(payload)
    return 0


def cmd_decode(args):
    _prepend_hacpp_root(args.hacpp_root)
    import math

    import torch

    device = args.device
    pc, dataset = _load_model(
        args.hacpp_root, args.model_path, device, decoded=args.decoded
    )
    if not os.path.isdir(args.bitstream_dir):
        raise FileNotFoundError("bitstream dir not found: %s" % args.bitstream_dir)
    log_info = pc.conduct_decoding(pre_path_name=args.bitstream_dir)
    payload = {"ok": True, "log": log_info}
    if args.render_out:
        from argparse import Namespace as _NS

        from scene import Scene
        from gaussian_renderer import render

        dataset.source_path = os.path.abspath(args.source_path) if args.source_path else dataset.source_path
        scene = Scene(dataset, pc, load_iteration=-1, shuffle=False, mode="eval")
        cameras = scene.getTestCameras() or scene.getTrainCameras()
        split = "test" if scene.getTestCameras() else "train"
        bg = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
        psnr_sum, count = 0.0, 0
        for cam in cameras:
            with torch.no_grad():
                image = render(cam, pc, _NS(debug=False), bg)["render"].clamp(0, 1)
            target = cam.original_image.to(device).clamp(0, 1)
            mse = torch.mean((image - target) ** 2).item()
            psnr_sum += -10.0 * math.log10(max(mse, 1e-12))
            count += 1
        payload["render"] = {
            "split": split,
            "views": count,
            "psnr": psnr_sum / max(count, 1),
        }
    _json_print(payload)
    return 0


def cmd_render(args):
    _prepend_hacpp_root(args.hacpp_root)
    import math

    import torch

    from argparse import Namespace as _NS

    from scene import Scene
    from gaussian_renderer import render

    device = args.device
    pc, dataset = _load_model(args.hacpp_root, args.model_path, device, decoded=args.decoded)
    dataset.source_path = os.path.abspath(args.source_path) if args.source_path else dataset.source_path
    scene = Scene(dataset, pc, load_iteration=-1, shuffle=False, mode="eval")
    test_cameras = scene.getTestCameras()
    cameras = test_cameras if test_cameras else scene.getTrainCameras()
    split = "test" if test_cameras else "train"
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    psnr_sum, count = 0.0, 0
    for cam in cameras:
        with torch.no_grad():
            image = render(cam, pc, _NS(debug=False), bg)["render"].clamp(0, 1)
        target = cam.original_image.to(device).clamp(0, 1)
        mse = torch.mean((image - target) ** 2).item()
        psnr_sum += -10.0 * math.log10(max(mse, 1e-12))
        count += 1
    _json_print({"ok": True, "split": split, "views": count, "psnr": psnr_sum / max(count, 1)})
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

    decode = sub.add_parser("decode")
    decode.add_argument("--model-path", required=True)
    decode.add_argument("--bitstream-dir", required=True)
    decode.add_argument("--decoded", action="store_true")
    decode.add_argument("--source-path", default=None)
    decode.add_argument("--render-out", action="store_true")

    render_cmd = sub.add_parser("render")
    render_cmd.add_argument("--model-path", required=True)
    render_cmd.add_argument("--decoded", action="store_true")
    render_cmd.add_argument("--source-path", default=None)

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
