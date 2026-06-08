#!/usr/bin/env python3
"""Run the complete SegAnyGaussians and PAHC-3DGS pipeline."""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_command(command, description, cwd, env, print_only=False):
    print("\n" + "=" * 72)
    print("[%s]" % description)
    print("cwd: %s" % cwd)
    print(shlex.join([str(item) for item in command]))
    print("=" * 72, flush=True)
    if print_only:
        return
    subprocess.run(
        [str(item) for item in command],
        cwd=str(cwd),
        env=env,
        check=True,
    )


def require_saga(saga_root, print_only=False):
    required = [
        "train_scene.py",
        "extract_segment_everything_masks.py",
        "get_scale.py",
        "get_clip_features.py",
        "train_contrastive_feature.py",
    ]
    missing = [name for name in required if not (saga_root / name).exists()]
    if missing and not print_only:
        raise FileNotFoundError(
            "SegAnyGaussians is incomplete at %s. Missing: %s"
            % (saga_root, ", ".join(missing))
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--source", required=True, type=Path)
    parser.add_argument("-m", "--model-path", required=True, type=Path)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument(
        "--saga-root",
        type=Path,
        default=REPO_ROOT / "third_party" / "SegAnyGAussians",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "pahc.yaml",
    )
    parser.add_argument("--scene-iterations", type=int, default=30000)
    parser.add_argument("--feature-iterations", "--iterations", type=int, default=10000)
    parser.add_argument("--num-sampled-rays", type=int, default=2000)
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--downsample-type", choices=["image", "mask"], default="image")
    parser.add_argument("--sam-checkpoint-path", type=Path, default=None)
    parser.add_argument("--sam-arch", default="vit_h")
    parser.add_argument("--skip-train", "--skip_train", action="store_true")
    parser.add_argument("--skip-mask", "--skip_mask", action="store_true")
    parser.add_argument("--skip-feature", "--skip_feature", action="store_true")
    parser.add_argument("--skip-compression", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--recluster", action="store_true")
    parser.add_argument("--labels-path", type=Path, default=None)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    model_path = args.model_path.resolve()
    saga_root = args.saga_root.resolve()
    config = args.config.resolve()
    output = args.output.resolve() if args.output else REPO_ROOT / "outputs" / ("%s.pahc.pt" % model_path.name)

    if not source.exists() and not args.print_only:
        parser.error("Source scene does not exist: %s" % source)
    if not config.exists() and not args.print_only:
        parser.error("PAHC config does not exist: %s" % config)
    require_saga(saga_root, args.print_only)

    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT)
    if current_pythonpath:
        env["PYTHONPATH"] += os.pathsep + current_pythonpath

    if not args.skip_train:
        run_command(
            [
                sys.executable,
                saga_root / "train_scene.py",
                "-s",
                source,
                "-m",
                model_path,
                "--iterations",
                args.scene_iterations,
                "--checkpoint_iterations",
                args.scene_iterations,
            ],
            "Step 1/4: Train basic 3D Gaussians",
            saga_root,
            env,
            args.print_only,
        )
    else:
        print("[Skip] Step 1/4: basic 3D Gaussian training")

    if not args.skip_mask:
        mask_command = [
            sys.executable,
            saga_root / "extract_segment_everything_masks.py",
            "--image_root",
            source,
            "--downsample",
            args.downsample,
            "--downsample_type",
            args.downsample_type,
            "--sam_arch",
            args.sam_arch,
        ]
        if args.sam_checkpoint_path:
            mask_command.extend(["--sam_checkpoint_path", args.sam_checkpoint_path.resolve()])
        run_command(
            mask_command,
            "Step 2a/4: Extract SAM masks",
            saga_root,
            env,
            args.print_only,
        )
        run_command(
            [
                sys.executable,
                saga_root / "get_scale.py",
                "--image_root",
                source,
                "--model_path",
                model_path,
            ],
            "Step 2b/4: Compute mask scales",
            saga_root,
            env,
            args.print_only,
        )
        run_command(
            [
                sys.executable,
                saga_root / "get_clip_features.py",
                "--image_root",
                source,
            ],
            "Step 2c/4: Extract CLIP features",
            saga_root,
            env,
            args.print_only,
        )
    else:
        print("[Skip] Step 2/4: SAM masks, scales, and CLIP features")

    if not args.skip_feature:
        run_command(
            [
                sys.executable,
                saga_root / "train_contrastive_feature.py",
                "-m",
                model_path,
                "--iterations",
                args.feature_iterations,
                "--num_sampled_rays",
                args.num_sampled_rays,
                "--target",
                "seg",
            ],
            "Step 3/4: Train contrastive feature Gaussians",
            saga_root,
            env,
            args.print_only,
        )
    else:
        print("[Skip] Step 3/4: contrastive feature training")

    if not args.skip_compression:
        compression_command = [
            sys.executable,
            REPO_ROOT / "scripts" / "compress.py",
            "--model-path",
            model_path,
            "--source-path",
            source,
            "--saga-root",
            saga_root,
            "--feature-iteration",
            args.feature_iterations,
            "--config",
            config,
            "--output",
            output,
        ]
        if args.skip_evaluation:
            compression_command.append("--skip-evaluation")
        if args.recluster:
            compression_command.append("--recluster")
        if args.labels_path:
            compression_command.extend(["--labels-path", args.labels_path.resolve()])
        run_command(
            compression_command,
            "Step 4/4: Run PAHC-3DGS compression",
            REPO_ROOT,
            env,
            args.print_only,
        )
    else:
        print("[Skip] Step 4/4: PAHC-3DGS compression")

    print("\nPipeline complete.")
    print("SAGA model: %s" % model_path)
    if not args.skip_compression:
        print("PAHC scene: %s" % output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
