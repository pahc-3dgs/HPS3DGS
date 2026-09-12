#!/usr/bin/env python3
"""Launch an existing HPS3DGS route in its own Python process.

Examples (launcher options precede the route; arguments after it are untouched):
  python scripts/hps3dgs.py --runtime configs/runtime.4090.json --dry-run geo33 -- --mode reference
  python scripts/hps3dgs.py --runtime configs/runtime.4090.json hac -- pack --raw RAW --bundle NEW_BUNDLE --scene SCENE
  python scripts/hps3dgs.py --runtime configs/runtime.4090.json hacpp-train -- -s DATASET -m NEW_MODEL

Relative runtime-config filenames and repository paths resolve from the release
root. Relative native CLI arguments retain the selected upstream working-directory
semantics; use absolute paths for datasets, checkpoints, and output directories.
The launcher imports no algorithm or CUDA module and never invokes a shell.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROUTES = {
    "hps-3dgs": ("saga", "scripts/run_hps_3dgs_pipeline.py", "."),
    "geo33": ("saga", "third_party/SegAnyGAussians/geo33.py", "third_party/SegAnyGAussians"),
    "fig7": ("saga", "third_party/SegAnyGAussians/fig7_taur.py", "third_party/SegAnyGAussians"),
    "hac": ("codec", "scripts/hac_backend.py", "."),
    "hacpp": ("codec", "scripts/hacpp_backend.py", "."),
    "hac-train": ("hac_train", "third_party/HAC/train.py", "third_party/HAC"),
    "hacpp-train": ("hacpp_train", "third_party/HAC-plus/train.py", "third_party/HAC-plus"),
}


def rooted(value, root):
    """Make paths absolute without dereferencing Conda interpreter symlinks."""
    if not isinstance(value, str) or not value:
        raise ValueError("Runtime paths must be nonempty strings")
    path = Path(value).expanduser()
    # A Conda Python may link to another environment's binary. Executing the
    # resolved target would select that other prefix and its incompatible libs.
    return Path(os.path.abspath(str(path if path.is_absolute() else root / path)))


def string_list(value, label):
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError("%s must be a list of nonempty strings" % label)
    return value


def build_plan(root, runtime_path, route, native_args):
    config = json.loads(runtime_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Expected runtime schema_version 1")
    role, script_name, cwd_name = ROUTES[route]
    profiles = config.get("runtimes", {})
    if role not in profiles:
        raise ValueError("Runtime configuration lacks profile %r" % role)
    profile = profiles[role]
    python = rooted(profile.get("python"), root)
    script = root / script_name
    cwd = root / cwd_name
    extra_pythonpath = [rooted(v, root) for v in string_list(profile.get("pythonpath", []), "pythonpath")]
    path_prepend = [python.parent] + [rooted(v, root) for v in string_list(profile.get("path_prepend", []), "path_prepend")]
    repo_pythonpath = [cwd.resolve()]
    if route == "hac":
        repo_pythonpath.append(root / "third_party/HAC")
    elif route == "hacpp":
        repo_pythonpath.append(root / "third_party/HAC-plus")
    pythonpath = list(dict.fromkeys(str(p) for p in repo_pythonpath + extra_pythonpath))
    env_values = {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}
    for source in (config.get("env", {}), profile.get("env", {})):
        if not isinstance(source, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in source.items()):
            raise ValueError("Runtime env must map strings to strings")
        if any(k in source for k in ("PYTHONPATH", "PYTHONHOME", "PATH")):
            raise ValueError("Use pythonpath/path_prepend fields instead of PATH, PYTHONPATH or PYTHONHOME in env")
        env_values.update(source)
    env_values["PYTHONPATH"] = os.pathsep.join(pythonpath)
    injected_args = []
    if route == "hps-3dgs":
        saga = str(root / "third_party/SegAnyGAussians")
        injected_args = ["--saga-root", saga]
        env_values["HPS_3DGS_SAGA_ROOT"] = saga
    elif route == "hac":
        injected_args = ["--hac-root", str(root / "third_party/HAC")]
    elif route == "hacpp":
        injected_args = ["--hacpp-root", str(root / "third_party/HAC-plus"), "--python", str(python)]
    forwarded = list(native_args)
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    command = [str(python), "-u", "-B", str(script)] + injected_args + forwarded
    required = [("Python executable", python, "file"), ("entry script", script, "file"), ("working directory", cwd, "dir")]
    required += [("PYTHONPATH directory", p, "dir") for p in repo_pythonpath + extra_pythonpath]
    required += [("PATH directory", p, "dir") for p in path_prepend]
    for value in string_list(profile.get("required_files", []), "required_files"):
        required.append(("runtime dependency", rooted(value, root), "file"))
    missing = [{"kind": label, "path": str(path)} for label, path, kind in required
               if not (path.is_file() if kind == "file" else path.is_dir())]
    return {
        "release_root": str(root), "runtime_config": str(runtime_path),
        "route": route, "runtime_profile": role, "command": command,
        "cwd": str(cwd.resolve()), "environment_overrides": env_values,
        "path_prepend": [str(p) for p in path_prepend],
        "cleared_environment_variables": ["PYTHONHOME"],
        "missing_paths": missing,
        "note": "Dry-run resolves paths only; it does not import modules or validate model/data compatibility.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release-root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="repository root (default: parent of this scripts directory)")
    parser.add_argument("--runtime", required=True, help="explicit JSON runtime configuration, relative to release root or absolute")
    parser.add_argument("--dry-run", action="store_true", help="print the resolved launch plan without running Python or requiring remote paths")
    parser.add_argument("route", choices=sorted(ROUTES), help="hac/hacpp select portable codecs; *-train selects native training")
    parser.add_argument("native_args", nargs=argparse.REMAINDER, help="original entry-point arguments, optionally preceded by --")
    args = parser.parse_args(argv)
    root = args.release_root.expanduser().resolve()
    try:
        plan = build_plan(root, rooted(args.runtime, root), args.route, args.native_args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    if plan["missing_paths"]:
        parser.error("Required paths are missing:\n" + "\n".join("  %(kind)s: %(path)s" % item for item in plan["missing_paths"]))
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.update(plan["environment_overrides"])
    env["PATH"] = os.pathsep.join(plan["path_prepend"] + [env.get("PATH", "")])
    try:
        return subprocess.call(plan["command"], cwd=plan["cwd"], env=env)
    except OSError as exc:
        parser.error("Cannot launch %s: %s" % (args.route, exc))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
