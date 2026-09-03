"""Subprocess bridge to an external HAC++ reference checkout.

Why a subprocess: both SegAnyGaussians and HAC++ expose top-level packages
named ``scene`` and ``gaussian_renderer``. Importing both in one process is a
name collision, so the bridge launches the standalone ``driver.py`` with the
HAC++ root at ``sys.path[0]`` and exchanges JSON on stdout.

What this bridge *does*:

* ``inspect``      - report whether the external HAC++ checkout and its CUDA
                     extensions (simple_knn, torch_scatter, the arithmetic
                     codec, diff_gaussian_rasterization) are importable.
* ``encode``       - run ``GaussianModel.conduct_encoding`` on a directory
                     produced by **HAC++'s own train.py**.
* ``decode``       - run ``GaussianModel.conduct_decoding`` against a bitstream
                     directory, optionally rendering test views.

What it deliberately does *not* do:

* restore HAC++ state from a SAGA/3DGS checkpoint. A HAC++ ``GaussianModel``
  stores anchors/offsets/anchor-features plus MLPs and a hash grid; the SAGA
  30k ``point_cloud.ply`` has neither. Call :func:`explain_incompatibility`
  for the wording to use in reports.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_HACPP_ROOT = "/disk3/ydz/code-worktrees/HAC-plus/all150-fa1-20260902"

REQUIRED_HACPP_FILES = (
    "scene/gaussian_model.py",
    "gaussian_renderer/__init__.py",
    "utils/entropy_models.py",
    "utils/encodings.py",
    "train.py",
)

INCOMPATIBILITY_NOTE = (
    "HAC++ cannot be initialised from a SAGA/3DGS point_cloud.ply: HAC++ stores "
    "anchors + K offsets + anchor features + shared MLPs + a binary hash grid, "
    "which is a different parameterisation from per-Gaussian xyz/SH/scale/rot/"
    "opacity. The supported path is: export the PAHC basis as an init cloud "
    "(src.hacpp.export.export_hacpp_init), train the HAC++ student with HAC++ "
    "train.py, then encode/decode it through this bridge."
)


class HacppBridgeError(RuntimeError):
    pass


def explain_incompatibility():
    return INCOMPATIBILITY_NOTE


@dataclass
class HacppBridge:
    """Adapter for an external HAC++ checkout (read-only, never modified)."""

    hacpp_root: str | Path = DEFAULT_HACPP_ROOT
    python_exe: str | None = None
    device: str = "cuda:0"
    timeout_s: int = 3600

    def __post_init__(self):
        self.hacpp_root = Path(self.hacpp_root).resolve()
        self.python_exe = self.python_exe or sys.executable
        missing = [name for name in REQUIRED_HACPP_FILES if not (self.hacpp_root / name).is_file()]
        if missing:
            raise HacppBridgeError(
                "HAC++ checkout at %s is missing %s" % (self.hacpp_root, ", ".join(missing))
            )

    # ------------------------------------------------------------------ run
    def _driver_argv(self, command: str, extra: list[str]):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.hacpp_root) + os.pathsep + env.get("PYTHONPATH", "")
        repo_root = Path(__file__).resolve().parents[2]
        env["PYTHONPATH"] = str(repo_root) + os.pathsep + env["PYTHONPATH"]
        return [
            self.python_exe,
            str(repo_root / "src" / "hacpp" / "driver.py"),
            "--hacpp-root",
            str(self.hacpp_root),
            "--device",
            self.device,
            command,
            *extra,
        ], env

    def run(self, command: str, extra: Optional[list[str]] = None):
        argv, env = self._driver_argv(command, list(extra or []))
        proc = subprocess.run(
            argv,
            env=env,
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
        )
        payload: Dict[str, Any] = {}
        if proc.stdout.strip():
            try:
                payload = json.loads(proc.stdout)
            except json.JSONDecodeError:
                payload = {"ok": False, "error": "unparsable driver output", "stdout": proc.stdout}
        if proc.returncode != 0:
            raise HacppBridgeError(
                "driver command %r failed (%s): %s"
                % (command, proc.returncode, proc.stderr.strip() or payload.get("error"))
            )
        return payload

    # -------------------------------------------------------------- commands
    def inspect(self):
        """Dependency report; ``ready_for_encode`` is the honest gate."""

        return self.run("inspect")

    def encode(self, model_path: str | Path, out_dir: str | Path, source_path: str | Path | None = None):
        """Encode a trained HAC++ model; ``out_dir`` receives the raw streams."""

        extra = ["--model-path", str(model_path), "--out-dir", str(out_dir)]
        if source_path is not None:
            extra += ["--source-path", str(source_path)]
        return self.run("encode", extra)

    def decode(self, model_path: str | Path, bitstream_dir: str | Path, render: bool = False, source_path: str | Path | None = None):
        extra = [
            "--model-path",
            str(model_path),
            "--bitstream-dir",
            str(bitstream_dir),
        ]
        if source_path is not None:
            extra += ["--source-path", str(source_path)]
        if render:
            extra.append("--render")
        return self.run("decode", extra)

    def render(self, model_path: str | Path, source_path: str | Path | None = None):
        extra = ["--model-path", str(model_path)]
        if source_path is not None:
            extra += ["--source-path", str(source_path)]
        return self.run("render", extra)

    def requires_trained_student(self):
        """Human-readable reminder for callers building pipelines."""

        return (
            "encode/decode need a HAC++ model directory from HAC++ train.py "
            "(chkpnt*.pth + cfg_args). PAHC only exports the init cloud; it "
            "does not convert SAGA weights."
        )
