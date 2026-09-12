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

All commands exchange JSON on stdout inside a fixed sentinel frame
(:data:`src.hacpp.driver.RESULT_BEGIN` / ``RESULT_END``); every other stdout
line is HAC++/GPCC progress noise and is returned as ``diagnostics``.

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
from typing import Any, Dict, Optional, Tuple

from .driver import RESULT_BEGIN, RESULT_END

DEFAULT_HACPP_ROOT = "/disk3/ydz/code-worktrees/HAC-plus/all150-fa1-20260902"

#: Diagnostics attached to a parsed payload are capped so that a runaway HAC++
#: log cannot flood the caller, but the cap is generous and always announced.
MAX_DIAGNOSTICS_CHARS = 200_000

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
    "opacity. The supported path is: export the HPS3DGS basis as an init cloud "
    "(src.hacpp.export.export_hacpp_init), train the HAC++ student with HAC++ "
    "train.py, then encode/decode it through this bridge."
)


class HacppBridgeError(RuntimeError):
    pass


def explain_incompatibility():
    return INCOMPATIBILITY_NOTE


def parse_driver_stdout(stdout: str) -> Tuple[Dict[str, Any], str]:
    """Extract the driver's final JSON from noisy stdout.

    HAC++ (and the GPCC ``tmc3`` binary it shells out to) print progress lines
    on stdout *before* the driver emits its result, so the JSON is located via
    the sentinel frame the driver writes, scanning **backwards** from the end:
    the last ``RESULT_END`` line closes the result, and the ``RESULT_BEGIN``
    line directly above it carries the payload. Everything outside the frame is
    returned as diagnostics - it is never silently dropped.

    Raises :class:`HacppBridgeError` when no complete frame is present (this is
    always a driver/protocol bug or a hard crash mid-write, never "no result").
    """

    lines = stdout.splitlines()
    end_index = None
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].strip() == RESULT_END:
            end_index = index
            break
    if end_index is None or end_index == 0:
        raise HacppBridgeError(
            "driver produced no %s frame (protocol violation); stdout tail: %s"
            % (RESULT_END, _tail(stdout))
        )
    begin_index = None
    candidate = end_index - 1
    if candidate >= 0 and lines[candidate].startswith(RESULT_BEGIN):
        begin_index = candidate
    if begin_index is None:
        raise HacppBridgeError(
            "driver result frame has %s but no %s line; stdout tail: %s"
            % (RESULT_END, RESULT_BEGIN, _tail(stdout))
        )
    payload_line = lines[begin_index][len(RESULT_BEGIN):].strip()
    try:
        payload = json.loads(payload_line)
    except json.JSONDecodeError as exc:
        raise HacppBridgeError(
            "driver result line is not valid JSON (%s); line: %.500s" % (exc, payload_line)
        ) from exc
    if not isinstance(payload, dict):
        raise HacppBridgeError("driver result is not a JSON object: %.200s" % (payload_line,))
    outside = [line for index, line in enumerate(lines) if index < begin_index or index > end_index]
    diagnostics = "\n".join(outside).strip("\n")
    return payload, diagnostics


def _tail(text: str, limit: int = 2000):
    text = text.strip("\n")
    if len(text) <= limit:
        return text
    return "...(truncated)..." + text[-limit:]


def _attach_diagnostics(payload: Dict[str, Any], diagnostics: str) -> Dict[str, Any]:
    """Keep the noisy stdout alongside the parsed result (never discard it)."""

    if not diagnostics:
        return payload
    truncated = len(diagnostics) > MAX_DIAGNOSTICS_CHARS
    if truncated:
        diagnostics = (
            "...(head dropped, %d chars)..." % (len(diagnostics) - MAX_DIAGNOSTICS_CHARS)
            + diagnostics[-MAX_DIAGNOSTICS_CHARS:]
        )
    payload = dict(payload)
    payload["diagnostics"] = diagnostics
    payload["diagnostics_truncated"] = truncated
    return payload


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
        try:
            payload, diagnostics = parse_driver_stdout(proc.stdout or "")
        except HacppBridgeError as exc:
            raise HacppBridgeError(
                "driver command %r exited %s without a valid result frame: %s; stderr: %s"
                % (command, proc.returncode, exc, _tail(proc.stderr or ""))
            ) from exc
        payload = _attach_diagnostics(payload, diagnostics)
        if proc.returncode != 0:
            raise HacppBridgeError(
                "driver command %r failed (%s): %s"
                % (command, proc.returncode, proc.stderr.strip() or payload.get("error"))
            )
        if not payload.get("ok", False):
            raise HacppBridgeError(
                "driver command %r reported failure: %s" % (command, payload.get("error"))
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

    def decode(
        self,
        bitstream_dir: str | Path,
        model_path: str | Path | None = None,
        render: bool = False,
        source_path: str | Path | None = None,
        max_cameras: int | None = None,
    ):
        """Decode a portable stream directory.

        ``model_path`` is optional for decode-only validation. Rendering still
        needs it because cameras/images are dataset assets, not codec state.
        """

        extra = ["--bitstream-dir", str(bitstream_dir)]
        if model_path is not None:
            extra += ["--model-path", str(model_path)]
        if source_path is not None:
            extra += ["--source-path", str(source_path)]
        if render:
            extra.append("--render")
        if max_cameras is not None:
            extra += ["--max-cameras", str(max_cameras)]
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
            "(chkpnt*.pth + cfg_args). HPS3DGS only exports the init cloud; it "
            "does not convert SAGA weights."
        )
