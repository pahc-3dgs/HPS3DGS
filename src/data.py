"""Dataset helpers for PAHC-3DGS experiments."""

from __future__ import annotations

import cv2
import glob
import Imath
import numpy as np
import OpenEXR as exr
from dataclasses import dataclass
from pathlib import Path
from scipy.spatial.transform import Rotation


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsics."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def matrix(self):
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )


@dataclass
class FrameData:
    """Paths and pose for a single RGB-D frame."""

    index: int
    image_path: Path
    depth_path: Path | None
    mask_path: Path | None
    c2w: np.ndarray


@dataclass
class RGBDTrajectoryConfig:
    """Configuration for the RGB-D trajectory reader migrated from datasets_np.py."""

    data_root: Path
    intrinsics: CameraIntrinsics
    scale: float = 1.0
    shift: tuple[float, float, float] = (0.0, 0.0, 0.0)
    depth_scale: float = 1.0
    max_depth: float | None = None


def read_exr_depth(filename: str | Path, channel: str = "FinalImageMovieRenderQueue_WorldDepth.R"):
    """Read a single-channel EXR depth image."""
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    exr_file = exr.InputFile(str(filename))
    window = exr_file.header()["dataWindow"]
    shape = (window.max.y - window.min.y + 1, window.max.x - window.min.x + 1)
    depth = np.frombuffer(exr_file.channel(channel, pt), dtype=np.float32)
    return depth.reshape(shape)


def load_traj_txt(traj_path: str | Path):
    """Load the ``traj.txt`` format used by the source RGB-D reader."""

    lines = Path(traj_path).read_text(encoding="utf-8").strip().splitlines()
    values: list[list[float]] = []
    for line in lines:
        parts = line.split()
        values.append([float(part.split("=")[1]) for part in parts[:6]])

    arr = np.asarray(values, dtype=np.float32)
    xyz = arr[:, :3] / 100.0
    pitch = -arr[:, 3]
    yaw = arr[:, 4]
    roll = -arr[:, 5]
    rotations = Rotation.from_euler(
        "xyz",
        np.stack([roll, pitch, yaw], axis=1),
        degrees=True,
    ).as_matrix()
    poses = np.concatenate([rotations, xyz[:, :, None]], axis=2)
    poses = np.concatenate([poses, np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (len(poses), 1, 1))], axis=1)
    poses = np.concatenate([poses[:, :, 1:2], -poses[:, :, 2:3], poses[:, :, 0:1], poses[:, :, 3:4]], axis=2)
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 2:3, :], poses[:, 0:1, :], poses[:, 3:4, :]], axis=1)
    return poses.astype(np.float32)


def discover_rgbd_frames(config: RGBDTrajectoryConfig):
    """Discover color/depth files and return normalized frame metadata."""

    root = Path(config.data_root)
    depth_candidates = sorted(glob.glob(str(root / "depth" / "*.*")))
    if depth_candidates and depth_candidates[0].lower().endswith(".exr"):
        image_paths = sorted(root.joinpath("color").glob("*.jpg"))
        depth_paths = [Path(p) for p in depth_candidates]
    else:
        image_paths = sorted(root.joinpath("color").glob("*.png"))
        depth_paths = sorted(root.joinpath("depth").glob("*.tiff"))

    poses = load_traj_txt(root / "traj.txt")
    shift = np.asarray(config.shift, dtype=np.float32)
    frames: list[FrameData] = []
    for idx, image_path in enumerate(image_paths):
        c2w = poses[idx].copy()
        c2w[:3, 3] -= shift
        c2w[:3, 3] /= 2.0 * config.scale
        frames.append(
            FrameData(
                index=idx,
                image_path=image_path,
                depth_path=depth_paths[idx] if idx < len(depth_paths) else None,
                mask_path=None,
                c2w=c2w,
            )
        )
    return frames


def read_depth(path: str | Path, depth_scale: float = 1.0, scale: float = 1.0, max_depth: float | None = None):
    """Read and normalize depth from EXR, TIFF, or common image formats."""

    path = Path(path)
    if path.suffix.lower() == ".exr":
        depth = read_exr_depth(path).astype(np.float32) / depth_scale
    else:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    depth = depth / (2.0 * scale)
    if max_depth is not None:
        depth = np.clip(depth, 0.0, max_depth)
    return depth
