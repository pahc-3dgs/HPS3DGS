"""HPS3DGS reference implementation.

This package contains the paper-specific compression code. The underlying
3DGS, semantic feature, scene loading, and rendering infrastructure is accessed
through the optional SegAnyGaussians backend in ``third_party/``.
"""

from .compression import HPS3DGSConfig, run_geo32_compression, run_geometry_compression
from .codec import CompactScene, load_compact_scene, save_compact_scene

__all__ = [
    "CompactScene",
    "HPS3DGSConfig",
    "load_compact_scene",
    "run_geo32_compression",
    "run_geometry_compression",
    "save_compact_scene",
]
