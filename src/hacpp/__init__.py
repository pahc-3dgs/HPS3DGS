"""HPS3DGS + shared-HAC++ backend.

One scene uses **one** HAC++ decoder (MLPs + entropy models) shared by every
component template; only anchors/offsets/owner streams are per-scene data.
Per-template HAC++ models are explicitly rejected (see
:mod:`src.hacpp.manifest`).
"""

from .bridge import HacppBridge, HacppBridgeError, explain_incompatibility
from .manifest import HacppManifest, ManifestError, SharedDecoder
from .owners import build_owner_index, gather_template_rows, owner_blocks_are_contiguous

__all__ = [
    "HacppBridge",
    "HacppBridgeError",
    "explain_incompatibility",
    "HacppManifest",
    "ManifestError",
    "SharedDecoder",
    "build_owner_index",
    "gather_template_rows",
    "owner_blocks_are_contiguous",
]
