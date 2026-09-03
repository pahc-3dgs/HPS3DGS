"""Runtime guards: HAC++ driver render-path notes."""

import sys
import types
import unittest
from unittest import mock

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.hacpp import driver  # noqa: E402


class Torch2RenderGuardTest(unittest.TestCase):
    def test_warning_records_the_evidence(self):
        for needle in ("25.17 GiB", "zxa1-12_init", "decoded_version", "torch"):
            self.assertIn(needle, driver.TORCH2_RENDER_WARNING)

    def test_guard_blocks_torch2_and_passes_torch1(self):
        stub = types.ModuleType("torch")
        with mock.patch.dict(sys.modules, {"torch": stub}):
            stub.__version__ = "2.4.1+cu121"
            with self.assertRaises(RuntimeError) as ctx:
                driver._check_render_runtime(allow=False)
            self.assertIn("25.17 GiB", str(ctx.exception))
            stub.__version__ = "1.12.1"
            self.assertIsNone(driver._check_render_runtime(allow=False))

    def test_guard_warns_but_continues_when_allowed(self):
        stub = types.ModuleType("torch")
        stub.__version__ = "2.4.1+cu121"
        with mock.patch.dict(sys.modules, {"torch": stub}):
            message = driver._check_render_runtime(allow=True)
        self.assertIn("decoded_version", message)

    def test_grid_config_matches_reference_checkpoint_shapes(self):
        # 3D grid rows: 18^3 + 11 * 2^13 = 95944 (x4 features),
        # 2D rows: ceil8(130^2) + 3 * 2^15 = 115208 (x4 features). These are
        # the shapes of every checkpoint in the reference checkout.
        rows3d = 18 ** 3 + 11 * 2 ** driver.ENCODING_CONFIG["log2_hashmap_size"]
        # GridEncoder rounds each level up to a multiple of 8 (130^2=16900 -> 16904).
        rows2d = 16904 + 3 * 2 ** driver.ENCODING_CONFIG["log2_hashmap_size_2D"]
        self.assertEqual(rows3d, 95944)
        self.assertEqual(rows2d, 115208)
        self.assertEqual(driver.ENCODING_CONFIG["n_features_per_level"], 4)


if __name__ == "__main__":
    unittest.main()
