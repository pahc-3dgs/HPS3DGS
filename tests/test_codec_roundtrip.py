import tempfile
import unittest
from pathlib import Path

import torch

from src.codec import (
    CompactScene,
    TemplateInstance,
    apply_instance_scaling,
    compute_instance_residuals,
    decode_compact_scene,
    load_compact_scene,
    save_compact_scene,
)


def _base_scene(mode="residual"):
    static = {
        "xyz": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "features_dc": torch.zeros(2, 3, 1),
        "features_rest": torch.zeros(2, 15, 3),
        "scaling": torch.zeros(2, 3),
        "rotation": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
        "opacity": torch.ones(2, 1),
    }
    instance = TemplateInstance(
        template_id=0,
        instance_id=7,
        rotation=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        translation=torch.tensor([2.0, 0.0, 0.0]),
        scale=torch.tensor([2.0]),
        center=torch.zeros(3),
        row_map=torch.tensor([1, 0, 1]),
    )
    return CompactScene(
        templates={"0": {"indices": torch.tensor([0, 1])}},
        static_gaussians=static,
        instances=[instance],
        scaling_domains={"scaling": "log"},
        sh_mode=mode,
        residual_attributes=["features_dc", "features_rest"],
    )


class CodecRoundtripTests(unittest.TestCase):
    def test_log_scaling_changes_physical_scale_by_instance_factor(self):
        raw = torch.tensor([[0.2, -0.5, 1.0]])
        transformed = apply_instance_scaling(raw, torch.tensor([3.0]), "log")
        torch.testing.assert_close(torch.exp(transformed), torch.exp(raw) * 3.0)

    def test_target_appearance_survives_save_reload_decode(self):
        for mode in ("residual", "full"):
            with self.subTest(mode=mode):
                scene = _base_scene(mode)
                target = {
                    "xyz": torch.tensor([[4.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
                    "features_dc": torch.arange(9, dtype=torch.float32).reshape(3, 3, 1),
                    "features_rest": torch.arange(135, dtype=torch.float32).reshape(3, 15, 3),
                }
                compute_instance_residuals(scene, [target])
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "scene.pt"
                    save_compact_scene(scene, path)
                    decoded = decode_compact_scene(load_compact_scene(path), require_residuals=True)
                torch.testing.assert_close(decoded["features_dc"][2:], target["features_dc"])
                torch.testing.assert_close(decoded["features_rest"][2:], target["features_rest"])
                torch.testing.assert_close(decoded["xyz"][2:], target["xyz"])

    def test_lossless_decode_rejects_missing_residual(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            decode_compact_scene(_base_scene("residual"), require_residuals=True)


if __name__ == "__main__":
    unittest.main()
