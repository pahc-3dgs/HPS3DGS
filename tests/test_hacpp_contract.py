import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.codec import CompactScene, TemplateInstance
from src.hacpp.export import SH_C0, basis_to_init_points
from src.hacpp.bridge import HacppBridge
from src.hacpp.instance_math import expand_instance_anchors
from src.hacpp.manifest import ManifestError, OwnerSection, verify_shared_decoder_file
from src.hacpp.pipeline import load_instances, save_instances
from src.hacpp.pipeline import build_bundle
from src.hacpp.bundle import read_bundle
from src.metrics import compression_report


class HacppContractTests(unittest.TestCase):
    def test_bridge_invokes_the_installed_src_driver_module(self):
        bridge = object.__new__(HacppBridge)
        bridge.python_exe = "python"
        bridge.hacpp_root = Path("/tmp/hacpp")
        bridge.device = "cuda:3"
        argv, _ = bridge._driver_argv("inspect", [])
        self.assertTrue(argv[1].replace("\\", "/").endswith("src/hacpp/driver.py"))

    def test_sh_dc_conversion_uses_band_zero_constant(self):
        xyz = torch.zeros(1, 3)
        dc = torch.ones(1, 3, 1)
        _, colors, _ = basis_to_init_points(xyz, dc)
        expected = int((SH_C0 + 0.5) * 255.0)
        np.testing.assert_array_equal(colors, np.full((1, 3), expected, dtype=np.uint8))

    def test_anchor_offset_broadcast_and_transform(self):
        anchors = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        offsets = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        voxel_scale = torch.tensor([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]])
        out = expand_instance_anchors(
            anchors, offsets, voxel_scale, torch.zeros(3),
            torch.tensor([1.0, 0.0, 0.0, 0.0]), torch.tensor([1.0, 0.0, 0.0]), 2.0,
        )
        expected = torch.tensor([[[5.0, 0.0, 0.0], [1.0, 6.0, 0.0]], [[7.0, 0.0, 0.0], [3.0, 6.0, 0.0]]])
        torch.testing.assert_close(out, expected)
        single = expand_instance_anchors(
            anchors, offsets[:, 0], voxel_scale, torch.zeros(3),
            torch.tensor([1.0, 0.0, 0.0, 0.0]), torch.zeros(3), 1.0,
        )
        self.assertEqual(tuple(single.shape), (2, 1, 3))

    def test_instances_npz_roundtrip_keeps_per_instance_shapes(self):
        instances = [
            TemplateInstance(0, 1, torch.tensor([1.0, 0.0, 0.0, 0.0]), torch.zeros(3), residuals={"features_dc": torch.arange(6.0).reshape(2, 3)}, row_map=torch.tensor([1, 0])),
            TemplateInstance(0, 2, torch.tensor([1.0, 0.0, 0.0, 0.0]), torch.ones(3), residuals={"features_rest": torch.arange(12.0).reshape(1, 4, 3), "opacity": torch.ones(1, 1)}, row_map=torch.tensor([0])),
        ]
        scene = CompactScene(templates={}, instances=instances)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "instances.npz"
            save_instances(path, scene)
            restored = load_instances(path)
        self.assertEqual(set(restored[0]["residuals"]), {"features_dc"})
        self.assertEqual(set(restored[1]["residuals"]), {"features_rest", "opacity"})
        torch.testing.assert_close(restored[0]["residuals"]["features_dc"], instances[0].residuals["features_dc"])
        torch.testing.assert_close(restored[1]["residuals"]["features_rest"], instances[1].residuals["features_rest"])
        torch.testing.assert_close(restored[1]["residuals"]["opacity"], instances[1].residuals["opacity"])

    def test_shared_decoder_rejects_nested_hash_and_owner_stability_needs_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.pt"
            good = Path(tmp) / "good.pt"
            torch.save({"module": {"encoding_xyz": {"params": torch.ones(1)}}}, bad)
            torch.save({"color_mlp": {"weight": torch.ones(1)}}, good)
            self.assertTrue(verify_shared_decoder_file(bad))
            self.assertEqual(verify_shared_decoder_file(good), [])
        with self.assertRaises(ManifestError):
            OwnerSection(num_anchors=1, num_owners=1, phase="stable").validate()
        OwnerSection(num_anchors=1, num_owners=1, phase="stable", provenance="propagated in training").validate()

    def test_actual_file_bytes_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bits.bin"
            path.write_bytes(b"123456789")
            report = compression_report(10, 5, final_payload={"x": torch.ones(100)}, bitstream_paths=path)
        self.assertEqual(report["bitstream_bytes"], 9)
        self.assertTrue(report["bitstream_is_actual"])
        self.assertEqual(report["estimated_tensor_bytes"], 400)

    def test_synthetic_bundle_validates_sizes_and_checksums(self):
        scene = CompactScene(
            templates={"0": {"indices": torch.tensor([0, 1])}},
            static_gaussians={"xyz": torch.zeros(2, 3)},
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            streams = root / "streams"
            streams.mkdir()
            (streams / "xyz_gpcc.npz").write_bytes(b"anchor")
            (streams / "hash.b").write_bytes(b"hash")
            (streams / "masks.b").write_bytes(b"mask")
            decoder = root / "shared_mlp.pt"
            torch.save({"color_mlp": {"weight": torch.ones(1)}}, decoder)
            bundle = root / "bundle"
            manifest = build_bundle(
                bundle,
                scene,
                hacpp_stream_dir=streams,
                shared_decoder_weights=decoder,
            )
            self.assertEqual(manifest.owners.phase, "init_only")
            loaded = read_bundle(bundle, verify=True)
            self.assertEqual(loaded.shared_decoder.weights, "hacpp/shared_mlp.pt")


if __name__ == "__main__":
    unittest.main()
