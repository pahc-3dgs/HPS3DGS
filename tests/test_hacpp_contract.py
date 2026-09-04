import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.codec import CompactScene, TemplateInstance
from src.hacpp.export import SH_C0, basis_to_init_points
from src.hacpp.bridge import HacppBridge, HacppBridgeError, parse_driver_stdout
from src.hacpp.instance_math import expand_instance_anchors
from src.hacpp.manifest import (
    ManifestError,
    OwnerSection,
    validate_shared_decoder_file,
    verify_shared_decoder_file,
)
from src.hacpp.pipeline import load_instances, save_instances
from src.hacpp.pipeline import build_bundle
from src.hacpp.bundle import bundle_artifact_bytes, read_bundle
from src.metrics import compression_report


class HacppContractTests(unittest.TestCase):
    @staticmethod
    def _decoder_config(parameter_bytes=20):
        return {
            "config_version": 1,
            "feat_dim": 50,
            "n_offsets": 10,
            "voxel_size": 0.005,
            "update_depth": 3,
            "update_init_factor": 16,
            "update_hierachy_factor": 4,
            "use_feat_bank": False,
            "n_features_per_level": 4,
            "log2_hashmap_size": 13,
            "log2_hashmap_size_2D": 15,
            "resolutions_list": [18, 24],
            "resolutions_list_2D": [130, 258],
            "use_2D": True,
            "ste_binary": True,
            "ste_multistep": False,
            "add_noise": False,
            "decoded_version": True,
            "white_background": False,
            "is_synthetic_nerf": False,
            "eval": True,
            "all_views_train_test": False,
            "Q": 1,
            "dtype": "float32",
            "shared_decoder_parameter_bytes": parameter_bytes,
        }

    @staticmethod
    def _decoder_state():
        return {
            name: {"weight": torch.ones(1)}
            for name in (
                "opacity_mlp", "cov_mlp", "color_mlp", "grid_mlp", "deform_mlp"
            )
        }

    def test_driver_protocol_extracts_last_adjacent_frame_and_keeps_noise(self):
        payload, diagnostics = parse_driver_stdout(
            "HAC log\n@@HACPP_RESULT@@ {\"ok\": true, \"views\": 1}\n"
            "@@HACPP_RESULT_END@@\ntrailing diagnostic\n"
        )
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["views"], 1)
        self.assertIn("HAC log", diagnostics)
        self.assertIn("trailing diagnostic", diagnostics)
        with self.assertRaises(HacppBridgeError):
            parse_driver_stdout("@@HACPP_RESULT@@ {\"ok\": true}\nnoise\n@@HACPP_RESULT_END@@\n")

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
                validate_shared_decoder_file(good)
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
            (streams / "x_bound_min.pkl").write_bytes(b"xmin")
            (streams / "x_bound_max.pkl").write_bytes(b"xmax")
            (streams / "feat_0_0.b").write_bytes(b"feat")
            (streams / "scaling_0.b").write_bytes(b"scale")
            (streams / "offsets_0.b").write_bytes(b"offset")
            (streams / "decoder_config.json").write_text(
                __import__("json").dumps(self._decoder_config()), encoding="utf-8"
            )
            decoder = root / "shared_mlp.pt"
            torch.save(self._decoder_state(), decoder)
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
            self.assertEqual(loaded.storage["paper_bound_bytes"], 24)
            self.assertEqual(loaded.storage["paper_total_bytes"], 73)
            self.assertEqual(loaded.storage["artifact_bytes"], bundle_artifact_bytes(bundle))

    def test_codec_only_bundle_needs_no_owner_sidecars(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            streams = root / "streams"
            streams.mkdir()
            payloads = {
                "xyz_gpcc.npz": b"a", "hash.b": b"h", "masks.b": b"m",
                "x_bound_min.pkl": b"x", "x_bound_max.pkl": b"y",
                "feat_0_0.b": b"f", "scaling_0.b": b"s", "offsets_0.b": b"o",
            }
            for name, value in payloads.items():
                (streams / name).write_bytes(value)
            decoder = streams / "shared_mlp.pt"
            torch.save(self._decoder_state(), decoder)
            (streams / "decoder_config.json").write_text(
                __import__("json").dumps(self._decoder_config()), encoding="utf-8"
            )
            bundle = root / "bundle"
            manifest = build_bundle(bundle, hacpp_stream_dir=streams, scene_id="synthetic")
            self.assertTrue(manifest.codec_only)
            self.assertFalse((bundle / "owners.npz").exists())
            self.assertFalse((bundle / "instances.npz").exists())
            read_bundle(bundle, verify=True)


if __name__ == "__main__":
    unittest.main()
