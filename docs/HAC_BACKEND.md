# Portable HAC backend

This is the codec-only counterpart of `src/hacpp`. It reuses the established
experiment and evaluation workflow, while retaining native HAC models, tensor
layouts, 16-bit coordinate quantization and Gaussian arithmetic coding.
It does not implement part clustering, template sharing, or owner propagation.

The source model must come from native HAC training with the small persistence
patch in this experiment: its `bitstreams/` directory contains the original
`_quantized_v.npy`, `patched_infos.json`, `x_bound_min.pkl`, and `x_bound_max.pkl`.
Original HAC checkpoints and HAC++ checkpoints are not interchangeable.

## Commands

Use the verified Torch 1.12.1 codec environment for this pinned implementation.
Set `CUDA_VISIBLE_DEVICES` before Python starts and use a free GPU.

```bash
python scripts/hac_backend.py --hac-root /path/to/HAC encode \
  --model /path/to/native/model --dataset /path/to/dataset --out /path/to/new/raw
python scripts/hac_backend.py pack \
  --raw /path/to/new/raw --bundle /path/to/new/bundle --scene scene_name
python scripts/hac_backend.py verify --bundle /path/to/new/bundle
python scripts/hac_backend.py --hac-root /path/to/HAC decode \
  --bundle /path/to/new/bundle --output /path/to/new/decode
python scripts/hac_backend.py --hac-root /path/to/HAC decode \
  --bundle /path/to/new/bundle --dataset /path/to/dataset --output /path/to/new/render
```

Decode always verifies bundle checksums before loading weights. It accepts no
source model/checkpoint path. Only the optional render step reads the dataset
for camera metadata and GT. The renderer currently checks the audited five-scene
protocol: 150 train / same 150 reconstruction cameras at 1280 x 720.

## Codec-specific decisions

- HAC's original `conduct_decoding` requires `[N_full,N_active,batch_size]` and
  existing target tensor shapes. The adapter persists counts and preallocates
  zero tensors before invoking that unchanged native routine.
- Hash embeddings are restored from `hash.b` before entropy predictions.
- Bounds are the exact training-time values, never re-estimated from a decoded
  or padded point cloud.
- Native training saves already-decoded coordinates. Re-encoding recovers their
  lattice indices with rounding and validates them against the original native
  integer anchor stream. It does not reapply the training-time floor quantizer.
- The shared decoder contains the opacity, covariance, color, entropy-grid and
  optional feature-bank MLPs. HAC's unused `mlp_deform` is excluded, consistent
  with native `get_mlp_size`; HAC++ uses that name for a different, required
  channel-context network.
- The fixed identity quaternion used for anchor visibility culling is recreated
  from the decoded count. Actual Gaussian covariance comes from the decoded MLP.

The `pahc-hac-codec-only-v1` bundle format is distinct from the HAC++ manifest.
It rejects missing arithmetic batches, side information, unknown files,
inventory changes and checksum mismatches. `artifact_bytes` includes the manifest
itself, decoder weights and all metadata. `native_accounted_bytes` separately
reproduces native HAC's 16-bit xyz + arithmetic payload + required raw MLP tensor
accounting, which omits serialization and side-information overhead.

`src/hac/evaluation.py` copies the previously verified camera and PNG metric
helpers unchanged; experiment `evaluation_reuse.json` records their source hash.

Validation: 8 new bundle contract tests, 13 existing HAC++ contract/runtime tests,
an actual CUDA round trip with inactive/padded anchors and byte-identical
re-encoding, followed by five-scene independent PNG and GT comparison.

```bash
python -m unittest tests.test_hac_bundle tests.test_hacpp_contract tests.test_hacpp_runtime_guard -v
```
