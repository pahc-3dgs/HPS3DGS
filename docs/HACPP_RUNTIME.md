# HAC++ runtime for the PAHC shared backend

The HAC++ reference checkout is used **read-only**; nothing in this repository
writes into it. This note records the runtime that its checkpoints and CUDA
extensions are known to work with, where the isolated runtime lives on this
machine, and the commands used for the Phase-0 evidence gates.

## Documented runtime (matches `HAC-plus/environment.yml`)

| component   | version                                     |
|-------------|---------------------------------------------|
| python      | 3.7.13                                       |
| CUDA        | cudatoolkit 11.6 (driver >= 11.6 is enough)  |
| torch       | 1.12.1                                       |
| torchvision | 0.13.1                                       |
| torchaudio  | 0.12.1                                       |
| scatter     | pytorch-scatter (conda `pyg` channel)        |

Hash-grid flags used by every model in the reference checkout (HAC++ `train.py`
defaults; `cfg_args` does not record them): `--n_features 4 --log2 13
--log2_2D 15`, resolutions `(18, 24, ..., 514)` and 2D `(130, 258, 514, 1026)`.
Verified from checkpoint shapes: 3D grid `18^3 + 11*2^13 = 95944` rows x 4
features, 2D grid `ceil8(130^2)=16904 + 3*2^15 = 115208` rows x 4 features.

## Isolated runtime on this machine

```
/disk3/ydz/.conda/envs/pahc_hacpp_py112     # python 3.7.13 / torch 1.12.1 / cu116
/disk3/ydz/pahc_hacpp_runtime/src/          # extensions unpacked from HAC++ zips
/disk3/ydz/pahc_hacpp_runtime/venv/         # torch-2.4 venv (see warning below)
/disk3/ydz/hacpp_e2e_smoke/                 # encode bitstreams + logs
```

Reference tree (read-only): `/disk3/ydz/code/test4gszip/HAC-plus`.
Scene: `/disk3/ydz/data/zxa1-12` (UE `cam_info.yaml`, 19 test views, fx=fy=531.2).

## Build the four extensions from the HAC++ submodule ZIPs

```
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH
for ext in arithmetic gridencoder simple-knn diff-gaussian-rasterization; do
  unzip -q -o $HACPLUS/submodules/$ext.zip -d $BUILD/$ext
  pip install --no-build-isolation --no-deps $BUILD/$ext/<inner-dir>
done
```

`arithmetic` and `_gridencoder` also exist as prebuilt `.so` files inside the
reference tree root; they are built for the torch in `HAC_env`, so on this
machine they import only from an environment with matching torch.

## tmc3 / GPCC

`tmc3` is not installed on this machine. `utils/gpcc_utils.py` in the reference
tree falls back to raw numpy anchor storage (`NUMPY:` prefix), so encode/decode
run but the anchor stream is **not GPCC-compressed**: reported sizes are larger
than the paper's and must be labelled `numpy_fallback`, not paper-equivalent.
Paper-equivalent numbers need an official MPEG-pcc `tmc3` build.

## Tested incompatibility (do not "fix" by switching to torch 2.x)

Rendering the `zxa1-12_init` checkpoint (May 7, `voxel_size=0.001`, 97,744
anchors, ~233k generated Gaussians, reference 19-view PSNR 43.066) under
torch 2.4.1+cu121 makes the rasterizer allocate a ~25.17 GiB
`(gaussian, tile)` binning buffer and OOM on a 24 GB card. This reproduced with
both the environment's `diff_gaussian_rasterization` and a fresh build of the
HAC++ submodule ZIP, so it is an extension/checkpoint/torch-ABI interaction,
not a missing build. `encode` is unaffected (it completed in 2.6 s, 2.961 MiB
reported). The driver prints this warning before rendering under torch >= 2;
`--allow-torch2-render` acknowledges it. Full-render evidence gates must run in
`pahc_hacpp_py112`.

## Evidence-gate commands

```
PY=/disk3/ydz/.conda/envs/pahc_hacpp_py112/bin/python
D=/disk3/ydz/code-worktrees/pahc3dgs/hacpp-backend-20260903/src/hacpp/driver.py
H=/disk3/ydz/code/test4gszip/HAC-plus
CUDA_VISIBLE_DEVICES=3 $PY $D --hacpp-root $H inspect
CUDA_VISIBLE_DEVICES=3 $PY $D --hacpp-root $H encode \
  --model-path $H/output/zxa1-12_init --source-path /disk3/ydz/data/zxa1-12 \
  --out-dir /disk3/ydz/hacpp_e2e_smoke/encode
CUDA_VISIBLE_DEVICES=3 $PY $D --hacpp-root $H decode \
  --model-path $H/output/zxa1-12_init --source-path /disk3/ydz/data/zxa1-12 \
  --bitstream-dir /disk3/ydz/hacpp_e2e_smoke/encode --render --max-cameras 1
```
