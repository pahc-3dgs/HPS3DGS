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
/disk3/ydz/.conda/envs/pahc_hacpp_py112     # python 3.8.18 / torch 1.12.1+cu113
/disk3/ydz/pahc_hacpp_runtime/src/          # extensions unpacked from HAC++ zips
/disk3/ydz/pahc_hacpp_runtime/bin/tmc3      # MPEG G-PCC encoder/decoder (see below)
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
machine they import only from an environment with matching torch. (Stale
torch-2.x copies of these `.so` shadow the correct ones when the HAC++ root is
prepended to `sys.path`; they have been moved to
`/disk3/ydz/pahc_hacpp_runtime/stale_so_backup/`.)

## tmc3 / GPCC

`tmc3` (MPEG G-PCC, `release-v23.0-rc2`) is built in the isolated runtime:

```
/disk3/ydz/pahc_hacpp_runtime/mpeg-pcc-tmc13/build/tmc3/tmc3
  -> symlink /disk3/ydz/pahc_hacpp_runtime/bin/tmc3
```

Build: clone `https://github.com/MPEGGroup/mpeg-pcc-tmc13`, then
`mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j`.

Add `/disk3/ydz/pahc_hacpp_runtime/bin` to `PATH` before running the driver so
`shutil.which("tmc3")` succeeds; `inspect` then reports `gpcc_mode: "gpcc"`
instead of `numpy_fallback`. With GPCC the zxa1-12 anchor stream drops from
0.1049 MiB (raw numpy fallback) to 0.0191 MiB.

## Render OOM (root cause: decoded_version, not torch 2.x)

The ~25.17 GiB `(gaussian, tile)` binning OOM on `zxa1-12_init` was initially
attributed to a torch-2.x rasterizer incompatibility. The real cause is that the
driver constructed `GaussianModel` with `decoded_version=False`, so
`get_scaling` re-applied `exp()` to the checkpoint's already-linear `_scaling`
and inflated every Gaussian to ~1.0 scales. `build_gaussians` now passes
`decoded_version=True` (matching train.py's `run_codec`), which fixes rendering
on torch 1.12.1; torch 2.x remains un-re-verified. `encode` was never affected.

## Evidence-gate commands

```
PY=/disk3/ydz/.conda/envs/pahc_hacpp_py112/bin/python
D=/disk3/ydz/code-worktrees/pahc3dgs/hacpp-backend-20260903/src/hacpp/driver.py
H=/disk3/ydz/code/test4gszip/HAC-plus
export PATH=/disk3/ydz/pahc_hacpp_runtime/bin:$PATH   # for tmc3
CUDA_VISIBLE_DEVICES=3 $PY $D --hacpp-root $H inspect
CUDA_VISIBLE_DEVICES=3 $PY $D --hacpp-root $H encode \
  --model-path $H/output/zxa1-12_init --source-path /disk3/ydz/data/zxa1-12 \
  --out-dir /disk3/ydz/hacpp_e2e_smoke/encode
CUDA_VISIBLE_DEVICES=3 $PY $D --hacpp-root $H decode \
  --model-path $H/output/zxa1-12_init --source-path /disk3/ydz/data/zxa1-12 \
  --bitstream-dir /disk3/ydz/hacpp_e2e_smoke/encode --render --max-cameras 1
```
