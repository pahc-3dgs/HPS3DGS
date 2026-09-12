<h1 align="center">HPS3DGS</h1>

<p align="center">
  <b>Hierarchical 3D Gaussian Compression for Aerospace Objects</b>
</p>

<p align="center">
  <a href="#overview">Overview</a> &bull;
  <a href="#installation">Installation</a> &bull;
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#project-structure">Project Structure</a> &bull;
  <a href="#testing">Testing</a>
</p>

HPS3DGS brings together PAHC, Geo33, independent Fig. 7 experiments, and portable HAC/HAC++ codecs in one versioned research codebase. It provides a common launcher while preserving each method's model, training procedure, and bitstream format.

**Release status:** `v0.1.0-rc.1` is the validated integration snapshot. Development currently uses `integrate/hps3dgs-20260912`; `main` still points to the earlier PAHC baseline. The snapshot covers code and representative decoding tests; complete training pipelines have not all been rerun.

## Overview

- **Geometry and appearance compression:** PAHC template-instance representations, Geo33 quantization-aware training, and native/compact storage formats.
- **Reproducible experiments:** independent Fig. 7 threshold runs with explicit input manifests and cached-candidate hashes.
- **Portable codecs:** separate HAC and HAC++ adapters for packing, independent decoding, and rendering.
- **Versioned validation:** pinned recursive dependencies, a source manifest, CPU tests, and GPU package regression tests.

| Route | Entry point | Purpose |
|---|---|---|
| `pahc` | `scripts/run_pahc_pipeline.py` | Scene training, masks, features, and PAHC compression |
| `geo33` | `third_party/SegAnyGAussians/geo33.py` | Geometry preparation, four-group QAT, native package decoding |
| `fig7` | `third_party/SegAnyGAussians/fig7_taur.py` | Independent threshold experiments and Fig. 7 package decoding |
| `hac` / `hacpp` | `scripts/hac_backend.py` / `scripts/hacpp_backend.py` | Separate portable codec interfaces |
| `hac-train` / `hacpp-train` | The corresponding backend's `train.py` | Native backend training |

HAC/HAC++ integration currently has `codec_only` scope. Geo33 and Fig. 7 retain their own quantization and decoding conventions.

## Installation

### 1. Get the code

```bash
git clone --branch integrate/hps3dgs-20260912 \
  https://github.com/pahc-3dgs/HPS3DGS.git
cd HPS3DGS
```

The current snapshot pins its dependencies to **server-local Git mirrors** under `/disk3/ydz/code/HPS3DGS-repositories`. On that server, initialize all six recursive submodules with:

```bash
GIT_LFS_SKIP_SMUDGE=1 git -c protocol.file.allow=always \
  submodule update --init --recursive
git submodule status --recursive
```

On another machine, first restore the mirrors or configure accessible remotes containing the **same pinned commits**, including the three nested SegAny dependencies. Cloning the parent repository alone does not provide these dependencies. See [dependency provenance](provenance/dependency_sources.json) and [backup and recovery](docs/VERSIONING.md). `GIT_LFS_SKIP_SMUDGE=1` retrieves source and model pointers without downloading the large CLIP weights.

### 2. Set up the runtime environments

The tested platform is Linux with an NVIDIA GPU. Use Git, Git LFS, Conda, and a CUDA toolchain compatible with the selected PyTorch build. CPU contract tests do not require a GPU.

The integration was validated with the following environments:

| Runtime profile | Routes | Python | PyTorch | PyTorch CUDA build |
|---|---|---|---|---|
| `saga` | PAHC, Geo33, Fig. 7 | 3.8.20 | 2.4.1 | 12.1 |
| `hac_train` / `hacpp_train` | Native training entry points | 3.8.20 | 2.4.1 | 12.1 |
| `codec` | Portable HAC/HAC++ codecs | 3.8.18 | 1.12.1 | 11.3 |

**PAHC / Geo33 / Fig. 7.** Start from a working Python 3.8 SAGA environment with PyTorch, torchvision, and its CUDA rasterizers installed. Follow the [pinned SAGA setup](third_party/SegAnyGAussians/README.md#installation) for its dependencies, then install the incremental PAHC package from the HPS3DGS root:

```bash
conda activate sacgs
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

`requirements.txt` adds PAHC dependencies; it does not install the SAGA rasterizers or select a GPU stack. Its current NumPy/OpenCV pins target Python 3.8. The historical upstream `environment.yml` files use Python 3.7 and are not a complete environment specification for this integration.

**Native HAC/HAC++ and portable codecs.** Keep their runtime environments separate from SAGA. The pinned [HAC](third_party/HAC/README.md#installation) and [HAC++](third_party/HAC-plus/README.md#installation) installation sections describe the backend dependencies. Their `submodules/` directories contain ZIP archives for `arithmetic`, `gridencoder`, `simple-knn`, and `diff-gaussian-rasterization`; extract and build these for the target environment. Use different Conda environment names when creating both backends. `pip install -e .` does not build these extensions.

For portable decoding, retain the Python 3.8 / PyTorch 1.12.1 codec environment in the table above. HAC++ also requires `tmc3` for G-PCC; place it on the configured `PATH`. See [HAC++ runtime notes](docs/HACPP_RUNTIME.md) for extension and G-PCC build details. Those notes include historical experiments; the current runtime paths and identities are recorded in [runtime.4090.json](configs/runtime.4090.json) and [runtime_assets.json](provenance/runtime_assets.json).

The reference server already has these environments and extensions. A clean installation on another machine must pass the checks below; it has not been established as a one-command environment rebuild.

### 3. Configure paths and model weights

On the reference server, use `configs/runtime.4090.json` directly. On another machine:

```bash
cp configs/runtime.example.json configs/runtime.local.json
# Edit runtime.local.json to point to your Python executables and extensions.
```

Each runtime profile defines `python`, `pythonpath`, `path_prepend`, and `required_files`. The launcher selects these explicitly; activating a Conda environment alone does not override the JSON configuration. Keep native arguments such as dataset and output paths absolute.

The full SAGA pipeline also requires SAM ViT-H and CLIP ViT-B/16 weights. Follow the [SAGA instructions](third_party/SegAnyGAussians/README.md#installation) and restore the CLIP LFS payloads from an accessible model source. On the reference server, verified copies are stored in `/disk3/ydz/pahc_hacpp_runtime/hps3dgs_weights`; their exact sizes and hashes are listed in [runtime_assets.json](provenance/runtime_assets.json). Source clones and Git bundles do not include the environments, model payloads, datasets, or checkpoints.

## Quick Start

All commands below run from the HPS3DGS root. Replace the `/absolute/path/...` values with your own inputs and use a new output directory for each run.

### Check the launcher

```bash
RUNTIME=configs/runtime.4090.json  # Use configs/runtime.local.json on another host.

python3 scripts/hps3dgs.py --runtime "$RUNTIME" --dry-run geo33 -- --help
python3 scripts/hps3dgs.py --runtime "$RUNTIME" geo33 -- --help
python3 scripts/hps3dgs.py --runtime "$RUNTIME" fig7 -- --help
python3 scripts/hps3dgs.py --runtime "$RUNTIME" hac -- --help
```

`--dry-run` prints the resolved command, working directory, environment, and missing paths. It does not import or execute the model. The upstream `hacpp-train` entry initializes CUDA before parsing arguments, so even its `--help` requires an available GPU; use launcher `--dry-run` when inspecting it without a GPU.

### Decode a Geo33 package

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/hps3dgs.py --runtime "$RUNTIME" geo33 -- \
  --mode decode \
  --package /absolute/path/to/scene.geo33.zip \
  --source-path /absolute/path/to/dataset \
  --output /absolute/path/to/new_decode_output \
  --eval-views 150
```

The dataset supplies cameras and reference images; the package supplies the compressed model. Geo33 compact storage packages use `run_storage_ablation.py decode`, not the native `geo33.py` decoder.

### Run PAHC

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/hps3dgs.py --runtime "$RUNTIME" pahc -- \
  -s /absolute/path/to/dataset \
  -m /absolute/path/to/new_model_directory \
  -o /absolute/path/to/new_output/scene.pahc.pt \
  --sam-checkpoint-path /absolute/path/to/sam_vit_h_4b8939.pth
```

This invokes scene training, mask/feature preparation, feature training, and compression. For an existing compatible SAGA model with the required features, add `--skip-train --skip-mask --skip-feature`. See the [original PAHC guide](docs/PAHC_ORIGINAL_README.md) for pipeline details.

Fig. 7 uses an explicit source manifest; training additionally requires a hash-bound candidate cache. See [Fig. 7 entry points](third_party/SegAnyGAussians/HPS3DGS_FIG7_ENTRYPOINTS.md). The migrated multi-scene, input-alignment, and HAC++ evaluation scripts are documented in [experiment entry points](docs/EXPERIMENT_ENTRYPOINTS.md).

## Project Structure

```text
HPS3DGS/
├── configs/
│   ├── pahc.yaml                   # PAHC algorithm configuration
│   ├── runtime.4090.json           # Validated reference-server runtimes
│   ├── runtime.example.json        # Template for another machine
│   ├── validation.4090.json        # Reference packages, metrics, and assertions
│   └── fig7_*.4090.json            # Fig. 7 inputs and candidate bindings
├── scripts/
│   ├── hps3dgs.py                  # Common launcher
│   ├── run_pahc_pipeline.py        # Original PAHC pipeline
│   ├── compress.py / decode.py / evaluate.py
│   ├── hac_backend.py / hacpp_backend.py
│   ├── validate_release.py         # Source, CPU, and GPU validation
│   ├── run_unittest_checks.py      # Strict unittest result reporting
│   ├── release_manifest.py         # Source and dependency manifest
│   ├── check_merge.py              # Read-only validation gate
│   └── experiments/                # Migrated experiment drivers
├── src/
│   ├── codec.py / compression.py   # PAHC representation and compression
│   ├── backend.py / data.py        # SAGA integration and input handling
│   ├── math_utils.py / quantization.py / metrics.py
│   └── hac/ / hacpp/               # Independent portable codec adapters
├── third_party/
│   ├── SegAnyGAussians/            # Pinned SAGA, Geo33, and Fig. 7 code
│   │   ├── geo33.py / geo33_storage_codec.py
│   │   ├── fig7_taur.py / fig7_packet.py
│   │   ├── CLIP-ViT-B-16-laion2B-s34B-b88K/  # Nested model/LFS repository
│   │   └── third_party/            # Nested kmeans_pytorch and segment-anything
│   ├── HAC/                        # Pinned native HAC backend
│   └── HAC-plus/                   # Pinned native HAC++ backend
├── tests/                          # PAHC and backend contract tests
├── docs/                           # Usage, validation, and versioning guides
├── provenance/                     # Imported source and runtime identities
├── release_manifest.json           # Recursive source-file hashes
├── requirements.txt                # Incremental PAHC dependencies
└── pyproject.toml                  # PAHC Python package metadata
```

Datasets, checkpoints, compressed packages, and validation outputs are kept outside the source tree.

## Testing

### CPU unit tests

Run the core contract tests in the SAGA environment from the repository root:

```bash
conda activate sacgs
CUDA_VISIBLE_DEVICES="" python -B -m unittest discover -s tests -v
```

This suite covers PAHC representation roundtrips and HAC/HAC++ contracts. It does not require the historical GPU regression packages. The release CPU profile below also runs the SegAny geometry and Fig. 7 tests, including a canonical fixture check.

### Source, CPU, and GPU validation

Validation operates on a **committed, clean checkout** with initialized submodules and a matching `release_manifest.json`. Keep local runtime/fixture overrides in the ignored `configs/runtime.local.json` and `configs/validation.local.json` files, or outside the repository.

The supplied `*.4090.json` files refer to datasets, cached candidates, and reference packages on the reference server. For another host, provide equivalent inputs with matching hashes and update the local fixture paths. Missing fixtures are reported as failures.

Run the following steps in the same Bash session. Set `GPU` to an idle physical device; the GPU profile checks occupancy and takes a shared device lock.

```bash
set -euo pipefail
RUNTIME=configs/runtime.4090.json
FIXTURES=configs/validation.4090.json
RUN_DIR="$(mktemp -d /tmp/hps3dgs-validation.XXXXXX)"
GPU=0

python3 scripts/validate_release.py --profile source \
  --runtime "$RUNTIME" --output "$RUN_DIR/source"

python3 scripts/validate_release.py --profile cpu \
  --runtime "$RUNTIME" --fixtures "$FIXTURES" --output "$RUN_DIR/cpu"

python3 scripts/validate_release.py --profile gpu --gpu "$GPU" \
  --runtime "$RUNTIME" --fixtures "$FIXTURES" --output "$RUN_DIR/gpu"

python3 scripts/check_merge.py \
  --reports "$RUN_DIR/source/report.json" \
            "$RUN_DIR/cpu/report.json" \
            "$RUN_DIR/gpu/report.json" \
  --require source cpu gpu
```

Each profile writes `report.json` with the candidate commit, source-manifest hash, dependency revisions, executed commands, exit codes, and check results. Preserve `RUN_DIR` if you need the evidence after temporary-directory cleanup. A nonzero exit code, skipped required test, or missing output is not a pass. The merge gate checks the evidence; it does not change Git branches or merge code.

The `v0.1.0-rc.1` snapshot passed the following checks:

| Check | Verified coverage |
|---|---|
| Source | 5,475 source files, six recursive submodules, and two LFS pointers |
| CPU | 37 unittest cases, plus Fig. 7 packet and canonical checks |
| GPU package regression | Eight packages, 150 views each; Geo33 native/storage, Fig. 7, HAC, and HAC++ |
| Instance integration | One synthetic nonempty instance; QAT gradients and independent three-view decoding |
| Recovery | Fresh recursive checkout followed by a 150-view package decode |

The eight package regressions reproduced their reference PSNR, SSIM, and LPIPS. The protocol is **150 train / 150 reconstruction views, using the same views**. These are reconstruction and migration checks. Full PAHC training/mask/feature/compression execution and native HAC/HAC++ short training remain outside this snapshot's completed validation. Synthetic instance tests do not establish real-scene template-sharing gains or owner-aware joint training.

## Development

Start a `fix/`, `feature/`, or `repro/` worktree from a verified baseline. Commit dependency changes before updating the parent gitlinks, refresh the source manifest, and validate the final commit before merging. Released tags remain immutable.

See [versioning and recovery](docs/VERSIONING.md), [validation policy](docs/VALIDATION_POLICY.md), and the [bug report](docs/templates/BUG_REPORT.md) / [change report](docs/templates/CHANGE_REPORT.md) templates. These detailed maintenance documents are currently in Chinese.

## Acknowledgements and License

This project builds on [SegAnyGAussians](third_party/SegAnyGAussians), [HAC](third_party/HAC), and [HAC++](third_party/HAC-plus), together with their Gaussian splatting, segmentation, and compression dependencies.

The parent repository includes an [Apache-2.0 license](LICENSE). Third-party code and model weights retain their respective licenses; consult each dependency before reuse.
