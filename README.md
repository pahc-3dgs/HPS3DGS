# PAHC-3DGS

Official code organization for **Physics-Aware Hierarchical 3DGS Compression:
An Efficient Representation for Aerospace Objects**.

PAHC-3DGS compresses an unstructured 3D Gaussian Splatting scene into a compact
hierarchical representation. The geometry layer discovers repetitive rigid
components and represents them with shared templates plus instance poses. The
appearance layer quantizes material-like Gaussian attributes with independent
codebooks.

## Project Layout

```text
pahc3dgs/
├── configs/pahc.yaml
├── scripts/
│   ├── run_pahc_pipeline.py
│   ├── compress.py
│   ├── decode.py
│   └── evaluate.py
├── pahc_3dgs/
│   ├── backend.py
│   ├── data.py
│   ├── compression.py
│   ├── math_utils.py
│   ├── quantization.py
│   ├── codec.py
│   └── metrics.py
├── third_party/SegAnyGAussians/
└── tests/
```

## Third-Party Backend

PAHC-3DGS uses SegAnyGaussians as the third-party backend for standard 3DGS,
semantic feature training, scene loading, and rendering.

```bash
git submodule update --init --recursive third_party/SegAnyGaussians
```

## Installation

PAHC-3DGS is installed incrementally on top of an existing
SegAnyGaussians (SAGA) environment. First install SAGA, PyTorch, torchvision,
and its CUDA rasterization extensions by following the upstream SAGA
instructions. Then install the PAHC-specific dependencies:

```bash
conda activate sacgs
pip install -r requirements.txt
pip install -e .
```

The `requirements.txt` file intentionally does not install or replace PyTorch,
torchvision, or the SAGA CUDA extensions. This avoids changing the GPU stack in
an environment where SAGA already runs correctly. The current setup has been
verified with Python 3.8.20, PyTorch 2.4.1+cu121, and torchvision 0.19.1+cu121
in the `sacgs` environment.

## Tests

```bash
python -m unittest discover -s tests
```

## Full Pipeline

Use the unified entry point for training, semantic feature extraction, feature
Gaussian training, and PAHC compression:

```bash
conda activate sacgs
cd /path/to/pahc3dgs
mkdir -p outputs logs

set -o pipefail
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
python scripts/run_pahc_pipeline.py \
  -s /path/to/scene \
  -m outputs/scene_3dgs \
  -o outputs/scene.pahc.pt \
  --saga-root /path/to/SegAnyGaussians \
  --sam-checkpoint-path /path/to/sam_vit_h_4b8939.pth \
  2>&1 | tee logs/scene_full_pipeline.log
```

`set -o pipefail` preserves a nonzero exit status when a pipeline stage fails,
while `tee` keeps the complete console output in a log file. Use a new model
directory for each from-scratch run because SAGA writes checkpoints and
configuration files into that directory.

To compress an existing trained SAGA scene, skip the first three stages:

```bash
python scripts/run_pahc_pipeline.py \
  -s /path/to/scene \
  -m /path/to/SegAnyGaussians/output/scene \
  --saga-root /path/to/SegAnyGaussians \
  --skip-train --skip-mask --skip-feature
```

The final stage follows the complete schedule: semantic-spatial
clustering, physics-aware matching, FPFH/RANSAC initialization, pose rejection,
seven M-step/E-step refinement cycles, target removal, basis quantization,
compact decoding, and unquantized and quantized rendering evaluation.

## Compact Representation

`pahc_3dgs.codec.CompactScene` stores:

- `templates`: Gaussian tensors retained as meta templates.
- `instances`: template id, instance id, rotation, translation, and optional scale.
- `codebooks`: appearance codebook centers and index maps.
- `metadata`: version, statistics, and experiment notes.

## Acknowledgements

This implementation builds on common 3DGS research infrastructure. It uses
[SegAnyGaussians](https://github.com/Jumpat/SegAnyGAussians) as a third-party backend and reorganizes the KMeans appearance
quantization ideas from [CompGS](https://github.com/UCDvision/compact3d) into the PAHC-3DGS appearance layer.
