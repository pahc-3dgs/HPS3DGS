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

Or use an existing checkout:

```bash
export PAHC3DGS_SAGA_ROOT=/path/to/SegAnyGAussians
```

The PAHC-specific code lives in `pahc_3dgs/`; third-party code is kept behind
`pahc_3dgs/backend.py`.

## Installation

```bash
conda activate sacgs
pip install -r requirements.txt
pip install -e .
```

Install SegAnyGaussians and its CUDA rasterization dependencies following the
upstream instructions when running full 3DGS training/rendering.

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

The final stage follows the complete `geo32.py` schedule: semantic-spatial
clustering, physics-aware matching, FPFH/RANSAC initialization, pose rejection,
seven M-step/E-step refinement cycles, target removal, basis quantization,
compact decoding, and unquantized and quantized rendering evaluation.

`run_pahc_pipeline.py` is the public end-to-end entry point. `compress.py`,
`decode.py`, and `evaluate.py` remain as focused tools for debugging and
inspection.

The appearance configuration controls the quantized attributes and codebook
sizes. As in `geo32.py`, PAHC applies one-shot KMeans quantization to the
retained basis under `torch.no_grad()`. Its reported quantized metric uses the
quantized basis together with the unquantized instance payloads created during
geometry refinement.

See `docs/TEST_ZXA1_12.md` for a concrete end-to-end test using the bundled
`/disk3/ydz/data/zxa1-12` dataset.

## Compact Representation

`pahc_3dgs.codec.CompactScene` stores:

- `templates`: Gaussian tensors retained as meta templates.
- `instances`: template id, instance id, rotation, translation, and optional scale.
- `codebooks`: appearance codebook centers and index maps.
- `metadata`: version, statistics, and experiment notes.

## Acknowledgements

This implementation builds on common 3DGS research infrastructure. It uses
SegAnyGaussians as a third-party backend and reorganizes the KMeans appearance
quantization ideas from CompGS into the PAHC-3DGS appearance layer.
