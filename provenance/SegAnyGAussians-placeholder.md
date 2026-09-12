# SegAnyGaussians Backend Placeholder

HPS3DGS uses SegAnyGaussians as the third-party 3DGS, semantic feature,
scene loading, and rendering backend.

Recommended setup:

```bash
git submodule update --init --recursive third_party/SegAnyGAussians
```

Alternatively, point to an existing checkout:

```bash
export HPS_3DGS_SAGA_ROOT=/path/to/SegAnyGAussians
```

The backend is not vendored here to avoid publishing checkpoints, generated
outputs, caches, and other experimental artifacts.
