# HAC++ backend status

This branch implements a Phase-0 PAHC-to-HAC++ adapter and an auditable codec
contract. It does **not** yet implement owner-aware HAC++ training.

## What is implemented

- PAHC scenes save, reload, and decode before their reported render evaluation.
- Standard 3DGS scale parameters are transformed in log space.
- Instance `features_dc` and higher-order SH are explicit residual/full/lossy
  streams; reported bitstream size is the actual saved-file size.
- `scripts/hacpp_backend.py export` writes `hacpp_init.ply`, `hacpp_init.npz`,
  `owners.npz`, and `init_config.json` for a new HAC++ student run.
- The HAC++ bridge runs in a subprocess to avoid top-level Python-package
  collisions with SegAnyGaussians. The bundle stores one `shared_mlp.pt`; the
  hash grid remains only in `hash.b`.

## Important boundary

`owners.npz` describes the exported initialization rows only. Stock HAC++
densification and pruning do not propagate PAHC owner IDs, so the mapping must
not be called stable after training. A manifest claiming `phase=stable` is
rejected unless it includes owner-propagation provenance. True joint
owner-aware HAC++ training is a later backend phase.

The checked-in `third_party/SegAnyGaussians` path is only a placeholder, even
though `.gitmodules` mentions it. Point PAHC at a real checkout using
`--saga-root /path/to/SegAnyGaussians` or `PAHC3DGS_SAGA_ROOT`.

## Commands

```bash
python scripts/hacpp_backend.py export \
  --basis outputs/scene.pahc.pt \
  --out-dir outputs/scene_hacpp_init \
  --voxel-size 0.005

python scripts/hacpp_backend.py verify --out-dir outputs/scene_hacpp_init

python scripts/hacpp_backend.py \
  --hacpp-root /path/to/HAC-plus \
  --python /path/to/hacpp/python \
  inspect
```

Train the HAC++ student from the generated `hacpp_init.ply` using the external
HAC++ checkout's `train.py --init_ply ...`. A SAGA/3DGS checkpoint cannot be
restored directly into `HAC++ GaussianModel`: anchors, offsets, learned
features, MLPs, masks, and the hash grid use a different parameterization.

Run CPU contract tests with:

```bash
python -m unittest discover -s tests -v
```
