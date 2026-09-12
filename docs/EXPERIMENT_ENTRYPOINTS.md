# Experiment entry points in HPS3DGS

These four entry points migrate verified experiment-side scripts into the release. They preserve the existing algorithm, codec representation and evaluation protocol. This migration does not train models or claim new reconstruction results.

## Code and input identity

All entry points accept `--release-root` (default: this release root) and `--code-manifest` (default: `release_manifest.json` at that root). The manifest must contain `files: {"release-relative/path": "sha256"}`. Each entry point checks the complete file map and its required source files before execution, rejects paths resolving outside the release, and rechecks code and manifest identity before successful completion. The publisher generates this manifest after staging changes; it must not list itself as a file whose hash it contains.

The PAHC backend comes from the release root; Geo33/SAGA and native HAC++ come from `third_party/SegAnyGAussians` and `third_party/HAC-plus`. Historical manifests remain read-only provenance inputs. Their old `repos`/`worktrees` code paths are never used to locate executable source.

Every `--output` must name a directory that does not exist. Input datasets, model/job directories and source files are not overwritten. Data bindings may still name existing immutable data or model files outside this release; those are inputs rather than executable dependencies. Physical GPU allocation remains the launcher's responsibility. Geo33 retains the existing per-GPU file lock and idle-memory check.

## Available commands

| Entry point | Required arguments | Additional arguments and purpose |
|---|---|---|
| `scripts/experiments/geo33_multiscene.py` | `--source-manifest`, `--scene`, `--gpu`, `--python`, `--output` | `--binding` selects an explicit aligned scene binding. `--geometry-plan` requires `--geometry-plan-sha256`; without this pair, the scene performs its own reference/prepare stages. The previous implicit zxa1-12 path is removed. |
| `scripts/experiments/align_acrim_source.py` | `--source-manifest`, `--camera-comparison`, `--source-cameras`, `--output` | `--source-cameras-sha256` optionally checks the expected original camera file. The source manifest supplies the `acrim-2` input binding. NumPy and plyfile are imported after argument parsing. |
| `scripts/experiments/hacpp_bundle_only_render.py` | `--bundle`, `--dataset`, `--output` | `--tmc3-bin` prepends the verified GPCC binary directory; `--gpu` optionally sets physical GPU visibility, otherwise inherited visibility is retained. Run with the verified HAC++ decoder Python runtime. |
| `scripts/experiments/hacpp_validate_backend.py` | `--job`, `--python`, `--output` | `--job` is an existing read-only directory containing `job.json` and `model/`. `--tmc3-bin` and `--gpu` have the same meanings as above. All new encode/pack/decode/render evidence goes under `--output`. |

For all four, `--help` works without GPU packages or Linux `fcntl`. The native workloads still require their original runtime dependencies.

## Preserved behavior

- Geo33 retains the source SHA checks, one fixed geometry plan, manifest-defined uniform K values, 10,000 QAT forwards, 150-view independent native/storage decoding, native package identity checks and fixed compact storage settings. Geometry plan SHA and binding/manifests are recorded and rechecked. Failed configurations still produce failure evidence; the migrated command additionally exits nonzero instead of returning process success after a recorded failure.
- Acrimsat alignment still solves a positive uniform scale and translation from 150 corresponding camera positions, requires identical rotations/intrinsics and residual below `1e-8`, transforms xyz and adds `log(scale)` to log-scale, and checks unchanged fields and semantic/raw xyz equality. Only input/output locations and provenance handling changed.
- HAC++ independent rendering retains `hash.b` installation **before** native entropy decoding and reconstruction of the fixed `[1, 0, 0, 0]` anchor-culling quaternion after decoding. It never loads source model/checkpoint state. The original native entropy decoder, hash tensor layout and MLP are unchanged.
- HAC++ remains at voxel size `0.005`, decoded model state, 150 training cameras and the same 150 reconstruction cameras, 1280x720, native background, `png_uint8_equivalent` PSNR/SSIM and LPIPS(VGG, `normalize=False`). The reader still refuses missing dataset PLYs that would trigger input mutation.
- HAC++ validation preserves all previous bundle, metric-difference (`1e-4`), camera, GT, PNG SHA and physical-byte checks, and checks the initialization PLY SHA plus source job/model metadata before/after. It invokes the migrated independent renderer from this release. Results explicitly declare `validation_only=true`; `all_model_training_fresh` is no longer unconditionally asserted by a validation-only operation and is copied from explicit job metadata when present (otherwise false).

## Original sources

The local source copies below were read and hashed during this migration. The four corresponding remote files were also directly SHA-checked during the 2026-09-12 code inventory, with identical values.

| New file | Original remote source | Original SHA256 |
|---|---|---|
| `geo33_multiscene.py` | `/disk3/ydz/experiments/gszip_multiscene_20260909/run_geo33_scene.py` | `dd57d26be88c86d8123305f1551e457a2a122ef095cad7a876356301bf3b7671` |
| `align_acrim_source.py` | `/disk3/ydz/experiments/gszip_multiscene_20260909/align_acrim_source.py` | `8d6174d918d76600acc11d45b77f6c95b3f9176153a3f2f01fabb24892dea957` |
| `hacpp_bundle_only_render.py` | `/disk3/ydz/experiments/gszip_pahc_hacpp_rd_20260910/bundle_only_render.py` | `46070af882243ef2b33290913a7af9a645da6e793a41ef618bb587d54a5f0272` |
| `hacpp_validate_backend.py` | `/disk3/ydz/experiments/gszip_pahc_hacpp_rd_20260910/validate_backend.py` | `54befb2a8507037e237eaf8eaaca721005eaa7d883c95464cedef310a52d40aa` |

Local original copies are in `.rd-multiscene-20260909/` and `.pahc-hacpp-rd-20260910/` under the gsZip workspace. They were not modified.

## Migration validation

On 2026-09-12, all four files parsed using Python 3.8 grammar and each `--help` returned exit code 0 under local Python 3.14.3. No hardcoded `/disk3/ydz/` path remains in executable source. AST comparisons against the original sources passed for:

- HAC++ `load_cameras`, `render_saved`, `restore_hash_before_entropy_decode`;
- the full Geo33 K-loop body;
- Acrimsat camera transformation math and the PLY xyz/log-scale transformation loop.

The integration workspace records these checks and source/output hashes in `experiment_entrypoint_validation.json`. No GPU experiment, training, codec decode or rendering was run by this migration subtask. Remote launch requires the published manifest, the pinned component source trees, the existing SAGA/Geo33 runtime (including compact3d and RGB/depth extensions), or the verified HAC++ runtime with its arithmetic/gridencoder/rasterizer extensions and GPCC binary. Full runtime acceptance belongs to the release validation, not to `--help` success.
