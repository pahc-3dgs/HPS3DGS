#!/usr/bin/env python3
"""Decode and render only a portable HPS3DGS/HAC++ bundle plus its dataset.

No source-model argument, Scene construction, cfg_args, trained point_cloud,
or source checkpoint is used. Camera transforms/loading and codec construction
are reused from the pinned HAC++ and HPS3DGS implementations.
Run in a fresh process with the verified py112 runtime and GPCC on PATH.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
from types import SimpleNamespace


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def snapshot(paths):
    return {str(p): {'bytes': p.stat().st_size, 'sha256': sha256(p)}
            for p in sorted(set(Path(p).resolve() for p in paths)) if p.is_file()}


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def inside(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def release_code(root, manifest_path):
    root, manifest_path = Path(root).resolve(), Path(manifest_path).resolve()
    files = json.loads(manifest_path.read_text()).get('files', {})
    required = {'src/hacpp/driver.py', 'src/hacpp/bundle.py',
                'third_party/HAC-plus/scene/gaussian_model.py',
                'third_party/HAC-plus/scene/dataset_readers.py',
                'third_party/HAC-plus/utils/camera_utils.py',
                'third_party/HAC-plus/gaussian_renderer/__init__.py',
                Path(__file__).resolve().relative_to(root).as_posix()}
    if not files or not required.issubset(files):
        raise ValueError('Release manifest lacks required decoder source files: ' + str(sorted(required - set(files))))
    for name, digest in files.items():
        path = (root / name).resolve()
        if not inside(path, root):
            raise ValueError('Code manifest path escapes release root: ' + name)
        if not path.is_file() or sha256(path) != digest:
            raise ValueError('Release source hash mismatch: ' + name)
    return {'path': str(manifest_path), 'sha256': sha256(manifest_path), 'verified_files': len(files)}


def load_cameras(dataset, config, output):
    """Reuse native readers; require existing PLYs so their fallback never writes."""
    from scene.dataset_readers import sceneLoadTypeCallbacks
    from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

    if (dataset / 'cam_info.yaml').is_file():
        kind = 'UE'
        dataset_ply = dataset / 'gt_2.ply'
        metadata = [dataset / 'cam_info.yaml']
        reader_kwargs = dict(eval=True, all_views_train_test=True)
    elif (dataset / 'sparse').is_dir():
        kind = 'Colmap'
        dataset_ply = dataset / 'sparse/0/points3D.ply'
        metadata = list((dataset / 'sparse/0').glob('*.bin'))
        metadata += list((dataset / 'sparse/0').glob('*.txt'))
        # Match the audited native default. The existing reader detects color.
        reader_kwargs = dict(images='images', eval=True, lod=0, all_views_train_test=True)
    else:
        raise ValueError('Only the audited UE and COLMAP datasets are supported: ' + str(dataset))
    if not dataset_ply.is_file():
        raise FileNotFoundError('Native reader would create a dataset PLY; refusing dataset mutation: ' + str(dataset_ply))
    metadata.append(dataset_ply)
    metadata_before = snapshot(metadata)
    info = sceneLoadTypeCallbacks[kind](str(dataset), **reader_kwargs)
    infos = info.test_cameras
    train_names = [c.image_name for c in info.train_cameras]
    test_names = [c.image_name for c in infos]
    if len(infos) != 150 or len(set(test_names)) != 150 or train_names != test_names:
        raise ValueError('Expected identical 150-unique-view train and test lists, got train=%s test=%s unique=%s'
                         % (len(train_names), len(test_names), len(set(test_names))))
    camera_args = SimpleNamespace(resolution=-1, data_device='cuda')
    cameras = cameraList_from_camInfos(infos, 1.0, camera_args)
    if any((cam.image_width, cam.image_height) != (1280, 720) for cam in cameras):
        raise ValueError('Expected the audited 1280x720 camera resolution for every view')
    camera_rows = [camera_to_JSON(i, c) for i, c in enumerate(infos)]
    write_json(output / 'cameras.json', camera_rows)
    source_images = [Path(c.image_path) for c in infos]
    return cameras, {
        'loader': kind, 'reader_kwargs': reader_kwargs, 'resolution_argument': -1,
        'train_views': len(train_names), 'test_views': len(test_names),
        'unique_test_views': len(set(test_names)), 'train_test_same_order': True,
        'held_out': False, 'resolution_width_height': [1280, 720],
        'camera_names': test_names, 'camera_sha256': sha256(output / 'cameras.json'),
        'dataset_metadata_before': metadata_before,
        'source_images_before': snapshot(source_images),
    }


def render_saved(cameras, gaussians, config, output):
    """Driver._render_psnr's PNG-domain formula, with per-view PNG evidence."""
    import numpy as np
    import torch
    import lpips
    from PIL import Image
    from gaussian_renderer import prefilter_voxel, render
    from utils.image_utils import psnr
    from utils.loss_utils import ssim

    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    background = torch.tensor([1, 1, 1] if config['white_background'] else [0, 0, 0],
                              dtype=torch.float32, device='cuda:0')
    lpips_fn = lpips.LPIPS(net='vgg').to('cuda:0').eval()
    rows, render_seconds = [], []
    for index, camera in enumerate(cameras):
        with torch.no_grad():
            visible_mask = prefilter_voxel(camera, gaussians, pipe, background)
            torch.cuda.synchronize()
            started = time.perf_counter()
            prediction = render(camera, gaussians, pipe, background,
                                visible_mask=visible_mask)['render'].clamp(0, 1)
            torch.cuda.synchronize()
            render_seconds.append(time.perf_counter() - started)
            target = camera.original_image.to('cuda:0').clamp(0, 1)
            if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(target).all()):
                raise ValueError('Nonfinite prediction or GT at view %d' % index)
            # Exact formula in src/hacpp/driver.py::_render_psnr.
            prediction_metric = torch.floor(prediction * 255.0 + 0.5).clamp(0, 255) / 255.0
            target_metric = torch.floor(target * 255.0 + 0.5).clamp(0, 255) / 255.0
            pred_batch, gt_batch = prediction_metric.unsqueeze(0), target_metric.unsqueeze(0)
            metrics = {
                'psnr': float(psnr(pred_batch, gt_batch).mean().item()),
                'ssim': float(ssim(pred_batch, gt_batch).mean().item()),
                'lpips': float(lpips_fn(pred_batch, gt_batch, normalize=False).mean().item()),
            }
        if not all(math.isfinite(v) for v in metrics.values()):
            raise ValueError('Nonfinite image metric at view %d: %r' % (index, metrics))
        filename = '%05d.png' % index
        pixels = {}
        for label, tensor in (('renders', prediction_metric), ('gt', target_metric)):
            arr = torch.round(tensor * 255.0).to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy()
            path = output / label / filename
            Image.fromarray(arr, mode='RGB').save(path)
            pixels[label] = {
                'file': label + '/' + filename, 'png_sha256': sha256(path),
                'rgb_sha256': hashlib.sha256(arr.tobytes()).hexdigest(),
                'shape_hwc': list(arr.shape),
            }
        with Image.open(output / 'renders' / filename) as saved_prediction:
            pred = np.asarray(saved_prediction, dtype=np.float64) / 255.0
        with Image.open(output / 'gt' / filename) as saved_target:
            gt = np.asarray(saved_target, dtype=np.float64) / 255.0
        mse = float(np.mean((pred - gt) ** 2))
        if mse <= 0 or not math.isfinite(mse):
            raise ValueError('Saved PNG does not have a finite PSNR at view %d' % index)
        cpu_psnr = -10.0 * math.log10(mse)
        if abs(cpu_psnr - metrics['psnr']) > 1e-4:
            raise ValueError('Saved PNG PSNR differs from evaluated PNG-domain tensor at view %d' % index)
        rows.append(dict(index=index, image_name=camera.image_name, metrics=metrics,
                         saved_png_cpu_psnr=cpu_psnr, files=pixels,
                         render_seconds=render_seconds[-1]))
        if (index + 1) % 25 == 0:
            print('BUNDLE_ONLY_RENDER %d/150' % (index + 1), flush=True)
    write_json(output / 'per_view_metrics.json', rows)
    timed = render_seconds[5:]
    return {
        'split': 'test', 'views': len(rows), 'views_total_in_split': len(cameras),
        'truncated': False, 'metric_domain': 'png_uint8_equivalent',
        'psnr': sum(r['metrics']['psnr'] for r in rows) / len(rows),
        'ssim': sum(r['metrics']['ssim'] for r in rows) / len(rows),
        'lpips': sum(r['metrics']['lpips'] for r in rows) / len(rows),
        'render_fps': 1.0 / (sum(timed) / len(timed)),
        'fps_warmup_views_excluded': 5,
        'saved_png_cpu_psnr': sum(r['saved_png_cpu_psnr'] for r in rows) / len(rows),
        'max_abs_saved_png_cpu_psnr_difference': max(abs(r['saved_png_cpu_psnr'] - r['metrics']['psnr']) for r in rows),
        'per_view_metrics_sha256': sha256(output / 'per_view_metrics.json'),
        'render_png_count': len(list((output / 'renders').glob('*.png'))),
        'gt_png_count': len(list((output / 'gt').glob('*.png'))),
        'lpips_network': 'vgg', 'lpips_normalize': False,
    }


def restore_hash_before_entropy_decode(gaussians, bitstream):
    """Restore hash.b before legacy conduct_decoding predicts entropy context.

    The native routine reads hash.b early but assigns it only after decoding
    feature/scaling/offset streams. That works when a source checkpoint already
    populated the grid. A fresh decoder must install this bundle-owned context
    first. Both bit decoding and tensor layout are the original HAC++ ones.
    """
    import torch
    from utils.encodings_cuda import decoder
    if not gaussians.ste_binary:
        raise ValueError('This audited portable path requires the binary hash-grid codec')
    count = gaussians.get_encoding_params().numel()
    embeddings = (decoder(count, str(bitstream / 'hash.b')) * 2 - 1).to(torch.float32)
    embeddings = embeddings.view(-1, gaussians.n_features_per_level)
    if gaussians.use_2D:
        grid3d = gaussians.encoding_xyz.encoding_xyz
        grids2d = [gaussians.encoding_xyz.encoding_xy, gaussians.encoding_xyz.encoding_xz,
                   gaussians.encoding_xyz.encoding_yz]
        n3d, n2d = grid3d.params.shape[0], grids2d[0].params.shape[0]
        if embeddings.shape[0] != n3d + 3*n2d:
            raise ValueError('Hash-grid stream shape does not match decoder metadata')
        grid3d.params = torch.nn.Parameter(embeddings[:n3d], requires_grad=False)
        for i, grid in enumerate(grids2d):
            grid.params = torch.nn.Parameter(embeddings[n3d+i*n2d:n3d+(i+1)*n2d], requires_grad=False)
    else:
        gaussians.encoding_xyz.params = torch.nn.Parameter(embeddings, requires_grad=False)
    if gaussians.get_encoding_params().numel() != count:
        raise ValueError('Installed hash-grid parameter count differs from hash.b')
    return int(count)


def execute(args):
    started = time.time()
    args.release_root = args.release_root.resolve()
    args.backend_root = args.release_root
    args.hacpp_root = args.release_root / 'third_party/HAC-plus'
    code_manifest = (args.code_manifest or args.release_root / 'release_manifest.json').resolve()
    code_record = release_code(args.release_root, code_manifest)
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    if args.tmc3_bin:
        os.environ['PATH'] = str(args.tmc3_bin.resolve()) + os.pathsep + os.environ.get('PATH', '')
    bundle, dataset, output = args.bundle.resolve(), args.dataset.resolve(), args.output.resolve()
    if inside(output, dataset) or inside(output, bundle):
        raise ValueError('Output must be outside the input dataset and bundle')
    if output.exists():
        raise FileExistsError('Output directory must be new: ' + str(output))
    if not shutil.which('tmc3'):
        raise RuntimeError('tmc3 missing from PATH; supply --tmc3-bin with its verified runtime directory')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'renders').mkdir()
    (output / 'gt').mkdir()

    sys.path.insert(0, str(args.backend_root.resolve()))
    from src.hacpp.bundle import read_bundle, bundle_artifact_bytes
    from src.hacpp import driver
    manifest = read_bundle(bundle, verify=True)
    bundle_before = snapshot(bundle.rglob('*'))
    config = json.loads((bundle / 'hacpp/decoder_config.json').read_text())
    if not config.get('all_views_train_test') or not config.get('decoded_version'):
        raise ValueError('Bundle must declare all_views_train_test and decoded_version')
    if float(config['voxel_size']) != 0.005:
        raise ValueError('Expected fixed voxel_size=0.005, got %r' % config['voxel_size'])
    driver._prepend_hacpp_root(str(args.hacpp_root.resolve()))
    import numpy as np
    import torch
    driver._check_render_runtime()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(0)
    gaussians = driver.build_gaussians_from_decoder_config(config, 'cuda:0')
    decoder_keys = driver.load_shared_decoder(gaussians, str(bundle / 'hacpp/shared_mlp.pt'), 'cuda:0')
    hash_value_count = restore_hash_before_entropy_decode(gaussians, bundle / 'hacpp')
    decode_log = gaussians.conduct_decoding(pre_path_name=str(bundle / 'hacpp'))
    # Native conduct_decoding reconstructs anchor/features/offsets/scales/masks,
    # but leaves the anchor-culling quaternion empty on a freshly built model.
    # Native create_from_pcd/anchor_growing initialize it to [1,0,0,0]; it is
    # only consumed by the nondifferentiable visible_filter. The actual Gaussian
    # orientation is produced by the decoded covariance MLP. Restore that fixed
    # runtime state from anchor count; never obtain it from a source checkpoint.
    anchor_count = int(gaussians.get_anchor.shape[0])
    culling_rotation = torch.zeros((anchor_count, 4), dtype=torch.float32, device='cuda:0')
    culling_rotation[:, 0] = 1.0
    gaussians._rotation = torch.nn.Parameter(culling_rotation, requires_grad=False)
    gaussians.eval()
    if int(gaussians.get_anchor.shape[0]) <= 0:
        raise ValueError('Bundle decoded to no anchors')
    cameras, protocol = load_cameras(dataset, config, output)
    metrics = render_saved(cameras, gaussians, config, output)
    bundle_after = snapshot(bundle.rglob('*'))
    metadata_after = snapshot(protocol['dataset_metadata_before'])
    images_after = snapshot(protocol['source_images_before'])
    preservation = {
        'bundle_unchanged': bundle_before == bundle_after,
        'dataset_metadata_unchanged': protocol['dataset_metadata_before'] == metadata_after,
        'dataset_images_unchanged': protocol['source_images_before'] == images_after,
    }
    if not all(preservation.values()):
        raise RuntimeError('Input preservation check failed: %r' % preservation)
    if sha256(code_manifest) != code_record['sha256']:
        raise RuntimeError('Release code manifest changed during rendering')
    release_code(args.release_root, code_manifest)
    source_paths = [Path(__file__).resolve(), args.backend_root / 'src/hacpp/driver.py',
                    args.backend_root / 'src/hacpp/bundle.py',
                    args.hacpp_root / 'scene/dataset_readers.py',
                    args.hacpp_root / 'utils/camera_utils.py',
                    args.hacpp_root / 'gaussian_renderer/__init__.py',
                    args.hacpp_root / 'scene/gaussian_model.py']
    result = {
        'ok': True, 'self_contained_decode': True, 'bundle_only_model_state': True,
        'source_model_used': False, 'source_checkpoint_used': False,
        'inputs': {'bundle': str(bundle), 'dataset': str(dataset)},
        'output': str(output), 'decoder_config': config, 'shared_decoder_keys': decoder_keys,
        'decoded': {'anchors': int(gaussians.get_anchor.shape[0]), 'feat_dim': int(gaussians.feat_dim),
                    'n_offsets': int(gaussians.n_offsets), 'voxel_size': float(gaussians.voxel_size)},
        'decode_log': decode_log, 'render_decoded': metrics, 'protocol': protocol,
        'fixed_runtime_state': {'anchor_culling_rotation': [1, 0, 0, 0],
                                'count': anchor_count, 'source_model_read': False},
        'entropy_initialization': {'hash_stream_preloaded_before_entropy_decode': True,
                                   'hash_values': hash_value_count, 'source': 'bundle/hacpp/hash.b',
                                   'hash_stream_sha256': sha256(bundle / 'hacpp/hash.b')},
        'storage': manifest.storage, 'actual_bundle_bytes': bundle_artifact_bytes(bundle),
        'bundle_files_before': bundle_before, 'preservation': preservation,
        'code_sha256': snapshot(source_paths),
        'release_code': code_record,
        'runtime': {'python': sys.version, 'executable': sys.executable, 'torch': torch.__version__,
                    'cuda': torch.version.cuda, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                    'device': torch.cuda.get_device_name(0), 'tmc3': shutil.which('tmc3')},
        'elapsed_seconds': time.time() - started,
    }
    write_json(output / 'bundle_only_render.json', result)
    print(json.dumps({'ok': True, 'output': str(output), 'render_decoded': metrics}, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--release-root', type=Path, default=Path(__file__).resolve().parents[2],
                        help='HPS3DGS release root; native HAC++ is third_party/HAC-plus')
    parser.add_argument('--code-manifest', type=Path, help='Defaults to release-root/release_manifest.json')
    parser.add_argument('--tmc3-bin', type=Path, help='Verified GPCC binary directory to prepend to PATH')
    parser.add_argument('--gpu', type=int, help='Optional physical GPU; otherwise preserve CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()
    execute(args)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
