"""Reuse of the audited 2026-09-10 HAC++ camera and PNG evaluation helpers.
The native HAC reader API has been aligned by the all-views reader patch.
Metric formulas, PNG quantization, LPIPS options and FPS timing are unchanged.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import time
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
