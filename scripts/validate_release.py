#!/usr/bin/env python3
"""Validate a committed HPS3DGS candidate and write auditable local evidence.

Profiles are independent: source checks repository identity, cpu runs existing
contract/algebra tests, gpu replays explicitly bound reference packages. Missing
fixtures and failed checks are failures. No source, model or baseline is updated.
"""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import hps3dgs
import release_manifest


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def git(root, *args):
    result = subprocess.run(['git', '-C', str(root)] + list(args), capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('git failed: %s\n%s' % (' '.join(args), result.stderr))
    return result.stdout.strip()


def get(value, key):
    for part in key.split('.'):
        value = value[part]
    return value


def package_state(path):
    """Exact physical bytes and hashes; do not dereference bundle symlinks."""
    path = Path(path)
    if path.is_file() and not path.is_symlink():
        return {'kind': 'file', 'bytes': path.stat().st_size, 'sha256': sha(path)}
    if not path.is_dir() or path.is_symlink():
        raise ValueError('Missing or symlinked package: ' + str(path))
    files = {}
    for item in sorted(path.rglob('*')):
        if item.is_symlink():
            raise ValueError('Bundle contains a symlink: ' + str(item))
        if item.is_file():
            files[item.relative_to(path).as_posix()] = {'bytes': item.stat().st_size, 'sha256': sha(item)}
    if not files:
        raise ValueError('Empty bundle: ' + str(path))
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {'kind': 'directory', 'bytes': sum(v['bytes'] for v in files.values()), 'sha256': digest, 'files': files}


def check_source(root, manifest_path):
    dirty = git(root, 'status', '--porcelain=v1', '--untracked-files=all')
    if dirty:
        raise ValueError('Candidate must be committed and clean:\n' + dirty)
    saved = json.loads(manifest_path.read_text(encoding='utf-8'))
    current = release_manifest.build_manifest(root, manifest_path)
    if current != saved:
        changed = [name for name in set(saved.get('files', {})) | set(current.get('files', {}))
                   if saved.get('files', {}).get(name) != current.get('files', {}).get(name)]
        raise ValueError('Release manifest is stale or incomplete; changed files: ' + repr(sorted(changed)[:30]))
    return saved


def runtime_plan(root, runtime, route, native_args):
    plan = hps3dgs.build_plan(root, runtime, route, native_args)
    if plan['missing_paths']:
        raise FileNotFoundError(json.dumps(plan['missing_paths']))
    env = os.environ.copy()
    env.pop('PYTHONHOME', None)
    env.update(plan['environment_overrides'])
    env['PATH'] = os.pathsep.join(plan['path_prepend'] + [env.get('PATH', '')])
    env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', MPLBACKEND='Agg')
    return plan, env


def execute(name, command, cwd, env, log_root, timeout=1800):
    log = log_root / (name + '.log')
    started = time.time()
    try:
        with log.open('w', encoding='utf-8') as handle:
            process = subprocess.run(command, cwd=str(cwd), env=env, stdout=handle,
                                     stderr=subprocess.STDOUT, timeout=timeout)
        result = {'name': name, 'status': 'passed' if process.returncode == 0 else 'failed',
                  'command': command, 'cwd': str(cwd), 'returncode': process.returncode,
                  'seconds': time.time() - started, 'log': str(log)}
    except subprocess.TimeoutExpired:
        result = {'name': name, 'status': 'failed', 'command': command, 'cwd': str(cwd),
                  'returncode': -1, 'seconds': time.time() - started, 'log': str(log), 'error': 'timeout'}
    if result['status'] != 'passed':
        result['log_tail'] = log.read_text(encoding='utf-8', errors='replace')[-6000:]
    return result


def check_syntax(root, manifest):
    paths = [p for p in manifest['files'] if p.endswith('.py') and not p.startswith('third_party/')]
    imports = json.loads((root / 'provenance/imported_files.json').read_text())
    for group in ('geo33', 'fig7', 'fig7_diagnostics'):
        paths += ['third_party/SegAnyGAussians/' + row['path'] for row in imports[group]]
    for name in sorted(set(paths)):
        ast.parse((root / name).read_text(encoding='utf-8-sig'), filename=name, feature_version=8)
    return {'name': 'managed_python_syntax', 'status': 'passed', 'files': len(set(paths)), 'grammar': 'Python 3.8'}


def cpu_checks(root, runtime, output, fixtures):
    results = []
    unit_runner = str(root / 'scripts/run_unittest_checks.py')
    for name, route, cwd, tail in [
        ('hps_3dgs_contracts', 'hps-3dgs', root, [unit_runner, '--discover', 'tests', '--report', str(output / 'hps-3dgs_unittest.json')]),
        ('geo33_geometry', 'geo33', root / 'third_party/SegAnyGAussians', [unit_runner, '--module', 'test_geo33_geometry_cpu', '--report', str(output / 'geometry_unittest.json')]),
        ('fig7_entrypoints', 'fig7', root / 'third_party/SegAnyGAussians', [unit_runner, '--module', 'test_fig7_entrypoint_bindings', '--report', str(output / 'fig7_unittest.json')]),
        ('fig7_packet', 'fig7', root / 'third_party/SegAnyGAussians', ['test_fig7_packet.py']),
    ]:
        plan, env = runtime_plan(root, runtime, route, [])
        env['CUDA_VISIBLE_DEVICES'] = ''
        results.append(execute(name, [plan['command'][0], '-B'] + tail, cwd, env, output))
    if fixtures and fixtures.get('canonical_point'):
        plan, env = runtime_plan(root, runtime, 'fig7', [])
        env['CUDA_VISIBLE_DEVICES'] = ''
        command = [plan['command'][0], '-B', str(root / 'third_party/SegAnyGAussians/test_canonical_fig7.py'),
                   '--point', fixtures['canonical_point']]
        results.append(execute('fig7_canonical', command, plan['cwd'], env, output))
    else:
        results.append({'name': 'fig7_canonical', 'status': 'failed', 'error': 'Missing canonical_point fixture'})
    return results


def gpu_checks(root, runtime, output, fixtures, physical_gpu):
    import fcntl
    if not fixtures or not fixtures.get('cases'):
        raise ValueError('GPU profile requires nonempty reference fixtures')
    required = set(fixtures.get('required_cases', []))
    cases = fixtures['cases']
    names = [case['name'] for case in cases]
    if not required or set(names) != required or len(names) != len(set(names)):
        raise ValueError('Fixture cases must exactly match unique required_cases')
    if any('/' in name or '\\' in name or name in {'.', '..'} for name in names):
        raise ValueError('Unsafe fixture name')
    for case in cases:
        if not case.get('args') or not case.get('result_file') or not case.get('route') or not case.get('format'):
            raise ValueError('Fixture lacks an executable workload/result contract: ' + case['name'])
        for key, tolerance in case.get('tolerance', {}).items():
            if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance < 0:
                raise ValueError('Metric tolerance must be finite and nonnegative: ' + key)
        for specification in case.get('images', {}).values():
            if type(specification.get('count')) is not int or specification['count'] <= 0 or not specification.get('glob'):
                raise ValueError('Image assertion requires a positive integer count and glob')
        if case.get('package'):
            required_keys = ('expected_package', 'expected_metrics', 'metrics_key', 'frames_key', 'expected_frames', 'images')
            if any(not case.get(key) for key in required_keys) or set(case['expected_metrics']) != {'psnr', 'ssim', 'lpips'}:
                raise ValueError('Package fixture lacks metric/frame/image assertions: ' + case['name'])
            if case['expected_frames'] != 150:
                raise ValueError('Published package regression requires all 150 reconstruction views')
            if any(spec.get('count') != 150 or not spec.get('glob') for spec in case['images'].values()):
                raise ValueError('Package image assertions must require 150 saved frames')
            if not all(math.isfinite(float(v)) for v in case['expected_metrics'].values()):
                raise ValueError('Nonfinite expected metrics')
        elif not case.get('required_fields') or not case.get('images'):
            raise ValueError('Synthetic fixture lacks explicit assertions: ' + case['name'])
    lock = open('/tmp/gszip_all150_gpu_%d.lock' % physical_gpu, 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    query = subprocess.run(['nvidia-smi', '-i', str(physical_gpu), '--query-gpu=memory.used',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    used = int(query.stdout.strip())
    if used >= 100:
        raise RuntimeError('Selected GPU is in use (%d MiB); select another idle GPU' % used)
    results = []
    # Hold the common per-GPU lock for this profile, including subprocess execution.
    for case in cases:
        case_root = output / case['name']
        case_root.mkdir()
        decoded = case_root / 'decoded'
        try:
            expected = case.get('expected_package')
            package = Path(case['package']) if case.get('package') else None
            original = package_state(package) if package else None
            if package:
                if expected is None or original != expected:
                    raise ValueError('Reference package identity/bytes differ from fixture: ' + str(package))
                # A package is copied alone: no trained model, encoded sidecar or
                # previous rendering output is exposed through its new directory.
                copied = case_root / ('package' if package.is_dir() else package.name)
                if package.is_dir():
                    shutil.copytree(package, copied)
                else:
                    shutil.copy2(package, copied)
                if package_state(copied) != original:
                    raise ValueError('Package-only copy is not byte identical')
                package = copied
            substitutions = {'root': str(root), 'output': str(decoded), 'package': str(package or ''),
                             'dataset': case.get('dataset', ''), 'raw': case.get('raw', ''),
                             'source_manifest': str(root / 'configs/fig7_sources.4090.json')}
            raw = case.get('raw')
            if raw and sha(raw) != case.get('raw_sha256'):
                raise ValueError('Synthetic fixture raw PLY identity mismatch')
            native_args = [value.format_map(substitutions) for value in case['args']]
            plan, env = runtime_plan(root, runtime, case['route'], native_args)
            command = plan['command']
            if case.get('script'):
                script = root / case['script']
                if root.resolve() not in script.resolve().parents:
                    raise ValueError('Fixture script escapes release root')
                command = [command[0], '-u', '-B', str(script)] + native_args
            env['CUDA_VISIBLE_DEVICES'] = str(physical_gpu)
            execution = execute(case['name'], command, plan['cwd'], env, case_root, case.get('timeout', 1800))
            results.append(execution)
            if execution['status'] != 'passed':
                continue
            record_path = decoded / case['result_file']
            record = json.loads(record_path.read_text(encoding='utf-8'))
            measured = get(record, case['metrics_key']) if case.get('metrics_key') else {}
            differences = {}
            for key, target in case.get('expected_metrics', {}).items():
                value = float(measured[key])
                if not math.isfinite(value):
                    raise ValueError('Nonfinite metric ' + key)
                differences[key] = value - target
                if abs(differences[key]) > case.get('tolerance', {}).get(key, 1e-4):
                    raise ValueError('Metric regression %s: measured %.10g expected %.10g' % (key, value, target))
            for key, value in case.get('required_fields', {}).items():
                if get(record, key) != value:
                    raise ValueError('Report field mismatch: ' + key)
            frames = get(record, case['frames_key']) if case.get('frames_key') else None
            if case.get('frames_key') and (type(frames) is not int or frames != case['expected_frames']):
                raise ValueError('Wrong reconstruction frame count')
            image_counts = {}
            for label, specification in case.get('images', {}).items():
                found = [path for path in decoded.glob(specification['glob']) if path.is_file()]
                image_counts[label] = len(found)
                if len(found) != specification['count']:
                    raise ValueError('Saved image count mismatch: ' + label)
            if package and package_state(package) != original:
                raise ValueError('Decoder modified the isolated package')
            if case.get('package') and package_state(Path(case['package'])) != original:
                raise ValueError('Reference package changed during validation')
            results.append({'name': case['name'] + '_output', 'status': 'passed',
                            'case': case['name'], 'format': case['format'], 'scene': case.get('scene'),
                            'result_file': str(record_path), 'metrics': measured, 'metric_differences': differences,
                            'frames': frames, 'image_counts': image_counts, 'package_bytes': original['bytes'] if original else None,
                            'package_only_copy': bool(package), 'synthetic_fixture': bool(case.get('raw'))})
        except Exception as exc:
            results.append({'name': case['name'] + '_output', 'status': 'failed', 'error': str(exc), 'traceback': traceback.format_exc()})
        print(json.dumps({'case': case['name'], 'latest': results[-1]['status']}), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['source', 'cpu', 'gpu'], required=True)
    parser.add_argument('--runtime', type=Path, default=Path('configs/runtime.4090.json'))
    parser.add_argument('--fixtures', type=Path, default=Path('configs/validation.4090.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0, help='Explicit physical GPU used only by gpu profile')
    parser.add_argument('--release-root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.release_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    manifest_path = root / 'release_manifest.json'
    report = {'schema_version': 1, 'profile': args.profile, 'status': 'failed',
              'release_root': str(root), 'root_head': git(root, 'rev-parse', 'HEAD'),
              'code_manifest_sha256': sha(manifest_path) if manifest_path.is_file() else None,
              'repositories': {}, 'checks': [], 'output': str(output),
              'protocol_note': '150 train / same 150 reconstruction views; not held-out. Synthetic fixture is separate.'}
    try:
        manifest = check_source(root, manifest_path)
        report['repositories'] = manifest['repositories']
        report['checks'].append({'name': 'committed_source_identity', 'status': 'passed', 'files': len(manifest['files'])})
        report['checks'].append(check_syntax(root, manifest))
        runtime = args.runtime if args.runtime.is_absolute() else root / args.runtime
        fixture_path = args.fixtures if args.fixtures.is_absolute() else root / args.fixtures
        report['configuration_inputs'] = {
            'runtime': {'path': str(runtime.resolve()), 'sha256': sha(runtime)} if runtime.is_file() else None,
            'fixtures': {'path': str(fixture_path.resolve()), 'sha256': sha(fixture_path)} if fixture_path.is_file() else None,
        }
        fixtures = json.loads(fixture_path.read_text(encoding='utf-8')) if fixture_path.is_file() else None
        if args.profile == 'cpu':
            report['checks'].extend(cpu_checks(root, runtime, output, fixtures))
        elif args.profile == 'gpu':
            report['checks'].extend(gpu_checks(root, runtime, output, fixtures, args.gpu))
        check_source(root, manifest_path)
        for value in report['configuration_inputs'].values():
            if value is not None and sha(value['path']) != value['sha256']:
                raise ValueError('Validation configuration changed during execution')
        if git(root, 'rev-parse', 'HEAD') != report['root_head'] or sha(manifest_path) != report['code_manifest_sha256']:
            raise ValueError('Candidate identity changed while validation ran')
        report['checks'].append({'name': 'source_unchanged_after_validation', 'status': 'passed'})
        report['status'] = 'passed' if all(check['status'] == 'passed' for check in report['checks']) else 'failed'
    except Exception as exc:
        report['checks'].append({'name': 'validation_error', 'status': 'failed', 'error': str(exc), 'traceback': traceback.format_exc()})
    report['seconds'] = time.time() - started
    write(output / 'report.json', report)
    print(json.dumps({'profile': args.profile, 'status': report['status'], 'root_head': report['root_head'],
                      'checks': len(report['checks']), 'report': str(output / 'report.json')}), flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
