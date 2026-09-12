"""Portable HAC streams with a single decoder and measured storage.

Uses the existing HAC++ bundle's checksum and fixed-point accounting design,
with a distinct HAC format. No owner-aware training is claimed by this format.
"""
import hashlib
import json
import re
import shutil
from pathlib import Path

FORMAT = 'pahc-hac-codec-only-v1'
EXACT = {'_quantized_v.npy', 'hash.b', 'masks.b', 'x_bound_min.pkl',
         'x_bound_max.pkl', 'patched_infos.json', 'shared_mlp.pt', 'decoder_config.json'}
CODED = re.compile(r'^(feat|scaling|offsets)_\d+(?:_\d+)?\.b$')

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()

def write_json(path, payload):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)+'\n')
    temp.replace(path)

def validate_config(config):
    if config.get('backend') != 'HAC' or config.get('format') != FORMAT:
        raise ValueError('Expected the HAC codec format; HAC++ models are incompatible')
    for key in ('architecture', 'patched_infos', 'white_background', 'all_views_train_test',
                'shared_decoder_parameter_bytes', 'native_mlp_parameter_bytes'):
        if key not in config:
            raise ValueError('Missing decoder configuration: '+key)
    nfull, nactive, batch = config['patched_infos']
    if not (isinstance(nfull,int) and isinstance(nactive,int) and isinstance(batch,int)
            and 0 < nactive <= nfull and batch==3000):
        raise ValueError('Invalid HAC patched_infos')
    arch = config['architecture']
    if not arch.get('decoded_version') or not arch.get('ste_binary'):
        raise ValueError('Portable HAC requires decoded state and binary hash coding')
    if 'is_synthetic_nerf' in arch:
        raise ValueError('HAC++-only constructor field in HAC configuration')
    return config

def classify(raw):
    raw = Path(raw)
    files = {p.name:p for p in raw.iterdir()}
    if any(not p.is_file() or p.is_symlink() for p in files.values()):
        raise ValueError('Encoder output must contain regular files only')
    unknown = set(files)-EXACT-{n for n in files if CODED.fullmatch(n)}
    if unknown or EXACT-set(files):
        raise ValueError('Invalid HAC streams: missing=%r unknown=%r' % (EXACT-set(files),unknown))
    config = validate_config(json.loads(files['decoder_config.json'].read_text()))
    info = json.loads(files['patched_infos.json'].read_text())
    if info != config['patched_infos']:
        raise ValueError('Batch metadata differs from decoder configuration')
    steps = (info[1]+info[2]-1)//info[2]
    for family in ('feat','scaling','offsets'):
        for step in range(steps):
            if not any(re.fullmatch(family+'_'+str(step)+r'(?:_\d+)?\.b', n) for n in files):
                raise ValueError('Missing HAC arithmetic batch: %s %s' % (family,step))
    return files, config

def pack(raw, bundle, scene_id):
    files, config = classify(raw)
    bundle = Path(bundle)
    if bundle.exists():
        raise FileExistsError('Refusing to overwrite bundle: '+str(bundle))
    (bundle/'hac').mkdir(parents=True)
    for name,path in files.items():
        shutil.copyfile(path,bundle/'hac'/name)
    index = {'hac/'+name:dict(bytes=path.stat().st_size,sha256=sha(path))
             for name,path in files.items()}
    stream_names = {n for n in files if n not in {'shared_mlp.pt','decoder_config.json'}}
    coded_bytes = sum(p.stat().st_size for n,p in files.items() if n.endswith('.b'))
    # Native HAC counts 16-bit coordinates, arithmetic files and raw MLP tensors;
    # it omits the .npy header, bounds, batch metadata and packaging overhead.
    native_bytes = config['patched_infos'][1]*6+coded_bytes+config['native_mlp_parameter_bytes']
    storage = dict(codec_stream_bytes=sum(files[n].stat().st_size for n in stream_names),
                   shared_decoder_bytes=files['shared_mlp.pt'].stat().st_size,
                   shared_decoder_parameter_bytes=config['shared_decoder_parameter_bytes'],
                   native_accounted_bytes=native_bytes, native_accounted_mib=native_bytes/1048576,
                   artifact_bytes=0, artifact_mib='0000000000.000000')
    manifest = dict(format=FORMAT,backend='HAC',codec_only=True,scene_id=scene_id,
                    decoder_kind='hac_shared_v1',shared_decoder='hac/shared_mlp.pt',
                    hash_in_shared_decoder=False,files=index,decoder_config=config,storage=storage,
                    definitions=dict(artifact='All bundle files including manifest and decoder; actual disk bytes',
                                     native_accounted='16-bit xyz payload + arithmetic files + raw required MLP parameters; excludes auxiliary metadata'))
    for _ in range(16):
        write_json(bundle/'manifest.json',manifest)
        actual = sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file())
        if actual == storage['artifact_bytes']:
            break
        storage['artifact_bytes']=actual
    else:
        raise ValueError('Artifact accounting did not converge')
    storage['artifact_mib']='%017.6f'%(actual/1048576)
    write_json(bundle/'manifest.json',manifest)
    return read_bundle(bundle)

def read_bundle(bundle):
    bundle = Path(bundle)
    manifest = json.loads((bundle/'manifest.json').read_text())
    if manifest.get('format')!=FORMAT or manifest.get('backend')!='HAC' or not manifest.get('codec_only'):
        raise ValueError('Incompatible bundle format')
    if manifest.get('hash_in_shared_decoder') is not False:
        raise ValueError('Hash must occur only in the per-scene stream')
    actual_files = {p.relative_to(bundle).as_posix():p for p in bundle.rglob('*') if p.is_file()}
    if set(actual_files)!=set(manifest['files'])|{'manifest.json'}:
        raise ValueError('Bundle inventory mismatch')
    for name,record in manifest['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or not name.startswith('hac/'):
            raise ValueError('Unsafe bundle path: '+name)
        path=actual_files[name]
        if path.is_symlink() or path.stat().st_size!=record['bytes'] or sha(path)!=record['sha256']:
            raise ValueError('Checksum or length mismatch: '+name)
    _,config=classify(bundle/'hac')
    if config!=manifest['decoder_config']:
        raise ValueError('Manifest and decoder configuration differ')
    actual=sum(p.stat().st_size for p in actual_files.values())
    if actual!=manifest['storage']['artifact_bytes']:
        raise ValueError('Artifact byte total mismatch')
    return manifest
