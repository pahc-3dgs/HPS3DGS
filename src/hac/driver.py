"""Native HAC encode and fresh-process decode. HAC++ helpers are reused only
for config loading, safe torch loading and shared architecture defaults.
HAC-specific batch metadata, quantization, bounds and decoder shapes are explicit.
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path
from .bundle import FORMAT, sha, validate_config, write_json
from ..hacpp.driver import ENCODING_CONFIG, _dataset_args, _torch_load, _prepend_hacpp_root

MODEL_FIELDS=('feat_dim','n_offsets','voxel_size','update_depth','update_init_factor',
              'update_hierachy_factor','use_feat_bank')

def modules(pc):
    result={'opacity_mlp':pc.mlp_opacity,'cov_mlp':pc.mlp_cov,
            'color_mlp':pc.mlp_color,'grid_mlp':pc.mlp_grid}
    if pc.use_feat_bank:
        result['mlp_feature_bank']=pc.mlp_feature_bank
    # HAC's deform_mlp is unused by renderer/entropy coding and excluded by its
    # own get_mlp_size. HAC++'s identically named module IS an entropy predictor.
    return result

def build_model(config, recover_quantized_checkpoint=False):
    validate_config(config)
    import torch
    from scene.gaussian_model import GaussianModel
    model_class=GaussianModel
    if recover_quantized_checkpoint:
        class DecodedCheckpointHac(GaussianModel):
            @property
            def get_quantized_v(self):
                # Native training uses floor on continuous coordinates. Here the
                # checkpoint ALREADY stores decoded lattice coordinates: recover
                # their integer indices, avoiding a second floor and 1-bin drift.
                from utils.encodings import Q_anchor
                interval=(self.x_bound_max-self.x_bound_min)*Q_anchor+1e-6
                values=torch.round((self._anchor-self.x_bound_min)/interval)
                active=self.get_mask_anchor
                reconstructed=values*interval+self.x_bound_min
                if not torch.allclose(reconstructed[active],self._anchor[active],atol=2e-6,rtol=1e-6):
                    raise ValueError('Checkpoint anchors are not on the saved HAC quantization lattice')
                return values.clamp(0,65535)
        model_class=DecodedCheckpointHac
    return model_class(**config['architecture']).to('cuda:0')

def load_model(model_path, dataset_path):
    import torch
    model_path=Path(model_path)
    dataset,_,_=_dataset_args(str(model_path),str(dataset_path))
    architecture={k:getattr(dataset,k) for k in MODEL_FIELDS}
    architecture.update(ENCODING_CONFIG)
    architecture.update(decoded_version=True,Q=1)
    native=model_path/'bitstreams'
    patched_infos=json.loads((native/'patched_infos.json').read_text())
    config=dict(format=FORMAT,backend='HAC',architecture=architecture,patched_infos=patched_infos,
                white_background=bool(dataset.white_background),all_views_train_test=bool(dataset.all_views_train_test),
                shared_decoder_parameter_bytes=0,native_mlp_parameter_bytes=0)
    pc=build_model(config,recover_quantized_checkpoint=True)
    iterations=sorted(int(p.name.split('_')[1]) for p in (model_path/'point_cloud').glob('iteration_*'))
    iteration=iterations[-1]
    base=model_path/'point_cloud'/('iteration_%d'%iteration)
    pc.load_ply_sparse_gaussian(str(base/'point_cloud.ply'))
    pc.load_mlp_checkpoints(str(base/'checkpoint.pth'))
    for attr in ('x_bound_min','x_bound_max'):
        setattr(pc,attr,_torch_load(str(native/(attr+'.pkl')),'cuda:0'))
    pc.eval()
    assert pc.get_anchor.shape[0]==patched_infos[0]
    config['source_iteration']=iteration
    config['native_mlp_parameter_bytes']=int(pc.get_mlp_size()[0]//8)
    return pc,config

def encode(model_path,dataset_path,out_dir):
    import numpy as np
    import torch
    out=Path(out_dir)
    if out.exists():
        raise FileExistsError('Refusing to overwrite encoder output: '+str(out))
    pc,config=load_model(model_path,dataset_path)
    native=Path(model_path)/'bitstreams'
    # Verify recovery against independently persisted native integer coordinates.
    q=pc.get_quantized_v[pc.get_mask_anchor].detach().cpu().numpy().astype(np.uint16)
    reference=np.load(native/'_quantized_v.npy',allow_pickle=False)
    if not np.array_equal(q,reference):
        raise ValueError('Recovered checkpoint grid differs from original native anchor stream')
    out.mkdir(parents=True)
    started=time.perf_counter()
    info,log=pc.conduct_encoding(str(out))
    seconds=time.perf_counter()-started
    if info!=config['patched_infos']:
        raise ValueError('Re-encoded active anchor counts changed')
    write_json(out/'patched_infos.json',info)
    for attr in ('x_bound_min','x_bound_max'):
        torch.save(getattr(pc,attr).detach().cpu(),str(out/(attr+'.pkl')))
    state={k:module.state_dict() for k,module in modules(pc).items()}
    config['shared_decoder_parameter_bytes']=sum(t.numel()*t.element_size() for m in state.values() for t in m.values())
    assert config['shared_decoder_parameter_bytes']==config['native_mlp_parameter_bytes']
    torch.save(state,str(out/'shared_mlp.pt'))
    write_json(out/'decoder_config.json',config)
    comparisons={p.name:sha(p)==sha(native/p.name) for p in out.iterdir()
                 if p.suffix in ('.b','.npy') and (native/p.name).is_file()}
    return dict(ok=True,backend='HAC',seconds=seconds,log=log,patched_infos=info,
                reencoded=True,recovered_integer_anchors_equal_native=True,
                native_stream_byte_equal=comparisons,
                config=config,shared_decoder_keys=sorted(state))

def restore_hash(pc,raw):
    """Reuse the proven HAC++ hash preload procedure (same grid layout in HAC)."""
    import torch
    from utils.encodings_cuda import decoder
    count=pc.get_encoding_params().numel()
    values=(decoder(count,str(raw/'hash.b'))*2-1).float().view(-1,pc.n_features_per_level)
    if pc.use_2D:
        xyz=pc.encoding_xyz.encoding_xyz
        planes=[pc.encoding_xyz.encoding_xy,pc.encoding_xyz.encoding_xz,pc.encoding_xyz.encoding_yz]
        n3,n2=xyz.params.shape[0],planes[0].params.shape[0]
        assert values.shape[0]==n3+3*n2
        xyz.params=torch.nn.Parameter(values[:n3],requires_grad=False)
        for i,plane in enumerate(planes):
            plane.params=torch.nn.Parameter(values[n3+i*n2:n3+(i+1)*n2],requires_grad=False)
    else:
        pc.encoding_xyz.params=torch.nn.Parameter(values,requires_grad=False)
    return count

def decode(raw):
    """Decode from bundle-owned state only; no model or checkpoint argument."""
    import torch
    raw=Path(raw)
    config=validate_config(json.loads((raw/'decoder_config.json').read_text()))
    pc=build_model(config)
    state=_torch_load(str(raw/'shared_mlp.pt'),'cuda:0')
    required=modules(pc)
    if set(state)!=set(required):
        raise ValueError('Shared HAC decoder module mismatch')
    for key,module in required.items():
        module.load_state_dict(state[key],strict=True)
    for attr in ('x_bound_min','x_bound_max'):
        value=_torch_load(str(raw/(attr+'.pkl')),'cuda:0')
        if value.numel()!=3 or not torch.isfinite(value).all():
            raise ValueError('Invalid HAC training bounds')
        setattr(pc,attr,value)
    if not torch.all(pc.x_bound_max>pc.x_bound_min):
        raise ValueError('Invalid bound order')
    nfull,nactive,_=config['patched_infos']
    # HAC checks target tensor shapes before installing decoded arrays.
    for name,shape in {'_anchor':(nfull,3),'_anchor_feat':(nfull,pc.feat_dim),
                       '_offset':(nfull,pc.n_offsets,3),'_scaling':(nfull,6),
                       '_mask':(nfull,pc.n_offsets,1)}.items():
        setattr(pc,name,torch.nn.Parameter(torch.zeros(shape,device='cuda:0'),requires_grad=False))
    count=restore_hash(pc,raw)
    started=time.perf_counter()
    log=pc.conduct_decoding(str(raw),config['patched_infos'])
    seconds=time.perf_counter()-started
    rotation=torch.zeros((nfull,4),device='cuda:0');rotation[:,0]=1
    pc._rotation=torch.nn.Parameter(rotation,requires_grad=False)
    pc.eval()
    assert int(pc.get_mask_anchor.sum())==nactive
    for name in ('_anchor','_anchor_feat','_offset','_scaling','_mask'):
        if not torch.isfinite(getattr(pc,name)).all():
            raise ValueError('Nonfinite decoded field: '+name)
    return pc,config,dict(ok=True,backend='HAC',self_contained_decode=True,
                         source_model_used=False,source_checkpoint_used=False,
                         patched_infos=config['patched_infos'],hash_preloaded=True,
                         hash_values=count,decode_seconds=seconds,log=log)
