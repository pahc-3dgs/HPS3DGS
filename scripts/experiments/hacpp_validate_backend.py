"""Reuse HPS3DGS's bridge and bundle contract; verify every fresh RD artifact."""
import argparse,hashlib,json,math,os,subprocess,sys,time
from pathlib import Path
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False))

def release_code(root, manifest_path):
 root=Path(root).resolve();manifest_path=Path(manifest_path).resolve();files=read(manifest_path).get('files',{})
 required={'src/hacpp/bridge.py','src/hacpp/pipeline.py','src/hacpp/bundle.py','src/hacpp/driver.py',
           'third_party/HAC-plus/scene/gaussian_model.py',
           'scripts/experiments/hacpp_bundle_only_render.py',Path(__file__).resolve().relative_to(root).as_posix()}
 if not files or not required.issubset(files):raise ValueError('Release manifest lacks required source files: '+str(sorted(required-set(files))))
 for name,digest in files.items():
  path=(root/name).resolve()
  try:path.relative_to(root)
  except ValueError:raise ValueError('Code manifest path escapes release root: '+name)
  if not path.is_file() or sha(path)!=digest:raise ValueError('Release source hash mismatch: '+name)
 return {'path':str(manifest_path),'sha256':sha(manifest_path),'verified_files':len(files)}

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--job',required=True,type=Path,help='Read-only existing job directory containing job.json and model/')
 p.add_argument('--output',required=True,type=Path,help='New validation output directory; must not exist')
 p.add_argument('--python',required=True,help='Verified Python runtime used by the HAC++ bridge and independent renderer')
 p.add_argument('--release-root',type=Path,default=Path(__file__).resolve().parents[2])
 p.add_argument('--code-manifest',type=Path,help='Defaults to release-root/release_manifest.json')
 p.add_argument('--tmc3-bin',type=Path,help='Verified GPCC binary directory to prepend to PATH')
 p.add_argument('--gpu',type=int,help='Optional physical GPU; otherwise preserve CUDA_VISIBLE_DEVICES')
 a=p.parse_args();root=a.release_root.resolve();code_manifest=(a.code_manifest or root/'release_manifest.json').resolve()
 code_record=release_code(root,code_manifest)
 input_job=a.job.resolve();spec=read(input_job/'job.json');model=input_job/'model';source=spec['input']['dataset'];job=a.output.resolve()
 for protected in [input_job,Path(source).resolve()]:
  try:job.relative_to(protected)
  except ValueError:continue
  raise ValueError('Output must be outside source job and dataset: '+str(protected))
 if job.exists():raise FileExistsError('Output directory already exists: '+str(job))
 init_ply=Path(spec['input']['init_ply']);assert sha(init_ply)==spec['input']['init_sha256'],'Initialization PLY source changed'
 source_inputs=[input_job/'job.json',model/'results.json',model/'cameras.json',model/'cfg_args',init_ply]
 source_before={str(path):sha(path) for path in source_inputs}
 if a.gpu is not None:os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
 if a.tmc3_bin:os.environ['PATH']=str(a.tmc3_bin.resolve())+os.pathsep+os.environ.get('PATH','')
 os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
 job.mkdir(parents=True,exist_ok=False)
 save(job/'source_provenance.json',{'job':str(input_job),'input_sha256':source_before,'release_code':code_record})
 save(job/'job.json',spec)
 sys.path.insert(0,str(root))
 from src.hacpp.bridge import HacppBridge
 from src.hacpp.pipeline import build_bundle
 from src.hacpp.bundle import read_bundle,bundle_artifact_bytes
 bridge=HacppBridge(hacpp_root=str(root/'third_party/HAC-plus'),python_exe=a.python,device='cuda:0')
 raw=job/'raw';bundle=job/'bundle'
 if (job/'encode.json').is_file():enc=read(job/'encode.json')
 else:enc=bridge.encode(model,raw,source);save(job/'encode.json',enc)
 assert enc['ok'] and enc['gpcc_mode']=='gpcc'
 if not (job/'pack.json').is_file():
  built=build_bundle(bundle,scene=None,hacpp_stream_dir=raw,scene_id=spec['scene']);save(job/'pack.json',built.to_dict())
 checked=read_bundle(bundle,verify=True);size=bundle_artifact_bytes(bundle)
 assert size==checked.storage['artifact_bytes']
 cfg=read(bundle/'hacpp'/'decoder_config.json');assert cfg['decoded_version'] and cfg['voxel_size']==.005 and cfg['all_views_train_test']
 if (job/'decode_only.json').is_file():dec=read(job/'decode_only.json')
 else:dec=bridge.decode(bundle/'hacpp');save(job/'decode_only.json',dec)
 assert dec['ok'] and dec['self_contained_decode']
 if (job/'decode_render.json').is_file():comparison=read(job/'decode_render.json')
 else:comparison=bridge.decode(bundle/'hacpp',model_path=model,source_path=source,render=True);save(job/'decode_render.json',comparison)
 result=comparison['render_decoded'];full=comparison['render_full'];native=read(model/'results.json')['ours_30000']
 for r in [full,result]:
  assert r['views']==r['views_total_in_split']==150 and r['split']=='test' and not r['truncated'] and r['metric_domain']=='png_uint8_equivalent'
  for key in ['psnr','ssim','lpips']:assert math.isfinite(r[key])
 deltas={key:result[key]-native[key.upper()] for key in ['psnr','ssim','lpips']}
 for key,delta in deltas.items():assert abs(delta)<1e-4,(key,delta)
 standalone=Path(__file__).resolve().with_name('hacpp_bundle_only_render.py');assert standalone.is_file(),'Independent render validator is required'
 cmd=[a.python,'-B',str(standalone),'--bundle',str(bundle),'--dataset',source,'--output',str(job/'bundle_only'),
  '--release-root',str(root),'--code-manifest',str(code_manifest)]
 if a.tmc3_bin:cmd+=['--tmc3-bin',str(a.tmc3_bin.resolve())]
 with (job/'bundle_only.log').open('w') as log:proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 assert proc.returncode==0,'Standalone render failed, inspect bundle_only.log'
 independent=read(job/'bundle_only'/'bundle_only_render.json')
 assert independent['ok'] and independent['self_contained_decode'] and independent['bundle_only_model_state']
 assert not independent['source_model_used'] and not independent['source_checkpoint_used']
 assert independent['entropy_initialization']['hash_stream_preloaded_before_entropy_decode']
 assert independent['entropy_initialization']['hash_stream_sha256']==sha(bundle/'hacpp'/'hash.b')
 assert independent['fixed_runtime_state']['anchor_culling_rotation']==[1,0,0,0] and not independent['fixed_runtime_state']['source_model_read']
 assert all(independent['preservation'].values()) and independent['actual_bundle_bytes']==size
 actual=independent['render_decoded'];protocol=independent['protocol']
 assert protocol['train_views']==protocol['test_views']==protocol['unique_test_views']==150 and protocol['train_test_same_order']
 assert actual['views']==actual['views_total_in_split']==actual['render_png_count']==actual['gt_png_count']==150
 assert actual['split']=='test' and not actual['truncated'] and actual['metric_domain']=='png_uint8_equivalent'
 independent_deltas={key:actual[key]-result[key] for key in ['psnr','ssim','lpips']}
 for key,delta in independent_deltas.items():assert math.isfinite(actual[key]) and abs(delta)<1e-4,(key,delta)
 assert actual['max_abs_saved_png_cpu_psnr_difference']<1e-4
 def cameras(path):return {c['img_name']:{k:v for k,v in c.items() if k!='id'} for c in read(path)}
 assert cameras(model/'cameras.json')==cameras(job/'bundle_only'/'cameras.json')
 from PIL import Image
 import numpy as np
 independent_rows=read(job/'bundle_only'/'per_view_metrics.json');assert len(independent_rows)==150
 png_equal=0
 for row in independent_rows:
  name=Path(row['files']['gt']['file']).name
  for label in ['renders','gt']:
   path=job/'bundle_only'/row['files'][label]['file'];assert sha(path)==row['files'][label]['png_sha256']
  g=np.asarray(Image.open(job/'bundle_only'/'gt'/name).convert('RGB'))
  native_gt=np.asarray(Image.open(model/'test'/'ours_30000'/'gt'/name).convert('RGB'))
  assert g.shape==(720,1280,3) and np.array_equal(g,native_gt)
  pred=np.asarray(Image.open(job/'bundle_only'/'renders'/name).convert('RGB'))
  b=np.asarray(Image.open(model/'test'/'ours_30000'/'renders'/name).convert('RGB'))
  png_equal+=int(np.array_equal(pred,b))
 save(job/'independent_execution.json',dict(command=cmd,returncode=proc.returncode))
 files={str(p.relative_to(bundle)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(bundle.rglob('*')) if p.is_file()}
 assert sum(r['bytes'] for r in files.values())==size
 data=dict(scene=spec['scene'],method='HPS3DGS-HAC++',role='backend_rd',setting=spec['setting'],config=spec['setting'],lmbda=spec['lmbda'],
  name=spec['setting'],group='lambda_sweep',size_mb=size/1e6,plotted_bytes=size,**{k:actual[k] for k in ['psnr','ssim','lpips']},
  n_train=150,n_eval=150,iterations=30000,gpu=spec['gpu'],model=str(model),bundle=str(bundle),bundle_files=files,
  codec_only=checked.codec_only,storage=checked.storage,metric_domain='png_uint8_equivalent',
  native_metric_deltas=deltas,independent_metric_deltas=independent_deltas,independent_render_metrics=actual,native_independent_gt_exact_frames=150,
  native_independent_render_exact_frames=png_equal,source_model_free_decode=dec['self_contained_decode'],independent_render_record=str(job/'bundle_only'/'bundle_only_render.json'),
  all_model_training_fresh=bool(spec.get('all_model_training_fresh',False)),validation_only=True,
  initialization_sha256=spec['input']['init_sha256'],update_until=spec['input']['update_until'],verified_at=time.time())
 assert all(sha(path)==digest for path,digest in source_before.items()),'Source job/model metadata changed'
 assert sha(code_manifest)==code_record['sha256'],'Release code manifest changed'
 release_code(root,code_manifest)
 data['source_inputs_unchanged']=True;data['release_code']=code_record;data['source_job']=str(input_job)
 data['lambda']=data['lmbda'];save(job/'verified.json',data);print(json.dumps(data,indent=2))
if __name__=='__main__':main()
