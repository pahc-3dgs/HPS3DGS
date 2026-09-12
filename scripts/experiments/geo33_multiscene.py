"""One fixed geometry plan and five uniform-K experiments using frozen Geo33."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time,traceback

def release_code(root, manifest_path, required):
 root=Path(root).resolve();manifest_path=Path(manifest_path).resolve()
 document=json.loads(manifest_path.read_text());files=document.get('files',{})
 if not files:raise ValueError('Release code manifest has no files')
 required=set(required)|{Path(__file__).resolve().relative_to(root).as_posix()}
 if not required.issubset(files):raise ValueError('Release manifest lacks required files: '+str(sorted(required-set(files))))
 for name,digest in files.items():
  path=(root/name).resolve()
  try:path.relative_to(root)
  except ValueError:raise ValueError('Code manifest path escapes release root: '+name)
  if not path.is_file() or sha(path)!=digest:raise ValueError('Release source hash mismatch: '+name)
 return {'path':str(manifest_path),'sha256':sha(manifest_path),'verified_files':len(files)}

def outside(path, protected):
 for source in protected:
  try:path.relative_to(Path(source).resolve())
  except ValueError:continue
  raise ValueError('Output must be outside input directory: '+str(source))
def sha(p):
 h=hashlib.sha256()
 with open(str(p),'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False));t.replace(p)
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--scene',required=True);ap.add_argument('--gpu',type=int,required=True)
 ap.add_argument('--source-manifest',type=Path,required=True,help='Read-only historical scene/input and codebook protocol manifest')
 ap.add_argument('--output',type=Path,required=True,help='New output directory for this scene; must not exist')
 ap.add_argument('--python',required=True,help='Python executable with the verified Geo33/SAGA runtime')
 ap.add_argument('--release-root',type=Path,default=Path(__file__).resolve().parents[2])
 ap.add_argument('--code-manifest',type=Path,help='Defaults to release-root/release_manifest.json')
 ap.add_argument('--binding',type=Path,help='Explicit scene binding, e.g. aligned_scene_binding.json')
 ap.add_argument('--geometry-plan',type=Path,help='Explicit frozen geometry_plan.pt; no implicit zxa1 plan reuse')
 ap.add_argument('--geometry-plan-sha256',help='Required with --geometry-plan')
 a=ap.parse_args()
 if bool(a.geometry_plan)!=bool(a.geometry_plan_sha256):ap.error('--geometry-plan and --geometry-plan-sha256 must be supplied together')
 a.release_root=a.release_root.resolve();code_manifest=(a.code_manifest or a.release_root/'release_manifest.json').resolve()
 work=a.release_root/'third_party/SegAnyGAussians';PY=a.python
 required=['third_party/SegAnyGAussians/'+n for n in ['geo32.py','geo33.py','geo33_geometry.py','geo33_appearance.py','geo33_storage_codec.py','run_storage_any.py']]
 code_record=release_code(a.release_root,code_manifest,required)
 source_digest=sha(a.source_manifest);manifest=json.loads(a.source_manifest.read_text())
 item=json.loads(a.binding.read_text()) if a.binding else manifest['scenes'][a.scene]
 files=item['files'];out=a.output.resolve()
 outside(out,[item['source_path']]+[Path(entry['path']).parent for entry in files.values()])
 if out.exists():raise FileExistsError('Output directory already exists: '+str(out))
 for entry in files.values():assert sha(entry['path'])==entry['sha256'],entry['path']
 if a.geometry_plan and sha(a.geometry_plan)!=a.geometry_plan_sha256:raise ValueError('Geometry plan SHA256 mismatch')
 source_record={'manifest':str(a.source_manifest.resolve()),'sha256':source_digest,'release_code':code_record}
 if a.binding:source_record['binding']={'path':str(a.binding.resolve()),'sha256':sha(a.binding)}
 import fcntl
 lock=open('/tmp/gszip_all150_gpu_%s.lock'%a.gpu,'a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 used=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip());assert used<100,used
 out.mkdir(parents=True,exist_ok=False)
 save(out/'scene_binding.json',item)
 save(out/'source_provenance.json',source_record)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
 state={'scene':a.scene,'gpu':a.gpu,'pid':os.getpid(),'state':'running','completed':[],'failed':[],'started':time.time()}
 def stage(name,cmd):
  state.update(stage=name,command=cmd);save(out/'status.json',state)
  with (out/(name+'.log')).open('w') as f:subprocess.run(cmd,cwd=str(work),env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 def base(mode,output):return [PY,'-B',str(work/'geo33.py'),'--mode',mode,'--source-path',item['source_path'],'--input-ply',files['raw']['path'],'--output',str(output),'--eval-views','150']
 try:
  for entry in files.values():assert sha(entry['path'])==entry['sha256']
  if a.geometry_plan:
   plan=a.geometry_plan.resolve()
   save(out/'geometry_reuse.json',{'path':str(plan),'sha256':sha(plan),'expected_sha256':a.geometry_plan_sha256,'release_code_verified':True})
  else:
   stage('reference',base('reference',out/'reference_raw'))
   prep=out/'geometry_prepare'
   stage('prepare',base('prepare',prep)+['--feature-ply',files['feature']['path'],'--labels',files['labels']['path'],
        '--scale-gate',files['scale_gate']['path'],'--pose-iters','300','--max-pairs','17','--max-psnr-drop','.5'])
   plan=prep/'geometry_plan.pt'
  assert plan.is_file()
  for k in manifest['codebook_values']:
   point=out/('k%04d'%k)
   try:
    stage('train_k%04d'%k,base('train',point)+['--geometry-plan',str(plan),'--steps','10000','--k',str(k),'--k-sh',str(k),'--k-dc',str(k)])
    native=json.loads((point/'verified.json').read_text())
    assert native['eval_frames']==150
    stage('storage_k%04d'%k,[PY,'-B',str(work/'run_storage_any.py'),'--source-package',str(point/'scene.geo33.zip'),
          '--expected-source-sha256',native['artifact_sha256'],'--expected-base-points',str(native['unique_points']),
          '--source',item['source_path'],'--output',str(point/'storage'),
          '--reference-psnr',str(native['metrics']['psnr']),'--reference-bytes',str(native['artifact_bytes'])])
    storage=json.loads((point/'storage/verified.json').read_text());assert storage['eval_frames']==150
    state['completed'].append({'k':k,'native_bytes':native['artifact_bytes'],'native_psnr':native['metrics']['psnr'],
                               'storage_bytes':storage['artifact_bytes'],'storage_psnr':storage['metrics']['psnr'],
                               'unique_points':native['unique_points'],'instances':native['instances'],'model':str(point)})
   except Exception as exc:
    failure={'k':k,'error':str(exc),'traceback':traceback.format_exc()};state['failed'].append(failure);save(point/'runner_failure.json',failure)
   save(out/'status.json',state)
  for entry in files.values():assert sha(entry['path'])==entry['sha256']
  assert sha(a.source_manifest)==source_digest,'Source manifest changed'
  if a.binding:assert sha(a.binding)==source_record['binding']['sha256'],'Scene binding changed'
  if a.geometry_plan:assert sha(a.geometry_plan)==a.geometry_plan_sha256,'Geometry plan changed'
  assert sha(code_manifest)==code_record['sha256'],'Release manifest changed'
  release_code(a.release_root,code_manifest,required)
  state.update(state='complete' if not state['failed'] else 'finished_with_failures',finished=time.time(),stage='finished')
 except Exception as exc:
  state.update(state='failed',error=str(exc),traceback=traceback.format_exc(),finished=time.time())
 save(out/'status.json',state);print(json.dumps(state),flush=True)
 if state['state']!='complete':raise SystemExit(1)
if __name__=='__main__':main()
