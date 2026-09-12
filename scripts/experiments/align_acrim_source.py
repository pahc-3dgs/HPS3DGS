"""Recover a normalization transform from 150 saved cameras; no image fitting."""
from pathlib import Path
import argparse,hashlib,json
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def release_code(root, manifest_path):
 root=Path(root).resolve();manifest_path=Path(manifest_path).resolve()
 files=json.loads(manifest_path.read_text()).get('files',{})
 own=Path(__file__).resolve().relative_to(root).as_posix()
 if not files or own not in files:raise ValueError('Release code manifest lacks this entry point')
 for name,digest in files.items():
  path=(root/name).resolve()
  try:path.relative_to(root)
  except ValueError:raise ValueError('Code manifest path escapes release root: '+name)
  if not path.is_file() or sha(path)!=digest:raise ValueError('Release source hash mismatch: '+name)
 return {'path':str(manifest_path),'sha256':sha(manifest_path),'verified_files':len(files)}

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--source-manifest',type=Path,required=True,help='Read-only historical scene/input manifest')
 parser.add_argument('--camera-comparison',type=Path,required=True,help='JSON containing source_training and current camera rows')
 parser.add_argument('--source-cameras',type=Path,required=True,help='Original saved cameras.json, retained as a provenance input')
 parser.add_argument('--source-cameras-sha256',help='Optional expected SHA256 of original cameras.json')
 parser.add_argument('--output',type=Path,required=True,help='New alignment output directory; must not exist')
 parser.add_argument('--release-root',type=Path,default=Path(__file__).resolve().parents[2])
 parser.add_argument('--code-manifest',type=Path,help='Defaults to release-root/release_manifest.json')
 args=parser.parse_args()
 code_manifest=(args.code_manifest or args.release_root/'release_manifest.json').resolve()
 code_record=release_code(args.release_root,code_manifest)
 inputs={str(p.resolve()):sha(p) for p in [args.source_manifest,args.camera_comparison,args.source_cameras]}
 if args.source_cameras_sha256 and sha(args.source_cameras)!=args.source_cameras_sha256:raise ValueError('Source camera SHA256 mismatch')
 OUT=args.output.resolve()
 if OUT.exists():raise FileExistsError('Output directory already exists: '+str(OUT))
 manifest=json.loads(args.source_manifest.read_text());item=manifest['scenes']['acrim-2'];original=item['files']
 for path in [item['source_path']]+[str(Path(x['path']).parent) for x in original.values()]:
  try:OUT.relative_to(Path(path).resolve())
  except ValueError:continue
  raise ValueError('Output must be outside input directory: '+path)
 for entry in original.values():assert sha(entry['path'])==entry['sha256'],entry['path']
 import numpy as np
 from plyfile import PlyData,PlyElement
 data=json.loads(args.camera_comparison.read_text())
 old={x['img_name']:x for x in data['source_training']};cur={x['img_name']:x for x in data['current']}
 assert len(old)==len(cur)==150 and set(old)==set(cur)
 names=sorted(cur);a=np.array([old[n]['position'] for n in names]);b=np.array([cur[n]['position'] for n in names])
 for n in names:
  assert np.allclose(old[n]['rotation'],cur[n]['rotation'],atol=1e-10,rtol=0)
  for k in ['width','height','fx','fy']:assert abs(old[n][k]-cur[n][k])<1e-9
 ac=a-a.mean(0);bc=b-b.mean(0);scale=float(np.sum(ac*bc)/np.sum(ac*ac));shift=b.mean(0)-scale*a.mean(0)
 residual=float(np.max(np.abs(a*scale+shift-b)))
 assert scale>0 and residual<1e-8,(scale,shift,residual)
 OUT.mkdir(parents=True,exist_ok=False)
 newitem=json.loads(json.dumps(item));checks={}
 for key in ['raw','feature']:
  src=Path(original[key]['path']);assert sha(src)==original[key]['sha256']
  ply=PlyData.read(str(src));v=ply['vertex'].data;updated=v.copy()
  for axis,k in enumerate(['x','y','z']):updated[k]=(v[k].astype(np.float64)*scale+shift[axis]).astype(np.float32)
  scaled=[k for k in v.dtype.names if k.startswith('scale_')]
  for k in scaled:updated[k]=(v[k].astype(np.float64)+np.log(scale)).astype(np.float32)
  changed=set(['x','y','z']+scaled)
  assert all(updated[k].tobytes()==v[k].tobytes() for k in v.dtype.names if k not in changed)
  dst=OUT/(key+'_normalized.ply');assert not dst.exists()
  elements=[PlyElement.describe(updated,'vertex') if el.name=='vertex' else el for el in ply.elements]
  PlyData(elements,text=ply.text,byte_order=ply.byte_order,comments=ply.comments,obj_info=ply.obj_info).write(str(dst))
  newitem['files'][key]={'path':str(dst),'sha256':sha(dst),'bytes':dst.stat().st_size}
  checks[key]={'source':original[key],'output':newitem['files'][key],'rows':len(v),'unchanged_fields_bitexact':True,'transformed_fields':sorted(changed)}
 raw=PlyData.read(newitem['files']['raw']['path'])['vertex'];feat=PlyData.read(newitem['files']['feature']['path'])['vertex']
 assert all(raw[k].tobytes()==feat[k].tobytes() for k in ['x','y','z'])
 report={'formula':'xyz_current = scale * xyz_saved + translation; log_scale_current = log_scale_saved + log(scale)',
         'scale':scale,'translation':shift.tolist(),'camera_max_abs_residual':residual,'camera_pairs':150,
         'same_rotations_and_intrinsics':True,'source_cameras_sha256':sha(args.source_cameras),
         'camera_comparison_sha256':sha(args.camera_comparison),'not_fitted_to_images':True,'fields':checks,
         'semantic_xyz_bitexact_after_transform':True,'original_inputs_unchanged':all(sha(x['path'])==x['sha256'] for x in original.values())}
 assert report['original_inputs_unchanged']
 assert all(sha(path)==digest for path,digest in inputs.items()),'Source provenance inputs changed'
 assert sha(code_manifest)==code_record['sha256'],'Release code manifest changed'
 release_code(args.release_root,code_manifest)
 report['source_provenance']={'inputs':inputs,'release_code':code_record}
 (OUT/'source_alignment.json').write_text(json.dumps(report,indent=2))
 newitem['alignment']=report;(OUT/'aligned_scene_binding.json').write_text(json.dumps(newitem,indent=2))
 print(json.dumps({k:report[k] for k in ['scale','translation','camera_max_abs_residual','camera_pairs','original_inputs_unchanged']},indent=2))
if __name__=='__main__':main()
