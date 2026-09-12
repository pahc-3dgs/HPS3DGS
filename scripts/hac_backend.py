#!/usr/bin/env python3
"""Independent HAC adapter: encode, pack, verify, and bundle-only decode/render."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.hac.bundle import pack,read_bundle,sha,write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hac-root',type=Path)
    sub=p.add_subparsers(dest='command',required=True)
    enc=sub.add_parser('encode');enc.add_argument('--model',type=Path,required=True)
    enc.add_argument('--dataset',type=Path,required=True);enc.add_argument('--out',type=Path,required=True)
    pk=sub.add_parser('pack');pk.add_argument('--raw',type=Path,required=True)
    pk.add_argument('--bundle',type=Path,required=True);pk.add_argument('--scene',required=True)
    vr=sub.add_parser('verify');vr.add_argument('--bundle',type=Path,required=True)
    dec=sub.add_parser('decode');dec.add_argument('--bundle',type=Path,required=True)
    dec.add_argument('--dataset',type=Path);dec.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='pack':
        result=pack(a.raw,a.bundle,a.scene)
    elif a.command=='verify':
        result=read_bundle(a.bundle)
    else:
        if a.hac_root is None:
            p.error('--hac-root is required for encode/decode')
        from src.hac import driver
        driver._prepend_hacpp_root(str(a.hac_root.resolve()))
        if a.command=='encode':
            result=driver.encode(a.model,a.dataset,a.out)
            write_json(a.out.parent/'encode.json',result)
        else:
            before=read_bundle(a.bundle)
            if a.output.exists():
                raise FileExistsError('Refusing to overwrite decode evidence')
            a.output.mkdir(parents=True)
            pc,config,result=driver.decode(a.bundle/'hac')
            if a.dataset:
                from src.hac.evaluation import load_cameras,render_saved,snapshot
                for name in ('renders','gt'):(a.output/name).mkdir()
                cams,protocol=load_cameras(a.dataset,config,a.output)
                result['render_decoded']=render_saved(cams,pc,config,a.output)
                result['protocol']=protocol
                result['preservation']={
                    'metadata_unchanged':snapshot(protocol['dataset_metadata_before'])==protocol['dataset_metadata_before'],
                    'images_unchanged':snapshot(protocol['source_images_before'])==protocol['source_images_before']}
                assert all(result['preservation'].values())
            after=read_bundle(a.bundle)
            assert before==after
            result.update(storage=before['storage'],bundle_unchanged=True)
            write_json(a.output/'decode.json',result)
    print('@@HAC_RESULT@@ '+json.dumps(result,sort_keys=True))

if __name__=='__main__':main()
