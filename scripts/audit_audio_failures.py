#!/usr/bin/env python3
"""Read-only MP4/decoder audit of failed recordings; evidence stays outside Git."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess

import numpy as np
import pandas as pd
from mpower_common import REPO, ffmpeg, save_json


def mp4_atoms(path):
    """Walk top-level boxes without loading or interpreting audio payloads."""
    size=Path(path).stat().st_size
    atoms=[];offset=0
    with Path(path).open('rb') as stream:
        while offset+8<=size:
            stream.seek(offset)
            length,kind=struct.unpack('>I4s',stream.read(8));header=8
            if length==1:
                extended=stream.read(8)
                if len(extended)!=8:break
                length=struct.unpack('>Q',extended)[0];header=16
            if length==0:length=size-offset
            valid=length>=header and offset+length<=size
            atoms.append({'type':kind.decode('ascii',errors='replace'),'offset':offset,
                          'size':length,'payload_bytes':max(0,length-header),'complete':valid})
            if not valid:break
            offset+=length
    return atoms


def decode_attempt(path, tolerant=False):
    options=['-err_detect','ignore_err','-fflags','+discardcorrupt'] if tolerant else []
    args=[ffmpeg(),'-v','error','-nostdin',*options,'-i',str(path),
          '-t','181','-ac','1','-ar','16000','-f','f32le','-']
    try:
        p=subprocess.run(args,capture_output=True,timeout=120)
        samples=np.frombuffer(p.stdout,dtype='<f4') if len(p.stdout)%4==0 else np.array([])
        return {'returncode':p.returncode,'output_bytes':len(p.stdout),'samples':len(samples),
                'duration_s':len(samples)/16000,'finite':bool(np.isfinite(samples).all()),
                'stderr':p.stderr.decode('utf-8',errors='replace')[-12000:]}
    except subprocess.TimeoutExpired:
        return {'returncode':None,'samples':0,'output_bytes':0,'stderr':'timeout after 120 seconds'}


def classify(atoms, normal, tolerant):
    kinds={a['type'] for a in atoms}
    if 'moov' not in kinds and 'mdat' not in kinds:
        return 'missing_metadata_and_audio_payload'
    if 'moov' not in kinds:return 'missing_movie_metadata'
    if 'mdat' not in kinds:return 'missing_audio_payload'
    if normal.get('samples',0)>0:return 'strict_decoder_produced_samples_review_required'
    if tolerant.get('samples',0)>0:return 'tolerant_decoder_produced_samples_review_required'
    return 'undecodable_audio_payload'


def audit(row):
    path=Path(row['audio_path']);atoms=mp4_atoms(path)
    normal=decode_attempt(path);tolerant=decode_attempt(path,tolerant=True)
    result={'recording_id':row['recording_id'],'audio_handle':str(row['audio_handle']),
            'audio_path':str(path),'file_size_bytes':path.stat().st_size,
            'md5':hashlib.md5(path.read_bytes()).hexdigest(),'atoms':atoms,
            'normal':normal,'tolerant':tolerant}
    result['classification']=classify(atoms,normal,tolerant)
    return result


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--root',default=os.environ.get('MPOWER_OUTPUT_ROOT'))
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    if not args.root:p.error('Set MPOWER_OUTPUT_ROOT or --root')
    root=Path(args.root).expanduser().resolve()
    if root==REPO or REPO in root.parents:p.error('Private root must be outside Git')
    if args.workers<1:p.error('--workers must be positive')
    os.umask(0o077)
    qc=pd.read_parquet(root/'processed/full_audio_quality_metrics.parquet')
    manifest=pd.read_parquet(root/'processed/recording_manifest.parquet')
    failed=qc.loc[qc.decode_status.ne('ok'),['recording_id']].merge(
        manifest[['recording_id','audio_path','audio_handle']],on='recording_id',validate='one_to_one')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=root/'eda/diagnostics'/f'failed_audio_audit_{stamp}'
    out.mkdir(parents=True,exist_ok=False)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results=list(pool.map(audit,failed.to_dict('records')))
    for r in results:save_json(out/f"{r['recording_id']}.json",r)
    pd.DataFrame([{'recording_id':r['recording_id'],'audio_handle':r['audio_handle'],
        'audio_path':r['audio_path'],'file_size_bytes':r['file_size_bytes'],'md5':r['md5'],
        'classification':r['classification'],'strict_returncode':r['normal']['returncode'],
        'strict_samples':r['normal']['samples'],'tolerant_returncode':r['tolerant']['returncode'],
        'tolerant_samples':r['tolerant']['samples']} for r in results]).to_csv(out/'failures.csv',index=False)
    summary={'recordings':len(results),'classifications':dict(Counter(r['classification'] for r in results)),
             'file_sizes':dict(Counter(r['file_size_bytes'] for r in results)),
             'strict_decoded_samples':sum(r['normal']['samples'] for r in results),
             'tolerant_decoded_samples':sum(r['tolerant']['samples'] for r in results),
             'source_files_modified':False,'canonical_outputs_modified':False}
    save_json(out/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)
    print('Private evidence:',out,flush=True)

if __name__=='__main__':main()
