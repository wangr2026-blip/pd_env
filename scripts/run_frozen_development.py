#!/usr/bin/env python3
"""Frozen development testing: pooled caches, participant balance, sealed test."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import yaml
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO),str(REPO/'scripts')]
from mpower_common import save_json,write_table
from src.data.frozen_protocol import common_windows,development_frame
from src.data.benchmark import digest
from src.models.frozen import ProtocolEncoder
from src.training.development import fit_development
from scripts.run_frozen_benchmark import initialize


def recording_vectors(encoder,wave,context,taps):
    windows,coverage=common_windows(wave,context)
    values=[encoder.represent(window,taps) for window in windows]
    result={}
    for name in values[0]:
        x=np.stack([v[name] for v in values]);result[name+'/window_mean']=x.mean(0)
        if name=='native_pooler' and context=='matched_2s':result[name+'/window_mean_sd']=np.r_[x.mean(0),x.std(0)]
    return result,coverage


def state_digest(model):
    sha=hashlib.sha256()
    for name,value in model.state_dict().items():sha.update(name.encode());sha.update(value.detach().cpu().contiguous().numpy().tobytes())
    return sha.hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);p.add_argument('--snapshot',required=True);p.add_argument('--model',required=True);p.add_argument('--context',choices=['full','matched_2s'],required=True)
    a=p.parse_args();os.umask(0o077);cfg=yaml.safe_load(Path(a.config).read_text());initialize(cfg)
    snapshot=Path(a.snapshot).resolve();ready=json.loads((snapshot/'READY.json').read_text());manifest=snapshot/'development_manifest.parquet'
    if hashlib.sha256(manifest.read_bytes()).hexdigest()!=ready['manifest_sha256']:raise ValueError('Snapshot changed')
    frame=development_frame(pd.read_parquet(manifest));frame=frame.reset_index(drop=True)
    root=Path(os.environ['MPOWER_PROTOCOL_ROOT']).resolve()
    if root==REPO or REPO in root.parents:raise ValueError('Outputs must be outside code checkout')
    out=root/'runs'/a.model/a.context;out.mkdir(parents=True,exist_ok=True)
    encoder=ProtocolEncoder(a.model,cfg,Path(os.environ['MPOWER_MODEL_ROOT']))
    if str(encoder.device)=='cpu' and cfg.get('require_gpu',True):raise RuntimeError('GPU required for scheduled full experiment')
    if encoder.amp:raise ValueError('Protocol baseline must start float32')
    taps=cfg['feature_taps'][encoder.kind]
    versions={p:importlib.metadata.version(p) for p in ['numpy','torch','transformers','scikit-learn','scipy']}
    code_digest=hashlib.sha256((REPO/'src/models/frozen.py').read_bytes()+(REPO/'src/data/frozen_protocol.py').read_bytes()).hexdigest()
    signature=digest({'model':encoder.provenance,'taps':taps,'context':a.context,'dtype':'float32','versions':versions,'code':code_digest})
    cache=root/'pooled_features'/a.model/a.context/signature;cache.mkdir(parents=True,exist_ok=True)
    save_json(out/'provenance.json',{'manifest_sha256':ready['manifest_sha256'],'model':encoder.provenance,'context':a.context,
        'feature_taps':taps,'signature':signature,'versions':versions,'test_labels_read':False,
        'selection_metric':'initial-assessment participant validation AUROC; log-loss then lexical tie break',
        'release_status':'provisional development experiment; task and near-duplicate review pending'})
    write_table(frame.drop(columns=[c for c in frame if c=='native_channel_qc']),out/'input_manifest.parquet')
    before=state_digest(encoder.model);features={};failures=[];coverage=[]
    extraction_start=time.monotonic()
    import torch
    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    for i,row in enumerate(frame.itertuples()):
        key=digest({'recording_id':row.recording_id,'source':row.file_sha256,'derived':row.mono16k_sha256,'signature':signature})
        path=cache/f'{key}.npz'
        try:
            if path.exists():
                with np.load(path,allow_pickle=False) as stored:values={k:stored[k].copy() for k in stored.files}
                _,info=common_windows(np.load(row.waveform_path,mmap_mode='r'),a.context)
            else:
                values,info=recording_vectors(encoder,np.load(row.waveform_path),a.context,taps)
                temporary=path.with_suffix('.tmp.npz');np.savez(temporary,**values);temporary.replace(path)
            if any(not np.isfinite(v).all() for v in values.values()):raise ValueError('nonfinite recording features')
            for name,v in values.items():features.setdefault(name,[]).append(v)
            coverage.append({'recording_id':row.recording_id,**info})
        except Exception as exc:failures.append({'recording_id':row.recording_id,'error':f'{type(exc).__name__}: {exc}'})
        if (i+1)%250==0:save_json(out/'progress.json',{'attempted':i+1,'total':len(frame),'failures':len(failures)});print(a.model,a.context,i+1,'/',len(frame),flush=True)
    write_table(pd.DataFrame(coverage),out/'coverage.parquet');write_table(pd.DataFrame(failures,columns=['recording_id','error']),out/'extraction_failures.csv')
    if failures:raise RuntimeError(f'{len(failures)} extraction failures; common cohort not silently changed')
    if before!=state_digest(encoder.model):raise AssertionError('Frozen encoder state changed')
    extraction_seconds=time.monotonic()-extraction_start
    peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3 if torch.cuda.is_available() else 0
    del encoder
    import gc,torch;gc.collect();torch.cuda.empty_cache()
    import joblib
    rows=[]
    for name,arrays in features.items():
        x=np.stack(arrays)
        for arm in ['technical_primary','pitch_sensitivity']:
            selected=np.ones(len(frame),dtype=bool) if arm=='technical_primary' else frame.pitch_detected_s.ge(2).to_numpy()
            f=frame.loc[selected].copy().reset_index(drop=True);f['initial_recording']=~f.duplicated('participant_id')
            trained=fit_development(x[selected],f,cfg['head'],cfg['seed'])
            for head,result in trained.items():
                directory=out/arm/name.replace('/','__')/head;directory.mkdir(parents=True,exist_ok=True)
                joblib.dump(result['state'],directory/'head.joblib')
                write_table(result['initial_validation_predictions'],directory/'initial_validation_predictions.csv')
                save_json(directory/'validation_metrics.json',result['metrics']);save_json(directory/'search.json',result['search'])
                rows.append({'model':a.model,'context':a.context,'arm':arm,'representation':name,'head':head,
                             'manifest_sha256':ready['manifest_sha256'],'peak_gpu_gib':peak_gpu_gib,
                             'seconds_per_recording':extraction_seconds/len(frame),**result['metrics']})
    write_table(pd.DataFrame(rows),out/'validation_leaderboard.csv')
    save_json(out/'DONE.json',{'model':a.model,'context':a.context,'recordings':len(frame),'pipelines':len(rows),
        'test_labels_read':False,'encoder_unchanged':True,'peak_gpu_gib':peak_gpu_gib,
        'seconds_per_recording':extraction_seconds/len(frame),'manifest_sha256':ready['manifest_sha256']})
    print((out/'DONE.json').read_text(),flush=True)
if __name__=='__main__':main()
