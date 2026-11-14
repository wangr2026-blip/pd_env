#!/usr/bin/env python3
"""Prepare immutable development-only waveforms/QC; run through Slurm."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import yaml
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'scripts')]
from mpower_common import save_json,write_table,version_info
from src.data.frozen_protocol import decode_native_and_derive,common_windows,development_frame
from create_participant_splits import verify
from inspect_audio import probe


def prepare_one(args):
    row,out=args;rid=row['recording_id'];directory=Path(out)/'waveforms';metadata=directory/f'{rid}.json';wave=directory/f'{rid}.npy'
    try:
        if wave.exists() and metadata.exists():return {**row,**json.loads(metadata.read_text()),'waveform_path':str(wave),'preprocess_status':'ok'}
        path=Path(row['audio_path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['file_sha256']:raise ValueError('source_hash_changed')
        rate=row.get('original_sample_rate');channels=row.get('original_channels')
        if pd.isna(rate) or pd.isna(channels):
            p=probe(path);rate=p['original_sample_rate'];channels=p['original_channels']
        y,info=decode_native_and_derive(path,int(rate),int(channels))
        _,windows=common_windows(y,'matched_2s');info['matched_windows']=windows
        from audio_quality import quality
        settings=yaml.safe_load((REPO/'configs/mpower_preprocessing.yaml').read_text())
        analysis=quality(y,16000,settings)
        analysis['frame_energy_dynamic_range_db']=analysis.pop('estimated_snr_db')
        analysis['flag_low_energy_dynamic_range']=analysis.pop('flag_low_estimated_snr')
        info['primary_span_qc']=analysis
        info['analysis_rms']=float(np.sqrt(np.mean(y.astype(float)**2)));info['analysis_peak']=float(np.abs(y).max())
        temporary=wave.with_suffix('.tmp.npy');np.save(temporary,y);temporary.replace(wave);save_json(metadata,info)
        return {**row,**info,'waveform_path':str(wave),'preprocess_status':'ok'}
    except Exception as exc:
        stderr=getattr(exc,'stderr',b'') or b''
        return {**row,'preprocess_status':'error','preprocess_error':f'{type(exc).__name__}: {exc}',
                'decoder_stderr':stderr.decode(errors='replace')[-4000:] if isinstance(stderr,bytes) else str(stderr)[-4000:]}


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=16);p.add_argument('--limit',type=int)
    a=p.parse_args();os.umask(0o077);cfg=yaml.safe_load(Path(a.config).read_text());out=Path(a.output).resolve()
    if out==REPO or REPO in out.parents:raise ValueError('Outputs must stay outside code checkout')
    if (out/'READY.json').exists():raise ValueError('Immutable snapshot already complete; choose a new output')
    out.mkdir(parents=True,exist_ok=True);(out/'waveforms').mkdir(exist_ok=True)
    source=Path(os.environ['MPOWER_OUTPUT_ROOT'])/'processed/cohort_A'
    # Intentionally never open test_manifest.csv or participant-wide label tables.
    frame=pd.concat([pd.read_csv(source/f'{split}_manifest.csv',dtype={'participant_id':str,'recording_id':str}) for split in ['train','val']],ignore_index=True)
    frame=development_frame(frame);verify(frame)
    frame=frame[frame.duration_s.ge(2) & frame.task.eq('sustained_phonation')].copy()
    frame=frame.sort_values(['participant_id','recorded_at','recording_id'],na_position='last')
    # Declared initial-assessment primary outcome; sensitivity retains all visits.
    frame['initial_recording']=~frame.duplicated('participant_id')
    if a.limit:
        # Bounded validation sample retains both classes in each development split.
        frame=frame.drop_duplicates('participant_id').groupby(['split','diagnosis'],group_keys=False).head(max(1,a.limit//4))
    # Add technical covariates from voice metadata, which contains no diagnosis labels.
    raw=Path(os.environ['MPOWER_DATA_ROOT'])/'voice_activity/Voice_Activity.csv'
    voice=pd.read_csv(raw,usecols=['recordId','phoneInfo','appVersion'],dtype=str)
    frame=frame.merge(voice.rename(columns={'recordId':'source_recording_id'}),on='source_recording_id',how='left',validate='one_to_one')
    frame=frame.reset_index(drop=True);write_table(frame,out/'input_manifest.parquet')
    save_json(out/'protocol.json',{'config':cfg,'source':'full cohort A train/val only; test sealed',
        'input_sha256':hashlib.sha256((out/'input_manifest.parquet').read_bytes()).hexdigest(),
        'versions':version_info(),'source_code_sha256':{str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['src','scripts'] for p in (REPO/folder).rglob('*.py')},
        'release_status':'provisional; source-column task compliance; near-duplicate review pending'})
    rows=[]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn')) as pool:
        for result in pool.map(prepare_one,[(r,str(out)) for r in frame.to_dict('records')],chunksize=4):
            rows.append(result)
            if len(rows)%1000==0:write_table(pd.DataFrame(rows),out/'preparation_checkpoint.parquet');print('Prepared',len(rows),'/',len(frame),flush=True)
    result=pd.DataFrame(rows);write_table(result,out/'preparation_results.parquet')
    good=result[result.preprocess_status.eq('ok')].copy()
    good['initial_recording']=~good.duplicated('participant_id')
    if good.empty:raise RuntimeError('No usable development recordings')
    # Parquet supports nested QC arrays; development runner reads scalar columns only.
    write_table(good,out/'development_manifest.parquet')
    exclusions=result[result.preprocess_status.ne('ok')];write_table(exclusions,out/'exclusions.parquet')
    retention=[]
    for field in ['diagnosis','split','sex','phoneInfo','appVersion','medication']:
        if field not in result:continue
        for value,g in result.groupby(field,dropna=False):retention.append({'field':field,'value':str(value),'attempted':len(g),'retained':int(g.preprocess_status.eq('ok').sum())})
    if 'age' in result:
        result['age_band']=pd.cut(result.age,bins=[0,30,50,65,80,120]).astype(str)
        for value,g in result.groupby('age_band'):retention.append({'field':'age_band','value':value,'attempted':len(g),'retained':int(g.preprocess_status.eq('ok').sum())})
    write_table(pd.DataFrame(retention),out/'retention.csv')
    if 'pitch_detected_s' in good:
        write_table(good.groupby('diagnosis').agg(recordings=('recording_id','size'),pitch_eligible=('pitch_detected_s',lambda x:int(x.ge(2).sum()))).reset_index(),out/'pitch_sensitivity_retention.csv')
    manifest_digest=hashlib.sha256((out/'development_manifest.parquet').read_bytes()).hexdigest()
    save_json(out/'READY.json',{'attempted':len(result),'retained':len(good),'excluded':len(exclusions),
        'participants':good.participant_id.nunique(),'manifest_sha256':manifest_digest,'test_opened':False})
    print((out/'READY.json').read_text(),flush=True)
if __name__=='__main__':main()
