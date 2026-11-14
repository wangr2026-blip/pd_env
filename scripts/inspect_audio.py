#!/usr/bin/env python3
"""Restartable bounded-memory inspection/extraction. Pilot by default; --full explicit."""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
import multiprocessing as mp
import faulthandler
import hashlib
import json
import logging
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import pandas as pd

from mpower_common import cache_key, configure, decode, ffmpeg, parser, save_json, version_info, write_table


def isolated_call(function, *args):
    """Contain even a native abort/segfault to one process and surface it as an error."""
    executor=ProcessPoolExecutor(max_workers=1,mp_context=mp.get_context('spawn'),initializer=faulthandler.enable)
    try:
        return executor.submit(function,*args).result(timeout=300)
    except Exception:
        for process in (executor._processes or {}).values():
            if process.is_alive(): process.terminate()
        raise
    finally:
        executor.shutdown(wait=True,cancel_futures=True)


def batch_results(batch,cfg,stage,scope,pool):
    """Return successful results and rows needing isolated native-crash recovery."""
    futures={}; completed={}; broken=False
    for row in batch:
        try: futures[pool.submit(work,row,cfg,stage,scope)]=row
        except BrokenProcessPool:
            broken=True; break
    try:
        for future in as_completed(futures,timeout=600):
            row=futures[future]
            try: completed[row['recording_id']]=future.result()
            except BrokenProcessPool: broken=True
            except Exception as exc:
                completed[row['recording_id']]={'recording_id':row['recording_id'],'decode_status':'error','feature_status':'error','error':str(exc),'scope':scope}
    except TimeoutError:
        broken=True
    pending=[row for row in batch if row['recording_id'] not in completed]
    return list(completed.values()),pending,broken


def choose_pilot(frame, n, seed):
    """Round-robin deterministic sampling across observed diagnosis and medication strata.

    One recording/participant first; fill remaining slots from repeat recordings.
    This is a coverage pilot, not a prevalence sample.
    """
    f=frame[frame.audio_exists].copy()
    f=f.sort_values('recording_id').sample(frac=1,random_state=seed)
    f['_stratum']=f.diagnosis.fillna('unknown')+'|'+f.medication.fillna('missing')
    f['_repeat']=f.duplicated('participant_id')
    f['_rank']=f.groupby(['_repeat','_stratum']).cumcount()
    return f.sort_values(['_repeat','_rank','_stratum']).head(n).drop(columns=['_stratum','_rank','_repeat'])


def probe(path):
    binary=shutil.which('ffprobe')
    if binary:
        p=subprocess.run([binary,'-v','error','-show_format','-show_streams','-of','json',str(path)],capture_output=True,text=True,check=True,timeout=30)
        info=json.loads(p.stdout)
        stream=next(s for s in info['streams'] if s['codec_type']=='audio')
        fmt=info['format']
        return {'original_format':fmt.get('format_name'),'original_codec':stream.get('codec_name'),
            'original_sample_rate':int(stream['sample_rate']),'original_channels':int(stream['channels']),
            'original_bit_depth':stream.get('bits_per_raw_sample',stream.get('bits_per_sample')),
            'original_bitrate':stream.get('bit_rate',fmt.get('bit_rate')),
            'container_duration_s':float(fmt.get('duration',stream.get('duration','nan'))),'probe_method':'ffprobe'}
    # FFmpeg bundled executable is available even when no system ffprobe is installed.
    p=subprocess.run([ffmpeg(),'-hide_banner','-nostdin','-i',str(path)],capture_output=True,text=True,timeout=30)
    text=p.stderr
    s=re.search(r'Audio: ([^,]+).*?, (\d+) Hz, ([^,\n]+)',text)
    d=re.search(r'Duration: (\d+):(\d+):([\d.]+)',text)
    form=re.search(r'Input #0, (.*?), from ',text)
    bitrate=re.search(r'bitrate: (\d+) kb/s',text)
    return {'original_format':form.group(1) if form else None,'original_codec':s.group(1) if s else None,
        'original_sample_rate':int(s.group(2)) if s else None,
        'original_channels':({'mono':1,'stereo':2}.get(s.group(3).strip())) if s else None,
        'original_bit_depth':None,'original_bitrate':int(bitrate.group(1))*1000 if bitrate else None,
        'container_duration_s':int(d.group(1))*3600+int(d.group(2))*60+float(d.group(3)) if d else None,
        'probe_method':'ffmpeg_header; bit_depth unavailable'}


def work(row,cfg,stage,scope):
    rid=row['recording_id']
    result={'recording_id':rid,'scope':scope,'decode_status':'error','feature_status':'not_requested'}
    try:
        key=cache_key(row,cfg,stage)
        target=cfg['output_root']/'cache'/f'{key}.json'
        if target.exists():
            cached=json.loads(target.read_text())
            if cached.get('decode_status')=='ok' and (stage!='features' or cached.get('feature_status')=='ok'):
                return {**cached,'recording_id':rid,'scope':scope,'cache_hit':True}
        path=Path(row['audio_path'])
        result.update(probe(path))
        result['file_size_bytes']=path.stat().st_size
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
        result['file_sha256']=h.hexdigest()
        y,sr=decode(path,cfg)
        # Analysis-normalized decoded hash: conservative duplicate signal flag.
        result['normalized_pcm_sha256']=hashlib.sha256(y.astype('<f4').tobytes()).hexdigest()
        # Full native-rate, native-channel decode hash; no sample-rate/channel conversion.
        native=subprocess.run([ffmpeg(),'-v','error','-nostdin','-i',str(path),'-t',str(cfg['max_decode_seconds']+1),'-f','hash','-hash','sha256','-c:a','pcm_f32le','-'],capture_output=True,text=True,timeout=120,check=True)
        result['native_pcm_sha256']=hashlib.sha256((str(result.get('original_sample_rate'))+'|'+str(result.get('original_channels'))+'|'+native.stdout.strip()).encode()).hexdigest()
        from audio_quality import quality
        result.update(quality(y,sr,cfg))
        result['decode_status']='ok'
        if stage=='features':
            from extract_acoustic_features import extract,sensitivity
            features,statuses,_=extract(y,sr,cfg)
            result.update(features)
            result['feature_status']='ok' if all(v=='ok' for k,v in statuses.items() if k!='praat_stable') else 'partial'
            result.update({'status__'+k:v for k,v in statuses.items()})
            if scope=='pilot': result.update(sensitivity(y,sr,cfg))
        result['cache_hit']=False
        # JSON standard does not support NaN; preserve undefined values as null.
        result={k:(None if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v) for k,v in result.items()}
        save_json(target,result)
    except Exception as exc:
        if stage == 'features':
            result['feature_status'] = 'error'
        result['error']=f'{type(exc).__name__}: {str(exc)[:1200]}'
    return result


def main(default_stage='qc'):
    p=parser(__doc__)
    p.add_argument('--stage',choices=['qc','features'],default=default_stage)
    p.add_argument('--full',action='store_true')
    p.add_argument('--limit',type=int)
    p.add_argument('--workers',type=int)
    args=p.parse_args()
    cfg=configure(args.config)
    out=cfg['output_root']
    logging.basicConfig(filename=out/'eda/diagnostics/extraction.log',level=logging.INFO)
    manifest=pd.read_parquet(out/'processed/recording_manifest.parquet')
    scope='full' if args.full else 'pilot'
    n=args.limit or cfg['pilot_size']
    selected=manifest if args.full else choose_pilot(manifest,n,cfg['seed'])
    write_table(selected,out/f'processed/{scope}_selection.parquet')
    save_json(out/f'eda/diagnostics/{scope}_execution.json',{'scope':scope,'stage':args.stage,'selected':len(selected),
        'config_digest':cfg['config_digest'],'seed':cfg['seed'],'versions':version_info(),
        'source_metadata_sha256':hashlib.sha256((cfg['data_root']/cfg['metadata']).read_bytes()).hexdigest(),
        'script_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        'sampling':'all source rows' if args.full else 'fixed-seed round-robin diagnosis/medication strata; unique participants first'})
    rows=[]
    # Keep task submission bounded to avoid a future per recording for large cohorts.
    workers=args.workers if args.workers is not None else cfg['workers']
    if workers < 1: raise ValueError('--workers must be positive')
    def make_pool():
        # Avoid forking an initialized Arrow/NumPy/native-library thread state.
        return ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),
                                   max_tasks_per_child=250,initializer=faulthandler.enable)
    pool=make_pool()
    try:
        for offset in range(0,len(selected),max(1,workers*4)):
            batch=selected.iloc[offset:offset+workers*4]
            values,pending,broken=batch_results(batch.to_dict('records'),cfg,args.stage,scope,pool)
            if broken:
                logging.error('Native worker failure: %d rows retried in isolated processes',len(pending))
                for process in (pool._processes or {}).values():
                    if process.is_alive(): process.terminate()
                pool.shutdown(wait=True,cancel_futures=True)
                # Independent child processes contain repeat crashes, while the
                # thread coordinator keeps recovery parallel across allocated CPUs.
                with ThreadPoolExecutor(max_workers=workers) as recovery:
                    retries={recovery.submit(isolated_call,work,row,cfg,args.stage,scope):row for row in pending}
                    for future in as_completed(retries):
                        row=retries[future]
                        try: value=future.result()
                        except Exception as exc:
                            value={'recording_id':row['recording_id'],'decode_status':'worker_error','feature_status':'error',
                                   'scope':scope,'error':f'isolated_native_worker_failure: {type(exc).__name__}: {exc}'}
                        values.append(value)
                pool=make_pool()
            for value in values:
                rows.append(value)
                if value.get('error'): logging.error('%s %s',value['recording_id'],value['error'])
            native_failures=sum(v.get('decode_status')=='worker_error' for v in values)
            if native_failures >= max(2, len(batch)//2):
                write_table(pd.DataFrame(rows),out/f'processed/{scope}_{args.stage}_checkpoint.parquet')
                raise RuntimeError(f'Systematic native worker failures ({native_failures}/{len(batch)}); '
                                   'checkpoint saved; fix the environment/extractor before resuming')
            interval=max(1,cfg.get('checkpoint_every',1000))
            checkpoint=(offset==0 or scope=='pilot' or len(rows)//interval>(len(rows)-len(batch))//interval or len(rows)==len(selected))
            if checkpoint:
                write_table(pd.DataFrame(rows),out/f'processed/{scope}_{args.stage}_checkpoint.parquet')
                print(f'{scope} {args.stage}: {len(rows)}/{len(selected)}; decoded {sum(r.get("decode_status")=="ok" for r in rows)}',flush=True)
    finally:
        pool.shutdown(wait=True,cancel_futures=True)
    result=pd.DataFrame(rows).sort_values('recording_id')
    id_cols=['recording_id','scope','decode_status','error','feature_status','cache_hit']
    feature_cols=[c for c in result if '__' in c or c.startswith(('interval_','pitch_detected'))]
    qc_cols=[c for c in result if c not in feature_cols]
    write_table(result[qc_cols],out/f'processed/{scope}_audio_quality_metrics.parquet')
    if args.stage=='features':
        write_table(result[[c for c in id_cols+feature_cols if c in result]],out/f'processed/{scope}_acoustic_features.parquet')
    print(f'Finished {scope}: {len(result)} attempted recordings',flush=True)


if __name__=='__main__': main()
