"""Signal-preserving development protocol. No test-manifest access."""
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np
from mpower_common import ffmpeg

SAMPLE_RATE=16000
MIN_SAMPLES=32000
MAX_SAMPLES=160000


def common_windows(waveform, context):
    span=np.asarray(waveform,dtype=np.float32)[:MAX_SAMPLES]
    if len(span)<MIN_SAMPLES:raise ValueError('below_2s_primary_minimum')
    if not np.isfinite(span).all():raise ValueError('nonfinite_primary_span')
    if context=='full':return [span],{'starts':[0],'real_samples':len(span),'summed_samples':len(span),'overlap_samples':0}
    if context!='matched_2s':raise ValueError('Unknown context policy')
    count=(len(span)+MIN_SAMPLES-1)//MIN_SAMPLES
    starts=np.rint(np.linspace(0,len(span)-MIN_SAMPLES,count)).astype(int)
    windows=[span[s:s+MIN_SAMPLES] for s in starts]
    return windows,{'starts':starts.tolist(),'real_samples':len(span),
                    'summed_samples':count*MIN_SAMPLES,'overlap_samples':count*MIN_SAMPLES-len(span)}


def longest_run(mask):
    edges=np.diff(np.r_[False,np.asarray(mask,dtype=bool),False].astype(int))
    starts=np.flatnonzero(edges==1);stops=np.flatnonzero(edges==-1)
    return int((stops-starts).max()) if len(starts) else 0


def native_channel_qc(samples):
    x=np.asarray(samples,dtype=float)
    if x.ndim==1:x=x[:,None]
    if not x.size or not np.isfinite(x).all():raise ValueError('empty_or_nonfinite_native_decode')
    channels=[]
    for channel in x.T:
        channels.append({'peak':float(np.abs(channel).max()),'rms':float(np.sqrt(np.mean(channel**2))),
            'dc':float(channel.mean()),'full_scale_fraction':float(np.mean(np.abs(channel)>=.999)),
            'max_full_scale_run_samples':longest_run(np.abs(channel)>=.999),
            'max_flat_top_run_samples':longest_run((np.abs(channel[1:]-channel[:-1])<=1e-8) & (np.abs(channel[1:])>=.95*np.abs(channel).max()) & (np.abs(channel[1:])>0)),
            'max_zero_run_samples':longest_run(np.abs(channel)<=1e-8)})
    denominator=np.mean([c['rms'] for c in channels])
    ratio=float(np.sqrt(np.mean(x.mean(axis=1)**2))/denominator) if denominator>0 else None
    correlations=[]
    for a in range(x.shape[1]):
        for b in range(a):
            if x[:,a].std()>0 and x[:,b].std()>0:correlations.append(float(np.corrcoef(x[:,a],x[:,b])[0,1]))
    return {'channels':channels,'downmix_rms_ratio':ratio,'cancellation_review':ratio is not None and ratio<.1,
            'minimum_channel_correlation':min(correlations) if correlations else None}


def decode_native_and_derive(path,rate,channels,max_seconds=180):
    args=[ffmpeg(),'-v','error','-nostdin','-i',str(path),'-t',str(max_seconds+1),'-f','f32le','-']
    native=subprocess.run(args,capture_output=True,timeout=120,check=True)
    x=np.frombuffer(native.stdout,dtype='<f4')
    if len(x)%channels:raise ValueError('native_channel_sample_mismatch')
    x=x.reshape(-1,channels)
    if len(x)>max_seconds*rate:raise ValueError('native_duration_exceeds_resource_limit')
    qc=native_channel_qc(x)
    # One decode from original media. Derive mono16k from that decoded native PCM.
    converted=subprocess.run([ffmpeg(),'-v','error','-f','f32le','-ar',str(rate),'-ac',str(channels),
        '-i','pipe:0','-ac','1','-ar','16000','-f','f32le','-'],input=native.stdout,
        capture_output=True,timeout=120,check=True)
    wave=np.frombuffer(converted.stdout,dtype='<f4').copy()
    if not wave.size or not np.isfinite(wave).all():raise ValueError('empty_or_nonfinite_derived_decode')
    metadata={'native_rate':rate,'native_channels':channels,'native_sample_count':len(x),
        'native_duration_s':len(x)/rate,'derived_sample_count':len(wave),
        'primary_sample_count':min(len(wave),MAX_SAMPLES),'capped_samples':max(0,len(wave)-MAX_SAMPLES),
        'native_channel_qc':qc,'decoder_warnings':native.stderr.decode(errors='replace')[-4000:],
        'resample_warnings':converted.stderr.decode(errors='replace')[-4000:],
        'mono16k_sha256':hashlib.sha256(wave.astype('<f4').tobytes()).hexdigest(),
        'frame_energy_dynamic_range_db_definition':'20log10(frame_RMS_P90/frame_RMS_P10); not environmental SNR'}
    common_windows(wave,'full')
    return wave[:MAX_SAMPLES],metadata


def participant_weights(ids):
    _,inverse,counts=np.unique(np.asarray(ids),return_inverse=True,return_counts=True)
    w=1/counts[inverse].astype(float)
    # Total weight equals the number of people, invariant to duplicated visits.
    return w


def development_frame(frame):
    # Filter BEFORE accessing labels. Unknown sealed rows cannot affect this path.
    d=frame.loc[frame.split.isin(['train','val'])].copy()
    if d.empty or d.groupby('participant_id').split.nunique().gt(1).any():raise ValueError('Invalid development split')
    if d.groupby('participant_id').diagnosis.nunique().gt(1).any():raise ValueError('Conflicting development labels')
    d['label']=d.diagnosis.map({'control':0,'pd':1})
    if d.label.isna().any():raise ValueError('Unresolved development labels')
    return d
