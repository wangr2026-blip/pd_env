"""Synthetic tests only: no participant data or real recordings."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_metadata import canonicalize, diagnosis
from create_participant_splits import assign_splits, verify
from audio_quality import quality
from extract_acoustic_features import extract
from inspect_audio import work, choose_pilot
from mpower_common import configure
import yaml


@pytest.fixture
def cfg(tmp_path):
    c=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/mpower_preprocessing.yaml').read_text())
    c.update(data_root=tmp_path, output_root=tmp_path/'out', config_digest='synthetic')
    (c['output_root']/'cache').mkdir(parents=True)
    return c


def test_identity_and_missing_labels_preserved(cfg):
    f=pd.DataFrame({'healthCode':['synthetic-a','synthetic-a','synthetic-b'],
        'recordId':['r1','r1','r3'],'audio_path':['missing.wav']*3,'audio_audio.m4a':['1','1','3']})
    r,p=canonicalize(f,cfg)
    assert len(r)==3 and r.recording_id.nunique()==3
    assert r.duplicate_recording_id.tolist()==[True,True,False]
    assert r.diagnosis.eq('unknown').all()
    assert p.recordings.sum()==len(r)


def test_longitudinal_label_conflict(cfg):
    cfg['columns']['diagnosis']='dx'; cfg['diagnosis_values']={'pd':['yes'],'control':['no']}
    f=pd.DataFrame({'healthCode':['synthetic-a']*2,'recordId':['r1','r2'],'audio_path':['x','y'],'audio_audio.m4a':['1','2'],'dx':['yes','no']})
    r,p=canonicalize(f,cfg)
    assert p.diagnosis.tolist()==['conflicting']
    assert r.diagnosis.eq('conflicting').all()


def test_missing_configured_key_fails(cfg):
    with pytest.raises(ValueError): canonicalize(pd.DataFrame({'participant':['x']}),cfg)


def test_duplicate_linked_participants_cannot_leak():
    rows=[]
    for i in range(60):
        for j in range(2): rows.append({'participant_id':f'synthetic-{i}','recording_id':f'{i}-{j}',
           'diagnosis':'pd' if i<30 else 'control','file_sha256':f'{i}-{j}',
           'native_pcm_sha256':f'{i}-{j}'})
    f=pd.DataFrame(rows); f.loc[f.participant_id.isin(['synthetic-0','synthetic-31']),'native_pcm_sha256']='shared'
    p,r=assign_splits(f,123)
    assert set(p.split)=={'train','val','test'}
    assert p[p.participant_id.isin(['synthetic-0','synthetic-31'])].split.nunique()==1
    verify(r)
    p2,r2=assign_splits(f,123)
    pd.testing.assert_frame_equal(p,p2)
    r.loc[r.index[0],'split']='invalid'
    with pytest.raises(AssertionError): verify(r)


def test_sine_and_silence_acoustics(cfg):
    sr=16000; t=np.arange(sr*3)/sr
    y=(.2*np.sin(2*np.pi*150*t)).astype(np.float32)
    q=quality(y,sr,cfg)
    assert q['duration_s']==3 and not q['flag_near_silent']
    f,s,_=extract(y,sr,cfg)
    assert abs(f['praat_full__f0_mean_hz']-150)<2
    assert s['opensmile']=='ok' and s['spectral']=='ok'
    assert len([k for k in f if k.startswith('egemaps__')])==88
    f2,_,_=extract(y,sr,cfg)
    np.testing.assert_allclose(list(f.values()),list(f2.values()),equal_nan=True)
    silence=np.zeros(sr*3,dtype=np.float32)
    assert quality(silence,sr,cfg)['flag_near_silent']
    f,_,_=extract(silence,sr,cfg)
    assert np.isnan(f['praat_full__jitter_local'])
    assert np.isnan(f['praat_full__shimmer_local'])


def test_failed_audio_is_recorded(cfg):
    r=work({'recording_id':'synthetic-bad','audio_path':str(cfg['data_root']/'missing.wav')},cfg,'features','pilot')
    assert r['decode_status']=='error' and 'error' in r


def test_pipeline_end_to_end(tmp_path):
    """Tiny synthetic file-backed smoke test, including reports and cohort manifests."""
    import os, subprocess
    import soundfile as sf
    root=tmp_path/'raw'; output=tmp_path/'private'; (root/'voice_activity/audio').mkdir(parents=True)
    rows=[]; sr=16000
    for i in range(2):
        audio=root/f'voice_activity/audio/synthetic-{i}.wav'
        t=np.arange(sr*3)/sr
        sf.write(audio,.2*np.sin(2*np.pi*(140+20*i)*t),sr)
        rows.append({'recordId':f'synthetic-r{i}','healthCode':f'synthetic-p{i}',
                     'audio_audio.m4a':str(i),'audio_path':str(audio),'medTimepoint':'synthetic-only',
                     'createdOn':'2025-01-01T00:00:00Z','dx':'yes' if i else 'no'})
    pd.DataFrame(rows).to_csv(root/'voice_activity/Voice_Activity.csv',index=False)
    (root/'demographics').mkdir()
    pd.DataFrame({'healthCode':['synthetic-p0','synthetic-p1'],'recordId':['demo-r0','demo-r1'],
        'professional-diagnosis':['False','True'],'age':[55,60],'gender':['Male','Female'],
        'createdOn':['2025-01-01T00:00:00Z']*2,'diagnosis-year':[None,2020],
        'medication-start-year':[None,2021]}).to_csv(root/'demographics/Demographics.csv',index=False)
    repo=Path(__file__).resolve().parents[1]
    config=yaml.safe_load((repo/'configs/mpower_preprocessing.yaml').read_text())
    config['columns']['diagnosis']='dx'; config['diagnosis_values']={'pd':['yes','true'],'control':['no','false']}
    config['pilot_size']=2; config['workers']=1
    path=tmp_path/'config.yaml'; path.write_text(yaml.safe_dump(config))
    env={**os.environ,'MPOWER_DATA_ROOT':str(root),'MPOWER_OUTPUT_ROOT':str(output),
         'MPLCONFIGDIR':str(tmp_path/'mpl'),'NUMBA_CACHE_DIR':str(tmp_path/'numba')}
    for script in ('audit_metadata.py','link_demographics.py','extract_acoustic_features.py','build_cohorts.py','create_participant_splits.py','report_characterization.py'):
        result=subprocess.run([sys.executable,str(repo/'scripts'/script),'--config',str(path)],env=env,capture_output=True,text=True)
        assert result.returncode==0,result.stdout+'\n'+result.stderr
    manifest=pd.read_parquet(output/'processed/cohort_A/recording_manifest.parquet')
    assert len(manifest)==2
    assert (output/'eda/MPOWER_FINAL_REPORT.md').is_file()
    assert (output/'eda/figures/audio_duration.svg').is_file()
    assert len(pd.read_parquet(output/'processed/pilot_acoustic_features.parquet'))==2


def test_demographic_link_integrity_and_label_policy(cfg):
    from link_demographics import link
    cfg['diagnosis_values']={'pd':['true'],'control':['false']}
    source=pd.DataFrame({'healthCode':['synthetic-a','synthetic-b','synthetic-c'],
        'recordId':['r1','r2','r3'],'audio_path':['x','y','z'],'audio_audio.m4a':['1','2','3'],
        'createdOn':['2015-03-01T00:00:00Z']*3,
        'medTimepoint':['Another time','Just after Parkinson medication (at your best)','Another time']})
    r,p=canonicalize(source,cfg)
    demo=pd.DataFrame({'healthCode':['synthetic-a','synthetic-b'], 'recordId':['d1','d2'],
        'professional-diagnosis':['True','False'],'age':['60','55'],'gender':['Female','Male'],
        'createdOn':['2015-02-28T00:00:00Z']*2,'diagnosis-year':['2010',None],'medication-start-year':['2011','0']})
    linked,participants=link(r,p,demo,cfg)
    assert len(linked)==len(r) and len(participants)==len(p)
    assert linked.set_index('participant_id').diagnosis.to_dict()=={'synthetic-a':'pd','synthetic-b':'conflicting','synthetic-c':'unknown'}
    assert participants.demographics_matched.sum()==2
    with pytest.raises(ValueError): link(r,p,pd.concat([demo,demo.iloc[:1]]),cfg)


def synthetic_native_crash():
    import os
    os._exit(17)


def test_native_crash_is_contained():
    from inspect_audio import isolated_call
    from concurrent.futures.process import BrokenProcessPool
    with pytest.raises(BrokenProcessPool): isolated_call(synthetic_native_crash)
    assert isolated_call(abs,-7)==7
