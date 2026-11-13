#!/usr/bin/env python3
"""Explicit cohort eligibility, exclusions, and exploratory threshold sensitivity."""
import numpy as np
import pandas as pd
from mpower_common import configure, parser, save_json, write_table


def main():
    p=parser(__doc__); p.add_argument('--scope',choices=['pilot','full'],default='pilot'); args=p.parse_args()
    cfg=configure(args.config); out=cfg['output_root']; scope=args.scope
    records=pd.read_parquet(out/'processed/recording_manifest.parquet')
    people=pd.read_parquet(out/'processed/participant_manifest.parquet')
    qc=pd.read_parquet(out/f'processed/{scope}_audio_quality_metrics.parquet')
    feature_path=out/f'processed/{scope}_acoustic_features.parquet'
    features=pd.read_parquet(feature_path) if feature_path.exists() else pd.DataFrame({'recording_id':pd.Series(dtype=str)})
    for metric,name,threshold in [('praat_full__pitch_valid_fraction','flag_low_pitch_coverage',cfg['qc']['min_pitch_fraction'])]:
        if metric in features:
            mapping=features.set_index('recording_id')[metric]
            values=qc.recording_id.map(mapping)
            qc[name]=values.notna() & values.lt(threshold)
    if 'praat_full__f0_cv' in features:
        qc['flag_pitch_cv_review']=qc.recording_id.map(features.set_index('recording_id')['praat_full__f0_cv']).gt(.2)
    write_table(qc,out/f'processed/{scope}_audio_quality_metrics.parquet')
    r=records.merge(qc,on='recording_id',how='left',validate='one_to_one')
    if 'pitch_detected_s' in features:
        r=r.merge(features[['recording_id','pitch_detected_s']],on='recording_id',how='left',validate='one_to_one')
    criteria={
        'missing_identity':r.missing_identity,
        'duplicate_source_identity':r.duplicate_recording_id,
        'ambiguous_diagnosis':~r.diagnosis.isin(['pd','control']),
        'not_processed':r.decode_status.isna(),
        'decode_failure':r.decode_status.notna() & ~r.decode_status.eq('ok'),
        'non_phonation':~r.task.eq('sustained_phonation'),
        'near_silent':r.get('flag_near_silent',pd.Series(True,index=r.index)).fillna(True),
        'short_duration':r.get('flag_short',pd.Series(True,index=r.index)).fillna(True),
        'insufficient_pitch_voicing':r.get('pitch_detected_s',pd.Series(np.nan,index=r.index)).fillna(0)<cfg['qc']['min_voiced_s'],
    }
    primary=['missing_identity','duplicate_source_identity','ambiguous_diagnosis','not_processed','decode_failure']
    qc_names=primary+['non_phonation','near_silent','short_duration','insufficient_pitch_voicing']
    exclusions=[]; flow=[]; keep=pd.Series(True,index=r.index)
    for name,mask in criteria.items():
        for row in r.loc[mask,['recording_id','participant_id','diagnosis']].to_dict('records'):
            exclusions.append({**row,'criterion':name,'scope':scope})
        if name in qc_names:
            keep &= ~mask
            for label,g in r.groupby('diagnosis'):
                surviving=g.loc[keep.loc[g.index]]
                flow.append({'criterion':name,'diagnosis':label,'remaining_recordings':len(surviving),
                             'remaining_participants':surviving.participant_id.nunique(),'scope':scope})
    write_table(pd.DataFrame(exclusions,columns=['recording_id','participant_id','diagnosis','criterion','scope']),out/'processed/cohort_exclusions.csv')
    write_table(pd.DataFrame(flow),out/'eda/tables/qc_exclusion_flow.csv')
    a=~pd.concat([criteria[k] for k in primary],axis=1).any(axis=1)
    b=~pd.concat([criteria[k] for k in qc_names],axis=1).any(axis=1)
    matching=people[people.participant_id.isin(r.loc[b,'participant_id']) & people.age.between(*cfg['age_restriction'])
                    & ~people.age_invalid & ~people.sex_conflict & people.sex.isin(['Male','Female'])].copy()
    matching['age_bin']=(matching.age//5).astype(int)
    selected_ids=[]; match_flow=[]
    rng=np.random.default_rng(cfg['seed'])
    for (sex,age_bin),g in matching.groupby(['sex','age_bin'],sort=True):
        groups={dx:g[g.diagnosis.eq(dx)].sort_values('participant_id') for dx in ('pd','control')}
        count=min(len(groups['pd']),len(groups['control']))
        for dx,group in groups.items():
            selected_ids.extend(rng.choice(group.participant_id.to_numpy(),size=count,replace=False).tolist())
            match_flow.append({'gender':sex,'age_bin_start':age_bin*5,'diagnosis':dx,'available_participants':len(group),'retained_participants':count})
    c=b & r.participant_id.isin(selected_ids)
    write_table(pd.DataFrame(match_flow),out/'eda/tables/age_gender_matching_flow.csv')
    counts={}
    for cohort,mask in [('A',a),('B',b),('C',c)]:
        selected=r[mask].copy(); d=out/f'processed/cohort_{cohort}'
        write_table(selected,d/'recording_manifest.parquet')
        write_table(people[people.participant_id.isin(selected.participant_id)],d/'participant_manifest.parquet')
        write_table(features[features.recording_id.isin(selected.recording_id)],d/'acoustic_features.parquet')
        write_table(qc[qc.recording_id.isin(selected.recording_id)],d/'audio_quality_metrics.parquet')
        counts[cohort]={'recordings':len(selected),'participants':selected.participant_id.nunique(),'scope':scope}
        for split in ('train','val','test'):
            write_table(selected.iloc[:0],d/f'{split}_manifest.csv')
    # Report technical sensitivity on processed rows even if diagnosis is unavailable.
    sensitivity=[]
    for duration in (2,3,5,7,9):
        for voicing in (1,2,3):
            mask=r.decode_status.eq('ok') & (r.duration_s>=duration)
            if 'pitch_detected_s' in r: mask &= r.pitch_detected_s>=voicing
            else: mask &= False
            for label,g in r.groupby('diagnosis'):
                s=g[mask.loc[g.index]]
                sensitivity.append({'min_duration_s':duration,'min_pitch_voiced_s':voicing,'diagnosis':label,'recordings':len(s),'participants':s.participant_id.nunique(),'scope':scope})
    write_table(pd.DataFrame(sensitivity),out/'eda/tables/qc_threshold_sensitivity.csv')
    save_json(out/'eda/tables/cohort_readiness.json',{'cohorts':counts,
        'release_status':'provisional processed-scope cohorts; full duplicate/QC audit pending' if a.any() else 'blocked: no eligible labels/decoded audio',
        'cohort_C':'Age 50–80; equal participant counts within 5-year age bins and reported gender; fixed-seed subsampling; residual age imbalance must be assessed.',
        'qc_policy':'Low amplitude, estimated SNR, clipping and unstable pitch are review flags; none alone removes a recording. Pitch-voicing criterion is exploratory and requires disease-bias sensitivity review.'})


if __name__=='__main__': main()
