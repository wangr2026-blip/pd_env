#!/usr/bin/env python3
"""Audited baseline-survey linkage for the observed syn5511429 schema."""
import hashlib
import json
import numpy as np
import pandas as pd
from mpower_common import configure, load_table, parser, save_json, write_table
from audit_metadata import diagnosis


def link(records, people, demo, cfg):
    required={'healthCode','recordId','professional-diagnosis','age','gender','createdOn','diagnosis-year','medication-start-year'}
    if required-set(demo): raise ValueError(f'Observed demographic schema changed; inspect {required-set(demo)}')
    invalid=demo.healthCode.isna() | demo.healthCode.duplicated(keep=False)
    if invalid.any():
        raise ValueError('Demographic participant key missing/duplicated: resolve explicitly before many-to-one linkage')
    d=demo.rename(columns={c:'demo__'+c for c in demo})
    p=people.drop(columns=[c for c in people if c.startswith('demo__') or c in {'source_diagnosis','demographics_matched','label_conflict','flag_control_medication_timing','flag_pd_no_medication','diagnosis_year_valid','medication_year_valid','demographics_at'}],errors='ignore')
    p=p.merge(d,left_on='participant_id',right_on='demo__healthCode',how='left',validate='one_to_one',indicator='_demo_join')
    p['demographics_matched']=p._demo_join.eq('both'); p=p.drop(columns='_demo_join')
    p['source_diagnosis']=p['demo__professional-diagnosis'].map(lambda v:diagnosis(v,cfg))
    p['diagnosis']=p.source_diagnosis
    p['age']=pd.to_numeric(p['demo__age'],errors='coerce')
    p['age_invalid']=p.age.notna() & ~p.age.between(18,120)
    p['age_min']=p['age_max']=p.age
    p['age_varies']=False
    # Retain the survey's exact gender field; no inference of biological sex.
    p['sex']=p['demo__gender'].where(p['demo__gender'].isin(['Male','Female']))
    p['sex_conflict']=False
    diag_year=pd.to_numeric(p['demo__diagnosis-year'],errors='coerce')
    med_year=pd.to_numeric(p['demo__medication-start-year'],errors='coerce')
    p['diagnosis_year_valid']=diag_year.between(1901,2026)
    p['medication_year_valid']=med_year.between(1901,2026)
    timed=records.medication.fillna('').str.contains('Immediately before Parkinson|Just after Parkinson',regex=True)
    any_timed=records.loc[timed].groupby('participant_id').size()
    no_meds=records.medication.fillna('').eq("I don't take Parkinson medications")
    any_no_meds=records.loc[no_meds].groupby('participant_id').size()
    p['flag_control_medication_timing']=p.source_diagnosis.eq('control') & p.participant_id.isin(any_timed.index)
    p['flag_pd_no_medication']=p.source_diagnosis.eq('pd') & p.participant_id.isin(any_no_meds.index)
    # Conservative primary policy; source diagnosis is always retained for sensitivity analyses.
    p['label_conflict']=p.source_diagnosis.eq('control') & (p.diagnosis_year_valid | p.medication_year_valid | p.flag_control_medication_timing)
    p.loc[p.label_conflict,'diagnosis']='conflicting'
    p['demographics_at']=pd.to_datetime(p['demo__createdOn'],errors='coerce',utc=True,format='mixed')
    numeric=pd.to_numeric(p['demo__createdOn'],errors='coerce'); epoch=numeric.between(1e11,1e14)
    p.loc[epoch,'demographics_at']=pd.to_datetime(numeric[epoch],unit='ms',utc=True)
    columns=['participant_id','diagnosis','source_diagnosis','age','sex','age_invalid','label_conflict',
             'demographics_matched','flag_control_medication_timing','flag_pd_no_medication','demographics_at']
    r=records.drop(columns=[c for c in columns if c!='participant_id' and c in records])
    r=r.merge(p[columns],on='participant_id',how='left',validate='many_to_one')
    r['diagnosis']=r.diagnosis.fillna('unknown')
    r['survey_to_recording_days']=(r.recorded_at-r.demographics_at).dt.total_seconds()/86400
    return r,p


def main():
    args=parser(__doc__).parse_args(); cfg=configure(args.config); out=cfg['output_root']
    path=cfg['data_root']/cfg['demographics']
    if not path.exists():
        print('No local demographics export; no linkage performed',flush=True); return
    demo=load_table(path)
    # Persist all schema evidence before validating linkage; no participant rows printed.
    save_json(out/'eda/diagnostics/demographics_schema.json',{'rows':len(demo),'columns':list(demo),
        'fields':[{'name':c,'dtype':str(demo[c].dtype),'missing':int(demo[c].isna().sum()),'distinct':int(demo[c].nunique())} for c in demo],
        'head':demo.head().fillna('').to_dict('records'),
        'categorical_values':{c:demo[c].value_counts(dropna=False).head(100).rename_axis('value').reset_index(name='count').astype(str).to_dict('records') for c in demo if demo[c].nunique()<100},
        'duplicate_rows':int(demo.duplicated().sum()),'missing_keys':int(demo.healthCode.isna().sum()),
        'duplicate_participant_keys':int(demo.healthCode.duplicated(keep=False).sum()),
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    records=pd.read_parquet(out/'processed/recording_manifest.parquet'); people=pd.read_parquet(out/'processed/participant_manifest.parquet')
    bad=demo[demo.healthCode.isna() | demo.healthCode.duplicated(keep=False)]
    write_table(bad,out/'eda/diagnostics/demographic_join_conflicts.csv')
    r,p=link(records,people,demo,cfg)
    write_table(p[~p.demographics_matched],out/'eda/diagnostics/voice_participants_without_demographics.csv')
    write_table(demo[~demo.healthCode.isin(people.participant_id)],out/'eda/diagnostics/demographics_without_voice.csv')
    write_table(p[p.label_conflict | p.flag_pd_no_medication],out/'eda/diagnostics/diagnosis_medication_audit.csv')
    write_table(r,out/'processed/recording_manifest.parquet'); write_table(p,out/'processed/participant_manifest.parquet')
    for name,group in [('source_diagnosis',p.source_diagnosis),('diagnosis',p.diagnosis)]:
        write_table(p.groupby(name).agg(participants=('participant_id','size'),recordings=('recordings','sum')).reset_index(),out/f'eda/tables/{name}_cohort_counts.csv')
    write_table(p.groupby('diagnosis').agg(participants=('participant_id','size'),recordings=('recordings','sum')).reset_index(),out/'eda/tables/cohort_characteristics.csv')
    write_table(p.groupby('diagnosis').agg(age_available=('age','count'),sex_available=('sex','count'),longitudinal=('recordings',lambda v:int(v.gt(1).sum()))).reset_index(),out/'eda/tables/demographic_coverage.csv')
    write_table(r.groupby(['diagnosis','medication'],dropna=False).size().reset_index(name='recordings'),out/'eda/tables/medication_by_diagnosis.csv')
    summary_path=out/'eda/tables/discovery_summary.json'; d=json.loads(summary_path.read_text())
    d.update(participant_diagnoses=p.diagnosis.value_counts().to_dict(),recording_diagnoses=r.diagnosis.value_counts().to_dict(),
             source_participant_diagnoses=p.source_diagnosis.value_counts().to_dict(),source_recording_diagnoses=r.source_diagnosis.value_counts().to_dict(),
             demographics_matched_participants=int(p.demographics_matched.sum()),demographics_unmatched_participants=int((~p.demographics_matched).sum()),
             demographics_matched_recordings=int(r.demographics_matched.fillna(False).sum()),label_conflict_participants=int(p.label_conflict.sum()),
             demographics_table='syn5511429',diagnosis_definition='Participant-reported professional diagnosis; not independently clinically adjudicated',
             blockers=['Full-audio processing, near-duplicate review and label/QC sensitivity remain pending'])
    d['relationships']['demographics']='syn5511429 baseline linked by healthCode; one-to-one participants / many-to-one recordings; duplicates fail closed; no temporal clinical-survey matching'
    save_json(summary_path,d)
    save_json(out/'eda/diagnostics/demographic_join_audit.json',{'source_rows':len(demo),'unique_participants':int(demo.healthCode.nunique()),
        'matched_voice_participants':int(p.demographics_matched.sum()),'unmatched_voice_participants':int((~p.demographics_matched).sum()),
        'matched_voice_recordings':int(r.demographics_matched.fillna(False).sum()),'unmatched_demographics':int((~demo.healthCode.isin(p.participant_id)).sum()),
        'participant_count_before':len(people),'participant_count_after':len(p),'recording_count_before':len(records),'recording_count_after':len(r),
        'cardinality':'one-to-one participant; many-to-one recording','clinical_timing':'baseline age/gender only; signed survey-to-recording days retained; no severity join',
        'label_policy':'false diagnosis plus valid diagnosis/medication year or explicit pre/post-PD-medication voice response => conflicting; source labels retained'})
    print(f'Demographic linkage: {int(p.demographics_matched.sum())}/{len(p)} voice participants; no row expansion',flush=True)


if __name__=='__main__':main()
