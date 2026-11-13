#!/usr/bin/env python3
"""Participant-disjoint, duplicate-component-aware fixed-seed split assignment."""
import numpy as np
import pandas as pd
from mpower_common import configure, parser, save_json, write_table


def assign_splits(recordings, seed, fractions=(.7,.15,.15)):
    if not np.isclose(sum(fractions),1) or min(fractions)<=0:
        raise ValueError('Split fractions must be positive and sum to one')
    if recordings.participant_id.isna().any(): raise ValueError('Missing participant identity')
    labels=recordings.groupby('participant_id').diagnosis.nunique()
    if labels.gt(1).any(): raise ValueError('Conflicting participant labels')
    ids=sorted(recordings.participant_id.unique())
    parent={x:x for x in ids}
    def find(x):
        while x!=parent[x]:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b): parent[find(b)]=find(a)
    for col in ('file_sha256','native_pcm_sha256','normalized_pcm_sha256','near_duplicate_group'):
        if col not in recordings: continue
        for _,g in recordings.dropna(subset=[col]).groupby(col):
            members=g.participant_id.unique()
            for x in members[1:]: union(members[0],x)
    p=recordings[['participant_id','diagnosis']].drop_duplicates().copy()
    p['duplicate_component']=p.participant_id.map(find)
    p['split']=''
    rng=np.random.default_rng(seed)
    totals={k:np.zeros(3,dtype=int) for k in p.diagnosis.unique()}
    target={k:np.array(fractions)*int(p.diagnosis.eq(k).sum()) for k in totals}
    components=[g for _,g in p.groupby('duplicate_component',sort=True)]
    rng.shuffle(components)
    components.sort(key=len,reverse=True)
    for group in components:
        counts=group.diagnosis.value_counts()
        costs=[]
        for index in range(3):
            cost=0
            for label,number in counts.items():
                before=(totals[label]-target[label])**2/(target[label]+1)
                new=totals[label].copy(); new[index]+=number
                cost+=float(np.sum((new-target[label])**2/(target[label]+1)-before))
            costs.append(cost)
        i=int(np.argmin(costs))
        p.loc[group.index,'split']=('train','val','test')[i]
        for label,number in counts.items(): totals[label][i]+=number
    merged=recordings.merge(p[['participant_id','split','duplicate_component']],on='participant_id',validate='many_to_one')
    verify(merged)
    return p,merged


def verify(frame):
    for col in ('participant_id','recording_id','file_sha256','native_pcm_sha256','normalized_pcm_sha256','near_duplicate_group'):
        if col in frame and frame.dropna(subset=[col]).groupby(col).split.nunique().gt(1).any():
            raise AssertionError(f'Leakage across splits: {col}')


def main():
    args=parser(__doc__).parse_args(); cfg=configure(args.config); out=cfg['output_root']
    manifest=pd.read_parquet(out/'processed/cohort_A/recording_manifest.parquet')
    if manifest.empty:
        save_json(out/'eda/diagnostics/split_status.json',{'status':'blocked','reason':'No eligible binary-labeled decoded recordings; no valid training splits'})
        return
    p,r=assign_splits(manifest,cfg['seed'],cfg['split_fractions'])
    write_table(p,out/'processed/participant_splits.parquet')
    for cohort in ('A','B','C'):
        d=out/f'processed/cohort_{cohort}'
        f=pd.read_parquet(d/'recording_manifest.parquet')
        f=f.merge(p[['participant_id','split','duplicate_component']],on='participant_id',how='left',validate='many_to_one')
        if f.split.isna().any(): raise AssertionError('Cohort contains unassigned participant')
        verify(f)
        for split in ('train','val','test'):
            write_table(f[f.split.eq(split)],d/f'{split}_manifest.csv')
        write_table(f.groupby(['split','diagnosis']).agg(recordings=('recording_id','size'),participants=('participant_id','nunique')).reset_index(),out/f'eda/tables/cohort_{cohort}_splits.csv')
    save_json(out/'eda/diagnostics/split_status.json',{'status':'provisional','seed':cfg['seed'],
        'reason':'Duplicate-connected participants kept together; full near-duplicate review required before release',
        'empty_partitions':[x for x in ('train','val','test') if not r.split.eq(x).any()],
        'demographic_stratification':'not implemented; age-restricted sensitivity cohort uses same assignments',
        'learned_transformations':'none fitted'})


if __name__=='__main__': main()
