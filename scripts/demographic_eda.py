#!/usr/bin/env python3
"""Participant/recording demographic EDA and participant-aware adjusted associations."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from mpower_common import summary, write_table, save_json


def run(cfg,scope):
    out=cfg['output_root']; tables=out/'eda/tables'; figs=out/'eda/figures'
    p=pd.read_parquet(out/'processed/participant_manifest.parquet')
    if 'source_diagnosis' not in p:return
    r=pd.read_parquet(out/'processed/recording_manifest.parquet')
    from report_characterization import export
    pd_rows=[]
    for level,frame in [('participant',p),('recording',r)]:
        for dx,g in frame.groupby('diagnosis'):
            pd_rows.append({'level':level,'diagnosis':dx,'n':len(g),'age_missing':int(g.age.isna().sum()),
                            'sex_missing':int(g.sex.isna().sum()),**{'age_'+k:v for k,v in summary(g.age).items()}})
    write_table(pd.DataFrame(pd_rows),tables/'demographics_by_diagnosis.csv')
    p['age_band']=pd.cut(p.age,bins=[17,29,39,49,59,69,79,120],right=True).astype('string')
    write_table(p.groupby(['diagnosis','sex','age_band'],dropna=False).size().reset_index(name='participants'),tables/'age_sex_diagnosis_contingency.csv')
    binary=p[p.diagnosis.isin(['pd','control']) & ~p.age_invalid]
    x=binary.loc[binary.diagnosis.eq('pd'),'age'].dropna().astype(float); y=binary.loc[binary.diagnosis.eq('control'),'age'].dropna().astype(float)
    pooled=np.sqrt((x.var()+y.var())/2)
    sex=pd.crosstab(binary.diagnosis,binary.sex,normalize='index')
    save_json(tables/'demographic_imbalance.json',{'participant_age_mean_difference':float(x.mean()-y.mean()),
         'participant_age_median_difference':float(x.median()-y.median()),'age_smd':float((x.mean()-y.mean())/pooled) if np.isfinite(pooled) and pooled>0 else None,
         'gender_proportions':sex.to_dict(),'definition':'baseline survey age and reported gender; primary labels exclude contradictions',
         'pd_participants_with_age':len(x),'control_participants_with_age':len(y)})
    fig,ax=plt.subplots(1,3,figsize=(14,4))
    groups=[]; names=[]
    for dx,g in binary.groupby('diagnosis'):
        v=g.age.dropna(); groups.append(v); names.append(dx)
        ax[0].hist(v,bins=np.arange(15,96,5),alpha=.5,label=dx,density=True)
        if len(v)>3 and v.nunique()>2: v.plot.kde(ax=ax[0],label=dx+' density')
    ax[0].legend(); ax[0].set(xlabel='Baseline age (years)',ylabel='Density',xlim=(15,95))
    if groups: ax[1].boxplot(groups,labels=names); ax[1].set(ylabel='Baseline age (years)')
    pd.crosstab(binary.diagnosis,binary.sex).plot.bar(ax=ax[2],rot=0); ax[2].set(ylabel='Participants',xlabel='Primary diagnosis')
    export(fig,figs,'demographic_distributions')
    # Report other observed baseline covariates without silently inferring clinical meaning.
    rows=[]
    for col in ['demo__education','demo__race','demo__smoked','demo__health-history','demo__deep-brain-stimulation']:
        if col in p:
            for (dx,value),g in p.groupby(['diagnosis',col],dropna=False): rows.append({'field':col,'diagnosis':dx,'value':value,'participants':len(g)})
    write_table(pd.DataFrame(rows),tables/'other_covariates.csv')
    q=pd.read_parquet(out/f'processed/{scope}_audio_quality_metrics.parquet')
    m=r.merge(q,on='recording_id',how='inner',validate='one_to_one')
    flags=[c for c in q if c.startswith('flag_')]
    quality_rows=[]
    for dx,g in m.groupby('diagnosis'):
        for c in flags: quality_rows.append({'diagnosis':dx,'criterion':c,'flagged_recordings':int(g[c].fillna(False).sum()),'recording_denominator':len(g),
            'flagged_participants':int(g.loc[g[c].fillna(False).astype(bool),'participant_id'].nunique()),'participant_denominator':g.participant_id.nunique(),'scope':scope})
    write_table(pd.DataFrame(quality_rows),tables/'quality_by_diagnosis.csv')
    fig,ax=plt.subplots(1,3,figsize=(14,4))
    for dx,g in m.groupby('diagnosis'):
        ax[0].hist(g.duration_s.dropna(),bins=20,alpha=.5,label=dx)
        ax[1].scatter(g.age,g.duration_s,s=12,alpha=.5,label=dx)
    ax[0].set(xlabel='Duration (s)',ylabel=f'{scope} recordings'); ax[0].legend(fontsize=8)
    ax[1].set(xlabel='Baseline age (years)',ylabel='Duration (s)')
    for dx,g in binary.groupby('diagnosis'):
        v=g.recordings.sort_values(); ax[2].plot(v,np.arange(1,len(v)+1)/len(v),label=dx)
    ax[2].set(xscale='log',xlabel='Recordings per participant',ylabel='Empirical cumulative fraction'); ax[2].legend()
    export(fig,figs,'diagnosis_duration_frequency')
    # Equal participant weighting: median feature and median recording-quality covariates.
    f=pd.read_parquet(out/f'processed/{scope}_acoustic_features.parquet')
    feature_cols=[c for c in f if c.startswith(('praat_','egemaps__','spectral__')) and pd.api.types.is_numeric_dtype(f[c])]
    joined=m[['recording_id','participant_id','rms_amplitude','clipping_fraction','estimated_snr_db']].merge(f[['recording_id']+feature_cols],on='recording_id',validate='one_to_one')
    agg=joined.groupby('participant_id').median(numeric_only=True).reset_index()
    agg=agg.merge(p[['participant_id','diagnosis','age','sex','age_invalid']],on='participant_id',validate='one_to_one')
    core=[c for c in ('praat_stable__jitter_local','praat_stable__shimmer_local','praat_full__hnr_db','praat_full__f0_cv','spectral__flux_mean') if c in agg]
    if core:
        fig,axes=plt.subplots(1,len(core),figsize=(4*len(core),4))
        for ax,col in zip(np.atleast_1d(axes),core):
            arrays=[agg.loc[agg.diagnosis.eq(dx),col].dropna().to_numpy() for dx in ('control','pd')]
            usable=[(i+1,v) for i,v in enumerate(arrays) if len(v)>1 and np.std(v)>0]
            if usable: ax.violinplot([v for _,v in usable],positions=[i for i,_ in usable],showmedians=True)
            ax.set_xticks([1,2],['control','pd']); ax.set_ylabel(col.replace('__','\n'),fontsize=8)
        export(fig,figs,'acoustic_by_diagnosis')
    agg=agg[agg.diagnosis.isin(['pd','control']) & ~agg.age_invalid & agg.sex.isin(['Male','Female'])]
    agg['pd_indicator']=agg.diagnosis.eq('pd').astype(int); agg['male_indicator']=agg.sex.eq('Male').astype(int)
    rows=[]
    for col in feature_cols:
        for model,covariates in [('age_gender',['pd_indicator','age','male_indicator']),('age_gender_quality',['pd_indicator','age','male_indicator','rms_amplitude','clipping_fraction','estimated_snr_db'])]:
            data=agg[[col]+covariates].replace([np.inf,-np.inf],np.nan).dropna().astype(float)
            if len(data)<max(20,len(covariates)+5) or data.pd_indicator.nunique()<2 or data[col].std()==0: continue
            covariates=[c for c in covariates if data[c].nunique()>1]
            design=sm.add_constant(data[covariates],has_constant='add')
            if np.linalg.matrix_rank(design.to_numpy())<len(design.columns):continue
            fitted=sm.OLS(data[col],design).fit(cov_type='HC3')
            ci=fitted.conf_int().loc['pd_indicator']
            rows.append({'feature':col,'model':model,'participants':len(data),'pd_participants':int(data.pd_indicator.sum()),
                'pd_coefficient':float(fitted.params['pd_indicator']),'ci_low':float(ci.iloc[0]),'ci_high':float(ci.iloc[1]),'p':float(fitted.pvalues['pd_indicator'])})
    result=pd.DataFrame(rows)
    if len(result):
        result['fdr_bh']=np.nan
        for model,g in result.groupby('model'): result.loc[g.index,'fdr_bh']=multipletests(g.p,method='fdr_bh')[1]
    write_table(result,tables/'covariate_adjusted_pd_associations.csv')
    # Per-cohort balance denominators, including matching residuals and split imbalance.
    balances=[]
    for cohort in ('A','B','C'):
        cp=pd.read_parquet(out/f'processed/cohort_{cohort}/participant_manifest.parquet')
        for dx,g in cp.groupby('diagnosis'):
            balances.append({'cohort':cohort,'diagnosis':dx,'participants':len(g),'age_mean':g.age.mean(),'age_median':g.age.median(),
                 'age_sd':g.age.std(),'age_missing':int(g.age.isna().sum()),'male':int(g.sex.eq('Male').sum()),'female':int(g.sex.eq('Female').sum()),'gender_missing':int(g.sex.isna().sum())})
        sp=out/f'eda/tables/cohort_{cohort}_splits.csv'
        assignment=out/'processed/participant_splits.parquet'
        if sp.exists() and assignment.exists():
            assigned=cp.merge(pd.read_parquet(assignment)[['participant_id','split']],on='participant_id',validate='one_to_one')
            write_table(assigned.groupby(['split','diagnosis']).agg(participants=('participant_id','size'),age_mean=('age','mean'),age_median=('age','median'),male=('sex',lambda s:int(s.eq('Male').sum())),female=('sex',lambda s:int(s.eq('Female').sum()))).reset_index(),tables/f'cohort_{cohort}_split_demographics.csv')
    write_table(pd.DataFrame(balances),tables/'cohort_demographic_balance.csv')
    write_table(p.groupby('diagnosis').agg(participants=('participant_id','size'),recordings=('recordings','sum'),
        mean_recordings=('recordings','mean'),median_recordings=('recordings','median'),
        median_followup_days=('followup_days','median'),longitudinal=('recordings',lambda v:int(v.gt(1).sum()))).reset_index(),tables/'longitudinal_by_diagnosis.csv')
    print('Demographic EDA and participant-median adjusted models complete',flush=True)
