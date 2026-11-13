#!/usr/bin/env python3
"""Computed Phase 1 tables, figures, and evidence report; no inferred diagnoses."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from mpower_common import configure, decode, parser, save_json, summary, write_table

plt.rcParams.update({'font.size':11,'figure.dpi':120,'savefig.dpi':300,'axes.spines.top':False,'axes.spines.right':False})


def export(fig,out,name):
    for ext in ('png','svg'): fig.savefig(out/f'{name}.{ext}',bbox_inches='tight')
    plt.close(fig)


def dictionary(features):
    rows=[]
    for col in features:
        if '__' not in col or col.startswith('status__'): continue
        namespace=col.split('__')[0]
        unit='dimensionless'
        if 'hz' in col.lower(): unit='Hz'
        elif 'db' in col.lower() or 'hnr' in col: unit='dB'
        elif 'absolute_s' in col: unit='s'
        elif 'mfcc' in col: unit='cepstral coefficient'
        elif any(k in col for k in ('rms','flux')): unit='linear amplitude / FFT scale'
        if namespace=='egemaps': unit='see openSMILE eGeMAPSv02 descriptor definition; full standard name preserved'
        interval='longest contiguous Praat-pitched interval >=0.5 s' if namespace=='praat_stable' else 'full recording'
        rows.append({'feature':col,'namespace':namespace,'algorithm': 'Praat via Parselmouth' if namespace.startswith('praat') or namespace.startswith('pitch_') or namespace.startswith('trim_') else 'openSMILE eGeMAPSv02' if namespace=='egemaps' else 'librosa STFT 1024/hop160; MFCC13',
            'units':unit,'interval':interval,'sample_rate_hz':16000,
            'validity':'finite; undefined retained as missing; perturbation requires tracked periods',
            'clinical_reference_range':'not established for this cohort'})
    return pd.DataFrame(rows)


def duplicates(r,out):
    counts={}
    for col in ('file_sha256','native_pcm_sha256','normalized_pcm_sha256'):
        if col not in r: continue
        dup=r[r[col].notna() & r[col].duplicated(keep=False)]
        write_table(dup[['recording_id','participant_id',col]],out/f'eda/diagnostics/{col}_duplicates.csv')
        counts[col]={'groups':int(dup[col].nunique()),'recordings':len(dup),
                     'cross_participant_groups':int(dup.groupby(col).participant_id.nunique().gt(1).sum())}
    counts['near_duplicates']='Not yet validated: no claim of exhaustive near-duplicate detection; release blocked pending review.'
    save_json(out/'eda/diagnostics/duplicate_summary.json',counts)
    return counts


def associations(r,f,out):
    cols=[c for c in f if c.startswith(('praat_','egemaps__','spectral__')) and pd.api.types.is_numeric_dtype(f[c])]
    joined=r[['recording_id','participant_id','diagnosis']].merge(f[['recording_id']+cols],on='recording_id',validate='one_to_one')
    # One median per participant: every participant receives equal weight.
    agg=joined.groupby(['participant_id','diagnosis'])[cols].median().reset_index()
    corr=agg[cols].corr(method='spearman',min_periods=10)
    write_table(corr.rename_axis('feature').reset_index(),out/'eda/tables/participant_spearman_correlations.csv')
    rows=[]
    for col in cols:
        x=agg.loc[agg.diagnosis.eq('pd'),col].dropna(); y=agg.loc[agg.diagnosis.eq('control'),col].dropna()
        if len(x)>=3 and len(y)>=3:
            u,p=stats.mannwhitneyu(x,y,alternative='two-sided')
            rows.append({'feature':col,'pd_participants':len(x),'control_participants':len(y),
                         'rank_biserial':float(2*u/(len(x)*len(y))-1),'p':float(p)})
    table=pd.DataFrame(rows,columns=['feature','pd_participants','control_participants','rank_biserial','p'])
    if len(table):
        from statsmodels.stats.multitest import multipletests
        table['fdr_bh']=multipletests(table.p,method='fdr_bh')[1]
    write_table(table,out/'eda/tables/exploratory_pd_associations.csv')
    return 'Blocked: no verified PD/control labels' if not len(table) else 'Participant-median exploratory rank tests; unadjusted for demographics; not causal; no biomarker claim.'


def representative(r,cfg,figs):
    categories={'decoded':r.decode_status.eq('ok')}
    for c in [x for x in r if x.startswith('flag_')]: categories[c]=r[c].fillna(False).astype(bool)
    selected=[]
    for name,mask in categories.items():
        group=r.loc[mask & r.decode_status.eq('ok')].sort_values('recording_id')
        if not len(group): continue
        row=group.iloc[0]; anon=hashlib.sha256(row.recording_id.encode()).hexdigest()[:10]
        try:
            y,sr=decode(row.audio_path,cfg)
            import parselmouth
            pitch=parselmouth.Sound(y,sampling_frequency=sr).to_pitch_ac(pitch_floor=cfg['pitch']['floor'],pitch_ceiling=cfg['pitch']['ceiling'])
            fig,ax=plt.subplots(3,1,figsize=(9,7))
            ax[0].plot(np.arange(len(y))/sr,y,lw=.5); ax[0].set(ylabel='Amplitude',title=f'{name}: example {anon}')
            ax[1].specgram(y,Fs=sr,NFFT=512,noverlap=384,cmap='magma'); ax[1].set(ylabel='Frequency (Hz)',ylim=(0,4000))
            f=pitch.selected_array['frequency']; f[f==0]=np.nan
            ax[2].plot(pitch.xs(),f,lw=1); ax[2].set(xlabel='Time (s)',ylabel='F0 (Hz)')
            export(fig,figs,'example_'+name)
            selected.append({'category':name,'anonymized_example':anon,'recording_id':row.recording_id,'rule':'lowest stable recording hash among category members'})
        except Exception as exc:
            selected.append({'category':name,'error':str(exc)})
    write_table(pd.DataFrame(selected),cfg['output_root']/'eda/diagnostics/representative_examples.csv')


def main():
    p=parser(__doc__); p.add_argument('--scope',choices=['pilot','full'],default='pilot'); args=p.parse_args(); cfg=configure(args.config)
    out=cfg['output_root']; figs=out/'eda/figures'; tables=out/'eda/tables'; scope=args.scope
    if scope == 'pilot':
        subprocess.run([sys.executable, str(Path(__file__).with_name('validate_pilot.py')),
                        '--config', args.config], check=True)
    from demographic_eda import run as demographic_eda
    demographic_eda(cfg,scope)
    r=pd.read_parquet(out/'processed/recording_manifest.parquet'); people=pd.read_parquet(out/'processed/participant_manifest.parquet')
    q=pd.read_parquet(out/f'processed/{scope}_audio_quality_metrics.parquet')
    f=pd.read_parquet(out/f'processed/{scope}_acoustic_features.parquet')
    r=r.merge(q,on='recording_id',how='inner',validate='one_to_one')
    d=json.loads((tables/'discovery_summary.json').read_text())
    counts=duplicates(r,out)
    statrows=[]
    for col in q:
        if pd.api.types.is_numeric_dtype(q[col]) and not pd.api.types.is_bool_dtype(q[col]): statrows.append({'metric':col,**summary(q[col])})
    write_table(pd.DataFrame(statrows),tables/f'{scope}_audio_summary.csv')
    feature_summary=[]
    for col in f:
        if '__' in col and not col.startswith('status__'):
            vals=pd.to_numeric(f[col],errors='coerce')
            feature_summary.append({'feature':col,**summary(vals),'valid_fraction':float(np.isfinite(vals).mean())})
    write_table(pd.DataFrame(feature_summary),tables/f'{scope}_feature_summary.csv')
    write_table(dictionary(f),out/'processed/feature_dictionary.csv')
    write_table(pd.DataFrame([{'threshold_s':k,'below_threshold':int((r.duration_s<k).sum()),'decoded_denominator':int(r.decode_status.eq('ok').sum()),'scope':scope} for k in (2,3,5,7,9)]),tables/'duration_threshold_counts.csv')
    assoc=associations(r,f,out)
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    people.diagnosis.value_counts().plot.bar(ax=ax[0],rot=0,color='#41799b'); ax[0].set(title='Participant diagnosis availability',ylabel='Participants')
    pd.Series(d['recording_diagnoses']).plot.bar(ax=ax[1],rot=0,color='#41799b'); ax[1].set(title='Recording diagnosis availability',ylabel='Recordings')
    export(fig,figs,'diagnosis_availability')
    fig,ax=plt.subplots(1,2,figsize=(10,4)); v=people.recordings.to_numpy()
    ax[0].hist(v,bins=np.geomspace(1,max(2,v.max()+1),40),color='#41799b'); ax[0].set(xscale='log',xlabel='Recordings per participant',ylabel='Participants')
    ax[1].plot(np.sort(v),np.arange(1,len(v)+1)/len(v)); ax[1].set(xscale='log',xlabel='Recordings per participant',ylabel='Empirical cumulative fraction')
    export(fig,figs,'longitudinal_recordings')
    fig,ax=plt.subplots(1,3,figsize=(13,4)); valid=r[r.decode_status.eq('ok')]
    ax[0].hist(valid.duration_s,bins=30,color='#41799b'); ax[0].set(xlabel='Decoded duration (s)',ylabel=f'{scope.capitalize()} recordings')
    for fmt,g in valid.groupby('original_format'): ax[1].hist(g.duration_s,bins=20,alpha=.5,label=fmt)
    if len(valid): ax[1].legend(fontsize=7)
    ax[1].set(xlabel='Duration (s)',ylabel='Recordings')
    ax[2].scatter(valid.duration_s,valid.energy_activity_s,s=10,alpha=.6); ax[2].set(xlabel='Total duration (s)',ylabel='Energy-active duration (s)')
    export(fig,figs,'audio_duration')
    metrics=['rms_amplitude','clipping_fraction','estimated_snr_db','silence_fraction']
    fig,ax=plt.subplots(2,2,figsize=(10,7))
    for a,col in zip(ax.flat,metrics):
        a.hist(pd.to_numeric(r[col],errors='coerce').dropna(),bins=25,color='#41799b'); a.set(xlabel=col.replace('_',' '),ylabel=f'{scope} recordings')
    export(fig,figs,'qc_distributions')
    core=[c for c in ['praat_stable__jitter_local','praat_stable__shimmer_local','praat_full__hnr_db','praat_full__f0_cv','spectral__flux_mean'] if c in f]
    if core:
        fig,axes=plt.subplots(1,len(core),figsize=(4*len(core),3.5)); axes=np.atleast_1d(axes)
        for a,col in zip(axes,core):
            a.hist(pd.to_numeric(f[col],errors='coerce').replace([np.inf,-np.inf],np.nan).dropna(),bins=25,color='#41799b'); a.set(xlabel=col.replace('__','\n'),ylabel=f'{scope} recordings')
        export(fig,figs,'acoustic_distributions')
        corr=f[core].corr(method='spearman')
        fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(corr,vmin=-1,vmax=1,cmap='coolwarm'); ax.set_xticks(range(len(core)),core,rotation=90,fontsize=7); ax.set_yticks(range(len(core)),core,fontsize=7); fig.colorbar(im,ax=ax,label='Spearman correlation (recording-level descriptive only)'); export(fig,figs,'selected_feature_correlations')
    representative(r,cfg,figs)
    # Overlapping definitions are compared descriptively, not treated as interchangeable.
    agreement=[]
    for a,b in [('praat_full__f0_mean_hz','egemaps__F0semitoneFrom27.5Hz_sma3nz_amean'),('praat_full__jitter_local','egemaps__jitterLocal_sma3nz_amean'),('praat_full__shimmer_db','egemaps__shimmerLocaldB_sma3nz_amean')]:
        if a in f and b in f:
            pair=f[[a,b]].dropna(); agreement.append({'praat':a,'opensmile':b,'paired':len(pair),'spearman':pair[a].corr(pair[b],method='spearman') if len(pair)>2 else np.nan,'note':'Definitions/units/intervals differ; correlation is not agreement in absolute units'})
    write_table(pd.DataFrame(agreement),tables/'praat_opensmile_comparison.csv')
    decoded=int(r.decode_status.eq('ok').sum()); failed=len(r)-decoded
    readiness=json.loads((tables/'cohort_readiness.json').read_text())
    split_status=json.loads((out/'eda/diagnostics/split_status.json').read_text())
    validation=json.loads((out/'eda/diagnostics/pilot_validation.json').read_text()) if scope=='pilot' else {'scope':'full; pilot validation saved separately'}
    report=f'''# mPower Phase 1 evidence report

## Executed scope
Metadata census: {d['recordings']:,} source recordings, {d['participants']:,} unique participants.
Audio: {scope}, {len(r)} attempted; {decoded} decoded; {failed} failures. No full-audio results are implied by a pilot.
Data-derived tables, identifiers, example mappings, and executed notebook stay in approved output storage.

## Diagnosis and demographics
Recording diagnosis counts: {json.dumps(d['recording_diagnoses'])}.
Participant diagnosis counts: {json.dumps(d['participant_diagnoses'])}.
The observed voice schema has no verified diagnosis, age, or sex. Medication timing is not a diagnosis.
PD/control prevalence, age/sex imbalance, covariate adjustment, medication-label contradictions, and age-balanced matching cannot be established.
Clinical questionnaires/severity measures are unavailable; no clinical time matching was performed.

## Repeated recordings
{d['longitudinal_participants']:,}/{d['participants']:,} participants have more than one recording;
{d['single_recording_participants']:,} have one. Recordings per participant: {json.dumps(d['recordings_per_participant'])}.
The most prolific 1% contribute {d['top_1_percent_recording_share']:.2%} of recordings.
See follow-up and timestamp summaries; invalid dates stay missing.

## Duration and QC ({scope})
Duration summary: {json.dumps(summary(r.duration_s))}.
Independent flag counts among {decoded} decoded recordings: {json.dumps({c:int(r.loc[r.decode_status.eq('ok'),c].fillna(False).sum()) for c in r if c.startswith('flag_')})}.
Thresholds are exploratory; noise is estimated from the 10th frame-RMS percentile, and SNR proxy is the 90th/10th percentile amplitude ratio.
This proxy is not a noise-reference SNR and can mischaracterize steady phonation. Energy activity is not validated voice activity.
Low intensity, unstable pitch, clipping and SNR flags are retained for review, not automatic primary exclusions.
See sequential QC flows and threshold sensitivity; disease-specific QC bias cannot be tested without labels.

## Acoustic validity and associations
Feature-specific valid denominators are in {scope}_feature_summary.csv; undefined perturbation measures remain missing.
Repeat-processing validation: {json.dumps(validation)}.
Full-recording and longest-contiguous-pitched-interval Praat features use distinct namespaces.
{assoc}. No causal conclusions or verified PD biomarkers are reported.
CPP, formants, openSMILE LLD exports and speech-contamination classifiers are not yet validated/configured.
Pilot pitch-range and 250-ms edge-trim sensitivity columns are retained. The synthetic repeat-processing test verifies deterministic extraction.

## Leakage and readiness
Source duplicate recording-ID rows: {d['duplicate_recording_ids']}.
Processed-audio duplicate audit: {json.dumps(counts)}.
Hash results cover only the processed scope. Different-codec lossy copies need a near-duplicate audit; native PCM hash equality does not find them all.
No labeled training dataset is released. Empty cohort/split manifests reflect missing eligibility information, not absence of available audio.
Cohort participant/recording counts: {json.dumps(readiness['cohorts'])}.
Partition status: {json.dumps(split_status)}. With no eligible labels, each train/validation/test partition contains zero participants and zero recordings.
All future participant recordings must share a split. Detected duplicate-connected participants share an assignment; source identity conflicts are quarantined.
No imputation, normalization, selection, PCA, embeddings or model fitting was performed.

## Recommendation and remaining work
Obtain authorized participant diagnosis/demographic metadata and audit join cardinality and temporal validity before binary prediction.
Validate QC against diagnosis after linkage. Run the restartable full-audio Slurm pipeline after pilot review and complete near-duplicate review.
Use cohort A as the future broad baseline and compare exploratory cohort B plus a genuinely age/sex-balanced cohort C once defensible definitions and labels are available.
Current cohorts are not ready for foundation-model fine-tuning. No additional datasets were downloaded.

## Manuscript Methods (executed scope)
We characterized the locally available mPower Voice Activity export, maintaining source table and row provenance.
The consolidated source was not concatenated with download shards. Participant records were grouped by the observed healthCode identifier;
recordId duplicates and missing identities were explicitly flagged. Audio was linked using the exported audio_path and file-handle fields,
without interpreting filenames as participant identifiers. Diagnosis was retained as unknown in the absence of an explicit verified field.
A fixed-seed coverage pilot prioritized distinct participants across medication and available diagnosis strata.
Original files remained unchanged. FFmpeg decoded one recording at a time to mono 16-kHz float PCM for analysis;
files beyond the configured resource limit were flagged rather than silently truncated. Container properties and original-rate/channel PCM hashes
were collected separately. QC summarized duration, frame energy, silence proxies, clipping, DC offset, spectral flatness and an explicitly labeled SNR proxy.
Praat through Parselmouth estimated pitch (60–500 Hz), jitter, shimmer and HNR, with maximum period/amplitude factors of 1.3/1.6.
Perturbation measurements were retained separately for complete recordings and the longest contiguous pitched interval of at least 0.5 s;
undefined estimates remained missing. openSMILE eGeMAPSv02 supplied 88 named functionals; librosa supplied spectral and MFCC summaries.
Exploratory comparisons, when eligible, use participant-level median features and Benjamini–Hochberg correction.
A provisional 70/15/15 participant allocation is implemented with fixed seed and duplicate-connected participant grouping,
but no valid supervised partition was created without confirmed binary labels. These descriptive analyses cannot establish causal biomarkers.
'''
    if d.get('demographics_table'):
        import re
        imbalance=json.loads((tables/'demographic_imbalance.json').read_text())
        meta=f'''## Diagnosis and demographics
The authorized syn5511429 demographics export was linked by observed healthCode with validated one-to-one participant / many-to-one recording cardinality.
Matched participants: {d['demographics_matched_participants']}/{d['participants']}; matched recordings: {d['demographics_matched_recordings']}/{d['recordings']}.
Unmatched participants remain unknown; no source rows were dropped or multiplied.
Participant source diagnosis: {json.dumps(d['source_participant_diagnoses'])}; recording source diagnosis: {json.dumps(d['source_recording_diagnoses'])}.
Primary participant labels: {json.dumps(d['participant_diagnoses'])}; recording labels: {json.dumps(d['recording_diagnoses'])}.
Diagnosis represents participant-reported professional diagnosis, not independent clinical adjudication.
Controls reporting a plausible diagnosis/PD-medication year or explicit pre/post-PD-medication voice timing are conservatively classified as conflicting; raw source labels remain for sensitivity analyses.
Label-conflict participants: {d['label_conflict_participants']}. Medication nonuse among PD participants is flagged, not automatically a contradiction.
Baseline demographic imbalance: {json.dumps(imbalance)}.
Age and reported gender are baseline survey covariates; signed survey-to-recording intervals are saved. No UPDRS/PDQ-8 severity data were retrieved or temporally linked.

## Repeated recordings'''
        report=re.sub(r'## Diagnosis and demographics.*?## Repeated recordings',lambda _:meta,report,flags=re.S)
        report=report.replace('No labeled training dataset is released. Empty cohort/split manifests reflect missing eligibility information, not absence of available audio.',
            'Labeled processed-scope cohorts and provisional splits are saved. This pilot does not establish full-cohort readiness; near-duplicate review and full processing remain pending.')
        report=report.replace('With no eligible labels, each train/validation/test partition contains zero participants and zero recordings.',
            'Exact per-cohort participant and recording split denominators are in cohort_A/B/C_splits.csv. These assignments describe processed audio only.')
        report=report.replace('disease-specific QC bias cannot be tested without labels.',
            'quality_by_diagnosis.csv reports diagnosis-specific flag rates with recording and participant denominators; pilot estimates are imprecise.')
        report=report.replace('Obtain authorized participant diagnosis/demographic metadata and audit join cardinality and temporal validity before binary prediction.',
            'Demographics have been retrieved and linkage audited. Review conservative label contradictions and selection bias before binary prediction.')
        report=report.replace('Current cohorts are not ready for foundation-model fine-tuning. No additional datasets were downloaded.',
            'Current processed-scope cohorts are provisional. Only the explicitly authorized demographics table was additionally downloaded; full-cohort readiness is pending.')
        report=report.replace('Diagnosis was retained as unknown in the absence of an explicit verified field.',
            'The baseline demographics survey was joined by healthCode with validated cardinality. Participant-reported professional diagnosis was mapped explicitly; unmatched, missing and contradictory responses remained outside primary binary cohorts.')
        report=report.replace('but no valid supervised partition was created without confirmed binary labels.',
            'with processed-scope supervised partitions remaining provisional pending full-audio and near-duplicate audits.')
        adjusted=tables/'covariate_adjusted_pd_associations.csv'
        if adjusted.exists() and adjusted.stat().st_size>2:
            adj=pd.read_csv(adjusted)
            report+='\n## Covariate-adjusted exploratory associations\nParticipant medians receive equal weight. OLS models use HC3 errors, baseline age/reported gender, and additionally median quality covariates. BH correction is applied separately by model.\n'
            report+=f'{len(adj)} feature/model comparisons; {int(adj.fdr_bh.lt(.05).sum()) if len(adj) else 0} pass FDR <0.05. These are pilot exploratory associations, with confounding and selection limitations; no causal or diagnostic biomarker claim. See covariate_adjusted_pd_associations.csv.\n'
    (out/'eda/MPOWER_FINAL_REPORT.md').write_text(report)
    print(f'Report complete: {scope}, {len(r)} attempted / {decoded} decoded',flush=True)


if __name__=='__main__': main()
