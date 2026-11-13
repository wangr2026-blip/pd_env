#!/usr/bin/env python3
"""Read-only snapshot audit for a quality/medication weighting analysis plan.

Writes aggregates only to separate approved storage; never changes EDA artifacts
or fits weights using held-out participants. No audio decoding or model inference.
"""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mpower_common import save_json


def summarize(values):
    x = pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    return {'n': len(x), **{f'q{q}': float(x.quantile(q / 100)) if len(x) else None for q in (0, 10, 25, 50, 75, 90, 99, 100)}}


def rows(frame, columns):
    return frame[columns].to_dict('records')


def audit(data, eda):
    inputs = {}
    def read(relative, columns=None):
        p = eda / relative
        before = p.stat()
        d = pd.read_parquet(p, columns=columns)
        after = p.stat()
        inputs[relative] = {'rows': len(d), 'bytes': before.st_size, 'mtime_ns': before.st_mtime_ns,
                            'changed_during_read': before.st_mtime_ns != after.st_mtime_ns or before.st_size != after.st_size}
        return d
    r = read('processed/recording_manifest.parquet')
    p = read('processed/participant_manifest.parquet')
    output = {'snapshot_utc': datetime.now(timezone.utc).isoformat(), 'inputs': inputs,
              'recordings': len(r), 'participants': len(p),
              'diagnosis_recordings': r.diagnosis.value_counts(dropna=False).to_dict(),
              'diagnosis_participants': p.diagnosis.value_counts(dropna=False).to_dict(),
              'source_diagnosis_participants': p.source_diagnosis.value_counts(dropna=False).to_dict()}
    output['medication_by_diagnosis'] = r.groupby(['diagnosis', 'medication'], dropna=False).agg(
        recordings=('recording_id', 'size'), participants=('participant_id', 'nunique')).reset_index().fillna('missing').to_dict('records')
    output['participant_demographics'] = []
    for label, group in p.groupby('diagnosis'):
        output['participant_demographics'].append({'diagnosis': label, 'n': len(group),
            'age': summarize(group.age), 'reported_gender': group.sex.fillna('missing').value_counts().to_dict(),
            'followup_days': summarize(group.followup_days), 'recordings': summarize(group.recordings)})
    output['covariate_coverage'] = {c: {'available': int(p[c].notna().sum()), 'total': len(p),
        'unique_nonmissing': int(p[c].nunique())} for c in p if c.startswith('demo__')}
    count = r.groupby('participant_id').size().sort_values(ascending=False)
    output['repeat_recordings'] = {'distribution': summarize(count), 'top_1_percent_share': float(count.head(int(np.ceil(len(count) * .01))).sum() / len(r)),
        'participants_with_multiple_recordings': int(count.gt(1).sum())}
    pd_rows = r[r.diagnosis.eq('pd')].copy()
    before = pd_rows.medication.fillna('').str.startswith('Immediately before')
    after = pd_rows.medication.fillna('').str.startswith('Just after')
    pre = set(pd_rows.loc[before, 'participant_id'])
    post = set(pd_rows.loc[after, 'participant_id'])
    pd_rows['utc_date'] = pd.to_datetime(pd_rows.recorded_at, errors='coerce', utc=True).dt.date
    dates_before = set(pd_rows.loc[before & pd_rows.utc_date.notna(), ['participant_id', 'utc_date']].itertuples(index=False, name=None))
    dates_after = set(pd_rows.loc[after & pd_rows.utc_date.notna(), ['participant_id', 'utc_date']].itertuples(index=False, name=None))
    output['medication_pair_readiness'] = {'pd_participants_with_pre': len(pre), 'pd_participants_with_post': len(post),
        'pd_participants_with_both': len(pre & post), 'same_utc_date_participant_days_with_both': len(dates_before & dates_after),
        'same_utc_date_participants_with_both': len({a for a, _ in dates_before & dates_after}),
        'caution': 'UTC dates are descriptive; not verified local dates or washout/peak-dose states'}
    if 'survey_to_recording_days' in r:
        output['survey_recording_gap_days'] = summarize(r.survey_to_recording_days)
    output['temporal_window_counts'] = []
    if 'survey_to_recording_days' in r:
        gap = pd.to_numeric(r.survey_to_recording_days, errors='coerce')
        for days in (1, 7, 30):
            selected = r[gap.between(0, days) & r.diagnosis.isin(['pd', 'control'])]
            output['temporal_window_counts'].append({'window': f'0_to_{days}_days_after_survey',
                'counts': selected.groupby('diagnosis').agg(recordings=('recording_id', 'size'),
                    participants=('participant_id', 'nunique')).reset_index().to_dict('records')})
    output['age_overlap_counts_50_80'] = p[p.age.between(50, 80)].groupby('diagnosis').size().to_dict()
    output['energy_proxy_synthetic_check'] = {}
    for kind in ('clean_sine', 'stationary_noise'):
        sr = 16000
        wave = .1 * np.sin(2 * np.pi * 150 * np.arange(sr * 3) / sr) if kind == 'clean_sine' else np.random.default_rng(1).normal(0, .1, sr * 3)
        frames = np.lib.stride_tricks.sliding_window_view(wave, 480)[::160]
        rms = np.sqrt(np.mean(frames ** 2, axis=1))
        value = float(20 * np.log10(np.quantile(rms, .9) / np.quantile(rms, .1)))
        output['energy_proxy_synthetic_check'][kind] = {'frame_dynamic_range_db': value, 'flagged_below_10': value < 10}
    output['quality_snapshots'] = {}
    for relative in ('processed/pilot_features_checkpoint.parquet', 'processed/full_features_checkpoint.parquet'):
        path = eda / relative
        if not path.exists():
            continue
        schema = pq.ParquetFile(path).schema.names
        desired = ['recording_id', 'decode_status', 'feature_status', 'duration_s', 'rms_amplitude', 'estimated_snr_db',
                   'frame_dynamic_range_db', 'clipping_fraction', 'pitch_detected_s', 'praat_full__pitch_valid_fraction',
                   'praat_full__f0_mean_hz', 'praat_full__jitter_local', 'praat_full__shimmer_local', 'praat_full__hnr_db']
        columns = [c for c in schema if c in desired or c.startswith('flag_') or c.startswith('status__')]
        q = read(relative, columns)
        q = q.merge(r[['recording_id', 'participant_id', 'diagnosis', 'medication']], on='recording_id', validate='one_to_one')
        item = {'recordings': len(q), 'participants': q.participant_id.nunique(),
                'decode_status': q.decode_status.fillna('missing').value_counts().to_dict(),
                'feature_status': q.feature_status.fillna('missing').value_counts().to_dict(),
                'by_diagnosis': [], 'numeric_summaries': {}}
        for label, group in q.groupby('diagnosis'):
            item['by_diagnosis'].append({'diagnosis': label, 'recordings': len(group), 'participants': group.participant_id.nunique(),
                'decoded': int(group.decode_status.eq('ok').sum()),
                'flags': {c: {'flagged': int(group[c].eq(True).sum()), 'evaluated': int(group[c].notna().sum())} for c in group if c.startswith('flag_')},
                'feature_nonmissing': {c: int(pd.to_numeric(group[c], errors='coerce').replace([np.inf, -np.inf], np.nan).notna().sum())
                    for c in desired if c in group and c.startswith('praat_')}})
        decoded = q[q.decode_status.eq('ok')]
        item['numeric_summaries'] = {c: summarize(decoded[c]) for c in desired if c in decoded and c not in ['recording_id', 'decode_status', 'feature_status']}
        output['quality_snapshots'][relative] = item
    # Device fields in the original voice table are distinct from baseline survey phoneInfo.
    voice_path = data / 'voice_activity/Voice_Activity.csv'
    header = pd.read_csv(voice_path, nrows=0).columns.tolist()
    device_columns = [c for c in header if any(token in c.lower() for token in ('phone', 'device', 'appversion', 'timezone', 'time_zone'))]
    output['voice_device_columns'] = device_columns
    if device_columns:
        devices = pd.read_csv(voice_path, usecols=device_columns, dtype='string')
        output['voice_device_coverage'] = {c: {'available': int(devices[c].notna().sum()), 'unique_nonmissing': int(devices[c].nunique())} for c in devices}
    output['interpretation'] = 'Descriptive snapshot only. In-progress full extraction is not a representative completed QC sample; no weighting coefficients fitted.'
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', help='New aggregate JSON under approved storage outside raw/EDA/repository directories')
    args = parser.parse_args()
    os.umask(0o077)
    data = Path(os.environ['MPOWER_DATA_ROOT']).resolve()
    eda = Path(os.environ['MPOWER_OUTPUT_ROOT']).resolve()
    root = Path(os.environ['MPOWER_BENCHMARK_ROOT']).resolve()
    repo = Path(__file__).resolve().parents[1]
    path = Path(args.output).resolve() if args.output else root / 'planning' / f"weighting_audit_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    for restricted in (data, eda, repo):
        if path == restricted or restricted in path.parents:
            raise ValueError('Audit output must stay outside raw data, EDA outputs and Git')
    result = audit(data, eda)
    save_json(path, result)
    print(path)
    print({k: result[k] for k in ('recordings', 'participants', 'diagnosis_recordings', 'diagnosis_participants', 'repeat_recordings', 'medication_pair_readiness')})


if __name__ == '__main__':
    main()
