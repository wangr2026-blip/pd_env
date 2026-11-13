#!/usr/bin/env python3
"""Inventory and audit observed tables without guessing diagnosis or linkage."""
import collections
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

from mpower_common import configure, load_table, parser, save_json, summary, write_table

TABLES = {'.csv', '.tsv', '.parquet', '.json', '.jsonl'}
AUDIO = {'.m4a', '.wav', '.mp3', '.aac', '.flac', '.ogg', '.aiff', '.caf', '.mp4', '.wma'}


def inventory(root):
    rows = []
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in {'.git', '__pycache__', '.pytest_cache', '.mpower_venv', '.venv', 'node_modules'})
        for name in sorted(files):
            p = Path(base) / name
            try:
                rows.append({'relative_path': str(p.relative_to(root)), 'extension': p.suffix.lower(),
                             'bytes': p.stat().st_size, 'kind': 'table' if p.suffix.lower() in TABLES else
                             'audio' if p.suffix.lower() in AUDIO else 'other'})
            except OSError as exc:
                rows.append({'relative_path': str(p.relative_to(root)), 'error': str(exc)})
            if len(rows) % 10000 == 0:
                print(f'Inventory: {len(rows):,} files inspected', flush=True)
    return pd.DataFrame(rows, columns=['relative_path', 'extension', 'bytes', 'kind', 'error'])


def diagnosis(value, cfg):
    if pd.isna(value):
        return 'unknown'
    text = str(value).strip().casefold()
    pd_values = {str(v).casefold() for v in cfg['diagnosis_values']['pd']}
    hc_values = {str(v).casefold() for v in cfg['diagnosis_values']['control']}
    if pd_values & hc_values:
        raise ValueError('Diagnosis mappings overlap')
    return 'pd' if text in pd_values else 'control' if text in hc_values else 'unknown'


def canonicalize(frame, cfg):
    c = cfg['columns']
    required = [c[k] for k in ('participant', 'recording', 'audio', 'handle')]
    missing = set(required) - set(frame)
    if missing:
        raise ValueError(f'Configured columns absent: {missing}')
    f = frame.copy()
    for col in f:
        f[col] = f[col].replace(r'^\s*$', pd.NA, regex=True)
    out = pd.DataFrame({'source_row': np.arange(len(f)) + 2, 'participant_id': f[c['participant']],
                        'source_recording_id': f[c['recording']], 'audio_path': f[c['audio']],
                        'audio_handle': f[c['handle']]})
    # Every source row survives. Duplicate recording IDs are quarantined, never silently merged.
    out['recording_id'] = [hashlib.sha256(f'{cfg["metadata"]}:{i}'.encode()).hexdigest()[:24] for i in out.source_row]
    out['source_table'] = cfg['metadata']
    # Preserve observed original fields without changing their semantics.
    for name in f:
        out['source__' + name] = f[name]
    out['task'] = 'sustained_phonation'  # mapping explicitly verified for audio_audio.m4a only
    if c['handle'] != 'audio_audio.m4a':
        out['task'] = 'unknown'
    out['duplicate_recording_id'] = f[c['recording']].notna() & f[c['recording']].duplicated(keep=False)
    out['duplicate_metadata_row'] = f.duplicated(keep=False)
    out['missing_identity'] = out.participant_id.isna() | out.source_recording_id.isna()
    root = cfg['data_root']
    def resolve(value):
        if pd.isna(value):
            return ''
        p = Path(value).expanduser()
        return str(p if p.is_absolute() else root / p)
    out['audio_path'] = out.audio_path.map(resolve)
    out['audio_exists'] = out.audio_path.map(lambda p: bool(p) and Path(p).is_file())
    diag = c.get('diagnosis')
    out['diagnosis_response'] = f[diag].map(lambda v: diagnosis(v, cfg)) if diag else 'unknown'
    out['medication'] = f[c['medication']] if c.get('medication') in f else pd.NA
    for key in ('age', 'sex'):
        col = c.get(key)
        out[key] = f[col] if col else pd.NA
    out['age'] = pd.to_numeric(out.age, errors='coerce')
    out['age_invalid'] = out.age.notna() & ~out.age.between(18, 120)
    tcol = c.get('timestamp')
    t = f[tcol] if tcol in f else pd.Series(pd.NA, index=f.index, dtype='string')
    numeric = pd.to_numeric(t, errors='coerce')
    # Epoch-ms is supported only when values have a plausible epoch-ms magnitude.
    epoch = numeric.between(1e11, 1e14)
    parsed = pd.to_datetime(t.where(~epoch), errors='coerce', utc=True, format='mixed')
    parsed.loc[epoch] = pd.to_datetime(numeric.loc[epoch], unit='ms', utc=True, errors='coerce')
    out['recorded_at'] = parsed
    people = []
    for pid, g in out.dropna(subset=['participant_id']).groupby('participant_id', sort=True):
        labels = set(g.diagnosis_response) & {'pd', 'control'}
        label = 'conflicting' if len(labels) > 1 else next(iter(labels)) if labels else 'unknown'
        ages = g.age.dropna()
        sexes = g.sex.dropna().unique()
        dates = g.recorded_at.dropna().sort_values()
        people.append({'participant_id': pid, 'diagnosis': label, 'recordings': len(g),
            'diagnosis_response_unknown': int(g.diagnosis_response.eq('unknown').sum()),
            'age': float(ages.median()) if len(ages) else np.nan, 'age_min': ages.min(), 'age_max': ages.max(),
            'age_varies': ages.nunique() > 1, 'age_invalid': bool(g.age_invalid.any()),
            'sex': sexes[0] if len(sexes) == 1 else pd.NA, 'sex_conflict': len(sexes) > 1,
            'first_recording': dates.min(), 'last_recording': dates.max(),
            'followup_days': (dates.max()-dates.min()).total_seconds()/86400 if len(dates) else np.nan,
            'median_interval_days': dates.diff().dt.total_seconds().median()/86400 if len(dates)>1 else np.nan})
    participants = pd.DataFrame(people)
    out = out.merge(participants[['participant_id', 'diagnosis']], how='left', on='participant_id', validate='many_to_one')
    out['diagnosis'] = out.diagnosis.fillna('unknown')
    return out, participants


def run(cfg):
    root, dest = cfg['data_root'], cfg['output_root']
    source = load_table(root/cfg['metadata'])
    inv = inventory(root)
    mapped = set()
    for value in source[cfg['columns']['audio']].dropna():
        p = Path(value)
        p = p if p.is_absolute() else root / p
        try:
            mapped.add(str(p.resolve().relative_to(root.resolve())))
        except ValueError:
            pass
    # Actual downloaded files may be .tmp or extensionless; use verified manifest links.
    inv.loc[inv.relative_path.isin(mapped), 'kind'] = 'audio'
    write_table(inv, dest/'eda/diagnostics/file_inventory.csv')
    repo_inv = inventory(Path(__file__).resolve().parents[1])
    write_table(repo_inv, dest/'eda/diagnostics/repository_inventory.csv')
    schemas = []
    for rel in inv.loc[inv.kind.eq('table'), 'relative_path']:
        try:
            f = source if rel == cfg['metadata'] else load_table(root/rel)
            candidates = [c for c in f if any(x in c.lower() for x in ('id', 'code', 'handle', 'time', 'date'))]
            inferred = f.infer_objects()
            info = {'table': rel, 'rows': len(f), 'columns': len(f.columns), 'duplicate_rows': int(f.duplicated().sum()),
                    'fields': [{'name': c, 'storage_dtype': str(f[c].dtype),
                       'numeric_parse_fraction': float(pd.to_numeric(f[c],errors='coerce').notna().mean()) if len(f) else 0,
                       'missing': int(f[c].isna().sum()), 'distinct': int(f[c].nunique())} for c in f],
                    'candidate_keys': {c: {'missing': int(f[c].isna().sum()), 'duplicated_nonmissing': int(f[c].dropna().duplicated().sum())} for c in candidates},
                    'categorical_values': {c: f[c].value_counts(dropna=False).head(100).rename_axis('value').reset_index(name='count').astype(str).to_dict('records') for c in f if f[c].nunique() <= 100},
                    'head': f.head(5).fillna('').to_dict('records')}
            if rel.startswith('voice_activity/metadata_parts/') and cfg['columns']['recording'] in f:
                key = cfg['columns']['recording']
                info['relation'] = 'download shard; not appended to consolidated table'
                info['unmatched_record_ids'] = int((~f[key].isin(source[key])).sum())
                shared = sorted(set(f.columns) & set(source.columns))
                left_hash = pd.util.hash_pandas_object(f[shared].fillna(''), index=False)
                right_hash = pd.util.hash_pandas_object(source[shared].fillna(''), index=False)
                info['rows_not_identical_to_consolidated'] = int((~left_hash.isin(set(right_hash))).sum())
            schemas.append(info)
        except Exception as exc:
            schemas.append({'table': rel, 'error': str(exc)})
        if len(schemas) % 25 == 0:
            print(f'Audited {len(schemas)} metadata tables', flush=True)
    save_json(dest/'eda/diagnostics/table_schemas.json', schemas)
    save_json(dest/'eda/diagnostics/audit_provenance.json', {
        'source_sha256': hashlib.sha256((root/cfg['metadata']).read_bytes()).hexdigest(),
        'config_digest': cfg['config_digest'],
        'source_scripts': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in Path(__file__).parent.glob('*.py')},
        'configuration': {k:v for k,v in cfg.items() if k not in ('data_root','output_root')},
    })
    recordings, participants = canonicalize(source, cfg)
    write_table(recordings, dest/'processed/recording_manifest.parquet')
    write_table(participants, dest/'processed/participant_manifest.parquet')
    write_table(recordings[recordings.duplicate_metadata_row | recordings.duplicate_recording_id | recordings.missing_identity], dest/'eda/diagnostics/identity_conflicts.csv')
    write_table(participants.groupby('diagnosis', dropna=False).agg(participants=('participant_id','size'), recordings=('recordings','sum')).reset_index(), dest/'eda/tables/cohort_characteristics.csv')
    write_table(recordings.groupby(['diagnosis','medication'], dropna=False).size().reset_index(name='recordings'), dest/'eda/tables/medication_by_diagnosis.csv')
    write_table(participants.groupby('diagnosis').agg(age_available=('age','count'), sex_available=('sex','count'), longitudinal=('recordings', lambda v: int(v.gt(1).sum()))).reset_index(), dest/'eda/tables/demographic_coverage.csv')
    save_json(dest/'eda/tables/discovery_summary.json', {
        'files': len(inv), 'total_bytes': int(inv.bytes.fillna(0).sum()), 'file_types': inv.extension.value_counts().to_dict(),
        'metadata_tables': len(schemas), 'recordings': len(recordings), 'participants': len(participants),
        'missing_audio': int((~recordings.audio_exists).sum()),
        'duplicate_recording_ids': int(recordings.duplicate_recording_id.sum()),
        'recording_diagnoses': recordings.diagnosis.value_counts().to_dict(),
        'participant_diagnoses': participants.diagnosis.value_counts().to_dict(),
        'recordings_per_participant': summary(participants.recordings),
        'longitudinal_participants': int(participants.recordings.gt(1).sum()),
        'single_recording_participants': int(participants.recordings.eq(1).sum()),
        'top_1_percent_recording_share': float(participants.recordings.nlargest(max(1,int(np.ceil(len(participants)*.01)))).sum()/len(recordings)) if len(recordings) else None,
        'followup_days': summary(participants.followup_days),
        'blockers': ['Diagnosis unavailable: no binary cohort or PD associations can be established'] if not cfg['columns']['diagnosis'] else [],
        'relationships': {'voice_to_participant': 'many-to-one, healthCode grouping; no external demographic or clinical join performed',
                          'shards': 'audited against consolidated source, never concatenated',
                          'countdown': 'handles inventoried; no local countdown mapping assumed; downloader retrieves only sustained phonation'}})
    print(f'Audit complete: {len(recordings)} recordings; {len(participants)} participants', flush=True)


if __name__ == '__main__':
    args = parser(__doc__).parse_args()
    run(configure(args.config))
