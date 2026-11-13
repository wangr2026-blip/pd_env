"""Read-only EDA integration. All benchmark artifacts live outside the repository."""
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts'))
from create_participant_splits import assign_splits, verify
from mpower_common import decode


def configure(path):
    cfg = yaml.safe_load(Path(path).read_text())
    os.umask(0o077)
    for key in ('data', 'eda', 'benchmark'):
        env = cfg[f'{key}_root_env']
        if not os.environ.get(env):
            raise ValueError(f'Set {env} to approved persistent storage outside Git')
        cfg[f'{key}_root'] = Path(os.environ[env]).expanduser().resolve()
    root = cfg['benchmark_root']
    if root == REPO or REPO in root.parents:
        raise ValueError('Benchmark artifacts must stay outside Git')
    for key in ('data_root', 'eda_root'):
        other = cfg[key]
        if root == other or root in other.parents or other in root.parents:
            raise ValueError('Benchmark output must not overlap raw data or EDA outputs')
    root.mkdir(parents=True, exist_ok=True)
    if cfg['audio']['sample_rate'] != 16000:
        raise ValueError('All three checkpoints require 16 kHz audio')
    if not 0 < cfg['audio']['segment_seconds'] <= 10:
        raise ValueError('Use segments in (0, 10] seconds to fit AST and Whisper')
    if cfg['audio']['hop_seconds'] <= 0 or cfg['audio']['max_segments'] < 1:
        raise ValueError('Positive hop_seconds and max_segments required')
    return cfg


def read_table(path):
    return pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path, dtype={'participant_id': str, 'recording_id': str})


def manifest(cfg, smoke=False):
    root = cfg['eda_root']
    cohort = root / f"processed/cohort_{cfg['cohort']}"
    explicit = cfg.get('manifest')
    status = 'provisional'
    if explicit:
        frame = read_table(root / explicit)
        source = str(root / explicit)
    elif all((cohort / f'{s}_manifest.csv').exists() for s in ('train', 'val', 'test')):
        parts = []
        for split in ('train', 'val', 'test'):
            part = read_table(cohort / f'{split}_manifest.csv')
            if 'split' in part and not part['split'].eq(split).all():
                raise ValueError('Canonical split filename disagrees with split column')
            part['split'] = split
            parts.append(part)
        frame = pd.concat(parts, ignore_index=True)
        source = str(cohort)
        if frame.empty:
            frame = read_table(cohort / 'recording_manifest.parquet')
    elif (cohort / 'recording_manifest.parquet').exists():
        frame = read_table(cohort / 'recording_manifest.parquet')
        source = str(cohort / 'recording_manifest.parquet')
    else:
        frame = read_table(root / 'processed/recording_manifest.parquet')
        source = str(root / 'processed/recording_manifest.parquet')
    required = {'participant_id', 'recording_id', 'audio_path', 'diagnosis'}
    if required - set(frame):
        raise ValueError(f'Manifest lacks {required - set(frame)}; run diagnosis linkage first')
    frame = frame[frame.diagnosis.isin(['pd', 'control'])].copy()
    if frame[list(required)].isna().any().any():
        raise ValueError('Missing identity, audio path or diagnosis')
    if frame.groupby('participant_id').diagnosis.nunique().gt(1).any():
        raise ValueError('Conflicting participant diagnoses')
    if frame.recording_id.duplicated().any():
        raise ValueError('Duplicate recording IDs')
    if 'decode_status' in frame:
        frame = frame[frame.decode_status.eq('ok')]
    if 'duration_s' in frame:
        frame = frame[frame.duration_s.ge(cfg['quality']['min_duration_s'])]
    for flag in cfg['quality']['reject_flags']:
        if flag in frame:
            values = frame[flag].astype(str).str.lower()
            frame = frame[~values.isin(['true', '1'])]
    def resolve_audio(value):
        p = Path(value).expanduser()
        return p if p.is_absolute() else cfg['data_root'] / p
    frame['audio_path'] = frame.audio_path.map(resolve_audio).astype(str)
    frame = frame[frame.audio_path.map(lambda p: Path(p).is_file())].copy()
    if frame.empty:
        raise ValueError('No eligible labeled recordings with accessible audio')
    # Reuse audited EDA hashes; otherwise hash actual file bytes before creating or
    # validating splits. Different recording IDs must not conceal identical audio.
    if 'file_sha256' not in frame:
        frame['file_sha256'] = None
    needed = frame.file_sha256.isna() | frame.file_sha256.astype(str).eq('')
    hashes = {}
    for index, path in frame.loc[needed, 'audio_path'].items():
        if path not in hashes:
            sha = hashlib.sha256()
            with Path(path).open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    sha.update(block)
            hashes[path] = sha.hexdigest()
        frame.loc[index, 'file_sha256'] = hashes[path]
    if 'split' in frame:
        if not frame.split.isin(['train', 'val', 'test']).all():
            raise ValueError('Invalid or partially assigned canonical splits')
        verify(frame)
        # EDA marks current assignments provisional; never promote them silently.
        status = 'reused canonical assignments; release validation not asserted'
    else:
        _, frame = assign_splits(frame, cfg['seed'], cfg['split_fractions'])
    if smoke:
        # Preserve canonical assignments, select small class-balanced subsets per split.
        selected = []
        for (_, _), group in frame.groupby(['split', 'diagnosis']):
            ids = sorted(group.participant_id.unique())[:cfg['smoke_participants_per_class']]
            selected.append(group[group.participant_id.isin(ids)].groupby('participant_id').head(1))
        frame = pd.concat(selected, ignore_index=True)
    verify(frame)
    if frame.groupby('audio_path').split.nunique().gt(1).any():
        raise AssertionError('Leakage across splits: audio_path')
    for split in ('train', 'val', 'test'):
        if set(frame.loc[frame.split.eq(split), 'diagnosis']) != {'pd', 'control'}:
            raise ValueError(f'{split} needs both diagnoses; insufficient labeled participants')
    frame['label'] = frame.diagnosis.eq('pd').astype(int)
    return frame.reset_index(drop=True), {'source': source, 'split_status': status, 'smoke_test': smoke}


def segments(path, cfg):
    audio_cfg = cfg['audio']
    y, sr = decode(path, audio_cfg)
    if len(y) / sr < cfg['quality']['min_duration_s']:
        raise ValueError('Recording below configured minimum duration')
    width = round(audio_cfg['segment_seconds'] * sr)
    hop = round(audio_cfg['hop_seconds'] * sr)
    result = []
    for start in range(0, len(y), hop):
        chunk = y[start:start + width]
        if len(chunk) < cfg['quality']['min_duration_s'] * sr:
            continue
        # Leave padding to checkpoint feature extractors; keep true waveform lengths.
        result.append((start, np.asarray(chunk, dtype=np.float32)))
        if len(result) >= audio_cfg['max_segments'] or start + width >= len(y):
            break
    if not result:
        raise ValueError('No segments survive duration filtering')
    return result


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()
