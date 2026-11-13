"""Shared configuration, private outputs, and provenance for Phase 1."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from functools import lru_cache

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--config', default=str(REPO / 'configs/mpower_preprocessing.yaml'))
    return p


def configure(path):
    os.umask(0o077)
    cfg = yaml.safe_load(Path(path).read_text())
    for key in ('data_root', 'output_root'):
        env = cfg[key + '_env']
        if not os.environ.get(env):
            raise ValueError(f'Set {env} to an approved directory outside the repository')
        cfg[key] = Path(os.environ[env]).expanduser().resolve()
        if cfg[key] == REPO or REPO in cfg[key].parents:
            raise ValueError(f'{env} must be outside the Git repository')
    if not cfg['data_root'].is_dir():
        raise FileNotFoundError(cfg['data_root'])
    if (cfg['output_root'] == cfg['data_root'] or cfg['output_root'] in cfg['data_root'].parents
            or cfg['data_root'] in cfg['output_root'].parents):
        raise ValueError('Output and raw data directories must not overlap')
    for d in ('processed', 'eda/tables', 'eda/figures', 'eda/diagnostics', 'cache'):
        (cfg['output_root'] / d).mkdir(parents=True, exist_ok=True)
    cfg['config_digest'] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return cfg


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=str, allow_nan=False))
    tmp.replace(path)


def write_table(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + '.tmp' + path.suffix)
    if path.suffix == '.parquet':
        frame.to_parquet(tmp, index=False)
    else:
        frame.to_csv(tmp, index=False)
    tmp.replace(path)


def load_table(path):
    path = Path(path)
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    if path.suffix in ('.json', '.jsonl'):
        return pd.read_json(path, lines=path.suffix == '.jsonl', dtype=False).astype('string')
    return pd.read_csv(path, sep='\t' if path.suffix == '.tsv' else ',', dtype='string', low_memory=False)


def ffmpeg():
    import shutil
    found = shutil.which('ffmpeg')
    if found:
        return found
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


@lru_cache(maxsize=1)
def version_info():
    from importlib.metadata import version, PackageNotFoundError
    names = ['numpy', 'pandas', 'scipy', 'librosa', 'praat-parselmouth', 'opensmile', 'pyarrow', 'soundfile']
    result = {'python': sys.version}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = 'unavailable'
    try:
        result['ffmpeg'] = subprocess.run([ffmpeg(), '-version'], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    except Exception as exc:
        result['ffmpeg'] = str(exc)
    return result


def decode(path, cfg):
    """Bound resources; reject long files rather than silently analyzing a truncation."""
    sr = cfg['sample_rate']
    limit = cfg['max_decode_seconds']
    proc = subprocess.run([ffmpeg(), '-v', 'error', '-nostdin', '-i', str(path),
        '-t', str(limit + 1), '-ac', '1', '-ar', str(sr), '-f', 'f32le', '-'],
        capture_output=True, timeout=120, check=True)
    y = np.frombuffer(proc.stdout, dtype='<f4').copy()
    if len(y) > limit * sr:
        raise ValueError('duration_exceeds_resource_limit; manual review required')
    if len(y) == 0 or not np.isfinite(y).all():
        raise ValueError('empty_or_nonfinite_decode')
    return y, sr


def cache_key(row, cfg, stage):
    path = Path(row['audio_path'])
    stat = path.stat()
    stamp = [stage, cfg['config_digest'], str(path), stat.st_size, stat.st_mtime_ns,
             hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             hashlib.sha256(Path(__file__).with_name('audio_quality.py').read_bytes()).hexdigest(),
             hashlib.sha256(Path(__file__).with_name('extract_acoustic_features.py').read_bytes()).hexdigest(), version_info()]
    stamp.append(hashlib.sha256(Path(__file__).with_name('inspect_audio.py').read_bytes()).hexdigest())
    return hashlib.sha256(json.dumps(stamp, sort_keys=True).encode()).hexdigest()


def summary(series):
    v = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan)
    return {'n': int(v.notna().sum()), 'missing': int(v.isna().sum()),
            **{k: (float(x) if pd.notna(x) else None) for k, x in
               {'mean': v.mean(), 'sd': v.std(), 'min': v.min(), 'max': v.max(),
                **{f'q{q:g}': v.quantile(q/100) for q in (1, 5, 25, 50, 75, 95, 99)}}.items()}}
