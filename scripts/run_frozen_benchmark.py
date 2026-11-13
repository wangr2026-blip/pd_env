#!/usr/bin/env python3
"""Unified frozen speech benchmark CLI; participant data never written into Git."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['ast', 'whisper', 'wav2vec2', 'all'], default='all')
    p.add_argument('--config', default=str(REPO / 'configs/frozen_benchmark.yaml'))
    p.add_argument('--smoke-test', action='store_true', help='Small real-recording end-to-end run; no performance claims')
    p.add_argument('--prepare-only', action='store_true', help='Download/load checkpoints without accessing recordings')
    p.add_argument('--data-smoke', action='store_true', help='Validate real manifests and decoding without pretrained models')
    return p


def initialize(cfg):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=cfg['threads'])
    import numpy as np
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(cfg['threads'])
    torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def data_smoke(cfg):
    from src.data.benchmark import manifest, segments
    frame, provenance = manifest(cfg, smoke=True)
    decoded = 0
    # Decode one real recording in every split/class, independently of checkpoint access.
    for _, group in frame.groupby(['split', 'diagnosis']):
        chunks = segments(group.iloc[0].audio_path, cfg)
        assert all(len(wave) > 0 for _, wave in chunks)
        decoded += 1
    return {**provenance, 'status': 'passed', 'recordings_selected': len(frame),
            'real_recordings_decoded': decoded,
            'participants_by_split_class': frame.groupby(['split', 'diagnosis']).participant_id.nunique().to_dict()}


def run(name, cfg, args):
    import joblib
    import numpy as np
    from mpower_common import save_json, write_table
    from src.data.benchmark import manifest, digest
    from src.models.frozen import FrozenEncoder, extract, pool
    from src.training.heads import train_head, attention_pool
    from src.evaluation.metrics import evaluate
    if args.prepare_only:
        encoder = FrozenEncoder(name, cfg, cfg['benchmark_root'] / 'models')
        save_json(cfg['benchmark_root'] / 'models' / f'{name}_provenance.json', encoder.provenance)
        return {'status': 'loaded', 'provenance': encoder.provenance}
    frame, source = manifest(cfg, smoke=args.smoke_test)
    tokens, entries, provenance = extract(frame, cfg, name)
    # Some eval-mode HF encoders still draw layer-drop RNG values. Decouple head
    # initialization from whether extraction ran or reused cached activations.
    initialize(cfg)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    mode = 'smoke' if args.smoke_test else 'baseline'
    out = cfg['benchmark_root'] / 'runs' / f'{stamp}_{name}_{mode}'
    out.mkdir(parents=True)
    save_json(out / 'hyperparameters.json', cfg)
    save_json(out / 'provenance.json', {**source, 'model': provenance,
              'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'torch', 'transformers']},
              'manifest_digest': digest(frame.to_dict('records')),
              'interpretation': 'Execution check only; not validated predictive performance' if args.smoke_test else 'Provisional cohort; research benchmark, pending full cohort audit'})
    # Immutable input snapshot identifies precisely which EDA version was consumed.
    write_table(frame, out / 'manifest.csv')
    save_json(out / 'activation_index.json', entries)
    if cfg['pooling'] == 'attention':
        x, attention = attention_pool(tokens, frame, cfg)
        import torch
        torch.save(attention, out / 'attention_pool.pt')
        save_json(out / 'attention_training_log.json', attention['logs'])
    else:
        x = pool(tokens, cfg['pooling'])
    np.save(out / 'recording_embeddings.npy', x)
    summaries = []
    for head in cfg['heads']:
        checkpoint, probabilities, logs = train_head(head, x, frame, cfg)
        directory = out / head
        directory.mkdir()
        joblib.dump(checkpoint, directory / 'head.joblib')
        prediction = frame[['recording_id', 'participant_id', 'split', 'label']].copy()
        prediction['probability'] = probabilities
        metrics, participants = evaluate(prediction, cfg['participant_aggregation'])
        save_json(directory / 'metrics.json', metrics)
        save_json(directory / 'training_log.json', logs)
        write_table(prediction, directory / 'recording_predictions.csv')
        write_table(participants, directory / 'participant_predictions.csv')
        summaries.append({'model': name, 'head': head, 'pooling': cfg['pooling'], 'mode': mode,
                          'run': str(out), **{k: v for k, v in metrics['participant']['test'].items() if k != 'calibration_bins'}})
    import pandas as pd
    write_table(pd.DataFrame(summaries), out / 'comparison.csv')
    return {'status': 'passed', 'run': str(out), 'smoke_test': args.smoke_test, 'recordings': len(frame), 'heads': cfg['heads']}


def main(argv=None):
    args = parser().parse_args(argv)
    # These bounds are set before torch/numpy imports and inherited by worker libraries.
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
    os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '10')
    os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '30')
    from src.data.benchmark import configure
    from mpower_common import save_json
    cfg = configure(args.config)
    if args.smoke_test:
        cfg['audio']['max_segments'] = min(cfg['audio']['max_segments'], 2)
        cfg['head']['epochs'] = min(cfg['head']['epochs'], 3)
    initialize(cfg)
    report = {'started_utc': datetime.now(timezone.utc).isoformat(), 'models': {}, 'arguments': vars(args)}
    failed = False
    if args.data_smoke:
        try:
            result = data_smoke(cfg)
            # JSON does not support tuple dict keys.
            result['participants_by_split_class'] = {f'{a}/{b}': n for (a, b), n in result['participants_by_split_class'].items()}
            report['data_smoke'] = result
        except Exception as exc:
            failed = True
            report['data_smoke'] = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
    else:
        names = ['ast', 'whisper', 'wav2vec2'] if args.model == 'all' else [args.model]
        for name in names:
            print(f'Starting {name}', flush=True)
            initialize(cfg)
            try:
                report['models'][name] = run(name, cfg, args)
            except Exception as exc:
                failed = True
                report['models'][name] = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
                traceback.print_exc()
            print(f"{name}: {report['models'][name]['status']}", flush=True)
            # Release cached GPU allocations before loading the next checkpoint.
            if 'torch' in sys.modules:
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path = cfg['benchmark_root'] / 'reports' / f'execution_{stamp}.json'
    save_json(path, report)
    print(json.dumps(report, indent=2, default=str), flush=True)
    print(f'Report: {path}', flush=True)
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
