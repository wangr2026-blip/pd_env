"""Synthetic correctness tests; tiny random models are not pretrained validation."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.benchmark import configure, manifest, segments
from src.evaluation.metrics import binary_metrics, evaluate
from src.models.frozen import FrozenEncoder, TokenStore, pool
from src.training.heads import train_head, attention_pool

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    return yaml.safe_load((REPO / 'configs/frozen_benchmark.yaml').read_text())


@pytest.fixture
def synthetic_frame(tmp_path):
    rows = []
    for split in ('train', 'val', 'test'):
        for label in (0, 1):
            for i in range(3):
                rows.append({'participant_id': f'synthetic-{split}-{label}-{i}',
                             'recording_id': f'synthetic-record-{split}-{label}-{i}',
                             'diagnosis': 'pd' if label else 'control', 'label': label,
                             'split': split, 'audio_path': str(tmp_path / 'synthetic.wav')})
    return pd.DataFrame(rows)


def test_metrics_and_participant_aggregation(synthetic_frame):
    f = synthetic_frame.copy()
    f['probability'] = np.where(f.label, .9, .1)
    metrics, participants = evaluate(f)
    assert len(participants) == len(f)
    for level in metrics.values():
        for result in level.values():
            assert result['auroc'] == result['auprc'] == result['f1'] == 1
            assert result['sensitivity'] == result['specificity'] == 1
            assert result['brier'] == pytest.approx(.01)
            assert result['ece_10_bins'] == pytest.approx(.1)
            assert sum(b['n'] for b in result['calibration_bins']) == result['n']
    assert binary_metrics([1, 1], [.1, .5])['auroc'] is None
    # Participant decisions average probabilities, not binary predictions.
    duplicate = f.iloc[[0]].copy()
    duplicate['recording_id'] = 'synthetic-repeat'
    duplicate['probability'] = .7
    _, people = evaluate(pd.concat([f, duplicate]))
    person = people[people.participant_id.eq(f.iloc[0].participant_id)].iloc[0]
    assert person.probability == pytest.approx(.4) and person.recordings == 2


def test_pooling_and_lazy_activations(tmp_path):
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    y = x + 10
    np.savez(tmp_path / 'a.npz', **{'-1': x})
    np.savez(tmp_path / 'b.npz', **{'-1': y})
    store = TokenStore([{'segments': [str(tmp_path / 'a.npz'), str(tmp_path / 'b.npz')]}], '-1')
    np.testing.assert_array_equal(store[0], np.concatenate([x, y]))
    np.testing.assert_allclose(pool(store, 'mean')[0], store[0].mean(0))
    np.testing.assert_allclose(pool(store, 'statistical')[0], np.r_[store[0].mean(0), store[0].std(0)])


@pytest.mark.parametrize('head', ['logistic', 'mlp'])
def test_heads_train_only_normalization(head, cfg, synthetic_frame):
    f = synthetic_frame
    rng = np.random.default_rng(4)
    x = rng.normal(size=(len(f), 8))
    x[f.split.eq('test')] += 1000  # Held-out distribution must never affect fitted scaler.
    cfg['head']['epochs'] = 3
    model, probabilities, logs = train_head(head, x, f, cfg)
    np.testing.assert_allclose(model[0].mean_, x[f.split.eq('train')].mean(0))
    assert probabilities.shape == (len(f),) and np.isfinite(probabilities).all()
    if head == 'mlp':
        assert len(logs) == 3


def test_attention_training_uses_train_only_statistics(cfg, synthetic_frame):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(1)
    torch.manual_seed(1)
    rng = np.random.default_rng(1)
    tokens = [rng.normal(size=(3 + i % 3, 8)).astype(np.float32) for i in range(len(synthetic_frame))]
    cfg['head']['epochs'] = 2
    x, state = attention_pool(tokens, synthetic_frame, cfg)
    train_tokens = np.concatenate([t for t, split in zip(tokens, synthetic_frame.split) if split == 'train'])
    np.testing.assert_allclose(state['mean'], train_tokens.mean(0), atol=1e-6)
    assert x.shape == (len(tokens), 8) and np.isfinite(x).all()
    assert len(state['logs']) == 2


def test_manifest_reuses_splits_and_rejects_leakage(tmp_path, cfg, synthetic_frame):
    f = synthetic_frame.copy()
    for index in f.index:
        audio = tmp_path / f'synthetic-{index}.wav'
        audio.write_bytes(f'synthetic-unique-audio-{index}'.encode())
        f.loc[index, 'audio_path'] = str(audio)
    path = tmp_path / 'manifest.csv'
    f.to_csv(path, index=False)
    cfg.update(eda_root=tmp_path, data_root=tmp_path, manifest='manifest.csv')
    result, provenance = manifest(cfg)
    assert result.split.tolist() == f.split.tolist()
    assert provenance['split_status'].startswith('reused')
    f.loc[6, 'participant_id'] = f.loc[0, 'participant_id']
    f.to_csv(path, index=False)
    with pytest.raises(AssertionError, match='Leakage'):
        manifest(cfg)
    f.loc[6, 'participant_id'] = 'synthetic-distinct-person'
    Path(f.loc[6, 'audio_path']).write_bytes(Path(f.loc[0, 'audio_path']).read_bytes())
    f.to_csv(path, index=False)
    with pytest.raises(AssertionError, match='file_sha256'):
        manifest(cfg)


def test_provisional_splits_duplicate_components(tmp_path, cfg, synthetic_frame):
    f = synthetic_frame.drop(columns='split').copy()
    for index in f.index:
        audio = tmp_path / f'synthetic-{index}.wav'
        audio.write_bytes(f'synthetic-unique-audio-{index}'.encode())
        f.loc[index, 'audio_path'] = str(audio)
    # Two participants share source content and must stay in the same partition.
    f['file_sha256'] = f.recording_id
    f.loc[[0, 1], 'file_sha256'] = 'synthetic-duplicate'
    f.to_csv(tmp_path / 'manifest.csv', index=False)
    cfg.update(eda_root=tmp_path, data_root=tmp_path, manifest='manifest.csv')
    result, provenance = manifest(cfg)
    assert provenance['split_status'] == 'provisional'
    assert result[result.file_sha256.eq('synthetic-duplicate')].split.nunique() == 1


def test_output_paths_and_sample_rate(tmp_path, cfg, monkeypatch):
    monkeypatch.setenv('MPOWER_DATA_ROOT', str(tmp_path / 'raw'))
    monkeypatch.setenv('MPOWER_OUTPUT_ROOT', str(tmp_path / 'eda'))
    monkeypatch.setenv('MPOWER_BENCHMARK_ROOT', str(tmp_path / 'benchmark'))
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump(cfg))
    assert configure(config)['benchmark_root'] == tmp_path / 'benchmark'
    monkeypatch.setenv('MPOWER_BENCHMARK_ROOT', str(tmp_path / 'eda' / 'benchmark'))
    with pytest.raises(ValueError, match='overlap'):
        configure(config)
    monkeypatch.setenv('MPOWER_BENCHMARK_ROOT', str(tmp_path / 'benchmark'))
    cfg['audio']['sample_rate'] = 8000
    config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='16 kHz'):
        configure(config)


def test_segmentation_and_native_resampling(tmp_path, cfg):
    import soundfile as sf
    sr = 22050
    y = .1 * np.sin(2 * np.pi * 150 * np.arange(sr * 12) / sr)
    path = tmp_path / 'synthetic.wav'
    sf.write(path, np.stack([y, y], axis=1), sr)
    chunks = segments(path, cfg)
    assert [offset for offset, _ in chunks] == [0, 80000, 160000]
    assert [len(wave) for _, wave in chunks] == [80000, 80000, 32000]
    assert all(wave.dtype == np.float32 and wave.ndim == 1 for _, wave in chunks)


@pytest.mark.parametrize('name', ['ast', 'whisper', 'wav2vec2'])
def test_tiny_random_backbone_frontend_freeze_and_shapes(name, cfg, tmp_path, monkeypatch):
    """Exercise real HF implementations locally; no checkpoint weights are downloaded."""
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    from transformers import (ASTConfig, ASTModel, ASTFeatureExtractor, WhisperConfig,
                              WhisperModel, WhisperFeatureExtractor, Wav2Vec2Config,
                              Wav2Vec2Model, Wav2Vec2FeatureExtractor)
    torch.set_num_threads(1)
    if name == 'ast':
        config = ASTConfig(hidden_size=24, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32)
        feature = ASTFeatureExtractor()
        cls = ASTModel
    elif name == 'whisper':
        config = WhisperConfig(d_model=24, encoder_layers=1, decoder_layers=1, encoder_attention_heads=2,
                               decoder_attention_heads=2, encoder_ffn_dim=32, decoder_ffn_dim=32)
        feature = WhisperFeatureExtractor()
        cls = WhisperModel
    else:
        config = Wav2Vec2Config(hidden_size=24, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
                                conv_dim=(8, 8, 8), conv_stride=(5, 4, 4), conv_kernel=(10, 8, 8),
                                num_conv_pos_embedding_groups=2, num_conv_pos_embeddings=16)
        feature = Wav2Vec2FeatureExtractor(return_attention_mask=False)
        cls = Wav2Vec2Model
    monkeypatch.setattr(transformers.AutoFeatureExtractor, 'from_pretrained', lambda *a, **k: feature)
    monkeypatch.setattr(transformers.AutoConfig, 'from_pretrained', lambda *a, **k: config)
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **k: object())
    monkeypatch.setattr(cls, 'from_pretrained', lambda *a, **k: cls(config))
    cfg['device'] = 'cpu'
    cfg['layers'] = [0, -1]
    encoder = FrozenEncoder(name, cfg, tmp_path)
    assert not encoder.model.training
    assert all(not p.requires_grad for p in encoder.model.parameters())
    waves = [np.random.default_rng(i).normal(0, .05, n).astype(np.float32) for i, n in enumerate([32000, 40000])]
    outputs = encoder(waves)
    assert len(outputs) == 2
    for layers in outputs:
        assert set(layers) == {'0', '-1'}
        for x in layers.values():
            assert x.ndim == 2 and x.shape[1] == 24 and np.isfinite(x).all()
    # Variable-length batch composition must not change the first recording's tokens.
    np.testing.assert_allclose(outputs[0]['-1'], encoder(waves[:1])[0]['-1'], atol=2e-5, rtol=2e-4)


def test_extraction_resumes_and_caches_waveforms(tmp_path, cfg, monkeypatch):
    from src.models import frozen
    frame = pd.DataFrame({'recording_id': ['synthetic-r'], 'audio_path': [str(tmp_path / 'synthetic.wav')]})
    frame_path = Path(frame.audio_path.iloc[0])
    frame_path.write_bytes(b'synthetic fake audio for cache test')
    cfg['benchmark_root'] = tmp_path / 'private'
    cfg['benchmark_root'].mkdir()
    calls = {'decode': 0, 'forward': 0}
    class FakeEncoder:
        def __init__(self, *args):
            self.provenance = {'test': 'synthetic-fake'}
        def __call__(self, waveforms):
            calls['forward'] += 1
            return [{'-1': np.ones((3, 4), dtype=np.float32) * wave.mean()} for wave in waveforms]
    def fake_segments(*args):
        calls['decode'] += 1
        return [(0, np.ones(32000, dtype=np.float32)), (32000, np.ones(32000, dtype=np.float32) * 2)]
    monkeypatch.setattr(frozen, 'FrozenEncoder', FakeEncoder)
    import src.data.benchmark as data
    monkeypatch.setattr(data, 'segments', fake_segments)
    first, entries, _ = frozen.extract(frame, cfg, 'ast')
    second, repeated, _ = frozen.extract(frame, cfg, 'ast')
    assert calls == {'decode': 1, 'forward': 1}
    assert entries == repeated
    np.testing.assert_array_equal(first[0], second[0])
    frame_path.write_bytes(b'changed content invalidates cache')
    frozen.extract(frame, cfg, 'ast')
    assert calls == {'decode': 2, 'forward': 2}


@pytest.mark.parametrize('name,pooling', [('ast', 'mean'), ('whisper', 'statistical'), ('wav2vec2', 'attention')])
def test_runner_artifacts_end_to_end_with_tiny_models(name, pooling, cfg, tmp_path, monkeypatch):
    """Full synthetic file-backed run through actual frontends/models and both heads."""
    import argparse
    import json
    import soundfile as sf
    import transformers
    from scripts.run_frozen_benchmark import run, initialize
    from transformers import ASTConfig, ASTModel, ASTFeatureExtractor, WhisperConfig, WhisperModel, WhisperFeatureExtractor
    from transformers import Wav2Vec2Config, Wav2Vec2Model, Wav2Vec2FeatureExtractor
    if name == 'ast':
        config = ASTConfig(hidden_size=24, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32)
        feature, cls = ASTFeatureExtractor(), ASTModel
    elif name == 'whisper':
        config = WhisperConfig(d_model=24, encoder_layers=1, decoder_layers=1, encoder_attention_heads=2,
                               decoder_attention_heads=2, encoder_ffn_dim=32, decoder_ffn_dim=32)
        feature, cls = WhisperFeatureExtractor(), WhisperModel
    else:
        config = Wav2Vec2Config(hidden_size=24, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
                                conv_dim=(8, 8, 8), conv_stride=(5, 4, 4), conv_kernel=(10, 8, 8),
                                num_conv_pos_embedding_groups=2, num_conv_pos_embeddings=16)
        feature, cls = Wav2Vec2FeatureExtractor(return_attention_mask=False), Wav2Vec2Model
    monkeypatch.setattr(transformers.AutoFeatureExtractor, 'from_pretrained', lambda *a, **k: feature)
    monkeypatch.setattr(transformers.AutoConfig, 'from_pretrained', lambda *a, **k: config)
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **k: object())
    monkeypatch.setattr(cls, 'from_pretrained', lambda *a, **k: cls(config))
    cfg.update(benchmark_root=tmp_path / 'benchmark', data_root=tmp_path, eda_root=tmp_path,
               manifest='manifest.csv', device='cpu', pooling=pooling)
    cfg['head']['epochs'] = 2
    cfg['benchmark_root'].mkdir()
    rows = []
    for index, (split, label) in enumerate((s, y) for s in ('train', 'val', 'test') for y in (0, 1)):
        path = tmp_path / f'synthetic-{index}.wav'
        sr = 16000
        time = np.arange(sr * 2) / sr
        sf.write(path, .1 * np.sin(2 * np.pi * (130 + index * 10) * time), sr)
        rows.append({'recording_id': f'synthetic-r{index}', 'participant_id': f'synthetic-p{index}',
                     'diagnosis': 'pd' if label else 'control', 'split': split, 'audio_path': str(path)})
    pd.DataFrame(rows).to_csv(tmp_path / 'manifest.csv', index=False)
    initialize(cfg)
    result = run(name, cfg, argparse.Namespace(prepare_only=False, smoke_test=True))
    assert result['status'] == 'passed' and result['recordings'] == 6
    output = Path(result['run'])
    assert np.load(output / 'recording_embeddings.npy').shape == (6, 48 if pooling == 'statistical' else 24)
    assert len(json.loads((output / 'activation_index.json').read_text())) == 6
    if pooling == 'attention':
        assert (output / 'attention_pool.pt').exists()
    for head in ('logistic', 'mlp'):
        metrics = json.loads((output / head / 'metrics.json').read_text())
        assert metrics['recording']['test']['n'] == 2
        predictions = pd.read_csv(output / head / 'recording_predictions.csv')
        assert len(predictions) == 6 and predictions.probability.between(0, 1).all()
        import joblib
        checkpoint = joblib.load(output / head / 'head.joblib')
        np.testing.assert_allclose(checkpoint.predict_proba(np.load(output / 'recording_embeddings.npy'))[:, 1], predictions.probability)
    # A second run reuses all activations and produces the same predictions under the same seed.
    initialize(cfg)
    repeated = run(name, cfg, argparse.Namespace(prepare_only=False, smoke_test=True))
    for head in ('logistic', 'mlp'):
        first = pd.read_csv(output / head / 'recording_predictions.csv')
        second = pd.read_csv(Path(repeated['run']) / head / 'recording_predictions.csv')
        np.testing.assert_array_equal(first.probability, second.probability)
