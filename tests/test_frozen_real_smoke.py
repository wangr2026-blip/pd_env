"""Opt-in real-audio integration using tiny RANDOM backbones, not pretrained scores.

Set MPOWER_RUN_REAL_SMOKE=1 and the three storage roots to run. Outputs remain on
approved private storage and are explicitly labeled as random-backbone tests.
"""
import argparse
import os
from pathlib import Path
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = pytest.mark.skipif(os.environ.get('MPOWER_RUN_REAL_SMOKE') != '1', reason='Opt-in private real-recording test')


@pytest.mark.parametrize('name,pooling', [('ast', 'mean'), ('whisper', 'statistical'), ('wav2vec2', 'attention')])
def test_real_audio_random_architecture_end_to_end(name, pooling, monkeypatch):
    import transformers
    from transformers import ASTConfig, ASTModel, ASTFeatureExtractor, WhisperConfig, WhisperModel, WhisperFeatureExtractor
    from transformers import Wav2Vec2Config, Wav2Vec2Model, Wav2Vec2FeatureExtractor
    from src.data.benchmark import configure
    from scripts.run_frozen_benchmark import initialize, run
    repo = Path(__file__).resolve().parents[1]
    cfg = configure(repo / 'configs/frozen_benchmark.yaml')
    cfg['benchmark_root'] = cfg['benchmark_root'] / 'random_architecture_checks'
    cfg['benchmark_root'].mkdir(exist_ok=True)
    cfg.update(device='cpu', pooling=pooling, smoke_participants_per_class=1)
    cfg['head']['epochs'] = 2
    cfg['audio']['max_segments'] = 1
    cfg['models'][name] = {'checkpoint': f'test-only/random-{name}-not-pretrained', 'revision': 'test-only'}
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
    initialize(cfg)
    result = run(name, cfg, argparse.Namespace(prepare_only=False, smoke_test=True))
    assert result['status'] == 'passed' and result['recordings'] == 6
    assert result['heads'] == ['logistic', 'mlp']
    output = Path(result['run'])
    assert all((output / head / 'head.joblib').is_file() for head in result['heads'])
