"""Offline architecture, mask and normalization checks for expanded backbones."""
from pathlib import Path
import sys
import numpy as np
import pytest
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


@pytest.mark.parametrize('kind,normalize,norm', [('wavlm',False,'group'),('wavlm',True,'layer'),('hubert',True,'layer')])
def test_waveform_native_frontend_and_padding(kind,normalize,norm,tmp_path,monkeypatch):
    import torch,transformers
    from src.models.frozen import FrozenEncoder
    torch.set_num_threads(1)
    cls,config_cls={'wavlm':(transformers.WavLMModel,transformers.WavLMConfig),
                    'hubert':(transformers.HubertModel,transformers.HubertConfig)}[kind]
    c=config_cls(hidden_size=24,num_hidden_layers=1,num_attention_heads=2,intermediate_size=32,
                 conv_dim=(8,8,8),conv_stride=(5,4,4),conv_kernel=(10,8,8),
                 num_conv_pos_embedding_groups=2,num_conv_pos_embeddings=16,feat_extract_norm=norm)
    feature=transformers.Wav2Vec2FeatureExtractor(do_normalize=normalize,return_attention_mask=True)
    monkeypatch.setattr(transformers.AutoConfig,'from_pretrained',lambda *a,**k:c)
    monkeypatch.setattr(transformers.AutoFeatureExtractor,'from_pretrained',lambda *a,**k:feature)
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**k:pytest.fail('No tokenizer required'))
    backbone=cls(c);calls=[];forward=backbone.forward
    def capture(**kwargs):
        calls.append(kwargs)
        return forward(**kwargs)
    monkeypatch.setattr(backbone,'forward',capture)
    monkeypatch.setattr(cls,'from_pretrained',lambda *a,**k:backbone)
    cfg={'models':{kind:{'checkpoint':'synthetic','revision':'test'}},'device':'cpu','mixed_precision':False,'layers':[-1]}
    encoder=FrozenEncoder(kind,cfg,tmp_path)
    waves=[np.random.default_rng(i).normal(.2,.01,n).astype('float32') for i,n in enumerate([16000,8000])]
    batch=encoder(waves);single=encoder(waves[:1])
    np.testing.assert_allclose(batch[0]['-1'],single[0]['-1'],atol=1e-6)
    assert len(calls)==3 and all(('attention_mask' in call)==(norm!='group') for call in calls)
    expected=feature(waves[:1],sampling_rate=16000,return_tensors='pt').input_values
    torch.testing.assert_close(calls[0]['input_values'],expected)
    assert batch[1]['-1'].shape[0]==int(backbone._get_feat_extract_output_lengths(len(waves[1])))
    assert all(np.isfinite(row['-1']).all() for row in batch)


def test_local_checkpoint_missing_fails_without_hub(tmp_path,monkeypatch):
    from src.models.frozen import FrozenEncoder
    monkeypatch.setenv('MPOWER_MODEL_ROOT',str(tmp_path/'missing'))
    cfg={'models':{'alias':{'checkpoint':'example/model','revision':'pinned','local_alias':'alias'}}}
    with pytest.raises(FileNotFoundError,match='Verified local checkpoint missing'):
        FrozenEncoder('alias',cfg,tmp_path)


def test_all_nine_config_entries_match_lock():
    import json
    repo=Path(__file__).resolve().parents[1]
    cfg=yaml.safe_load((repo/'configs/frozen_benchmark_local.yaml').read_text())
    lock=json.loads((repo/'configs/pretrained_models.lock.json').read_text())['models']
    assert len(cfg['models'])==9
    for alias,c in cfg['models'].items():
        assert c['local_alias']==alias and c['revision']==lock[alias]['revision']
        assert c['checkpoint']==lock[alias]['repo_id'] and not lock[alias]['gated']


@pytest.mark.parametrize('mel_bins',[80,128])
def test_whisper_native_mel_bins_and_padding(mel_bins,tmp_path,monkeypatch):
    import torch,transformers
    from src.models.frozen import FrozenEncoder
    torch.set_num_threads(1)
    c=transformers.WhisperConfig(d_model=24,encoder_layers=1,decoder_layers=1,
        encoder_attention_heads=2,decoder_attention_heads=2,encoder_ffn_dim=32,
        decoder_ffn_dim=32,num_mel_bins=mel_bins)
    feature=transformers.WhisperFeatureExtractor(feature_size=mel_bins)
    monkeypatch.setattr(transformers.AutoConfig,'from_pretrained',lambda *a,**k:c)
    monkeypatch.setattr(transformers.AutoFeatureExtractor,'from_pretrained',lambda *a,**k:feature)
    monkeypatch.setattr(transformers.WhisperModel,'from_pretrained',lambda *a,**k:transformers.WhisperModel(c))
    cfg={'models':{'variant':{'checkpoint':'synthetic','revision':'test'}},'device':'cpu','mixed_precision':False,'layers':[-1]}
    model=FrozenEncoder('variant',cfg,tmp_path)
    wave=np.random.default_rng(2).normal(0,.1,32000).astype('float32')
    result=model([wave])[0]['-1']
    assert result.shape==(100,24) and np.isfinite(result).all()


def test_local_provenance_mismatch_rejected(tmp_path,monkeypatch):
    import json
    from src.models.frozen import FrozenEncoder
    root=tmp_path/'checkpoints/alias';root.mkdir(parents=True)
    (root/'download_provenance.json').write_text(json.dumps({'repo_id':'example/model','revision':'wrong'}))
    monkeypatch.setenv('MPOWER_MODEL_ROOT',str(tmp_path))
    cfg={'models':{'alias':{'checkpoint':'example/model','revision':'pinned','local_alias':'alias'}}}
    with pytest.raises(ValueError,match='provenance'):
        FrozenEncoder('alias',cfg,tmp_path)


def test_reverification_timestamp_preserves_model_identity(tmp_path,monkeypatch):
    import json
    import transformers
    from src.models.frozen import FrozenEncoder
    c=transformers.ASTConfig(hidden_size=24,num_hidden_layers=1,num_attention_heads=2,intermediate_size=32)
    monkeypatch.setattr(transformers.AutoConfig,'from_pretrained',lambda *a,**k:c)
    monkeypatch.setattr(transformers.AutoFeatureExtractor,'from_pretrained',lambda *a,**k:transformers.ASTFeatureExtractor())
    monkeypatch.setattr(transformers.ASTModel,'from_pretrained',lambda *a,**k:transformers.ASTModel(c))
    root=tmp_path/'checkpoints/alias';root.mkdir(parents=True)
    monkeypatch.setenv('MPOWER_MODEL_ROOT',str(tmp_path))
    cfg={'models':{'alias':{'checkpoint':'synthetic/ast','revision':'pinned','local_alias':'alias'}},'device':'cpu','mixed_precision':False,'layers':[-1]}
    identities=[]
    for stamp in ['first-verification','later-verification']:
        (root/'download_provenance.json').write_text(json.dumps({'repo_id':'synthetic/ast','revision':'pinned','files':[],'verified_at':stamp}))
        identities.append(FrozenEncoder('alias',cfg,tmp_path).provenance)
    assert identities[0]==identities[1]
