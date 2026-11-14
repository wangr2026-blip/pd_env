"""Protocol boundaries, native signal preservation, participant balance and sealing."""
import copy
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'scripts')]
from src.data.frozen_protocol import common_windows,native_channel_qc,decode_native_and_derive,participant_weights,development_frame


@pytest.mark.parametrize('seconds',[1.999,2,2.001,9.7,10,11])
def test_span_and_complete_window_coverage(seconds):
    y=np.arange(round(seconds*16000),dtype=np.float32)
    if seconds<2:
        with pytest.raises(ValueError,match='below_2s'):common_windows(y,'matched_2s')
        return
    windows,info=common_windows(y,'matched_2s');n=min(len(y),160000)
    assert info['starts'][0]==0 and info['starts'][-1]+32000==n
    assert all(len(w)==32000 for w in windows)
    assert all(step<=32000 for step in np.diff(info['starts']))
    covered=np.zeros(n,dtype=bool)
    for w,start in zip(windows,info['starts']):
        np.testing.assert_array_equal(w,y[start:start+32000]);covered[start:start+32000]=True
    assert covered.all() and info['overlap_samples']==len(windows)*32000-n
    whole,_=common_windows(y,'full');np.testing.assert_array_equal(whole[0],y[:160000])


def test_native_cancellation_and_clipping_flags():
    y=np.sin(np.arange(48000)*2*np.pi*150/48000)
    d=native_channel_qc(np.stack([y,-y],axis=1))
    assert d['cancellation_review'] and d['minimum_channel_correlation']==pytest.approx(-1)
    d=native_channel_qc(np.r_[np.ones(100),np.zeros(400)])
    assert d['channels'][0]['max_full_scale_run_samples']==100
    assert d['channels'][0]['max_zero_run_samples']==400


@pytest.mark.parametrize('rate,channels',[(16000,1),(44100,2),(48000,1)])
def test_native_decode_resampling_pitch_duration(tmp_path,rate,channels):
    import soundfile as sf
    t=np.arange(round(rate*2.2))/rate;y=.2*np.sin(2*np.pi*150*t)
    raw=y if channels==1 else np.stack([y,y],axis=1)
    path=tmp_path/'synthetic.wav';sf.write(path,raw,rate,subtype='FLOAT')
    actual,info=decode_native_and_derive(path,rate,channels)
    assert info['native_sample_count']==len(y) and info['native_channels']==channels
    assert abs(len(actual)/16000-2.2)<1/16000
    spectrum=np.abs(np.fft.rfft(actual));freq=np.fft.rfftfreq(len(actual),1/16000)[spectrum.argmax()]
    assert abs(freq-150)<1


def test_participant_weights_and_sealed_label_perturbation():
    ids=['a','b','b','c'];w=participant_weights(ids)
    assert {p:w[np.array(ids)==p].sum() for p in set(ids)}=={'a':1.,'b':1.,'c':1.}
    f=pd.DataFrame({'participant_id':['a','b','sealed'],'split':['train','val','test'],'diagnosis':['pd','control','whatever']})
    first=development_frame(f);f.loc[2,'diagnosis']='perturbed';pd.testing.assert_frame_equal(first,development_frame(f))


def test_weighted_fit_test_refusal_and_train_moments():
    from src.training.development import fit_development
    rng=np.random.default_rng(1);rows=[];features=[]
    for split in ['train','val']:
        for i in range(12):
            for visit in range(1+i%3):
                rows.append({'participant_id':f'{split}-{i}','recording_id':f'{split}-{i}-{visit}',
                    'duplicate_component':f'{split}-{i}','split':split,'diagnosis':'pd' if i%2 else 'control','label':i%2,'initial_recording':visit==0})
                features.append(rng.normal(size=4)+(i%2))
    frame=pd.DataFrame(rows);x=np.asarray(features)
    options={'cv_folds':2,'C_grid':[.1,1.],'hidden_dim':4,'learning_rate':.01,'weight_decay':.001,'epochs':2,'batch_size':8,'specificity_target':.9}
    result=fit_development(x,frame,options,4)
    train=frame.split.eq('train').to_numpy();weights=participant_weights(frame.loc[train].participant_id)
    np.testing.assert_allclose(result['logistic']['state']['scaler'].mean_,np.average(x[train],axis=0,weights=weights))
    assert set(result)=={'logistic','mlp'} and result['mlp']['initial_validation_predictions'].participant_id.nunique()==12
    altered=frame.copy();altered.loc[0,'split']='test'
    with pytest.raises(ValueError,match='sealed'):fit_development(x,altered,options,4)
    # Duplicating visits cannot alter participant-weighted scaler/logistic optimization.
    duplicate=pd.concat([frame,frame.loc[train]],ignore_index=True);xd=np.concatenate([x,x[train]])
    duplicate.loc[len(frame):,'initial_recording']=False
    again=fit_development(xd,duplicate,options,4)
    np.testing.assert_allclose(result['logistic']['state']['scaler'].mean_,again['logistic']['state']['scaler'].mean_)
    np.testing.assert_allclose(result['logistic']['probabilities'],again['logistic']['probabilities'],atol=1e-6)


@pytest.mark.parametrize('rate,channels',[(16000,1),(44100,2),(48000,2)])
def test_synthetic_alac_m4a_native_decode(tmp_path,rate,channels):
    import subprocess,soundfile as sf
    from mpower_common import ffmpeg
    y=.2*np.sin(2*np.pi*170*np.arange(int(rate*2.1))/rate)
    path=tmp_path/'tone.wav';target=tmp_path/'tone.m4a'
    sf.write(path,y if channels==1 else np.stack([y,y],axis=1),rate)
    subprocess.run([ffmpeg(),'-v','error','-i',str(path),'-c:a','alac',str(target)],check=True)
    result,info=decode_native_and_derive(target,rate,channels)
    assert info['native_sample_count']==len(y) and abs(len(result)/16000-2.1)<1/16000


def test_single_window_std_and_ast_pooler(tmp_path,monkeypatch):
    import torch,transformers
    from src.models.frozen import ProtocolEncoder
    torch.set_num_threads(1)
    c=transformers.ASTConfig(hidden_size=24,num_hidden_layers=2,num_attention_heads=2,intermediate_size=32)
    feature=transformers.ASTFeatureExtractor();model=transformers.ASTModel(c)
    monkeypatch.setattr(transformers.AutoConfig,'from_pretrained',lambda *a,**k:c)
    monkeypatch.setattr(transformers.AutoFeatureExtractor,'from_pretrained',lambda *a,**k:feature)
    monkeypatch.setattr(transformers.ASTModel,'from_pretrained',lambda *a,**k:model)
    cfg={'models':{'a':{'checkpoint':'synthetic','revision':'test'}},'device':'cpu','mixed_precision':False,'layers':[-1]}
    encoder=ProtocolEncoder('a',cfg,tmp_path);wave=np.random.default_rng(4).normal(0,.1,32000).astype('float32')
    with torch.inference_mode():expected=model(**feature([wave],sampling_rate=16000,return_tensors='pt')).pooler_output[0].numpy()
    vectors=encoder.represent(wave,['native_pooler','patches'])
    np.testing.assert_allclose(vectors['native_pooler'],expected,atol=1e-6)
    from scripts.run_frozen_development import recording_vectors
    v,coverage=recording_vectors(encoder,wave,'matched_2s',['native_pooler'])
    np.testing.assert_array_equal(v['native_pooler/window_mean_sd'][24:],np.zeros(24))


@pytest.mark.parametrize('kind',['wav2vec2','wavlm','hubert'])
def test_named_convolution_projection_and_layer_taps(kind,tmp_path,monkeypatch):
    import torch,transformers
    from src.models.frozen import ProtocolEncoder
    torch.set_num_threads(1)
    cls,config_cls={'wav2vec2':(transformers.Wav2Vec2Model,transformers.Wav2Vec2Config),
                   'wavlm':(transformers.WavLMModel,transformers.WavLMConfig),
                   'hubert':(transformers.HubertModel,transformers.HubertConfig)}[kind]
    c=config_cls(hidden_size=24,num_hidden_layers=2,num_attention_heads=2,intermediate_size=32,
                 conv_dim=(8,8,8),conv_stride=(5,4,4),conv_kernel=(10,8,8),
                 num_conv_pos_embedding_groups=2,num_conv_pos_embeddings=16)
    monkeypatch.setattr(transformers.AutoConfig,'from_pretrained',lambda *a,**k:c)
    monkeypatch.setattr(transformers.AutoFeatureExtractor,'from_pretrained',lambda *a,**k:transformers.Wav2Vec2FeatureExtractor())
    monkeypatch.setattr(cls,'from_pretrained',lambda *a,**k:cls(c))
    cfg={'models':{'a':{'checkpoint':'synthetic','revision':'test'}},'device':'cpu','mixed_precision':False,'layers':[-1]}
    encoder=ProtocolEncoder('a',cfg,tmp_path)
    y=np.random.default_rng(1).normal(0,.1,32000).astype('float32')
    before={k:v.clone() for k,v in encoder.model.state_dict().items()}
    out=encoder.represent(y,['conv_raw','conv_normalized','encoder_input','block_middle','last_hidden'])
    assert out['conv_raw/mean'].shape==(8,)
    assert out['conv_normalized/mean'].shape==(8,)
    assert out['encoder_input/mean'].shape==(24,)
    assert out['block_1/mean'].shape==(24,)
    assert out['last_hidden/mean_sd'].shape==(48,)
    for k,v in encoder.model.state_dict().items():torch.testing.assert_close(before[k],v,rtol=0,atol=0)
    assert all(not p.requires_grad for p in encoder.model.parameters())
