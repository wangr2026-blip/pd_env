"""Checkpoint-native frontends and frozen encoders; no decoder or CTC inference."""
from contextlib import nullcontext
from pathlib import Path
import json
import os

import numpy as np


class FrozenEncoder:
    def __init__(self, name, cfg, cache_dir):
        import torch
        from transformers import AutoConfig, AutoFeatureExtractor
        from transformers import ASTModel, WhisperModel, Wav2Vec2Model, WavLMModel, HubertModel
        self.torch = torch
        self.name = name
        self.cfg = cfg
        model_cfg = cfg['models'][name]
        kwargs = dict(cache_dir=str(cache_dir), revision=model_cfg['revision'])
        checkpoint = model_cfg['checkpoint']
        source_provenance = None
        if model_cfg.get('local_alias'):
            root = Path(os.environ.get(cfg.get('model_root_env', 'MPOWER_MODEL_ROOT'),
                                       str(cache_dir))).expanduser().resolve()
            directory = root / 'checkpoints' / model_cfg['local_alias']
            if not directory.is_dir():
                raise FileNotFoundError(f'Verified local checkpoint missing: {directory}; run download_pretrained_models.py')
            source_provenance = json.loads((directory / 'download_provenance.json').read_text())
            # Verification timestamps change on rechecks; asset hashes/revisions do not.
            source_provenance.pop('verified_at', None)
            if (source_provenance['revision'] != model_cfg['revision'] or
                    source_provenance['repo_id'] != model_cfg['checkpoint']):
                raise ValueError('Local checkpoint provenance disagrees with pinned benchmark config')
            checkpoint = str(directory)
            kwargs = dict(local_files_only=True)
        config = AutoConfig.from_pretrained(checkpoint, **kwargs)
        self.kind = config.model_type
        resolved = getattr(config, '_commit_hash', None)
        if resolved and source_provenance is None:
            kwargs['revision'] = resolved
        self.feature = AutoFeatureExtractor.from_pretrained(checkpoint, **kwargs)
        if self.feature.sampling_rate != 16000:
            raise ValueError('Checkpoint frontend does not expect 16 kHz')
        classes = {'audio-spectrogram-transformer': ASTModel, 'whisper': WhisperModel,
                   'wav2vec2': Wav2Vec2Model, 'wavlm': WavLMModel, 'hubert': HubertModel}
        if self.kind not in classes:
            raise ValueError(f'Unsupported backbone architecture: {self.kind}')
        # Acoustic encoder inputs require a feature extractor, not an ASR tokenizer.
        model = classes[self.kind].from_pretrained(checkpoint, **kwargs)
        self.model = model.get_encoder() if self.kind == 'whisper' else model
        self.model.requires_grad_(False).eval()
        device = cfg['device']
        self.device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available() else ('cpu' if device == 'auto' else device))
        self.model.to(self.device)
        self.amp = bool(cfg['mixed_precision'] and self.device.type == 'cuda')
        self.provenance = {'checkpoint': checkpoint, 'requested_revision': model_cfg['revision'],
                           'resolved_revision': source_provenance['revision'] if source_provenance else resolved,
                           'source_repository': model_cfg['checkpoint'], 'architecture': self.kind,
                           'local_assets': source_provenance,
                           'feature_extractor': self.feature.to_dict(), 'config': config.to_dict(),
                           'device': str(self.device), 'mixed_precision': self.amp}
        if any(p.requires_grad for p in self.model.parameters()):
            raise AssertionError('Backbone must be frozen')

    def __call__(self, waveforms):
        torch = self.torch
        # Group-normalized wav2vec2-base is sensitive to waveform padding. Single-item
        # forwards preserve exact valid inputs independently of batch composition.
        if self.kind in ('wav2vec2', 'wavlm', 'hubert') and len(waveforms) > 1:
            return [self([waveform])[0] for waveform in waveforms]
        if self.kind == 'whisper':
            inputs = self.feature(waveforms, sampling_rate=16000, return_tensors='pt',
                                  padding='max_length', truncation=True, return_attention_mask=True)
            valid_mel_frames = inputs.attention_mask.sum(-1)
            inputs = {'input_features': inputs.input_features}
        elif self.kind == 'audio-spectrogram-transformer':
            inputs = self.feature(waveforms, sampling_rate=16000, return_tensors='pt')
            inputs = {'input_values': inputs.input_values}
        else:
            native = self.feature(waveforms, sampling_rate=16000, return_tensors='pt',
                                  padding=True, return_attention_mask=True)
            inputs = {'input_values': native.input_values}
            # Group-normalized base models were pretrained without encoder masks;
            # layer-normalized variants receive their checkpoint-specific masks.
            if getattr(self.model.config, 'feat_extract_norm', 'group') != 'group':
                inputs['attention_mask'] = native.attention_mask
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        context = torch.autocast('cuda', dtype=torch.float16) if self.amp else nullcontext()
        with torch.inference_mode(), context:
            output = self.model(**inputs, output_hidden_states=True, return_dict=True)
        result = []
        for index, waveform in enumerate(waveforms):
            layers = {}
            for layer in self.cfg['layers']:
                hidden = output.hidden_states[layer][index]
                if self.kind == 'audio-spectrogram-transformer':
                    # AST has CLS/distillation tokens followed by frequency-major patches.
                    hidden = hidden[2:]
                    c = self.model.config
                    time_patches = (c.max_length - c.patch_size) // c.time_stride + 1
                    frequency_patches = (c.num_mel_bins - c.patch_size) // c.frequency_stride + 1
                    valid_frames = max(1, (len(waveform) - 400) // 160 + 1)
                    valid_time = max(1, min(time_patches, (valid_frames - c.patch_size) // c.time_stride + 1))
                    hidden = hidden.reshape(frequency_patches, time_patches, -1)[:, :valid_time].reshape(-1, hidden.shape[-1])
                elif self.kind == 'whisper':
                    # Encoder stride is 2 over 10 ms Mel frames; remove trailing padding tokens.
                    valid = min(hidden.shape[0], (int(valid_mel_frames[index]) + 1) // 2)
                    hidden = hidden[:max(1, valid)]
                elif self.kind in ('wav2vec2', 'wavlm', 'hubert'):
                    valid = int(self.model._get_feat_extract_output_lengths(len(waveform)))
                    hidden = hidden[:max(1, min(hidden.shape[0], valid))]
                layers[str(layer)] = hidden.float().cpu().numpy()
            result.append(layers)
        return result


def extract(frame, cfg, name):
    """Content-addressed segment inputs and hidden states, atomically written/resumable."""
    import hashlib
    from src.data.benchmark import digest, segments
    from mpower_common import save_json, version_info
    root = cfg['benchmark_root']
    encoder = FrozenEncoder(name, cfg, root / 'models')
    save_json(root / 'models' / f'{name}_provenance.json', encoder.provenance)
    implementation = hashlib.sha256(Path(__file__).read_bytes()
        + Path(__file__).parents[1].joinpath('data/benchmark.py').read_bytes()
        + Path(__file__).parents[2].joinpath('scripts/mpower_common.py').read_bytes()).hexdigest()
    import importlib.metadata
    versions = {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'numpy')}
    signature = digest({'model': encoder.provenance, 'layers': cfg['layers'], 'audio': cfg['audio'],
                        'versions': versions, 'implementation': implementation})
    token_dir = root / 'activations' / name / signature
    token_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = root / 'segments'
    audio_dir.mkdir(exist_ok=True)
    entries = []
    for row in frame.itertuples():
        path = Path(row.audio_path)
        # Hash file bytes, not participant identifiers; catches in-place source replacement.
        sha = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                sha.update(block)
        audio_key = digest({'sha256': sha.hexdigest(), 'audio': cfg['audio'],
                            'min_duration': cfg['quality']['min_duration_s'], 'implementation': implementation,
                            'ffmpeg': version_info()['ffmpeg']})
        audio_file = audio_dir / f'{audio_key}.npz'
        if audio_file.exists():
            with np.load(audio_file, allow_pickle=False) as cached:
                chunks = [(int(k.split('_')[1]), cached[k].copy()) for k in cached.files]
        else:
            chunks = segments(path, cfg)
            temporary = audio_file.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, **{f's_{offset}': wave for offset, wave in chunks})
            temporary.replace(audio_file)
        paths = [token_dir / f'{audio_key}_{offset}.npz' for offset, _ in chunks]
        missing = [i for i, p in enumerate(paths) if not p.exists()]
        for start in range(0, len(missing), cfg['batch_size']):
            indices = missing[start:start + cfg['batch_size']]
            outputs = encoder([chunks[i][1] for i in indices])
            for i, layers in zip(indices, outputs):
                if any(not np.isfinite(x).all() for x in layers.values()):
                    raise ValueError('Nonfinite backbone activations')
                tmp = paths[i].with_suffix('.tmp.npz')
                np.savez_compressed(tmp, **layers)
                tmp.replace(paths[i])
        entries.append({'recording_id': row.recording_id, 'segments': [str(p) for p in paths],
                        'waveforms': str(audio_file), 'signature': signature})
    return TokenStore(entries, str(cfg['layers'][-1])), entries, encoder.provenance


class TokenStore:
    """Lazy disk-backed recording token bags, bounded to one recording in memory."""
    def __init__(self, entries, layer):
        self.entries, self.layer = entries, layer

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        arrays = []
        for path in self.entries[index]['segments']:
            with np.load(path, allow_pickle=False) as cached:
                arrays.append(cached[self.layer].copy())
        return np.concatenate(arrays)


def pool(tokens, kind):
    if kind == 'mean':
        return np.stack([x.mean(0) for x in tokens])
    if kind == 'statistical':
        return np.stack([np.concatenate([x.mean(0), x.std(0)]) for x in tokens])
    raise ValueError(f'Unknown fixed pooling: {kind}')


class ProtocolEncoder(FrozenEncoder):
    """Selective named taps, native AST pooling, pooled vectors only."""
    def represent(self, waveform, taps=None):
        torch=self.torch
        taps=taps or (['native_pooler','patches'] if self.kind=='audio-spectrogram-transformer'
                      else ['last_hidden','block_middle'])
        feature_options={'sampling_rate':16000,'return_tensors':'pt'}
        length=len(waveform);captures={};handles=[]
        def capture(name,transpose=False):
            def hook(module,args,output):
                value=output[0] if isinstance(output,tuple) else output
                captures[name]=value.transpose(1,2) if transpose else value
            return hook
        def before(name):
            def hook(module,args):captures[name]=args[0]
            return hook
        if self.kind in ('wav2vec2','wavlm','hubert'):
            native=self.feature([waveform],padding=False,return_attention_mask=True,**feature_options)
            inputs={'input_values':native.input_values}
            if getattr(self.model.config,'feat_extract_norm','group')!='group':inputs['attention_mask']=native.attention_mask
            if 'conv_raw' in taps:handles.append(self.model.feature_extractor.register_forward_hook(capture('conv_raw',True)))
            if 'conv_normalized' in taps:handles.append(self.model.feature_projection.projection.register_forward_pre_hook(before('conv_normalized')))
            layers=self.model.encoder.layers
            if 'encoder_input' in taps:handles.append(layers[0].register_forward_pre_hook(before('encoder_input')))
        elif self.kind=='whisper':
            native=self.feature([waveform],padding='max_length',truncation=True,return_attention_mask=True,do_normalize=False,**feature_options)
            inputs={'input_features':native.input_features}
            valid_whisper=(int(native.attention_mask[0].sum())+1)//2
            layers=self.model.layers
        else:
            native=self.feature([waveform],**feature_options);inputs={'input_values':native.input_values}
            layers=self.model.encoder.layer
        middle=max(1,len(layers)//2)
        if 'block_middle' in taps:handles.append(layers[middle-1].register_forward_hook(capture('block_middle')))
        try:
            with torch.inference_mode():
                output=self.model(**{k:v.to(self.device) for k,v in inputs.items()},output_hidden_states=False,return_dict=True)
                captures['last_hidden']=output.last_hidden_state
                result={}
                for tap in taps:
                    if tap=='native_pooler':
                        if self.kind!='audio-spectrogram-transformer':raise ValueError('native_pooler applies to AST')
                        result['native_pooler']=output.pooler_output[0].float().cpu().numpy();continue
                    sequence=captures['last_hidden'] if tap=='patches' else captures[tap]
                    sequence=sequence[0]
                    if self.kind=='audio-spectrogram-transformer':
                        sequence=sequence[2:];c=self.model.config
                        nt=(c.max_length-c.patch_size)//c.time_stride+1;nf=(c.num_mel_bins-c.patch_size)//c.frequency_stride+1
                        frames=max(1,(length-400)//160+1);valid=max(1,min(nt,(frames-c.patch_size)//c.time_stride+1))
                        sequence=sequence.reshape(nf,nt,-1)[:,:valid].reshape(-1,sequence.shape[-1])
                    elif self.kind=='whisper':sequence=sequence[:max(1,valid_whisper)]
                    else:
                        valid=int(self.model._get_feat_extract_output_lengths(length));sequence=sequence[:max(1,valid)]
                    sequence=sequence.float()
                    mean=sequence.mean(0);sd=sequence.std(0,correction=0)
                    name=f'block_{middle}' if tap=='block_middle' else tap
                    result[name+'/mean']=mean.cpu().numpy()
                    result[name+'/mean_sd']=torch.cat([mean,sd]).cpu().numpy()
                if any(not np.isfinite(v).all() for v in result.values()):raise ValueError('Nonfinite pooled features')
                return result
        finally:
            for handle in handles:handle.remove()
