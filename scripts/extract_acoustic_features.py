#!/usr/bin/env python3
"""Established Praat/openSMILE/librosa features; undefined measurements stay NaN.

Run through inspect_audio.py --stage features to share decoding and checkpoints.
"""
from functools import lru_cache

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
import opensmile


def zero_crossing_rate_numpy(y, frame_length=1024, hop_length=160):
    """Librosa's default mono ZCR semantics without its native Numba gufunc.

    Edge centering, threshold=1e-10, zero_pos=True, pad=False. Prefix sums
    count transitions within each frame in O(samples), without a framed copy.
    """
    y = np.asarray(y)
    if y.ndim != 1 or not y.size or not np.isfinite(y).all():
        raise ValueError('ZCR requires nonempty finite mono audio')
    if frame_length < 1 or hop_length < 1:
        raise ValueError('Frame and hop lengths must be positive')
    padded = np.pad(y, (frame_length // 2, frame_length // 2), mode='edge')
    negative = (padded < 0) & (np.abs(padded) > 1e-10)
    cumulative = np.r_[0, np.cumsum(negative[1:] != negative[:-1])]
    starts = np.arange(0, len(padded) - frame_length + 1, hop_length)
    return (cumulative[starts + frame_length - 1] - cumulative[starts]) / frame_length


@lru_cache(maxsize=1)
def smile_extractor():
    """One frontend per worker, reused across recordings."""
    return opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                           feature_level=opensmile.FeatureLevel.Functionals,
                           num_workers=1)


def praat(y, sr, settings, prefix='praat_full__'):
    sound = parselmouth.Sound(y, sampling_frequency=sr)
    floor, ceiling = settings['floor'], settings['ceiling']
    pitch = sound.to_pitch_ac(time_step=.01, pitch_floor=floor, pitch_ceiling=ceiling)
    f = pitch.selected_array['frequency']
    valid = f[f>0]
    result = {'f0_mean_hz':np.mean(valid) if len(valid) else np.nan,
              'f0_median_hz':np.median(valid) if len(valid) else np.nan,
              'f0_sd_hz':np.std(valid) if len(valid) else np.nan,
              'f0_min_hz':np.min(valid) if len(valid) else np.nan,
              'f0_max_hz':np.max(valid) if len(valid) else np.nan,
              'f0_range_hz':np.ptp(valid) if len(valid) else np.nan,
              'f0_cv':np.std(valid)/np.mean(valid) if len(valid) else np.nan,
              'pitch_valid_fraction':float(np.mean(f>0)), 'unvoiced_fraction':float(np.mean(f==0))}
    points = call(sound, 'To PointProcess (periodic, cc)', floor, ceiling)
    for name, method in [('jitter_local','local'), ('jitter_absolute_s','local, absolute'), ('jitter_rap','rap'), ('jitter_ppq5','ppq5'), ('jitter_ddp','ddp')]:
        try:
            result[name] = call(points, f'Get jitter ({method})', 0, 0, 1/ceiling, 1/floor, settings['max_period_factor'])
        except parselmouth.PraatError:
            result[name] = np.nan
    for name, method in [('shimmer_local','local'), ('shimmer_db','local_dB'), ('shimmer_apq3','apq3'), ('shimmer_apq5','apq5'), ('shimmer_apq11','apq11'), ('shimmer_dda','dda')]:
        try:
            result[name] = call([sound, points], f'Get shimmer ({method})', 0, 0, 1/ceiling, 1/floor, settings['max_period_factor'], settings['max_amplitude_factor'])
        except parselmouth.PraatError:
            result[name] = np.nan
    try:
        result['hnr_db'] = call(sound.to_harmonicity_cc(minimum_pitch=floor), 'Get mean', 0, 0)
        result['nhr_from_mean_hnr'] = 10**(-result['hnr_db']/10)
    except parselmouth.PraatError:
        result['hnr_db'] = result['nhr_from_mean_hnr'] = np.nan
    # CPP omitted: a validated, version-tested CPP implementation is not yet configured.
    return {prefix+k:float(v) for k,v in result.items()}, pitch


def longest_interval(mask, times, duration):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    starts, stops = np.where(edges==1)[0], np.where(edges==-1)[0]
    if not len(starts):
        return None
    i = np.argmax(stops-starts)
    a, b = float(times[starts[i]]), min(duration,float(times[stops[i]-1])+.01)
    return a,b


def extract(y, sr, cfg):
    features, pitch = praat(y, sr, cfg['pitch'])
    f = pitch.selected_array['frequency']
    interval = longest_interval(f>0, pitch.xs(), len(y)/sr)
    # Stable here means contiguous pitch tracking, not low disease-related variability.
    # No jitter or amplitude threshold selects the interval.
    features['interval_start_s'] = interval[0] if interval else np.nan
    features['interval_end_s'] = interval[1] if interval else np.nan
    features['pitch_detected_s'] = min(len(y)/sr,float(np.count_nonzero(f)*pitch.dx))
    statuses = {'praat_full':'ok','praat_stable':'insufficient_contiguous_voicing','opensmile':'pending','spectral':'pending'}
    if interval and interval[1]-interval[0]>=.5:
        seg = y[int(interval[0]*sr):int(interval[1]*sr)]
        stable,_ = praat(seg,sr,cfg['pitch'],prefix='praat_stable__')
        features.update(stable)
        statuses['praat_stable']='ok'
    smile = smile_extractor()
    try:
        eg = smile.process_signal(y,sr)
        if len(eg.columns)!=88:
            raise ValueError(f'Expected 88 eGeMAPSv02 functionals, got {len(eg.columns)}')
        features.update({'egemaps__'+k:float(v) for k,v in eg.iloc[0].items()})
        statuses['opensmile']='ok'
    except Exception as exc:
        statuses['opensmile']=f'{type(exc).__name__}: {exc}'
    try:
        spec = np.abs(librosa.stft(y,n_fft=1024,hop_length=160))
        arrays = {'centroid_hz':librosa.feature.spectral_centroid(S=spec,sr=sr)[0],
            'bandwidth_hz':librosa.feature.spectral_bandwidth(S=spec,sr=sr)[0],
            'rolloff_hz':librosa.feature.spectral_rolloff(S=spec,sr=sr)[0],
            'flatness':librosa.feature.spectral_flatness(S=spec)[0],
            'zcr':zero_crossing_rate_numpy(y,frame_length=1024,hop_length=160),
            'rms':librosa.feature.rms(S=spec,frame_length=1024)[0],
            'flux':np.sqrt(np.mean(np.diff(spec,axis=1)**2,axis=0))}
        # Spectral contrast requires all subbands below Nyquist.
        contrast = librosa.feature.spectral_contrast(S=spec,sr=sr,n_bands=5)
        arrays.update({f'contrast_{i}':v for i,v in enumerate(contrast)})
        mel = librosa.feature.melspectrogram(S=spec**2, sr=sr, n_mels=40)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel),sr=sr,n_mfcc=13)
        arrays.update({f'mfcc_{i+1}':v for i,v in enumerate(mfcc)})
        for k,v in arrays.items():
            features['spectral__'+k+'_mean']=float(np.mean(v))
            features['spectral__'+k+'_sd']=float(np.std(v))
        statuses['spectral']='ok'
    except Exception as exc:
        statuses['spectral']=f'{type(exc).__name__}: {exc}'
    return features, statuses, pitch


def sensitivity(y,sr,cfg):
    result={}
    for lo,hi in cfg['pitch']['sensitivity_ranges']:
        s={**cfg['pitch'],'floor':lo,'ceiling':hi}
        v,_=praat(y,sr,s,prefix='')
        result.update({f'pitch_{lo}_{hi}__'+k:v[k] for k in ('f0_mean_hz','pitch_valid_fraction','jitter_local','shimmer_local')})
    if len(y)>sr:
        v,_=praat(y[int(.25*sr):-int(.25*sr)],sr,cfg['pitch'],prefix='')
        result.update({'trim_025s__'+k:v[k] for k in ('f0_mean_hz','pitch_valid_fraction','jitter_local','shimmer_local')})
    return result


if __name__ == '__main__':
    from inspect_audio import main
    main(default_stage='features')
