"""Transparent descriptive QC: energy activity is not a clinical voice detector."""
import numpy as np
import librosa


def quality(y, sr, cfg):
    q = cfg['qc']
    n = max(32, int(sr*q['frame_ms']/1000))
    hop = max(1, int(sr*q['hop_ms']/1000))
    frames = librosa.util.frame(np.pad(y, (0, max(0,n-len(y)))), frame_length=n, hop_length=hop)
    rms = np.sqrt(np.mean(frames.astype(float)**2, axis=0))
    threshold = max(q['near_silence_rms'], float(rms.max())*10**(-q['energy_db_below_peak']/20))
    active = rms > threshold
    # Frame centers yield bounded time estimates; not a validated VAD or SNR measurement.
    active_times = np.flatnonzero(active)*hop/sr
    activity_s = min(len(y)/sr, float(active.sum()*hop/sr))
    peak = float(np.max(np.abs(y)))
    overall = float(np.sqrt(np.mean(y.astype(float)**2)))
    noise = float(np.quantile(rms,.1))
    signal = float(np.quantile(rms,.9))
    snr = float(20*np.log10(max(signal,1e-12)/max(noise,1e-12)))
    clipped = float(np.mean(np.abs(y)>=q['clipping_amplitude']))
    flat = librosa.feature.spectral_flatness(y=y, n_fft=1024, hop_length=hop)[0]
    result = {'duration_s': len(y)/sr, 'analysis_samples':len(y), 'analysis_sample_rate':sr,
        'peak_amplitude':peak, 'rms_amplitude':overall, 'dc_offset':float(y.mean()),
        'frame_dynamic_range_db':snr, 'estimated_snr_db':snr,
        'noise_floor_rms_p10':noise, 'energy_activity_s':activity_s,
        'silence_fraction':float(1-active.mean()), 'low_energy_fraction':float(np.mean(rms<q['low_amplitude_rms'])),
        'leading_silence_s':float(active_times[0]) if len(active_times) else len(y)/sr,
        'trailing_silence_s': max(0., len(y)/sr-float(active_times[-1])-n/sr) if len(active_times) else len(y)/sr,
        'clipping_fraction':clipped, 'spectral_flatness_mean':float(np.mean(flat)),
        'flag_near_silent':overall<q['near_silence_rms'], 'flag_low_amplitude':overall<q['low_amplitude_rms'],
        'flag_short':len(y)/sr<q['min_duration_s'], 'flag_short_activity':activity_s<q['min_voiced_s'],
        'flag_clipping':clipped>q['clipping_fraction'], 'flag_low_estimated_snr':snr<q['estimated_snr_db']}
    return result
