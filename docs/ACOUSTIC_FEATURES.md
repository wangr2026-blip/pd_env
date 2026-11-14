# Extracted acoustic measurements

The extractor uses mono 16-kHz decoded audio and emits 184 core feature columns,
plus interval information, processing status and optional pilot sensitivity
measurements. Measurements are recording-level. Undefined values remain missing;
these columns are not clinically validated diagnostic thresholds.

| Prefix | Count | Contents |
|---|---:|---|
| `praat_full__` | 22 | Whole-recording Praat measurements |
| `praat_stable__` | 22 | Same measurements on longest contiguous pitch-tracked interval >=0.5 s |
| `egemaps__` | 88 | Complete openSMILE eGeMAPSv02 Functionals |
| `spectral__` | 52 | Mean/SD of 26 spectral, energy and cepstral trajectories |

Each Praat set contains six F0 summaries (mean, median, SD, min, max, range),
F0 coefficient of variation, pitch-valid/unvoiced fractions, five jitter
measurements (local, absolute seconds, RAP, PPQ5, DDP), six shimmer measurements
(local, dB, APQ3, APQ5, APQ11, DDA), mean HNR in dB and the dimensionless ratio
`10**(-mean_hnr_db/10)`. Fractional perturbation values are not percentages.
The derived ratio is not an average of per-frame linear NHR values.
Praat uses autocorrelation pitch at 10-ms steps, 60–500 Hz, periodic
cross-correlation point processes, maximum period factor 1.3 and amplitude
factor 1.6. The selected interval does not certify physiological stability.
Interval start/end and total pitch-detected seconds are saved separately.

The 88 openSMILE columns include F0 and loudness distributions/slopes, spectral
flux, MFCCs 1–4, jitter, shimmer, HNR, H1–H2/H1–A3 harmonic differences,
F1–F3 frequencies/bandwidths/relative amplitudes, voiced/unvoiced spectral
balance/slopes, segment rates/durations and equivalent sound level. Native
standard column names are preserved. These include formant descriptors; a
separate Praat formant or CPP analysis is not implemented.

The spectral trajectories are centroid, bandwidth, rolloff, flatness,
zero-crossing rate, RMS, raw magnitude spectral flux, contrast indices 0–5,
and 13 MFCCs. Each exports mean and population SD. The STFT has 1,024 samples
and 160-sample hop (64/10 ms); MFCCs use 40 mel bands. `mfcc_1` through
`mfcc_13` number the returned coefficients starting at one. Contrast values
are in dB. RMS/flux depend on recording amplitude. ZCR uses a NumPy prefix-sum
implementation matching Librosa's edge centering, 1e-10 threshold,
positive-zero convention and no artificial crossing at frame starts.

Pilot-only sensitivity measurements repeat F0 mean, pitch-valid fraction,
local jitter and local shimmer for 50–400 Hz, 75–600 Hz, and removal of 250 ms
at each edge (12 additional values). The full run retains all 184 core columns
but omits these extra pilot checks. QC tables separately retain duration,
amplitude, energy/silence proxies, clipping, original codec/rate and hashes.

Implementation: `scripts/extract_acoustic_features.py`; settings:
`configs/mpower_preprocessing.yaml`; private per-field dictionary:
`$MPOWER_OUTPUT_ROOT/processed/feature_dictionary.csv`.
