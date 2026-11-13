# QC execution and interpretation

This document preserves the initial voice-only QC interpretation. Subsequent
demographic linkage provides labels; see [current results](CURRENT_RESULTS.md)
and the [weighting audit](PD_QUALITY_MEDICATION_WEIGHTING_PLAN.md) for the latest
counts, processing failures and limitations. References below to unknown labels
describe the earlier phase, not the current canonical manifest.

The generated evidence report is `$MPOWER_OUTPUT_ROOT/eda/MPOWER_FINAL_REPORT.md`; actual distributions, threshold sensitivities, feature validity, and sequential exclusions are written alongside it. Audio outputs carry `pilot` or `full` scope. Do not cite synthetic test measurements as participant results.

The pilot selects 120 recordings with fixed seed, one recording per participant first, using round-robin sampling over available diagnosis and medication strata. Device stratification is not implemented. It is a coverage sample, not a random prevalence estimate. All diagnosis values are currently unknown.

QC computes decoded mono 16-kHz duration/sample count, peak, RMS, DC offset, frame-RMS dynamic range, clipping fraction, low-energy fraction, energy activity, silence proxies and spectral flatness. Original container/codec/rate/channel/bitrate fields come from ffprobe when present, otherwise FFmpeg headers; bit depth can be unavailable. Native decoded sample counts are not presently exported; the analysis sample count is after resampling. Compressed bit depth is generally not meaningful.

Energy activity uses 30-ms frames / 10-ms hops and threshold max(0.0001, peak frame RMS minus 35 dB). It is not a speech/phonation classifier. The SNR proxy is 20 log10(frame-RMS p90/p10), with p10 as an estimated noise floor. This can be misleading for steady vowels and is not a genuine noise-reference SNR. Pitch-detected duration is a separate Praat estimate. Segmentation does not certify task compliance.

Flags: near silence RMS <0.0001; low amplitude RMS <0.001; total duration <2 s; energy activity <2 s; samples >=0.999 exceeding 1%; estimated SNR <10 dB. Flags overlap. Low SNR, clipping, low amplitude and unstable pitch require review and remain available to sensitivity analyses. No composite quality score is imposed. No disease-specific quality conclusion can be reached without diagnosis labels.

Representative figures select the lowest deterministic recording hash among decoded examples in each observed category. Waveforms, spectrograms and pitch contours use secondary anonymized IDs. Lookup maps stay in private output storage. No audio is redistributed.

Exact SHA256 hashes cover source bytes, native-rate/channel float PCM and analysis-normalized mono 16-kHz float PCM. Native hash identity requires matching rate and channel layout. Lossy recoding and near copies may evade exact hashes. Exhaustive near-duplicate detection, task-contamination classification, and manual review are pending and explicitly block final release.
