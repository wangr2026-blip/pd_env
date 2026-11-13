# Reproducible Phase 1 execution

Read `AGENTS.md`, `data_layout.md` and `configs/mpower_preprocessing.yaml` first. No raw files are changed and no additional datasets are downloaded. Participant-bearing manifests, schemas, caches, figures and executed notebooks go to approved storage outside Git. Source notebook outputs are clear.

Use Python 3.12 (tested), not the current Python 3.15 `pd_env` interpreter. Create a separate compatible environment and install `requirements-preprocessing.txt`. Direct dependencies are pinned; the Slurm job records the complete installed environment and FFmpeg version. A bundled imageio FFmpeg executable is a fallback when system FFmpeg is absent. A pre-staged environment archive may be unpacked to compute-node temporary storage with `MPOWER_ENV_ARCHIVE`; it must be compatible with compute-node OS and the base interpreter must be accessible.

```bash
export MPOWER_DATA_ROOT=/approved/raw/root
export MPOWER_OUTPUT_ROOT=/approved/derived/mpower_phase1
export MPOWER_PYTHON=/approved/environment/bin/python
mkdir -p logs
sbatch --export=ALL scripts/preprocess_mpower.slurm
```

All large inventories, real-audio pilot extraction, reporting and full runs must execute through Slurm. The default is a 120-recording pilot, 2 CPUs, 12 GB RAM, 12-hour limit. Check pilot extraction statuses and sensitivity analyses before submitting `MPOWER_SCOPE=full`; full extraction must not be launched locally. The batch pipeline inventories all raw tables, audits/canonicalizes metadata, extracts audio/QC/features, builds cohorts, attempts splits, creates tables/figures/report and executes the notebook to private storage. The scheduler resource request may need adjustment after pilot timing. Do not automatically scale an unreviewed pilot.

```bash
# After pilot review only:
export MPOWER_SCOPE=full
# Reuse an audited unchanged metadata census only after verifying inputs:
export MPOWER_SKIP_AUDIT=1
sbatch --export=ALL scripts/preprocess_mpower.slurm
```

Intermediate outputs are atomic. Each recording has a cache key including audio path, size, modification time, config hash, extraction/QC/common-code hashes and library versions. Successful results are reused; errors/partial extraction retry. Raw-file stat keys are not a substitute for cryptographic raw integrity checks; file hashes are also saved. Work submission and decoded audio memory are bounded; pilot checkpoints update every worker batch, while full checkpoints update initially, approximately every 1,000 results and at completion. Successful per-recording caches are atomic between checkpoints. Spawned workers recycle after 250 tasks. If a native worker exits abruptly, successful results are retained and unresolved batch rows retry in isolated processes; repeat native failures receive explicit worker-error status. Errors do not abort the recording loop, though dependency/configuration errors should be fixed before rerunning. Configured maximum duration is 180 s: longer files are flagged for review rather than silently truncated. Source recordings remain unchanged.

Praat uses autocorrelation pitch 60–500 Hz, 10-ms steps, periodic-cc point processes, period factor 1.3 and amplitude factor 1.6. Local/absolute/RAP/PPQ5/DDP jitter, local/dB/APQ3/APQ5/APQ11/DDA shimmer, F0 summaries and HNR are extracted. NHR derived from mean HNR is explicitly named and is not the mean linear NHR. Perturbation values are fractions, except absolute jitter (seconds) and shimmer dB. Full-recording and longest contiguous pitched interval >=0.5 s measurements are separated. The interval selection does not minimize jitter or shimmer and does not certify stationarity. Undefined values remain missing. Pilot repeats pitch ranges 50–400 / 75–600 Hz and removes 250 ms from both edges for sensitivity.

openSMILE eGeMAPSv02 provides 88 full-recording functionals with original names. librosa provides STFT 1024-point / 160-sample hop features: centroid, bandwidth, rolloff, flatness, five-band contrast, RMS, ZCR, raw FFT spectral-flux summaries and 13 MFCC mean/SD values from a 40-band log-mel power spectrogram. Flux is not amplitude-normalized. Formants, CPP, LLD exports, DDK and language features are not reported. No transformer embeddings or model training occur.

Participant-level median aggregation avoids counting repeated recordings as independent in optional exploratory rank comparisons; rank-biserial effects and BH FDR are implemented. After authorized demographics linkage, `demographic_eda.py` computes baseline-age/reported-gender distributions, SMD, PD-stratified duration/QC/frequency summaries, other available covariates, and participant-median OLS with HC3 errors adjusted for age/gender and additionally median recording quality. BH FDR is applied separately to the two adjustment specifications. These are exploratory pilot models, not clinical validation. Cohort C matches equal participant counts within 5-year age bins and gender. Baseline demographic covariates are not time-updated clinical measurements. Separate descriptive recording-level correlations must not be interpreted as participant-independent inference.

Validation: `python -m pytest -q tests` uses synthetic metadata, periodic signals and silence to check source-row retention, missing-label handling, longitudinal label conflicts, join/config integrity, corrupt-file error recording, expected F0, all 88 eGeMAPS features, undefined perturbation values, deterministic extraction and duplicate-linked participant split isolation. No synthetic participant records are presented as observed data.

Algorithm references: [Praat jitter definitions](https://www.fon.hum.uva.nl/praat/manual/Voice_2__Jitter.html), [Praat automated voice analysis](https://www.fon.hum.uva.nl/praat/manual/Voice_6__Automating_voice_analysis_with_a_script.html), [Parselmouth API](https://parselmouth.readthedocs.io/en/stable/api_reference.html), [openSMILE usage](https://audeering.github.io/opensmile-python/usage.html), [librosa documentation](https://librosa.org/doc/0.10.2/).

The generated final report includes an executed-scope manuscript Methods draft and answers each requested analysis category with computed results or an explicit blocker. It must be revised after metadata linkage and full processing; it is not a completed PD biomarker manuscript.

The authorized demographics-only downloader and its Slurm wrapper are `scripts/download_mpower_demographics.py` and `.slurm`. They use the existing credential file without logging it, export only syn5511429, download no audio, and refuse to overwrite an existing snapshot. `scripts/linked_pilot_mpower.slurm` preserves the original voice-only pilot, reuses the audited voice census, links demographics, runs a diagnosis-stratified pilot and regenerates reports. The download does not authorize downloading UPDRS/PDQ-8 surveys or any other table.

## Executed jobs in this session

- `27497408`: voice-only census and 120-recording pilot, completed; archived under `archive/voice_only_pilot/`.
- `27497846`: authorized metadata-only demographics export, completed.
- `27497956`: audited demographic linkage and diagnosis-stratified 120-recording pilot, completed; archived before full processing under `archive/linked_pilot/`.
- `27498284`: initial full run failed when a native worker exited abruptly; no full result is claimed. Slurm peak RSS was about 2.1 GB, without an OOM state.
- `27498642`: full run resubmitted with spawned workers, periodic worker recycling, isolated retries and native crash tracing; 16 CPUs / 48 GB / 24-hour limit. Submission is not evidence of successful full completion. Check scheduler state and job logs before citing full results.

Observed linked census: 970 participants report professional PD diagnosis, 3,992 report no diagnosis and 864 are unknown (including unmatched surveys). A conservative contradiction audit quarantines 102 of the self-reported controls, leaving 3,890 primary controls. Source labels remain intact. Primary PD/control age mean difference is 27.64 years and standardized difference is 2.28, so unadjusted disease comparisons are strongly confounded.

The linked pilot decoded/extracted 120/120 recordings; stable-interval jitter was defined for 117/120. Five repeat extractions reproduced. Pilot cohorts A/B/C contain 56/55/12 participants respectively, each with one selected recording; these are coverage-sample counts, not full-cohort yields. The pilot A partitions have 40 train, 8 validation and 8 test participants/recordings. No pilot adjusted acoustic comparison passed BH FDR <0.05 among 368 feature/model comparisons; this is not evidence of equivalence or absence of a disease signal. The full job must generate its own denominators and inferential summaries.

Final synthetic verification: 9 tests pass, including deliberately terminated native-worker containment followed by successful task execution. This does not establish that every real recording is decodable; use the per-recording full-run statuses.
