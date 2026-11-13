# mPower: execution history, weighting plan, and frozen-pilot handoff

Consolidated November 12, 2026, America/New_York. Files timestamped
`20261113T03...Z` were created that evening in local time. This document records
completed work and its evidence; it does not assert that proposed weighting or
pretrained experiments have executed.

## Current result

The frozen-model code is implemented and tested with tiny random models,
including real-recording integration tests. Requested pretrained weights have
not loaded or been cached. The weighting protocol is written and its descriptive
audit ran, but calibrated technical weights, weighted Huber aggregation and
participant/day-weighted classification loss have not been implemented in the
benchmark runner or empirically fitted. A 56-participant pilot manifest is
snapshotted independently of EDA updates, and its real-audio data smoke passed.
No new Slurm job or pretrained baseline was launched in these steps.

## 1. Frozen-model implementation and verification

Implemented sources and configuration:

- `src/data/benchmark.py`: read-only canonical manifest integration, explicit
  labels, participant/content-duplicate split verification, technical filtering,
  segmentation and 16-kHz decoding.
- `src/models/frozen.py`: AST, Whisper encoder, wav2vec2 backbone, checkpoint
  frontends, frozen parameters, hidden-state extraction, reusable waveform and
  activation caches, mean/statistical pooling.
- `src/training/heads.py`: logistic and MLP heads, train-only scaling, optional
  learned attention pooling. Current head fitting uses ordinary unweighted loss.
- `src/evaluation/metrics.py`: recording/participant metrics and calibration.
- `scripts/run_frozen_benchmark.py`: unified CLI and execution reports.
- `scripts/compare_frozen_benchmarks.py`: completed-run comparisons, excluding
  smoke outputs by default.
- `scripts/frozen_benchmark.slurm`: separate CPU-safe batch wrapper.
- `configs/frozen_benchmark.yaml` and `requirements-benchmark.txt`.

The local `.benchmark_venv` reuses installed dependencies in `.mpower_venv` and
the existing `plm-benchmark` environment. It does not modify those environments.
A package installation attempt failed resolving pypi.org; environment reuse
allowed local tests to proceed.

| Executed check | Result | Evidence under `$MPOWER_BENCHMARK_ROOT` |
|---|---|---|
| Initial real-data smoke, default cohort B | Passed; 28 selected records, six real recordings decoded | `reports/execution_20261112T232803445913Z.json` |
| Online pretrained smoke, `--model all --smoke-test` | AST, Whisper and wav2vec2 all failed fetching frontend assets due to Hugging Face DNS failures; no pretrained extraction/head training | `reports/execution_20261112T233048418302Z.json`, `pretrained_smoke.log` |
| Initial synthetic/frontend tests | First 12, then 13 passed as coverage expanded | Subsequent complete test logs supersede these intermediate runs |
| Expanded benchmark tests | A cached-rerun attention determinism test initially failed; seed reset after extraction fixed it | Explanation in `docs/FROZEN_BENCHMARK_STATUS.md` |
| Benchmark suite after the seed correction | 16 passed | `unit_tests.log` |
| Real-recording architecture checks | Three passed: AST mean, Whisper statistical, wav2vec2 attention; each used six real records and both classifier heads with tiny random backbones | `real_random_architecture_tests.log`; outputs in `random_architecture_checks/` |
| Full regression suite | 28 passed, three warnings, 285.53 s; includes nine existing EDA tests | `validation.log` |
| Final targeted verification after last benchmark code changes | 19 passed, three warnings, 56.74 s | `final_benchmark_validation.log` |
| Comparison aggregation | Executed successfully on explicitly labeled random-model smoke outputs | `comparison_check.log`, `random_architecture_checks/comparison.csv` |
| CLI help, Python compilation, Slurm shell syntax and whitespace checks | Passed | Commands listed below |

The warnings concern the AST frontend's Mel filter bank. Tiny random encoder
checks validate code and tensor processing, not requested checkpoint weights.
Random-model checkpoints are explicitly named `test-only/random-*` in provenance.
Smoke metrics are not validated predictive performance.

Cache-related corrections verified by tests include file-content invalidation,
resumable segment activations, train-only transformations, and training reseeding
after extraction. Some wav2vec2 eval code consumes RNG draws; head initialization
must not depend on whether activations came from a cache.

## 2. Quality/medication weighting work already completed

Read the existing data layout, canonical manifests, cohort/split logic, QC
implementation and medication linkage. Created and executed
`scripts/audit_weighting_readiness.py`. This is a read-only aggregate audit with
bounded CPU settings; it neither decodes the full dataset nor changes EDA jobs.

Two preserved audit snapshots:

- `planning/weighting_audit_20261113T032955Z.json`: initial metadata and QC review.
- `planning/weighting_audit_20261113T033321Z.json`: expanded audit with temporal
  windows, overlapping-age counts and synthetic energy-proxy checks.

Aggregate checks passed: diagnosis counts sum to source row/participant counts,
medication cells sum to all records, checkpoint status counts reconcile, and
none of the audited parquet inputs changed during the reads. The audit script
passed Python compilation.

Important findings:

| Finding | Consequence for the proposed method |
|---|---|
| 65,022 recordings; 5,826 participants | Preserve participant identity and split disjointness. |
| 970 PD, 3,890 control, 864 unknown and 102 conflicting participants | Primary binary training uses explicit PD/control labels; other labels remain audited. |
| Most active 1% supplied 30.4% of all recordings | Equalize participant contributions. |
| PD supplies about 63.4% of binary-labeled recordings but 20.0% of binary-labeled participants | Recording-level training silently changes class influence. |
| Median age PD 61 versus controls 29 | Audit demographic shortcuts and common support. |
| PD median follow-up 10.36 days versus control approximately 2.34 minutes | Separate first-assessment screening from longitudinal phenotype aggregation. |
| 422 PD participants reported both pre/post states; 243 had both on at least one UTC date | Candidate paired analysis; pharmacologic ON/OFF and local-day timing remain unverified. |
| Recording-specific device/app fields complete: 11 phoneInfo and three appVersion values | Device nuisance audits are feasible with validated source linkage. |
| Pilot: 120 successfully extracted records from 120 people | No within-person repeatability calibration from this pilot alone. |
| Full checkpoint: 6,016 attempts, 5,991 worker errors, 16 other errors, nine successes | Recover processing before fitting empirical quality weights. No successful PD entries in that snapshot. |
| Legacy `estimated_snr_db` flags 79/120 pilot recordings below 10 dB | Do not use this unvalidated threshold as a quality exclusion or weight. |

The last item was checked against `audio_quality.py`: the field equals
`20 log10(P90(frame RMS)/P10(frame RMS))`, a frame-energy dynamic-range proxy.
An independent synthetic calculation gives approximately 0 dB for a clean
steady sine and 0.73 dB for stationary noise; both trigger the 10 dB flag.
This demonstrates a limitation of that proxy, not measured noise levels in
participant recordings.

The expanded audit also records post-survey window support:

| Window | PD participants / recordings | Control participants / recordings |
|---|---:|---:|
| 0–1 days | 827 / 2,304 | 3,192 / 8,093 |
| 0–7 days | 915 / 7,832 | 3,529 / 13,660 |
| 0–30 days | 936 / 19,174 | 3,596 / 17,892 |

These are metadata eligibility counts before completed full-audio QC. They do
not establish usable audio counts. The existing 50–80 age sensitivity range
contains 805 PD and 466 control participants before additional filtering.

## 3. Weighting specification to carry forward

The full scientific plan is
[PD_QUALITY_MEDICATION_WEIGHTING_PLAN.md](PD_QUALITY_MEDICATION_WEIGHTING_PLAN.md).
Its literature references and assumptions remain part of this handoff.

Keep these components separate:

1. **Technical validity and reliability.** Feature-specific validity masks and
   training-calibrated, bounded reliability weights. Start with uniform weights
   until calibration is defensible. Low intensity, irregular pitch, jitter,
   shimmer and HNR are potential biology, not automatic recording-quality penalties.
2. **Participant/session/day balance.** Stop prolific contributors and overlapping
   segments from dominating summaries or training loss.
3. **Medication state.** Retain reported pre-dose, post-dose, another time,
   nonuse and missing separately. No arbitrary medication weight and no use of
   medication text as a predictor in the primary voice-only PD classifier.
4. **Demographic/device adjustment.** Define separate nuisance baselines,
   common-support comparisons and adjusted analyses; do not fold all covariates
   into a composite “quality” score.

Within participant/session/state, the ordinary feature-specific weighted mean is:

`mu = sum_r(m_r * q_r * x_r) / sum_r(m_r * q_r)`.

The robust scalar-feature proposal is:

`mu_hat = argmin_mu sum_r(m_r * q_r * rho_c((x_r - mu) / s_k))`.

Huber tuning comparisons start at 1.345, with explicit scale floors and sparse-
repeat handling. Retain dispersion/tails alongside robust location. For frozen
embeddings, a quality-weighted arithmetic mean is the initial proposal.

The proposed participant/day-balanced classification loss uses:

`a_idr = (1 / D_i) * q_idr / sum_j_in_day(q_idj)`

`L = (1 / N_participants) * sum_i sum_d,r a_idr * BCE(y_i, p_idr)`.

Segment weights sum to their recording's weight. Class balancing is a separate
optional choice. Test metrics remain ordinary participant-level metrics with
coverage and calibration reporting. Unknown/conflicting labels are not repaired
by assigning small quality weights.

If “kappa mean” means kappa-sigma clipping, compare it only among repeated
same-state measurements. Cohen kappa measures reviewer agreement, not a feature
mean or loss weight.

**Implementation status:** the above weighting formulas are a proposed next
stage. The current benchmark implements ordinary mean/statistical/attention
pooling, train-only scaling, unweighted logistic/MLP fitting and unweighted
mean/median participant probability aggregation. The prepared pilot YAML does
not activate the proposed weighted objective. Applying it requires explicit
implementation and validation; calibrated q values are currently unavailable.

## 4. Latest Slurm inspection and impact assessment

Commands attempted: `squeue -u wangr7`, `scontrol show job 27498642`, and DNS
resolution of `respublica-01` and `huggingface.co`. Both Slurm queries failed
because the controller hostname could not resolve. No live node allocation,
GPU availability, complete job list, or scheduler state was verified.

The latest identifiable job is **27498642**, associated with
`scripts/full_mpower.slurm`. Its submission script requests 16 CPUs, 48 GB RAM,
24 hours and no GPU. These are script requests, not independently verified live
allocations. Its error log was actively updating around 22:47 local time.

Evidence:

- `logs/mpower_full_27498642.out`: last progress line 6,016/65,022, decoded nine.
- `logs/mpower_full_27498642.err`: repeated fatal segmentation faults through
  `numba/np/ufunc/gufunc.py` → `librosa.core.audio.zero_crossings` →
  `librosa.feature.zero_crossing_rate` → `extract_acoustic_features.py`.
- `processed/full_features_checkpoint.parquet`: the processing-failure counts
  described above.

The frozen runner does not use this handcrafted zero-crossing feature-extraction
path and can operate independently. Shared disk traffic and cluster capacity
can still cause contention. A separate modest allocation, one model at a time,
and independent output storage limit interference. EDA may later rewrite cohort
manifests; use the prepared snapshot below to keep experiment membership fixed.
The existing EDA job was not stopped, restarted or modified.

## 5. Frozen-pilot snapshot and latest data check

Prepared private directory:

`$MPOWER_BENCHMARK_ROOT/planning/frozen_pilot_20261113T034808Z/`

Files: `manifest.parquet`, `config.yaml`, `readiness.json`.

Uses cohort A to avoid inheriting cohort B's exploratory pitch-voicing exclusion
as the only frozen-model pilot population. Both retain provisional canonical
participant assignments. Cohort A has 56 participants: 35 PD, 21 controls.

| Split | PD | Controls | Total |
|---|---:|---:|---:|
| Train | 25 | 15 | 40 |
| Validation | 5 | 3 | 8 |
| Test | 5 | 3 | 8 |

The private snapshot was reloaded and checked for semantic dataframe equality,
label integrity and split disjointness. Initial digest comparisons exposed
CSV float/null round-trip differences and parquet null-representation differences;
these were snapshot serialization checks, not audio/model failures. The final
snapshot uses parquet, passed the dataframe check, and records the reloaded
manifest digest:

`7bb068bfdc30a1d7bbd8fafe36e8e868487437897e599bde5726d35630ae4d36`.

Executed `--data-smoke` against this exact config: passed, 28 records selected,
six real recordings decoded across all split/class combinations. Evidence:
`reports/execution_20261113T034927039844Z.json`.

This supports a small exploratory frozen-backbone pilot. Eight test participants
cannot support precise predictive-performance conclusions. No requested
pretrained weights are present; DNS/download and scheduler access remain
launch blockers. Quality calibration and the full acoustic EDA need not be
finished to exercise a frozen pilot, but a larger scientifically validated run
requires adequate data/QC and a locked evaluation design.

## 6. Reproducible command index

From the repository root, configure approved, distinct storage directories:

```bash
export MPOWER_DATA_ROOT=/approved/storage/mpower
export MPOWER_OUTPUT_ROOT=/approved/storage/mpower_phase1
export MPOWER_BENCHMARK_ROOT=/approved/storage/mpower_benchmarks
```

Commands executed during this work, with these environment variables pointed
at this project's existing private storage:

```bash
.benchmark_venv/bin/python -m pytest -q tests/test_frozen_benchmark.py
MPOWER_RUN_REAL_SMOKE=1 .benchmark_venv/bin/python -m pytest -q tests/test_frozen_real_smoke.py
MPOWER_RUN_REAL_SMOKE=1 .benchmark_venv/bin/python -m pytest -q tests
MPOWER_RUN_REAL_SMOKE=1 .benchmark_venv/bin/python -m pytest -q tests/test_frozen_benchmark.py tests/test_frozen_real_smoke.py
.benchmark_venv/bin/python scripts/run_frozen_benchmark.py --model all --smoke-test
.benchmark_venv/bin/python scripts/run_frozen_benchmark.py --data-smoke
.benchmark_venv/bin/python scripts/audit_weighting_readiness.py
.benchmark_venv/bin/python scripts/run_frozen_benchmark.py --help
.benchmark_venv/bin/python -m compileall -q src scripts/run_frozen_benchmark.py scripts/compare_frozen_benchmarks.py
.benchmark_venv/bin/python -m py_compile scripts/audit_weighting_readiness.py
bash -n scripts/frozen_benchmark.slurm
git diff --check
```

Tests used bounded CPU/thread settings; opt-in real checks wrote private
artifacts outside Git. Comparison aggregation was executed with
`--include-smoke` and the benchmark root set to the separate
`random_architecture_checks` directory. The latest snapshot data smoke used:

```bash
.mpower_venv/bin/python scripts/run_frozen_benchmark.py \
  --config "$MPOWER_BENCHMARK_ROOT/planning/frozen_pilot_20261113T034808Z/config.yaml" \
  --data-smoke
```

Next pretrained commands, **prepared but not successfully executed on this
snapshot**, once model access is available:

```bash
.benchmark_venv/bin/python scripts/run_frozen_benchmark.py \
  --model all \
  --config "$MPOWER_BENCHMARK_ROOT/planning/frozen_pilot_20261113T034808Z/config.yaml" \
  --prepare-only
.benchmark_venv/bin/python scripts/run_frozen_benchmark.py \
  --model all \
  --config "$MPOWER_BENCHMARK_ROOT/planning/frozen_pilot_20261113T034808Z/config.yaml" \
  --smoke-test
```

After all requested pretrained smoke checks pass, the same command without
`--smoke-test` runs ordinary pilot heads. Slurm submission requires a reachable
controller and a complete Python environment, with
`MPOWER_BENCHMARK_PYTHON` and `MPOWER_BENCHMARK_CONFIG` set appropriately.
Proposed reliability/Huber/weighted-loss comparisons are an additional
implementation stage, not an implicit option in these commands.

## 7. Newly requested immediate frozen run

After the request to run the frozen models now, attempted Slurm submission with
`sbatch --parsable scripts/frozen_benchmark.slurm --smoke-test`, exporting the
prepared snapshot config, storage roots and complete benchmark Python path.
The controller hostname remained unresolved. The submission client repeatedly
retried and was interrupted with exit status 130; no job ID or acceptance was
returned. This stopped the new submission client, not the existing EDA job.

A fresh DNS probe failed for both `respublica-01` and `huggingface.co`.
Then executed the actual frozen runner against the prepared snapshot for all
three models, using `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` to check available
local weights without repeating known-failing downloads. The runner attempted
AST, Whisper and wav2vec2; each failed because pretrained assets were absent.
It exited with status 1 and wrote:

`reports/execution_20261113T040455001124Z.json`.

No pretrained forward pass, embedding extraction, head training or baseline
experiment completed in this attempt. The previously successful tests remain
random-backbone architecture/integration checks. This is an execution blocker,
not a requirement to wait for quality-weight calibration.

New logs: `frozen_submit_attempt.log`, `frozen_connectivity_check.log`,
`frozen_local_attempt.log`. Consolidated copies of plans, audit JSON, execution
reports and test/attempt logs are preserved in this private bundle:

/mnt/isilon/wang_lab/wangr7/projects/pd_data/mpower_benchmarks/handoff/consolidated_20261113T040616Z
