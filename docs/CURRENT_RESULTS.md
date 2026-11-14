# Current results and execution limits

Snapshot: November 13, 2026. Historical test evidence is from November 12
local time / November 13 UTC; the HPC cleanup audit is newer. These are aggregate
execution/QC findings, not a claim of validated
clinical predictive performance. Private evidence remains in `pd_data`.

## Verification

| Check | Result |
|---|---|
| Full regression suite | 28 passed, 3 warnings, 285.53 seconds |
| Final targeted benchmark/integration checks | 19 passed, 3 warnings, 56.74 seconds |
| Real-recording tiny-random-encoder integration | All three architectures passed with six recordings each and both heads |
| Original cohort-B data-only smoke | Passed; 28 selected, six real recordings decoded |
| Fixed cohort-A snapshot data-only smoke | Passed; 28 selected, six real recordings decoded |
| Requested pretrained AST, Whisper-small and wav2vec2-base-960h | None loaded; downloads blocked and no cached requested weights |
| Latest requested Slurm submission | Controller DNS failed; retrying client interrupted; no job ID returned |
| Pretrained baseline experiments | None completed |

The 19 targeted tests overlap the 28-test suite; these are not 47 independent
tests. AST frontend warnings concern Mel filters. The integration tests use
explicitly named `test-only/random-*` backbones. Their metrics must not be cited
as pretrained-model performance.

## Data and pilot readiness

The source has 65,022 recordings from 5,826 participants. Primary participant
labels: 970 PD, 3,890 controls, 864 unknown and 102 conflicting. Diagnosis is
participant-reported professional diagnosis with a conservative contradiction
policy, not independent clinical adjudication.

The fixed cohort-A pilot uses one recording per participant:

| Split | PD | Controls | Total |
|---|---:|---:|---:|
| Train | 25 | 15 | 40 |
| Validation | 5 | 3 | 8 |
| Test | 5 | 3 | 8 |

The pilot is sufficient for an exploratory execution run once weights are
available. Eight held-out participants cannot support precise predictive
conclusions. Existing splits remain provisional pending full duplicate/QC review.

The November 13 full acoustic checkpoint contained 31,040 attempted rows:
30,990 worker errors, 41 other errors and nine successes. Error logs implicated
Librosa/Numba zero-crossing extraction. These are processing failures and do not
establish the quality of participant audio. This code path is not used by the
frozen encoder runner. Outside-sandbox Slurm queries on November 13 verified that
job 27498642 timed out after 24 hours; no mPower jobs were active. See the
[reorganization audit](REORGANIZATION_AUDIT.md).

## Weighting findings and status

- The most active 1% supplied 30.4% of recordings. Equal participant contributions
  are essential to the proposed training design.
- Median age was 61 in PD and 29 in controls; demographic shortcuts need auditing.
- 422 PD participants reported both pre/post medication recordings; 243 did so
  on at least one UTC date. These are candidate comparisons, not verified
  pharmacologic ON/OFF pairs.
- The legacy `estimated_snr_db` is a frame-energy percentile ratio, not true SNR.
  Its 10-dB flag affected 79/120 pilot recordings; a synthetic clean steady tone
  also triggers it. It is not a validated exclusion or weighting rule.
- Proposed work separates technical reliability, participant/day balance,
  medication state and demographic/device adjustment. Weighted Huber summaries
  and participant/day-weighted loss are documented, not implemented or fitted.

For detailed assumptions and evidence, see the [weighting protocol](PD_QUALITY_MEDICATION_WEIGHTING_PLAN.md),
[complete run history](MPOWER_RUN_HISTORY_AND_WEIGHTING.md),
[frozen benchmark setup](FROZEN_BENCHMARK_README.md), and
[pd_data context](PD_DATA_CONTEXT.md).
