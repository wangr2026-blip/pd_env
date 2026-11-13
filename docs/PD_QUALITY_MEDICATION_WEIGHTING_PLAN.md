# mPower voice: quality, medication, aggregation, and weighted-loss analysis plan

Status: proposed protocol grounded in a read-only data audit, November 12, 2026
(local time). The audit snapshot is November 13 UTC. No empirical weighting
coefficients, clinical thresholds, medication effects, or predictive results have
been validated. Existing EDA files and jobs were not changed.

## 1. Scientific objective and recommendation

Use **participant-balanced, technically reliable aggregation**, with medication
state explicitly retained. A single score multiplying quality, age, medication,
and disease severity would mix measurement error with genuine biology and
change the population being studied.

The proposed default is:

1. Resolve technical processing failures independently of recording quality.
2. Apply narrowly defined technical eligibility checks and feature-specific
   validity masks; retain disease-related dysphonia.
3. Build a calibrated, bounded technical-reliability weight from training data.
4. Aggregate recordings within participant, session/day, and medication state;
   use a weighted Huber location for scalar acoustic features and a weighted
   arithmetic mean for frozen-model embeddings.
5. Give each participant equal total contribution to the training loss. Keep
   medication response and demographic adjustment as separate analyses.
6. Preserve ordinary participant-level evaluation and report coverage, uncertainty,
   and subgroup performance alongside any weighting gains.

Until quality calibration is available, use **uniform technical weights** with
participant/day balancing. Do not invent coefficients such as “medication ON =
0.5,” “SNR importance = 40%,” or “jitter abnormality = poor quality.”

If “kappa mean” means a kappa-sigma-clipped mean, treat it as an outlier-robust
comparison confined to repeat measurements in the same participant/state.
Cohen's kappa instead measures reviewer agreement; it is not a mean or a
recording-reliability weight. Huber estimation supplies the smooth downweighting
that is useful here, without hard deletion of every extreme repeat.

## 2. What the current data establish

The audit reads the canonical manifests, existing pilot/full feature checkpoints,
and the voice-table device-field header/coverage. It performs no full audio scan,
no new download, and no model fitting. Aggregates and input timestamps are saved
outside Git under `$MPOWER_BENCHMARK_ROOT/planning/weighting_audit_*.json`.

| Observation at the audit snapshot | Consequence |
|---|---|
| 65,022 recordings from 5,826 participants | Recording count is not independent sample size. |
| Primary labels: 970 PD and 3,890 controls; 864 unknown and 102 conflicting participants | Keep unknown/conflicting labels outside primary supervised training; retain source labels for sensitivity analysis. |
| Median age: PD 61, controls 29 | A large age confounding risk requires common-support and age-only baselines. |
| Reported male gender: 630/970 PD and 3,142/3,890 controls; some missing values | Preserve reported gender terminology; assess imbalance and performance by group. |
| Median recordings: PD 8, controls 3; most active 1% of all participants supplied 30.4% of recordings | Normalize training contributions by participant; prevent prolific contributors dominating means and loss. Among binary labels, PD supplies about 63.4% of recordings but only 20.0% of participants. |
| Median follow-up: PD 10.36 days, controls approximately 2.34 minutes | Screening at a fixed acquisition budget must be separated from longitudinal aggregation. |
| PD medication responses: 9,033 pre, 9,786 post, 17,638 another time, 2,879 reported nonuse, 52 missing recordings | Medication state is heterogeneous; missing and “another time” are not OFF states. |
| 422 PD participants have both pre and post reports; 243 have both on at least one UTC date | Paired analyses are plausible candidates, pending valid audio and timing checks. These are not 422 confirmed pharmacologic ON/OFF pairs. |
| Voice-table `phoneInfo` and `appVersion` are populated for every row: 11 and 3 distinct values | Recover recording-specific device/app information into a new derived manifest; baseline survey device is an inferior substitute. |
| Pilot: 120 recordings from 120 people, all extracted successfully | Useful for initial QC inspection; insufficient for within-person repeatability estimation or population QC prevalence. |
| Full checkpoint: 6,016 attempted rows, 5,991 worker errors, 16 other errors, 9 successes; no successful PD rows in that snapshot | Full-dataset quality/feature inference is blocked. Worker errors cannot be coded as zero quality or evidence that PD audio is worse. |
| Current `estimated_snr_db` flags 79/120 pilot records below 10 dB | This is not a validated rejection threshold. |

The worker-error strings match `BrokenProcessPool`/abrupt process termination.
The 16 other entries match `CalledProcessError`; their individual causes still
need diagnosis. These facts establish a processing failure, not its root cause.
Do not stop or rewrite the running EDA process as part of this plan. Diagnose in
an isolated job, export successful retry results separately, and later merge
through a versioned recovery manifest.

**Critical QC measurement issue.** `audio_quality.py` defines both
`frame_dynamic_range_db` and `estimated_snr_db` as
`20 log10(P90(frame RMS)/P10(frame RMS))`. It does not isolate noise-only from
voice frames. A synthetic clean constant-amplitude sine produces 0.0 dB by this
formula, whereas stationary noise produces approximately 0.73 dB; both trigger
the existing 10 dB flag. Keep the legacy column for provenance, but use the name
`frame_energy_dynamic_range_db` in new analyses and do not treat it as true SNR.
A real noise estimate requires validated background intervals or a validated
noise estimator; if unavailable, record SNR as unavailable.

## 3. Define the estimands before changing the cohort

### A. Primary: new-participant PD screening at an initial assessment

Outcome: participant-reported professional diagnosis, with the existing
conservative contradiction policy. This is not clinician-adjudicated diagnosis.

Use the first eligible assessment session and a fixed acquisition budget, with
exactly one recording as the essential baseline. A maximum-three-recording
session aggregate is a prespecified secondary comparison. Specify a session
boundary from task timestamps before examining predictive performance. Report
people with only one/two recordings separately; never require three silently.
Exclude recording count, follow-up length, medication-response text, and metadata
missingness indicators from the primary voice-only classifier inputs.

Audit temporal label validity: recordings before the baseline diagnosis survey
are not automatically confirmed PD recordings. Compare post-survey windows
0–1, 0–7, and 0–30 days using the saved audit counts. Choose and lock the target
window based on the intended deployment and data support, not the best AUROC.
Record diagnosis/onset years as coarse self-reports, not exact diagnosis dates.

### B. Secondary: longitudinal voice phenotype

Describe each participant over a prespecified horizon, initially the first 30
post-survey days, with equal day contributions. Sensitivities: first day, first
7 days, and all available follow-up. Explicitly report the change in population
and acquisition effort. An all-follow-up score is not a first-visit screening
score. Keep state-specific feature summaries rather than calling a mixture an
“unmedicated phenotype.”

### C. Separate exploratory analysis: medication-associated voice change in PD

Compare pre/post recordings within the same participant, preferably within a
verified local day/session and known device. Retain the observational wording
“medication-associated change.” No dosage, verified washout, exact dose time,
peak exposure, or full severity adjustment is established by the current fields.
A reported nonuser differs from a treated person recording before the next dose.

## 4. Criteria by factor

| Factor | Proposed treatment | Guardrail |
|---|---|---|
| Missing file, empty/nonfinite decoded audio, impossible format | Technical ineligibility with an explicit reason; retry infrastructure errors separately | A transient decoder/worker failure is not irreversible audio rejection. |
| Duplicate source/content/near-duplicate | One analysis unit per duplicate; all linked participants stay in the same split | No increased weight from copied recordings or overlapping windows. |
| Wrong task, competing speaker, interrupted task | Blinded review and task-validity flag; affected segments masked | Do not conflate irregular vowel phonation with wrong task. |
| Too little usable phonation | Feature-specific unavailable status; 2 seconds is an initial feasibility marker | Test 1/2/3/5-second sensitivity and estimator stability before adopting cutoffs. |
| Clipping/saturation/dropout | Inspect fraction and sustained runs/flat tops; downweight only after calibration; mask confirmed damaged intervals | Existing >1% clipping is a review trigger, not a universal clinical exclusion. Compression/AGC can hide clipping below full scale. |
| Background interference/reverberation | Reviewer labels plus validated signal measures; feature-specific uncertainty | Energy percentile ratio, spectral flatness, and HNR alone cannot certify environmental noise. |
| Low RMS/intensity | Retain; audit device/gain/distance and noise context | Hypophonia can be biological. Uncalibrated smartphone amplitude is not calibrated vocal loudness. |
| Pitch-tracker failures | Mark jitter/shimmer/F0 feature validity separately; compare pitch settings and algorithms | Aperiodicity may be PD-related. Failure of periodicity analysis does not invalidate the embedding or recording. |
| High jitter/shimmer, low HNR, high F0 variability | Retain as candidate disease features | Never use the same feature's “abnormality” as its quality penalty. |
| Medication timing | Five explicit categories: pre, post, another time, reported nonuse, missing | No arbitrary medication quality multiplier; do not infer diagnosis from medication. |
| Age and reported gender | Flexible age adjustment, common-support comparisons, subgroup reporting | Separate voice-only from demographic-augmented models. Do not residualize out disease signal indiscriminately. |
| Device/app/native sample rate/codec | Recording-level linkage, nuisance-only baseline, device-held-out sensitivity if supported | Resampling cannot restore bandwidth absent from the original recording. |
| Smoking/health history/DBS/surgery | Validate coding, missingness and availability, then stratify or adjust when justified | Missing is not absent. DBS and duration are PD-only effect modifiers, not generic control-group covariates. |
| Disease duration/onset/severity | PD-only descriptive/interaction analyses; severity remains incompletely observed | Do not assign severity weights from recording count, age, or medication use. |
| Time of day/fatigue/practice | Within-person modeling, task order and time-since-enrollment terms | UTC is not local time; fatigue and practice are hypotheses unless measured. |
| Repeated participation/selection/missing audio | Participant/day balance, retention tables and coverage reporting | Dropout and missingness can be informative; no confident missing-at-random claim. |

For acoustic analysis, keep full-phonation and standardized central-window
measurements in parallel. Selecting the smoothest window can stabilize an
estimator while removing the pathological instability of interest. Do not use
an adaptive “best” window as the only phenotype. Retain onset/offset, pauses,
and instability as separate measures where estimable. Run raw-audio primary
analysis; denoising/amplitude normalization is a sensitivity branch with feature
preservation checks, not a default correction.

## 5. Medication coding and analysis

Preserve original text and map only these observed values:

- `Immediately before Parkinson medication` → reported pre-dose.
- `Just after Parkinson medication (at your best)` → reported post-dose.
- `Another time` → treated-state timing unspecified, not midpoint or OFF.
- `I don't take Parkinson medications` → reported nonuse at that assessment.
- Missing → unknown, never imputed as nonuse.

Participants may legitimately change responses over time; do not force one
lifetime medication category. PD nonuse is not itself a label contradiction.
Keep the existing conservative control contradiction quarantine for the primary
analysis and repeat a sensitivity analysis using direct source diagnosis.

Medication text is a near-direct label shortcut in this dataset: most controls
report nonuse. It must not be a predictor of the primary PD-versus-control
voice model. A metadata-only diagnostic baseline can measure this shortcut,
explicitly as an audit rather than a useful voice biomarker.

For paired analyses, first audit timestamp chronology, matching candidates,
quality, device consistency, and local time availability. Use nonoverlapping
nearest pre/post matches within a prespecified feasible time window; do not
reuse one recording in multiple pairs. Compare 1/3/6-hour matching tolerances
as descriptive sensitivity, since the dose time is unknown. Same UTC date alone
is insufficient proof of a valid medication pair. Require repeated paired days
for individual response claims; a single pair supports only a noisy descriptive
contrast. Count paired participants and days after every filter.

For feature k, an exploratory PD-only model is:

`x_itk = participant_intercept_ik + beta_k * reported_post_it + f_k(time_of_day_it)
         + g_k(time_since_enrollment_it) + technical_covariates_it + error_itk`.

Include medication slope variation by participant only with adequate data.
Use date/session effects or matched-day contrasts as supported; assess residual
autocorrelation and use participant/day block bootstrap or appropriate temporal
errors. If local time is unavailable, label UTC-time adjustment inadequate for
circadian control and restrict claims. Missing severity and dose information
prevent a definitive causal treatment-effect interpretation. Correct exploratory
feature-family tests for multiplicity and report effect sizes with intervals.

## 6. Establish technical weights rather than guessing them

### Calibration data

After a recovery pilot succeeds, select approximately 400 training-set recordings
from distinct participants initially, with coverage of diagnoses, age/gender,
recording devices, medication states, and technical flags. This is a proposed
starting budget, not a power calculation. Add matched repeated recordings and
at least 100 blinded repeat judgments to assess agreement. Reviewers must have
appropriate research-data access and work within approved storage.

Two reviewers independently mark external interference, clipping/dropouts, task
completion, competing voices, and usable spans, without diagnosis or medication
labels. They explicitly distinguish dysphonia from technical contamination.
Adjudicate disagreements. Compute raw agreement, confusion matrices, and
weighted Cohen kappa with participant-bootstrap intervals for ordinal ratings;
inspect prevalence effects rather than relying on a single kappa cutoff.
Estimate sensitivity for severe technical contamination. If reviewer agreement
or subgroup reliability is poor, revise the rubric before fitting a quality model.

Augment suitable training recordings with controlled noise, clipping, dropout,
codec and reverberation changes. Measure feature distortion relative to each
recording's original, not relative to the healthy group. Synthetic degradation
requires validation against naturally degraded reviewed recordings. If human
quality judgments are sampled by flags/class, calibrate to the intended training
population using known sampling probabilities or a representative calibration
subset; an enriched review sample is not a prevalence sample.

### Weight definition

For feature k, fit an out-of-participant estimate of **technical measurement
error variance**, `v_hat_irk`, using corruption experiments and reviewed technical
signals. Estimate variance on training-scaled feature units. Use a positive
regularizer `tau_k^2` and cap weight ratios, initially comparing maximum ratios
1, 2 and 5, with 10 only as a sensitivity analysis:

`q_irk = bounded_rescale(1 / (v_hat_irk + tau_k^2))`, with `0 < q_irk <= 1`.

A validity mask is separate: `m_irk = 0` means feature k cannot be estimated;
otherwise `m_irk = 1`. A recording can have unavailable jitter and valid spectral
or embedding features. No universal scalar weight is assumed optimal for all
features or for all three encoders. Encoder reliability should be evaluated
using embedding stability under technical corruption and reviewed recordings.

A calibrated probability of technical usability is an alternative provisional
weight, but it is not automatically inverse-variance weighting. Compare it
explicitly rather than claiming the two have the same meaning. Do not multiply
correlated proxy subscores for duration, energy, HNR and pitch confidence.

Use repeated same-state/session measurements to assess reliability, but do not
call all within-person variance technical error; biological state variation is
real. Never fit quality weights from PD classifier loss, target labels, or how
similar a voice is to healthy voices. Uniform weights remain the primary
fallback if calibration is unsupported. Missing quality measurements receive an
explicit missing flag and a prespecified fallback, not automatic zero weight.

## 7. Weighted means, Huber means, and training loss

### Ordinary weighted mean baseline

Within a participant/session/state, a feature-specific mean is:

`mu_ik = sum_r(m_irk * q_irk * x_irk) / sum_r(m_irk * q_irk)`.

A zero denominator means unavailable, not zero feature value. This is the first
comparator and is straightforward to audit.

### Preferred robust acoustic summary

Within the same participant, session, medication state and feature, estimate:

`mu_hat = argmin_mu sum_r(m_r * q_r * rho_c((x_r - mu) / s_k))`.

Huber rho is quadratic near zero and linear beyond c. Iterative effective
weights are proportional to `m_r * q_r * min(1, c / abs(u_r))`, where
`u_r = (x_r - mu) / s_k`. Start comparisons at `c = 1.345`, with 1.5/2 and an
ordinary weighted mean as sensitivity settings; these are statistical tuning
constants, not clinical boundaries. Use a prespecified robust scale, with a
positive training-derived floor and explicit handling of one/two observations
or zero MAD. Retain within-state dispersion, tails and intermittent abnormalities
as separate features: robust location does not establish that an extreme repeat
is technical corruption. For positive skewed features, compare a training-prespecified log
transform; report back-transformed summaries as geometric-scale summaries,
not arithmetic means.

Summarize days within state using equal-day robust aggregation. Across people,
use one summary per participant; do not Huber-downweight severe participants
because they are extreme compared with controls. Never robustify across pre and
post states in a way that deletes genuine medication transitions. With too few
repeats, report a single-observation estimate and uncertainty/coverage flag.

For vector embeddings, use a quality-weighted arithmetic mean as primary;
coordinate-wise Huber depends on embedding axes. A within-state geometric-median
comparison is optional and requires separate validation.

### Participant/day-balanced recording-level loss

For participant i, day d, recording r, let q_idr be validated technical reliability
and D_i the number of eligible days in the chosen horizon:

`a_idr = (1 / D_i) * q_idr / sum_j_in_day(q_idj)`.

Then `sum_d,r a_idr = 1` for each participant. Optimize:

`L = (1 / N_participants) * sum_i sum_d,r a_idr * BCE(y_i, p_idr)`.

Segment weights first sum to their recording weight, so multiple overlapping
segments do not multiply influence. Sessions may replace days in the initial
assessment estimand. Normalize within the appropriate eligible feature/recording
set and document missing timestamps and day-boundary rules.

This normalization intentionally makes technical weights relative within a
participant/day: a participant with all low-reliability recordings still has
unit total weight. Globally downweighting those participants would change the
target population. Handle inadequate evidence through calibrated abstention
and explicit coverage, or evaluate participant-level reliability weighting as
a labeled sensitivity analysis with ordinary unweighted evaluation.

Class balancing is separate and optional: participant-class factors such as
`N/(2*N_class)` change the training objective and can affect calibration. Never
apply class weights implicitly through recording counts. Recalibrate on an
independent validation sample for the intended target prevalence if justified;
mPower's selected sample does not establish community PD prevalence.

Huber robustness belongs in continuous feature aggregation or a clearly
specified continuous prediction task. It is not a reason to suppress large
binary-classification losses, which may be genuine difficult PD cases.

## 8. Demographic and acquisition adjustment

First estimate age/gender overlap at the **participant** level. Compare all-age
screening with a prespecified overlapping-age cohort; 50–80 is an existing
sensitivity range, not a claim that it is optimal. Age-only, demographic-only,
device-only and quality-only baselines quantify shortcut performance. Evaluate
voice-only and demographic-plus-voice models separately.

For feature associations, fit participant-level age splines/reported-gender
models with smoking/device terms as justified by support and missingness.
Do not indiscriminately control for RMS, HNR, jitter, medication use or disease
severity when those lie on the disease-to-voice pathway. Document the causal
question and adjustment set for each analysis. Optional overlap weighting or
matching uses training-derived nuisance models, positivity checks and bounded
weights; it defines an overlap population and is separate from measurement
reliability. Do not multiply every possible weight together by default.

Record-specific `phoneInfo`/`appVersion` must be linked by validated source-row
or recording identity with strict cardinality checks. Audit whether device
changes within people, missingness, or app versions track diagnosis. Smoking,
DBS, health history and treatment fields have differing denominators and skip
patterns; establish codebooks before treating values as comparable covariates.
Do not infer absence of a comorbidity from missing survey data.

## 9. Validation, comparisons, and adoption gates

Keep existing participant/duplicate assignments where valid. Extend full-cohort
splits using duplicate components while respecting locked assignments. If full
stratification requires a new split, version it before modeling and keep it
separate from previously evaluated splits. All QC calibration, scaling,
regularization, clipping, feature selection and hyperparameter choices occur
inside training folds. Blinding reviewers to diagnosis does not authorize using
test participants to calibrate weights.

Run these comparisons with the same participant splits and acquisition budget:

1. One-recording and ordinary participant-mean/median baselines.
2. Participant/day-balanced training with uniform quality weights.
3. Calibrated quality-weighted arithmetic summaries/loss.
4. Weighted Huber acoustic summaries; separately evaluate embedding means.
5. Medication-stratified analyses and matched pre/post sensitivity.
6. Demographic/device common-support and nuisance-baseline analyses.

The current benchmark CLI uses unweighted means/medians and unweighted head
fitting; this protocol is **not yet wired into it**. Future implementation must
verify the installed sklearn head APIs for sample weights, or use explicit
per-recording PyTorch losses/sampling. Do not assume that changing participant
prediction aggregation automatically changes the training loss.

Report participant AUROC, AUPRC, balanced accuracy, sensitivity, specificity,
Brier score, log loss and calibration, with participant-cluster bootstrap CIs.
Report abstention coverage and conditional performance together. Include fixed
sensitivity operating points chosen on validation data. Do not quality-weight
the primary test metrics to make low-quality cases disappear.

Report retention, total weight, effective sample size
`ESS = (sum w)^2 / sum(w^2)`, feature availability and error rates by diagnosis,
age, reported gender, device, and medication group. ESS is a weight-concentration
diagnostic, not an independent sample-size claim; report participant counts and
cluster uncertainty separately. Inspect whether weights preferentially remove
low-volume, irregular or dysphonic PD voices, and whether within-person
pre/post contrasts are preserved.

Adopt weighting only if independent validation supports improved technical
measurement fidelity or stability, acceptable calibration and coverage, and no
unexplained clinically important subgroup degradation. Predefine a practically
meaningful margin with the study team or a power/precision calculation. A small
AUROC increase alone is insufficient. Lock the method before one final test
assessment and seek external-device/cohort validation when available.

## 10. Work packages and concrete outputs

| Order | Work | Output / gate |
|---|---|---|
| 1 | Isolated worker-failure diagnosis and small diverse retry pilot | Recovery manifest; failure causes separated from audio quality; successes across PD/control, devices and medication categories before a full rerun. |
| 2 | Versioned time/device/label linkage and complete descriptive QC | Data dictionary, missingness/retention tables, validated recording-specific device joins and fixed analysis windows. |
| 3 | Blinded technical-review rubric and reliability study | Annotation guide, interval labels, agreement report; refine rubric when unreliable. |
| 4 | Training-only feature/encoder reliability calibration | Versioned q-model, feature masks, capped-weight policy, natural/synthetic corruption checks. |
| 5 | Reusable aggregation and loss implementation | Explicit participant/session/state summaries, weights, denominators, ESS and abstention flags. |
| 6 | Controlled comparisons and sensitivity analyses | Participant-level results with CIs, state/demographic/device reports and selection-bias audit. |
| 7 | Lock protocol, evaluate held-out data, external validation | Final documented operating policy; no re-tuning after test results. |

The small retrospective plan audit is complete. The worker recovery, additional
review labels, weight calibration, weighted training and medication-effect
analyses are proposed next work; none is claimed as already completed here.

## Sources supporting the design

- [Omberg et al., remote mPower monitoring, Nature Biotechnology (2022)](https://www.nature.com/articles/s41587-021-00974-9): directly documents demographic and repeated-participant confounding in mPower. Its conclusions motivate participant splitting and nuisance baselines, not a particular weight formula.
- [Chaibub Neto et al., PLOS ONE (2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9352058/): shows how daily timing and serial correlation complicate observational medication-response analyses. Its empirical application uses tapping; transferring the design principles to voice here is a methodological proposal, not validation of a voice medication effect.
- [Objective Methods of Sample Selection in Acoustic Analysis of Voice](https://pmc.ncbi.nlm.nih.gov/articles/PMC3212407/): demonstrates that window selection affects acoustic-measurement stability in Parkinson-related voice samples, supporting explicit parallel window policies rather than an unexplained best-window rule.
- [statsmodels HuberT documentation](https://www.statsmodels.org/stable/generated/statsmodels.robust.norms.HuberT.html): documents the Huber M-estimation function and default tuning constant; this is not a PD-specific calibration.
- [scikit-learn Cohen kappa documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html): defines agreement and ordinal weighting for reviewer labels, distinct from weighting a feature mean.
