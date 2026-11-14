# Frozen development protocol implementation — November 14, 2026

This implements the frozen-testing stages of FROZEN_MODEL_PREPROCESSING_PLAN.md.
Fine-tuning and final evaluation follow after development-only selection and
release review; neither is launched by this workflow. HeAR remains excluded.

## Protocol

The source is full cohort A's training and validation manifests. No test
manifest or final-test labels are opened. Existing participant/duplicate-linked
splits are reused and verified. The technical candidate requires the source
sustained-phonation task and at least two decoded seconds. Task compliance is
currently a source-column mapping, not a validated detector or manual review.
Near-duplicate release review is still pending; all experiments are provisional.
Quietness, aperiodicity, F0 variability, clipping proxies and HNR/jitter
availability do not exclude primary candidates. Pitch-detected material >=2 s
is a separately labeled sensitivity arm, with diagnosis-specific retention.

Each native recording is decoded once to its original rate/channels. The audit
records native sample counts, duration, per-channel RMS/peak/DC, full-scale
fractions/runs, repeated high-amplitude flat-top runs, zero runs and channel
correlation/downmix cancellation review. FFmpeg derives mono16k float32 from
that PCM. There is no manual denoising, loudness normalization, preemphasis or
adaptive trimming. Analysis QC and decoder warnings remain private. The legacy
energy percentile ratio is named frame_energy_dynamic_range_db; it is not SNR
and is not used as a weight.

Every model receives the same first min(N,160000) real samples, with spans below
32000 rejected explicitly. Native/full derived sample counts and capped samples
are recorded. In the matched condition, ceil(N/32000) complete two-second windows
are endpoint-anchored using rounded linspace starts. Their union covers every
primary-span sample without fabricated tails. Window starts, summed/unique
samples and overlap are logged; equal window weights produce one recording
vector. A one-window population SD is zero.

Two contexts are scheduled for every checkpoint: full and matched_2s. AST's
native pooler is the primary AST representation, with final-normalized patch
mean/statistical features labeled separately. Waveform and Whisper adapters
capture named last_hidden and middle transformer block taps; waveform adapters
also compare normalized pre-projection convolution features. Raw convolution
and encoder-input taps are implemented and tested but omitted from the initial
budget. Middle block names encode resolved depth. All inference is float32,
frozen eval, with no augmentation. Checkpoint frontends/masks are retained;
Whisper explicitly disables waveform normalization. No ASR tokenizer is needed.
AST/Whisper output cropping does not reverse internal attention to padding.

Only pooled vectors are cached by default. Cache identity includes model/source
hashes, context, named taps, dtype, software versions and relevant source code.
Encoder state hashes before/after extraction must match. Extraction failures
are logged and stop head fitting rather than silently changing the common
cohort. Every run records the shared immutable manifest digest.

## Heads and selection

Logistic regression uses inverse-visit-count sample weights, with total weight
one per training participant, for both scaler moments and loss. This fixed
normalization preserves the regularization objective when identical visits are
duplicated. The declared C grid is [0.01,0.1,1], selected using three-fold
StratifiedGroupKFold within training, grouped by duplicate-connected component.
The small PyTorch MLP uses the same weighted train-only scaler and uniform
participant sampling (one randomly chosen visit per person per epoch), giving
participant-balanced expected loss without version-dependent sklearn weighting.
It trains ten epochs; validation AUROC and then log loss select its checkpoint.
No learned transform uses validation/test data to fit its parameters.

The primary validation outcome uses the first eligible recording per person,
with all visits remaining in their assigned fold. Logistic/MLP predictions are
compared on that same acquisition rule. The operating threshold targets 90%
validation specificity and is stored for later carry-forward. Baselines use
age only, device/app only, and age+device on initial assessments, with training-
only imputation/encoding/scaling; medication is not a predictor. Retention and
common-support tables cover available diagnosis, age, reported gender, device,
app and medication covariates.

The final aggregator refuses missing tasks or differing manifest digests. It
ranks technical-primary pipelines by validation participant AUROC, log loss,
then lexical keys, selecting two distinct affordable checkpoints. Declared
budgets are 32 GiB peak allocated GPU memory and 30 extraction seconds per
recording. Only validation leaderboards are exported. The old comparison.csv
with test scores is never read. Final testing, clinical claims and fine-tuning
remain gated on the prespecified shortlist and release review.

## Execution and persistence

`scripts/submit_frozen_testing.py` copies source/configuration into a new private
experiment directory before submitting work. Existing jobs therefore do not
consume edited checkout files. It submits:

1. CPU preparation (16 CPUs, 48 GB, CPU partition's 45-day maximum).
2. Eighteen dependent GPU array tasks (9 checkpoints x 2 contexts), at most
   three concurrently, each with 4 CPUs, 64 GB RAM and the GPU partition's
   five-day maximum.
3. A validation-only summary job depending on success of the entire array.

A validation-job dependency can gate preparation as well. Invalid failed
prerequisites cancel dependent jobs. Submission receipts list each task/model,
job IDs, private code snapshot and output root. Models remain in the shared
verified local store; original audio and completed acoustic outputs are read
only. Private waveforms, manifests, features and predictions stay under
MPOWER_BENCHMARK_ROOT/protocol_experiments/.

```bash
# Set approved roots and interpreter, as in .env.example, then:
python scripts/submit_frozen_testing.py --plan-only
python scripts/submit_frozen_testing.py --max-concurrent 3
```

Local validation covers exact 1.999/2/2.001/9.7/10/>10-second boundaries, complete
coverage, cancellation/clipping, native mono/stereo WAV/ALAC at 16/44.1/48 kHz,
pitch/duration preservation, native AST pooling, raw/normalized/projected taps,
frozen state, participant-weighted moments, duplicate-visit invariance and
sealed-test refusal/label perturbation. Actual compute-node validation and
submission status are recorded in docs/results/frozen_protocol_submission.json.

## Submitted execution

Validation job 27563989 completed in 6:23. It prepared 64 development recordings,
ran all nine checkpoints in both contexts, completed 364 head/representation/
sensitivity pipelines, and generated the validation-only summary. Encoder state
hashes matched and no test labels were read. Its deliberately small validation
budget used two MLP epochs, two CV folds and one C value; its shortlist is a
software check, not the scientific shortlist from the full experiments.

The full configured budget is queued under preparation 27564044, GPU array
27564045 (tasks 0–17, concurrency three) and summary 27564046. The full code and
config snapshot is private under protocol_experiments/20261114T074847Z/code.
These full experiments are not yet completed. The technical metadata gate
(duration >=2 s and source phonation task) is applied before waveform preparation;
its aggregate exclusion count is recorded in the submission summary. Later
native-decode failures are recorded individually in exclusions.parquet. Retention
tables describe the eligible development candidates, not the held-out test set.

Fifty-two distinct unit checks passed across protocol (19), expanded adapters
(9), original benchmark (16) and downloader (8) tests. Code compilation and Slurm
shell syntax also passed. Runtime versions are recorded in the aggregate
submission summary and every private run provenance file. The small validation
results do not establish final model rankings or clinical performance.
