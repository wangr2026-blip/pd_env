# Interpretability pipeline

Which interpretable acoustic features (pitch, jitter, shimmer, HNR, spectral
shape, formants, loudness) does each frozen audio encoder represent, layer by
layer? Are they used for PD prediction, and does a coverage map predict which
feature fusions help? This package is the config-driven pipeline for that
analysis. It reuses the repository's existing decoder, manifests, splits and
features rather than reimplementing them.

The sections marked **Methods** are written so they can go straight into a
paper's methods section.

## Status

| Stage | Module | Status |
|---|---|---|
| 0 Cohort, QC, folds, matched subset | `pipeline.cohort` | **implemented, tested** |
| Synthetic dataset (smoke tests) | `pipeline.datasets.synthetic` | **implemented, tested** |
| 1 Handcrafted features (+CPP, Praat formants) | `pipeline.features` | planned |
| 2 Layer-wise embeddings (+random-init controls) | `pipeline.embed` | planned |
| 3 Predictive baselines (3 age regimes) | `pipeline.baselines` | planned |
| 4 Coverage map | `pipeline.coverage` | planned |
| 5 Erasure (LEACE) and fusion | `pipeline.causal` | planned |
| 6–8 Hypothesis test, fairness, extensions | — | planned (stubs) |

## Quick start

```bash
python -m pip install -r requirements-pipeline.txt

# Synthetic smoke test: no participant data, runs in ~15 s on a laptop.
export PIPELINE_OUTPUT_ROOT=/tmp/pd_pipeline_smoke
python -m pipeline.cohort --config configs/pipeline/smoke_synthetic.yaml
python -m pytest -q tests/pipeline

# mPower on the HPC (development participants only; test sealed).
export MPOWER_OUTPUT_ROOT=/approved/storage/pd_data/mpower_phase1
export MPOWER_BENCHMARK_ROOT=/approved/storage/pd_data/mpower_benchmarks
export MPOWER_PYTHON=/path/to/env/bin/python
sbatch --export=ALL,PIPELINE_STAGE=cohort,PIPELINE_CONFIG=configs/pipeline/mpower_vowel.yaml \
       scripts/pipeline/cpu_stage.slurm
```

## Layout

```text
pipeline/
  cohort.py               Stage 0 CLI
  lib/config.py           YAML loading (extends:, ${ENV} expansion), stage cache keys
  lib/cache.py            content-addressed stage directories + completion manifest
  lib/folds.py            sealed-test guard, participant folds, participant aggregation
  datasets/base.py        Dataset interface (language, task, standard columns)
  datasets/mpower.py      mPower adapter over existing train/val manifests
  datasets/synthetic.py   synthetic vowels with planted acoustics
configs/pipeline/         base.yaml, smoke_synthetic.yaml, mpower_vowel.yaml
scripts/pipeline/         SLURM templates
tests/pipeline/           synthetic and fake-manifest tests
```

`lib` is not called `core` because the repository `.gitignore` excludes any
path named `core`.

## Configuration and caching

One YAML file per analysis. Dataset configs `extends: base.yaml` and override
only what differs. `${VAR}` expands environment variables and fails if unset.
For real data, `output_root` must be outside the Git checkout.

Each stage writes to `<output_root>/<run_name>/<stage>/<key>/`. The key is a
SHA-256 over:

- the config sections the stage depends on
- the keys of upstream stages
- the stage's source code
- package versions
- a fingerprint of the input data (mPower: SHA-256 of the manifests; synthetic: the generator parameters)

A stage is complete only once `manifest.json` has been written, atomically and
last. Re-running with an unchanged key is a cache hit. Changing a later
stage's settings never invalidates an earlier stage's outputs. `--force`
recomputes. `LATEST` in each stage directory names the most recent key.

## Methods: Stage 0 (cohort and folds)

**Source and sealed test.** We used the existing participant-level 70/15/15
partition (seed 20261112), in which participants linked by identical file,
native-PCM or normalized-PCM hashes form indivisible components. Only the
training and validation partitions (the "development set") were loaded. The
mPower adapter opens only `cohort_<X>/{train,val}_manifest.csv`. It never
opens the test manifest or participant-wide label tables. Every stage
additionally refuses any row whose source split is not train/val. The final
test set was not used for any analysis reported here.

**Labels and demographics.** Diagnosis is the participant-reported
professional diagnosis, with the repository's conservative contradiction
policy. Controls reporting a diagnosis year, a medication year or PD-medication
voice timing are marked conflicting and excluded upstream. Age and sex come
from the baseline demographics survey. Ages flagged invalid upstream are
treated as missing.

**Audio and QC.** Recordings are decoded by FFmpeg to mono 16 kHz float32
(`scripts/mpower_common.decode`) with no denoising, loudness normalization,
pre-emphasis or trimming. QC records decoded duration, RMS level (dBFS),
peak, and Praat-detected voiced time. Voiced time comes from autocorrelation
pitch (75–500 Hz, 10 ms step) and is reported as voiced frames × step. Audio
is flagged, never trimmed. For mPower, QC uses the completed full-dataset
decode and Praat pitch-detected duration (`qc_source: precomputed`). For
other datasets it is computed directly (`qc_source: compute`, parallelizable
with `cohort.workers`). Duration and voicing distributions by diagnosis are
reported in `qc_summary.csv` and `duration_distribution.png`.

**Exclusion criteria.** Criteria are applied in order and reported both
sequentially (`exclusion_flow.csv`: remaining recordings and participants by
diagnosis after each criterion) and per recording, with overlapping reasons
(`exclusions.csv`):

1. missing or non-binary label
2. missing age
3. age outside 18–100
4. missing sex
5. task other than sustained phonation
6. decode failure
7. duration < 2.0 s
8. voiced time < 1.0 s ("no voiced audio")
9. medication-timepoint filter (optional, off by default)

The 2 s minimum matches the existing frozen-model protocol and is a
feasibility rule, not a clinical cutoff. The 1.0 s voicing minimum was chosen
empirically. On synthetic whisper-like noise (formant-filtered white noise, no
glottal source), Praat reports 0.5–0.7 s of spurious pitch per ~5 s clip, so
a 0.5 s threshold does not separate unvoiced clips. Genuinely voiced
synthetic clips had at least 3.5 s. The medication filter keeps only listed
timepoint responses, separately for PD and controls (e.g. PD recordings made
"Immediately before Parkinson medication").

**Initial recording.** Each participant's initial recording is the earliest
eligible one by timestamp, with ties broken by recording ID. Analyses that
need one recording per participant use it; others use all recordings.

**Folds.** Participants are assigned to 5 folds with scikit-learn
`StratifiedGroupKFold` (shuffled, seed 20261112). Stratification is by PD
label; sex and age bins are optional. Groups are the duplicate-linked
components, so neither a participant nor any duplicate-linked participant
appears in more than one fold. Assignment is checked automatically: no
participant or group spans folds, no fold is empty, and every fold contains
both classes. All recordings of a participant inherit its fold. Every
downstream classifier and probe uses these folds (`lib.folds.iter_folds`).

**Age/sex-matched cohort.** Within eligible development participants aged
50–80 with known sex, participants are binned by 5-year age bin × sex. Each
stratum keeps equal numbers of PD and control participants, sampled without
replacement with a fixed seed. This is the rule of the repository's
cohort C, rebuilt inside the development set because the original cohort C
includes sealed-test participants. Matched participants keep their main-fold
assignment, so raw-cohort and matched-cohort results share folds. Coarsened
matching leaves residual age differences within bins. These are reported in
`cohort_summary.csv` and `fold_summary.csv`. No acoustic or model output is
used for selection. If a fold of the matched subset lacks a class, the stage
records a warning.

**Participant aggregation.** Recording-level probabilities are averaged per
participant (mean by default, median optional:
`aggregation.participant`). Metrics are reported at both recording and
participant level.

**Outputs.** `cohort.csv` (eligible recordings with fold, initial-recording
and matched flags, QC), `folds.csv` (one row per participant),
`cohort_summary.csv` (N, recordings, age mean/SD/median/IQR/range, % female
by cohort × group), `fold_summary.csv`, `exclusions.csv`,
`exclusion_flow.csv`, `qc.csv`, `qc_summary.csv`, `matching_flow.csv`. These
contain participant-level rows and must stay in private storage.

## Methods: synthetic dataset

Synthetic sustained /a/ vowels are used to smoke-test the pipeline and to give
probes a known answer. Each recording is a Rosenberg glottal pulse train:

- 40% open and 16% closing phase, evaluated in continuous time so there is no sample-grid quantization of jitter
- lip-radiation differentiation
- a cascade of three formant resonators (700/1220/2600 Hz; bandwidths 80/90/120 Hz)
- additive formant-filtered aspiration noise scaled to a target HNR
- 50 ms onset/offset ramps, a 0.2–0.6 s leading silence and a −60 dBFS noise floor

Per-cycle perturbations are i.i.d. Gaussian, with relative SD = target / 1.128,
so the expected local jitter and shimmer equal the targets. F0 has a slow 1%,
0.4 Hz drift.

Participants have a planted PD effect:

| | PD | Control |
|---|---|---|
| Local jitter (median) | 1.1% | 0.5% |
| Local shimmer | 6% | 3.5% |
| HNR | 15 dB | 21 dB |
| Age | mean 64 | mean 46 |

Age also adds a small jitter increase and HNR decrease. Sex sets F0 (male ≈120 Hz, female ≈205 Hz). Recordings vary around the participant baseline.

The generator also exercises every Stage 0 path:

- silent, too-short (1.2 s) and whisper-like unvoiced clips
- unlabeled participants, and missing age or sex
- duplicate-linked participant pairs
- mPower medication-timepoint wording
- rows in a sealed test split, which must never appear downstream

Audio is generated on demand from `(seed, recording)` and is bit-identical
across calls.

**Verification.** On 40 voiced synthetic clips, Praat estimates track the
planted values. Spearman ρ: F0 1.00, HNR 0.94, local jitter 0.89, local shimmer
0.71. Absolute scales differ: formant ringing and noise lower measured jitter
and raise measured shimmer relative to the nominal targets. Probe tests should
therefore use ranks or measured features, not nominal magnitudes. This check
runs in `tests/pipeline/test_stage0.py`.
