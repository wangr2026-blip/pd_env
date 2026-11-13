# Parkinson's disease voice research

Reproducible mPower metadata linkage, audio QC/acoustic analysis, and frozen
speech-model benchmarking. The GitHub repository contains code, synthetic
fixtures, an output-free notebook, methods, and aggregate execution summaries.
Real research data and private experiment outputs live separately in `pd_data`.

Start with [current results and blockers](docs/CURRENT_RESULTS.md),
[pd_data context](docs/PD_DATA_CONTEXT.md), and
[the consolidated run history](docs/MPOWER_RUN_HISTORY_AND_WEIGHTING.md).

## Repository structure

```text
configs/       EDA/QC and unified frozen-model YAML
scripts/       download/linkage, EDA, benchmark CLIs and Slurm wrappers
src/
  data/        canonical manifests, splits, decoding and segmentation
  models/      frozen AST, Whisper encoder and wav2vec2 backbone
  training/    logistic/MLP heads and learned attention pooling
  evaluation/  recording/participant metrics and calibration
notebooks/     unexecuted mPower characterization notebook
tests/        synthetic correctness tests and opt-in real integration tests
docs/         methods, aggregate results, weighting plan and run history
```

## Storage and configuration

`pd_data` is a sibling private storage directory, not part of this Git repository:

```text
pd_data/mpower/             source recordings and metadata
pd_data/mpower_phase1/      canonical manifests, QC/features, EDA and archives
pd_data/mpower_benchmarks/  model caches, activations, private runs and evidence
```

Use distinct approved paths outside Git:

```bash
export MPOWER_DATA_ROOT=/approved/storage/pd_data/mpower
export MPOWER_OUTPUT_ROOT=/approved/storage/pd_data/mpower_phase1
export MPOWER_BENCHMARK_ROOT=/approved/storage/pd_data/mpower_benchmarks
```

See [data layout](docs/data_layout.md), [observed schema](docs/voice_activity_schema.md)
and [data dictionary](docs/MPOWER_DATA_DICTIONARY.md). Participant metadata,
recordings, embeddings, predictions, credentials and raw execution logs are
excluded from publication. Cloning this repository does not provide the dataset.

## Execution

Use Python 3.12 and install the requirements appropriate to the task:

```bash
python -m pip install -r requirements-preprocessing.txt
python -m pip install -r requirements-benchmark.txt
python -m pytest -q tests/test_frozen_benchmark.py
python scripts/run_frozen_benchmark.py --model all --config configs/frozen_benchmark.yaml --prepare-only
python scripts/run_frozen_benchmark.py --model all --config configs/frozen_benchmark.yaml --smoke-test
```

Tests exercising PyTorch require the benchmark dependencies. Real data require
applicable research permissions and approved storage. `--prepare-only` caches
public model assets; the runner still requires all three storage environment
variables. See [benchmark setup and output structure](docs/FROZEN_BENCHMARK_README.md)
for complete commands, GPU/CPU options and offline-cache context.

EDA/QC batch entry points are `scripts/preprocess_mpower.slurm` and
`scripts/full_mpower.slurm`; frozen models use `scripts/frozen_benchmark.slurm`.
Set `MPOWER_PYTHON` or `MPOWER_BENCHMARK_PYTHON` to a complete environment for the
selected job. Do not assume the existing Conda environment named `pd_env` has
all benchmark dependencies. Large workloads belong in Slurm allocations.

## Current status

The full regression suite passed 28 tests; 19 overlapping targeted checks were
also rerun and passed. Tiny random AST, Whisper and wav2vec2 backbones passed
real-recording integration checks. Requested pretrained weights have **not**
loaded: shell network connections were denied, and the assets were not cached.
No pretrained baseline experiment completed. The fixed pilot has 56 participants
with provisional participant-disjoint assignments and only eight test people.
Smoke metrics are execution checks, not validated predictive performance.

The full acoustic EDA snapshot also contains substantial worker failures, which
must be separated from audio quality. The legacy SNR-named energy proxy is not
a validated noise measurement. See [execution status](docs/FROZEN_BENCHMARK_STATUS.md)
and [machine-readable aggregate summary](docs/results/verification_summary.json).

The [quality/medication weighting protocol](docs/PD_QUALITY_MEDICATION_WEIGHTING_PLAN.md)
proposes calibrated technical weights, robust scalar aggregation and equal
participant/day contributions. Medication state and demographic/device effects
are separate analyses. These weights and weighted loss are **not yet fitted or
implemented** in the current runner.
