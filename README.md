# Parkinson's disease voice research

Reproducible mPower metadata linkage, audio QC/acoustic analysis, and frozen
speech-model benchmarking. The GitHub repository contains code, synthetic
fixtures, an output-free notebook, methods, and aggregate execution summaries.
Real research data and private experiment outputs live separately in `pd_data`.

Start with [current results and blockers](docs/CURRENT_RESULTS.md),
[pd_data context](docs/PD_DATA_CONTEXT.md), and
[the consolidated run history](docs/MPOWER_RUN_HISTORY_AND_WEIGHTING.md).

To obtain pretrained weights on another machine, follow the
[step-by-step download and Slurm guide](docs/DOWNLOAD_PRETRAINED_MODELS.md).
It downloads nine public checkpoints at pinned revisions and verifies them
without requiring PyTorch, Hugging Face login or access to participant data.
For the QC audit, exact input/window policies and staged classifier training,
see the [sustained-vowel preprocessing and frozen-model plan](docs/FROZEN_MODEL_PREPROCESSING_PLAN.md).

## Repository structure

```text
configs/       EDA/QC and unified frozen-model YAML
scripts/       download/linkage, EDA, benchmark CLIs and versioned Slurm templates
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
pd_data/environments/       HPC Python environments and worker archive
pd_data/slurm/              submitted job snapshots and runtime logs
```

Use distinct approved paths outside Git:

```bash
export MPOWER_DATA_ROOT=/approved/storage/pd_data/mpower
export MPOWER_OUTPUT_ROOT=/approved/storage/pd_data/mpower_phase1
export MPOWER_BENCHMARK_ROOT=/approved/storage/pd_data/mpower_benchmarks
export MPOWER_SLURM_ROOT=/approved/storage/pd_data/slurm
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
python scripts/run_frozen_benchmark.py --model all --config configs/frozen_benchmark_local.yaml --prepare-only
python scripts/run_frozen_benchmark.py --model all --config configs/frozen_benchmark_local.yaml --smoke-test
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
Copy `.env.example` to `.env` and set approved HPC paths. Submit with:

```bash
bash scripts/submit_mpower.sh frozen_benchmark --smoke-test
```

The helper keeps job snapshots and logs in
`MPOWER_SLURM_ROOT` while running source code from this checkout. Use standard
`SBATCH_*` environment variables for scheduler options. See the
[reorganization audit](docs/REORGANIZATION_AUDIT.md) for the latest verified status.

See the [extraction repair and parallel benchmarks](docs/EXTRACTION_RELIABILITY.md)
and the [complete acoustic feature catalog](docs/ACOUSTIC_FEATURES.md) for the
184 core measurements and pilot sensitivity checks.

## Current status

All nine public pretrained checkpoints are now downloaded and hash-verified on
HPC (12.02 GB). Offline GPU smoke checks passed for all nine, using six real
recordings per checkpoint and both classifier heads. Thirty-three benchmark,
download and adapter unit checks passed. These are execution checks; no full-
cohort pretrained experiment or validated predictive performance is claimed.
See the [local benchmark guide](docs/LOCAL_PRETRAINED_BENCHMARK.md) and
[aggregate validation summary](docs/results/pretrained_integration_summary.json).
HeAR remains excluded pending access.

Full acoustic preprocessing completed: 64,973/65,022 recordings succeeded.
The remaining 49 files match their malformed/empty Synapse source objects;
see the [failure audit](docs/FAILED_AUDIO_AUDIT.md). The legacy SNR-named energy
proxy is not
a validated noise measurement. See [execution status](docs/FROZEN_BENCHMARK_STATUS.md)
and [machine-readable aggregate summary](docs/results/verification_summary.json).

The [quality/medication weighting protocol](docs/PD_QUALITY_MEDICATION_WEIGHTING_PLAN.md)
proposes calibrated technical weights, robust scalar aggregation and equal
participant/day contributions. Medication state and demographic/device effects
are separate analyses. These weights and weighted loss are **not yet fitted or
implemented** in the current runner.
