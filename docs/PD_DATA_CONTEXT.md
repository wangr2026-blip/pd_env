# How pd_env and pd_data fit together

`pd_env` is the code repository. Its sibling `pd_data` is approved persistent
HPC storage, not another source-code repository and not part of a Git clone.
The name `pd_env` also names an existing Conda environment; the repository and
that environment are distinct. A working Python environment must contain the
dependencies required by the selected pipeline.

```text
projects/
  pd_env/                           GitHub: wangr2026-blip/pd_env
    configs/                        preprocessing and frozen-model YAML
    scripts/                        download, EDA/QC, runners and Slurm wrappers
    src/{data,models,training,evaluation}/
    tests/                          synthetic tests and opt-in real integration tests
    notebooks/                      unexecuted characterization notebook
    docs/                           methods, aggregate status, plans and handoff
    requirements-*.txt
  pd_data/                          private HPC storage; not in Git
    README.md                       local storage guide
    mpower/                         MPOWER_DATA_ROOT
      voice_activity/
        Voice_Activity.csv
        metadata_parts/
        audio/
      demographics/Demographics.csv
      synapse_cache/
    mpower_phase1/                  MPOWER_OUTPUT_ROOT
      processed/                    canonical/private manifests, features, cohorts, splits
      eda/                          private tables, figures and diagnostics
      cache/                        preprocessing caches
      archive/                      earlier EDA snapshots
    mpower_benchmarks/              MPOWER_BENCHMARK_ROOT
      models/                       intended pretrained HF cache; currently unavailable
      segments/                     waveform caches when extraction succeeds
      activations/                  per-model hidden-state caches
      runs/                         real pretrained experiment outputs, when successful
      reports/                      execution status, including failed attempts
      planning/                     aggregate weighting audits and fixed pilot snapshot
      random_architecture_checks/   clearly labeled tiny-random-model integration outputs
      handoff/                      consolidated private plans, evidence and logs
```

Directory names such as `models/`, `runs/`, and `activations/` show the intended
layout; they may not exist until a successful relevant command creates them.
They are not proof of downloaded weights or completed experiments.

## Configuration

Set three distinct, nonoverlapping roots outside the repository:

```bash
export MPOWER_DATA_ROOT=/approved/storage/pd_data/mpower
export MPOWER_OUTPUT_ROOT=/approved/storage/pd_data/mpower_phase1
export MPOWER_BENCHMARK_ROOT=/approved/storage/pd_data/mpower_benchmarks
```

On the original HPC these are siblings under
`/mnt/isilon/wang_lab/wangr7/projects/pd_data`. Runtime code reads the environment;
this host-specific path is context, not a portable default. Metadata `audio_path`
links to actual recordings; do not reconstruct participant identity from audio
filenames. The canonical tables link voice activity to the separately downloaded
baseline demographics with explicit identity, label and cardinality checks.

`MPOWER_PYTHON` selects the EDA Python executable. `MPOWER_BENCHMARK_PYTHON` selects
the frozen-model batch executable, and `MPOWER_BENCHMARK_CONFIG` can select a
fixed private pilot config. The local `.mpower_venv` and `.benchmark_venv` are
ignored host-specific environments, not portable artifacts committed to Git.

## What each output means

- `mpower/` contains real research recordings and source metadata. Do not edit
  source exports to repair labels or append download shards as new records.
- `mpower_phase1/processed/` contains participant-linked analysis tables and
  provisional splits. Preserve unknown/conflicting labels. Exact hashes and
  duplicate groups help prevent recording/participant leakage.
- `mpower_benchmarks/planning/` contains descriptive aggregate audits and an
  immutable 56-person cohort-A pilot snapshot. The snapshot prevents a running
  EDA process changing experiment membership midway through a run. It remains
  a small provisional pilot, not a validated release cohort.
- `random_architecture_checks/` contains real-recording tests with tiny RANDOM
  backbones. These verify processing and artifacts, not pretrained prediction.
- `reports/` includes failures as well as successes. Read the status and model
  provenance; a report or metrics file alone does not prove pretrained execution.
- `handoff/` preserves plans, audit snapshots and execution logs. It is private;
  raw logs can contain identifiers or paths and are not published wholesale.

The version-controlled [current results](CURRENT_RESULTS.md) and
[run history](MPOWER_RUN_HISTORY_AND_WEIGHTING.md) summarize actual outcomes.
The [weighting protocol](PD_QUALITY_MEDICATION_WEIGHTING_PLAN.md) is proposed
future methodology; it has not been fitted or wired into current model training.

## Git, cloud work, and transfers

Publish source code, synthetic fixtures, an output-free notebook, documentation,
and selected aggregate status summaries. Keep recordings, source exports,
participant-level manifests/predictions, embeddings, raw logs, credentials and
model checkpoints outside Git, even for a private repository. `.gitignore`
provides additional protection but does not replace content review.

A cloud checkout contains code and aggregate documentation only. It has no
implied permission or direct access to the HPC research data. Use synthetic
fixtures for development. Public pretrained model assets may be downloaded in
a network-enabled environment, packaged with revisions/checksums, and copied
through an authorized transfer to the shared model cache without transferring
participant recordings or metadata.

The original command-execution session denied network sockets: external HTTPS
and the configured DNS server returned `Operation not permitted`; Hugging Face,
PyPI and the Slurm controller could not resolve. The connected GitHub service
has separate connectivity, allowing publication of reviewed repository files.
This does not grant Python downloads or shell Git/Slurm network access.
