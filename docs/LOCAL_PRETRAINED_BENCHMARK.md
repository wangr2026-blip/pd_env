# Local pretrained model integration

Nine public checkpoints are configured in `configs/frozen_benchmark_local.yaml`.
Each entry records its repository, pinned commit and downloader alias. The CLI
and Slurm template default to this configuration. The older
`configs/frozen_benchmark.yaml` remains available explicitly for the original
three-model setup and synthetic regression checks.

Set `MPOWER_MODEL_ROOT` to approved model storage, normally
`$MPOWER_BENCHMARK_ROOT/models`. The runner looks under
`$MPOWER_MODEL_ROOT/checkpoints/<alias>/`, validates downloaded provenance against
the pinned configuration and loads with `local_files_only=True`. A missing
checkpoint fails explicitly instead of falling back to a network download.
Run downloader `--public-only --verify-only` to check file hashes before use.
HeAR is excluded pending access.

The nine keys accepted by `--model` are:

```text
ast_audioset           hubert_large             wav2vec2_base_960h
wav2vec2_large_robust  wavlm_base_plus          wavlm_large
whisper_large_v3      whisper_medium           whisper_small
```

`--model all` iterates the entries of the selected configuration. Architecture
selection comes from the checkpoint config, allowing multiple checkpoints of
one architecture. WavLM and HuBERT use acoustic encoders and their own feature
extractors; an ASR tokenizer is not required. Checkpoint normalization is
preserved. Group-normalized waveform models do not receive an encoder mask;
layer-normalized variants receive it. Waveform recordings are forwarded one
at a time to preserve independence from padding/batch composition, and valid
output lengths are computed from the encoder convolution stack.

Whisper uses its own Mel frontend (80 bins for Small/Medium, 128 for Large-v3)
and native padded window. Only the encoder is used; pooled tokens are limited
to valid input frames. AST removes special tokens and trailing time patches.
Cropping output tokens does not undo attention to padding inside AST/Whisper.
All backbones are frozen in evaluation mode. The local config uses 10-second
segments and hop, preserving pitch/timing. Residual tails below the configured
minimum duration are excluded consistently across models. Heads/scalers use
training data only; existing participant-disjoint assignments are reused.

Example after setting the three research roots plus `MPOWER_MODEL_ROOT` and
`MPOWER_BENCHMARK_PYTHON` to a complete Python 3.12 benchmark environment:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$MPOWER_BENCHMARK_PYTHON" scripts/run_frozen_benchmark.py --model whisper_medium --smoke-test
# On an approved GPU partition, using the standard private job/log helper:
SBATCH_PARTITION=gpuq SBATCH_GRES=gpu:1 SBATCH_MEM=64G \
  bash scripts/submit_mpower.sh frozen_benchmark --smoke-test
```

Smoke mode selects one participant of each class in each split (six recordings
per model), at most two accepted segments per recording and three MLP epochs.
It verifies inference, cached activations, both heads and artifact export; it
is not a full-cohort experiment or validated predictive performance. No full
cohort model training is launched automatically by downloading or validation.

Public asset verification status is in `$MPOWER_MODEL_ROOT/download_status.json`.
Private model-run reports and outputs stay under `$MPOWER_BENCHMARK_ROOT/reports`
and `runs/`. Read those reports before claiming that a checkpoint ran successfully.

## Verified HPC outcome — November 14, 2026

All nine public checkpoints, 66 files and 12,018,426,907 bytes, were downloaded
into private `pd_data/mpower_benchmarks/models/` and matched the pinned sizes
and upstream hashes. Slurm job 27563234 repeated verification and ran all nine
pretrained models with `HF_HUB_OFFLINE=1`, completing in 2:11 on an A100 80GB.
Each processed the same six real, participant-disjoint recordings and completed
both logistic and MLP heads. Saved provenance records CUDA use and pinned model
identity. These are execution checks on two people per split, not performance
estimates or full-cohort baselines.

Private execution evidence: `reports/execution_20261114T071701929673Z.json` and
`runs/` under `MPOWER_BENCHMARK_ROOT`, plus Slurm logs for job 27563234. The
[aggregate machine-readable summary](results/pretrained_integration_summary.json)
contains no participant records. Thirty-three distinct local checks passed:
16 original benchmark tests, eight download-integrity tests, and nine expanded
adapter/configuration/provenance tests. The final provenance-only correction
excludes re-verification timestamps from cache identity; it does not alter
inference. Identical asset hashes/revisions therefore retain cache identity
when the downloader re-verifies them.

Warnings observed include AST Mel-filter coverage, a Transformers mask-dtype
deprecation in WavLM, and an uninitialized wav2vec2 `masked_spec_embed` tensor.
The latter is a training-only masking tensor unused by these frozen eval-mode
forwards; no encoder layer was replaced with random weights. GPU smoke checks
passed despite these warnings. HeAR remains excluded pending access.
