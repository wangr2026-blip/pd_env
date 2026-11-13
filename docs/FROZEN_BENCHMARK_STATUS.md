# Execution status — November 12, 2026

Implementation is present for all three frozen speech backbones, both heads,
mean/statistical/learned attention pooling, participant-disjoint split reuse,
reusable waveform/activation caches, metrics, artifact export, comparison CLI,
and Slurm execution. Pretrained validation remains blocked by network access.

## What actually ran

- **16 benchmark tests passed**: synthetic fixtures, real Transformers frontends
  and tiny randomly initialized encoder implementations, both heads, train-only
  normalization, leakage checks, calibration/aggregation, waveform resampling,
  cache reuse/invalidation, checkpoint reload and deterministic cached reruns.
- **Three real-recording architecture integration tests passed**: AST with mean
  pooling, Whisper with statistical pooling, wav2vec2 with learned attention.
  Each processed six real labeled recordings, extracted embeddings, trained and
  evaluated logistic and MLP heads, and saved complete outputs. These use tiny
  random backbones, explicitly identified as `test-only/random-*` in provenance.
- **Real data-only smoke passed**: canonical cohort B splits were reused; 28
  labeled recordings selected and six decoded (one in each split/class).
- The CLI help, Python compilation and Slurm shell syntax checks passed.
- **Final benchmark verification: 19 passed** after the duplicate-file protection
  and deterministic-seeding fixes (`final_benchmark_validation.log`).
- Automated comparison aggregation executed successfully on the explicitly
  labeled random-backbone smoke outputs (`comparison_check.log`).
- **Full regression suite: 28 passed**, including the nine existing EDA tests,
  16 new benchmark tests and three opt-in real-recording architecture checks
  (`validation.log`, 285.53 seconds). The final 19 benchmark checks were also
  rerun after the last benchmark code changes and passed.

An integration test exposed cache-dependent attention initialization because
wav2vec2 can consume RNG values even in eval mode. Training is now reseeded
independently after extraction; the cached-rerun regression passes.

## Pretrained status and blockers

| Requested checkpoint | Pretrained loaded | Download/cache | Real pretrained smoke |
|---|---|---|---|
| MIT/ast-finetuned-audioset-10-10-0.4593 | No | Blocked | Failed before extraction |
| openai/whisper-small | No | Blocked | Failed before extraction |
| facebook/wav2vec2-base-960h | No | Blocked | Failed before extraction |

All three requested checkpoints were attempted online using `--model all
--smoke-test`. Hugging Face DNS resolution failed on each checkpoint's frontend
files. The installed Transformers classes import and run locally, but this is
not evidence that pretrained weights loaded. No requested pretrained weights
were found in the existing local Hugging Face cache, and none were downloaded.
A pip dependency installation attempt likewise failed resolving pypi.org.

There is no visible GPU on this host (`nvidia-smi` unavailable). Slurm inspection
failed because controller `respublica-01` could not be resolved. No baseline
experiment or Slurm job was launched. GPU/mixed-precision execution remains
untested. Existing EDA inputs were read only; benchmark code did not submit,
cancel or modify EDA jobs. Current EDA splits remain provisional pending the
full duplicate/QC review.

**No smoke-test or random-backbone metric is validated predictive performance.**
The only trained heads so far are integration-test heads, not pretrained-model
baselines.

## Artifacts and rerun

Private artifacts are under:

```text
/mnt/isilon/wang_lab/wangr7/projects/pd_data/mpower_benchmarks/
  reports/execution_20261112T232803445913Z.json  # real data smoke
  reports/execution_20261112T233048418302Z.json  # all pretrained attempts failed
  pretrained_smoke.log                         # DNS errors and tracebacks
  unit_tests.log                              # 16 benchmark tests
  real_random_architecture_tests.log           # three real architecture checks
  final_benchmark_validation.log               # final 19 benchmark/real checks
  validation.log                              # full regression run
  random_architecture_checks/                  # clearly labeled private test outputs
```

Set the three storage environment variables as described in the benchmark
README. On a host with approved network access and available resources:

```bash
.benchmark_venv/bin/python scripts/run_frozen_benchmark.py --model all --config configs/frozen_benchmark.yaml --prepare-only
.benchmark_venv/bin/python scripts/run_frozen_benchmark.py --model all --config configs/frozen_benchmark.yaml --smoke-test
# After all pretrained smoke checks pass and compute capacity is confirmed:
sbatch scripts/frozen_benchmark.slurm
```

For Slurm, activate a complete environment or export
`MPOWER_BENCHMARK_PYTHON` to its absolute Python executable. `.benchmark_venv`
reuses existing host dependencies without modifying their environments.
Outside this host, install `requirements-benchmark.txt` into a fresh environment.
