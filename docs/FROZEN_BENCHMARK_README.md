# Frozen mPower speech benchmarks

For the November 14 nine-checkpoint local integration and successful offline GPU
smoke validation, see [the local benchmark guide](LOCAL_PRETRAINED_BENCHMARK.md).
Earlier three-model/network-blocked status below is historical. The default
CLI/Slurm config is now `configs/frozen_benchmark_local.yaml`; the original
three-model configuration remains available explicitly.

One runner supports AST (`MIT/ast-finetuned-audioset-10-10-0.4593`), Whisper
(`openai/whisper-small`, encoder only), and wav2vec 2.0
(`facebook/wav2vec2-base-960h`, backbone only). Backbone parameters are frozen and
inference runs in eval/inference mode. Logistic regression and MLP heads use
training-only standardization. MLP epochs are selected by validation log loss.

## Setup

From the repository root, in a Python 3.12 environment with matching PyTorch/CUDA:

```bash
python -m pip install -r requirements-benchmark.txt
export MPOWER_DATA_ROOT=/approved/storage/mpower
export MPOWER_OUTPUT_ROOT=/approved/storage/mpower_phase1
export MPOWER_BENCHMARK_ROOT=/approved/storage/mpower_benchmarks
export MPOWER_SLURM_ROOT=/approved/storage/slurm
```

Choose three distinct directories outside Git. Benchmark outputs must not overlap
raw storage or EDA storage. No benchmark command changes EDA files or waits for
EDA completion. Private derived manifests, predictions, audio segments, and model
artifacts stay under `MPOWER_BENCHMARK_ROOT`, with a restrictive umask.

The local `.benchmark_venv` created for this session reuses installed dependencies
from `.mpower_venv` and the existing `plm-benchmark` environment through a `.pth`
file. It does not modify those environments. It is specific to this HPC host;
use the installation above elsewhere. Torchaudio is optional: Transformers AST
can use its NumPy/Kaldi-compatible frontend. If installing torchaudio, match its
version to torch.

## Run

```bash
python scripts/run_frozen_benchmark.py --model all --config configs/frozen_benchmark.yaml --prepare-only
python scripts/run_frozen_benchmark.py --model ast --config configs/frozen_benchmark.yaml --smoke-test
python scripts/run_frozen_benchmark.py --model whisper --config configs/frozen_benchmark.yaml --smoke-test
python scripts/run_frozen_benchmark.py --model wav2vec2 --config configs/frozen_benchmark.yaml --smoke-test
python scripts/run_frozen_benchmark.py --model all --config configs/frozen_benchmark.yaml
python scripts/compare_frozen_benchmarks.py --config configs/frozen_benchmark.yaml
```

On this host, use `.benchmark_venv/bin/python` for the commands. The full command
runs both heads for each model. `--smoke-test` selects at most six participants
per class per split and one recording each, uses at most two segments per
recording and three training epochs. Its metrics check execution only and **are
not validated predictive performance**. `--data-smoke` independently checks
canonical manifests and decodes six real recordings without model downloads.
`--prepare-only` loads and caches all requested model configs, weights and feature
extractors, plus Whisper/wav2vec2 tokenizers (AST does not have a tokenizer).
Errors produce a nonzero exit code and an execution JSON report; `--model all`
attempts every model even if one fails.

For batch execution, export the three storage variables, activate the intended
environment (or set `MPOWER_BENCHMARK_PYTHON`), then:

```bash
bash scripts/submit_mpower.sh frozen_benchmark --smoke-test
MPOWER_MODEL=whisper bash scripts/submit_mpower.sh frozen_benchmark
```

Defaults request 2 CPUs and 16 GB, with no GPU. When GPU capacity is available,
set `SBATCH_PARTITION` to the institution's GPU partition and `SBATCH_GRES=gpu:1` before submission. `device:
auto` selects CUDA and enables float16 autocast only on CUDA. Benchmark jobs do
not submit or cancel EDA jobs. Avoid scheduling a full benchmark until capacity
is known and all model smoke checks pass.

## Data, preprocessing, and leakage

`configs/frozen_benchmark.yaml` controls checkpoint revisions, segments, batch
size, layer selection, pooling, participant aggregation, heads, and QC. Cohort B
is the default. Canonical `train/val/test_manifest.csv` files are reused, with
participant/recording/hash/near-duplicate leakage checks from the EDA splitter.
The current EDA assignments remain provisional until the full duplicate and QC
review is complete. An explicit CSV/Parquet `manifest` relative to EDA storage
may be supplied. Required columns: `participant_id`, `recording_id`, `audio_path`,
`diagnosis` (`pd` or `control`); optional `split` (`train`, `val`, `test`).
Absent cohort files, the canonical recording manifest is used. No diagnosis is
inferred from medication or invented from raw Voice Activity rows. Run existing
EDA diagnosis linkage if no labeled canonical manifest exists.

When EDA file hashes are absent, source files are hashed before splitting or
validating existing assignments; identical bytes cannot cross partitions under
different recording IDs. Existing audited hashes are reused. This can require
substantial I/O for a large unaudited manifest.

When splits are absent, the existing duplicate-component-aware splitter makes
fixed-seed **provisional** assignments. Conflicting labels, duplicate recording
IDs, partial split assignments and class-empty partitions fail explicitly.
Smoke subsampling preserves existing split assignments. The consumed manifest
is snapshotted with a digest. Model comparison tables include that digest so
runs on different EDA snapshots can be identified.

FFmpeg decodes mono 16 kHz without amplitude normalization; nonfinite or overly
long recordings fail rather than being silently truncated. Minimum duration and
available QC flags filter recordings. Five-second segments with five-second hops
are the defaults; partial tails below minimum duration are dropped. The maximum
number of segments bounds work per recording. Maximum permitted configured
segment duration is 10 seconds, within all checkpoint input limits.

All frontends are loaded from their checkpoints. AST uses its native normalized
128-bin log-Mel frontend, fixed 1024-frame input, and spatial patch tokens with
CLS/distillation and trailing padded patches excluded from pooling. Whisper
uses its native 80-bin log-Mel frontend, 30-second padding and frozen encoder;
trailing encoder padding tokens are excluded from pooling. Wav2vec2 uses its
native normalized raw waveform frontend without an attention mask for this
group-norm checkpoint. Its forward calls use individual unpadded waveforms to
avoid batch-padding effects; AST and Whisper use configurable batches.

Mean and statistical (mean plus population standard deviation) pooling operate
on valid tokens concatenated across each recording's segments. `attention`
learns a token scorer with an auxiliary binary linear head on training recordings,
selects an epoch by validation loss, and supplies pooled embeddings to both
classifier heads. Its token normalization uses training tokens only. The scorer,
statistics, and validation log are saved. Participant probabilities aggregate
recording probabilities by configurable `mean` or `median`. All metrics use a
fixed 0.5 decision threshold; no test-set threshold or calibration fitting occurs.

## Outputs and reuse

```text
$MPOWER_BENCHMARK_ROOT/
  models/                     Hugging Face cache + resolved model provenance
  segments/                   content-addressed, decoded segment waveform NPZs
  activations/<model>/<hash>/  per-segment hidden-state NPZs for configured layers
  reports/                    execution status, including failures
  runs/<timestamp>_<model>_<smoke|baseline>/
    manifest.csv              immutable input and split snapshot
    provenance.json           resolved revision, package versions, manifest digest
    hyperparameters.json
    activation_index.json     recording -> segment activation/input paths
    recording_embeddings.npy  rows aligned with manifest.csv
    attention_pool.pt         only when learned attention pooling is selected
    comparison.csv
    logistic/ and mlp/
      head.joblib             fitted scaler and trained classifier
      training_log.json
      recording_predictions.csv
      participant_predictions.csv
      metrics.json
  comparison.csv              aggregate completed baseline runs
```

Both levels report AUROC, AUPRC, F1, balanced accuracy, sensitivity, specificity,
Brier score, log loss, ten-bin ECE and reliability bins for every split. Undefined
single-class AUROC/AUPRC are null; training requires both classes in each split.
Head and pooling checkpoint files are trusted local artifacts; load them only
from a trusted benchmark run.

Segment keys hash source file contents and preprocessing; activation keys also
include resolved model config, feature extractor, precision/device, library
versions, implementation, and layer selection. Atomic writes permit interrupted
extraction to resume. Tokens load lazily from disk, bounded by one recording;
select only needed layers to control disk use. Waveform segments are reusable
across models. Set `layers: [0, 3, 6, -1]` for intermediate-layer probing; the last
listed layer feeds the current head. PCA/SAE transforms must be fitted on
training participants only. Requesting new layers requires encoder inference
but can reuse decoded segments. Currently all-layer hidden states are requested
from the model, but only selected layers are persisted.

Seeds, deterministic PyTorch algorithms and fixed checkpoint provenance support
repeatability within a compatible environment. For long-term reproducibility,
replace each `revision: main` with the resolved commit recorded in provenance.
Activations/heads require the same pinned revisions for inference. Fine-tuning
should be implemented as a separate mode with fresh caches, unfreezing only
explicitly selected layers and preserving the same participant split and
training-only transforms.

## Verification

```bash
python -m pytest -q tests/test_frozen_benchmark.py
python scripts/run_frozen_benchmark.py --data-smoke --config configs/frozen_benchmark.yaml
```

Unit tests use clearly marked synthetic fixtures and tiny randomly initialized
HF encoders, never participant data in Git. They check preprocessing/input
shapes, freeze/eval state, batch invariance, segmentation/resampling, train-only
normalization, duplicate-aware splits, probability aggregation, calibration
metrics, attention training and extraction cache reuse. Real pretrained smoke
validation is a separate CLI invocation and must pass for each model before
claiming a working pretrained pipeline. See `FROZEN_BENCHMARK_STATUS.md` for the
actual execution results and remaining infrastructure blockers.

To opt into the real-recording architecture integration tests:

```bash
MPOWER_RUN_REAL_SMOKE=1 python -m pytest -q tests/test_frozen_real_smoke.py
```

These use tiny **random** encoders with real HF frontends, six real recordings
per model, and both heads. Artifacts are explicitly named `test-only/random-*`
and stored under `random_architecture_checks/`. They verify data-to-artifact
integration without claiming pretrained behavior or predictive performance.

Frontend references: [AST documentation](https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer),
[Whisper documentation](https://huggingface.co/docs/transformers/model_doc/whisper),
and [the wav2vec2 checkpoint frontend](https://huggingface.co/facebook/wav2vec2-base-960h/blob/main/preprocessor_config.json).
