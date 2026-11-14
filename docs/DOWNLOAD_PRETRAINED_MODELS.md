# Download the pretrained models on another machine

This guide reproduces the nine public checkpoint downloads verified in the
local checkout. It downloads **12.02 GB** of weights and supporting files at
the revisions in `configs/pretrained_models.lock.json`. Have at least **20 GB
free** in the destination. Downloading needs Python 3.10 or newer and curl;
Python 3.12 is the project's tested HPC version. No GPU, PyTorch installation,
Hugging Face account or participant recordings are needed for this step.

HeAR is excluded because access is pending. Do not use `--refresh-lock` when
reproducing this download: it would select newer upstream revisions.

## 1. Get the code

For an existing checkout, open a terminal **in that checkout's `pd_env`
directory**, then run:

```bash
git status --short
git pull --ff-only origin main
```

Preserve any local changes if Git reports a conflict. Do not reset or delete
the existing checkout to make the pull succeed.

For a new checkout, run this instead from the parent projects directory:

```bash
git clone https://github.com/wangr2026-blip/pd_env.git
cd pd_env
```

All commands below assume the current directory is this repository root.

## 2. Select Python and persistent storage

On the HPC, activate the existing environment:

```bash
conda activate pd_env
python3 --version
command -v curl
```

On another computer, an existing Python 3.10+ installation is sufficient;
Conda activation is optional. There is no pip install for the downloader.

These paths follow the documented sibling `pd_env` / `pd_data` layout. If your
institution uses another approved persistent location, change the exports
before running them. Do not point `MPOWER_MODEL_ROOT` at raw recording storage
or temporary job scratch.

```bash
export PD_PROJECTS_ROOT="$(dirname "$(pwd -P)")"
export MPOWER_BENCHMARK_ROOT="$PD_PROJECTS_ROOT/pd_data/mpower_benchmarks"
export MPOWER_MODEL_ROOT="$MPOWER_BENCHMARK_ROOT/models"
export MPOWER_SLURM_ROOT="$PD_PROJECTS_ROOT/pd_data/slurm"
export MPOWER_DOWNLOAD_PYTHON="$(command -v python3)"

mkdir -p "$MPOWER_MODEL_ROOT"
df -h "$MPOWER_MODEL_ROOT"
"$MPOWER_DOWNLOAD_PYTHON" -c 'import sys; assert sys.version_info >= (3, 10); print(sys.executable)'
```

Keep these exports in the current terminal. A new terminal needs the same
values again, or your existing ignored `.env` can store them. The downloader
does not require `MPOWER_DATA_ROOT` or `MPOWER_OUTPUT_ROOT` and does not read
participant data.

## 3. Check the plan without downloading

```bash
"$MPOWER_DOWNLOAD_PYTHON" scripts/download_pretrained_models.py --public-only --plan-only
```

The output must say `Pinned assets: 12.018 GB`, name the directory selected in
step 2, and list **nine** aliases with `gated=False`:

| Alias | Upstream checkpoint | Approximate GB |
|---|---|---:|
| `ast_audioset` | `MIT/ast-finetuned-audioset-10-10-0.4593` | 0.346 |
| `wav2vec2_base_960h` | `facebook/wav2vec2-base-960h` | 0.378 |
| `wav2vec2_large_robust` | `facebook/wav2vec2-large-robust` | 1.270 |
| `wavlm_base_plus` | `microsoft/wavlm-base-plus` | 0.378 |
| `wavlm_large` | `microsoft/wavlm-large` | 1.262 |
| `hubert_large` | `facebook/hubert-large-ll60k` | 1.262 |
| `whisper_small` | `openai/whisper-small` | 0.971 |
| `whisper_medium` | `openai/whisper-medium` | 3.060 |
| `whisper_large_v3` | `openai/whisper-large-v3` | 3.092 |

The total includes weights and frontend/tokenizer/configuration files. Only
one PyTorch weight format is downloaded per checkpoint. Large-v3 is stored
in half precision upstream, so its similar disk size to Medium does not mean
the two models have similar parameter counts.

## 4. Download using one of these routes

### Route A: a network-enabled login node or computer

Use this route where local policy allows long downloads. Both
`huggingface.co` and its redirected file hosts must be reachable over HTTPS.

```bash
"$MPOWER_DOWNLOAD_PYTHON" scripts/download_pretrained_models.py --public-only --workers 2
```

Wait for the command to finish. Success produces a `VERIFIED <alias>` line
for every checkpoint and exits with status zero. Transfers can take time;
do not close the terminal while they are running.

To resume after an interruption, run **the identical command again**. Existing
files are hash-checked, partial files resume from their received bytes, and
only missing or invalid assets are downloaded. Do not run two download jobs
against the same destination at once.

For an initial single-model check, use:

```bash
"$MPOWER_DOWNLOAD_PYTHON" scripts/download_pretrained_models.py --public-only --models ast_audioset
```

Then run the all-model command above; it will reuse AST.

### Route B: a Slurm node with outbound internet

Use this route only if that partition permits outbound HTTPS. From the same
repository-root terminal with the step-2 variables still exported:

```bash
model_job="$(bash scripts/submit_mpower.sh download_pretrained_models)"
model_job="${model_job%%;*}"
echo "$model_job"
squeue -j "$model_job"
```

If your cluster requires a specific partition or account, set its real values
using `SBATCH_PARTITION` and `SBATCH_ACCOUNT` before submission. The template
requests two CPUs, 2 GB RAM and four hours, with no GPU. It uses the absolute
Python executable selected in step 2. This guide has not submitted a job.

Check completion and logs:

```bash
sacct -j "$model_job" --format=JobID,JobName,State,ExitCode,Elapsed
cat "$MPOWER_SLURM_ROOT/logs/mpower_models_${model_job}.out"
cat "$MPOWER_SLURM_ROOT/logs/mpower_models_${model_job}.err"
```

The batch step should show `COMPLETED` and `0:0`. Submitted template snapshots
are under `$MPOWER_SLURM_ROOT/jobs`. A timed-out or failed download is resumable
by submitting the same command again. Confirm that the earlier job has stopped
before starting its replacement.

### Route C: download elsewhere and transfer to HPC

If the HPC has no permitted internet connection, complete Route A on a machine
that does. Copy its **entire `checkpoints/` directory**, including configuration
files and per-model provenance, to the HPC model directory selected in step 2.
For example, replace both paths and `your_user@your_hpc_login` below with your
actual source and destination:

```bash
rsync -av --partial /absolute/source/models/checkpoints/ your_user@your_hpc_login:/absolute/hpc/models/checkpoints/
```

Create the destination model directory first on the HPC. Run step 5 there using
the same Git checkout and pinned lock. This transfer contains public model
assets; there is no reason to transfer participant recordings or metadata.

## 5. Verify on the machine that will run the models

This command makes no network requests when the pinned lock exists. It checks
file sizes and hashes, rather than relying on filenames or the existence of
directories:

```bash
"$MPOWER_DOWNLOAD_PYTHON" scripts/download_pretrained_models.py --public-only --verify-only
```

It must print nine `VERIFIED` lines and exit successfully. To check status:

```bash
"$MPOWER_DOWNLOAD_PYTHON" -m json.tool "$MPOWER_MODEL_ROOT/download_status.json"
```

Each public entry should have `"status": "verified"`, its pinned revision,
byte count and file count. A previous failed HeAR entry can remain in this
file; `--public-only` excludes it from the verification result.

Expected structure:

```text
$MPOWER_MODEL_ROOT/
  download_status.json
  checkpoints/
    ast_audioset/
    wav2vec2_base_960h/
    wav2vec2_large_robust/
    wavlm_base_plus/
    wavlm_large/
    hubert_large/
    whisper_small/
    whisper_medium/
    whisper_large_v3/
```

Every model directory has its original configuration/frontend assets, one
weight format (`model.safetensors` or `pytorch_model.bin`) and
`download_provenance.json`. A `.part` file alone is an incomplete transfer.

## 6. Use the local files in model code

Model loading requires the benchmark Python/PyTorch/Transformers environment;
that is separate from downloading. Point `from_pretrained` at the **local
checkpoint directory**, for example
`$MPOWER_MODEL_ROOT/checkpoints/whisper_small`, and use
`local_files_only=True`. Export `MPOWER_MODEL_ROOT` in the Slurm job as well.
Use each model's own processor and architecture class. See the
[input and local-loading guide](PRETRAINED_MODELS_AND_INPUTS.md) for an example.

The runner now supports all nine public checkpoints through
`configs/frozen_benchmark_local.yaml`, which is its default configuration.
It resolves pinned aliases under `$MPOWER_MODEL_ROOT/checkpoints/` and loads
locally. WavLM/HuBERT adapters, checkpoint-specific normalization and masks,
and valid-token pooling are implemented. See the
[local benchmark guide](LOCAL_PRETRAINED_BENCHMARK.md) for setup and smoke-test
commands. Asset verification and inference validation remain separate checks.

## If a step fails

- **`--public-only` is unrecognized or the lock is missing:** confirm that the
  new commit was pulled and that the terminal is in the repository root.
- **DNS/connection errors:** try a permitted network-enabled login node or
  Route C. More retries cannot fix a node whose firewall blocks outbound HTTPS.
- **Interrupted transfer or timeout:** rerun the same download command; keep
  the `.part` files so it can resume.
- **Hash verification fails:** rerun the download command without
  `--verify-only`. The downloader replaces an invalid final file after fetching
  and verifying a valid copy.
- **Slurm reports a partition/account/time-limit error:** use your cluster's
  permitted `SBATCH_*` settings. The template's resource request is a default.
- **Model code tries to contact Hugging Face after verification:** it is
  probably still using a remote model ID or another cache directory. Supply
  the local checkpoint path and `local_files_only=True`.
- **HeAR returns 401/403:** leave it excluded. Public-model downloads use
  `--public-only` and do not need credentials. Never put tokens in this guide,
  Git, a submitted template or chat.
