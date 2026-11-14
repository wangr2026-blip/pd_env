#!/usr/bin/env bash
#SBATCH --job-name=mpower_frozen_development
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
set -euo pipefail
umask 077
cd "${MPOWER_CODE_ROOT:?}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
: "${MPOWER_PROTOCOL_ROOT:?}"
mapfile -t task < <("${MPOWER_BENCHMARK_PYTHON:?}" -c 'import os,yaml; c=yaml.safe_load(open("configs/frozen_model_testing.yaml")); tasks=[(m,k) for m in c["models"] for k in c["protocol"]["contexts"]]; print("\n".join(tasks[int(os.environ["SLURM_ARRAY_TASK_ID"])]))')
"$MPOWER_BENCHMARK_PYTHON" -u scripts/run_frozen_development.py \
  --config configs/frozen_model_testing.yaml --snapshot "$MPOWER_PROTOCOL_ROOT/snapshot" --model "${task[0]}" --context "${task[1]}"
