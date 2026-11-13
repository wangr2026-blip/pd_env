#!/usr/bin/env bash
#SBATCH --job-name=mpower_frozen
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/mpower_frozen_%j.out
#SBATCH --error=logs/mpower_frozen_%j.err
# CPU-safe defaults. Request --gres=gpu:1 and the local GPU partition at submission
# only when capacity is available; the YAML device:auto then selects CUDA.
set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?}"
: "${MPOWER_DATA_ROOT:?}"
: "${MPOWER_OUTPUT_ROOT:?}"
: "${MPOWER_BENCHMARK_ROOT:?}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export TOKENIZERS_PARALLELISM=false
PYTHON_BIN="${MPOWER_BENCHMARK_PYTHON:-python}"
exec "$PYTHON_BIN" scripts/run_frozen_benchmark.py --model "${MPOWER_MODEL:-all}" --config "${MPOWER_BENCHMARK_CONFIG:-configs/frozen_benchmark.yaml}" "$@"
