#!/usr/bin/env bash
#SBATCH --job-name=mpower_protocol_prepare
#SBATCH --partition=defq
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=45-00:00:00
set -euo pipefail
umask 077
cd "${MPOWER_CODE_ROOT:?}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${MPOWER_PROTOCOL_ROOT:?}"
"${MPOWER_BENCHMARK_PYTHON:?}" -u scripts/prepare_frozen_protocol.py \
  --config configs/frozen_model_testing.yaml --output "$MPOWER_PROTOCOL_ROOT/snapshot" --workers "${SLURM_CPUS_PER_TASK:-16}"

"$MPOWER_BENCHMARK_PYTHON" scripts/protocol_covariate_baselines.py --snapshot "$MPOWER_PROTOCOL_ROOT/snapshot"
