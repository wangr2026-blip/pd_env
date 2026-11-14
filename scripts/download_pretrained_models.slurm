#!/usr/bin/env bash
#SBATCH --job-name=mpower_models
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=04:00:00
#SBATCH --output=logs/mpower_models_%j.out
#SBATCH --error=logs/mpower_models_%j.err
# Use a partition with outbound HTTPS; no GPU, torch or participant data needed.
set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${MPOWER_MODEL_ROOT:?Set MPOWER_MODEL_ROOT to the shared public model directory}"
PYTHON_BIN="${MPOWER_DOWNLOAD_PYTHON:-python3}"
exec "$PYTHON_BIN" scripts/download_pretrained_models.py --public-only --workers 2 "$@"
