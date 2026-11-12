#!/usr/bin/env bash

#SBATCH --job-name=mpower_voice
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/mpower_%j.out
#SBATCH --error=logs/mpower_%j.err

set -euo pipefail
umask 077

cd "${SLURM_SUBMIT_DIR:?}"

: "${MPOWER_DATA_ROOT:?Set MPOWER_DATA_ROOT before sbatch}"
: "${MPOWER_PYTHON:?Set MPOWER_PYTHON before sbatch}"

TOKEN_FILE="$HOME/.config/mpower/synapse_token"

if [[ ! -s "$TOKEN_FILE" ]]; then
    echo "ERROR: Synapse token file missing"
    exit 1
fi

mkdir -p "$MPOWER_DATA_ROOT"

REPO_ROOT="$(pwd -P)"
DATA_ROOT="$(cd "$MPOWER_DATA_ROOT" && pwd -P)"

case "$DATA_ROOT/" in
    "$REPO_ROOT/"*)
        echo "ERROR: Data directory is inside the Git repository"
        exit 1
        ;;
esac

export SYNAPSE_AUTH_TOKEN="$(cat "$TOKEN_FILE")"

trap 'unset SYNAPSE_AUTH_TOKEN' EXIT

echo "======================================"
echo "mPower Voice Download"
echo "======================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Python: ${MPOWER_PYTHON}"
echo "Data root: ${MPOWER_DATA_ROOT}"
echo "Batch size: ${MPOWER_CHUNK:-250}"
echo "Row limit: ${MPOWER_LIMIT:-0}"
echo "Started: $(date)"
echo "======================================"

"$MPOWER_PYTHON" -u scripts/download_mpower.py \
    --out "$MPOWER_DATA_ROOT" \
    --chunk "${MPOWER_CHUNK:-250}" \
    --limit "${MPOWER_LIMIT:-0}"

echo "======================================"
echo "Finished: $(date)"
echo "======================================"
