#!/usr/bin/env bash
#SBATCH --job-name=mpower_demographics
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/mpower_demographics_%j.out
#SBATCH --error=logs/mpower_demographics_%j.err
set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?}"
: "${MPOWER_DATA_ROOT:?Set approved HPC raw storage}"
: "${MPOWER_PYTHON:?Set Python with existing synapseclient installed}"
TOKEN_FILE="$HOME/.config/mpower/synapse_token"
if [[ ! -s "$TOKEN_FILE" ]]; then
    echo 'Missing configured Synapse token file' >&2
    exit 1
fi
export SYNAPSE_AUTH_TOKEN="$(cat "$TOKEN_FILE")"
trap 'unset SYNAPSE_AUTH_TOKEN' EXIT
"$MPOWER_PYTHON" -u scripts/download_mpower_demographics.py --authorized
