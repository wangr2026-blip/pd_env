#!/usr/bin/env bash
#SBATCH --job-name=mpower_linked_pilot
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=logs/mpower_linked_%j.out
#SBATCH --error=logs/mpower_linked_%j.err
set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?}"
: "${MPOWER_OUTPUT_ROOT:?}"
ARCHIVE="$MPOWER_OUTPUT_ROOT/archive/voice_only_pilot"
if [[ ! -e "$ARCHIVE" ]]; then
    mkdir -p "$ARCHIVE"
    cp -a "$MPOWER_OUTPUT_ROOT/eda" "$ARCHIVE/eda"
    mkdir -p "$ARCHIVE/processed"
    cp -a "$MPOWER_OUTPUT_ROOT"/processed/pilot_* "$ARCHIVE/processed/"
fi
export MPOWER_SKIP_AUDIT=1
export MPOWER_SCOPE=pilot
bash scripts/preprocess_mpower.slurm
