#!/usr/bin/env bash
# Snapshot a versioned Slurm template and keep job files/logs in private storage.
set -euo pipefail
umask 077
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
: "${MPOWER_SLURM_ROOT:?Set MPOWER_SLURM_ROOT to private HPC job storage outside Git}"
if [[ $# -lt 1 ]]; then
    echo 'Usage: bash scripts/submit_mpower.sh TEMPLATE_NAME [script arguments...]' >&2
    exit 2
fi
TEMPLATE="$1"
shift
case "$TEMPLATE" in
    download_mpower|download_mpower_demographics|preprocess_mpower|linked_pilot_mpower|full_mpower|frozen_benchmark) ;;
    *) echo "Unknown Slurm template: $TEMPLATE" >&2; exit 2 ;;
esac
# Resolve symlinks before validating; never create job outputs inside the checkout.
JOB_ROOT="$(realpath -m -- "$MPOWER_SLURM_ROOT")"
case "$JOB_ROOT/" in
    "$REPO_ROOT/"*|//) echo 'MPOWER_SLURM_ROOT must be outside the repository and not /' >&2; exit 2 ;;
esac
mkdir -p -- "$JOB_ROOT/jobs" "$JOB_ROOT/logs"
SNAPSHOT="$(mktemp "$JOB_ROOT/jobs/${TEMPLATE}_XXXXXXXX.slurm")"
cp -- "$REPO_ROOT/scripts/$TEMPLATE.slurm" "$SNAPSHOT"
# Scheduler options may be supplied through Slurm's standard SBATCH_*
# environment variables.
sbatch --parsable --chdir="$REPO_ROOT" \
    --output="$JOB_ROOT/logs/%x_%j.out" \
    --error="$JOB_ROOT/logs/%x_%j.err" "$SNAPSHOT" "$@"
