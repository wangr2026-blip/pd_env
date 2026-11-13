#!/usr/bin/env bash
#SBATCH --job-name=mpower_phase1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=logs/mpower_phase1_%j.out
#SBATCH --error=logs/mpower_phase1_%j.err
set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?}"
: "${MPOWER_DATA_ROOT:?Set approved raw data root}"
: "${MPOWER_OUTPUT_ROOT:?Set approved private derived output root outside repository}"
: "${MPOWER_PYTHON:?Set compatible Python executable}"
# Optional immutable wheel environment archive staged by the submitting session.
if [[ -n "${MPOWER_ENV_ARCHIVE:-}" ]]; then
    JOB_ENV="${SLURM_TMPDIR:-/tmp}/mpower_phase1_${SLURM_JOB_ID}"
    mkdir -p "$JOB_ENV"
    tar -xzf "$MPOWER_ENV_ARCHIVE" -C "$JOB_ENV"
    MPOWER_PYTHON="$JOB_ENV/mpower-phase1-py312/bin/python"
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$MPOWER_OUTPUT_ROOT/cache/matplotlib"
export NUMBA_CACHE_DIR="$MPOWER_OUTPUT_ROOT/cache/numba"
export JUPYTER_RUNTIME_DIR="$MPOWER_OUTPUT_ROOT/cache/jupyter"
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$JUPYTER_RUNTIME_DIR"
export PATH="$(dirname "$MPOWER_PYTHON"):$PATH"
CONFIG="${MPOWER_CONFIG:-configs/mpower_preprocessing.yaml}"
SCOPE="${MPOWER_SCOPE:-pilot}"
if [[ "$SCOPE" != pilot && "$SCOPE" != full ]]; then
    echo 'MPOWER_SCOPE must be pilot or full' >&2
    exit 2
fi
# Census is reusable only when explicitly requested by operator after checking raw inputs.
if [[ "${MPOWER_SKIP_AUDIT:-0}" != 1 ]]; then
    "$MPOWER_PYTHON" -u scripts/audit_metadata.py --config "$CONFIG"
fi
"$MPOWER_PYTHON" -u scripts/link_demographics.py --config "$CONFIG"
if [[ "$SCOPE" == full ]]; then
    "$MPOWER_PYTHON" -u scripts/inspect_audio.py --config "$CONFIG" --stage features --full --workers "${SLURM_CPUS_PER_TASK:-2}"
else
    "$MPOWER_PYTHON" -u scripts/inspect_audio.py --config "$CONFIG" --stage features --workers "${SLURM_CPUS_PER_TASK:-2}"
fi
"$MPOWER_PYTHON" -u scripts/build_cohorts.py --config "$CONFIG" --scope "$SCOPE"
"$MPOWER_PYTHON" -u scripts/create_participant_splits.py --config "$CONFIG"
"$MPOWER_PYTHON" -u scripts/report_characterization.py --config "$CONFIG" --scope "$SCOPE"
"$MPOWER_PYTHON" -u scripts/execute_characterization_notebook.py
"$MPOWER_PYTHON" -m pip freeze > "$MPOWER_OUTPUT_ROOT/eda/diagnostics/environment.lock.txt"
