#!/usr/bin/env bash
#SBATCH --job-name=mpower_full_phase1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=45-00:00:00
# defq maximum: --time=0 was accepted but left the job pending for PartitionTimeLimit.
# Override with SBATCH_TIMELIMIT if the target cluster has a different maximum.
#SBATCH --output=logs/mpower_full_%j.out
#SBATCH --error=logs/mpower_full_%j.err
set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?}"
: "${MPOWER_OUTPUT_ROOT:?}"
: "${MPOWER_PYTHON:?}"
"$MPOWER_PYTHON" -c 'import json,os,pathlib; p=pathlib.Path(os.environ["MPOWER_OUTPUT_ROOT"])/"eda/diagnostics/pilot_validation.json"; d=json.loads(p.read_text()); assert d["all_repeated_reproducible"] and not d["repeat_errors"], "Pilot repeat validation did not pass"'
ARCHIVE="$MPOWER_OUTPUT_ROOT/archive/linked_pilot"
if [[ ! -e "$ARCHIVE" ]]; then
    mkdir -p "$ARCHIVE"
    cp -a "$MPOWER_OUTPUT_ROOT/eda" "$ARCHIVE/eda"
    cp -a "$MPOWER_OUTPUT_ROOT/processed" "$ARCHIVE/processed"
fi
export MPOWER_SKIP_AUDIT=1
export MPOWER_SCOPE=full
bash scripts/preprocess_mpower.slurm
