#!/usr/bin/env bash
# CPU stage of the interpretability pipeline (Stage 0 now; Stages 1, 3-5 later).
#   sbatch --export=ALL,PIPELINE_STAGE=cohort,PIPELINE_CONFIG=configs/pipeline/mpower_vowel.yaml \
#          scripts/pipeline/cpu_stage.slurm
# Stage 0 with qc_source=precomputed reads existing manifests only; it is light.
#SBATCH --job-name=pd_pipeline_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-06:00:00
set -euo pipefail
umask 077
cd "${PIPELINE_CODE:-$SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
: "${PIPELINE_STAGE:?Set PIPELINE_STAGE (e.g. cohort)}"
: "${PIPELINE_CONFIG:?Set PIPELINE_CONFIG to a configs/pipeline/*.yaml file}"
: "${MPOWER_PYTHON:?Set MPOWER_PYTHON to an environment with requirements-pipeline.txt}"
"$MPOWER_PYTHON" -u -m "pipeline.${PIPELINE_STAGE}" --config "$PIPELINE_CONFIG" ${PIPELINE_ARGS:-}
