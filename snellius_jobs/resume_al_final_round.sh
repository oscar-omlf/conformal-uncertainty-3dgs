#!/bin/bash
set -eu

# Submit a Slurm job that resumes only the final post-training stage of an AL run.
#
# Required:
#   SCENE=mipnerf/bonsai
#   AL_ROOT=/scratch-shared/$USER/output/active_learning_mipnerf/bonsai_popgs20_camera
#   METHOD=conformal_visibility
#
# Optional:
#   SEED=0 ROUND=16 ALPHA=0.1 FINAL_ITERS=21000

if [ -z "${SCENE:-}" ]; then
    echo "ERROR: set SCENE=mipnerf/<dataset>"
    exit 1
fi
if [ -z "${AL_ROOT:-}" ]; then
    echo "ERROR: set AL_ROOT=/scratch-shared/\$USER/output/active_learning_mipnerf/<run_name>"
    exit 1
fi
if [ -z "${METHOD:-}" ]; then
    echo "ERROR: set METHOD=uniform|conformal_color|conformal_visibility|conformal_sensitivity|raw_sensitivity"
    exit 1
fi

SEED="${SEED:-0}"
ROUND="${ROUND:-16}"
ALPHA="${ALPHA:-0.1}"
FINAL_ITERS="${FINAL_ITERS:-21000}"

echo "Submitting final-round resume:"
echo "  SCENE=$SCENE"
echo "  AL_ROOT=$AL_ROOT"
echo "  METHOD=$METHOD"
echo "  SEED=$SEED ROUND=$ROUND FINAL_ITERS=$FINAL_ITERS"

sbatch --export=ALL,SCENE="$SCENE",AL_ROOT="$AL_ROOT",METHOD="$METHOD",SEED="$SEED",ROUND="$ROUND",ALPHA="$ALPHA",FINAL_ITERS="$FINAL_ITERS" \
    snellius_jobs/resume_al_final_round.job
