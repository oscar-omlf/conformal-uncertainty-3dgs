#!/bin/bash
set -eu

if [ -z "${DATASET_NAME:-}" ]; then
    echo "ERROR: set DATASET_NAME, e.g. DATASET_NAME=garden ./snellius_jobs/submit_mipnerf_popgs20.sh"
    exit 1
fi

ALPHA="${ALPHA:-0.1}"
SEED="${SEED:-0}"
CALIB="${CALIB:-10%}"
METHODS="${METHODS:-uniform conformal_color conformal_visibility conformal_sensitivity raw_sensitivity}"
SCENE="${SCENE:-mipnerf/${DATASET_NAME}}"
OUTPUT_BASE="${OUTPUT_BASE:-/scratch-shared/$USER/output}"
AL_ROOT="${AL_ROOT:-$OUTPUT_BASE/active_learning_mipnerf/${DATASET_NAME}_popgs20}"

for method in $METHODS; do
    echo "Submitting MipNeRF POp-GS 20-view scene=$SCENE method=$method seed=$SEED"
    sbatch --export=ALL,SCENE="$SCENE",AL_ROOT="$AL_ROOT",METHOD="$method",SEED="$SEED",PRESET=popgs20,CALIB="$CALIB",ALPHA="$ALPHA" snellius_jobs/02_active_learning_loop.job
done
