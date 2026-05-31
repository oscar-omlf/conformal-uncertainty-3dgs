#!/bin/bash
set -eu

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <dataset> <start_round> [method]"
    echo "Example: $0 counter 10 conformal_visibility"
    exit 1
fi

DATASET="$1"
START_ROUND="$2"
METHOD="${3:-conformal_visibility}"

RUN_NAME="${DATASET}_popgs20"
SCENE="mipnerf/${DATASET}"
AL_ROOT="/scratch-shared/$USER/output/active_learning_mipnerf/${RUN_NAME}"

echo "Submitting resume:"
echo "  dataset     = $DATASET"
echo "  scene       = $SCENE"
echo "  al_root     = $AL_ROOT"
echo "  method      = $METHOD"
echo "  start_round = $START_ROUND"

sbatch --export=ALL,SCENE="$SCENE",AL_ROOT="$AL_ROOT",METHOD="$METHOD",PRESET=popgs20,START_ROUND="$START_ROUND" \
    snellius_jobs/02_active_learning_loop.job