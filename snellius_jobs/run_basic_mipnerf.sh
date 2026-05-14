#!/bin/bash
set -eu

if [ -z "${DATASET_NAME:-}" ]; then
    echo "ERROR: set DATASET_NAME, e.g. DATASET_NAME=garden ./snellius_jobs/run_basic_mipnerf.sh"
    exit 1
fi

ITERS="${ITERS:-30000}"
ALPHA="${ALPHA:-0.1}"
SCENE="${SCENE:-mipnerf/${DATASET_NAME}}"
OUTPUT_BASE="${OUTPUT_BASE:-/scratch-shared/$USER/output}"
OUT="${OUT:-$OUTPUT_BASE/basic_mipnerf_${DATASET_NAME}_${ITERS}}"

sbatch --export=ALL,SCENE="$SCENE",OUT="$OUT",ITERS="$ITERS",ALPHA="$ALPHA" snellius_jobs/01_full_pipeline.job
