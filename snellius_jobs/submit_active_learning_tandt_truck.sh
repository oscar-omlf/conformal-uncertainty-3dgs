#!/bin/bash
set -eu

ITERS="${ITERS:-30000}"
ALPHA="${ALPHA:-0.1}"
SEED="${SEED:-0}"
ROUNDS="${ROUNDS:-5}"
INIT_TRAIN="${INIT_TRAIN:-10%}"
CALIB="${CALIB:-10%}"
TEST="${TEST:-20%}"
ADD_K="${ADD_K:-5}"
METHODS="${METHODS:-conformal_color conformal_visibility conformal_sensitivity raw_sensitivity}"
SCENE="${SCENE:-tandt/truck}"
OUTPUT_BASE="${OUTPUT_BASE:-/scratch-shared/$USER/output}"
AL_ROOT="${AL_ROOT:-$OUTPUT_BASE/active_learning/tandt_truck}"

for method in $METHODS; do
    echo "Submitting scene=$SCENE method=$method seed=$SEED"
    sbatch --export=ALL,SCENE="$SCENE",AL_ROOT="$AL_ROOT",METHOD="$method",SEED="$SEED",ROUNDS="$ROUNDS",INIT_TRAIN="$INIT_TRAIN",CALIB="$CALIB",TEST="$TEST",ADD_K="$ADD_K",ITERS="$ITERS",ALPHA="$ALPHA" snellius_jobs/02_active_learning_loop.job
done
