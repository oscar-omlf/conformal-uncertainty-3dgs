#!/bin/bash
set -eu

ITERS="${ITERS:-30000}"
ALPHA="${ALPHA:-0.1}"
SCENE="${SCENE:-tandt/truck}"
OUTPUT_BASE="${OUTPUT_BASE:-/scratch-shared/$USER/output}"
OUT="${OUT:-$OUTPUT_BASE/basic_tandt_truck_${ITERS}}"

sbatch --export=ALL,SCENE="$SCENE",OUT="$OUT",ITERS="$ITERS",ALPHA="$ALPHA" snellius_jobs/01_full_pipeline.job
