#!/bin/bash
set -eu

# Resume only the final post-training part of an active-learning run.
# Use this when round_16 training finished but the job failed during final
# rendering, conformal metrics, LPIPS, aggregation, or plotting.
#
# Required:
#   SCENE=mipnerf/bonsai
#   AL_ROOT=/scratch-shared/$USER/output/active_learning_mipnerf/bonsai_popgs20_camera
#   METHOD=conformal_visibility
#
# Optional:
#   SEED=0
#   ROUND=16
#   ALPHA=0.1
#   FINAL_ITERS=21000

if [ -z "${SCENE:-}" ]; then
    echo "ERROR: set SCENE=/path/to/scene, e.g. SCENE=mipnerf/bonsai"
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
ROUND_PAD="$(printf "%02d" "$ROUND")"
ALPHA="${ALPHA:-0.1}"
FINAL_ITERS="${FINAL_ITERS:-21000}"
TARGET_COVERAGE="$(python -c "print(1.0 - float('${ALPHA}'))")"

RUN_ROOT="$AL_ROOT/$METHOD/seed_${SEED}"
ROUND_OUT="$RUN_ROOT/round_${ROUND_PAD}"
SPLIT_DIR="$ROUND_OUT/splits"
OUTPUT_ROOT="$(dirname "$ROUND_OUT")"

if [ ! -d "$ROUND_OUT/point_cloud" ]; then
    echo "ERROR: trained point_cloud not found under $ROUND_OUT"
    echo "This script expects final-round training to have completed."
    exit 1
fi
if [ ! -d "$SPLIT_DIR" ]; then
    echo "ERROR: split dir not found: $SPLIT_DIR"
    exit 1
fi

NEED_COLOR=0
NEED_SENSITIVITY=0
NEED_VISIBILITY=0
case "$METHOD" in
    random|uniform)
        ;;
    conformal_color|raw_color|color)
        NEED_COLOR=1
        ;;
    conformal_visibility|raw_visibility|visibility)
        NEED_VISIBILITY=1
        ;;
    conformal_sensitivity|raw_sensitivity|sensitivity|fisher|pup)
        NEED_SENSITIVITY=1
        ;;
    conformal_combined|raw_combined|combined)
        NEED_COLOR=1
        NEED_SENSITIVITY=1
        NEED_VISIBILITY=1
        ;;
    *)
        echo "ERROR: unknown METHOD=$METHOD"
        exit 1
        ;;
esac

module purge
module load 2025
module load Anaconda3/2025.06-1
module load CUDA/12.9.1
export CUDA_HOME=$CUDA_ROOT
export TORCH_CUDA_ARCH_LIST="8.0"

cd $HOME/conformal-uncertainty-3dgs

set +u
eval "$(conda shell.bash hook)"
conda activate gaussian_splatting
set -u

echo "================================================================"
echo "Resume AL final round"
echo "Scene      : $SCENE"
echo "AL root    : $AL_ROOT"
echo "Method     : $METHOD"
echo "Round      : $ROUND_PAD"
echo "Round out  : $ROUND_OUT"
echo "Signals    : color=$NEED_COLOR sensitivity=$NEED_SENSITIVITY visibility=$NEED_VISIBILITY"
echo "================================================================"

echo "[round $ROUND_PAD] Rendering RGB test views for final image metrics"
python render.py -m "$ROUND_OUT" --split_dir "$SPLIT_DIR" --iteration "$FINAL_ITERS" --skip_train --skip_calib --skip_candidate

if [ "$NEED_COLOR" -eq 1 ]; then
    echo "[round $ROUND_PAD] Rendering final color sigma for calib+test"
    python scripts/render_color.py \
        -m "$ROUND_OUT" \
        --split_dir "$SPLIT_DIR" \
        --iteration "$FINAL_ITERS" \
        --output_root "$OUTPUT_ROOT" \
        --skip_train \
        --skip_candidate
fi

if [ "$NEED_SENSITIVITY" -eq 1 ]; then
    echo "[round $ROUND_PAD] Computing/rendering final sensitivity sigma for calib+test"
    python scripts/compute_sensitivity.py -m "$ROUND_OUT" --iteration "$FINAL_ITERS"
    python scripts/render_per_gaussian_scalar.py \
        -m "$ROUND_OUT" \
        --split_dir "$SPLIT_DIR" \
        --iteration "$FINAL_ITERS" \
        --output_root "$OUTPUT_ROOT" \
        --skip_train \
        --skip_candidate \
        --modality sensitivity \
        --scores_path "$ROUND_OUT/uncertainty/fishers.npz" \
        --score_key fishers_log_dets \
        --score_transform shift_positive
fi

if [ "$NEED_VISIBILITY" -eq 1 ]; then
    echo "[round $ROUND_PAD] Computing/rendering final visibility sigma for calib+test"
    python scripts/compute_visibility.py -m "$ROUND_OUT" --iteration "$FINAL_ITERS"
    python scripts/render_per_gaussian_scalar.py \
        -m "$ROUND_OUT" \
        --split_dir "$SPLIT_DIR" \
        --iteration "$FINAL_ITERS" \
        --output_root "$OUTPUT_ROOT" \
        --skip_train \
        --skip_candidate \
        --modality visibility \
        --scores_path "$ROUND_OUT/uncertainty/visibility.npz" \
        --score_key uncertainty_log
fi

if [ "$NEED_COLOR" -eq 1 ]; then
    python scripts/conformal_prediction.py --run_dir "$ROUND_OUT" --iteration "$FINAL_ITERS" --modality color --sigma_norm minmax_calib --alpha "$ALPHA" --calib_sample_ratio 0.1
fi
if [ "$NEED_SENSITIVITY" -eq 1 ]; then
    python scripts/conformal_prediction.py --run_dir "$ROUND_OUT" --iteration "$FINAL_ITERS" --modality sensitivity --sigma_norm minmax_calib --alpha "$ALPHA" --calib_sample_ratio 0.1
fi
if [ "$NEED_VISIBILITY" -eq 1 ]; then
    python scripts/conformal_prediction.py --run_dir "$ROUND_OUT" --iteration "$FINAL_ITERS" --modality visibility --sigma_norm minmax_calib --alpha "$ALPHA" --calib_sample_ratio 0.1
fi

echo "[round $ROUND_PAD] Final image metrics with LPIPS"
python metrics.py -m "$ROUND_OUT"

if [ "$NEED_COLOR" -eq 1 ] || [ "$NEED_SENSITIVITY" -eq 1 ] || [ "$NEED_VISIBILITY" -eq 1 ]; then
    echo "[round $ROUND_PAD] Final conformal aggregation"
    python scripts/aggregate_results.py --run_dir "$ROUND_OUT"
fi

echo "[done] Aggregating active-learning root"
python scripts/active_learning/aggregate.py --al_root "$AL_ROOT"

echo "[done] Plotting active-learning curves"
python scripts/active_learning/plot.py --al_root "$AL_ROOT" --coverage_target "$TARGET_COVERAGE"

echo "================================================================"
echo "Final-round resume complete: $ROUND_OUT"
echo "================================================================"
