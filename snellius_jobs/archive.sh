#!/bin/bash
set -eu

# Stage a compact archive of MipNeRF active-learning results from Snellius
# scratch, then optionally rsync it somewhere else.
#
# Usage on Snellius:
#   AL_RUN=garden_popgs20 ./snellius_jobs/archive_active_learning_mipnerf.sh
#   AL_RUN=bonsai_popgs20 ./snellius_jobs/archive_active_learning_mipnerf.sh
#
# Optional:
#   OUTPUT_BASE=/scratch-shared/$USER/output
#   AL_ROOT=$OUTPUT_BASE/active_learning_mipnerf/<run_name>
#   DATASET_NAME=garden RUN_SUFFIX=popgs20  # convenience alternative to AL_RUN
#   ARCHIVE_BASE=$HOME/al_archives
#   PREVIEW_FRAMES="00000 00008"  # test-frame stems to keep across rounds
#   DEST=""   # intentionally empty by default; set to rsync destination if useful
#
# Note: to copy to your laptop, it is usually easier to run rsync FROM your
# laptop and pull from Snellius. See the command printed at the end.

OUTPUT_BASE="${OUTPUT_BASE:-/scratch-shared/$USER/output}"
AL_RUN="${AL_RUN:-}"
DATASET_NAME="${DATASET_NAME:-}"
RUN_SUFFIX="${RUN_SUFFIX:-popgs20}"
if [ -z "$AL_RUN" ]; then
    if [ -z "$DATASET_NAME" ]; then
        echo "ERROR: set AL_RUN=<dataset>_popgs20 or DATASET_NAME=<dataset>"
        exit 1
    fi
    AL_RUN="${DATASET_NAME}_${RUN_SUFFIX}"
fi
AL_ROOT="${AL_ROOT:-$OUTPUT_BASE/active_learning_mipnerf/$AL_RUN}"
SRC="${SRC:-$AL_ROOT}"
ARCHIVE_BASE="${ARCHIVE_BASE:-$HOME/al_archives}"
ARCHIVE_DIR="${ARCHIVE_DIR:-$ARCHIVE_BASE/${AL_RUN}_minimal}"
PREVIEW_FRAMES="${PREVIEW_FRAMES:-00000 00008}"
DEST="${DEST:-}"

if [ ! -d "$SRC" ]; then
    echo "ERROR: source not found: $SRC"
    exit 1
fi

mkdir -p "$ARCHIVE_DIR"

echo "Source : $SRC"
echo "Archive: $ARCHIVE_DIR"

# Root summaries and generated plots.
rsync -a --prune-empty-dirs \
    --include='*/' \
    --include='active_learning_summary.csv' \
    --include='active_learning_summary.md' \
    --include='final_metrics.csv' \
    --include='final_metrics.md' \
    --include='figures/***' \
    --include='figures_poster/***' \
    --exclude='*' \
    "$SRC/" "$ARCHIVE_DIR/"

# Per-round lightweight provenance and PSNR-curve inputs.
rsync -a --prune-empty-dirs \
    --include='*/' \
    --include='results.json' \
    --include='per_view.json' \
    --include='splits/*.txt' \
    --include='splits/summary.json' \
    --exclude='*' \
    "$SRC/" "$ARCHIVE_DIR/"

# Final-round uncertainty metrics and rankings only.
rsync -a --prune-empty-dirs \
    --include='*/' \
    --include='round_16/conformal/**/metrics.json' \
    --include='round_16/active_learning/**/view_signal_scores.csv' \
    --include='round_16/active_learning/**/view_rankings.json' \
    --exclude='*' \
    "$SRC/" "$ARCHIVE_DIR/"

# Tiny qualitative progression subset: keep a few fixed test frames across all
# rounds/methods. These are enough for report figures showing improvement over
# training views without archiving every rendered image.
for frame in $PREVIEW_FRAMES; do
    for suffix in png jpg jpeg JPG JPEG; do
        rsync -a --prune-empty-dirs \
            --include='*/' \
            --include="test/ours_*/gt/${frame}.${suffix}" \
            --include="test/ours_*/render/${frame}.${suffix}" \
            --exclude='*' \
            "$SRC/" "$ARCHIVE_DIR/"
    done
done

du -sh "$ARCHIVE_DIR"

if [ -n "$DEST" ]; then
    echo "Rsyncing archive to: $DEST"
    rsync -avh --progress "$ARCHIVE_DIR/" "$DEST/"
else
    echo
    echo "DEST is empty, so nothing was copied off Snellius."
    echo "To pull this archive from your laptop, run something like:"
    echo
    echo "  rsync -avh --progress <snellius_user>@snellius.surf.nl:$ARCHIVE_DIR/ ./$(basename "$ARCHIVE_DIR")/"
    echo
fi
