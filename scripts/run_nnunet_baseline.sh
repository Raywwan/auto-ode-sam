#!/usr/bin/env bash
# =============================================================================
# run_nnunet_baseline.sh — nnU-Net AMOS22-liver single-fold baseline
#
# Trains a 3d_fullres single-fold nnUNetv2 on AMOS22 liver-binary, then
# evaluates on the SAME N=100 val volumes used by the V2 headline so the
# comparison is apples-to-apples.
#
# Pipeline:
#   1. Install nnunetv2 (idempotent — skip if already installed)
#   2. Convert AMOS22 → Dataset511_AMOS22Liver/ (imagesTr/labelsTr/imagesVa/)
#      Liver only (class 6 → 1, all else → 0). N_train=240, N_val=100.
#   3. nnUNetv2_plan_and_preprocess
#   4. nnUNetv2_train ... 3d_fullres 0  (fold 0 only, NOT 5-fold)
#   5. nnUNetv2_predict on the val set
#   6. Bridge predictions through V2's metric pipeline for apples-to-apples comparison.
#
# Output:
#   thesis/results/amos22_liver_nnunet/per_volume_metrics.csv  (same schema as V2)
#   thesis/results/amos22_liver_nnunet/summary.json
#
# Wall-clock budget: ~30 GPU-h on RTX 4090 (preprocess 1 h, train 24 h, predict 3 h, eval 2 h)
# =============================================================================
set -u

ROOT="C:/Users/Raywa/Desktop/VoluFormer3D_V4"
PYTHON="C:/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe"
AMOS_ROOT="C:/Users/Raywa/Desktop/LiteSAM3D/data/amos22"
NNUNET_ROOT="$ROOT/nnUNet_workspace"

export nnUNet_raw="$NNUNET_ROOT/nnUNet_raw"
export nnUNet_preprocessed="$NNUNET_ROOT/nnUNet_preprocessed"
export nnUNet_results="$NNUNET_ROOT/nnUNet_results"

mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(stamp)] [nnUNet] $*"; }

# -----------------------------------------------------------------------------
# Step 1 — install nnUNetv2 (idempotent)
# -----------------------------------------------------------------------------
log "Step 1: ensure nnunetv2 installed"
if ! "$PYTHON" -c "import nnunetv2" 2>/dev/null; then
    log "  installing nnunetv2..."
    "$PYTHON" -m pip install --no-warn-script-location nnunetv2 2>&1 | tail -20
else
    log "  nnunetv2 already importable"
fi

# -----------------------------------------------------------------------------
# Step 2 — Convert AMOS22 → nnU-Net liver-binary Dataset511
# -----------------------------------------------------------------------------
DSET_NAME="Dataset511_AMOS22Liver"
DSET_DIR="$nnUNet_raw/$DSET_NAME"
log "Step 2: build $DSET_NAME at $DSET_DIR"

if [ -f "$DSET_DIR/dataset.json" ]; then
    log "  $DSET_NAME already exists — skipping conversion"
else
    "$PYTHON" "$ROOT/scripts/build_nnunet_amos22_liver.py" \
        --amos_root "$AMOS_ROOT" \
        --dataset_dir "$DSET_DIR" \
        --max_val 100
    log "  conversion done"
fi

# -----------------------------------------------------------------------------
# Step 3 — plan_and_preprocess
# -----------------------------------------------------------------------------
log "Step 3: plan_and_preprocess"
"$PYTHON" -m nnunetv2.experiment_planning.plan_and_preprocess_entrypoints \
    -d 511 -c 3d_fullres --verify_dataset_integrity 2>&1 | tail -40

# -----------------------------------------------------------------------------
# Step 4 — train fold 0 only
# -----------------------------------------------------------------------------
log "Step 4: train 3d_fullres fold 0 (single fold, NOT 5-fold)"
"$PYTHON" -m nnunetv2.run.run_training \
    511 3d_fullres 0 --npz 2>&1 | tee "$ROOT/logs/nnunet_train_511_3d_fullres_fold0.log"

# -----------------------------------------------------------------------------
# Step 5 — predict on imagesVa (the converted N=100 val set)
# -----------------------------------------------------------------------------
PRED_DIR="$ROOT/thesis/results/amos22_liver_nnunet/predictions"
mkdir -p "$PRED_DIR"
log "Step 5: predict imagesVa → $PRED_DIR"
# IMPORTANT: `python -m nnunetv2.inference.predict_from_raw_data` runs a
# hardcoded `Dataset004_Hippocampus` demo (its __main__ ignores CLI args).
# Use the installed `nnUNetv2_predict` entrypoint instead.
NNUNET_PREDICT="C:/Users/Raywa/AppData/Local/Programs/Python/Python312/Scripts/nnUNetv2_predict.exe"
"$NNUNET_PREDICT" \
    -i "$DSET_DIR/imagesVa" \
    -o "$PRED_DIR" \
    -d 511 -c 3d_fullres -f 0 \
    -chk checkpoint_best.pth \
    --save_probabilities 2>&1 | tail -30

# -----------------------------------------------------------------------------
# Step 6 — bridge to V2's metric pipeline (DSC/HD95/NSD + extras)
# -----------------------------------------------------------------------------
log "Step 6: evaluate predictions with V2's 3D metric pipeline"
"$PYTHON" "$ROOT/scripts/eval_nnunet_predictions.py" \
    --pred_dir "$PRED_DIR" \
    --amos_root "$AMOS_ROOT" \
    --out_dir "$ROOT/thesis/results/amos22_liver_nnunet"

log "DONE — results at $ROOT/thesis/results/amos22_liver_nnunet/"
