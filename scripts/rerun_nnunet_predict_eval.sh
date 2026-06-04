#!/usr/bin/env bash
# =============================================================================
# rerun_nnunet_predict_eval.sh — Re-run nnUNet Step 5 (predict) + Step 6 (eval)
# after the 2026-05-05 queue's predict step failed silently.
#
# Root cause: `python -m nnunetv2.inference.predict_from_raw_data` is NOT a
# proper argparse entrypoint — its __main__ runs a hardcoded
# `Dataset004_Hippocampus / nnUNetTrainer_5epochs` demo regardless of CLI args.
# The correct entrypoint is the installed `nnUNetv2_predict.exe` script.
#
# Training is already done; checkpoint exists at:
#   nnUNet_workspace/nnUNet_results/Dataset511_AMOS22Liver/
#     nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth
#
# Pre-condition: GPU is FREE (run after seed=43 rescue finishes).
# Wallclock: ~30-60 min (100 vols, sliding-window inference + TTA).
#
# Usage:
#   bash scripts/rerun_nnunet_predict_eval.sh 2>&1 | tee logs/rerun_nnunet_predict_eval.log
# =============================================================================
set -u

ROOT="C:/Users/Raywa/Desktop/VoluFormer3D_V4"
PYTHON="C:/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe"
NNUNET_PREDICT="C:/Users/Raywa/AppData/Local/Programs/Python/Python312/Scripts/nnUNetv2_predict.exe"
AMOS_ROOT="C:/Users/Raywa/Desktop/LiteSAM3D/data/amos22"
NNUNET_ROOT="$ROOT/nnUNet_workspace"
STATUS_DIR="$ROOT/logs/queue_status"
mkdir -p "$STATUS_DIR"

export nnUNet_raw="$NNUNET_ROOT/nnUNet_raw"
export nnUNet_preprocessed="$NNUNET_ROOT/nnUNet_preprocessed"
export nnUNet_results="$NNUNET_ROOT/nnUNet_results"

cd "$ROOT" || exit 99

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
log()   { echo "[$(stamp)] [NNUNET-RERUN] $*"; }

DSET_NAME="Dataset511_AMOS22Liver"
DSET_DIR="$nnUNet_raw/$DSET_NAME"
PRED_DIR="$ROOT/thesis/results/amos22_liver_nnunet/predictions"
mkdir -p "$PRED_DIR"

CKPT="$nnUNet_results/$DSET_NAME/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth"
if [ ! -f "$CKPT" ]; then
    log "ABORT — checkpoint missing at $CKPT"
    exit 1
fi
if [ ! -d "$DSET_DIR/imagesVa" ]; then
    log "ABORT — imagesVa missing at $DSET_DIR/imagesVa"
    exit 1
fi

# -------------------------------------------------------------------------
# Step 5 — predict using the proper entrypoint
# -------------------------------------------------------------------------
log "Step 5: nnUNetv2_predict → $PRED_DIR"
log "  imagesVa: $DSET_DIR/imagesVa"
log "  checkpoint: $CKPT"

# Clear any stale partial predictions from the failed run (none expected, but be safe)
rm -f "$PRED_DIR"/*.nii.gz "$PRED_DIR"/*.npz "$PRED_DIR"/*.pkl 2>/dev/null

"$NNUNET_PREDICT" \
    -i "$DSET_DIR/imagesVa" \
    -o "$PRED_DIR" \
    -d 511 -c 3d_fullres -f 0 \
    -chk checkpoint_best.pth \
    --save_probabilities \
    2>&1 | tee "$ROOT/logs/nnunet_predict_step5.log"
rc=${PIPESTATUS[0]}
echo "$rc" > "$STATUS_DIR/rerun_nnunet_step5_predict.exit"
if [ "$rc" -ne 0 ]; then
    log "FAIL Step 5 (exit $rc)"
    exit "$rc"
fi
log "OK   Step 5"

# Quick sanity: did any predictions land?
N_PRED=$(ls "$PRED_DIR"/*.nii.gz 2>/dev/null | wc -l)
log "  predictions written: $N_PRED .nii.gz files"
if [ "$N_PRED" -eq 0 ]; then
    log "ABORT — Step 5 returned 0 but wrote 0 predictions; something is off"
    exit 2
fi

# -------------------------------------------------------------------------
# Step 6 — bridge through V2's 3D metric pipeline
# -------------------------------------------------------------------------
log "Step 6: eval predictions with V2's pipeline"
"$PYTHON" "$ROOT/scripts/eval_nnunet_predictions.py" \
    --pred_dir "$PRED_DIR" \
    --amos_root "$AMOS_ROOT" \
    --out_dir "$ROOT/thesis/results/amos22_liver_nnunet" \
    2>&1 | tee "$ROOT/logs/nnunet_eval_step6.log"
rc=${PIPESTATUS[0]}
echo "$rc" > "$STATUS_DIR/rerun_nnunet_step6_eval.exit"
if [ "$rc" -ne 0 ]; then
    log "FAIL Step 6 (exit $rc)"
    exit "$rc"
fi
log "OK   Step 6"

log "DONE — see $ROOT/thesis/results/amos22_liver_nnunet/summary.json"
