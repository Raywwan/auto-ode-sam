#!/usr/bin/env bash
# =============================================================================
# run_curve_queue.sh — 3D-DSC-vs-epoch curve sweep for both seed=43 and fwdonly.
#
# Runs scripts/eval_ckpt_curve.sh against each per-epoch ckpt of:
#   1. seed=43 retrain (V2 reproducibility curve)
#   2. forward-only ODE ablation (bidirectional contribution curve)
#
# Outputs publication-ready CSVs under:
#   thesis/results/curves/seed43/curve.csv
#   thesis/results/curves/fwdonly/curve.csv
#
# Wallclock budget: ~9 GPU-h total (9 ckpts × 30 min × 2 runs).
# Designed to be launched AFTER run_gpu_queue.sh finishes — does not interfere
# with the main queue's ckpt selection.
#
# Usage:
#   bash scripts/run_curve_queue.sh 2>&1 | tee logs/curve_queue.log
# =============================================================================
set -u

ROOT="C:/Users/Raywa/Desktop/VoluFormer3D_V4"
STATUS_DIR="$ROOT/logs/queue_status"
mkdir -p "$STATUS_DIR"

cd "$ROOT" || exit 99

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
log()   { echo "[$(stamp)] [CURVE] $*"; }

# -----------------------------------------------------------------------------
# Step C1 — seed=43 reproducibility curve
# -----------------------------------------------------------------------------
SEED43_RUN="checkpoints/phase3_odesam_v2_seed43_40ep"
SEED43_CFG="configs/phase3_odesam_v2_seed43.yaml"
SEED43_OUT="thesis/results/curves/seed43"

log "Step C1: seed=43 ckpt-curve sweep"
if [ -d "$SEED43_RUN" ]; then
    bash "$ROOT/scripts/eval_ckpt_curve.sh" \
        "$SEED43_RUN" "$SEED43_CFG" "$SEED43_OUT" \
        > "$ROOT/logs/curve_step_c1_seed43.log" 2>&1
    rc=$?
    echo "$rc" > "$STATUS_DIR/curve_step_c1_seed43.exit"
    if [ "$rc" -eq 0 ]; then
        log "OK   C1 seed=43 — see $SEED43_OUT/curve.csv"
    else
        log "FAIL C1 seed=43 (rc=$rc) — see logs/curve_step_c1_seed43.log"
    fi
else
    log "SKIP C1 — $SEED43_RUN does not exist"
    echo "skipped" > "$STATUS_DIR/curve_step_c1_seed43.exit"
fi

# -----------------------------------------------------------------------------
# Step C2 — fwdonly ablation curve
# -----------------------------------------------------------------------------
FWD_RUN="checkpoints/phase3_odesam_v2_fwdonly_40ep"
FWD_CFG="configs/phase3_odesam_v2_fwdonly.yaml"
FWD_OUT="thesis/results/curves/fwdonly"

log "Step C2: fwdonly ckpt-curve sweep"
if [ -d "$FWD_RUN" ]; then
    bash "$ROOT/scripts/eval_ckpt_curve.sh" \
        "$FWD_RUN" "$FWD_CFG" "$FWD_OUT" \
        > "$ROOT/logs/curve_step_c2_fwdonly.log" 2>&1
    rc=$?
    echo "$rc" > "$STATUS_DIR/curve_step_c2_fwdonly.exit"
    if [ "$rc" -eq 0 ]; then
        log "OK   C2 fwdonly — see $FWD_OUT/curve.csv"
    else
        log "FAIL C2 fwdonly (rc=$rc) — see logs/curve_step_c2_fwdonly.log"
    fi
else
    log "SKIP C2 — $FWD_RUN does not exist"
    echo "skipped" > "$STATUS_DIR/curve_step_c2_fwdonly.exit"
fi

log "CURVE QUEUE COMPLETE — see $STATUS_DIR for per-step exit codes"
