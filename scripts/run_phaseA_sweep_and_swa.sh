#!/usr/bin/env bash
# =============================================================================
# run_phaseA_sweep_and_swa.sh — Sequential GPU queue for A0 Option-A FULL plan,
# Phase A (post-nnUNet).
#
# Steps:
#   A1. Threshold sweep on V2 publication ckpt   (~1.0 h)
#   A2. Threshold sweep on seed=43 ckpt          (~1.0 h)
#   A3. Threshold sweep on fwdonly ckpt          (~1.0 h)
#   A4. SWA top5 eval (in-domain AMOS22)         (~0.5 h)
#   A5. SWA top5 eval (TotalSeg cross-dataset)   (~0.5 h)
#
# Each step writes its own log + exit code. The script continues to the next
# step even on failure so we don't have to babysit. Final status surfaced in
# logs/queue_status/.
# =============================================================================
set -u

ROOT="C:/Users/Raywa/Desktop/VoluFormer3D_V4"
PYTHON="C:/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe"
STATUS_DIR="$ROOT/logs/queue_status"
LOG_DIR="$ROOT/logs"
mkdir -p "$STATUS_DIR"

cd "$ROOT" || exit 99

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
log()   { echo "[$(stamp)] [PHASE-A] $*"; }

V2_CKPT="C:/Users/Raywa/Desktop/VoluFormer3D/checkpoints/phase3_odesam_v2_256px_liver/phase3_odesam_v2_256px_liver_best.pt"
V2_CFG="$ROOT/configs/phase3_odesam_v2.yaml"
S43_CKPT="$ROOT/checkpoints/phase3_odesam_v2_seed43_40ep/phase3_odesam_v2_seed43_40ep_best.pt"
S43_CFG="$ROOT/configs/phase3_odesam_v2_seed43.yaml"
FWD_CKPT="$ROOT/checkpoints/phase3_odesam_v2_fwdonly_40ep/phase3_odesam_v2_fwdonly_40ep_best.pt"
FWD_CFG="$ROOT/configs/phase3_odesam_v2_fwdonly.yaml"
SWA_CKPT="$ROOT/checkpoints/phase3_odesam_v2_seed43_40ep/swa_top5.pt"

run_step() {
    local name="$1"; shift
    local logf="$1"; shift
    log "START $name"
    log "  cmd: $*"
    PYTHONUNBUFFERED=1 "$@" 2>&1 | tee "$logf"
    local rc=${PIPESTATUS[0]}
    echo "$rc" > "$STATUS_DIR/$name.exit"
    if [ "$rc" -eq 0 ]; then log "OK    $name"; else log "FAIL  $name (exit $rc)"; fi
}

# ---- A1. V2 publication threshold sweep ----
run_step "phaseA_a1_threshold_sweep_v2" "$LOG_DIR/phaseA_a1_threshold_sweep_v2.log" \
    "$PYTHON" scripts/eval_with_threshold_sweep.py \
    --checkpoint "$V2_CKPT" --config "$V2_CFG" \
    --output thesis/results/threshold_sweep_v2 \
    --img-size 256 --n-slices 8 \
    --thr-lo 0.30 --thr-hi 0.70 --thr-step 0.025

# ---- A2. seed=43 threshold sweep ----
run_step "phaseA_a2_threshold_sweep_seed43" "$LOG_DIR/phaseA_a2_threshold_sweep_seed43.log" \
    "$PYTHON" scripts/eval_with_threshold_sweep.py \
    --checkpoint "$S43_CKPT" --config "$S43_CFG" \
    --output thesis/results/threshold_sweep_seed43 \
    --img-size 256 --n-slices 8 \
    --thr-lo 0.30 --thr-hi 0.70 --thr-step 0.025

# ---- A3. fwdonly threshold sweep ----
run_step "phaseA_a3_threshold_sweep_fwdonly" "$LOG_DIR/phaseA_a3_threshold_sweep_fwdonly.log" \
    "$PYTHON" scripts/eval_with_threshold_sweep.py \
    --checkpoint "$FWD_CKPT" --config "$FWD_CFG" \
    --output thesis/results/threshold_sweep_fwdonly \
    --img-size 256 --n-slices 8 \
    --thr-lo 0.30 --thr-hi 0.70 --thr-step 0.025

# ---- A4. SWA top5 in-domain eval ----
run_step "phaseA_a4_swa_indomain" "$LOG_DIR/phaseA_a4_swa_indomain.log" \
    "$PYTHON" scripts/eval_amos22_liver_indomain.py \
    --checkpoint "$SWA_CKPT" --config "$S43_CFG" \
    --output thesis/results/amos22_liver_indomain_swa_top5 \
    --img-size 256 --n-slices 8

# ---- A5. SWA top5 TotalSeg cross-dataset eval ----
run_step "phaseA_a5_swa_totalseg" "$LOG_DIR/phaseA_a5_swa_totalseg.log" \
    "$PYTHON" scripts/eval_totalseg_liver.py \
    --checkpoint "$SWA_CKPT" --config "$S43_CFG" \
    --output thesis/results/totalseg_liver_crossds_swa_top5 \
    --img-size 256 --n-slices 8 --max-volumes 100

log "DONE — see $STATUS_DIR/phaseA_a*.exit"
