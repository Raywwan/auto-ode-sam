#!/usr/bin/env bash
# =============================================================================
# rescue_seed43_evals.sh — Re-run the two seed=43 eval steps that crashed
# in run #2 of the GPU queue (2026-05-04 20:15) due to underscored flags
# (--out_dir / --img_size / --n_slices) being passed to argparse names that
# are registered with hyphens (--output / --img-size / --n-slices).
#
# The seed=43 best.pt at ep35 (val_dice 0.9231) is intact — only the eval
# step failed. This script reproduces queue_step1b + queue_step1c with the
# correct flag names.
#
# Pre-condition: GPU is FREE (run after fwdonly training + eval finishes).
# Wallclock: ~25 + ~25 = ~50 min on the 4090.
#
# Usage (after `queue_step2b_fwdonly_indomain_eval` shows OK in queue_status):
#   bash scripts/rescue_seed43_evals.sh 2>&1 | tee logs/rescue_seed43_evals.log
# =============================================================================
set -u

ROOT="C:/Users/Raywa/Desktop/VoluFormer3D_V4"
PYTHON="C:/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe"
STATUS_DIR="$ROOT/logs/queue_status"
mkdir -p "$STATUS_DIR"

cd "$ROOT" || exit 99

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
log()   { echo "[$(stamp)] [RESCUE] $*"; }

S43_BEST="checkpoints/phase3_odesam_v2_seed43_40ep/phase3_odesam_v2_seed43_40ep_best.pt"
if [ ! -f "$S43_BEST" ]; then
    log "ABORT — $S43_BEST missing"
    exit 1
fi

run_step() {
    local name="$1"; shift
    local logfile="$ROOT/logs/${name}.log"
    log "START $name"
    log "  cmd: $*"
    "$@" > "$logfile" 2>&1
    local rc=$?
    echo "$rc" > "$STATUS_DIR/${name}.exit"
    if [ "$rc" -eq 0 ]; then
        log "OK    $name"
    else
        log "FAIL  $name (exit $rc) — see $logfile"
    fi
    return "$rc"
}

# Step 1b — in-domain N=100 eval (AMOS22 val split)
run_step "queue_step1b_seed43_indomain_eval" \
    "$PYTHON" scripts/eval_amos22_liver_indomain.py \
        --checkpoint "$S43_BEST" \
        --config     configs/phase3_odesam_v2_seed43.yaml \
        --output     thesis/results/amos22_liver_indomain_seed43 \
        --img-size   256 --n-slices 8

# Step 1c — cross-dataset eval (TotalSegmentator liver)
run_step "queue_step1c_seed43_totalseg_eval" \
    "$PYTHON" scripts/eval_totalseg_liver.py \
        --checkpoint "$S43_BEST" \
        --config     configs/phase3_odesam_v2_seed43.yaml \
        --output     thesis/results/totalseg_liver_crossds_seed43 \
        --img-size   256 --n-slices 8 --max-volumes 100

log "DONE — see $STATUS_DIR for exit codes"
