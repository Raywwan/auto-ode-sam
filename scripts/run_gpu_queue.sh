#!/usr/bin/env bash
# =============================================================================
# run_gpu_queue.sh — Sequential GPU job runner for the A0 thesis push.
#
# Order:
#   1. V2 seed=43 retrain (40 ep) → eval N=100 in-domain → eval N=100 TotalSeg
#   2. Forward-only ODE retrain (40 ep) → eval N=100 in-domain
#   3. nnU-Net AMOS22 liver baseline → eval N=100 in-domain
#
# Each step writes its full log under logs/ and an exit-code marker under
# logs/queue_status/. If any step fails, the queue records the failure but
# CONTINUES to the next independent job (we'd rather get partial results
# than block on a single failure).
#
# Usage:
#   bash scripts/run_gpu_queue.sh 2>&1 | tee logs/gpu_queue.log
# =============================================================================
set -u  # NOT -e: we want to continue on individual job failure

ROOT="C:/Users/Raywa/Desktop/VoluFormer3D_V4"
PYTHON="C:/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe"
STATUS_DIR="$ROOT/logs/queue_status"
mkdir -p "$STATUS_DIR"

cd "$ROOT" || exit 99

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(stamp)] [QUEUE] $*"; }

run_step() {
    local name="$1"
    shift
    local logfile="$ROOT/logs/${name}.log"
    log "START $name"
    log "  cmd: $*"
    log "  log: $logfile"
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

# -----------------------------------------------------------------------------
# Step 1: V2 seed=43 — 40 epoch fresh cosine
# -----------------------------------------------------------------------------
run_step "queue_step1_v2_seed43_train" \
    "$PYTHON" train.py --config configs/phase3_odesam_v2_seed43.yaml

# Eval seed=43 if best.pt exists
S43_BEST="checkpoints/phase3_odesam_v2_seed43_40ep/phase3_odesam_v2_seed43_40ep_best.pt"
if [ -f "$S43_BEST" ]; then
    run_step "queue_step1b_seed43_indomain_eval" \
        "$PYTHON" scripts/eval_amos22_liver_indomain.py \
            --checkpoint "$S43_BEST" \
            --config     configs/phase3_odesam_v2_seed43.yaml \
            --output     thesis/results/amos22_liver_indomain_seed43 \
            --img-size   256 --n-slices 8

    run_step "queue_step1c_seed43_totalseg_eval" \
        "$PYTHON" scripts/eval_totalseg_liver.py \
            --checkpoint "$S43_BEST" \
            --config     configs/phase3_odesam_v2_seed43.yaml \
            --output     thesis/results/totalseg_liver_crossds_seed43 \
            --img-size   256 --n-slices 8 --max-volumes 100
else
    log "SKIP seed43 evals — best.pt missing at $S43_BEST"
fi

# -----------------------------------------------------------------------------
# Step 2: Forward-only ODE — 40 epoch
# -----------------------------------------------------------------------------
run_step "queue_step2_fwdonly_train" \
    "$PYTHON" train.py --config configs/phase3_odesam_v2_fwdonly.yaml

FWD_BEST="checkpoints/phase3_odesam_v2_fwdonly_40ep/phase3_odesam_v2_fwdonly_40ep_best.pt"
if [ -f "$FWD_BEST" ]; then
    run_step "queue_step2b_fwdonly_indomain_eval" \
        "$PYTHON" scripts/eval_amos22_liver_indomain.py \
            --checkpoint "$FWD_BEST" \
            --config     configs/phase3_odesam_v2_fwdonly.yaml \
            --output     thesis/results/amos22_liver_indomain_fwdonly \
            --img-size   256 --n-slices 8
else
    log "SKIP fwdonly eval — best.pt missing at $FWD_BEST"
fi

# -----------------------------------------------------------------------------
# Step 3: nnU-Net AMOS22 liver baseline (env install + train + eval)
# -----------------------------------------------------------------------------
# nnU-Net deployment is handled by a dedicated bash, NOT inline here, so the
# queue can resume cleanly if step 3 needs interactive intervention.
log "Step 3 trigger: bash scripts/run_nnunet_baseline.sh"
bash "$ROOT/scripts/run_nnunet_baseline.sh" \
    > "$ROOT/logs/queue_step3_nnunet.log" 2>&1
echo $? > "$STATUS_DIR/queue_step3_nnunet.exit"

log "GPU QUEUE COMPLETE — see $STATUS_DIR for per-step exit codes"
