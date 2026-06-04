#!/usr/bin/env bash
# =============================================================================
# run_phaseB_seed4445_eval_and_ensemble.sh — Phase B GPU queue:
#   B0. Block until seed=45 training finishes (seed45_train.exit appears with 0)
#   B1. 3D eval seed=44 best.pt -> AMOS22 N=100
#   B2. 3D eval seed=45 best.pt -> AMOS22 N=100
#   B3. 4-seed ensemble (42,43,44,45) -> AMOS22 N=100 at default tau=0.5
#   B4. 4-seed ensemble (42,43,44,45) -> AMOS22 N=100 at tau=0.40 (best from V2 sweep)
#
# Each step writes its own log + exit code in logs/queue_status/phaseB_*.exit.
# Continues on failure. seed44 eval does NOT block on seed45 — it runs after the
# seed45 trainer has released the GPU.
# =============================================================================
set -u

ROOT="C:/Users/Raywa/Desktop/VoluFormer3D_V4"
PYTHON="C:/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe"
STATUS_DIR="$ROOT/logs/queue_status"
LOG_DIR="$ROOT/logs"
mkdir -p "$STATUS_DIR"
cd "$ROOT" || exit 99

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
log()   { echo "[$(stamp)] [PHASE-B] $*"; }

V2_CKPT="C:/Users/Raywa/Desktop/VoluFormer3D/checkpoints/phase3_odesam_v2_256px_liver/phase3_odesam_v2_256px_liver_best.pt"
S43_CFG="$ROOT/configs/phase3_odesam_v2_seed43.yaml"
S43_CKPT="$ROOT/checkpoints/phase3_odesam_v2_seed43_40ep/phase3_odesam_v2_seed43_40ep_best.pt"
S44_CKPT="$ROOT/checkpoints/phase3_odesam_v2_seed44_40ep/phase3_odesam_v2_seed44_40ep_best.pt"
S45_CKPT="$ROOT/checkpoints/phase3_odesam_v2_seed45_40ep/phase3_odesam_v2_seed45_40ep_best.pt"

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

# ---- B0. Wait for seed=45 to finish ----
log "B0: waiting for seed45_train.exit"
until [ -f "$STATUS_DIR/seed45_train.exit" ]; do
    sleep 60
done
S45_RC="$(cat "$STATUS_DIR/seed45_train.exit")"
log "B0: seed45 train exited with code $S45_RC"
if [ "$S45_RC" != "0" ]; then
    log "B0: seed45 train FAILED — continuing B1 (seed44 eval) but skipping B2-B4 which need seed45"
fi

# ---- B1. 3D eval seed=44 (AMOS22 in-domain) ----
if [ -f "$S44_CKPT" ]; then
    run_step "phaseB_b1_eval_seed44_indomain" "$LOG_DIR/phaseB_b1_eval_seed44_indomain.log" \
        "$PYTHON" scripts/eval_amos22_liver_indomain.py \
        --checkpoint "$S44_CKPT" --config "$S43_CFG" \
        --output thesis/results/amos22_liver_indomain_seed44 \
        --img-size 256 --n-slices 8
else
    log "B1 SKIPPED: $S44_CKPT not found"
fi

# ---- B2. 3D eval seed=45 (AMOS22 in-domain) ----
if [ "$S45_RC" = "0" ] && [ -f "$S45_CKPT" ]; then
    run_step "phaseB_b2_eval_seed45_indomain" "$LOG_DIR/phaseB_b2_eval_seed45_indomain.log" \
        "$PYTHON" scripts/eval_amos22_liver_indomain.py \
        --checkpoint "$S45_CKPT" --config "$S43_CFG" \
        --output thesis/results/amos22_liver_indomain_seed45 \
        --img-size 256 --n-slices 8
else
    log "B2 SKIPPED: seed45 not OK or ckpt missing"
fi

# ---- B3. 4-seed sigmoid ensemble at default tau=0.5 (AMOS22) ----
if [ "$S45_RC" = "0" ] && [ -f "$S45_CKPT" ] && [ -f "$S44_CKPT" ] && [ -f "$S43_CKPT" ] && [ -f "$V2_CKPT" ]; then
    run_step "phaseB_b3_ensemble_4seed_tau050" "$LOG_DIR/phaseB_b3_ensemble_4seed_tau050.log" \
        "$PYTHON" scripts/eval_4seed_ensemble.py \
        --checkpoints "$V2_CKPT" "$S43_CKPT" "$S44_CKPT" "$S45_CKPT" \
        --config "$S43_CFG" \
        --output thesis/results/ensemble_4seed_tau050 \
        --img-size 256 --n-slices 8 --threshold 0.5
else
    log "B3 SKIPPED: one of the four ckpts missing"
fi

# ---- B4. 4-seed ensemble at tau=0.40 (best from V2 single-ckpt sweep) ----
if [ "$S45_RC" = "0" ] && [ -f "$S45_CKPT" ] && [ -f "$S44_CKPT" ] && [ -f "$S43_CKPT" ] && [ -f "$V2_CKPT" ]; then
    run_step "phaseB_b4_ensemble_4seed_tau040" "$LOG_DIR/phaseB_b4_ensemble_4seed_tau040.log" \
        "$PYTHON" scripts/eval_4seed_ensemble.py \
        --checkpoints "$V2_CKPT" "$S43_CKPT" "$S44_CKPT" "$S45_CKPT" \
        --config "$S43_CFG" \
        --output thesis/results/ensemble_4seed_tau040 \
        --img-size 256 --n-slices 8 --threshold 0.40
else
    log "B4 SKIPPED: one of the four ckpts missing"
fi

log "DONE — see $STATUS_DIR/phaseB_b*.exit"
