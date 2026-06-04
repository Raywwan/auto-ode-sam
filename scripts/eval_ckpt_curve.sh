#!/usr/bin/env bash
# =============================================================================
# eval_ckpt_curve.sh — Sweep saved per-epoch ckpts through the 3D N=100 eval
# pipeline so we can plot a 3D-DSC-vs-epoch curve (publication figure).
#
# Reuses scripts/eval_amos22_liver_indomain.py (the validated headline eval).
# Each per-ckpt eval runs ~25-40 min; with ~9 ckpts per run that's ~4-6 GPU-h.
#
# Usage:
#   bash scripts/eval_ckpt_curve.sh <run_ckpt_dir> <config> <out_base>
#
# Example:
#   bash scripts/eval_ckpt_curve.sh \
#       checkpoints/phase3_odesam_v2_seed43_40ep \
#       configs/phase3_odesam_v2_seed43.yaml \
#       thesis/results/curves/seed43
# =============================================================================
set -u

if [ "$#" -lt 3 ]; then
    echo "usage: $0 <run_ckpt_dir> <config> <out_base>"
    exit 64
fi

RUN_DIR="$1"
CONFIG="$2"
OUT_BASE="$3"

ROOT="C:/Users/Raywa/Desktop/VoluFormer3D_V4"
PYTHON="C:/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe"

mkdir -p "$OUT_BASE"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
log()   { echo "[$(stamp)] [curve] $*"; }

log "run_dir : $RUN_DIR"
log "config  : $CONFIG"
log "out_base: $OUT_BASE"

# Sort ckpts by epoch number (natural sort).
ckpts=$(ls -1 "$RUN_DIR" 2>/dev/null | grep -E '_epoch[0-9]+\.pt$' | sort)
if [ -z "$ckpts" ]; then
    log "ERROR: no _epochNNN.pt ckpts found in $RUN_DIR"
    exit 65
fi

n_ckpts=$(echo "$ckpts" | wc -l | tr -d ' ')
log "found $n_ckpts per-epoch ckpts"

i=0
for fname in $ckpts; do
    i=$((i + 1))
    ckpt="$RUN_DIR/$fname"
    epoch=$(echo "$fname" | grep -oE 'epoch[0-9]+' | grep -oE '[0-9]+' | sed 's/^0*//')
    if [ -z "$epoch" ]; then epoch=0; fi
    out_dir="$OUT_BASE/ep${epoch}"
    if [ -f "$out_dir/per_volume_metrics.csv" ] && [ -f "$out_dir/summary.json" ]; then
        log "  ($i/$n_ckpts) ep${epoch} already evaluated — skipping"
        continue
    fi
    log "  ($i/$n_ckpts) eval ep${epoch}: $ckpt"
    mkdir -p "$out_dir"
    "$PYTHON" "$ROOT/scripts/eval_amos22_liver_indomain.py" \
        --checkpoint "$ckpt" \
        --config     "$CONFIG" \
        --output     "$out_dir" \
        --img-size   256 --n-slices 8 \
        > "$out_dir/eval.log" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
        log "  ($i/$n_ckpts) ep${epoch} FAILED (rc=$rc) — see $out_dir/eval.log"
    fi
done

log "aggregating into $OUT_BASE/curve.csv"
"$PYTHON" "$ROOT/scripts/aggregate_curve.py" \
    --base_dir "$OUT_BASE" \
    --out_csv  "$OUT_BASE/curve.csv"

log "DONE — curve at $OUT_BASE/curve.csv"
