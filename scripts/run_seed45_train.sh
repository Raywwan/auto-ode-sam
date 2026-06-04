#!/usr/bin/env bash
# =============================================================================
# run_seed45_train.sh — n=4 reproducibility seed=45 retrain (40 epochs)
# Byte-equal to seed=44 except experiment.name/seed.
# Exit code captured to logs/queue_status/seed45_train.exit.
# =============================================================================
set -u
cd "$(dirname "$0")/.."

LOGDIR="logs/queue_status"
mkdir -p "$LOGDIR"
LOG="logs/seed45_train.log"

PYBIN="/c/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe"

echo "[seed45] start $(date -Iseconds)" | tee -a "$LOG"
PYTHONUNBUFFERED=1 "$PYBIN" -u train.py \
  --config configs/phase3_odesam_v2_seed45.yaml \
  2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "[seed45] end $(date -Iseconds) exit=$RC" | tee -a "$LOG"
echo "$RC" > "$LOGDIR/seed45_train.exit"
exit "$RC"
