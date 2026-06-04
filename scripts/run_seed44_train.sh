#!/usr/bin/env bash
# =============================================================================
# run_seed44_train.sh — n=4 reproducibility seed=44 retrain (40 epochs)
# Launches V2 architecture with seed=44, identical hyperparams to seed=43.
# Exit code captured to logs/queue_status/seed44_train.exit.
# =============================================================================
set -u
cd "$(dirname "$0")/.."

LOGDIR="logs/queue_status"
mkdir -p "$LOGDIR"
LOG="logs/seed44_train.log"

PYBIN="/c/Users/Raywa/AppData/Local/Programs/Python/Python312/python.exe"

echo "[seed44] start $(date -Iseconds)" | tee -a "$LOG"
PYTHONUNBUFFERED=1 "$PYBIN" -u train.py \
  --config configs/phase3_odesam_v2_seed44.yaml \
  2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "[seed44] end $(date -Iseconds) exit=$RC" | tee -a "$LOG"
echo "$RC" > "$LOGDIR/seed44_train.exit"
exit "$RC"
