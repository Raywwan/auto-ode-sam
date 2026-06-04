#!/bin/bash
# Sequential test runs: MultiScaleISA → FCA-SAM → ACM-SAM
# Each: 20 epochs, 256px, liver only (organ 6)
# Results saved to checkpoints/ and logs/

PYTHON="/c/Users/Raywa/Desktop/LiteSAM3D/.venv/Scripts/python"
DIR="C:/Users/Raywa/Desktop/VoluFormer3D"

cd "C:/Users/Raywa/Desktop/VoluFormer3D"

echo "============================================================"
echo "[1/3] MultiScaleISA — $(date)"
echo "============================================================"
$PYTHON train.py --config configs/test_multiscale_isa_a1.yaml
echo "MultiScaleISA done — $(date)"

echo ""
echo "============================================================"
echo "[2/3] FCA-SAM — $(date)"
echo "============================================================"
$PYTHON train.py --config configs/test_fca_sam_a1.yaml
echo "FCA-SAM done — $(date)"

echo ""
echo "============================================================"
echo "[3/3] ACM-SAM — $(date)"
echo "============================================================"
$PYTHON train.py --config configs/test_acm_sam_a1.yaml
echo "ACM-SAM done — $(date)"

echo ""
echo "============================================================"
echo "ALL RUNS COMPLETE — $(date)"
echo "============================================================"
