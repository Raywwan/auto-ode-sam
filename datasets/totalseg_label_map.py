"""AMOS22 ↔ TotalSegmentator label mapping for Phase I cross-dataset pretrain.

Source: TotalSegmentator v2 (Wasserthal et al, RSNA 2023). Class IDs follow the
official `total` task (104 classes). https://github.com/wasserthal/TotalSegmentator

AMOS22 organ IDs (1-indexed; 0 = bg):
  1=spleen, 2=r_kidney, 3=l_kidney, 4=gallbladder, 5=esophagus, 6=liver,
  7=stomach, 8=aorta, 9=ivc, 10=pancreas, 11=r_adrenal, 12=l_adrenal,
  13=duodenum, 14=bladder, 15=prostate_uterus

13 organs have direct counterparts in TotalSeg `total` task. Bladder is in
TotalSeg under `urinary_bladder` (id 104). Prostate_uterus has no direct
counterpart (TotSeg has prostate-male only via separate task) — we treat it
as AMOS-only for Phase I and rely on AMOS22 fine-tune to recover it.
"""
from __future__ import annotations
from typing import Dict, List
import numpy as np

# AMOS organ id -> list of TotalSeg class IDs that should remap to it.
# Some AMOS classes correspond to multiple TS classes (e.g., L1-L5 vertebrae
# would not but kidneys are split L/R in both).
AMOS22_TO_TOTALSEG: Dict[int, List[int]] = {
    1: [1],         # spleen
    2: [2],         # kidney_right
    3: [3],         # kidney_left
    4: [4],         # gallbladder
    5: [42],        # esophagus
    6: [5],         # liver
    7: [6],         # stomach
    8: [7],         # aorta
    9: [8, 9],      # inferior_vena_cava + portal_vein_and_splenic_vein (closest)
    10: [10],       # pancreas
    11: [11],       # adrenal_gland_right
    12: [12],       # adrenal_gland_left
    13: [55],       # duodenum
}

SHARED_AMOS_INDICES: List[int] = sorted(AMOS22_TO_TOTALSEG.keys())  # 13 indices
AMOS_ONLY_INDICES: List[int] = [14, 15]                              # bladder, prostate_uterus

# Reverse lookup: TS class id -> AMOS organ id
TOTALSEG_TO_AMOS22: Dict[int, int] = {}
for amos_id, ts_list in AMOS22_TO_TOTALSEG.items():
    for ts_id in ts_list:
        TOTALSEG_TO_AMOS22[ts_id] = amos_id


def remap_label_volume(label: np.ndarray) -> np.ndarray:
    """Map a TotalSeg label volume to AMOS22 indices.

    Any TS class not in TOTALSEG_TO_AMOS22 is mapped to 0 (background).
    Output dtype is always int64 (forced via dtype= override). Preserves shape.
    """
    out = np.zeros_like(label, dtype=np.int64)
    for ts_id, amos_id in TOTALSEG_TO_AMOS22.items():
        out[label == ts_id] = amos_id
    return out
