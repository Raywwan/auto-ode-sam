"""Label map between TotalSegmentator (104 classes) and AMOS22 (15 organs).

13 of AMOS22's 15 organs have direct or near-direct counterparts in TotalSeg.
Bladder and prostate_uterus are AMOS-only (not in TotalSeg core 104).
"""
import pytest
from datasets.totalseg_label_map import (
    AMOS22_TO_TOTALSEG,
    TOTALSEG_TO_AMOS22,
    SHARED_AMOS_INDICES,
    AMOS_ONLY_INDICES,
    remap_label_volume,
)
import numpy as np


def test_shared_indices_count():
    assert len(SHARED_AMOS_INDICES) == 13


def test_amos_only_is_bladder_and_prostate():
    assert set(AMOS_ONLY_INDICES) == {14, 15}


def test_amos_to_totalseg_keys_are_shared():
    assert set(AMOS22_TO_TOTALSEG.keys()) == set(SHARED_AMOS_INDICES)


def test_round_trip_consistency():
    for amos_id in SHARED_AMOS_INDICES:
        ts_ids = AMOS22_TO_TOTALSEG[amos_id]
        for ts_id in ts_ids:
            assert TOTALSEG_TO_AMOS22[ts_id] == amos_id


def test_remap_drops_unmapped_classes():
    # TotalSeg label 200 (not in our map) must become 0 (bg).
    vol = np.array([[0, 1, 200, 14]], dtype=np.int64)  # ts 1=spleen → amos 1
    out = remap_label_volume(vol)
    assert out[0, 0] == 0
    assert out[0, 1] == 1  # spleen
    assert out[0, 2] == 0  # unknown → bg
    assert out[0, 3] == 0  # ts 14 (spinal_cord) is unmapped → bg
    assert out.dtype == np.int64
