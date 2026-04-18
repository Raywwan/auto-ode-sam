import numpy as np
import pytest
import torch
from datasets.amos22_multiorgan import AMOS22MultiOrgan3D_Dataset

DATA_ROOT = r"C:\Users\Raywa\Desktop\LiteSAM3D\data\amos22"


@pytest.mark.skipif(not __import__("pathlib").Path(DATA_ROOT).exists(),
                    reason="AMOS22 data not present on this machine")
def test_dataset_shape_and_dtype():
    ds = AMOS22MultiOrgan3D_Dataset(
        data_root=DATA_ROOT, split="train",
        img_size=256, depth=8, modality="ct",
    )
    assert len(ds) > 0
    sample = ds[0]
    assert sample["image"].shape == (8, 1, 256, 256)
    assert sample["masks"].shape == (15, 8, 256, 256)
    assert sample["present_mask"].shape == (15,)
    assert sample["image"].dtype == torch.float32
    assert sample["masks"].dtype == torch.uint8
    assert sample["present_mask"].sum() >= 1
