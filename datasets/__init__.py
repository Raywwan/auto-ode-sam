# =============================================================================
# datasets/__init__.py
# Dataset registry and factory function.
# =============================================================================

from datasets.amos22 import AMOS22Dataset, AMOS22_3D_Dataset
from datasets.amos22_multiorgan import AMOS22MultiOrgan3D_Dataset
from datasets.base_dataset import MedicalImageDataset, MODALITY_ID


def get_dataset(cfg, split: str = "train", copypaste_prob: float = 0.0) -> MedicalImageDataset:
    """
    Factory function: instantiate the correct dataset from config.

    Args:
        cfg: OmegaConf config. cfg.data.dataset must be one of:
             "amos22", "amos22_3d"
        split: "train", "val", or "test"
        copypaste_prob: Probability of copy-paste augmentation per sample.
                        Only applied to AMOS22_3D_Dataset train splits.
                        Val/test always get 0.0 — pass explicitly from caller.

    Returns:
        Instantiated dataset object.
    """
    dataset_name = cfg.data.dataset.lower()
    modality = getattr(cfg.data, "modality", "ct")
    target_organs = getattr(cfg.data, "target_organs", None)

    if dataset_name == "amos22":
        return AMOS22Dataset(
            cfg=cfg,
            split=split,
            modality=modality,
            target_organs=target_organs,
        )
    elif dataset_name == "amos22_3d":
        n_slices = getattr(cfg.data, "slices_per_volume", 16)
        return AMOS22_3D_Dataset(
            cfg=cfg,
            split=split,
            modality=modality,
            target_organs=target_organs,
            n_slices=n_slices,
            copypaste_prob=copypaste_prob,
        )
    elif dataset_name == "amos22_multiorgan":
        aug_flag = getattr(cfg.data, "augment", None)
        if aug_flag is not None and split != "train":
            aug_flag = False  # safeguard: never augment val/test
        slabs_per_volume = int(getattr(cfg.data, "slabs_per_volume", 1))
        light_aug = bool(getattr(cfg.data, "light_aug", False))
        return AMOS22MultiOrgan3D_Dataset(
            data_root=cfg.data.data_root,
            split=split,
            img_size=getattr(cfg.data, "img_size", 256),
            depth=getattr(cfg.data, "slices_per_volume", 8),
            modality=modality,
            augment=aug_flag,
            slabs_per_volume=slabs_per_volume,
            light_aug=light_aug,
        )
    else:
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. "
            f"Available: 'amos22', 'amos22_3d', 'amos22_multiorgan'"
        )


__all__ = [
    "MedicalImageDataset",
    "AMOS22Dataset",
    "AMOS22_3D_Dataset",
    "MODALITY_ID",
    "get_dataset",
]
