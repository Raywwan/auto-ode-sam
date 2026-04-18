# =============================================================================
# models/__init__.py — Model Factory
# =============================================================================

from omegaconf import DictConfig


def build_model(cfg: DictConfig):
    """
    Build the model specified by cfg.model.architecture.

    Supported architectures:
        "multiscale_isa"  — MultiScaleISA (Stage 2 DA-ISA) [default]
        "fca_sam"         — FCA-SAM (FFT frequency cross-slice adapter)
        "acm_sam"         — ACM-SAM (Anatomical Context Memory bank)
    """
    arch = getattr(cfg.model, "architecture", "multiscale_isa")

    if arch == "multiscale_isa":
        from models.voluformer3d import VoluFormer3D
        return VoluFormer3D(cfg)
    elif arch == "fca_sam":
        from models.fca_sam import FCASAM
        return FCASAM(cfg)
    elif arch == "acm_sam":
        from models.acm_sam import ACMSAM
        return ACMSAM(cfg)
    elif arch == "trimamba_sam":
        from models.trimamba_sam import TriMambaSAM
        return TriMambaSAM(cfg)
    elif arch == "ode_sam":
        from models.ode_sam import ODESAM
        return ODESAM(cfg)
    elif arch == "auto_ode_sam":
        from models.auto_ode_sam import AutoODESAM
        return AutoODESAM(cfg)
    elif arch == "organflow_sam2":
        from models.organflow_sam2 import OrganFlowSAM2
        return OrganFlowSAM2(cfg)
    elif arch == "litesam3d_v2":
        raise NotImplementedError(
            "litesam3d_v2 is not available in VoluFormer3D. "
            "Use architecture='multiscale_isa' instead."
        )
    else:
        raise ValueError(
            f"Unknown architecture: {arch!r}. "
            f"Options: multiscale_isa, fca_sam, acm_sam, trimamba_sam, ode_sam, auto_ode_sam, organflow_sam2"
        )
