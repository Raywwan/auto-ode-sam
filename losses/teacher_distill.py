"""V9 Novel #3 — Teacher-ensemble distillation loss.

Distills multi-teacher feature representations into the SwinUNETR-3D proposer:
  - SAM2 image encoder      (strong mask-aware visual priors)
  - DINOv2-ViT-L/14         (strong self-supervised general vision features)
  - BiomedCLIP              (medical text-vision joint features)

Novelty claim (spec §3, row #3): multi-teacher feature matching with
*adaptive per-organ teacher weighting*. Different organs benefit from
different teachers (esophagus — tiny, boundary-driven → SAM2; kidney —
textured → DINOv2; pancreas — requires semantic disambiguation → BiomedCLIP).
A learnable softmax-normalized `(n_organs, n_teachers)` matrix selects the
effective teacher mixture per-organ at each optimization step.

Design discipline (spec §7b — loss preservation clause):
  - `TeacherDistillLoss` returns a scalar; when the config weight is 0 the
    caller skips calling it entirely and pays zero forward cost.
  - Teacher loading is **lazy** and **optional**. If a teacher fails to load
    (no internet, missing weights, CUDA OOM) we warn and continue with the
    remaining teachers. Falls back to an identity mapping if zero teachers
    load — the loss becomes a no-op rather than crashing training.
  - Teachers are frozen. Their parameters never appear in the optimizer.
  - Teacher forward passes happen inside `torch.no_grad()` and features are
    detached before feeding the matching objective.

Feature matching strategy:
  - We expose three student taps from SwinUNETR via
    `SwinUNETRProposer.register_distill_hooks(...)` (layers 1/2/3 MLP outputs).
  - Teacher features are extracted as the global-pool token (or CLS token).
  - Student features are global-avg-pooled across spatial dims to match.
  - A light projection head (Linear + GELU + Linear) maps the pooled student
    dim → teacher dim at each tap.
  - Loss = mean over (sample, tap, teacher) of a per-organ weighted cosine
    distance: `1 - cosine(proj_student, teacher)`.

Batch shape assumptions:
  - student features: Dict[str, Tensor] with shape (B, C_s, D, H, W)
  - teacher features: Dict[str, Tensor] with shape (B, C_t) already pooled
  - organ_id: (B,) long in [0, n_organs)
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Teacher wrappers — each one is optional. Each returns a (B, C_t) pooled feat.
# ---------------------------------------------------------------------------

class _TeacherBase(nn.Module):
    name: str = "base"
    out_dim: int = 0

    def is_available(self) -> bool:
        return False

    @torch.no_grad()
    def forward(self, img_2d: torch.Tensor) -> Optional[torch.Tensor]:
        """img_2d: (B, 3, 224, 224) normalized to ImageNet stats.
        Returns (B, out_dim) pooled feature or None if teacher is disabled."""
        return None


class SAM2ImageEncoderTeacher(_TeacherBase):
    """Uses MedSAM2 / SAM2 image encoder (we already have this in the codebase).

    We reuse `models/medsam2_encoder.py` to keep dependencies minimal. Pools
    the output feature map with global average to get (B, C_t).
    """
    name = "sam2"

    def __init__(self, ckpt_path: Optional[str] = None):
        super().__init__()
        self.encoder: Optional[nn.Module] = None
        self.out_dim = 256  # placeholder; set when we load
        try:
            from models.medsam2_encoder import MedSAM2Encoder  # lazy import
            self.encoder = MedSAM2Encoder(ckpt_path=ckpt_path)
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.out_dim = getattr(self.encoder, "embed_dim", 256)
            self._ok = True
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"[TeacherDistill] SAM2 teacher unavailable: {e}")
            self._ok = False

    def is_available(self) -> bool:
        return self._ok and self.encoder is not None

    @torch.no_grad()
    def forward(self, img_2d: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.is_available():
            return None
        # MedSAM2 expects (B, 3, 1024, 1024) but 224 is fine for distillation —
        # features are pooled and used as regression targets.
        try:
            feats = self.encoder(img_2d)
            if isinstance(feats, dict):
                feats = feats.get("image_embed", next(iter(feats.values())))
            return feats.flatten(2).mean(-1)  # (B, C_t)
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"[TeacherDistill] SAM2 forward failed: {e}")
            return None


class DINOv2Teacher(_TeacherBase):
    """DINOv2-ViT-L/14 via torch.hub. Returns CLS token (B, 1024)."""
    name = "dinov2"

    def __init__(self, variant: str = "dinov2_vitl14"):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.out_dim = 1024
        try:
            # torch.hub will hit the internet on first call; cache is used on reuse.
            self.model = torch.hub.load("facebookresearch/dinov2", variant,
                                         trust_repo=True, verbose=False)
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
            self.out_dim = self.model.embed_dim
            self._ok = True
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"[TeacherDistill] DINOv2 teacher unavailable: {e}")
            self._ok = False

    def is_available(self) -> bool:
        return self._ok and self.model is not None

    @torch.no_grad()
    def forward(self, img_2d: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.is_available():
            return None
        try:
            # DINOv2 needs 14-multiple input; 224 is fine.
            return self.model(img_2d)  # (B, embed_dim)
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"[TeacherDistill] DINOv2 forward failed: {e}")
            return None


class BiomedCLIPTeacher(_TeacherBase):
    """BiomedCLIP via open_clip. Returns image-encoder pooled token (B, 512)."""
    name = "biomedclip"

    def __init__(self,
                 pretrained: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.out_dim = 512
        try:
            import open_clip  # lazy import
            self.model, _, _ = open_clip.create_model_and_transforms(pretrained)
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
            self._ok = True
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"[TeacherDistill] BiomedCLIP teacher unavailable: {e}")
            self._ok = False

    def is_available(self) -> bool:
        return self._ok and self.model is not None

    @torch.no_grad()
    def forward(self, img_2d: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.is_available():
            return None
        try:
            feats = self.model.encode_image(img_2d)
            return feats
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"[TeacherDistill] BiomedCLIP forward failed: {e}")
            return None


# ---------------------------------------------------------------------------
# Main distillation loss
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _prep_teacher_input(volume: torch.Tensor, center_slice: Optional[torch.Tensor] = None,
                        size: int = 224) -> torch.Tensor:
    """Convert a (B, 1, D, H, W) CT patch into (B, 3, size, size) normalized for teachers.

    Picks the center slice (or per-sample slice if given), replicates to RGB,
    and resizes with bilinear.
    """
    B, _, D, H, W = volume.shape
    if center_slice is None:
        z = torch.full((B,), D // 2, device=volume.device, dtype=torch.long)
    else:
        z = center_slice.clamp(0, D - 1).to(volume.device, dtype=torch.long)
    # Gather per-sample center slices.
    idx = z.view(B, 1, 1, 1, 1).expand(-1, 1, 1, H, W)
    slc = volume.gather(2, idx).squeeze(2)              # (B, 1, H, W)
    slc = slc.expand(-1, 3, -1, -1)                      # (B, 3, H, W)
    slc = F.interpolate(slc, size=(size, size), mode="bilinear", align_corners=False)
    mean = _IMAGENET_MEAN.to(slc.device, slc.dtype)
    std = _IMAGENET_STD.to(slc.device, slc.dtype)
    return (slc - mean) / std


class _ProjHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class TeacherDistillLoss(nn.Module):
    """Novel #3 — multi-teacher distillation with organ-adaptive weighting.

    Usage:
      loss_mod = TeacherDistillLoss(student_taps={
          "layer1": student_ch_at_tap1,
          "layer2": student_ch_at_tap2,
          "layer3": student_ch_at_tap3,
      }, teachers=["sam2", "dinov2", "biomedclip"], n_organs=15)
      l = loss_mod(student_feats, volume, organ_id)

    `student_feats`: Dict[tap_name, Tensor(B, C_s, D_f, H_f, W_f)] from the
                     SwinUNETR distill hooks (or any intermediate features).
    `volume`:        raw (B, 1, D, H, W) CT — used to build teacher inputs.
    `organ_id`:      (B,) long.
    """

    def __init__(
        self,
        student_taps: Dict[str, int],
        teachers: Sequence[str] = ("sam2", "dinov2", "biomedclip"),
        n_organs: int = 15,
        teacher_ckpts: Optional[Dict[str, str]] = None,
        teacher_input_size: int = 224,
    ):
        super().__init__()
        self.teacher_input_size = teacher_input_size
        self.n_organs = n_organs
        self.tap_names = list(student_taps.keys())
        teacher_ckpts = teacher_ckpts or {}

        # Build teachers (each may fail silently and become unavailable).
        self.teachers = nn.ModuleDict()
        for t in teachers:
            if t == "sam2":
                self.teachers[t] = SAM2ImageEncoderTeacher(ckpt_path=teacher_ckpts.get("sam2"))
            elif t == "dinov2":
                self.teachers[t] = DINOv2Teacher()
            elif t == "biomedclip":
                self.teachers[t] = BiomedCLIPTeacher()
            else:
                warnings.warn(f"[TeacherDistill] unknown teacher name '{t}' skipped")

        active = [n for n, t in self.teachers.items() if t.is_available()]
        print(f"[TeacherDistill] active teachers: {active} / requested {list(teachers)}")
        self._active = active

        # Projection heads: student tap dim → each teacher's dim.
        self.proj = nn.ModuleDict()
        for tap_name, s_dim in student_taps.items():
            for t_name in self._active:
                key = f"{tap_name}__{t_name}"
                self.proj[key] = _ProjHead(s_dim, self.teachers[t_name].out_dim)

        # Organ-adaptive teacher weight matrix (n_organs, n_active_teachers).
        # Initialized uniform; softmaxed on access.
        n_active = max(len(self._active), 1)
        self.organ_weights = nn.Parameter(torch.zeros(n_organs, n_active))

    def is_noop(self) -> bool:
        return len(self._active) == 0

    def _teacher_features(self, volume: torch.Tensor,
                          center_slice: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Run all active teachers; return {teacher_name: (B, C_t)} detached."""
        out: Dict[str, torch.Tensor] = {}
        if self.is_noop():
            return out
        img = _prep_teacher_input(volume, center_slice, size=self.teacher_input_size)
        for name in self._active:
            t = self.teachers[name]
            with torch.no_grad():
                f = t(img)
            if f is not None:
                out[name] = f.detach()
        return out

    def forward(
        self,
        student_feats: Dict[str, torch.Tensor],
        volume: torch.Tensor,
        organ_id: torch.Tensor,
        center_slice: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Returns {'loss': scalar, 'per_teacher': {name: scalar}}."""
        if self.is_noop() or not student_feats:
            return {"loss": volume.new_zeros(()), "per_teacher": {}}

        teacher_feats = self._teacher_features(volume, center_slice)
        if not teacher_feats:
            return {"loss": volume.new_zeros(()), "per_teacher": {}}

        B = volume.shape[0]
        # Organ weights → (B, n_active)
        w_all = F.softmax(self.organ_weights, dim=-1)         # (n_organs, n_active)
        w_b = w_all[organ_id.clamp(0, self.n_organs - 1)]     # (B, n_active)

        per_teacher: Dict[str, torch.Tensor] = {}
        tap_teacher_losses: List[torch.Tensor] = []
        for tap_name, feat in student_feats.items():
            # Pool student feature over spatial dims to (B, C_s).
            if feat.dim() == 5:                               # (B, C, D, H, W)
                s_pooled = feat.flatten(2).mean(-1)
            elif feat.dim() == 4:                             # (B, C, H, W)
                s_pooled = feat.flatten(2).mean(-1)
            elif feat.dim() == 3:                             # (B, L, C)
                s_pooled = feat.mean(dim=1)
            else:
                s_pooled = feat.flatten(1)

            for ti, t_name in enumerate(self._active):
                if t_name not in teacher_feats:
                    continue
                proj = self.proj[f"{tap_name}__{t_name}"]
                s_proj = proj(s_pooled)                        # (B, C_t)
                t_feat = teacher_feats[t_name]                 # (B, C_t)
                # Align shapes if teacher dim drifted (sanity).
                if s_proj.shape[-1] != t_feat.shape[-1]:
                    d = min(s_proj.shape[-1], t_feat.shape[-1])
                    s_proj, t_feat = s_proj[..., :d], t_feat[..., :d]
                cos = F.cosine_similarity(s_proj, t_feat, dim=-1)   # (B,)
                per_sample = (1.0 - cos) * w_b[:, ti]
                tap_teacher_losses.append(per_sample.mean())
                key = f"{tap_name}__{t_name}"
                per_teacher[key] = tap_teacher_losses[-1].detach()

        if not tap_teacher_losses:
            return {"loss": volume.new_zeros(()), "per_teacher": {}}
        loss = torch.stack(tap_teacher_losses).mean()
        return {"loss": loss, "per_teacher": per_teacher}
