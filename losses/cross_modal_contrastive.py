"""V9 Novel #4 — Cross-modal InfoNCE alignment.

Aligns per-organ pooled image features with BiomedCLIP text embeddings of
organ names (e.g., "CT scan of a liver"). Intended to sharpen class-conditional
representations — a common failure mode on AMOS22 is a strong left-vs-right
confusion for paired organs (left/right kidney, left/right adrenal).

Novelty claim (spec §3, row #4): InfoNCE with *hard-negative mining across
organs*. Unlike plain image-text CLIP, we build the positive set from the
current batch's organ-pooled features and the negative set from the text
embeddings of the *most visually similar* distractor organs (confusion-pair
bias). This targets specifically the L/R symmetry failures.

Design discipline:
  - Text embeddings are precomputed once on first forward and cached.
    If BiomedCLIP is unavailable, the loss becomes a no-op (returns 0).
  - Config weight λ defaults to 0 → callers skip this module entirely.
  - The module does NOT own the image encoder. It accepts an organ-pooled
    feature tensor from the student model.

Inputs:
  organ_pooled: (B, C_img)  — per-sample feature pooled within the organ mask
                              (or a projection of one of the decoder heads).
  organ_id:     (B,)        — ground-truth organ index.

Output: {'loss': scalar, 'acc@1': scalar}.
"""
from __future__ import annotations

import warnings
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_ORGAN_NAMES: Sequence[str] = (
    "spleen",
    "right kidney",
    "left kidney",
    "gallbladder",
    "esophagus",
    "liver",
    "stomach",
    "aorta",
    "inferior vena cava",
    "pancreas",
    "right adrenal gland",
    "left adrenal gland",
    "duodenum",
    "bladder",
    "prostate or uterus",
)

# Known confusion pairs (indices into DEFAULT_ORGAN_NAMES) — prioritized as
# hard negatives. Pairs are bidirectional.
DEFAULT_CONFUSION_PAIRS = (
    (1, 2),    # right kidney ↔ left kidney
    (10, 11),  # right adrenal ↔ left adrenal
    (5, 6),    # liver ↔ stomach (boundary confusion)
    (7, 8),    # aorta ↔ IVC
    (9, 12),   # pancreas ↔ duodenum
    (3, 5),    # gallbladder ↔ liver
)


class CrossModalInfoNCE(nn.Module):
    """Novel #4 — per-organ image ↔ organ-text InfoNCE with hard negatives.

    Args:
        embed_dim: dim of the organ-pooled image feature incoming to `forward`.
        n_organs: number of organ classes (default 15 — AMOS22).
        organ_names: human-readable organ names used to build text embeddings.
        temperature: InfoNCE temperature.
        hard_negative_weight: upweight factor on confusion-pair negatives.
    """

    def __init__(
        self,
        embed_dim: int,
        n_organs: int = 15,
        organ_names: Sequence[str] = DEFAULT_ORGAN_NAMES,
        temperature: float = 0.07,
        hard_negative_weight: float = 2.0,
        text_template: str = "a CT scan of a {}",
    ):
        super().__init__()
        assert len(organ_names) == n_organs, "organ_names length must match n_organs"
        self.n_organs = n_organs
        self.organ_names = list(organ_names)
        self.text_template = text_template
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight

        # BiomedCLIP is lazy-loaded on first forward.
        self._clip = None
        self._tokenizer = None
        self._text_feats: Optional[torch.Tensor] = None       # (n_organs, C_t)
        self._available: Optional[bool] = None

        # Image-side projection head — maps (B, embed_dim) → (B, C_t).
        # C_t is filled in when we know BiomedCLIP's text dim; until then
        # we lazily build the Linear.
        self.embed_dim = embed_dim
        self.img_proj: Optional[nn.Linear] = None

        # Confusion-pair mask (n_organs, n_organs) of hard-negative weights.
        self.register_buffer("neg_weight", self._build_neg_weight())

    def _build_neg_weight(self) -> torch.Tensor:
        w = torch.ones(self.n_organs, self.n_organs)
        for a, b in DEFAULT_CONFUSION_PAIRS:
            if a < self.n_organs and b < self.n_organs:
                w[a, b] = self.hard_negative_weight
                w[b, a] = self.hard_negative_weight
        # Zero the diagonal; positives aren't "negatives".
        w.fill_diagonal_(0.0)
        return w

    def _try_load_clip(self, device: torch.device) -> bool:
        if self._available is not None:
            return self._available
        try:
            import open_clip
            model, _, _ = open_clip.create_model_and_transforms(
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            )
            tokenizer = open_clip.get_tokenizer(
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            )
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            model.to(device)
            self._clip = model
            self._tokenizer = tokenizer

            # Encode all organ names once.
            prompts = [self.text_template.format(n) for n in self.organ_names]
            tok = tokenizer(prompts).to(device)
            with torch.no_grad():
                t_feats = model.encode_text(tok)
                t_feats = F.normalize(t_feats, dim=-1)
            self._text_feats = t_feats.detach()
            self.img_proj = nn.Linear(self.embed_dim, t_feats.shape[-1]).to(device)
            self._available = True
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"[CrossModalInfoNCE] BiomedCLIP unavailable: {e}")
            self._available = False
        return self._available

    def is_noop(self) -> bool:
        return self._available is False

    def forward(
        self,
        organ_pooled: torch.Tensor,     # (B, embed_dim)
        organ_id: torch.Tensor,         # (B,)
    ) -> Dict[str, torch.Tensor]:
        device = organ_pooled.device
        if not self._try_load_clip(device):
            return {"loss": organ_pooled.new_zeros(()),
                    "acc@1": organ_pooled.new_zeros(())}

        assert self.img_proj is not None and self._text_feats is not None
        text_feats = self._text_feats.to(device)           # (K, C_t)
        img_feats = F.normalize(self.img_proj(organ_pooled), dim=-1)  # (B, C_t)

        # Logits: (B, K) = img @ text.T / τ
        logits = img_feats @ text_feats.t() / self.temperature

        # Apply hard-negative weighting. We do it by rescaling the non-target
        # logits (target index gets untouched; negatives multiplied by
        # neg_weight[target] before softmax exponentiation). Since multiplying
        # a logit is *not* identical to upweighting a probability, we instead
        # add log(neg_weight[...]) to non-target logits — equivalent to
        # multiplying their exp(logit) before normalization.
        B = organ_pooled.shape[0]
        idx = organ_id.clamp(0, self.n_organs - 1)
        # log-weights: same shape as logits
        nw = self.neg_weight.to(device)                    # (K, K)
        log_nw = torch.log(nw.clamp(min=1e-6))[idx]        # (B, K)
        # Zero-out weighting on the positive column (diagonal already 0 → log = -inf) —
        # restore the positive column to 0 so it isn't masked to -inf.
        log_nw[torch.arange(B, device=device), idx] = 0.0
        # For non-confusion pairs (weight=1 → log=0) this is a no-op.
        weighted_logits = logits + log_nw

        loss = F.cross_entropy(weighted_logits, idx)
        with torch.no_grad():
            acc = (weighted_logits.argmax(-1) == idx).float().mean()
        return {"loss": loss, "acc@1": acc}
