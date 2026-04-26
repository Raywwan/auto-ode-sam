"""V9 Stage-1 proposer: a 3D SwinUNETR that produces a coarse 15-organ
probability volume from a full CT.

Design notes:
  - Wraps MONAI's SwinUNETR (it ships pretrained weights for medical imaging).
  - Output is (B, K, 96, 96, 96) logits at the patch resolution. Sliding-window
    inference in eval time stitches patches into full volumes.
  - Supports loading MONAI's public BTCV/AMOS22 pretraining for warm-start.
  - Exposes intermediate encoder features so the teacher-ensemble distillation
    loss (Novel #3) can target layer-wise features.

This module is used as the **proposer** in the V9 cascade; the refiner
(OrganFlowSAM2) reads its output both as input (ROI gate) and for the cascade
consistency loss.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from monai.networks.nets import SwinUNETR as _MONAISwinUNETR
    _MONAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MONAI_AVAILABLE = False
    _MONAISwinUNETR = None


class SwinUNETRProposer(nn.Module):
    """3D SwinUNETR wrapper that emits (B, n_organs, D, H, W) probabilities.

    Args:
        n_organs: number of organ classes (AMOS22: 15, no background channel).
                 An extra background channel is added internally and softmax'd
                 across all K+1 classes; only the organ channels are returned.
        patch_size: model input patch size (96 is MONAI default).
        feature_size: SwinUNETR base feature width (24 → 24M params, 48 → 60M).
        use_v2: use SwinUNETR v2 (residual blocks, ~5 % gain on AMOS22).
        pretrained_weights: optional path to MONAI pretrained SwinUNETR weights
                            (AMOS22 or BTCV). Loaded with strict=False.
        lora_rank: if >0, freeze the swinViT encoder and inject LoRA adapters
                   of the given rank into every qkv/proj/fc1/fc2 Linear inside
                   swinViT. Decoder stays fully trainable. Required for 1B+
                   models (VoCo-H) on 24 GB 4090.
    """

    def __init__(
        self,
        n_organs: int = 15,
        patch_size: int = 96,
        feature_size: int = 48,
        use_v2: bool = True,
        pretrained_weights: Optional[str] = None,
        deep_supervision: bool = False,
        deep_sup_layers: Tuple[str, ...] = ("decoder3", "decoder4", "decoder5"),
        lora_rank: int = 0,
        lora_alpha: Optional[float] = None,
        use_organmoe: bool = False,
        moe_n_experts: int = 8,
        moe_rank: int = 16,
        moe_alpha: float = 16.0,
        moe_top_k: int = 2,
    ) -> None:
        super().__init__()
        if not _MONAI_AVAILABLE:
            raise RuntimeError(
                "MONAI is not installed; cannot build SwinUNETRProposer. "
                "Install via `pip install monai`."
            )
        self.n_organs = n_organs
        self.patch_size = patch_size
        self.feature_size = feature_size

        self.net = _MONAISwinUNETR(
            in_channels=1,
            out_channels=n_organs + 1,          # +1 background
            feature_size=feature_size,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=True,
            use_v2=use_v2,
            spatial_dims=3,
        )

        self._distill_hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._distill_features: Dict[str, torch.Tensor] = {}

        # Deep supervision: capture intermediate decoder feature maps via
        # forward hooks, then apply a 1x1x1 conv head per layer to produce
        # auxiliary logits at reduced resolution. Heads are created lazily on
        # first forward so we don't need to know the MONAI internal channel
        # sizes up front.
        self.deep_supervision = bool(deep_supervision)
        self.deep_sup_layers = tuple(deep_sup_layers)
        self._aux_hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._aux_features: Dict[str, torch.Tensor] = {}
        self.aux_heads = nn.ModuleDict()
        if self.deep_supervision:
            for name in self.deep_sup_layers:
                mod = self._resolve(name)
                if mod is None:
                    print(f"[SwinUNETRProposer] WARN: deep-sup layer `{name}` not found, skipped")
                    continue
                h = mod.register_forward_hook(self._make_aux_hook(name))
                self._aux_hooks.append(h)

        if pretrained_weights is not None:
            self.load_pretrained(pretrained_weights)

        # LoRA: freeze encoder + inject low-rank adapters. Decoder stays
        # trainable. Run AFTER pretrained load so the frozen base carries the
        # pretrained weights, not random init.
        self.lora_rank = int(lora_rank)
        if self.lora_rank > 0:
            alpha = float(lora_alpha) if lora_alpha is not None else float(lora_rank)
            for p in self.net.swinViT.parameters():
                p.requires_grad = False
            n_inj = _inject_lora_into_swin(self.net.swinViT, rank=self.lora_rank, alpha=alpha)
            n_lora = sum(p.numel() for p in self.net.swinViT.parameters() if p.requires_grad)
            n_dec = sum(
                p.numel() for n, p in self.net.named_parameters()
                if not n.startswith("swinViT.") and p.requires_grad
            )
            print(f"[SwinUNETRProposer] LoRA enabled: rank={self.lora_rank} alpha={alpha}")
            print(f"[SwinUNETRProposer]   injected into {n_inj} Linear layers")
            print(f"[SwinUNETRProposer]   trainable LoRA params: {n_lora/1e6:.2f}M")
            print(f"[SwinUNETRProposer]   trainable decoder params: {n_dec/1e6:.2f}M")

        # OrganMoE-3D: replace plain LoRA with sparse-MoE LoRA experts.
        # Mutually exclusive with lora_rank>0 plain-LoRA injection.
        if use_organmoe and self.lora_rank > 0:
            raise ValueError("use_organmoe=True is mutually exclusive with lora_rank>0; pick one.")
        if use_organmoe:
            from models.organmoe_3d import inject_organmoe_into_swin, PresenceHead
            for p in self.net.swinViT.parameters():
                p.requires_grad = False
            n_replaced = inject_organmoe_into_swin(
                self.net.swinViT,
                n_experts=int(moe_n_experts),
                rank=int(moe_rank),
                alpha=float(moe_alpha),
                n_organs=int(n_organs),
                top_k=int(moe_top_k),
            )
            self._n_organmoe_modules = int(n_replaced)
            deepest_ch = int(self.net.swinViT.embed_dim) * 16
            self.presence_head = PresenceHead(in_channels=deepest_ch, n_organs=int(n_organs))
            self._use_organmoe = True
            print(f"[SwinUNETRProposer] OrganMoE enabled: n_experts={moe_n_experts} rank={moe_rank} "
                  f"alpha={moe_alpha} top_k={moe_top_k}")
            print(f"[SwinUNETRProposer]   replaced {n_replaced} Linear layers with MoELoRALinear")
            print(f"[SwinUNETRProposer]   presence_head in_channels={deepest_ch}")
        else:
            self._use_organmoe = False
            self._n_organmoe_modules = 0
            self.presence_head = None

    def load_pretrained(self, path: str, strict: bool = False) -> None:
        """Load MONAI-format pretrained weights.

        Supports two checkpoint flavors:
        1. **SSL SwinViT backbone** (`module.patch_embed...`) from the MONAI
           self-supervised pretrain release. Keys are remapped to `swinViT.*`
           so they land inside `self.net.swinViT`. Decoder + head stay random.
        2. **Full SwinUNETR checkpoint** (keys already prefixed with
           `swinViT.` or already rooted at the SwinUNETR module). Loaded
           directly; the 16-class output head is mismatched with strict=False.
        """
        state = torch.load(path, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state.get("model_state_dict", state))

        # Strip leading "module." (DDP artifact).
        state = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in state.items()
        }

        # Detect SSL backbone format (all keys look like `patch_embed.*`,
        # `layers1.*`, etc.) and remap to `swinViT.*`.
        sample_keys = list(state.keys())[:10]
        looks_ssl = (
            not any(k.startswith("swinViT.") for k in sample_keys)
            and any(k.startswith(("patch_embed", "layers1", "layers2", "layers3", "layers4")) for k in sample_keys)
        )
        if looks_ssl:
            # Drop SSL-only heads that cannot map into SwinUNETR.
            drop_prefixes = ("rotation_head.", "contrastive_head.", "mask_token", "convTrans3d.")
            state = {
                f"swinViT.{k}": v
                for k, v in state.items()
                if not k.startswith(drop_prefixes)
            }
            # MONAI renamed MLP blocks: fc1/fc2 → linear1/linear2.
            remap = {}
            for k, v in state.items():
                nk = k.replace(".mlp.fc1.", ".mlp.linear1.").replace(".mlp.fc2.", ".mlp.linear2.")
                remap[nk] = v
            state = remap

        # Drop keys whose shape doesn't match the target — happens when the
        # checkpoint uses feature_size != current model (e.g. SSL is fs=48 but
        # the smoke uses fs=24). strict=False only handles missing keys, not
        # shape mismatches, so we must filter here.
        target = self.net.state_dict()
        kept, dropped_shape = {}, 0
        for k, v in state.items():
            if k in target and target[k].shape == v.shape:
                kept[k] = v
            else:
                dropped_shape += int(k in target)
        state = kept

        missing, unexpected = self.net.load_state_dict(state, strict=strict)
        # Count how many encoder params actually got filled.
        filled = sum(
            1 for k in self.net.state_dict()
            if k.startswith("swinViT.") and k in state
        )
        total_enc = sum(1 for k in self.net.state_dict() if k.startswith("swinViT."))
        print(f"[SwinUNETRProposer] loaded {path}")
        print(f"[SwinUNETRProposer]   encoder keys filled: {filled}/{total_enc}   shape-dropped: {dropped_shape}")
        print(f"[SwinUNETRProposer]   missing={len(missing)} unexpected={len(unexpected)}")

    def forward(self, volume: torch.Tensor) -> Dict[str, torch.Tensor]:
        """volume: (B, 1, D, H, W) HU-clipped float32. D=H=W=patch_size for train."""
        if getattr(self, "_use_organmoe", False):
            from models.organmoe_3d import MoELoRALinear
            stage_features = self.net.swinViT(volume)
            deep_feat = stage_features[-1]
            pres_logits = self.presence_head(deep_feat)
            organ_presence = torch.sigmoid(pres_logits).detach()
            self._last_presence_logits = pres_logits
            for m in self.modules():
                if isinstance(m, MoELoRALinear):
                    m._cached_presence = organ_presence
        self._aux_features.clear()
        logits = self.net(volume)                                    # (B, K+1, D, H, W)
        probs = F.softmax(logits, dim=1)
        organ_probs = probs[:, 1:]                                  # drop bg
        organ_logits = logits[:, 1:]
        aux_logits: Dict[str, torch.Tensor] = {}
        if self.deep_supervision:
            for name, feat in self._aux_features.items():
                # ModuleDict keys must be str. Layer dotted names contain '.'
                # which ModuleDict disallows, so swap to '__'.
                key = name.replace(".", "__")
                head = self.aux_heads[key] if key in self.aux_heads else None
                if head is None:
                    c_in = feat.shape[1]
                    # Create in float32 (model's natural dtype) — autocast wraps the
                    # forward op; mixing bf16 weights with fp32 inputs in eval mode
                    # otherwise raises a dtype mismatch.
                    head = nn.Conv3d(c_in, self.n_organs + 1, kernel_size=1).to(
                        device=feat.device
                    )
                    nn.init.normal_(head.weight, std=1e-3)
                    nn.init.zeros_(head.bias)
                    self.aux_heads[key] = head
                aux_logits[name] = head(feat)
        return {
            "logits": organ_logits,                                  # (B, K, D, H, W)
            "probs": organ_probs,                                    # (B, K, D, H, W)
            "bg_prob": probs[:, 0:1],                                # (B, 1, D, H, W)
            "full_logits": logits,                                   # (B, K+1, ...) for CE
            "full_probs": probs,
            "distill_feats": dict(self._distill_features),           # may be empty
            "aux_logits": aux_logits,                                # may be empty
        }

    @property
    def last_presence_logits(self):
        """Most recent presence-head output (B, n_organs). None if OrganMoE off."""
        return getattr(self, "_last_presence_logits", None)

    def collect_last_gates(self):
        """Mean gate weights across all MoELoRALinear modules. None if OrganMoE off."""
        from models.organmoe_3d import MoELoRALinear
        gates = []
        for m in self.modules():
            if isinstance(m, MoELoRALinear) and getattr(m, "_last_gate", None) is not None:
                g = m._last_gate
                gates.append(g.mean(dim=-1, keepdim=True))
        if not gates:
            return None
        return torch.stack(gates, dim=0).mean(dim=0)

    # -- teacher-distillation hook management (Novel #3) --------------------

    def register_distill_hooks(self, layer_names: Tuple[str, ...] = (
        "swinViT.layers1.blocks.1.mlp",
        "swinViT.layers2.blocks.1.mlp",
        "swinViT.layers3.blocks.1.mlp",
    )) -> None:
        """Register forward hooks on selected encoder layers to capture their
        activations for teacher-distillation feature matching. Call once after
        module init; features populate `out['distill_feats']` on each forward."""
        self.clear_distill_hooks()
        for name in layer_names:
            mod = self._resolve(name)
            if mod is None:
                print(f"[SwinUNETRProposer] WARN: distill layer `{name}` not found, skipped")
                continue
            h = mod.register_forward_hook(self._make_hook(name))
            self._distill_hooks.append(h)

    def clear_distill_hooks(self) -> None:
        for h in self._distill_hooks:
            h.remove()
        self._distill_hooks.clear()
        self._distill_features.clear()

    def _resolve(self, dotted: str) -> Optional[nn.Module]:
        mod: nn.Module = self.net
        for part in dotted.split("."):
            if part.isdigit():
                mod = mod[int(part)]
            else:
                mod = getattr(mod, part, None)
                if mod is None:
                    return None
        return mod

    def _make_hook(self, key: str):
        def _hook(_mod, _inp, out):
            self._distill_features[key] = out
        return _hook

    def _make_aux_hook(self, key: str):
        def _hook(_mod, _inp, out):
            self._aux_features[key] = out
        return _hook

    def clear_aux_hooks(self) -> None:
        for h in self._aux_hooks:
            h.remove()
        self._aux_hooks.clear()
        self._aux_features.clear()
        self.deep_supervision = False

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


class _SwinLoRALinear(nn.Module):
    """LoRA-adapted nn.Linear. y = W x + (B A) x * (alpha / r). Base W is frozen."""

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = rank
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale


def _inject_lora_into_swin(module: nn.Module, rank: int, alpha: float) -> int:
    """Replace nn.Linear named qkv/proj/fc1/fc2 inside swinViT with LoRA wrappers."""
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in ("qkv", "proj", "fc1", "fc2"):
            setattr(module, name, _SwinLoRALinear(child, rank=rank, alpha=alpha))
            n += 1
        else:
            n += _inject_lora_into_swin(child, rank, alpha)
    return n
