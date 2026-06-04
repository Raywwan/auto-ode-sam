"""
Diagnose why val_dice ~= 0 at ep5 despite loss descending.

Plan T11 Step 4 checks:
  (a) _init_adjacency buffer loaded correctly
  (b) LoRA init correct (B was zero at start; check nonzero now)
  (c) flow_targets magnitudes

Extras:
  (d) val path asymmetry: replay the trainer's val logic on 10 samples,
      reporting per-sample sigmoid range, threshold-crossing pixels,
      the picked organ_id, and per-organ train-loss-equivalent dice
      (all present organs, not just first).
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from omegaconf import OmegaConf
from models.organflow_sam2 import OrganFlowSAM2
from datasets.amos22_multiorgan import AMOS22MultiOrgan3D_Dataset

CKPT = r"C:\Users\Raywa\Desktop\VoluFormer3D_V4\checkpoints\v4_organflow_sam2_256px\v4_organflow_sam2_256px_epoch005.pt"
CFG  = r"C:\Users\Raywa\Desktop\VoluFormer3D_V4\configs\v4_organflow_sam2_256px.yaml"

print("=" * 70)
print("EP5 GO-GATE FAIL DIAGNOSTIC")
print("=" * 70)

cfg = OmegaConf.load(CFG)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\nLoading model from {CKPT}")
model = OrganFlowSAM2(cfg).to(device)
state = torch.load(CKPT, map_location=device, weights_only=False)
sd = state.get("model_state_dict", state.get("model", state))
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"  loaded, missing={len(missing)}, unexpected={len(unexpected)}")
model.eval()

# ------------------------------------------------------------------
# (a) _init_adjacency buffer
# ------------------------------------------------------------------
print("\n--- (a) _init_adjacency buffer ---")
dec = model.decoder
# The AnatomyGraphAttention layer is inside the transformer; find it
found = False
for name, mod in model.named_modules():
    if hasattr(mod, "_init_adjacency"):
        init_adj = mod._init_adjacency
        adj     = mod.adjacency
        offdiag = getattr(mod, "_offdiag_mask", None)
        print(f"  {name}: _init_adjacency shape={tuple(init_adj.shape)}, "
              f"mean={init_adj.mean().item():.4f}, "
              f"nonzero={int(init_adj.nonzero().numel()/2)}/{init_adj.numel()}")
        print(f"  {name}: learned adjacency mean={adj.mean().item():.4f}, "
              f"std={adj.std().item():.4f}, "
              f"|A - A_init| L2 = {(adj - init_adj).norm().item():.4f}")
        if offdiag is not None:
            print(f"  {name}: _offdiag_mask diag sum = {offdiag.diag().sum().item():.1f} "
                  f"(should be 0 after self-loop zeroing)")
        found = True
if not found:
    print("  WARN: no module with _init_adjacency found — check wiring")

# ------------------------------------------------------------------
# (b) LoRA init / training progress
# ------------------------------------------------------------------
print("\n--- (b) LoRA B weight magnitudes after ep5 ---")
lora_bs = []
lora_as = []
for name, p in model.named_parameters():
    if "lora_B" in name:
        lora_bs.append((name, p.detach().abs().mean().item(), p.detach().abs().max().item()))
    elif "lora_A" in name:
        lora_as.append((name, p.detach().abs().mean().item(), p.detach().abs().max().item()))
print(f"  {len(lora_as)} lora_A params, {len(lora_bs)} lora_B params")
if lora_bs:
    bmean = sum(x[1] for x in lora_bs) / len(lora_bs)
    bmax  = max(x[2] for x in lora_bs)
    zero_b = sum(1 for x in lora_bs if x[2] < 1e-8)
    print(f"  lora_B: mean(|B|)={bmean:.2e}, max(|B|)={bmax:.2e}, exactly_zero_count={zero_b}/{len(lora_bs)}")
if lora_as:
    amean = sum(x[1] for x in lora_as) / len(lora_as)
    amax  = max(x[2] for x in lora_as)
    print(f"  lora_A: mean(|A|)={amean:.2e}, max(|A|)={amax:.2e}")

# ------------------------------------------------------------------
# Load one val batch
# ------------------------------------------------------------------
print("\n--- loading val set ---")
data_cfg = OmegaConf.merge(cfg.data, OmegaConf.create({"split": "val"}))
# Manually construct the dataset matching datasets/__init__.py logic
ds = AMOS22MultiOrgan3D_Dataset(
    data_root=cfg.data.data_root,
    split="val",
    img_size=cfg.data.img_size,
    depth=cfg.data.slices_per_volume,
    modality=cfg.data.modality,
)
print(f"  val set: {len(ds)} volumes")
samples = [ds[i] for i in range(min(6, len(ds)))]

# ------------------------------------------------------------------
# (c) flow_targets magnitudes + (d) val-path forward pass
# ------------------------------------------------------------------
print("\n--- (c) flow_targets magnitudes + (d) val-path replay ---")
print(f"  {'sample':<8} {'organ':<6} {'pred_min':<10} {'pred_mean':<10} {'pred_max':<10} "
      f"{'>0.5 pct':<10} {'val_dice':<10} {'all_organ_dice':<18}")

with torch.no_grad():
    for i, s in enumerate(samples):
        img      = s["image"].unsqueeze(0).to(device)        # (1, D, 1, H, W)
        masks_gt = s["masks"].unsqueeze(0).to(device)        # (1, 15, D, H, W)
        present  = s["present_mask"].unsqueeze(0).to(device) # (1, 15) bool
        if img.shape[2] == 1:
            img = img.repeat(1, 1, 3, 1, 1)

        organ_id = torch.argmax(present.int(), dim=1) + 1    # trainer pattern
        out = model(img, organ_id, is_3d=True)

        pm = out["masks"]                                    # (1, 15, h, w)
        idx = (organ_id - 1).long()
        H, W = pm.shape[-2:]
        picked_logit = pm.gather(1, idx.view(-1,1,1,1).expand(-1,1,H,W)).squeeze(1)
        picked_prob  = torch.sigmoid(picked_logit)
        pct_above = (picked_prob > 0.5).float().mean().item() * 100

        # GT for picked organ, center slice
        D = masks_gt.shape[2]
        gt_center = masks_gt[:, :, D // 2].gather(
            1, idx.view(-1,1,1,1).expand(-1,1,*masks_gt.shape[-2:])
        ).squeeze(1).float()
        # Replicate trainer: downsample GT 256->pred size (64) with nearest
        Hp, Wp = picked_prob.shape[-2:]
        if gt_center.shape[-2:] != (Hp, Wp):
            gt_center = torch.nn.functional.interpolate(
                gt_center.unsqueeze(1), size=(Hp, Wp), mode="nearest"
            ).squeeze(1)
        # Single-organ binary dice (picked_prob thresholded at 0.5)
        pred_bin = (picked_prob > 0.5).float()
        inter = (pred_bin * gt_center).sum()
        denom = pred_bin.sum() + gt_center.sum()
        val_dice = (2 * inter / denom.clamp(min=1e-6)).item() if denom > 0 else float('nan')

        # All-present-organ dice (what train effectively supervises)
        all_probs = torch.sigmoid(pm)
        all_bin   = (all_probs > 0.5).float()
        gt_all    = masks_gt[:, :, D // 2].float()
        # Downsample GT to pred size to match trainer
        if gt_all.shape[-2:] != (Hp, Wp):
            gt_all = torch.nn.functional.interpolate(
                gt_all, size=(Hp, Wp), mode="nearest"
            )
        present_idx = present[0].nonzero(as_tuple=False).squeeze(-1).tolist()
        all_dices = []
        for oid in present_idx:
            p = all_bin[0, oid]
            g = gt_all[0, oid]
            inter_o = (p * g).sum()
            denom_o = p.sum() + g.sum()
            d_o = (2 * inter_o / denom_o.clamp(min=1e-6)).item() if denom_o > 0 else float('nan')
            all_dices.append(d_o)
        all_avg = sum(all_dices) / max(len(all_dices), 1)

        flow_targets = out.get("flow_targets", {})
        if i == 0 and flow_targets:
            for key in ("fwd_pairs", "bwd_pairs"):
                if key in flow_targets:
                    tgt = flow_targets[key]["target"]
                    prd = flow_targets[key]["pred"]
                    print(f"  flow_targets[{key}]: pred range "
                          f"[{prd.min().item():.3f}, {prd.max().item():.3f}], "
                          f"target range [{tgt.min().item():.3f}, {tgt.max().item():.3f}], "
                          f"MSE={((prd-tgt)**2).mean().item():.4f}")

        print(f"  s{i:<7} {organ_id.item():<6} "
              f"{picked_prob.min().item():<10.4f} "
              f"{picked_prob.mean().item():<10.4f} "
              f"{picked_prob.max().item():<10.4f} "
              f"{pct_above:<10.2f} "
              f"{val_dice:<10.4f} "
              f"{all_avg:<10.4f} ({len(present_idx)} organs)")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
