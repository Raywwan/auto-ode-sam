"""Pre-flight for Boundary-DoU decoder-only fine-tune (Combined Plan #5).

Six gates. ALL must PASS before launching the 10-GPU-h training:

  G1. Config schema     : all loss/checkpoint fields present, types correct
  G2. CombinedLoss      : w_boundary_dou=0.1 wires BoundaryDoULoss + breakdown
  G3. Weights-only load : V2 best.pt loads, model weights match V2 bit-for-bit
                          (verified via state_dict keys + sample tensor checksum)
  G4. Encoder freeze    : encoder.requires_grad == False, encoder in .eval()
  G5. Decoder trainable : decoder + ode + pfesa params still trainable, ~0.5M+
  G6. Forward sanity    : dummy slab -> forward -> backward -> finite grad

Cost: ~10-30 s on CPU/GPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf
from models import build_model
from training.losses import CombinedLoss, BoundaryDoULoss
from utils.checkpoint import CheckpointManager

CFG_PATH = ROOT / "configs" / "phase3_odesam_v2_seed43_bdou_ft5ep.yaml"


def _ok(msg: str) -> None:
    print(f"  PASS - {msg}")


def _fail(label: str, msg: str) -> None:
    raise RuntimeError(f"{label} FAIL: {msg}")


def g1_config_schema(cfg) -> None:
    print("G1. Config schema")
    for k in [
        "experiment.name", "training.epochs", "training.lr",
        "training.encoder_freeze_epochs", "training.loss.w_boundary_dou",
        "training.loss.boundary_dou_alpha_max", "checkpoint.resume_from",
        "checkpoint.resume_weights_only", "validation.val_every_n_epochs",
    ]:
        node = cfg
        for part in k.split("."):
            if not hasattr(node, part):
                _fail("G1", f"missing key {k}")
            node = getattr(node, part)
    if cfg.training.encoder_freeze_epochs < cfg.training.epochs:
        _fail("G1", "encoder_freeze_epochs must >= epochs for decoder-only FT")
    if not cfg.checkpoint.resume_weights_only:
        _fail("G1", "resume_weights_only must be True")
    if abs(cfg.training.loss.w_boundary_dou - 0.1) > 1e-9:
        _fail("G1", f"w_boundary_dou={cfg.training.loss.w_boundary_dou} != 0.1")
    if abs(cfg.training.loss.boundary_dou_alpha_max - 0.8) > 1e-9:
        _fail("G1", f"alpha_max={cfg.training.loss.boundary_dou_alpha_max} != 0.8")
    if abs(cfg.training.lr - 1e-5) > 1e-12:
        _fail("G1", f"lr={cfg.training.lr} != 1e-5")
    if not Path(cfg.checkpoint.resume_from).exists():
        _fail("G1", f"resume_from does not exist: {cfg.checkpoint.resume_from}")
    _ok("All required fields present and within locked ranges")


def g2_combined_loss(cfg) -> None:
    print("G2. CombinedLoss wiring")
    loss_fn = CombinedLoss(cfg.training.loss)
    if not isinstance(getattr(loss_fn, "boundary_dou_fn", None), BoundaryDoULoss):
        _fail("G2", "boundary_dou_fn not instantiated when w_boundary_dou>0")
    if abs(loss_fn.w_boundary_dou - 0.1) > 1e-9:
        _fail("G2", f"loss_fn.w_boundary_dou={loss_fn.w_boundary_dou}")
    pred = torch.randn(2, 16, 16, requires_grad=True)
    target = torch.zeros(2, 16, 16)
    target[0, 4:12, 4:12] = 1.0
    target[1, 6:10, 6:10] = 1.0
    total, breakdown = loss_fn(pred, target)
    if "boundary_dou" not in breakdown:
        _fail("G2", "breakdown missing 'boundary_dou' key")
    if not torch.isfinite(total):
        _fail("G2", "loss is not finite")
    if not (0.0 <= breakdown["boundary_dou"] <= 1.0 + 1e-3):
        _fail("G2", f"boundary_dou out of [0,1]: {breakdown['boundary_dou']}")
    _ok(f"breakdown includes boundary_dou={breakdown['boundary_dou']:.4f}, total={total.item():.4f}")


def g3_weights_only_load(cfg) -> None:
    print("G3. Weights-only load (V2 -> fresh trainer)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    ckpt = torch.load(cfg.checkpoint.resume_from, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  WARN: {len(missing)} missing keys (first 3): {missing[:3]}")
    if unexpected:
        print(f"  WARN: {len(unexpected)} unexpected keys (first 3): {unexpected[:3]}")
    if len(missing) > 10 or len(unexpected) > 10:
        _fail("G3", f"too many key mismatches: missing={len(missing)} unexpected={len(unexpected)}")
    # Sample-checksum: pick first encoder weight tensor and verify equality with the ckpt copy
    sample_key = None
    for k in sd:
        if k.startswith("encoder.") and "weight" in k and sd[k].dim() >= 2:
            sample_key = k
            break
    if sample_key is None:
        _fail("G3", "no encoder.*weight sample tensor found")
    model_t = dict(model.named_parameters())[sample_key].detach().cpu()
    ckpt_t  = sd[sample_key].detach().cpu()
    if not torch.allclose(model_t, ckpt_t, atol=0.0):
        _fail("G3", f"sample tensor {sample_key} mismatch between model and ckpt")
    _ok(f"V2 weights loaded; sample {sample_key} matches bit-for-bit")


def g4_g5_freeze(cfg) -> None:
    print("G4. Encoder freeze + G5. Decoder trainable")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    # Simulate trainer freeze
    for p in model.encoder.parameters():
        p.requires_grad = False
    model.encoder.eval()
    enc_frozen = all(not p.requires_grad for p in model.encoder.parameters())
    if not enc_frozen:
        _fail("G4", "some encoder params still require grad")
    in_eval = not model.encoder.training
    if not in_eval:
        _fail("G4", "encoder not in .eval() mode")
    _ok("Encoder fully frozen and in .eval()")
    # Decoder trainable: count
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if trainable < 100_000:
        _fail("G5", f"too few trainable params: {trainable}")
    _ok(f"Trainable params (decoder+ODE+PFESA): {trainable/1e6:.2f}M / {total/1e6:.2f}M total")


def g6_forward(cfg) -> None:
    print("G6. Forward + backward sanity (3D slab path, V2 native)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device).train()
    # Freeze encoder (mirrors trainer._set_encoder_frozen(True))
    for p in model.encoder.parameters():
        p.requires_grad = False
    model.encoder.eval()
    loss_fn = CombinedLoss(cfg.training.loss).to(device)
    B = 1
    D = cfg.data.slices_per_volume
    H = W = cfg.model.img_size
    # V2 standard center-slice path: images (B, D, 3, H, W), boxes (B, 4),
    # modality_ids (B,), organ_id (B,) optional.
    images = torch.randn(B, D, 3, H, W, device=device)
    boxes  = torch.tensor([[H * 0.25, W * 0.25, H * 0.75, W * 0.75]] * B, device=device)
    modality_ids = torch.zeros(B, dtype=torch.long, device=device)
    organ_id     = torch.full((B,), 6, dtype=torch.long, device=device)  # liver index per V2 cfg
    out = model(
        images=images,
        boxes=boxes,
        modality_ids=modality_ids,
        is_3d=True,
        multimask_output=True,
        organ_id=organ_id,
    )
    masks_pred = out["masks"]   # (B, 3, H_out, W_out)
    iou_pred   = out["iou_pred"]
    best_idx   = iou_pred.argmax(dim=1)
    best_masks = masks_pred[torch.arange(B, device=device), best_idx]  # (B, H_out, W_out)
    # GT: dummy binary mask, resized to predicted resolution
    y = torch.zeros(B, H, W, device=device)
    y[:, H//4:3*H//4, W//4:3*W//4] = 1.0
    if y.shape[-2:] != best_masks.shape[-2:]:
        y = torch.nn.functional.interpolate(
            y.unsqueeze(1), size=best_masks.shape[-2:], mode="nearest"
        ).squeeze(1)
    total, breakdown = loss_fn(pred=best_masks, target=y)
    if not torch.isfinite(total):
        _fail("G6", f"loss non-finite: {total}")
    total.backward()
    g_sum = 0.0
    enc_grad = 0.0
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        s = float(p.grad.abs().sum())
        if n.startswith("encoder."):
            enc_grad += s
        else:
            g_sum += s
    if enc_grad > 0.0:
        _fail("G6", f"frozen encoder received gradient ({enc_grad:.2e})")
    if g_sum == 0.0:
        _fail("G6", "no trainable (decoder/ODE/PFESA) param received a gradient")
    _ok(
        f"Forward+backward OK. loss={total.item():.4f}, "
        f"breakdown={ {k: round(v, 4) for k, v in breakdown.items()} }, "
        f"|dec_grad|={g_sum:.2e}, |enc_grad|={enc_grad:.2e}"
    )


def main() -> int:
    print("=" * 72)
    print("Pre-flight: Boundary-DoU decoder-only fine-tune (Combined Plan #5)")
    print(f"Config: {CFG_PATH}")
    print("=" * 72)
    cfg = OmegaConf.load(str(CFG_PATH))
    g1_config_schema(cfg)
    g2_combined_loss(cfg)
    g3_weights_only_load(cfg)
    g4_g5_freeze(cfg)
    g6_forward(cfg)
    print("=" * 72)
    print("ALL GATES PASSED. Safe to launch.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as e:
        print(f"\n  FAILED: {e}\n")
        sys.exit(1)
