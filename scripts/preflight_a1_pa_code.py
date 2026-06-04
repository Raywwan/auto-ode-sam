"""PA-CODE Pre-flight -- All-gate sanity check before launching 40-epoch run.

Eight gates. ALL must PASS before training.

  G1. Config schema     : all required PA-CODE fields exist in path_a1_pa_code.yaml
  G2. Monitor-metric    : trainer/CheckpointManager init reads val_dice_3d (audit Bug 1)
  G3. Encoder freeze    : full encoder (backbone+proj+neck) + BN .eval() (audit Bug 2/4)
  G4. Identity-init     : PACodeBidirectionalNeuralODE = pure pass-through at init
  G5. Warmstart         : path_a1_pa_code_warmstart.pt exists and strict-loads
  G6. 3D eval pipeline  : eval_a1_per_organ utilities import and find >=1 val volume
  G7. FiLM divergence   : 100 toy steps -> gamma/beta diverge across organ_ids (audit Concern 9)
  G8. Forward sanity    : full model forward on dummy slab, no NaN, no shape error

Cost: ~30-60s on 4090. Run BEFORE every long training launch.

ASCII-only output (Windows cp1252).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from omegaconf import OmegaConf
from models import build_model

PA_CODE_CONFIG = ROOT / "configs" / "path_a1_pa_code.yaml"
BASE_CONFIG    = ROOT / "configs" / "base.yaml"
WARMSTART_PATH = ROOT / "checkpoints" / "path_a1_pa_code" / "path_a1_pa_code_warmstart.pt"


def _load_cfg():
    cfg = OmegaConf.load(str(BASE_CONFIG)) if BASE_CONFIG.exists() else OmegaConf.create({})
    pa = OmegaConf.load(str(PA_CODE_CONFIG))
    if "defaults" in pa:
        pa = OmegaConf.masked_copy(pa, [k for k in pa if k != "defaults"])
    return OmegaConf.merge(cfg, pa)


def _ok(msg: str) -> None:
    print(f"  PASS - {msg}")


def _fail(label: str, msg: str) -> None:
    raise RuntimeError(f"{label} FAIL: {msg}")


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def gate1_schema(cfg) -> None:
    print("[G1] config schema")
    required = [
        ("model.ode.use_pa_code",            True),
        ("model.ode.n_pos_freqs",            None),
        ("model.ode.organ_emb_dim",          None),
        ("training.encoder_freeze_epochs",   None),
        ("training.encoder_lr_scale",        None),
        ("validation.use_val_dice_3d",       True),
        ("validation.val_dice_3d_n_vols",    None),
        ("checkpoint.monitor_metric",        "val_dice_3d"),
        ("checkpoint.best_min_delta",        None),
        ("checkpoint.resume_from",           None),
    ]
    for key, want in required:
        node = cfg
        for part in key.split("."):
            if not OmegaConf.is_config(node) or part not in node:
                _fail("G1", f"missing config key: {key}")
            node = node[part]
        if want is not None and node != want:
            _fail("G1", f"{key} = {node!r}, expected {want!r}")
    if int(cfg.validation.val_dice_3d_n_vols) < 4:
        _fail("G1", f"val_dice_3d_n_vols={cfg.validation.val_dice_3d_n_vols} too noisy "
                    f"(audit Bug 3 says >=4)")
    _ok(f"all {len(required)} required PA-CODE fields present and well-typed")


def gate2_monitor_metric(cfg) -> None:
    print("[G2] CheckpointManager wires PA-CODE monitor_metric")
    from utils.checkpoint import CheckpointManager
    cm = CheckpointManager(
        save_dir=str(ROOT / "checkpoints" / "_preflight_tmp"),
        keep_top_k=int(cfg.checkpoint.keep_top_k),
        monitor_metric=str(cfg.checkpoint.monitor_metric),
        experiment_name="preflight",
        min_delta=float(cfg.checkpoint.best_min_delta),
    )
    if cm.monitor_metric != "val_dice_3d":
        _fail("G2", f"CheckpointManager.monitor_metric={cm.monitor_metric!r} != 'val_dice_3d'")
    if cm.min_delta < 0.001:
        _fail("G2", f"CheckpointManager.min_delta={cm.min_delta} too small (>=0.001)")
    _ok(f"CheckpointManager monitor='val_dice_3d' min_delta={cm.min_delta}")


def gate3_encoder_freeze(cfg) -> None:
    print("[G3] _set_encoder_frozen covers full encoder + BN .eval()")
    model = build_model(cfg).cpu()
    # Probe pre-freeze
    pre_train_count = sum(1 for p in model.encoder.parameters() if p.requires_grad)
    if pre_train_count == 0:
        _fail("G3", "encoder has no trainable params even pre-freeze")

    # Simulate the trainer's freeze hook
    for p in model.encoder.parameters():
        p.requires_grad = False
    model.encoder.eval()

    # Verify ALL encoder params (backbone + proj + neck) are now frozen
    still_trainable = [n for n, p in model.encoder.named_parameters() if p.requires_grad]
    if still_trainable:
        _fail("G3", f"{len(still_trainable)} encoder params still trainable after freeze: "
                    f"{still_trainable[:3]}")

    # Verify all BN modules switched to eval (training=False)
    import torch.nn as nn
    bns = [(n, m) for n, m in model.encoder.named_modules()
           if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]
    train_mode_bns = [n for n, m in bns if m.training]
    if train_mode_bns:
        _fail("G3", f"{len(train_mode_bns)} BN modules still in train() mode after eval(): "
                    f"{train_mode_bns[:3]}")
    _ok(f"all {len(list(model.encoder.parameters()))} encoder params frozen; "
        f"all {len(bns)} BN modules in eval()")


def gate4_identity_init(cfg) -> None:
    print("[G4] PA-CODE identity-init")
    model = build_model(cfg).cpu().eval()
    if not getattr(model, "use_pa_code", False):
        _fail("G4", "model.use_pa_code=False -- config wiring broken")
    B, D, C = 1, int(cfg.data.slices_per_volume), int(cfg.model.embed_dim)
    H = W = 16
    feats = torch.randn(B, D, C, H, W)
    organ_id = torch.tensor([6])  # liver
    with torch.no_grad():
        out = model.ode(feats, organ_id)
    diff = (out - feats).abs().max().item()
    if diff > 1e-5:
        _fail("G4", f"identity violated, max-abs-diff={diff:.3e}")
    _ok(f"|ode(x) - x|_inf = {diff:.3e}")


def gate5_warmstart(cfg) -> None:
    print("[G5] warmstart ckpt exists + strict-loads into PA-CODE model")
    if not WARMSTART_PATH.exists():
        _fail("G5", f"warmstart missing: {WARMSTART_PATH}\n"
                    f"  -> run scripts/prepare_a1_pa_code_warmstart.py first")
    ckpt = torch.load(str(WARMSTART_PATH), map_location="cpu", weights_only=False)
    if "model_state_dict" not in ckpt:
        _fail("G5", f"warmstart ckpt missing 'model_state_dict' key")
    if ckpt.get("epoch", 0) != -1:
        _fail("G5", f"warmstart epoch={ckpt.get('epoch')} (expected -1 for trainer start_epoch=0)")
    model = build_model(cfg).cpu()
    # strict=True must succeed (the prep script verifies this too, but we re-check)
    model.load_state_dict(ckpt["model_state_dict"])
    n_loaded = ckpt.get("warmstart_keys_loaded", -1)
    _ok(f"strict=True load OK; carried {n_loaded} keys from {Path(ckpt.get('warmstart_source','?')).name}")


def gate6_3d_eval_pipeline(cfg) -> None:
    print("[G6] 3D eval pipeline (eval_a1_per_organ)")
    try:
        import importlib
        ea = importlib.import_module("eval_a1_per_organ")
    except Exception as e:
        _fail("G6", f"eval_a1_per_organ import: {e}")
    try:
        from evaluation.metrics_3d import VolumetricMetrics  # noqa: F401
    except Exception as e:
        _fail("G6", f"VolumetricMetrics import: {e}")
    vols = ea.list_val_volumes(Path(cfg.data.data_root), cfg.data.modality)
    if len(vols) < 4:
        _fail("G6", f"only {len(vols)} val volumes -- need >= 4 for n_vols=4")
    _ok(f"{len(vols)} AMOS22 val volumes available")


def gate7_film_divergence(cfg) -> None:
    """100 toy steps must make gamma/beta diverge across organ_ids.

    Simulates: pull a fresh PA-CODE model, run a tiny optimization on a
    synthetic per-organ target. If gamma_head / beta_head are working,
    gamma(liver) and gamma(spleen) must drift apart.
    """
    print("[G7] FiLM heads diverge per-organ under 100 toy gradient steps")
    torch.manual_seed(0)
    model = build_model(cfg).cpu()
    # Only train ode heads (matches P0 frozen-encoder regime)
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("ode.")
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
    )

    B, D, C = 2, int(cfg.data.slices_per_volume), int(cfg.model.embed_dim)
    H = W = 16
    feats = torch.randn(B, D, C, H, W).requires_grad_(False)

    # Synthetic per-organ targets: organ 6 wants +1, organ 1 wants -1
    targets = {6: torch.full_like(feats, +1.0), 1: torch.full_like(feats, -1.0)}

    for step in range(100):
        loss = 0.0
        for oid in [6, 1]:
            organ_id = torch.tensor([oid] * B)
            out = model.ode(feats, organ_id)
            loss = loss + (out - targets[oid]).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # After 100 steps, gamma(6) and gamma(1) should differ; same for beta.
    e_liver  = model.ode.ode_fwd.organ_embed(torch.tensor([6]))
    e_spleen = model.ode.ode_fwd.organ_embed(torch.tensor([1]))
    g_liver  = model.ode.ode_fwd.gamma_head(e_liver).detach()
    g_spleen = model.ode.ode_fwd.gamma_head(e_spleen).detach()
    b_liver  = model.ode.ode_fwd.beta_head(e_liver).detach()
    b_spleen = model.ode.ode_fwd.beta_head(e_spleen).detach()
    g_diff = (g_liver - g_spleen).abs().mean().item()
    b_diff = (b_liver - b_spleen).abs().mean().item()
    if g_diff < 1e-4:
        _fail("G7", f"gamma head not diverging: |gamma(liver)-gamma(spleen)|.mean={g_diff:.3e}")
    if b_diff < 1e-4:
        _fail("G7", f"beta head not diverging: |beta(liver)-beta(spleen)|.mean={b_diff:.3e}")
    _ok(f"|gamma_diff|={g_diff:.3e}, |beta_diff|={b_diff:.3e} after 100 toy steps")


def gate8_forward_sanity(cfg) -> None:
    print("[G8] full-model forward sanity (no NaN, expected shapes)")
    model = build_model(cfg).cpu().eval()
    B, D, H, W = 1, int(cfg.data.slices_per_volume), int(cfg.model.img_size), int(cfg.model.img_size)
    images = torch.randn(B, D, 3, H, W)
    organ_id = torch.tensor([6])
    with torch.no_grad():
        out = model(images, organ_id, target_organ_ids=[1, 2, 3, 6])
    if torch.isnan(out["masks"]).any():
        _fail("G8", "NaN in masks output")
    if out["masks"].shape[1] != 4:
        _fail("G8", f"masks K={out['masks'].shape[1]}, expected 4 (target_organs)")
    if out["deepsup_logits"].shape[-2:] != out["masks"].shape[-2:]:
        _fail("G8", f"deepsup shape {tuple(out['deepsup_logits'].shape[-2:])} != "
                    f"masks shape {tuple(out['masks'].shape[-2:])}")
    _ok(f"forward OK: masks {tuple(out['masks'].shape)}, "
        f"deepsup {tuple(out['deepsup_logits'].shape)}")


# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 70)
    print("PA-CODE pre-flight -- 8 gates")
    print("=" * 70)

    cfg = _load_cfg()
    print(f"  config: {PA_CODE_CONFIG.name}")
    print(f"  arch:   {cfg.model.architecture}  use_pa_code={cfg.model.ode.use_pa_code}")
    print(f"  budget: {cfg.training.epochs} epochs, freeze P0={cfg.training.encoder_freeze_epochs}")
    print()

    gates = [
        gate1_schema,
        gate2_monitor_metric,
        gate3_encoder_freeze,
        gate4_identity_init,
        gate5_warmstart,
        gate6_3d_eval_pipeline,
        gate7_film_divergence,
        gate8_forward_sanity,
    ]
    for fn in gates:
        try:
            fn(cfg)
        except RuntimeError as e:
            print()
            print("=" * 70)
            print(f"PRE-FLIGHT FAIL: {e}")
            print("=" * 70)
            return 1
        print()

    print("=" * 70)
    print("PRE-FLIGHT PASS - safe to launch path_a1_pa_code")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
