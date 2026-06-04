"""Probe the gbound (Run #3) saved checkpoint to verify the bounded-gamma fix.

Sister to scripts/diag_pa_code_filmhead_drift.py — but reads checkpoints from
checkpoints/path_a1_pa_code_gbound/ and reports the BOUNDED gamma per organ
(applying gamma = 1 + bound * tanh(gamma_raw - 1)), not the raw drift.

Output goes to thesis/results/pa_code_drift/run3_bounded_*.txt for use in
§7.4 of the A0 thesis (PA-CODE negative result + bounded-gamma fix).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CKPT_DIR = ROOT / "checkpoints" / "path_a1_pa_code_gbound"
WARMSTART = ROOT / "checkpoints" / "path_a1_pa_code" / "path_a1_pa_code_warmstart.pt"

GAMMA_BOUND = 0.5  # matches gbound config

P = "ode.ode_fwd."
ORGANS = {"liver": 6, "spleen": 1, "R-kidney": 2, "L-kidney": 3}

TRACK_KEYS = [
    f"{P}organ_embed.weight",
    f"{P}gamma_head.0.weight",
    f"{P}gamma_head.0.bias",
    f"{P}gamma_head.2.weight",
    f"{P}gamma_head.2.bias",
    f"{P}beta_head.0.weight",
    f"{P}beta_head.0.bias",
    f"{P}beta_head.2.weight",
    f"{P}beta_head.2.bias",
]


def load_state(p: Path) -> dict:
    ck = torch.load(str(p), map_location="cpu", weights_only=False)
    return ck.get("model_state_dict", ck.get("state_dict", ck))


def apply_gamma_head(state: dict, prefix: str, embed_vec: torch.Tensor) -> torch.Tensor:
    """Re-apply gamma_head MLP without instantiating the full model."""
    h = embed_vec
    h = torch.nn.functional.linear(
        h,
        state[f"{prefix}.0.weight"],
        state.get(f"{prefix}.0.bias"),
    )
    h = torch.nn.functional.silu(h)
    h = torch.nn.functional.linear(
        h,
        state[f"{prefix}.2.weight"],
        state.get(f"{prefix}.2.bias"),
    )
    return h


def report(label: str, state: dict) -> None:
    print(f"\n[{label}]")
    emb = state[f"{P}organ_embed.weight"]

    gamma_per_organ = {}
    beta_per_organ = {}
    for name, oid in ORGANS.items():
        e = emb[oid]
        gamma_raw = apply_gamma_head(state, f"{P}gamma_head", e)
        gamma = 1.0 + GAMMA_BOUND * torch.tanh(gamma_raw - 1.0)
        beta = apply_gamma_head(state, f"{P}beta_head", e)
        gamma_per_organ[name] = gamma
        beta_per_organ[name] = beta

    # Per-organ stats
    print(f"  gamma (BOUNDED via 1 + {GAMMA_BOUND}*tanh(raw-1)):")
    all_g = torch.stack([gamma_per_organ[n] for n in ORGANS])
    print(f"    cross-organ std (mean over channels) = {all_g.std(dim=0).mean().item():.4e}  <-- gamma_sigma")
    print(f"    cross-organ |max - min|     (mean)   = {(all_g.max(dim=0).values - all_g.min(dim=0).values).mean().item():.4e}")
    for name in ORGANS:
        g = gamma_per_organ[name]
        print(f"    {name:10s}: min={g.min().item():+.4f}  max={g.max().item():+.4f}  mean={g.mean().item():+.4f}  std={g.std().item():.4e}")

    print(f"\n  beta (UNBOUNDED — future-work bound candidate):")
    all_b = torch.stack([beta_per_organ[n] for n in ORGANS])
    print(f"    cross-organ std (mean over channels) = {all_b.std(dim=0).mean().item():.4e}  <-- beta_sigma")
    for name in ORGANS:
        b = beta_per_organ[name]
        print(f"    {name:10s}: min={b.min().item():+.4f}  max={b.max().item():+.4f}  mean={b.mean().item():+.4f}  std={b.std().item():.4e}")

    # Pairwise gamma diffs
    print(f"\n  pairwise |gamma| diffs (mean over channels):")
    names = list(ORGANS.keys())
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = (gamma_per_organ[a] - gamma_per_organ[b]).abs().mean().item()
            print(f"    |gamma({a:8s}) - gamma({b:8s})| = {d:.4e}")


def main() -> None:
    print("=" * 78)
    print("PA-CODE Run #3 (bounded-gamma) — saved-checkpoint probe")
    print(f"  gamma_bound = {GAMMA_BOUND}  ->  gamma in [{1-GAMMA_BOUND:.2f}, {1+GAMMA_BOUND:.2f}]")
    print("=" * 78)

    # 1. Warmstart baseline (identity init, before any training)
    if WARMSTART.exists():
        report("warmstart (identity init)", load_state(WARMSTART))
    else:
        print(f"\n[warmstart NOT FOUND at {WARMSTART}]")

    # 2. ep0 — first saved gbound checkpoint
    ep0 = CKPT_DIR / "path_a1_pa_code_gbound_epoch000.pt"
    if ep0.exists():
        report("Run #3 ep0 (after 1 epoch with frozen encoder + gamma_bound=0.5)", load_state(ep0))

    # 3. Tensor-level drift table (parallel to the unbounded diag)
    print("\n" + "=" * 78)
    print("Tensor delta L2 vs warmstart (gbound run, ep0 only)")
    print("=" * 78)
    if WARMSTART.exists() and ep0.exists():
        ws = load_state(WARMSTART)
        ck = load_state(ep0)
        print(f"\n  {'key':45s}  {'init_L2':>10s}  {'ep0 dL2':>10s}  {'ep0 rel':>10s}")
        print("  " + "-" * 90)
        for k in TRACK_KEYS:
            if k not in ws or k not in ck:
                continue
            a = ws[k].float()
            b = ck[k].float()
            init = a.norm().item()
            d = (b - a).norm().item()
            rel = (d / init * 100) if init > 1e-9 else float("inf")
            print(f"  {k:45s}  {init:10.4e}  {d:10.4e}  {rel:>9.2f}%")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
