"""Did gamma_head and beta_head actually move during the failed PA-CODE run?
If not -> Agent 1's gradient-trap-in-real-training claim holds and we have a
deeper architecture issue than a simple LR retune can fix.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CKPT_DIR = ROOT / "checkpoints" / "path_a1_pa_code"
CKPTS = {
    "warmstart": CKPT_DIR / "path_a1_pa_code_warmstart.pt",
    "epoch000":  CKPT_DIR / "path_a1_pa_code_epoch000.pt",
    "epoch001":  CKPT_DIR / "path_a1_pa_code_epoch001.pt",
    "epoch003":  CKPT_DIR / "path_a1_pa_code_epoch003.pt",
    "epoch005":  CKPT_DIR / "path_a1_pa_code_epoch005.pt",
}

# We track the FORWARD ODE module (ode.ode_fwd.*). Bidirectional has fwd+bwd
# but they are independent; one is enough to confirm gradient flow.
P = "ode.ode_fwd."
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
    f"{P}net.0.weight",
    f"{P}net.2.weight",
    f"{P}net.4.weight",
]


def load_state(p: Path) -> dict:
    ck = torch.load(str(p), map_location="cpu", weights_only=False)
    return ck.get("model_state_dict", ck.get("state_dict", ck))


def main() -> None:
    states = {name: load_state(p) for name, p in CKPTS.items()}

    print(f"{'key':45s}  {'init_L2':>10s}  {'ep0 dL2':>10s}  {'ep1 dL2':>10s}  {'ep3 dL2':>10s}  {'ep5 dL2':>10s}  {'ep5 rel':>8s}")
    print("-" * 130)
    for k in TRACK_KEYS:
        ws = states["warmstart"][k].float()
        init_l2 = ws.norm().item()
        deltas = []
        for ep in ("epoch000", "epoch001", "epoch003", "epoch005"):
            cur = states[ep][k].float()
            d = (cur - ws).norm().item()
            deltas.append(d)
        rel = deltas[-1] / max(init_l2, 1e-12)
        ddisplay = "  ".join(f"{d:10.4e}" for d in deltas)
        print(f"{k.replace(P, ''):45s}  {init_l2:10.4e}  {ddisplay}  {rel:8.2%}")

    # Per-organ FiLM probe: forward through gamma_head and beta_head for the
    # 4 target organs and report inter-organ spread.
    print()
    print("Per-organ FiLM probe (forward fwd-ODE gamma_head, beta_head):")
    print("-" * 130)
    organs = {"liver": 6, "spleen": 1, "R-kidney": 2, "L-kidney": 3}

    for ep_name in ("warmstart", "epoch001", "epoch003", "epoch005"):
        s = states[ep_name]
        emb_w = s[f"{P}organ_embed.weight"]      # (16, 64)
        gw0 = s[f"{P}gamma_head.0.weight"]; gb0 = s[f"{P}gamma_head.0.bias"]
        gw1 = s[f"{P}gamma_head.2.weight"]; gb1 = s[f"{P}gamma_head.2.bias"]
        bw0 = s[f"{P}beta_head.0.weight"];  bb0 = s[f"{P}beta_head.0.bias"]
        bw1 = s[f"{P}beta_head.2.weight"];  bb1 = s[f"{P}beta_head.2.bias"]

        gammas = []
        betas  = []
        for name, oid in organs.items():
            e = emb_w[oid]
            hg = F.gelu(e @ gw0.T + gb0)
            g  = hg @ gw1.T + gb1
            hb = F.gelu(e @ bw0.T + bb0)
            b  = hb @ bw1.T + bb1
            gammas.append(g)
            betas.append(b)
        all_g = torch.stack(gammas)   # (4, 256)
        all_b = torch.stack(betas)
        g_mean = all_g.mean().item()
        g_std_org = all_g.std(0).mean().item()  # mean over channels of std-across-organs
        b_mean = all_b.mean().item()
        b_std_org = all_b.std(0).mean().item()

        print(f"\n  [{ep_name}]")
        print(f"    gamma:  mean across (organ x channel) = {g_mean:+.5f}  (target init 1.0)")
        print(f"            mean( std-across-organs ) per-channel = {g_std_org:.4e}")
        for i, (name, _) in enumerate(organs.items()):
            print(f"            {name:10s}: gamma mean={gammas[i].mean().item():+.5f}  std={gammas[i].std().item():.4e}")
        print(f"    beta:   mean across (organ x channel) = {b_mean:+.5f}  (target init 0.0)")
        print(f"            mean( std-across-organs ) per-channel = {b_std_org:.4e}")
        for i, (name, _) in enumerate(organs.items()):
            print(f"            {name:10s}: beta mean={betas[i].mean().item():+.5f}  std={betas[i].std().item():.4e}")
        # Liver vs spleen vs kidney pairwise
        print(f"    pairwise |gamma| diffs (mean over 256 channels):")
        names = list(organs.keys())
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                d = (gammas[i] - gammas[j]).abs().mean().item()
                print(f"      |gamma({names[i]:8s}) - gamma({names[j]:8s})| = {d:.4e}")


if __name__ == "__main__":
    main()
