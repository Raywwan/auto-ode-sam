"""Generate the §7.4 PA-CODE diagnostic figure for the thesis.

Two stacked panels:
  (top)    γσ trajectory for Runs #2 (unbounded) and #3 (tanh-bounded);
           dashed horizontal at σ_max=1.0 (the maximum elementwise spread
           that the bound γ ∈ [0.5, 1.5] can support).
  (bottom) val_dice_3d trajectory for all three PA-CODE runs, with
           gate threshold at 0.30 highlighted.

Reads the parsed metric history from the run logs directly so the figure
re-generates verbatim from the artifact tree in the appendix.

Output:
    thesis/figures/pa_code_gamma_drift.pdf
"""
from __future__ import annotations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "thesis" / "figures" / "pa_code_gamma_drift.pdf"


# Trajectories extracted from the run logs (epoch 0–6 for runs #2/#3, 0–5 for run #1).
RUN1 = {
    "label":       r"Run \#1 (LR $2{\times}10^{-4}$, no $\gamma$-bound)",
    "epoch":       [0, 1, 2, 3, 4, 5],
    "val_dice_3d": [0.1961, 0.2232, 0.0915, 0.2082, 0.1354, 0.1646],
    "gamma_sigma": [None, None, None, None, None, None],
    "color":       "#888888",
    "ls":          ":",
    "marker":      "s",
}
RUN2 = {
    "label":       r"Run \#2 (LR $1{\times}10^{-4}$, no $\gamma$-bound)",
    "epoch":       [0, 1, 2, 3, 4],
    "val_dice_3d": [0.1961, 0.2396, 0.0805, 0.2229, 0.1725],
    "gamma_sigma": [0.000, 0.054, 0.214, 0.363, 0.494],
    "color":       "#d05050",
    "ls":          "--",
    "marker":      "o",
}
RUN3 = {
    "label":       r"Run \#3 (LR $1{\times}10^{-4}$, $\gamma{=}1{+}0.5\tanh(\gamma_{\mathrm{raw}}{-}1)$)",
    "epoch":       [0, 1, 2, 3, 4, 5, 6],
    "val_dice_3d": [0.1961, 0.2154, 0.2204, 0.0952, 0.2014, 0.1769, 0.1692],
    "gamma_sigma": [0.000, 0.040, 0.221, 0.394, 0.666, 0.958, 1.238],
    "color":       "#3060a0",
    "ls":          "-",
    "marker":      "o",
}


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family":     "serif",
        "font.size":       10,
        "axes.labelsize":  10,
        "axes.titlesize":  11,
        "legend.fontsize": 8.5,
        "text.usetex":     False,
    })

    fig, (ax_g, ax_d) = plt.subplots(2, 1, figsize=(6.0, 5.4), sharex=True,
                                     gridspec_kw={"hspace": 0.12})

    # ----- top: gamma sigma -----
    for run in (RUN2, RUN3):
        xs = [e for e, v in zip(run["epoch"], run["gamma_sigma"]) if v is not None]
        ys = [v for v in run["gamma_sigma"] if v is not None]
        ax_g.plot(xs, ys, color=run["color"], ls=run["ls"],
                  marker=run["marker"], lw=1.6, ms=5, label=run["label"])
    ax_g.axhline(1.0, color="black", ls=":", lw=0.8, alpha=0.6)
    ax_g.text(0.05, 1.04, "max elementwise spread allowed by bound",
              transform=ax_g.get_yaxis_transform(), fontsize=8.2,
              color="black", alpha=0.7)
    ax_g.set_ylabel(r"per-organ spread $\sigma_\gamma$")
    ax_g.set_ylim(-0.05, 1.4)
    ax_g.legend(loc="upper left", framealpha=0.92)
    ax_g.grid(alpha=0.25)
    ax_g.set_title(r"PA-CODE $\gamma$-drift: bound holds elementwise but per-organ spread is unbounded",
                   pad=4)

    # ----- bottom: val_dice_3d -----
    for run in (RUN1, RUN2, RUN3):
        ax_d.plot(run["epoch"], run["val_dice_3d"], color=run["color"],
                  ls=run["ls"], marker=run["marker"], lw=1.6, ms=5,
                  label=run["label"])
    ax_d.axhline(0.30, color="black", ls=":", lw=0.8, alpha=0.6)
    ax_d.text(0.05, 0.31, "ep4 gate threshold (0.30)",
              transform=ax_d.get_yaxis_transform(), fontsize=8.2,
              color="black", alpha=0.7)
    ax_d.axhline(0.1961, color="black", ls=(0, (1, 3)), lw=0.7, alpha=0.4)
    ax_d.text(0.05, 0.205, "ep0 baseline (0.196)",
              transform=ax_d.get_yaxis_transform(), fontsize=8.0,
              color="black", alpha=0.6)
    ax_d.set_xlabel("epoch")
    ax_d.set_ylabel("val Dice 3D (4-volume probe)")
    ax_d.set_ylim(0.0, 0.42)
    ax_d.set_xticks(range(0, 7))
    ax_d.legend(loc="upper right", framealpha=0.92)
    ax_d.grid(alpha=0.25)

    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
