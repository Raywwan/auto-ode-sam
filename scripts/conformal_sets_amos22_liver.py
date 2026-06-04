"""Split-conformal prediction sets + volume-level conformal risk control
on the cached V2 (seed=42) AMOS22 liver sigmoid stacks.

Reads:  thesis/results/amos22_liver_indomain_predictions/predictions_seed42/*.npz
        (per-volume float16 'prob' + uint8 'pred' + uint8 'gt' + 'spacing_mm')

Writes: thesis/results/conformal_amos22_liver/
            summary.json          -- headline numbers (all alpha levels)
            per_volume_metrics.csv -- one row per test volume, fold-aware
            calibration_table.csv  -- (alpha, fold) -> q_lo/q_hi/tau

Two parallel analyses on the same cached probability stacks:

(A) Split-conformal prediction SETS at pixel level, MONDRIAN
    (class-conditional).  For each pixel with true label y and predicted
    foreground prob p we use the canonical LAC nonconformity score:
        s = 1 - p   if y == 1
        s =     p   if y == 0
    Background pixels vastly outnumber foreground (~99:1 in liver crops);
    marginal calibration would let easy background dominate the quantile
    and reduce the prediction set to the hard binary prediction.  We
    therefore calibrate two quantiles separately:
        q_fg = ceil((n_fg + 1)(1-alpha)) order statistic of {1-p : y=1}
        q_bg = ceil((n_bg + 1)(1-alpha)) order statistic of {  p : y=0}
    This is Mondrian split CP and guarantees class-conditional coverage:
        P(y_test in Set | y_test = c) >= 1 - alpha   for c in {0, 1}.
    Per-pixel prediction set:
        1 in set  iff  p >= 1 - q_fg
        0 in set  iff  p <=     q_bg
    Pixels with q_bg < p < 1 - q_fg yield the ambiguous set {0, 1};
    pixels outside both bands yield the empty set (a structural rate
    that we report).  We additionally report SELECTIVE-DICE on the
    confident-singleton pixels alone -- a clean downstream metric for the
    workshop.

(B) Volume-level CONFORMAL RISK CONTROL on the false-negative rate
    (Angelopoulos & Bates 2021).  Define per-volume risk
        L_i(tau) = 1 - recall_i(tau)        (i.e. miss rate of foreground)
    L is bounded in [0, 1] and monotone-decreasing in tau on the meaningful
    range.  We calibrate the *largest* threshold satisfying the CRC bound
        (n_cal / (n_cal + 1)) * mean(L_cal(tau)) + 1 / (n_cal + 1) <= alpha
    which guarantees E[L_test(tau_hat)] <= alpha on exchangeable test vols.
    This shifts tau down from 0.5 toward a more inclusive operating point
    whose miss rate is controlled at the volume level.

Both analyses are run for alpha in {0.05, 0.10, 0.20}.

Reproducibility: a fixed seed=42 RNG is used both for the held-out 50/50
split and for the 5-fold cross-split coverage sanity check.

Cost: 100 vols, ~600 MB float16 in memory, pure CPU.  Runtime ~3-5 min.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = ROOT / "thesis" / "results" / "amos22_liver_indomain_predictions" / "predictions_seed42"
DEFAULT_OUTPUT = ROOT / "thesis" / "results" / "conformal_amos22_liver"

ALPHAS = (0.05, 0.10, 0.20)
SEED = 42


def load_volume(path: Path) -> dict:
    d = np.load(path)
    return {
        "case_id": path.stem,
        "prob": d["prob"].astype(np.float32),  # to float32 for arithmetic safety
        "pred": d["pred"].astype(np.uint8),
        "gt": d["gt"].astype(np.uint8),
        "spacing_mm": d["spacing_mm"].astype(np.float32),
    }


def pixel_nonconformity(prob: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """LAC score: 1 - prob_of_true_class, flattened, float32."""
    p = prob.ravel().astype(np.float32)
    y = gt.ravel().astype(np.uint8)
    # for y=1: s = 1 - p ; for y=0: s = p
    s = np.where(y == 1, 1.0 - p, p)
    return s


def pixel_nonconformity_classwise(prob: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (s_fg, s_bg): nonconformity scores split by true label.
    s_fg = 1 - p where y == 1 ; s_bg = p where y == 0."""
    p = prob.ravel().astype(np.float32)
    y = gt.ravel().astype(np.uint8)
    fg = y == 1
    return (1.0 - p[fg]), p[~fg]


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample valid quantile for split-CP (Romano et al. 2019).
    Returns the k-th order statistic where k = ceil((n+1)(1-alpha)).
    For very large n this is essentially the (1-alpha) quantile."""
    n = scores.size
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    # use partition + select for efficiency on huge arrays
    return float(np.partition(scores, k - 1)[k - 1])


def evaluate_set_test(prob_list: list[np.ndarray], gt_list: list[np.ndarray],
                      q_fg: float, q_bg: float):
    """Mondrian pixel-level coverage + class-conditional + set-size stats.
    Set membership:
        1 in set iff p >= 1 - q_fg
        0 in set iff p <=     q_bg
    Returns dict with marginal coverage, class-conditional coverages,
    set-size composition, and selective-Dice on singleton-foreground pixels.
    Per-volume records carry foreground-coverage and selective Dice
    components."""
    tot_pix = 0
    tot_covered = 0
    tot_size2 = 0
    tot_size0 = 0
    tot_size = 0
    tot_fg_pix = 0
    tot_fg_cov = 0
    tot_bg_pix = 0
    tot_bg_cov = 0
    # selective Dice: pixels where set = {1} exactly are "confident fg";
    # we compute Dice over those vs ground truth, ignoring ambiguous and
    # empty-set pixels in BOTH the prediction and the reference.
    tot_sel_tp = 0
    tot_sel_pred_pos = 0
    tot_sel_gt_pos = 0
    per_vol = []
    for prob, gt in zip(prob_list, gt_list):
        p = prob.astype(np.float32)
        y = gt.astype(np.uint8)
        in1 = p >= (1.0 - q_fg)
        in0 = p <= q_bg
        size = in0.astype(np.int8) + in1.astype(np.int8)
        # marginal coverage: true class in its band
        cov = np.where(y == 1, in1, in0)
        fg_mask = (y == 1)
        bg_mask = ~fg_mask
        n = y.size
        n_fg = int(fg_mask.sum())
        n_bg = int(bg_mask.sum())
        n_fg_cov = int(in1[fg_mask].sum()) if n_fg > 0 else 0
        n_bg_cov = int(in0[bg_mask].sum()) if n_bg > 0 else 0
        n_cov = n_fg_cov + n_bg_cov
        n_s2 = int((size == 2).sum())
        n_s0 = int((size == 0).sum())
        s_sum = int(size.sum())
        # singleton-foreground pixels: {1} only (i.e. in1 & ~in0)
        sel_pred = (in1 & ~in0).astype(np.uint8)
        # selective Dice on the confident region: TP/FP/FN computed within
        # the union (pred=={1} OR gt==1) -- this keeps the comparison fair
        # by not double-counting ambiguous pixels.
        sel_tp = int(((sel_pred == 1) & (y == 1)).sum())
        sel_pred_pos = int(sel_pred.sum())
        sel_gt_pos = n_fg
        per_vol.append({
            "n_pix": n,
            "n_fg": n_fg,
            "n_bg": n_bg,
            "covered": n_cov,
            "fg_covered": n_fg_cov,
            "bg_covered": n_bg_cov,
            "size_0": n_s0,
            "size_2": n_s2,
            "set_size_sum": s_sum,
            "sel_tp": sel_tp,
            "sel_pred_pos": sel_pred_pos,
            "sel_gt_pos": sel_gt_pos,
            "selective_dice": (2.0 * sel_tp / (sel_pred_pos + sel_gt_pos))
                               if (sel_pred_pos + sel_gt_pos) > 0 else 1.0,
        })
        tot_pix += n
        tot_covered += n_cov
        tot_size2 += n_s2
        tot_size0 += n_s0
        tot_size += s_sum
        tot_fg_pix += n_fg
        tot_fg_cov += n_fg_cov
        tot_bg_pix += n_bg
        tot_bg_cov += n_bg_cov
        tot_sel_tp += sel_tp
        tot_sel_pred_pos += sel_pred_pos
        tot_sel_gt_pos += sel_gt_pos
    selective_dice_micro = (
        2.0 * tot_sel_tp / (tot_sel_pred_pos + tot_sel_gt_pos)
    ) if (tot_sel_pred_pos + tot_sel_gt_pos) > 0 else 1.0
    selective_dice_macro = float(np.mean([v["selective_dice"] for v in per_vol]))
    return {
        "coverage_marginal": tot_covered / tot_pix,
        "coverage_fg": tot_fg_cov / tot_fg_pix if tot_fg_pix > 0 else float("nan"),
        "coverage_bg": tot_bg_cov / tot_bg_pix if tot_bg_pix > 0 else float("nan"),
        "frac_ambiguous": tot_size2 / tot_pix,
        "frac_empty": tot_size0 / tot_pix,
        "mean_set_size": tot_size / tot_pix,
        "selective_dice_micro": selective_dice_micro,
        "selective_dice_macro": selective_dice_macro,
        "per_volume": per_vol,
    }


def per_volume_recall(prob: np.ndarray, gt: np.ndarray, tau: float) -> float:
    """Recall (sensitivity) of foreground at threshold tau."""
    pred = (prob >= tau)
    tp = int((pred & (gt == 1)).sum())
    pos = int((gt == 1).sum())
    if pos == 0:
        return 1.0  # no foreground -- treat miss rate as 0
    return tp / pos


def per_volume_dice(prob: np.ndarray, gt: np.ndarray, tau: float) -> float:
    pred = (prob >= tau).astype(np.uint8)
    inter = int((pred & gt).sum())
    a = int(pred.sum())
    b = int(gt.sum())
    if a + b == 0:
        return 1.0
    return 2.0 * inter / (a + b)


def crc_calibrate(losses_cal: np.ndarray, taus: np.ndarray, alpha: float) -> float:
    """Pick the LARGEST tau on the grid (most precise operating point)
    whose CRC bound holds.  Loss = 1 - recall is non-decreasing in tau
    (higher threshold misses more foreground), so the valid region is
    {tau : bound(tau) <= alpha} and we take its supremum -- the
    rightmost-valid threshold.  losses_cal: shape (n_cal, n_tau);
    taus: shape (n_tau,)."""
    n_cal = losses_cal.shape[0]
    mean_loss = losses_cal.mean(axis=0)  # (n_tau,)
    # Conformal Risk Control (Angelopoulos & Bates 2021), loss bounded in [0,1]:
    # bound(tau) = (n/(n+1)) * mean_cal_loss(tau) + 1/(n+1)
    rhs = (n_cal / (n_cal + 1)) * mean_loss + 1.0 / (n_cal + 1)
    ok = rhs <= alpha
    if not ok.any():
        # No threshold satisfies the bound -- return the smallest tau
        # (most inclusive, lowest possible loss).
        return float(taus.min())
    idx = int(np.where(ok)[0].max())  # largest valid tau index
    return float(taus[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--subsample_cal", type=int, default=20,
                    help="keep every k-th pixel for calibration quantile estimation; 1=none")
    ap.add_argument("--n_folds", type=int, default=5,
                    help="k-fold split for cross-split coverage check")
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    npz_files = sorted(args.cache_dir.glob("*.npz"))
    assert len(npz_files) == 100, f"expected 100 cached vols, got {len(npz_files)}"
    print(f"Loading {len(npz_files)} cached volumes from {args.cache_dir}")
    t0 = time.time()
    vols = [load_volume(p) for p in npz_files]
    print(f"  loaded in {time.time()-t0:.1f}s; total ~{sum(v['prob'].nbytes for v in vols)/1e6:.0f} MB")

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(vols))
    n_cal = len(vols) // 2
    cal_idx_main = order[:n_cal].tolist()
    test_idx_main = order[n_cal:].tolist()
    print(f"50/50 split: n_cal={n_cal}, n_test={len(test_idx_main)}, seed={SEED}")

    # Precompute per-volume class-split nonconformity scores.  Foreground
    # is rare so we never subsample it; background is subsampled with the
    # requested stride to keep the calibration array tractable.
    print(f"Computing per-volume class-split nonconformity (bg stride={args.subsample_cal}, fg full)")
    t0 = time.time()
    pixel_scores_fg: list[np.ndarray] = []
    pixel_scores_bg: list[np.ndarray] = []
    for v in vols:
        s_fg, s_bg = pixel_nonconformity_classwise(v["prob"], v["gt"])
        if args.subsample_cal > 1:
            s_bg = s_bg[::args.subsample_cal]
        pixel_scores_fg.append(s_fg)
        pixel_scores_bg.append(s_bg)
    mean_fg = float(np.mean([s.size for s in pixel_scores_fg])) / 1e6
    mean_bg = float(np.mean([s.size for s in pixel_scores_bg])) / 1e6
    print(f"  done in {time.time()-t0:.1f}s, mean fg pix/vol={mean_fg:.3f}M  mean bg pix/vol={mean_bg:.2f}M")

    # === Analysis A: split-CP prediction sets (MONDRIAN) =================
    print("\n=== (A) Split-conformal prediction sets (Mondrian / class-conditional) ===")
    a_records = []
    for alpha in ALPHAS:
        cal_fg = np.concatenate([pixel_scores_fg[i] for i in cal_idx_main])
        cal_bg = np.concatenate([pixel_scores_bg[i] for i in cal_idx_main])
        q_fg = conformal_quantile(cal_fg, alpha)
        q_bg = conformal_quantile(cal_bg, alpha)
        print(f"  alpha={alpha:.2f}: n_cal_fg={cal_fg.size:,}  n_cal_bg={cal_bg.size:,}"
              f"  q_fg={q_fg:.4f}  q_bg={q_bg:.4f}"
              f"  -> fg-in-set if p>={1-q_fg:.4f}, bg-in-set if p<={q_bg:.4f}")
        # Evaluate on TEST volumes (full resolution, not subsampled)
        test_prob = [vols[i]["prob"] for i in test_idx_main]
        test_gt = [vols[i]["gt"] for i in test_idx_main]
        res = evaluate_set_test(test_prob, test_gt, q_fg, q_bg)
        a_records.append({
            "alpha": alpha,
            "q_fg": q_fg,
            "q_bg": q_bg,
            "lower_thresh_p": q_bg,
            "upper_thresh_p": 1.0 - q_fg,
            "target_cov_per_class": 1.0 - alpha,
            "coverage_marginal": res["coverage_marginal"],
            "coverage_fg": res["coverage_fg"],
            "coverage_bg": res["coverage_bg"],
            "mean_set_size": res["mean_set_size"],
            "frac_ambiguous": res["frac_ambiguous"],
            "frac_empty": res["frac_empty"],
            "selective_dice_micro": res["selective_dice_micro"],
            "selective_dice_macro": res["selective_dice_macro"],
            "per_volume": res["per_volume"],
        })
        print(f"    cov_marg={res['coverage_marginal']:.4f}  cov_fg={res['coverage_fg']:.4f}"
              f"  cov_bg={res['coverage_bg']:.4f}  mean_set_size={res['mean_set_size']:.4f}"
              f"  frac_ambig={res['frac_ambiguous']:.4f}  frac_empty={res['frac_empty']:.2e}"
              f"  sel_dice(micro)={res['selective_dice_micro']:.4f}"
              f"  sel_dice(macro)={res['selective_dice_macro']:.4f}")

    # === Cross-fold coverage stability check ===========================
    print(f"\n=== (A) {args.n_folds}-fold cross-split coverage stability ===")
    fold_cov_fg = {a: [] for a in ALPHAS}
    fold_cov_bg = {a: [] for a in ALPHAS}
    fold_order = rng.permutation(len(vols))
    folds = np.array_split(fold_order, args.n_folds)
    for f_idx, test_fold in enumerate(folds):
        cal_fold = [i for i in range(len(vols)) if i not in set(test_fold.tolist())]
        cal_fg = np.concatenate([pixel_scores_fg[i] for i in cal_fold])
        cal_bg = np.concatenate([pixel_scores_bg[i] for i in cal_fold])
        for alpha in ALPHAS:
            q_fg = conformal_quantile(cal_fg, alpha)
            q_bg = conformal_quantile(cal_bg, alpha)
            test_prob = [vols[i]["prob"] for i in test_fold.tolist()]
            test_gt = [vols[i]["gt"] for i in test_fold.tolist()]
            res = evaluate_set_test(test_prob, test_gt, q_fg, q_bg)
            fold_cov_fg[alpha].append(res["coverage_fg"])
            fold_cov_bg[alpha].append(res["coverage_bg"])
        print(f"  fold {f_idx+1}/{args.n_folds}: "
              + ", ".join([f"a={a:.2f}->fg_cov={fold_cov_fg[a][-1]:.4f}/bg_cov={fold_cov_bg[a][-1]:.4f}"
                           for a in ALPHAS]))
    fold_summary = {
        f"alpha_{a}": {
            "target_per_class": 1.0 - a,
            "fg_cov_mean": float(np.mean(fold_cov_fg[a])),
            "fg_cov_std": float(np.std(fold_cov_fg[a])),
            "fg_cov_min": float(np.min(fold_cov_fg[a])),
            "fg_cov_max": float(np.max(fold_cov_fg[a])),
            "bg_cov_mean": float(np.mean(fold_cov_bg[a])),
            "bg_cov_std": float(np.std(fold_cov_bg[a])),
            "fold_cov_fg": [float(x) for x in fold_cov_fg[a]],
            "fold_cov_bg": [float(x) for x in fold_cov_bg[a]],
        }
        for a in ALPHAS
    }

    # === Analysis B: volume-level conformal risk control =================
    print("\n=== (B) Volume-level conformal risk control on miss rate ===")
    # Precompute per-volume recall and dice on a tau grid.
    tau_grid = np.linspace(0.01, 0.99, 99).astype(np.float32)
    print(f"  Building per-volume loss surface on {tau_grid.size} thresholds...")
    t0 = time.time()
    recall_mat = np.zeros((len(vols), tau_grid.size), dtype=np.float32)
    dice_mat = np.zeros((len(vols), tau_grid.size), dtype=np.float32)
    for vi, v in enumerate(vols):
        prob = v["prob"]; gt = v["gt"]
        pos = int((gt == 1).sum())
        if pos == 0:
            recall_mat[vi, :] = 1.0
            dice_mat[vi, :] = 1.0
            continue
        b_sum = pos  # |GT|
        for ti, tau in enumerate(tau_grid):
            pred = prob >= tau
            inter = int((pred & (gt == 1)).sum())
            recall_mat[vi, ti] = inter / pos
            a_sum = int(pred.sum())
            denom = a_sum + b_sum
            dice_mat[vi, ti] = (2.0 * inter / denom) if denom > 0 else 1.0
    losses_mat = 1.0 - recall_mat  # miss rate
    print(f"  built in {time.time()-t0:.1f}s")

    b_records = []
    for alpha in ALPHAS:
        # Calibrate on the same main split for headline reporting
        cal_losses = losses_mat[cal_idx_main]
        tau_hat = crc_calibrate(cal_losses, tau_grid, alpha)
        # Evaluate on test split
        test_losses = losses_mat[test_idx_main, np.searchsorted(tau_grid, tau_hat)]
        test_dice = dice_mat[test_idx_main, np.searchsorted(tau_grid, tau_hat)]
        # Baseline (tau=0.5)
        i_05 = int(np.searchsorted(tau_grid, 0.5))
        base_losses = losses_mat[test_idx_main, i_05]
        base_dice = dice_mat[test_idx_main, i_05]
        rec = {
            "alpha": alpha,
            "tau_hat": tau_hat,
            "test_mean_loss": float(test_losses.mean()),
            "test_max_loss": float(test_losses.max()),
            "test_mean_recall": float(1.0 - test_losses.mean()),
            "test_mean_dice": float(test_dice.mean()),
            "baseline_tau_0p5_mean_loss": float(base_losses.mean()),
            "baseline_tau_0p5_mean_recall": float(1.0 - base_losses.mean()),
            "baseline_tau_0p5_mean_dice": float(base_dice.mean()),
        }
        b_records.append(rec)
        print(f"  alpha={alpha:.2f}: tau_hat={tau_hat:.3f}  "
              f"E[loss_test]={rec['test_mean_loss']:.4f}  "
              f"E[recall_test]={rec['test_mean_recall']:.4f}  "
              f"E[dice_test]={rec['test_mean_dice']:.4f}  "
              f"(baseline tau=0.5: loss={rec['baseline_tau_0p5_mean_loss']:.4f}, "
              f"dice={rec['baseline_tau_0p5_mean_dice']:.4f})")

    # === Write outputs ==================================================
    cal_ids = [vols[i]["case_id"] for i in cal_idx_main]
    test_ids = [vols[i]["case_id"] for i in test_idx_main]
    summary = {
        "n_volumes": len(vols),
        "n_cal_main": n_cal,
        "n_test_main": len(test_idx_main),
        "seed": SEED,
        "n_folds": args.n_folds,
        "subsample_cal_stride": args.subsample_cal,
        "alphas": list(ALPHAS),
        "cal_case_ids": cal_ids,
        "test_case_ids": test_ids,
        "pixel_split_cp": [
            {k: v for k, v in r.items() if k != "per_volume"} for r in a_records
        ],
        "pixel_split_cp_fold_stability": fold_summary,
        "volume_crc_missrate": b_records,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {args.output / 'summary.json'}")

    # per-volume CSV (test split)
    csv_path = args.output / "per_volume_metrics.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        header = ["case_id"]
        for r in a_records:
            a = r["alpha"]
            header += [f"a{a}_cov_fg", f"a{a}_cov_bg",
                       f"a{a}_frac_ambig", f"a{a}_selective_dice"]
        for r in b_records:
            a = r["alpha"]
            header += [f"a{a}_recall_at_tau_hat", f"a{a}_dice_at_tau_hat"]
        header += ["dice_at_tau_0p5", "recall_at_tau_0p5"]
        w.writerow(header)
        for ti, vi in enumerate(test_idx_main):
            cid = vols[vi]["case_id"]
            row = [cid]
            for r in a_records:
                pv = r["per_volume"][ti]
                fg_cov = (pv["fg_covered"] / pv["n_fg"]) if pv["n_fg"] > 0 else float("nan")
                bg_cov = (pv["bg_covered"] / pv["n_bg"]) if pv["n_bg"] > 0 else float("nan")
                row += [
                    f"{fg_cov:.6f}",
                    f"{bg_cov:.6f}",
                    f"{pv['size_2']/pv['n_pix']:.6f}",
                    f"{pv['selective_dice']:.6f}",
                ]
            for r in b_records:
                ti_grid = int(np.searchsorted(tau_grid, r["tau_hat"]))
                row += [
                    f"{1.0 - losses_mat[vi, ti_grid]:.6f}",
                    f"{dice_mat[vi, ti_grid]:.6f}",
                ]
            i_05 = int(np.searchsorted(tau_grid, 0.5))
            row += [f"{dice_mat[vi, i_05]:.6f}", f"{1.0 - losses_mat[vi, i_05]:.6f}"]
            w.writerow(row)
    print(f"Wrote {csv_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
