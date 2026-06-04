"""AMOS22 fairness audit: subset-DSC/HD95/NSD by sex, age, manufacturer, site.

Joins the canonical per-volume metric CSV (in-domain V2 seed=42) with the
AMOS22 demographics CSV (downloaded from Zenodo 7262581) on amos_id, then
reports stratified means + non-parametric tests.

WHY this matters (workshop / thesis fairness section):
    A model can have a strong headline DSC but systematically under-perform
    on a protected subgroup (e.g. one sex, an age band, a specific scanner
    vendor). Reporting subgroup performance and a statistical test rules
    out (or surfaces) this failure mode.

INPUTS:
    thesis/results/amos22_liver_indomain/per_volume_metrics.csv
    thesis/results/fairness/labeled_data_meta_0000_0599.csv      (Zenodo)

OUTPUTS:
    thesis/results/fairness/fairness_audit.json       summary + per-strata
    thesis/results/fairness/fairness_audit.tex        booktabs row snippet

DISCIPLINE NOTES:
    - All three metrics reported per stratum (not DSC-only). The metric-axis
      blindness lesson from close(2) applies here too.
    - Mann-Whitney U for binary strata (sex, site). Kruskal-Wallis for
      manufacturer (4 groups). Spearman correlation for age (continuous).
    - Multiple-comparison correction (Bonferroni) applied to the *family*
      of subgroup tests (typically 3-4 tests).
    - "No evidence of bias" is reported only when (i) all subgroup means are
      within ~1 DSC point of overall, AND (ii) corrected p > 0.05.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

THESIS = Path(__file__).resolve().parent.parent / "thesis"
METRICS_CSV = THESIS / "results" / "amos22_liver_indomain" / "per_volume_metrics.csv"
DEMOG_CSV   = THESIS / "results" / "fairness" / "labeled_data_meta_0000_0599.csv"
OUT_JSON    = THESIS / "results" / "fairness" / "fairness_audit.json"
OUT_TEX     = THESIS / "results" / "fairness" / "fairness_audit.tex"


# ---------- IO ----------

def load_metrics() -> dict[int, dict]:
    """case_id 'amos_0008' -> {dsc, hd95_mm, nsd, amos_id=8}."""
    out: dict[int, dict] = {}
    with METRICS_CSV.open() as f:
        for r in csv.DictReader(f):
            aid = int(r["case_id"].split("_")[1])
            out[aid] = {
                "case_id": r["case_id"],
                "amos_id": aid,
                "dsc":     float(r["dsc"]),
                "hd95_mm": float(r["hd95_mm"]),
                "nsd":     float(r["nsd"]),
            }
    return out


def load_demographics() -> dict[int, dict]:
    """amos_id -> {sex, age, manufacturer, site, model}."""
    out: dict[int, dict] = {}
    with DEMOG_CSV.open() as f:
        for r in csv.DictReader(f):
            aid = int(r["amos_id"])
            age_str = r["Patient's Age"]
            age = None
            if age_str and age_str.endswith("Y"):
                try: age = int(age_str[:-1])
                except ValueError: pass
            out[aid] = {
                "sex":          r["Patient's Sex"].strip() or None,
                "age":          age,
                "manufacturer": r["Manufacturer"].strip() or None,
                "site":         r["Site"].strip() or None,
                "model":        r["Manufacturer's Model Name"].strip() or None,
            }
    return out


# ---------- stratification helpers ----------

def stratum_stats(values: list[float], higher_better: bool = True) -> dict:
    a = np.asarray(values)
    return {
        "n":      int(a.size),
        "mean":   float(a.mean()),
        "std":    float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "median": float(np.median(a)),
        "min":    float(a.min()),
        "max":    float(a.max()),
    }


def mannwhitney(g1: list[float], g2: list[float]) -> dict:
    """Two-sided MWU. Returns U, p, effect (rank-biserial)."""
    if len(g1) < 3 or len(g2) < 3:
        return {"U": None, "p_value": None, "note": "n<3 in a group"}
    u, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
    # rank-biserial effect size = 1 - 2U / (n1 * n2)
    rb = 1.0 - 2.0 * u / (len(g1) * len(g2))
    return {"U": float(u), "p_value": float(p), "rank_biserial": float(rb),
            "n1": len(g1), "n2": len(g2)}


def kruskal(groups: dict[str, list[float]]) -> dict:
    g_keys = [k for k, v in groups.items() if len(v) >= 3]
    if len(g_keys) < 2:
        return {"H": None, "p_value": None, "note": "<2 groups with n>=3"}
    h, p = stats.kruskal(*[groups[k] for k in g_keys])
    return {"H": float(h), "p_value": float(p), "groups": g_keys,
            "ns": {k: len(groups[k]) for k in g_keys}}


def spearman(x: list[float], y: list[float]) -> dict:
    if len(x) < 5:
        return {"rho": None, "p_value": None, "note": "n<5"}
    r, p = stats.spearmanr(x, y)
    return {"rho": float(r), "p_value": float(p), "n": len(x)}


# ---------- main analysis ----------

def stratify_binary(rows: list[dict], key: str, val_a: str, val_b: str,
                    metric_key: str) -> dict:
    g_a = [r[metric_key] for r in rows if r.get(key) == val_a
           and r.get(metric_key) is not None]
    g_b = [r[metric_key] for r in rows if r.get(key) == val_b
           and r.get(metric_key) is not None]
    return {
        f"{val_a}": stratum_stats(g_a),
        f"{val_b}": stratum_stats(g_b),
        "test":     mannwhitney(g_a, g_b),
    }


def stratify_kway(rows: list[dict], key: str, metric_key: str) -> dict:
    groups: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r.get(key) is not None and r.get(metric_key) is not None:
            groups[r[key]].append(r[metric_key])
    summary = {k: stratum_stats(v) for k, v in groups.items()}
    summary["_test"] = kruskal(groups)
    return summary


def age_correlation(rows: list[dict], metric_key: str) -> dict:
    x = [r["age"] for r in rows if r.get("age") is not None
         and r.get(metric_key) is not None]
    y = [r[metric_key] for r in rows if r.get("age") is not None
         and r.get(metric_key) is not None]
    return spearman(x, y)


def main() -> None:
    if not DEMOG_CSV.exists():
        print(f"[fairness] MISSING demographics: {DEMOG_CSV}")
        sys.exit(1)
    if not METRICS_CSV.exists():
        print(f"[fairness] MISSING canonical metrics: {METRICS_CSV}")
        sys.exit(1)

    metrics = load_metrics()
    demog = load_demographics()

    # Join on amos_id
    rows: list[dict] = []
    missing = []
    for aid, m in metrics.items():
        d = demog.get(aid)
        if d is None:
            missing.append(aid)
            continue
        rows.append({**m, **d})

    if missing:
        print(f"[fairness] WARN: {len(missing)} cases missing demog: {missing[:10]}")

    print(f"[fairness] joined {len(rows)} cases "
          f"(metrics={len(metrics)}, demog={len(demog)})")

    metric_keys = ("dsc", "hd95_mm", "nsd")
    overall = {k: stratum_stats([r[k] for r in rows]) for k in metric_keys}

    # ----- per-axis stratifications -----
    audit: dict = {
        "n_total":       len(rows),
        "overall":       overall,
        "sex":           {k: stratify_binary(rows, "sex", "M", "F", k)
                          for k in metric_keys},
        "site":          {k: stratify_binary(rows, "site", "people", "center", k)
                          for k in metric_keys},
        "manufacturer":  {k: stratify_kway(rows, "manufacturer", k)
                          for k in metric_keys},
        "age_spearman":  {k: age_correlation(rows, k) for k in metric_keys},
    }

    # ----- Bonferroni correction: family of 3 axis tests per metric (sex,
    #       site, manufacturer) + 1 age correlation = 4 tests per metric.
    BONF = 4
    for axis, key in (("sex","sex"), ("site","site"),
                      ("manufacturer","manufacturer")):
        for mk in metric_keys:
            test_node = audit[axis][mk].get("test")
            if test_node is None and axis == "manufacturer":
                test_node = audit[axis][mk].get("_test")
            if test_node and test_node.get("p_value") is not None:
                pc = min(1.0, test_node["p_value"] * BONF)
                test_node["p_bonferroni"] = pc
                test_node["significant"]  = bool(pc < 0.05)
    for mk in metric_keys:
        node = audit["age_spearman"][mk]
        if node.get("p_value") is not None:
            pc = min(1.0, node["p_value"] * BONF)
            node["p_bonferroni"] = pc
            node["significant"]  = bool(pc < 0.05)

    # ----- write JSON -----
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as f:
        json.dump(audit, f, indent=2)

    # ----- write a small LaTeX snippet (sex + site only — main table) -----
    overall_dsc = overall["dsc"]["mean"]
    overall_hd95 = overall["hd95_mm"]["mean"]
    overall_nsd  = overall["nsd"]["mean"]
    sex_M = audit["sex"]["dsc"]["M"]; sex_F = audit["sex"]["dsc"]["F"]
    site_p = audit["site"]["dsc"]["people"]; site_c = audit["site"]["dsc"]["center"]

    tex = (
        "% Auto-generated by scripts/fairness_audit.py — do not edit\n"
        "\\begin{tabular}{lrrrrr}\n"
        "\\toprule\n"
        "Stratum & $n$ & DSC & HD95 (mm) & NSD@1mm & MWU $p_{\\mathrm{Bonf}}$ \\\\\n"
        "\\midrule\n"
        f"Overall            & {overall['dsc']['n']} & "
        f"{overall_dsc:.4f} & {overall_hd95:.3f} & {overall_nsd:.4f} & --- \\\\\n"
        f"Sex = M            & {sex_M['n']} & "
        f"{sex_M['mean']:.4f} & {audit['sex']['hd95_mm']['M']['mean']:.3f} & "
        f"{audit['sex']['nsd']['M']['mean']:.4f} & "
        f"{audit['sex']['dsc']['test'].get('p_bonferroni', float('nan')):.3f} \\\\\n"
        f"Sex = F            & {sex_F['n']} & "
        f"{sex_F['mean']:.4f} & {audit['sex']['hd95_mm']['F']['mean']:.3f} & "
        f"{audit['sex']['nsd']['F']['mean']:.4f} & \\\\\n"
        f"Site = people      & {site_p['n']} & "
        f"{site_p['mean']:.4f} & {audit['site']['hd95_mm']['people']['mean']:.3f} & "
        f"{audit['site']['nsd']['people']['mean']:.4f} & "
        f"{audit['site']['dsc']['test'].get('p_bonferroni', float('nan')):.3f} \\\\\n"
        f"Site = center      & {site_c['n']} & "
        f"{site_c['mean']:.4f} & {audit['site']['hd95_mm']['center']['mean']:.3f} & "
        f"{audit['site']['nsd']['center']['mean']:.4f} & \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    OUT_TEX.write_text(tex, encoding="utf-8")

    # ----- console summary -----
    print("---")
    print(f"  Overall:    DSC {overall_dsc:.4f}  HD95 {overall_hd95:.3f} mm  "
          f"NSD {overall_nsd:.4f}  (n={len(rows)})")
    for axis in ("sex", "site", "manufacturer", "age_spearman"):
        print(f"\n[{axis}]")
        for mk in metric_keys:
            node = audit[axis][mk]
            if axis == "age_spearman":
                rho = node.get("rho"); p = node.get("p_value")
                pc = node.get("p_bonferroni")
                print(f"  {mk:8s} rho={rho}  p={p}  p_bonf={pc}")
                continue
            # stratum means
            keys = [k for k in node if not k.startswith("_") and k != "test"]
            means = "  ".join(
                f"{k}={node[k]['mean']:.4f}(n={node[k]['n']})" for k in keys)
            test = node.get("test") or node.get("_test", {})
            pc = test.get("p_bonferroni")
            print(f"  {mk:8s} {means}  p_bonf={pc}")

    print(f"\n[fairness] wrote {OUT_JSON}")
    print(f"[fairness] wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
