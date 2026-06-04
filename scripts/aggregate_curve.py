"""
aggregate_curve.py — Aggregate per-ckpt summary.json files into one curve CSV.

Reads every <base_dir>/ep<N>/summary.json produced by eval_amos22_liver_indomain.py
and emits <out_csv> with one row per epoch and the headline 3D metrics.

Usage:
    python aggregate_curve.py --base_dir thesis/results/curves/seed43 --out_csv thesis/results/curves/seed43/curve.csv
"""
from __future__ import annotations
import argparse
import csv
import json
import re
from pathlib import Path

FIELDS = [
    "epoch", "n_volumes",
    "dsc_mean", "dsc_std",
    "hd95_mm_mean", "hd95_mm_std",
    "nsd_mean", "nsd_std",
    "iou_mean", "assd_mm_mean",
    "sensitivity_mean", "precision_mean", "specificity_mean",
    "vol_sim_mean", "nsd_at_2.0mm_mean",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    base = Path(args.base_dir)
    if not base.exists():
        raise SystemExit(f"[aggregate] base_dir does not exist: {base}")

    rows = []
    for ep_dir in sorted(base.glob("ep*")):
        if not ep_dir.is_dir():
            continue
        m = re.match(r"ep(\d+)$", ep_dir.name)
        if not m:
            continue
        epoch = int(m.group(1))
        summary_path = ep_dir / "summary.json"
        if not summary_path.exists():
            print(f"[aggregate] WARN no summary.json in {ep_dir}, skipping")
            continue
        s = json.loads(summary_path.read_text())
        overall = (s.get("summary") or {}).get("overall") or {}
        row = {"epoch": epoch, "n_volumes": s.get("n_volumes", 0)}
        for k in FIELDS[2:]:
            v = overall.get(k, float("nan"))
            row[k] = v if v == v else float("nan")
        rows.append(row)

    rows.sort(key=lambda r: r["epoch"])
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[aggregate] wrote {out} with {len(rows)} epoch rows")
    if rows:
        best = max(rows, key=lambda r: r["dsc_mean"] if r["dsc_mean"] == r["dsc_mean"] else -1)
        print(f"[aggregate] peak DSC: {best['dsc_mean']:.4f} at ep{best['epoch']}")


if __name__ == "__main__":
    main()
