"""Deliverable 3/5 continued: is ANY conservative variant of the hedge positive,
and what does the metric say is even POSSIBLE?

Three things:
  1. BY-CONSTRUCTION bounds. E[F1](k) = C_k * 2/(k+1) <= 2/(k+1), so any row (or
     relation) whose top-1 hit probability exceeds 2/3 can NEVER be improved by
     k=2, and 1/2 for k=3, etc. Count the rows that are even eligible.
  2. k-CAP sweep of the rank-aware adaptive rule (cap 2..8), cross-fitted,
     measured with the official scorer, paired bootstrap on train+val pooled.
  3. MARGIN sweep: only hedge when the predicted gain exceeds delta.
     Reported as a SENSITIVITY SURFACE, not as a selection. Picking the argmax
     of a sweep is exactly what turned a claimed +0.0364 into a measured +0.0000
     on the background-lift lever; the same discipline applies here.

Run:
  cd /Users/maksimsilchenko/AKBC/pipeline && source ~/mac-ml-setup/.venv/bin/activate \
    && python3 topk_report4.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import TEST_ROWS, TOTAL_TEST_ROWS
from scorer import paired_bootstrap
from topk_numeric import KMAX, build_table, default_args
from topk_report2 import flatten, score_policy_rows
from topk_report3 import RankAwareIso, attach, crossfit_rank, measured, ranks_of

PAIRS = [("hasCapacity", "cap_recite"), ("hasArea", "area_recite")]
OUT = Path(__file__).resolve().parent.parent / "analysis"


def choose_k_capped(probs, kcap, margin=0.0):
    """argmax_k C_k*2/(k+1) over k <= kcap, and only leave k=1 if the predicted
    gain over k=1 exceeds `margin`."""
    if not probs:
        return 1
    base = min(1.0, probs[0])
    best_k, best_v, c = 1, base, 0.0
    for k, p in enumerate(probs[:kcap], start=1):
        c = min(1.0, c + p)
        v = c * 2 / (k + 1)
        if v > best_v + 1e-12:
            best_k, best_v = k, v
    if best_k > 1 and (best_v - base) <= margin:
        return 1
    return best_k


def main() -> int:
    args = default_args()
    report = {}
    for relation, channel in PAIRS:
        print("=" * 78)
        print(f"{relation} / {channel}")
        tr = build_table(channel, "train", args)
        va = build_table(channel, "val", args)
        te = build_table(channel, "test", args)
        splits = [(tr, "train"), (va, "val")]
        x, y, g = flatten([tr, va])
        r = ranks_of([tr, va])
        p_rank = crossfit_rank(x, r, y, g)
        attach([tr, va], p_rank, "p_rank")
        full = RankAwareIso().fit(x, r, y)
        for row in te:
            n = len(row["shares"])
            row["p_rank"] = list(full.predict(np.array(row["shares"]),
                                              np.arange(1, n + 1))) if n else []

        rep = {}

        # ---------------- 1. eligibility by construction
        allrows = [row for t in (tr, va) for row in t]
        elig = {}
        for k in (2, 3, 4):
            thr = 2.0 / (k + 1)
            elig[k] = {
                "threshold_on_p1": round(thr, 4),
                "trainval_rows_with_p1_below": sum(1 for row in allrows
                                                   if row["p_rank"] and row["p_rank"][0] < thr),
                "trainval_rows": len(allrows),
                "test_rows_with_p1_below": sum(1 for row in te
                                               if row["p_rank"] and row["p_rank"][0] < thr),
                "test_rows": len(te),
            }
        print("\n  1. eligibility: k can only beat k=1 on a row whose top-1 hit prob < 2/(k+1)")
        for k, v in elig.items():
            print(f"     k={k}: p1 < {v['threshold_on_p1']:.4f} -> "
                  f"{v['trainval_rows_with_p1_below']}/{v['trainval_rows']} train+val rows, "
                  f"{v['test_rows_with_p1_below']}/{v['test_rows']} test rows")
        rep["eligibility"] = elig

        # ---------------- 2/3. cap x margin sweep
        k1 = {row["subject"]: 1 for row in allrows}
        base_f1 = measured(splits, relation, k1)
        base = sum(base_f1) / len(base_f1)
        print(f"\n  2/3. cap x margin sweep, cross-fitted, measured on train+val pooled "
              f"(baseline shipped k=1 = {base:.4f})")
        print("       cap  margin   rows_hedged   macro-F1    delta   90% CI            test_rows_k>1")
        grid = []
        for kcap in (2, 3, 4, 6, 8):
            for margin in (0.0, 0.02, 0.05, 0.10):
                ks = {row["subject"]: choose_k_capped(row["p_rank"], kcap, margin)
                      for row in allrows}
                f1 = measured(splits, relation, ks)
                m = sum(f1) / len(f1)
                b = paired_bootstrap(base_f1, f1, n_boot=4000)
                nh = sum(1 for v in ks.values() if v > 1)
                kt = {row["subject"]: choose_k_capped(row["p_rank"], kcap, margin)
                      for row in te}
                nt = sum(1 for v in kt.values() if v > 1)
                grid.append({"kcap": kcap, "margin": margin, "rows_hedged": nh,
                             "macro_f1": round(m, 4), "delta": round(m - base, 4),
                             "ci_lo": round(b["ci_lo"], 4), "ci_hi": round(b["ci_hi"], 4),
                             "test_rows_hedged": nt,
                             "overall_impact_if_transfers":
                                 round((m - base) * TEST_ROWS[relation] / TOTAL_TEST_ROWS, 5)})
                print(f"       {kcap:>3d}  {margin:>5.2f}   {nh:>10d}   {m:.4f}  "
                      f"{m-base:+.4f}   [{b['ci_lo']:+.4f},{b['ci_hi']:+.4f}]   {nt:>10d}")
        rep["sweep"] = grid
        rep["baseline_trainval_k1"] = round(base, 4)
        best = max(grid, key=lambda d: d["delta"])
        rep["best_cell_do_not_trust"] = best
        n_pos = sum(1 for d in grid if d["delta"] > 0)
        rep["cells_positive"] = n_pos
        rep["cells_total"] = len(grid)
        print(f"       cells with a positive point estimate: {n_pos}/{len(grid)}; "
              f"best cell {best['kcap']}/{best['margin']} at {best['delta']:+.4f} "
              f"CI [{best['ci_lo']:+.4f},{best['ci_hi']:+.4f}]")
        report[relation] = rep

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "topk_report4.json", "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"\nwrote {OUT/'topk_report4.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
