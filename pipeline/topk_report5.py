"""Closing checks.

  A. GROUP-LEVEL adaptivity. Per-row k is the highest-variance form of the
     policy. The lowest-variance form that is still adaptive: bucket rows by
     their predicted top-1 hit probability, and pick ONE k per bucket from the
     empirical E[F1] curve inside that bucket (fitted out-of-fold). If even this
     fails, the failure is not "the calibrator is noisy", it is that the metric
     does not pay for hedging on these pools.

  B. The TEST cov@1 identity. With exactly one gold and k=1, row F1 is 1 on a
     hit and 0 on a miss, so the relation's macro-F1 IS cov@1. The board has
     already measured that: hasCapacity 0.3367, hasArea 0.8700 (submission 6,
     confirmed 0.7060). That converts the break-even condition into a statement
     about a MEASURED quantity rather than a projected one.

  C. Projected test cov@k from the calibrator, against the break-even line.

Run:
  cd /Users/maksimsilchenko/AKBC/pipeline && source ~/mac-ml-setup/.venv/bin/activate \
    && python3 topk_report5.py
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
from topk_report2 import NFOLD, SHIPPED_TEST_F1, flatten
from topk_report3 import RankAwareIso, attach, crossfit_rank, measured, ranks_of

PAIRS = [("hasCapacity", "cap_recite"), ("hasArea", "area_recite")]
OUT = Path(__file__).resolve().parent.parent / "analysis"
NBUCKET = 4


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
        attach([tr, va], crossfit_rank(x, r, y, g), "p_rank")
        allrows = [row for t in (tr, va) for row in t]
        rep = {}

        # ---------------- A. group-level adaptive, cross-fitted by row
        rng = np.random.RandomState(20260812)
        fold = rng.permutation(len(allrows)) % NFOLD
        p1 = np.array([row["p_rank"][0] if row["p_rank"] else 0.0 for row in allrows])
        edges = np.quantile(p1, np.linspace(0, 1, NBUCKET + 1)[1:-1])
        bucket = np.digitize(p1, edges)

        ks = {}
        chosen_log = {}
        for f in range(NFOLD):
            trn = fold != f
            for b in range(NBUCKET):
                sel = trn & (bucket == b)
                if sel.sum() == 0:
                    continue
                # empirical E[F1] of uniform k inside this bucket, on TRAIN folds
                best_k, best_v = 1, -1.0
                for k in range(1, KMAX + 1):
                    vals = []
                    for i in np.where(sel)[0]:
                        row = allrows[i]
                        kk = min(k, max(len(row["values"]), 1))
                        hit = row["hit_at"] is not None and row["hit_at"] <= kk
                        vals.append((2 / (kk + 1)) if hit else 0.0)
                    v = float(np.mean(vals))
                    if v > best_v:
                        best_k, best_v = k, v
                chosen_log.setdefault(b, []).append(best_k)
                for i in np.where((fold == f) & (bucket == b))[0]:
                    ks[allrows[i]["subject"]] = best_k
        for row in allrows:
            ks.setdefault(row["subject"], 1)

        k1 = {row["subject"]: 1 for row in allrows}
        base_f1 = measured(splits, relation, k1)
        base = sum(base_f1) / len(base_f1)
        grp_f1 = measured(splits, relation, ks)
        grp = sum(grp_f1) / len(grp_f1)
        b = paired_bootstrap(base_f1, grp_f1, n_boot=10000)
        print(f"\n  A. group-level adaptive ({NBUCKET} buckets on predicted p1, k chosen "
              f"out-of-fold from the bucket's own E[F1] curve)")
        print(f"     bucket edges on p1: {[round(float(e),4) for e in edges]}")
        print(f"     k chosen per bucket across folds: "
              f"{ {bb: sorted(set(v)) for bb, v in sorted(chosen_log.items())} }")
        print(f"     shipped k=1 {base:.4f}  ->  group-adaptive {grp:.4f}   "
              f"delta {grp-base:+.4f}  90% CI [{b['ci_lo']:+.4f},{b['ci_hi']:+.4f}]"
              f"  up {b['rows_up']} / down {b['rows_down']}")
        rep["group_adaptive"] = {
            "buckets": NBUCKET,
            "edges_p1": [round(float(e), 4) for e in edges],
            "k_per_bucket_across_folds": {str(bb): sorted(set(v)) for bb, v in chosen_log.items()},
            "baseline_k1": round(base, 4), "group_adaptive": round(grp, 4),
            "delta": round(grp - base, 4),
            "ci": [round(b["ci_lo"], 4), round(b["ci_hi"], 4)],
            "rows_up": b["rows_up"], "rows_down": b["rows_down"],
        }

        # ---------------- B/C. test-side break-even, anchored on a MEASURED cov@1
        full = RankAwareIso().fit(x, r, y)
        for row in te:
            n = len(row["shares"])
            row["p_rank"] = list(full.predict(np.array(row["shares"]),
                                              np.arange(1, n + 1))) if n else []
        cov_proj = []
        for k in range(1, KMAX + 1):
            cov_proj.append(float(np.mean([min(1.0, sum(row["p_rank"][:k]))
                                           for row in te])))
        cov1_measured = SHIPPED_TEST_F1[relation]
        scale = cov1_measured / cov_proj[0] if cov_proj[0] else 1.0
        cov_anchor = [min(1.0, c * scale) for c in cov_proj]
        print(f"\n  B/C. TEST break-even. cov@1 on test is MEASURED by the board "
              f"(k=1 macro-F1 = cov@1) = {cov1_measured:.4f}")
        print("     k              " + "".join(f"{k:>9d}" for k in range(1, KMAX + 1)))
        print("     cov@k proj     " + "".join(f"{c:>9.4f}" for c in cov_proj))
        print("     cov@k anchored " + "".join(f"{c:>9.4f}" for c in cov_anchor))
        print("     need > (uniform break-even vs k=1: cov@1*(k+1)/2)")
        print("     required       " + "".join(
            f"{min(cov1_measured*(k+1)/2, 9.9):>9.4f}" for k in range(1, KMAX + 1)))
        print("     E[F1] anchored " + "".join(
            f"{cov_anchor[k-1]*2/(k+1):>9.4f}" for k in range(1, KMAX + 1)))
        rep["test_breakeven"] = {
            "cov1_measured_by_board": cov1_measured,
            "cov_at_k_projected": [round(c, 4) for c in cov_proj],
            "cov_at_k_anchored": [round(c, 4) for c in cov_anchor],
            "required_for_uniform_k_to_beat_k1":
                [round(cov1_measured * (k + 1) / 2, 4) for k in range(1, KMAX + 1)],
            "E_F1_anchored": [round(cov_anchor[k - 1] * 2 / (k + 1), 4)
                              for k in range(1, KMAX + 1)],
            "any_uniform_k_beats_k1": any(cov_anchor[k - 1] * 2 / (k + 1) > cov1_measured
                                          for k in range(2, KMAX + 1)),
        }
        report[relation] = rep

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "topk_report5.json", "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"\nwrote {OUT/'topk_report5.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
