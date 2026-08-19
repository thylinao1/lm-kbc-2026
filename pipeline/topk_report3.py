"""Deliverable 3/4 continued: WHY the expected-F1 projection and the actual
measurement disagree, and whether a rank-aware calibrator fixes it.

report2 produced a contradiction that has to be resolved before anything is
claimed. On hasCapacity the calibrated rule PROJECTS +0.0432 on test, while the
same rule, cross-fitted, MEASURES -0.0048 on train+val where gold is public.
One of the two is wrong. This script settles it by running the projection and
the measurement on the SAME rows.

Three checks:
  A. projection vs measurement on train+val (out-of-fold probabilities).
  B. rank-conditional reliability: is a share-only calibrator biased against the
     rank-1 candidate? (rank 1 is the shipped selector's own pick, which carries
     information beyond its vote share.)
  C. a rank-aware calibrator (isotonic fitted separately for rank 1, rank 2 and
     rank >= 3), cross-fitted, re-scored, paired-bootstrapped.

Run:
  cd /Users/maksimsilchenko/AKBC/pipeline && source ~/mac-ml-setup/.venv/bin/activate \
    && python3 topk_report3.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import TEST_ROWS, TOTAL_TEST_ROWS
from scorer import paired_bootstrap, score_one_relation
from topk_numeric import KMAX, build_table, default_args, emit
from topk_report2 import (NFOLD, SHIPPED_TEST_F1, choose_k, fit_iso, flatten,
                          reliability, score_policy_rows)

PAIRS = [("hasCapacity", "cap_recite"), ("hasArea", "area_recite")]
OUT = Path(__file__).resolve().parent.parent / "analysis"


def ranks_of(tables):
    r = []
    for t in tables:
        for row in t:
            r += list(range(1, len(row["shares"]) + 1))
    return np.array(r)


def rank_bucket(j):
    return 0 if j == 1 else (1 if j == 2 else 2)


class RankAwareIso:
    """One isotonic per rank bucket {1}, {2}, {>=3}."""

    def __init__(self):
        self.models = {}
        self.fallback = None

    def fit(self, x, r, y):
        self.fallback = fit_iso(x, y)
        for b in (0, 1, 2):
            m = np.array([rank_bucket(v) == b for v in r])
            if m.sum() >= 30:
                self.models[b] = fit_iso(x[m], y[m])
        return self

    def predict(self, x, r):
        out = np.zeros(len(x))
        for i, (xi, ri) in enumerate(zip(x, r)):
            m = self.models.get(rank_bucket(ri), self.fallback)
            out[i] = m.predict([xi])[0]
        return out


def crossfit_rank(x, r, y, g, nfold=NFOLD, seed=20260812):
    rng = np.random.RandomState(seed)
    rows = np.unique(g)
    fold_of_row = {row: i for row, i in zip(rows, rng.permutation(len(rows)) % nfold)}
    fold = np.array([fold_of_row[v] for v in g])
    p = np.zeros(len(x))
    for f in range(nfold):
        tr, te = fold != f, fold == f
        p[te] = RankAwareIso().fit(x[tr], r[tr], y[tr]).predict(x[te], r[te])
    return p


def attach(tables, probs, key="p"):
    i = 0
    for t in tables:
        for row in t:
            m = len(row["shares"])
            row[key] = list(probs[i:i + m])
            i += m


def projected(tables, key, ks):
    """The policy's OWN expected F1 under its probabilities: mean of C_k*2/(k+1)."""
    tot = n = 0.0
    for t in tables:
        for row in t:
            k = ks if isinstance(ks, int) else ks[row["subject"]]
            k = min(k, max(len(row[key]), 1))
            c = min(1.0, sum(row[key][:k])) if row[key] else 0.0
            tot += c * 2 / (k + 1)
            n += 1
    return tot / n


def measured(tables_splits, relation, ks):
    f1 = []
    for t, sp in tables_splits:
        f1 += score_policy_rows(t, relation, sp, ks)["f1_vector"]
    return f1


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

        # ---- share-only calibrator, cross-fitted
        from topk_report2 import crossfit_probs
        p_share = crossfit_probs(x, y, g)
        attach([tr, va], p_share, "p_share")
        # ---- rank-aware calibrator, cross-fitted
        p_rank = crossfit_rank(x, r, y, g)
        attach([tr, va], p_rank, "p_rank")

        rep = {}

        # ================= B. rank-conditional reliability
        print("\n  B. rank-conditional reliability of the SHARE-ONLY calibrator (cross-fitted)")
        print("     rank    n   mean_share  mean_pred_p   empirical_hit   bias(pred-emp)")
        rk = []
        for j in range(1, KMAX + 1):
            m = r == j
            if m.sum() == 0:
                continue
            row = {"rank": j, "n": int(m.sum()),
                   "mean_share": round(float(x[m].mean()), 4),
                   "mean_pred": round(float(p_share[m].mean()), 4),
                   "empirical": round(float(y[m].mean()), 4)}
            row["bias"] = round(row["mean_pred"] - row["empirical"], 4)
            rk.append(row)
            print(f"     {j:>4d} {row['n']:>5d}      {row['mean_share']:.4f}"
                  f"       {row['mean_pred']:.4f}          {row['empirical']:.4f}"
                  f"        {row['bias']:+.4f}")
        rep["rank_conditional_share_only"] = rk

        rk2 = []
        print("\n     same table for the RANK-AWARE calibrator (cross-fitted)")
        print("     rank    n   mean_pred_p   empirical_hit   bias(pred-emp)")
        for j in range(1, KMAX + 1):
            m = r == j
            if m.sum() == 0:
                continue
            row = {"rank": j, "n": int(m.sum()),
                   "mean_pred": round(float(p_rank[m].mean()), 4),
                   "empirical": round(float(y[m].mean()), 4)}
            row["bias"] = round(row["mean_pred"] - row["empirical"], 4)
            rk2.append(row)
            print(f"     {j:>4d} {row['n']:>5d}       {row['mean_pred']:.4f}"
                  f"          {row['empirical']:.4f}        {row['bias']:+.4f}")
        rep["rank_conditional_rank_aware"] = rk2

        rel_r, ece_r, brier_r = reliability(p_rank, y)
        print(f"     rank-aware ECE(oof)={ece_r}  Brier(oof)={brier_r}")
        rep["rank_aware_reliability"] = rel_r
        rep["rank_aware_ece_oof"] = ece_r
        rep["rank_aware_brier_oof"] = brier_r

        # ================= A + C. projection vs measurement, both calibrators
        print("\n  A/C. projection vs MEASUREMENT on the same 165-ish train+val rows")
        print("       policy                 projected E[F1]   measured macro-F1   projection error")
        results = {}
        k1 = {row["subject"]: 1 for t in (tr, va) for row in t}
        for key, label in (("p_share", "share-only"), ("p_rank", "rank-aware")):
            kad = {row["subject"]: choose_k(row[key]) for t in (tr, va) for row in t}
            for name, ks in (("k=1", k1), ("adaptive", kad)):
                pr = projected([tr, va], key, ks)
                f1 = measured(splits, relation, ks)
                me = sum(f1) / len(f1)
                tag = f"{label} {name}"
                print(f"       {tag:<22s}   {pr:.4f}            {me:.4f}"
                      f"             {pr-me:+.4f}")
                results[tag] = {"projected": round(pr, 4), "measured": round(me, 4),
                                "error": round(pr - me, 4), "f1": f1, "k": ks}
        rep["projection_vs_measurement"] = {
            k: {kk: vv for kk, vv in v.items() if kk not in ("f1", "k")}
            for k, v in results.items()}

        base = results["share-only k=1"]["f1"]
        print("\n  paired bootstrap vs shipped k=1 (train+val pooled):")
        boots = {}
        for tag in ("share-only adaptive", "rank-aware adaptive"):
            b = paired_bootstrap(base, results[tag]["f1"], n_boot=10000)
            boots[tag] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in b.items()}
            print(f"     {tag:<22s} point {b['point']:+.4f}  90% CI "
                  f"[{b['ci_lo']:+.4f},{b['ci_hi']:+.4f}]  up {b['rows_up']} / down {b['rows_down']}")
        rep["bootstrap_vs_k1"] = boots

        # k distribution for the rank-aware rule
        kd = {}
        for v in results["rank-aware adaptive"]["k"].values():
            kd[v] = kd.get(v, 0) + 1
        print(f"     rank-aware adaptive k distribution (train+val): {dict(sorted(kd.items()))}")
        rep["k_distribution_trainval_rank_aware"] = {str(k): v for k, v in sorted(kd.items())}

        # ================= TEST, rank-aware, with the projection bias applied
        full = RankAwareIso().fit(x, r, y)
        for row in te:
            n = len(row["shares"])
            row["p_rank"] = list(full.predict(np.array(row["shares"]),
                                              np.arange(1, n + 1))) if n else []
        k_test = {row["subject"]: (choose_k(row["p_rank"]) if row["p_rank"] else 1)
                  for row in te}
        changed = sum(1 for v in k_test.values() if v > 1)
        kdt = {}
        for v in k_test.values():
            kdt[v] = kdt.get(v, 0) + 1
        pr1 = projected([te], "p_rank", 1)
        pra = projected([te], "p_rank", k_test)
        raw_delta = pra - pr1
        # bias correction measured on train+val: projection error of the adaptive
        # policy minus projection error of k=1
        bias = (results["rank-aware adaptive"]["error"] - results["rank-aware k=1"]["error"])
        corrected = raw_delta - bias
        known = SHIPPED_TEST_F1[relation]
        print(f"\n  TEST ({len(te)} rows), rank-aware calibrator fitted on all train+val:")
        print(f"     rows emitting k>1: {changed}/{len(te)}  k dist {dict(sorted(kdt.items()))}")
        print(f"     projected E[F1] k=1 {pr1:.4f} (board-MEASURED k=1 = {known:.4f})")
        print(f"     projected E[F1] adaptive {pra:.4f}")
        print(f"     RAW projected delta        {raw_delta:+.4f}")
        print(f"     projection bias measured on train+val {bias:+.4f}")
        print(f"     BIAS-CORRECTED delta       {corrected:+.4f}")
        print(f"     overall-score impact (bias-corrected) "
              f"{corrected * TEST_ROWS[relation] / TOTAL_TEST_ROWS:+.5f}")
        rep["test"] = {
            "n_rows": len(te), "rows_changed": changed,
            "k_distribution": {str(k): v for k, v in sorted(kdt.items())},
            "projected_k1": round(pr1, 4), "projected_adaptive": round(pra, 4),
            "raw_projected_delta": round(raw_delta, 4),
            "projection_bias_from_trainval": round(bias, 4),
            "bias_corrected_delta": round(corrected, 4),
            "board_measured_k1": known,
            "overall_impact_bias_corrected":
                round(corrected * TEST_ROWS[relation] / TOTAL_TEST_ROWS, 5),
            "measured_trainval_delta":
                round(results["rank-aware adaptive"]["measured"]
                      - results["rank-aware k=1"]["measured"], 4),
            "overall_impact_if_trainval_delta_transfers":
                round((results["rank-aware adaptive"]["measured"]
                       - results["rank-aware k=1"]["measured"])
                      * TEST_ROWS[relation] / TOTAL_TEST_ROWS, 5),
        }
        report[relation] = rep

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "topk_report3.json", "w") as fh:
        json.dump(report, fh, indent=1)
    print("\n" + "=" * 78)
    print("overall impact if the MEASURED train+val delta transfers: "
          f"{sum(v['test']['overall_impact_if_trainval_delta_transfers'] for v in report.values()):+.5f}")
    print("overall impact under the BIAS-CORRECTED projection: "
          f"{sum(v['test']['overall_impact_bias_corrected'] for v in report.values()):+.5f}")
    print(f"wrote {OUT/'topk_report3.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
