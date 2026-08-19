"""Deliverables 3, 4, 5: per-row adaptive top-k under a calibrated hit model.

Decision rule (derived in topk_numeric.py's docstring, and note the brief's
version of the inequality was inverted):

    emit k candidates, k* = argmax_k  C_k * 2/(k+1),
    C_k = sum_{j<=k} p_j  (candidates are tolerance-separated, so the per
                           candidate hit events are mutually exclusive)

p_j is a calibrated hit probability for the j-th candidate, obtained by
isotonic regression of the candidate's vote share onto the 0/1 hit label,
fitted on train (demo-excluded) + val pooled.

Honesty controls built in:
  * CROSS-FITTED evaluation. The calibrator that scores a row is never fitted
    on that row (GroupKFold by row). The in-sample number is printed too, so
    the optimism is visible rather than hidden.
  * ORACLE bound. The best any per-row policy could do on this candidate list,
    which says whether the calibration or the list is the binding constraint.
  * paired bootstrap of per-row F1 on train+val POOLED, using the campaign's
    own scorer.paired_bootstrap.

Run:
  cd /Users/maksimsilchenko/AKBC/pipeline && source ~/mac-ml-setup/.venv/bin/activate \
    && python3 topk_report2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import TEST_ROWS, TOTAL_TEST_ROWS
from scorer import paired_bootstrap, score_one_relation
from topk_numeric import KMAX, build_table, default_args, emit

PAIRS = [("hasCapacity", "cap_recite"), ("hasArea", "area_recite")]
# Board-measured TEST macro-F1 of the shipped k=1 configuration, from the
# confirmed 0.7060 submission. Used only to express the projection as a delta
# on a known quantity; nothing here reads test gold.
SHIPPED_TEST_F1 = {"hasCapacity": 0.3367, "hasArea": 0.8700}
OUT = Path(__file__).resolve().parent.parent / "analysis"
NFOLD = 5


def flatten(tables: list[list[dict]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(share, hit, row_group) over every candidate of every row."""
    x, y, g = [], [], []
    gid = 0
    for t in tables:
        for r in t:
            for j, s in enumerate(r["shares"]):
                x.append(s)
                y.append(1.0 if r["hit_at"] == j + 1 else 0.0)
                g.append(gid)
            gid += 1
    return np.array(x), np.array(y), np.array(g)


def fit_iso(x, y):
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True,
                             out_of_bounds="clip")
    iso.fit(x, y)
    return iso


def crossfit_probs(x, y, g, nfold=NFOLD, seed=20260812):
    """Out-of-fold calibrated probabilities, folds split by ROW."""
    rng = np.random.RandomState(seed)
    rows = np.unique(g)
    fold_of_row = {r: i for r, i in zip(rows, rng.permutation(len(rows)) % nfold)}
    fold = np.array([fold_of_row[v] for v in g])
    p = np.zeros_like(x, dtype=float)
    for f in range(nfold):
        tr, te = fold != f, fold == f
        p[te] = fit_iso(x[tr], y[tr]).predict(x[te])
    return p


def reliability(p, y, nbins=10):
    """Equal-count bins on the predicted probability."""
    order = np.argsort(p, kind="stable")
    out = []
    for chunk in np.array_split(order, nbins):
        if len(chunk) == 0:
            continue
        out.append({
            "n": int(len(chunk)),
            "pred_lo": round(float(p[chunk].min()), 4),
            "pred_hi": round(float(p[chunk].max()), 4),
            "mean_pred": round(float(p[chunk].mean()), 4),
            "empirical": round(float(y[chunk].mean()), 4),
        })
    ece = sum(b["n"] * abs(b["mean_pred"] - b["empirical"]) for b in out) / len(p)
    brier = float(np.mean((p - y) ** 2))
    return out, round(ece, 4), round(brier, 4)


def choose_k(probs: list[float]) -> int:
    """argmax_k  (sum_{j<=k} p_j) * 2/(k+1),  capped at the number available."""
    best_k, best_v, c = 1, -1.0, 0.0
    for k, p in enumerate(probs, start=1):
        c = min(1.0, c + p)
        v = c * 2 / (k + 1)
        if v > best_v + 1e-12:
            best_k, best_v = k, v
    return best_k


def per_row_f1(table, relation, split, ks):
    res = score_policy_rows(table, relation, split, ks)
    return res


def score_policy_rows(table, relation, split, ks):
    preds = {r["subject"]: emit(r["values"][:(ks if isinstance(ks, int) else ks[r["subject"]])])
             for r in table}
    return score_one_relation(preds, relation, split,
                              subjects={r["subject"] for r in table})


def main() -> int:
    args = default_args()
    report = {}

    for relation, channel in PAIRS:
        print("=" * 78)
        print(f"{relation} / {channel}")
        tr = build_table(channel, "train", args)
        va = build_table(channel, "val", args)
        te = build_table(channel, "test", args)
        print(f"  rows: train(demo-excluded)={len(tr)}  val={len(va)}  test={len(te)}")

        x, y, g = flatten([tr, va])
        n_rows = len(np.unique(g))

        # ---------------- calibration
        iso_full = fit_iso(x, y)
        p_in = iso_full.predict(x)
        p_oof = crossfit_probs(x, y, g)
        rel_in, ece_in, brier_in = reliability(p_in, y)
        rel_oof, ece_oof, brier_oof = reliability(p_oof, y)
        print(f"\n  calibration on train+val pooled: {len(x)} candidates from {n_rows} rows, "
              f"base hit rate {y.mean():.4f}")
        print("  reliability (cross-fitted, equal-count deciles):")
        print("     bin  n   pred_range          mean_pred  empirical")
        for i, b in enumerate(rel_oof):
            print(f"     {i+1:>3d} {b['n']:>4d}  [{b['pred_lo']:.4f},{b['pred_hi']:.4f}]"
                  f"      {b['mean_pred']:.4f}     {b['empirical']:.4f}")
        print(f"  ECE(oof)={ece_oof}  Brier(oof)={brier_oof}   "
              f"| ECE(in-sample)={ece_in} Brier(in)={brier_in}")

        # map probabilities back onto rows
        def attach(tables, probs):
            i = 0
            gid = 0
            for t in tables:
                for r in t:
                    m = len(r["shares"])
                    r["p"] = list(probs[i:i + m])
                    i += m
                    gid += 1
        attach([tr, va], p_oof)
        for r in tr + va:
            r["p_in"] = list(iso_full.predict(np.array(r["shares"]))) if r["shares"] else []

        # ---------------- policies
        pol = {}
        for name, key in (("adaptive_oof", "p"), ("adaptive_insample", "p_in")):
            pol[name] = {r["subject"]: choose_k(r[key]) for t in (tr, va) for r in t}
        pol["shipped_k1"] = {r["subject"]: 1 for t in (tr, va) for r in t}
        # oracle: the best k a per-row policy could pick knowing the answer
        pol["oracle"] = {r["subject"]: (r["hit_at"] or 1) for t in (tr, va) for r in t}

        scored = {}
        for name, ks in pol.items():
            f1s, subs = [], []
            for t, sp in ((tr, "train"), (va, "val")):
                res = score_policy_rows(t, relation, sp, ks)
                f1s += res["f1_vector"]
                subs += res["subjects"]
            scored[name] = {"f1": f1s, "macro": sum(f1s) / len(f1s)}
        # best uniform k on pooled train+val, for reference
        for k in range(1, KMAX + 1):
            f1s = []
            for t, sp in ((tr, "train"), (va, "val")):
                f1s += score_policy_rows(t, relation, sp, k)["f1_vector"]
            scored[f"uniform_k{k}"] = {"f1": f1s, "macro": sum(f1s) / len(f1s)}

        print("\n  pooled train+val macro-F1 (official scorer, per-row vectors concatenated):")
        for name in ["shipped_k1"] + [f"uniform_k{k}" for k in range(2, KMAX + 1)] + \
                    ["adaptive_insample", "adaptive_oof", "oracle"]:
            print(f"     {name:>20s}  {scored[name]['macro']:.4f}")

        base = scored["shipped_k1"]["f1"]
        boots = {}
        for name in ("adaptive_oof", "adaptive_insample", "uniform_k2"):
            boots[name] = paired_bootstrap(base, scored[name]["f1"], n_boot=10000)
            b = boots[name]
            print(f"\n  paired bootstrap {name} - shipped_k1 (train+val pooled, n={b['n_rows']}):"
                  f"\n     point {b['point']:+.4f}  90% CI [{b['ci_lo']:+.4f},{b['ci_hi']:+.4f}]"
                  f"  rows up {b['rows_up']} / down {b['rows_down']}")

        kdist = {}
        for s, k in pol["adaptive_oof"].items():
            kdist[k] = kdist.get(k, 0) + 1
        print(f"\n  adaptive_oof k distribution on train+val: {dict(sorted(kdist.items()))}")

        # ---------------- TEST projection
        for r in te:
            r["p"] = list(iso_full.predict(np.array(r["shares"]))) if r["shares"] else []
        k_test = {r["subject"]: (choose_k(r["p"]) if r["p"] else 1) for r in te}
        changed = sum(1 for v in k_test.values() if v > 1)
        kdist_t = {}
        for v in k_test.values():
            kdist_t[v] = kdist_t.get(v, 0) + 1

        def proj(ks):
            tot = 0.0
            for r in te:
                k = ks if isinstance(ks, int) else ks[r["subject"]]
                k = min(k, max(len(r["p"]), 1))
                c = min(1.0, sum(r["p"][:k])) if r["p"] else 0.0
                tot += c * 2 / (k + 1)
            return tot / len(te)

        proj_k1, proj_ad = proj(1), proj(k_test)
        delta = proj_ad - proj_k1
        known = SHIPPED_TEST_F1[relation]
        print(f"\n  TEST projection ({len(te)} rows, calibrator fitted on ALL train+val):")
        print(f"     rows where the rule emits k>1: {changed}/{len(te)}   "
              f"k distribution {dict(sorted(kdist_t.items()))}")
        print(f"     projected E[F1] k=1      : {proj_k1:.4f}   (board-MEASURED k=1 = {known:.4f})")
        print(f"     projected E[F1] adaptive : {proj_ad:.4f}")
        print(f"     projected DELTA          : {delta:+.4f}")
        print(f"     delta-anchored test F1   : {known + delta:.4f}  "
              f"(= measured {known:.4f} + projected delta)")
        print(f"     overall-score impact     : {delta * TEST_ROWS[relation] / TOTAL_TEST_ROWS:+.5f}")

        report[relation] = {
            "n_train_scored": len(tr), "n_val": len(va), "n_test": len(te),
            "n_candidates_trainval": int(len(x)), "base_hit_rate": round(float(y.mean()), 4),
            "reliability_oof": rel_oof, "ece_oof": ece_oof, "brier_oof": brier_oof,
            "reliability_insample": rel_in, "ece_insample": ece_in, "brier_insample": brier_in,
            "pooled_macro": {k: round(v["macro"], 4) for k, v in scored.items()},
            "bootstrap": {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                              for kk, vv in v.items()} for k, v in boots.items()},
            "k_distribution_trainval": {str(k): v for k, v in sorted(kdist.items())},
            "test_rows_changed": changed,
            "k_distribution_test": {str(k): v for k, v in sorted(kdist_t.items())},
            "test_projected_k1": round(proj_k1, 4),
            "test_projected_adaptive": round(proj_ad, 4),
            "test_projected_delta": round(delta, 4),
            "test_board_measured_k1": known,
            "test_delta_anchored_projection": round(known + delta, 4),
            "overall_score_impact": round(delta * TEST_ROWS[relation] / TOTAL_TEST_ROWS, 5),
        }

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "topk_report2.json", "w") as fh:
        json.dump(report, fh, indent=1)
    tot = sum(v["overall_score_impact"] for v in report.values())
    print("\n" + "=" * 78)
    print(f"TOTAL projected overall-score impact of adaptive hedging: {tot:+.5f}")
    print(f"wrote {OUT/'topk_report2.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
