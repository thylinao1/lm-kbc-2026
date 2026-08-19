"""Two closing checks on the expected-F1 proposal.

PART 1. Per-direction effective tau. ef1_ischaracter.py showed the EV rule is a
global vote-share threshold on 4 of 6 relations when the calibrator sees all the
data. Under the cross-fit each direction gets its own curve, so each direction
has its own effective tau. Printing it turns "per-row expected-F1 maximiser"
into a number on an axis the board has already bracketed.

PART 2. PERMUTATION CONTROL on the oracle-prefix ceiling, which is the
proposal's "the class is not empty, the pricing is what fails" claim (+0.108
overall for a perfect per-row cut). The campaign's own finding is that oracle
ceilings on wide pools are inflated by coincidence ("coverage is not
knowledge": 35.5 of 84.5 capacity coverage points are chance). An oracle
that picks the best k per row ON THE SPLIT IT IS SCORED ON has the same defect.
Control: give every subject ANOTHER subject's gold and recompute the oracle
gain. Whatever survives is what a rule could not possibly have known.

Nothing here writes to configs/, submissions/ or docs/.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import expected_f1 as E
from common import TEST_ROWS, TOTAL_TEST_ROWS
from scorer import _ev

TOL = 0.05


# ---------------------------------------------------------------- part 1

def effective_tau() -> None:
    print("=" * 96)
    print("PART 1. EFFECTIVE GLOBAL TAU of the EV rule, PER CROSS-FIT DIRECTION")
    print("=" * 96)
    print(f"{'relation':30s} {'eval on':8s} {'n':>4s} {'eff tau':>9s} {'match':>9s} "
          f"{'EV delta':>10s} {'tau delta':>10s}")
    for rel in E.RELCFG:
        data = {sp: E.build_rows(rel, sp) for sp in ("train", "val")}
        for use, fit in (("val", "train"), ("train", "val")):
            rng = np.random.default_rng(E.SEED)
            cal = E.fit_calibrator("iso", data[fit])
            size = E.fit_size_model(rel, data[fit])
            n_mc = 1200 if rel == "awardWonBy" else 4000
            rows = data[use]
            ks_e = {r["subject"]: E.choose_k(r, cal, size, rel, rng, n_mc)[0]
                    for r in rows}
            ks_s = {r["subject"]: E.shipped_k(r, rel) for r in rows}
            grid = sorted({s for r in rows for s in r["shares"]} | {0.0, 1.01})
            best = (0.0, -1)
            for tau in grid:
                m = sum(1 for r in rows
                        if sum(1 for s in r["shares"] if s >= tau) == ks_e[r["subject"]])
                if m > best[1]:
                    best = (tau, m)
            ks_t = {r["subject"]: sum(1 for s in r["shares"] if s >= best[0])
                    for r in rows}
            f_s = E.score_prefixes(rows, ks_s, rel, use)["macro_f1"]
            f_e = E.score_prefixes(rows, ks_e, rel, use)["macro_f1"]
            f_t = E.score_prefixes(rows, ks_t, rel, use)["macro_f1"]
            print(f"{rel[:30]:30s} {use:8s} {len(rows):4d} {best[0]:9.4f} "
                  f"{best[1]:4d}/{len(rows):<4d} {f_e-f_s:+10.4f} {f_t-f_s:+10.4f}")


# ---------------------------------------------------------------- part 2

def oracle_gain(rows_by_split: dict, rel: str) -> float:
    """Pooled oracle-prefix gain over the shipped rule, official scorer."""
    ship, orc = [], []
    for sp, rows in rows_by_split.items():
        ks_s = {r["subject"]: E.shipped_k(r, rel) for r in rows}
        a = E.score_prefixes(rows, ks_s, rel, sp)
        best_f, best_k = {}, {}
        maxK = max((len(r["surfaces"]) for r in rows), default=0)
        cap = min(maxK, E.RELCFG[rel]["kmax"] or maxK)
        for k in range(0, cap + 1):
            s = E.score_prefixes(rows, {r["subject"]: k for r in rows}, rel, sp)
            for subj, f in zip(s["subjects"], s["f1_vector"]):
                if subj not in best_f or f > best_f[subj] + 1e-12:
                    best_f[subj], best_k[subj] = f, k
        b = E.score_prefixes(rows, best_k, rel, sp)
        ship += a["f1_vector"]
        orc += b["f1_vector"]
    n = len(ship)
    return (sum(orc) - sum(ship)) / n


def f1_row(preds: list[str], gold: list, rel_type: str) -> float:
    ev = _ev()
    g = [x if isinstance(x, list) else [x] for x in gold]
    tp = ev.true_positives(preds, g, rel_type=rel_type, tolerance=TOL)
    if not preds and not g:
        return 1.0
    p = 1.0 if not preds else tp / len(preds)
    r = 1.0 if not g else tp / len(g)
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def oracle_gain_direct(rows: list[dict], rel: str, golds: list) -> float:
    """Same quantity computed row-by-row with the grader's own true_positives,
    so gold can be permuted without touching the gold FILE."""
    ev = _ev()
    rt = ev.RELATION_TYPE[rel]
    cap = E.RELCFG[rel]["kmax"] or 10 ** 9
    tot_s = tot_o = 0.0
    for r, g in zip(rows, golds):
        K = min(len(r["surfaces"]), cap)
        fs = f1_row(r["surfaces"][:E.shipped_k(r, rel)], g, rt)
        fo = max(f1_row(r["surfaces"][:k], g, rt) for k in range(0, K + 1))
        tot_s += fs
        tot_o += fo
    n = len(rows)
    return (tot_o - tot_s) / n


def permutation_control(rel: str, n_perm: int = 200) -> None:
    rows = E.build_rows(rel, "train") + E.build_rows(rel, "val")
    golds = [r["gold"] for r in rows]
    real = oracle_gain_direct(rows, rel, golds)
    rng = random.Random(20260812)
    sham = []
    for _ in range(n_perm):
        perm = golds[:]
        rng.shuffle(perm)
        sham.append(oracle_gain_direct(rows, rel, perm))
    w = TEST_ROWS[rel] / TOTAL_TEST_ROWS
    m, sd = float(np.mean(sham)), float(np.std(sham))
    print(f"  {rel:30s} n={len(rows):3d}  oracle gain {real:+.4f} "
          f"({real*w:+.5f} overall) | shuffled-gold {m:+.4f} +- {sd:.4f} "
          f"({m*w:+.5f} overall) | EXCESS {real-m:+.4f} ({(real-m)*w:+.5f} overall)")


def main() -> int:
    effective_tau()
    print()
    print("=" * 96)
    print("PART 2. PERMUTATION CONTROL on the oracle-prefix ceiling "
          "(pooled train+val, 200 shuffles)")
    print("    'shuffled-gold' = the same oracle given ANOTHER subject's gold. "
          "That part is coincidence.")
    print("=" * 96)
    for rel in ("personHasCityOfDeath", "hasCapacity", "hasArea",
                "companyTradesAtStockExchange", "countryLandBordersCountry"):
        permutation_control(rel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
