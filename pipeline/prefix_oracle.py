"""How much is left on the table for a PREFIX rule on the two set relations?

Every deployed rule (any tau, any per-row expected-F1 maximiser) emits a prefix
of the vote-share ranking. So the best achievable by ANY such rule is the
per-row oracle prefix: pick, with gold in hand, the k that maximises F1.

That is the honest ceiling to compare a proposed cleverness against -- not the
"keep exactly the correct ones" ceiling, which no ranking-based rule can reach.

Also reports the best GLOBAL tau on the same rows, so the gap between
"best single threshold" and "best per-row cut" is measured, not asserted.
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "/Users/maksimsilchenko/AKBC/pipeline")

from aggregate import vote_shares
from channels import CHANNELS
from common import load_pool, rows_for, spec_for_channel
from scorer import _ev

ARGS = argparse.Namespace(
    model="google/gemma-4-31B",
    revision="5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89",
    temperature=0.7, top_p=0.95, demo_seed=1234, seed_base=7000)

AWARD_DEMOS = ("FAI Gold Air Medal", "Fields medal", "Fulbright Prize",
               "Nobel Prize in Physics", "Time Person of the Year", "Ballon d'Or")


def ranked(draws: list[str], ch: str) -> list[tuple[float, str]]:
    """Candidates ordered by vote share, descending; ties broken by surface form."""
    sh = vote_shares(draws, ch)
    out = [(share, surf) for share, surf in sh.values()]
    out.sort(key=lambda x: (-x[0], x[1]))
    return out


def f1_of(preds: list[str], golds: list[list[str]]) -> float:
    ev = _ev()
    seen, flat = set(), []
    for p in preds:
        k = ev.normalize_string(p)
        if k not in seen:
            seen.add(k)
            flat.append(p)
    if not flat and not golds:
        return 1.0
    tp = ev.string_true_positives(flat, golds)
    return 2 * tp / (len(flat) + len(golds)) if (flat or golds) else 1.0


def analyse(rel: str, ch: str, ship_tau: float, taus: list[float]) -> None:
    print("=" * 88)
    print(f"{rel}  channel={ch}  shipped tau={ship_tau}")
    print("=" * 88)
    rows = []          # (split, subject, ranked, gold)
    for split in ("train", "val"):
        pool = load_pool(spec_for_channel(CHANNELS[ch], split, ARGS))
        gm = {r["SubjectEntity"]: (r.get("ObjectEntities") or [])
              for r in rows_for(split, rel)}
        for s in sorted(pool):
            if rel == "awardWonBy" and split == "train" and s in AWARD_DEMOS:
                continue                      # demo leakage guard
            rows.append((split, s, ranked(pool[s], ch), gm[s]))

    # global tau curve on the pooled rows
    print(f"\n{'tau':>6s} {'train F1':>9s} {'val F1':>8s} {'pooled F1':>10s}")
    for tau in taus:
        per = {"train": [], "val": []}
        for split, _s, rk, g in rows:
            preds = [surf for share, surf in rk if share >= tau]
            per[split].append(f1_of(preds, g))
        allf = per["train"] + per["val"]
        mark = "   <- shipped" if abs(tau - ship_tau) < 1e-9 else ""
        print(f"{tau:6.3f} {sum(per['train'])/len(per['train']):9.4f} "
              f"{sum(per['val'])/len(per['val']):8.4f} "
              f"{sum(allf)/len(allf):10.4f}{mark}")

    # oracle prefix, per row
    print(f"\nORACLE PREFIX (best k per row, gold in hand)")
    print(f"{'split':>5s} {'subject':44s} {'G':>4s} {'K':>4s} {'k@tau':>6s} "
          f"{'F1@tau':>7s} {'k*':>4s} {'F1*':>7s} {'gain':>7s}")
    ship, orc = [], []
    for split, s, rk, g in rows:
        preds = [surf for share, surf in rk if share >= ship_tau]
        f_ship = f1_of(preds, g)
        best_k, best_f = 0, f1_of([], g)
        for k in range(1, len(rk) + 1):
            f = f1_of([surf for _, surf in rk[:k]], g)
            if f > best_f + 1e-12:
                best_k, best_f = k, f
        ship.append(f_ship)
        orc.append(best_f)
        print(f"{split:>5s} {s[:44]:44s} {len(g):4d} {len(rk):4d} {len(preds):6d} "
              f"{f_ship:7.4f} {best_k:4d} {best_f:7.4f} {best_f-f_ship:+7.4f}")
    n = len(ship)
    print(f"\nPOOLED train+val, n={n}: shipped {sum(ship)/n:.4f} -> "
          f"oracle-prefix {sum(orc)/n:.4f}  (gap {sum(orc)/n - sum(ship)/n:+.4f})")
    for split in ("train", "val"):
        idx = [i for i, r in enumerate(rows) if r[0] == split]
        print(f"   {split:5s} n={len(idx):3d}: shipped {sum(ship[i] for i in idx)/len(idx):.4f}"
              f" -> oracle-prefix {sum(orc[i] for i in idx)/len(idx):.4f}")


if __name__ == "__main__":
    analyse("countryLandBordersCountry", "borders_list", 0.15,
            [0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60])
    print()
    analyse("awardWonBy", "award_list", 0.10,
            [0.001, 0.034, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50])
