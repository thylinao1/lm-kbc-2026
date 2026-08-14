"""awardWonBy: is the relation recall-capped by the POOL, or by our threshold?

Everything here reads only cached draws and the public train/val gold. No
network, no external fact source.

Three questions, all answered by measurement:
  1. What are per-row P / R / F1 under the shipped rule (award_list, tau=0.10)?
  2. How do F1, emitted-set size and tp respond as tau falls to 0 (full union)?
  3. What is the POOL CEILING: emit exactly the pool candidates that are
     correct (P = 1.0 by construction), F1 = 2m/(G+m) with m = matched
     candidates and G = gold size. No decision rule can beat this.
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "/Users/maksimsilchenko/AKBC/pipeline")

from aggregate import predict_set, vote_shares
from channels import CHANNELS
from common import load_pool, rows_for, spec_for_channel
from scorer import _ev, score_one_relation

REL = "awardWonBy"
CH = "award_list"
DEMOS = ("FAI Gold Air Medal", "Fields medal", "Fulbright Prize",
         "Nobel Prize in Physics", "Time Person of the Year", "Ballon d'Or")

ARGS = argparse.Namespace(
    model="google/gemma-4-31B",
    revision="5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89",
    temperature=0.7, top_p=0.95, demo_seed=1234, seed_base=7000)


def gold_map(split: str) -> dict[str, list[list[str]]]:
    return {r["SubjectEntity"]: (r.get("ObjectEntities") or [])
            for r in rows_for(split, REL)}


def tp_of(preds: list[str], golds: list[list[str]]) -> int:
    """Official bipartite matcher, unmodified."""
    return _ev().string_true_positives(preds, golds)


def row_prf(preds: list[str], golds: list[list[str]]) -> tuple[float, float, float, int]:
    ev = _ev()
    # dedup by normalized form exactly as the grader does
    seen, flat = set(), []
    for p in preds:
        k = ev.normalize_string(p)
        if k in seen:
            continue
        seen.add(k)
        flat.append(p)
    tp = tp_of(flat, golds)
    p = tp / len(flat) if flat else 1.0
    r = tp / len(golds) if golds else 1.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f1, len(flat)


def pool_union(draws: list[str]) -> list[str]:
    """Every distinct candidate the pool ever produced (tau -> 0)."""
    return [surf for _, surf in vote_shares(draws, CH).values()]


def main() -> None:
    for split in ("train", "val"):
        pool = load_pool(spec_for_channel(CHANNELS[CH], split, ARGS))
        gm = gold_map(split)
        subs = sorted(pool)
        scorable = [s for s in subs if s not in DEMOS] if split == "train" else subs

        print(f"\n{'='*78}\n{split.upper()}  ({len(scorable)} scorable rows"
              f"{', %d demo rows excluded' % (len(subs)-len(scorable)) if split=='train' else ''})\n{'='*78}")

        # ---- 1. shipped rule, per row
        print(f"{'subject':52s} {'G':>4s} {'np':>4s} {'tp':>4s} {'P':>6s} {'R':>6s} {'F1':>6s}")
        tot = []
        for s in scorable:
            preds = predict_set(pool[s], CH, tau=0.10)
            p, r, f1, n = row_prf(preds, gm[s])
            tot.append((p, r, f1))
            print(f"{s[:52]:52s} {len(gm[s]):4d} {n:4d} "
                  f"{tp_of(preds, gm[s]):4d} {p:6.3f} {r:6.3f} {f1:6.3f}")
        n = len(tot)
        print(f"{'MACRO (tau=0.10, shipped)':52s} {'':4s} {'':4s} {'':4s} "
              f"{sum(x[0] for x in tot)/n:6.3f} {sum(x[1] for x in tot)/n:6.3f} "
              f"{sum(x[2] for x in tot)/n:6.3f}")

        # ---- 2. tau response curve
        print(f"\n{'tau':>6s} {'macroP':>7s} {'macroR':>7s} {'macroF1':>8s} "
              f"{'mean#pred':>10s} {'mean tp':>8s}")
        for tau in (0.90, 0.70, 0.50, 0.30, 0.20, 0.15, 0.10, 0.07, 0.05,
                    0.034, 0.001):
            ps = rs = fs = npd = tps = 0.0
            for s in scorable:
                preds = predict_set(pool[s], CH, tau=tau)
                p, r, f1, k = row_prf(preds, gm[s])
                ps += p; rs += r; fs += f1; npd += k; tps += tp_of(preds, gm[s])
            print(f"{tau:6.3f} {ps/n:7.4f} {rs/n:7.4f} {fs/n:8.4f} "
                  f"{npd/n:10.1f} {tps/n:8.1f}")

        # ---- 3. pool ceiling
        print(f"\nPOOL CEILING (union of all 30 draws; oracle keeps only the correct ones)")
        print(f"{'subject':52s} {'G':>5s} {'|pool|':>7s} {'match':>6s} "
              f"{'cover':>6s} {'oracleF1':>9s}")
        orc = []
        for s in scorable:
            u = pool_union(pool[s])
            g = gm[s]
            m = tp_of(u, g)
            f1 = 2 * m / (len(g) + m) if (len(g) + m) else 1.0
            orc.append(f1)
            print(f"{s[:52]:52s} {len(g):5d} {len(u):7d} {m:6d} "
                  f"{m/max(len(g),1):6.3f} {f1:9.4f}")
        print(f"{'MACRO ORACLE (pool ceiling)':52s} {'':5s} {'':7s} {'':6s} "
              f"{'':6s} {sum(orc)/len(orc):9.4f}")

        # ---- scorer cross-check of the shipped number
        preds = {s: predict_set(pool[s], CH, tau=0.10) for s in subs}
        res = score_one_relation(preds, REL, split,
                                 subjects=set(scorable) if split == "train" else None)
        print(f"\ncross-check via official scorer: macro_f1={res['macro_f1']:.4f} "
              f"(n={res['n_rows']})")

    # ---- TEST: how many rows would change, and pool size (no gold available)
    pool = load_pool(spec_for_channel(CHANNELS[CH], "test", ARGS))
    print(f"\n{'='*78}\nTEST (gold hidden): emitted-set size by tau\n{'='*78}")
    print(f"{'tau':>6s} {'mean#pred':>10s} {'rows differing vs tau=0.10':>28s}")
    base = {s: predict_set(pool[s], CH, tau=0.10) for s in pool}
    for tau in (0.90, 0.50, 0.30, 0.20, 0.15, 0.10, 0.07, 0.05, 0.034, 0.001):
        cur = {s: predict_set(pool[s], CH, tau=tau) for s in pool}
        diff = sum(1 for s in pool if sorted(cur[s]) != sorted(base[s]))
        mp = sum(len(v) for v in cur.values()) / len(cur)
        print(f"{tau:6.3f} {mp:10.1f} {diff:28d}")
    print(f"\nTEST pool union size per subject:")
    for s in sorted(pool):
        print(f"  {s[:52]:52s} |pool|={len(pool_union(pool[s])):4d} "
              f"|tau=.10|={len(base[s]):4d}")


if __name__ == "__main__":
    main()
