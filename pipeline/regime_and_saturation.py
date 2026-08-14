"""(1) borders: what a threshold actually decides, per split.
(2) award: does the model run out of names? coverage vs number of draws.

(1) A threshold rule only has work to do on candidates that are neither
unanimous nor near-absent. Counting them per split says whether train+val can
calibrate anything for the regime TEST is in.

(2) If the union of correct names saturates as draws are added, the ceiling is
the model's enumeration, not my selection rule, and no aggregation change can
reach the gold set. Reported as matched-gold coverage m/G at n draws.
"""
from __future__ import annotations

import sys
from collections import Counter

sys.path.insert(0, "/Users/maksimsilchenko/AKBC/pipeline")

from aggregate import vote_shares
from channels import CHANNELS
from common import load_pool, rows_for, spec_for_channel
from ev_prefix_rule import ARGS, AWARD_DEMOS
from scorer import _ev

BANDS = [(0.0, 0.05), (0.05, 0.15), (0.15, 0.50), (0.50, 0.90), (0.90, 1.01)]


def borders_regime() -> None:
    ch, rel = "borders_list", "countryLandBordersCountry"
    print("=" * 84)
    print("(1) borders: vote-share distribution of pool candidates, by split")
    print("=" * 84)
    print(f"{'split':>6s} {'rows':>5s} {'cands':>6s} | " +
          " | ".join(f"[{a:.2f},{b:.2f})" for a, b in BANDS))
    for split in ("train", "val", "test"):
        pool = load_pool(spec_for_channel(CHANNELS[ch], split, ARGS))
        cnt = Counter()
        tot = 0
        contested_rows = 0
        for s, draws in pool.items():
            has = False
            for share, _ in vote_shares(draws, ch).values():
                tot += 1
                for a, b in BANDS:
                    if a <= share < b:
                        cnt[(a, b)] += 1
                        break
                if 0.05 <= share < 0.90:
                    has = True
            contested_rows += has
        cells = " | ".join(f"{cnt[k]:12d}" for k in BANDS)
        print(f"{split:>6s} {len(pool):5d} {tot:6d} | {cells}")
        print(f"{'':>6s} rows containing at least one contested candidate "
              f"(share in [0.05,0.90)): {contested_rows}/{len(pool)}")
    print("\nThe threshold's entire decision surface is the contested band. On train+val")
    print("it is a handful of candidates, so nothing about the middle of the range can")
    print("be calibrated from the splits that have gold.")


def award_saturation() -> None:
    ch, rel = "award_list", "awardWonBy"
    ev = _ev()
    print("\n" + "=" * 84)
    print("(2) award: distinct correct names recovered as draws are added")
    print("=" * 84)
    for split in ("train", "val"):
        pool = load_pool(spec_for_channel(CHANNELS[ch], split, ARGS))
        gm = {r["SubjectEntity"]: (r.get("ObjectEntities") or [])
              for r in rows_for(split, rel)}
        subs = [s for s in sorted(pool)
                if not (split == "train" and s in AWARD_DEMOS)]
        print(f"\n{split.upper()}  ({len(subs)} scorable rows)")
        print(f"{'#draws':>7s} {'mean |union|':>13s} {'mean matched m':>15s} "
              f"{'mean m/G':>9s} {'macro 2m/(G+m)':>15s}")
        for nd in (1, 2, 3, 5, 10, 15, 20, 25, 30):
            u_sz = m_sz = cov = ceil = 0.0
            for s in subs:
                draws = pool[s][:nd]
                union = [surf for _, surf in vote_shares(draws, ch).values()]
                g = gm[s]
                m = ev.string_true_positives(union, g)
                u_sz += len(union)
                m_sz += m
                cov += m / max(len(g), 1)
                ceil += 2 * m / (len(g) + m) if (len(g) + m) else 1.0
            n = len(subs)
            print(f"{nd:7d} {u_sz/n:13.1f} {m_sz/n:15.1f} {cov/n:9.3f} {ceil/n:15.4f}")

        # marginal yield of the last 10 draws, per row
        print(f"\n  per-row: new correct names contributed by draws 21-30")
        for s in subs:
            g = gm[s]
            u20 = [surf for _, surf in vote_shares(pool[s][:20], ch).values()]
            u30 = [surf for _, surf in vote_shares(pool[s][:30], ch).values()]
            m20 = ev.string_true_positives(u20, g)
            m30 = ev.string_true_positives(u30, g)
            print(f"    {s[:50]:50s} G={len(g):4d}  m@20={m20:4d}  m@30={m30:4d}  "
                  f"(+{m30-m20})  still missing {len(g)-m30}")

    # TEST: enumeration volume only (no gold)
    pool = load_pool(spec_for_channel(CHANNELS[ch], "test", ARGS))
    print(f"\nTEST enumeration volume (no gold available):")
    for nd in (1, 5, 10, 20, 30):
        u = sum(len(vote_shares(d[:nd], ch)) for d in pool.values()) / len(pool)
        print(f"   {nd:2d} draws -> mean union {u:6.1f} distinct names")
    names_per_draw = [len(CHANNELS[ch].parse(d)) for dr in pool.values() for d in dr]
    print(f"   names per single draw on TEST: mean {sum(names_per_draw)/len(names_per_draw):.1f}, "
          f"max {max(names_per_draw)} (parser caps a draw at 60)")


if __name__ == "__main__":
    borders_regime()
    award_saturation()
