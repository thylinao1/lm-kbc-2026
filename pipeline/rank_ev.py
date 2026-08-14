"""The expected-value arithmetic any reranker of the capacity pool must beat.

A reranker cannot invent candidates. All it does is move some row's emitted
value from the frequency rank-1 entry to a lower-ranked one. Under this metric,
with exactly one gold per numeric row, that trade is worth

    +1 row   if the promoted candidate is inside tolerance and rank-1 is not
     0 rows  if both are inside tolerance, or neither is
    -1 row   if rank-1 is inside tolerance and the promoted candidate is not

so a rule that fires on a subset S of rows is EV-positive if and only if, INSIDE
S, the promoted candidate is correct more often than rank-1 is. The base rates
below therefore state the bar exactly, without reference to any particular
signal: they say how enriched a firing set has to be before the swap pays.

This is the pre-commitment instrument. Any proposed reranker can be scored
against this table BEFORE it is built, by asking what enrichment its mechanism
plausibly delivers. A signal that fires roughly at random loses by construction.

Note the distinction that a careless version of this analysis gets wrong:
"is the rank-k entry correct" is NOT "is the first correct entry at rank k".
Tolerance separation keeps entries >5% apart from EACH OTHER, but two of them
can still both sit within 5% of the same gold. The swap arithmetic needs the
first quantity; the coverage decomposition (oracle@k) needs the second. Both are
printed, and they do not agree, which is the point.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate import numeric_candidates
from channels import CHANNELS, demo_union
from common import load_pool, spec_for_channel
from rescore_ceiling import golds_for, hit, ranked_candidates


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="cap_recite")
    ap.add_argument("--splits", default="val")
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--model", default="google/gemma-4-31B")
    ap.add_argument("--revision", default="5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--demo-seed", type=int, default=1234)
    ap.add_argument("--seed-base", type=int, default=7000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ch = CHANNELS[args.channel]
    rel = ch.relation
    out = []
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        pool = load_pool(spec_for_channel(ch, split, args))
        keep = (set(pool) - demo_union(rel, args.demo_seed)) if split == "train" else set(pool)
        golds = golds_for(split, rel)
        shortlists = {}
        for s in sorted(keep):
            shortlists[s] = ranked_candidates(numeric_candidates(pool[s], args.channel), True)

        n = len(keep)
        print(f"\n=== {args.channel} / {split}  n={n} ===")
        print("A. Is the rank-k ENTRY inside tolerance?  (what a swap to rank k buys)")
        print(f"{'k':>3s} {'rows with a rank-k':>19s} {'entry correct':>14s} {'rate':>7s}")
        per_rank = []
        for k in range(1, args.depth + 1):
            have = cor = 0
            for s in sorted(keep):
                o, g = shortlists[s], golds.get(s)
                if len(o) < k or g is None:
                    continue
                have += 1
                cor += hit(o[k - 1], g)
            per_rank.append({"k": k, "rows": have, "correct": cor,
                             "rate": round(cor / max(have, 1), 4)})
            print(f"{k:3d} {have:19d} {cor:14d} {cor / max(have, 1):7.3f}")

        print("\nB. Swap arithmetic against rank-1, over rows that HAVE a rank-k entry")
        print(f"{'k':>3s} {'gain':>6s} {'loss':>6s} {'neutral':>8s} {'net rows':>9s} "
              f"{'enrichment needed':>18s}")
        swaps = []
        for k in range(2, args.depth + 1):
            gain = loss = neutral = 0
            for s in sorted(keep):
                o, g = shortlists[s], golds.get(s)
                if len(o) < k or g is None:
                    continue
                a, b = hit(o[0], g), hit(o[k - 1], g)
                if b and not a:
                    gain += 1
                elif a and not b:
                    loss += 1
                else:
                    neutral += 1
            ratio = loss / gain if gain else float("inf")
            swaps.append({"k": k, "gain": gain, "loss": loss, "neutral": neutral,
                          "net": gain - loss,
                          "enrichment_needed": None if gain == 0 else round(ratio, 2)})
            print(f"{k:3d} {gain:6d} {loss:6d} {neutral:8d} {gain - loss:+9d} "
                  f"{(f'{ratio:.2f}:1' if gain else 'inf'):>18s}")

        # The best a PERFECT selector over the shortlist can do, and the best a
        # perfect selector restricted to the top-2 can do.
        best_any = sum(1 for s in sorted(keep)
                       if golds.get(s) is not None
                       and any(hit(c, golds[s]) for c in shortlists[s]))
        best_top2 = sum(1 for s in sorted(keep)
                        if golds.get(s) is not None
                        and any(hit(c, golds[s]) for c in shortlists[s][:2]))
        r1 = per_rank[0]["correct"]
        print(f"\nC. rank-1 alone {r1}/{n} = {r1 / n:.4f}")
        print(f"   perfect choice within top-2   {best_top2}/{n} = {best_top2 / n:.4f}")
        print(f"   perfect choice within the whole shortlist {best_any}/{n} = {best_any / n:.4f}")
        print(f"   a coin-flip between rank-1 and rank-2 would score about "
              f"{(r1 + per_rank[1]['correct']) / (2 * n):.4f}")

        out.append({"channel": args.channel, "split": split, "n": n,
                    "per_rank": per_rank, "swaps": swaps,
                    "rank1": r1, "perfect_top2": best_top2, "perfect_any": best_any})

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
