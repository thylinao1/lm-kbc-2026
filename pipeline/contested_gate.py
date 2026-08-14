"""Where is a reranker allowed to fire, and what is the bar once it gets there?

rank_ev.py says a blanket rank-1 -> rank-2 swap must invert a 1.92:1 base rate,
which is a bar no weak signal clears. But a reranker does not have to fire
everywhere. If it fires only where the frequency ranking has no confident
opinion, the base rate inside the firing set is what it must beat, and that can
be much closer to even.

THE GATE MUST READ NO GOLD, or the free parameter is fitted and the lever joins
background-lift on the pile of sweep peaks. Two gates are evaluated, both
computable from draws alone on the TEST split:

  SPLIT-HALF  fire where two disjoint halves of the subject's own draws select
              different answers. Zero free parameters. This is the model telling
              me, in its own sampling variance, that it has not decided.
  SHARE-RATIO fire where support(rank2)/support(rank1) >= r. One free parameter,
              so r is reported as a curve rather than chosen here; any use of it
              must fix r by a draws-only rule declared before scoring.

For each gate I report the count of rows it fires on (on val AND on test, since
the gate needs no gold), and inside the firing set the contested base rate
23:12-style, which IS the accuracy a verifier must exceed to break even.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate import numeric_candidates, predict_numeric, tolerance_support
from channels import CHANNELS, demo_union
from common import load_pool, spec_for_channel
from rescore_ceiling import golds_for, hit, ranked_candidates

TOL = 0.05


def unstable(draws: list[str], channel: str) -> bool:
    """Deterministic stride split. No gold, no randomness, no parameter."""
    a = predict_numeric(draws[0::2], channel)
    b = predict_numeric(draws[1::2], channel)
    if not a or not b:
        return bool(a) != bool(b)
    try:
        x, y = float(a[0]), float(b[0])
    except ValueError:
        return True
    return abs(x - y) / max(x, y) > TOL


def report(channel: str, split: str, args) -> dict:
    ch = CHANNELS[channel]
    rel = ch.relation
    pool = load_pool(spec_for_channel(ch, split, args))
    keep = (set(pool) - demo_union(rel, args.demo_seed)) if split == "train" else set(pool)
    golds = golds_for(split, rel)
    has_gold = bool(golds)

    rows = []
    for s in sorted(keep):
        vals = numeric_candidates(pool[s], channel)
        order = ranked_candidates(vals, True)
        if len(order) < 2:
            continue
        sup = [len(tolerance_support(vals, c, TOL)) for c in order[:2]]
        g = golds.get(s)
        rows.append({
            "subject": s,
            "ratio": sup[1] / sup[0] if sup[0] else 1.0,
            "unstable": unstable(pool[s], channel),
            "r1_ok": hit(order[0], g) if g is not None else None,
            "r2_ok": hit(order[1], g) if g is not None else None,
            "any_ok": any(hit(c, g) for c in order) if g is not None else None,
        })

    def summarise(sel: list[dict], tag: str) -> dict:
        n = len(sel)
        d = {"gate": tag, "rows_fired": n,
             "share_of_split": round(n / max(len(keep), 1), 4)}
        if has_gold and n:
            g1 = sum(1 for r in sel if r["r1_ok"] and not r["r2_ok"])
            g2 = sum(1 for r in sel if r["r2_ok"] and not r["r1_ok"])
            d.update({
                "contested": g1 + g2,
                "rank1_wins": g1, "rank2_wins": g2,
                "break_even_accuracy": (round(g1 / (g1 + g2), 4) if g1 + g2 else None),
                "rank1_correct": sum(1 for r in sel if r["r1_ok"]),
                "shortlist_covered": sum(1 for r in sel if r["any_ok"]),
            })
        return d

    out = {"channel": channel, "split": split, "rows_with_2plus": len(rows),
           "n_split": len(keep), "gates": []}
    out["gates"].append(summarise(rows, "all rows with >=2 candidates"))
    out["gates"].append(summarise([r for r in rows if r["unstable"]], "split-half unstable"))
    out["gates"].append(summarise([r for r in rows if not r["unstable"]], "split-half stable"))
    for r0 in (0.5, 0.7, 0.8, 0.9, 1.0):
        out["gates"].append(summarise([r for r in rows if r["ratio"] >= r0],
                                      f"share ratio >= {r0}"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="cap_recite")
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--model", default="google/gemma-4-31B")
    ap.add_argument("--revision", default="5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--demo-seed", type=int, default=1234)
    ap.add_argument("--seed-base", type=int, default=7000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    out = []
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        r = report(args.channel, split, args)
        out.append(r)
        print(f"\n=== {args.channel} / {split}  n={r['n_split']} "
              f"(rows with >=2 candidates: {r['rows_with_2plus']}) ===")
        hdr = (f"{'gate':28s} {'fires':>6s} {'contested':>10s} {'r1':>4s} {'r2':>4s} "
               f"{'break-even acc':>15s}")
        print(hdr)
        for g in r["gates"]:
            if "contested" in g:
                be = g["break_even_accuracy"]
                print(f"{g['gate']:28s} {g['rows_fired']:6d} {g['contested']:10d} "
                      f"{g['rank1_wins']:4d} {g['rank2_wins']:4d} "
                      f"{(f'{be:.3f}' if be is not None else 'n/a'):>15s}")
            else:
                print(f"{g['gate']:28s} {g['rows_fired']:6d} "
                      f"{'(no gold on this split)':>10s}")
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
