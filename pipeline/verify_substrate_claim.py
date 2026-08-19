"""Adversarial verification of the substrate/union-coverage proposal.

Independent re-implementation, NOT a call into the proposal's own code, of:
  1. the permutation (null) control on SINGLE-frame and UNION coverage for
     hasCapacity, which is the load-bearing number in the proposal's section 0;
  2. the free-parameter sensitivity of the hasArea confidence router, which is
     the proposal's only concrete "do this instead" recommendation.

Reads only cached pools and public train/val gold. No test gold, no network.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate import numeric_candidates, predict_numeric
from channels import CHANNELS
from common import gold_primary, load_pool, rows_for, spec_for_channel
from substrate_ev import _args, frames_for, hit, top_share

NPERM = 400


def cand_sets(relation: str, split: str, frames: list[str]) -> tuple[list[str], dict]:
    a = _args()
    pools = {n: load_pool(spec_for_channel(CHANNELS[n], split, a)) for n in frames}
    gold = {r["SubjectEntity"]: r for r in rows_for(split, relation)}
    subs = [s for s in sorted(gold)
            if all(s in p for p in pools.values()) and gold_primary(gold[s])]
    per = {}
    for n in frames:
        per[n] = {s: numeric_candidates(pools[n][s], n) for s in subs}
    return subs, per


def coverage(subs, cands_by_sub, golds) -> float:
    return sum(1 for s in subs if any(hit(c, golds[s]) for c in cands_by_sub[s])) / len(subs)


def null_coverage(subs, cands_by_sub, golds, nperm=NPERM, seed=0) -> float:
    """Chance rate: give every subject ANOTHER subject's gold."""
    rng = random.Random(seed)
    order = list(subs)
    tot = 0.0
    for _ in range(nperm):
        perm = list(order)
        rng.shuffle(perm)
        # derangement-ish: reject fixed points by rotating them out
        fixed = [i for i in range(len(perm)) if perm[i] == order[i]]
        for i in fixed:
            j = rng.randrange(len(perm))
            perm[i], perm[j] = perm[j], perm[i]
        fake = {order[i]: golds[perm[i]] for i in range(len(order))}
        tot += coverage(order, cands_by_sub, fake)
    return tot / nperm


def windows(cands: list[float]) -> int:
    """How many disjoint 5% slots the candidate set spans (the haystack size)."""
    if not cands:
        return 0
    xs = sorted(set(cands))
    n, last = 0, None
    for x in xs:
        if last is None or not hit(x, last):
            n += 1
            last = x
    return n


def part1() -> None:
    rel, split = "hasCapacity", "val"
    frames = frames_for(rel)
    subs, per = cand_sets(rel, split, frames)
    gold = {r["SubjectEntity"]: r for r in rows_for(split, rel)}
    golds = {s: float(gold_primary(gold[s])[0]) for s in subs}
    print(f"=== permutation control, {rel}/{split}, n={len(subs)}, {NPERM} shuffles")
    print(f"  {'frames':>7} {'coverage':>9} {'chance':>7} {'chance-corrected':>17} {'median 5% slots':>16}")
    # order frames by marginal contribution, matching the proposal's greedy union
    order = ["cap_current", "cap_recite", "cap_rich", "cap_official",
             "cap_disambig", "cap_listing", "cap_infobox"]
    order = [f for f in order if f in per] + [f for f in per if f not in order]
    rows = []
    for k in (1, 2, 3, 4, 5, 6, 7):
        use = order[:k]
        merged = {s: [c for f in use for c in per[f][s]] for s in subs}
        cov = coverage(subs, merged, golds)
        ch = null_coverage(subs, merged, golds)
        cc = (cov - ch) / (1 - ch) if ch < 1 else float("nan")
        med = sorted(windows(merged[s]) for s in subs)[len(subs) // 2]
        rows.append((k, cov, ch, cc, med))
        print(f"  {k:7d} {cov:9.4f} {ch:7.4f} {cc:17.4f} {med:16d}")
    # single-frame, shipped frame alone
    merged = {s: per["cap_recite"][s] for s in subs}
    cov = coverage(subs, merged, golds)
    ch = null_coverage(subs, merged, golds)
    print(f"\n  cap_recite alone: coverage {cov:.4f}  chance {ch:.4f}  "
          f"chance-corrected {(cov-ch)/(1-ch):.4f}")
    print(f"  SEVEN-FRAME UNION: coverage {rows[-1][1]:.4f}  chance {rows[-1][2]:.4f}  "
          f"chance-corrected {rows[-1][3]:.4f}")
    print(f"  proposal's claim: 'only 6.2% of val rows have no correct value anywhere'")
    print(f"  chance-corrected residual deficit = {1-rows[-1][3]:.4f}")


def router_pair(relation: str, split: str, base: str, other: str,
                tie_to_base: bool = True) -> tuple[float, float, int]:
    a = _args()
    pa = load_pool(spec_for_channel(CHANNELS[base], split, a))
    pb = load_pool(spec_for_channel(CHANNELS[other], split, a))
    gold = {r["SubjectEntity"]: r for r in rows_for(split, relation)}
    subs = [s for s in sorted(gold) if s in pa and s in pb and gold_primary(gold[s])]
    na = nr = flips = 0
    for s in subs:
        g = float(gold_primary(gold[s])[0])
        _, sa = top_share(pa[s], base)
        _, sb = top_share(pb[s], other)
        ga, gb = predict_numeric(pa[s], base), predict_numeric(pb[s], other)
        oka = int(bool(ga) and hit(float(ga[0]), g))
        okb = int(bool(gb) and hit(float(gb[0]), g))
        take_b = (sb > sa) if tie_to_base else (sb >= sa)
        na += oka
        nr += okb if take_b else oka
        if take_b:
            flips += 1
    n = len(subs)
    return na / n, nr / n, flips


def router_testrows(relation: str, base: str, other: str, tie_to_base: bool = True) -> int:
    """How many TEST rows the router changes. Uses no test gold, only whether
    the emitted value differs."""
    a = _args()
    pa = load_pool(spec_for_channel(CHANNELS[base], "test", a))
    pb = load_pool(spec_for_channel(CHANNELS[other], "test", a))
    subs = [s for s in pa if s in pb]
    ch = 0
    for s in subs:
        _, sa = top_share(pa[s], base)
        _, sb = top_share(pb[s], other)
        take_b = (sb > sa) if tie_to_base else (sb >= sa)
        if not take_b:
            continue
        ga, gb = predict_numeric(pa[s], base), predict_numeric(pb[s], other)
        va = float(ga[0]) if ga else None
        vb = float(gb[0]) if gb else None
        if va != vb:
            ch += 1
    return ch


def part2() -> None:
    print("\n=== hasArea confidence router: EVERY pair, and the tie-break flipped")
    print(f"  {'pair':34s} {'tie':>6} {'A':>7} {'router':>7} {'delta':>8} {'val flips':>9} {'test rows':>9}")
    base = "area_recite"
    for other in [f for f in frames_for("hasArea") if f != base]:
        for tie in (True, False):
            a_, r_, fl = router_pair("hasArea", "val", base, other, tie)
            tr = router_testrows("hasArea", base, other, tie)
            print(f"  {base+'+'+other:34s} {'A' if tie else 'B':>6} {a_:7.4f} {r_:7.4f} "
                  f"{r_-a_:+8.4f} {fl:9d} {tr:9d}")
    print("\n=== same router on hasCapacity for reference (shipped base cap_recite)")
    for other in [f for f in frames_for("hasCapacity") if f != "cap_recite"]:
        a_, r_, fl = router_pair("hasCapacity", "val", "cap_recite", other, True)
        print(f"  cap_recite+{other:22s} {a_:7.4f} -> {r_:7.4f}  {r_-a_:+8.4f}")


if __name__ == "__main__":
    part1()
    part2()
