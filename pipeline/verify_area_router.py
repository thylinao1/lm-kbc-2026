"""Is the hasArea confidence router real? Paired statistics, done honestly.

The proposal reports 'val 0.8500 -> 0.8800, paired bootstrap 90% CI
[+0.0100,+0.0600], excludes zero' for area_recite + area_lead. This recomputes
the paired structure from scratch: how many rows move up, how many down, the
exact McNemar readout, and a paired bootstrap over rows.

Also asks the question the proposal does not: the partner frame was chosen from
four candidates on the same split the result is read on.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate import predict_numeric
from channels import CHANNELS, demo_union
from common import gold_primary, load_pool, rows_for, spec_for_channel
from substrate_ev import _args, frames_for, hit, top_share


def paired(relation: str, split: str, base: str, other: str, drop_demos: bool):
    a = _args()
    pa = load_pool(spec_for_channel(CHANNELS[base], split, a))
    pb = load_pool(spec_for_channel(CHANNELS[other], split, a))
    gold = {r["SubjectEntity"]: r for r in rows_for(split, relation)}
    subs = [s for s in sorted(gold) if s in pa and s in pb and gold_primary(gold[s])]
    if drop_demos:
        demos = demo_union(relation)
        subs = [s for s in subs if s not in demos]
    d = []
    for s in subs:
        g = float(gold_primary(gold[s])[0])
        _, sa = top_share(pa[s], base)
        _, sb = top_share(pb[s], other)
        ga, gb = predict_numeric(pa[s], base), predict_numeric(pb[s], other)
        oka = int(bool(ga) and hit(float(ga[0]), g))
        okb = int(bool(gb) and hit(float(gb[0]), g))
        okr = okb if sb > sa else oka
        d.append((oka, okr))
    return d


def report(tag, d):
    n = len(d)
    if n == 0:
        print(f"  {tag:34s} NO ROWS")
        return
    up = sum(1 for a, r in d if r > a)
    dn = sum(1 for a, r in d if r < a)
    fa = sum(a for a, _ in d) / n
    fr = sum(r for _, r in d) / n
    rng = random.Random(7)
    deltas = []
    for _ in range(20000):
        idx = [rng.randrange(n) for _ in range(n)]
        deltas.append(sum(d[i][1] - d[i][0] for i in idx) / n)
    deltas.sort()
    lo, hi = deltas[int(0.05 * len(deltas))], deltas[int(0.95 * len(deltas))]
    mc = abs(up - dn) / (up + dn) ** 0.5 if up + dn else 0.0
    print(f"  {tag:34s} n={n:3d}  A={fa:.4f} router={fr:.4f} delta={fr-fa:+.4f}  "
          f"up={up} down={dn}  90% CI [{lo:+.4f},{hi:+.4f}]  McNemar z={mc:.2f}")


if __name__ == "__main__":
    print("=== hasArea, val (the split the partner frame was CHOSEN on)")
    for other in [f for f in frames_for("hasArea") if f != "area_recite"]:
        report(f"area_recite+{other}", paired("hasArea", "val", "area_recite", other, False))

    print("\n=== hasArea, leak-free TRAIN (demo_union excluded)")
    for other in [f for f in frames_for("hasArea") if f != "area_recite"]:
        report(f"area_recite+{other}", paired("hasArea", "train", "area_recite", other, True))

    print("\n=== hasCapacity, val, for comparison")
    for other in [f for f in frames_for("hasCapacity") if f != "cap_recite"]:
        report(f"cap_recite+{other}", paired("hasCapacity", "val", "cap_recite", other, False))
