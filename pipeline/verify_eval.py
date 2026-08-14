"""Scoring for the yes/no verification probe. Rule fixed before the pools exist.

THE BAR IS DELIBERATELY HIGHER THAN THE DUELS'. This is the third instrument tried on
hasCapacity. Two have already measured DEAD. Continuing to test the same relation until
something clears a 90 percent interval is how a campaign talks itself into shipping noise, so
this probe has to clear the duel bar AND hold its sign inside both strata of a gate that was
defined on other grounds and reads no gold.

    SHIP  val macro-F1 beats the shipped selector, paired bootstrap 90% CI excludes zero,
          AND the sign holds inside both the split-half-stable and split-half-unstable strata.
    DEAD  anything else.

The split-half gate is a deterministic stride split of the draws, with agreement judged by the
grader's own 5 percent tolerance. Zero free parameters. On val it fires on 37 of 97 rows,
where the shipped selector scores 0.1892 against 0.4667 on the stable rows.

ACCEPT-EVERYTHING CONTROL is printed whatever the verdict. A verification probe whose margin
is positive for nearly every candidate is reporting a yes-bias, and its argmax is then the
ranking of that bias rather than of knowledge. That is the documented failure mode for this
family and it is the first thing to check.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate import numeric_candidates, predict_numeric, tolerance_support
from channels import CHANNELS, demo_union
from common import POOL_DIR, load_pool, rows_for, spec_for_channel

TOL = 0.05
SEED = 20260812


def load_probe(split: str, channel: str, k: int) -> dict:
    p = POOL_DIR / "hasCapacity" / f"verify_{channel}_{split}_k{k}.jsonl"
    if not p.exists():
        return {}
    with open(p) as fh:
        return {json.loads(l)["subject"]: json.loads(l) for l in fh if l.strip()}


def unstable(draws: list[str], ch: str) -> bool:
    a, b = draws[0::2], draws[1::2]
    pa, pb = predict_numeric(a, ch), predict_numeric(b, ch)
    if not pa or not pb:
        return True
    x, y = float(pa[0]), float(pb[0])
    return abs(x - y) / max(x, y) > TOL


def paired_bootstrap(a: list[float], b: list[float], n: int = 5000):
    d = [y - x for x, y in zip(a, b)]
    rng = random.Random(SEED)
    m = sorted(statistics.fmean(rng.choices(d, k=len(d))) for _ in range(n))
    return {"point": statistics.fmean(d), "ci_lo": m[int(0.05 * n)], "ci_hi": m[int(0.95 * n)],
            "up": sum(1 for x in d if x > 0), "down": sum(1 for x in d if x < 0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="cap_recite")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--model", default="google/gemma-4-31B")
    ap.add_argument("--revision", default="5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--demo-seed", type=int, default=1234)
    ap.add_argument("--seed-base", type=int, default=7000)
    args = ap.parse_args()
    ch = args.channel

    for split in ("val", "train"):
        probe = load_probe(split, ch, args.k)
        if not probe:
            print(f"[{split}] no probe file")
            continue
        pool = load_pool(spec_for_channel(CHANNELS[ch], split, args))
        gold = {r["SubjectEntity"]: float(r["ObjectEntities"][0][0])
                for r in rows_for(split, "hasCapacity") if r.get("ObjectEntities")}
        banned = demo_union("hasCapacity", args.demo_seed)
        man = POOL_DIR / "hasCapacity" / f"verify_{ch}_{split}_k{args.k}.manifest.json"
        if man.exists():
            banned = banned | set(json.load(open(man))["demo_subjects"])
        subs = [s for s in probe if s in gold and not (split == "train" and s in banned)]
        if not subs:
            print(f"[{split}] 0 leak-free rows, skipping")
            continue

        allm = [m for s in subs for m in probe[s]["margins"].values()]
        print(f"\n[{split}] n={len(subs)}  ACCEPT-EVERYTHING CONTROL: "
              f"mean margin {statistics.fmean(allm):+.3f}, "
              f"accepted at margin>0 {sum(1 for m in allm if m>0)/len(allm):.3f}, "
              f"median spread within a row "
              f"{statistics.median([max(probe[s]['margins'].values())-min(probe[s]['margins'].values()) for s in subs if len(probe[s]['margins'])>1]):.3f}")

        strata = {"unstable": {"s": [], "p": []}, "stable": {"s": [], "p": []}}
        S, P = [], []
        for s in subs:
            g = gold[s]
            hit = lambda v: 1.0 if (v is not None and abs(v - g) / g <= TOL) else 0.0
            sp_ = predict_numeric(pool[s], ch)
            shipped = hit(float(sp_[0]) if sp_ else None)
            m = probe[s]["margins"]
            if m:
                vals = numeric_candidates(pool[s], ch)
                n = max(len(vals), 1)
                tie = {c: len(tolerance_support(vals, float(c), TOL)) / n for c in m}
                best = max(m, key=lambda c: (m[c], tie.get(c, 0.0)))
                pv = hit(float(best))
            else:
                pv = 0.0
            S.append(shipped); P.append(pv)
            k = "unstable" if unstable(pool[s], ch) else "stable"
            strata[k]["s"].append(shipped); strata[k]["p"].append(pv)

        bs = paired_bootstrap(S, P)
        print(f"  shipped {statistics.fmean(S):.4f}   verify {statistics.fmean(P):.4f}   "
              f"delta {bs['point']:+.4f}  90% CI [{bs['ci_lo']:+.4f},{bs['ci_hi']:+.4f}]  "
              f"up {bs['up']}/dn {bs['down']}")
        signs = []
        for k, v in strata.items():
            if not v["s"]:
                continue
            d = statistics.fmean(v["p"]) - statistics.fmean(v["s"])
            signs.append(d)
            print(f"    {k:9s} n={len(v['s']):3d}  shipped {statistics.fmean(v['s']):.4f}  "
                  f"verify {statistics.fmean(v['p']):.4f}  delta {d:+.4f}")
        if split == "val":
            ok = bs["ci_lo"] > 0 and all(x > 0 for x in signs)
            print(f"\n  PRE-COMMITTED VERDICT: {'SHIP' if ok else 'DEAD'}  "
                  f"(needs CI low > 0 AND both strata positive)")
            print(f"  if the point estimate held on test: {bs['point']*98/475:+.4f} overall")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
