"""Scoring for the forced-choice duel instrument, with the decision rule fixed in advance.

WRITTEN AND COMMITTED BEFORE THE DUEL POOLS EXISTED. That is the whole point. The last
capacity lever this campaign tried reported +0.0364 and measured +0.0000 once its free
parameter was fixed by a rule declared ahead of time, because the +0.0364 was the argmax of an
eleven point sweep whose sign flipped one grid step away. So everything below is pinned now.

PRE-COMMITTED PRIMARY RULE, zero free parameters:
    score(i) = mean over j != i of P(i beats j), averaged over both presentation orders.
    Prediction = argmax. Ties broken by the shipped frame's own vote share.
This is a Borda count over the duel matrix. No fitting, no threshold, no blend weight.

PRE-COMMITTED SECONDARY RULES, reported for sensitivity, NEVER used to select:
    Copeland   count of pairwise wins at P > 0.5, Borda as tie-break.
    Blend      mean of (duel Borda) and (vote share), both min-max normalised per row.
Reporting a sensitivity family and then shipping its best member is the failure this protocol
exists to prevent. The primary is the primary whatever the others do.

PRE-COMMITTED SHIP RULE, per relation:
    SHIP  only if the pooled clean-row paired bootstrap 90% CI excludes zero from above AND
          the sign of the delta agrees on train-clean and val taken separately.
    PROBE if the CI excludes zero but the two splits disagree in sign.
    DEAD  otherwise.
The two-split sign agreement is there because this campaign's single most repeated failure is
a pooled number carried entirely by one split, which is exactly how the cross-frame agreement
reranker died earlier today at +0.2206 train and +0.0000 val.

CLEAN ROWS. val in full (no val subject is ever a demonstration), plus train rows that are in
no channel's demo set for the relation (channels.demo_union) and are not among the eight
subjects used as duel demonstrations. Anything less than this is the leak that already burned
one result today.

KNOWN AND DISCLOSED LIMIT. Duels rank the primary frame's top-6 tolerance-separated
candidates. The shipped log-cluster selector picks a value outside that set on 2 of 97 val and
2 of 98 test capacity rows, and 0 of 100 area rows. On those rows the two systems differ in
which candidates exist, not only in how they are ordered. Reported, not hidden.
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


def load_duels(relation: str, channel: str, split: str, k: int) -> dict:
    p = POOL_DIR / relation / f"duels_{channel}_{split}_k{k}.jsonl"
    if not p.exists():
        return {}
    out = {}
    for line in open(p):
        if line.strip():
            r = json.loads(line)
            out[r["subject"]] = r
    return out


def borda(rec: dict) -> dict[str, float]:
    """Mean probability each candidate beats a uniformly chosen opponent."""
    cands = rec["candidates"]
    wins: dict[str, list[float]] = {c: [] for c in cands}
    for key, p_a in rec["pair_probs"].items():
        a, b = key.split("|")
        if a in wins:
            wins[a].append(p_a)
        if b in wins:
            wins[b].append(1.0 - p_a)
    return {c: (statistics.fmean(v) if v else 0.5) for c, v in wins.items()}


def copeland(rec: dict) -> dict[str, float]:
    cands = rec["candidates"]
    sc = {c: 0.0 for c in cands}
    for key, p_a in rec["pair_probs"].items():
        a, b = key.split("|")
        if a in sc and b in sc:
            sc[a] += 1.0 if p_a > 0.5 else 0.0
            sc[b] += 1.0 if p_a < 0.5 else 0.0
    return sc


def vote_share(draws: list[str], channel: str, cands: list[str]) -> dict[str, float]:
    vals = numeric_candidates(draws, channel)
    n = max(len(vals), 1)
    return {c: len(tolerance_support(vals, float(c), TOL)) / n for c in cands}


def _norm(d: dict[str, float]) -> dict[str, float]:
    if not d:
        return {}
    lo, hi = min(d.values()), max(d.values())
    return {k: (0.5 if hi == lo else (v - lo) / (hi - lo)) for k, v in d.items()}


def pick(rec: dict, draws: list[str], channel: str, method: str) -> float | None:
    cands = rec["candidates"]
    if not cands:
        return None
    if len(cands) == 1:
        return float(cands[0])
    vs = vote_share(draws, channel, cands)
    if method == "borda":
        sc, tie = borda(rec), vs
    elif method == "copeland":
        b = borda(rec)
        sc, tie = copeland(rec), b
    elif method == "blend":
        b, nb = borda(rec), None
        nb, nv = _norm(b), _norm(vs)
        sc = {c: 0.5 * nb[c] + 0.5 * nv[c] for c in cands}
        tie = vs
    else:
        raise ValueError(method)
    best = max(cands, key=lambda c: (sc.get(c, 0.0), tie.get(c, 0.0)))
    return float(best)


def paired_bootstrap(a: list[float], b: list[float], n: int = 5000):
    d = [y - x for x, y in zip(a, b)]
    rng = random.Random(SEED)
    m = sorted(statistics.fmean(rng.choices(d, k=len(d))) for _ in range(n))
    return {"point": statistics.fmean(d), "ci_lo": m[int(0.05 * n)], "ci_hi": m[int(0.95 * n)],
            "up": sum(1 for x in d if x > 0), "down": sum(1 for x in d if x < 0), "n": len(d)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--relation", default="hasCapacity")
    ap.add_argument("--channel", default="cap_recite")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--rows", type=int, default=0, help="test row count for overall units")
    ap.add_argument("--model", default="google/gemma-4-31B")
    ap.add_argument("--revision", default="5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--demo-seed", type=int, default=1234)
    ap.add_argument("--seed-base", type=int, default=7000)
    args = ap.parse_args()
    rel, ch = args.relation, args.channel

    man = POOL_DIR / rel / f"duels_{ch}_val_k{args.k}.manifest.json"
    duel_demos = set(json.load(open(man))["demo_subjects"]) if man.exists() else set()
    banned = demo_union(rel, args.demo_seed) | duel_demos
    print(f"clean-row filter: {len(banned)} train subjects excluded "
          f"({len(duel_demos)} of them duel demonstrations)")

    per_split: dict[str, dict[str, list[float]]] = {}
    for split in ("train", "val"):
        duels = load_duels(rel, ch, split, args.k)
        if not duels:
            print(f"  [{split}] no duel file yet")
            continue
        pool = load_pool(spec_for_channel(CHANNELS[ch], split, args))
        gold = {r["SubjectEntity"]: r for r in rows_for(split, rel)}
        subs = [s for s in duels if s in gold and not (split == "train" and s in banned)]
        if not subs:
            # Expected for hasArea: area_lead100 uses all 100 train rows as
            # demonstrations, so the relation has no leak-free train row at all
            # and val is the only honest split. Say so rather than crash.
            print(f"  [{split}] 0 leak-free rows (every train subject is some "
                  f"channel's demonstration). Skipping this split.")
            continue
        acc: dict[str, list[float]] = {m: [] for m in ("shipped", "borda", "copeland", "blend")}
        for s in subs:
            g = gold[s].get("ObjectEntities") or []
            gv = float(g[0][0]) if g and g[0] else 0.0
            hit = lambda v: 1.0 if (v is not None and gv and abs(v - gv) / gv <= TOL) else 0.0
            p = predict_numeric(pool[s], ch)
            acc["shipped"].append(hit(float(p[0]) if p else None))
            for m in ("borda", "copeland", "blend"):
                acc[m].append(hit(pick(duels[s], pool[s], ch, m)))
        per_split[split] = acc
        print(f"  [{split}] n={len(subs)}  " +
              "  ".join(f"{m}={statistics.fmean(v):.4f}" for m, v in acc.items()))

    if "val" not in per_split:
        print("\nval duels missing; cannot apply the ship rule yet.")
        return 0

    print("\nPOOLED CLEAN ROWS (train-clean + val), against the shipped selector:")
    pooled = {m: sum((per_split[s][m] for s in per_split), []) for m in
              ("shipped", "borda", "copeland", "blend")}
    rows = args.rows or {"hasCapacity": 98, "hasArea": 100}.get(rel, 100)
    verdicts = {}
    for m in ("borda", "copeland", "blend"):
        bs = paired_bootstrap(pooled["shipped"], pooled[m])
        signs = []
        for s in per_split:
            d = statistics.fmean(per_split[s][m]) - statistics.fmean(per_split[s]["shipped"])
            signs.append(d)
        agree = all(x > 0 for x in signs) or all(x < 0 for x in signs)
        tag = ("SHIP" if (bs["ci_lo"] > 0 and agree and signs[0] > 0)
               else "PROBE" if bs["ci_lo"] > 0 else "DEAD")
        verdicts[m] = tag
        print(f"  {m:9s} {statistics.fmean(pooled[m]):.4f}  delta {bs['point']:+.4f}  "
              f"90% CI [{bs['ci_lo']:+.4f},{bs['ci_hi']:+.4f}]  up {bs['up']}/dn {bs['down']}  "
              f"per-split {[round(x,4) for x in signs]}  -> {tag}"
              f"{'  (PRIMARY)' if m=='borda' else ''}")
        print(f"            if it held on test: {bs['point']*rows/475:+.4f} overall")

    print(f"\nPRE-COMMITTED VERDICT for {rel}: primary rule 'borda' is {verdicts['borda']}.")
    print("Secondary rules are sensitivity only and must not be shipped over the primary.")

    duels_t = load_duels(rel, ch, "test", args.k)
    if duels_t:
        pool_t = load_pool(spec_for_channel(CHANNELS[ch], "test", args))
        changed = sum(1 for s in duels_t
                      if (lambda a, b: a is None or b is None or abs(a - b) / max(a, 1e-9) > TOL)(
                          pick(duels_t[s], pool_t[s], ch, "borda"),
                          (lambda p: float(p[0]) if p else None)(predict_numeric(pool_t[s], ch))))
        print(f"TEST: borda changes {changed}/{len(duels_t)} rows (direction unknown, no gold).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
