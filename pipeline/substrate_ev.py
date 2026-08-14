"""What a SECOND SUBSTRATE could be worth, measured on cached draws only.

Everything here reads the local pool cache and the public train/val gold. No
network, no external KB, no test gold. Closed book applies to the analysis too.

Three questions, in order:
  A. How much of hasCapacity is a KNOWLEDGE deficit shared by every frame?
     (per-frame coverage, and the union-coverage curve as frames are added)
  B. How well does a single substrate know when it knows? (calibration of the
     top-candidate tolerance share against correctness)
  C. If a second, ERROR-DECORRELATED substrate existed, what could any router
     extract from it? Simulated under the assumption most favourable to the
     pro-swap case: B's errors are fully independent of A's and B knows when it
     knows exactly as well as A does. That makes the output an UPPER BOUND.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate import numeric_candidates, predict_numeric
from channels import CHANNELS, demo_union
from common import TEST_ROWS, TOTAL_TEST_ROWS, gold_primary, load_pool, rows_for, spec_for_channel

TOL = 0.05


def _args():
    return argparse.Namespace(
        model="google/gemma-4-31B",
        revision="5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89",
        temperature=0.7, top_p=0.95, demo_seed=1234, seed_base=7000)


def hit(pred: float, gold: float) -> bool:
    return gold != 0 and abs(pred - gold) / abs(gold) <= TOL


def frames_for(relation: str) -> list[str]:
    return [n for n, c in CHANNELS.items()
            if c.relation == relation and c.demo_strategy != "nn"]


def load_frames(relation: str, split: str, names: list[str]) -> dict[str, dict[str, list[str]]]:
    a = _args()
    out = {}
    for n in names:
        try:
            out[n] = load_pool(spec_for_channel(CHANNELS[n], split, a))
        except Exception as e:                       # pool not generated for this split
            print(f"  [skip {n}/{split}: {type(e).__name__}]")
    return out


def clean_subjects(relation: str, split: str) -> list[str]:
    """Subjects usable for a CROSS-FRAME analysis: on train, drop the union of
    every frame's demos (channels.demo_union docstring explains why the single
    channel guard is not enough). val and test contain no demos at all."""
    rows = rows_for(split, relation)
    if split != "train":
        return [r["SubjectEntity"] for r in rows]
    demos = demo_union(relation)
    return [r["SubjectEntity"] for r in rows if r["SubjectEntity"] not in demos]


# ------------------------------------------------------- A. coverage structure


def coverage_table(relation: str, split: str) -> dict:
    names = frames_for(relation)
    pools = load_frames(relation, split, names)
    subs = [s for s in clean_subjects(relation, split)]
    gold = {r["SubjectEntity"]: r for r in rows_for(split, relation)}

    per_frame_cov, per_frame_real, covered_sets = {}, {}, {}
    for name, pool in pools.items():
        cov, real, seen = set(), set(), 0
        for s in subs:
            if s not in pool:
                continue
            seen += 1
            g = [float(x) for x in gold_primary(gold[s])]
            if not g:
                continue
            cands = numeric_candidates(pool[s], name)
            if any(hit(c, g[0]) for c in cands):
                cov.add(s)
            got = predict_numeric(pool[s], name)
            if got and hit(float(got[0]), g[0]):
                real.add(s)
        per_frame_cov[name] = len(cov) / seen if seen else 0.0
        per_frame_real[name] = len(real) / seen if seen else 0.0
        covered_sets[name] = cov

    scorable = [s for s in subs if all(s in p for p in pools.values())]
    n = len(scorable)
    if n == 0:
        return {"relation": relation, "split": split, "n_scorable": 0,
                "per_frame_coverage": per_frame_cov, "per_frame_realized": per_frame_real,
                "union_order": [], "union_curve": [], "union_coverage": 0.0,
                "covered_sets": {}, "scorable": [],
                "note": "no leak-free rows: some frame's demo set covers the whole split"}
    # union curve, greedily ordered by marginal contribution (best case ordering)
    order, remaining, curve = [], dict(covered_sets), []
    acc: set[str] = set()
    while remaining:
        best = max(remaining, key=lambda k: len((acc | remaining[k]) & set(scorable)))
        acc |= remaining.pop(best)
        order.append(best)
        curve.append(len(acc & set(scorable)) / n)
    return {"relation": relation, "split": split, "n_scorable": n,
            "per_frame_coverage": per_frame_cov, "per_frame_realized": per_frame_real,
            "union_order": order, "union_curve": curve,
            "union_coverage": curve[-1] if curve else 0.0,
            "covered_sets": {k: sorted(v & set(scorable)) for k, v in covered_sets.items()},
            "scorable": scorable}


# ------------------------------------------------------- B. calibration of A


def top_share(draws: list[str], channel: str) -> tuple[float, float]:
    """(winning value, share of candidate mass within the grader's tolerance of it)."""
    cands = numeric_candidates(draws, channel)
    if not cands:
        return 0.0, 0.0
    best, bestn = None, -1
    for c in sorted(set(cands)):
        k = sum(1 for d in cands if hit(d, c))
        if k > bestn:
            best, bestn = c, k
    return float(best), bestn / len(cands)


def calibration(relation: str, split: str, channel: str, bins=(0, .2, .4, .6, .8, 1.01)) -> dict:
    a = _args()
    pool = load_pool(spec_for_channel(CHANNELS[channel], split, a))
    gold = {r["SubjectEntity"]: r for r in rows_for(split, relation)}
    rows = []
    for s in clean_subjects(relation, split):
        if s not in pool:
            continue
        g = [float(x) for x in gold_primary(gold[s])]
        if not g:
            continue
        _, share = top_share(pool[s], channel)
        got = predict_numeric(pool[s], channel)
        ok = bool(got) and hit(float(got[0]), g[0])
        rows.append({"subject": s, "share": share, "correct": int(ok)})
    out = []
    for lo, hi in zip(bins, bins[1:]):
        sel = [r for r in rows if lo <= r["share"] < hi]
        out.append({"band": f"[{lo},{hi})", "n": len(sel),
                    "acc": sum(r["correct"] for r in sel) / len(sel) if sel else None})
    return {"channel": channel, "split": split, "n": len(rows),
            "acc": sum(r["correct"] for r in rows) / len(rows), "bands": out, "rows": rows}


# ------------------------------------------------------- C. decorrelated-B bound


def simulate_second_substrate(rows: list[dict], beta: float, n_rep: int = 4000,
                              seed: int = 0, shared: float = 0.0) -> dict:
    """Upper bound on what a router can extract from a decorrelated substrate B.

    rows: A's real per-row (share, correct) on this relation.
    beta: B's marginal accuracy on the relation (the thing we do not know).
    shared: 0.0 = B's errors are FULLY independent of A's (most favourable to the
            swap case). 1.0 = B fails on exactly the rows A fails on.

    Construction. B is given A's OWN empirical calibration curve, so B knows when
    it knows exactly as well as A does; its per-row correctness is then resampled
    to hit marginal accuracy beta. Router is parameter free: emit B's value iff
    B's confidence exceeds A's. Also reports the tuned-threshold optimum (an
    oracle over the router's own free parameter) and the row-level oracle.
    """
    rng = random.Random(seed)
    n = len(rows)
    shares = [r["share"] for r in rows]
    corr_a = [r["correct"] for r in rows]
    alpha = sum(corr_a) / n
    # A's calibration as a monotone map share -> P(correct); fitted on A itself.
    pairs = sorted(zip(shares, corr_a))
    def p_of_share(x: float) -> float:
        near = sorted(pairs, key=lambda p: abs(p[0] - x))[:25]
        return sum(c for _, c in near) / len(near)

    router, tuned, oracle, always_b = [], [], [], []
    for rep in range(n_rep):
        idx = list(range(n))
        rng.shuffle(idx)
        share_b = [shares[j] for j in idx]           # B's confidence: same distribution
        p_b = [p_of_share(s) for s in share_b]
        z = beta / (sum(p_b) / n) if sum(p_b) else 0.0
        corr_b = []
        for i in range(n):
            p = min(1.0, p_b[i] * z)
            if shared > 0 and corr_a[i] == 0:
                p *= (1 - shared)
            corr_b.append(1 if rng.random() < p else 0)
        r = sum(corr_b[i] if share_b[i] > shares[i] else corr_a[i] for i in range(n)) / n
        router.append(r)
        best = max(
            sum(corr_b[i] if (shares[i] < s and share_b[i] >= t) else corr_a[i]
                for i in range(n)) / n
            for s in (0.2, 0.3, 0.4, 0.5, 0.6) for t in (0.3, 0.4, 0.5, 0.6, 0.8))
        tuned.append(best)
        oracle.append(sum(max(corr_a[i], corr_b[i]) for i in range(n)) / n)
        always_b.append(sum(corr_b) / n)
    m = lambda v: sum(v) / len(v)
    return {"beta": beta, "shared": shared, "alpha_A": alpha,
            "B_alone": m(always_b), "router_argmax_conf": m(router),
            "router_tuned_oracle_params": m(tuned), "row_oracle": m(oracle),
            "delta_router": m(router) - alpha, "delta_tuned": m(tuned) - alpha,
            "delta_oracle": m(oracle) - alpha}


def phi(a: list[int], b: list[int]) -> float:
    n = len(a)
    n11 = sum(1 for i in range(n) if a[i] and b[i])
    n10 = sum(1 for i in range(n) if a[i] and not b[i])
    n01 = sum(1 for i in range(n) if not a[i] and b[i])
    n00 = n - n11 - n10 - n01
    den = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    return (n11 * n00 - n10 * n01) / den if den else 0.0


def main() -> int:
    out: dict = {}
    for rel in ("hasCapacity", "hasArea"):
        for split in ("val", "train"):
            print(f"\n=== coverage {rel} / {split}")
            t = coverage_table(rel, split)
            print(f"  n_scorable={t['n_scorable']}")
            for k in t["union_order"]:
                print(f"    {k:14s} cov={t['per_frame_coverage'][k]:.4f} "
                      f"realized={t['per_frame_realized'][k]:.4f}")
            print(f"  union curve (greedy best-case order): "
                  f"{[round(x,4) for x in t['union_curve']]}")
            print(f"  UNION COVERAGE = {t['union_coverage']:.4f}  "
                  f"-> knowledge deficit {1-t['union_coverage']:.4f}")
            out[f"cov_{rel}_{split}"] = {k: v for k, v in t.items()
                                        if k not in ("covered_sets", "scorable")}

    print("\n=== calibration cap_recite")
    for split in ("val", "train"):
        c = calibration("hasCapacity", split, "cap_recite")
        print(f"  {split}: n={c['n']} acc={c['acc']:.4f}")
        for b in c["bands"]:
            print(f"    share {b['band']:12s} n={b['n']:3d} acc="
                  f"{'-' if b['acc'] is None else format(b['acc'],'.4f')}")
        out[f"calib_{split}"] = {k: v for k, v in c.items() if k != "rows"}
        if split == "val":
            val_rows = c["rows"]

    print("\n=== within-substrate error correlation (val, hasCapacity)")
    a = _args()
    corr = {}
    for name in frames_for("hasCapacity"):
        try:
            pool = load_pool(spec_for_channel(CHANNELS[name], "val", a))
        except Exception:
            continue
        gold = {r["SubjectEntity"]: r for r in rows_for("val", "hasCapacity")}
        v = []
        for r in val_rows:
            s = r["subject"]
            g = [float(x) for x in gold_primary(gold[s])]
            got = predict_numeric(pool[s], name) if s in pool else []
            v.append(int(bool(got) and hit(float(got[0]), g[0])))
        corr[name] = v
    base = corr["cap_recite"]
    for name, v in corr.items():
        if name == "cap_recite":
            continue
        print(f"  phi(cap_recite, {name:14s}) = {phi(base, v):+.4f}   "
              f"acc={sum(v)/len(v):.4f}")
    out["phi"] = {n: phi(base, v) for n, v in corr.items() if n != "cap_recite"}

    print("\n=== decorrelated-B upper bound (val hasCapacity, n=%d)" % len(val_rows))
    print("  beta = B's own accuracy on the relation.  shared=0 means B's errors")
    print("  are FULLY independent of A's, which is the best case for a swap.")
    print(f"  {'beta':>6} {'shared':>7} {'B alone':>8} {'router':>8} {'tuned':>8} {'oracle':>8}"
          f" {'d(router)':>10} {'d(tuned)':>9}")
    sims = []
    for beta in (0.15, 0.20, 0.25, 0.30, 0.3608, 0.40):
        for shared in (0.0, 0.5):
            s = simulate_second_substrate(val_rows, beta, shared=shared)
            sims.append(s)
            print(f"  {beta:6.3f} {shared:7.2f} {s['B_alone']:8.4f} "
                  f"{s['router_argmax_conf']:8.4f} {s['router_tuned_oracle_params']:8.4f} "
                  f"{s['row_oracle']:8.4f} {s['delta_router']:+10.4f} {s['delta_tuned']:+9.4f}")
    out["sims"] = sims

    print("\n=== overall-score leverage (test row weights)")
    for r, w in sorted(TEST_ROWS.items(), key=lambda kv: -kv[1]):
        print(f"  {r:30s} {w:3d} rows  d(overall)/d(F1) = {w/TOTAL_TEST_ROWS:.5f}"
              f"   1 row = {1/TOTAL_TEST_ROWS:.5f} overall")
    dest = Path("out"
                "30a8a394-1df6-488a-9dd1-a63c2c72b2a9/scratchpad/substrate_ev.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1, default=str))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
