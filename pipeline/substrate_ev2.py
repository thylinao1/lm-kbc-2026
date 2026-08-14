"""Part 2: validate the decorrelated-B simulator against REAL frame pairs, and
measure the same coverage structure on personHasCityOfDeath.

The simulator in substrate_ev.py extrapolates to "errors fully decorrelated".
That extrapolation is only worth anything if the simulator reproduces what
actually happens at the correlation levels we can measure. So: for every real
frame pair on hasCapacity, measure (a) the phi of their error vectors, (b) what
a parameter-free confidence router actually scores on that pair, then (c) run
the simulator with `shared` calibrated to reproduce that phi and check it lands
on the same number. Only then read the phi=0 column.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate import normalize, numeric_candidates, predict_numeric
from channels import CHANNELS
from common import gold_aliases, gold_primary, load_pool, rows_for, spec_for_channel
from substrate_ev import _args, calibration, frames_for, hit, phi, top_share


def per_row_frame_data(relation: str, split: str) -> dict[str, list[dict]]:
    """{frame: [{subject, share, value, correct}]} aligned on a common subject list."""
    a = _args()
    gold = {r["SubjectEntity"]: r for r in rows_for(split, relation)}
    subs = sorted(gold)
    out = {}
    for name in frames_for(relation):
        try:
            pool = load_pool(spec_for_channel(CHANNELS[name], split, a))
        except Exception:
            continue
        rows = []
        for s in subs:
            g = [float(x) for x in gold_primary(gold[s])] if s in gold else []
            if s not in pool or not g:
                rows.append(None)
                continue
            val, share = top_share(pool[s], name)
            got = predict_numeric(pool[s], name)
            rows.append({"subject": s, "share": share,
                         "value": float(got[0]) if got else 0.0,
                         "correct": int(bool(got) and hit(float(got[0]), g[0]))})
        out[name] = rows
    return out


def real_pair_routers(relation: str, split: str) -> list[dict]:
    data = per_row_frame_data(relation, split)
    base = "cap_recite" if relation == "hasCapacity" else "area_recite"
    A = data[base]
    res = []
    for name, B in data.items():
        if name == base:
            continue
        idx = [i for i in range(len(A)) if A[i] and B[i]]
        a = [A[i]["correct"] for i in idx]
        b = [B[i]["correct"] for i in idx]
        n = len(idx)
        # parameter-free router: take whichever frame is more concentrated
        router = sum(b[k] if B[idx[k]]["share"] > A[idx[k]]["share"] else a[k]
                     for k in range(n)) / n
        oracle = sum(max(a[k], b[k]) for k in range(n)) / n
        res.append({"pair": f"{base}+{name}", "n": n,
                    "phi": phi(a, b), "A": sum(a) / n, "B": sum(b) / n,
                    "router_argmax_conf": router, "row_oracle": oracle,
                    "delta_router": router - sum(a) / n,
                    "delta_oracle": oracle - sum(a) / n})
    return res


def sim_with_phi(rows: list[dict], beta: float, shared: float, n_rep: int = 3000,
                 seed: int = 1) -> dict:
    """Same construction as substrate_ev.simulate_second_substrate, but also
    returns the INDUCED phi so the knob can be read against measured values."""
    rng = random.Random(seed)
    n = len(rows)
    shares = [r["share"] for r in rows]
    corr_a = [r["correct"] for r in rows]
    alpha = sum(corr_a) / n
    pairs = sorted(zip(shares, corr_a))

    def p_of_share(x: float) -> float:
        near = sorted(pairs, key=lambda p: abs(p[0] - x))[:25]
        return sum(c for _, c in near) / len(near)

    phis, routers, oracles, balone = [], [], [], []
    for _ in range(n_rep):
        idx = list(range(n))
        rng.shuffle(idx)
        share_b = [shares[j] for j in idx]
        p_b = [p_of_share(s) for s in share_b]
        mean_p = sum(p_b) / n
        z = beta / mean_p if mean_p else 0.0
        corr_b = []
        for i in range(n):
            p = min(1.0, p_b[i] * z)
            if corr_a[i] == 0:
                p *= (1 - shared)
            else:
                p = min(1.0, p * (1 + shared))
            corr_b.append(1 if rng.random() < p else 0)
        phis.append(phi(corr_a, corr_b))
        routers.append(sum(corr_b[i] if share_b[i] > shares[i] else corr_a[i]
                           for i in range(n)) / n)
        oracles.append(sum(max(corr_a[i], corr_b[i]) for i in range(n)) / n)
        balone.append(sum(corr_b) / n)
    m = lambda v: sum(v) / len(v)
    return {"beta_target": beta, "shared": shared, "induced_phi": m(phis),
            "alpha": alpha, "B_alone": m(balone), "router": m(routers),
            "oracle": m(oracles), "delta_router": m(routers) - alpha,
            "delta_oracle": m(oracles) - alpha}


# ------------------------------------------------ cityOfDeath coverage structure


def death_coverage(split: str) -> dict:
    a = _args()
    gold = {r["SubjectEntity"]: r for r in rows_for(split, "personHasCityOfDeath")}
    names = [n for n, c in CHANNELS.items()
             if c.relation == "personHasCityOfDeath" and c.demo_strategy != "nn"
             and n != "death_year"]
    pools = {}
    for n in names:
        try:
            pools[n] = load_pool(spec_for_channel(CHANNELS[n], split, a))
        except Exception as e:
            print(f"  [skip {n}: {type(e).__name__}]")
    subs = [s for s in sorted(gold) if all(s in p for p in pools.values())]
    nonempty = [s for s in subs if gold_primary(gold[s])]
    per, cov_sets = {}, {}
    for name, pool in pools.items():
        ch = CHANNELS[name]
        got = set()
        for s in nonempty:
            aliases = {normalize(x) for grp in gold_aliases(gold[s]) for x in grp}
            drawn = set()
            for d in pool[s]:
                for c in ch.parse(d):
                    drawn.add(normalize(c))
            if drawn & aliases:
                got.add(s)
        per[name] = len(got) / len(nonempty)
        cov_sets[name] = got
    union = set().union(*cov_sets.values()) if cov_sets else set()
    return {"split": split, "n_rows": len(subs), "n_nonempty": len(nonempty),
            "per_frame_coverage_on_nonempty": per,
            "union_coverage_on_nonempty": len(union) / len(nonempty)}


def main() -> int:
    out = {}
    print("=== REAL frame pairs on hasCapacity/val: measured phi and measured router")
    print(f"  {'pair':32s} {'n':>4} {'phi':>7} {'A':>7} {'B':>7} {'router':>8} {'oracle':>8} {'d(router)':>10}")
    real = real_pair_routers("hasCapacity", "val")
    for r in real:
        print(f"  {r['pair']:32s} {r['n']:4d} {r['phi']:+7.3f} {r['A']:7.4f} {r['B']:7.4f} "
              f"{r['router_argmax_conf']:8.4f} {r['row_oracle']:8.4f} {r['delta_router']:+10.4f}")
    out["real_pairs_capacity"] = real

    print("\n=== REAL frame pairs on hasArea/val")
    reala = real_pair_routers("hasArea", "val")
    for r in reala:
        print(f"  {r['pair']:32s} {r['n']:4d} {r['phi']:+7.3f} {r['A']:7.4f} {r['B']:7.4f} "
              f"{r['router_argmax_conf']:8.4f} {r['row_oracle']:8.4f} {r['delta_router']:+10.4f}")
    out["real_pairs_area"] = reala

    print("\n=== SIMULATOR VALIDATION: match beta and phi to a real pair, compare routers")
    valrows = calibration("hasCapacity", "val", "cap_recite")["rows"]
    checks = []
    for r in real:
        best = None
        for shared in [x / 100 for x in range(0, 96, 5)]:
            s = sim_with_phi(valrows, r["B"], shared, n_rep=800)
            if best is None or abs(s["induced_phi"] - r["phi"]) < abs(best["induced_phi"] - r["phi"]):
                best = s
        checks.append({"pair": r["pair"], "measured_phi": r["phi"],
                       "sim_phi": best["induced_phi"], "shared": best["shared"],
                       "measured_router": r["router_argmax_conf"], "sim_router": best["router"],
                       "measured_oracle": r["row_oracle"], "sim_oracle": best["oracle"]})
        print(f"  {r['pair']:32s} phi meas {r['phi']:+.3f} / sim {best['induced_phi']:+.3f} "
              f"(shared={best['shared']:.2f})  router meas {r['router_argmax_conf']:.4f} / "
              f"sim {best['router']:.4f}   oracle meas {r['row_oracle']:.4f} / sim {best['oracle']:.4f}")
    out["sim_validation"] = checks

    print("\n=== phi=0 extrapolation (a genuinely decorrelated second substrate)")
    print(f"  {'beta':>6} {'phi':>7} {'B alone':>8} {'router':>8} {'oracle':>8} {'d(router) rel':>14} {'d overall':>10}")
    grid = []
    for beta in (0.15, 0.20, 0.25, 0.30, 0.3367, 0.40):
        s = sim_with_phi(valrows, beta, 0.0, n_rep=3000)
        s["d_overall"] = s["delta_router"] * 98 / 475
        s["d_overall_oracle"] = s["delta_oracle"] * 98 / 475
        grid.append(s)
        print(f"  {beta:6.3f} {s['induced_phi']:+7.3f} {s['B_alone']:8.4f} {s['router']:8.4f} "
              f"{s['oracle']:8.4f} {s['delta_router']:+14.4f} {s['d_overall']:+10.4f}")
    out["phi0_grid"] = grid

    print("\n=== personHasCityOfDeath coverage structure")
    for split in ("val",):
        d = death_coverage(split)
        print(f"  {split}: rows={d['n_rows']} non-empty gold={d['n_nonempty']}")
        for k, v in sorted(d["per_frame_coverage_on_nonempty"].items(), key=lambda kv: -kv[1]):
            print(f"    {k:16s} coverage-on-answerable = {v:.4f}")
        print(f"    UNION = {d['union_coverage_on_nonempty']:.4f}  -> knowledge deficit "
              f"{1-d['union_coverage_on_nonempty']:.4f} of answerable rows")
        out[f"death_{split}"] = d

    dest = Path("out"
                "30a8a394-1df6-488a-9dd1-a63c2c72b2a9/scratchpad/substrate_ev2.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1, default=str))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
