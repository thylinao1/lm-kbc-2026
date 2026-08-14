"""Accuracy as a function of the number of draws: the decisive test of whether
a BETTER ESTIMATE of the sampling distribution can be worth anything.

THE LOGIC, and it is an argument about the estimator rather than a val delta.
Sample frequency over N draws is a Monte-Carlo estimate of one functional: the
probability that this prompt, at this temperature, emits this value. Exact
teacher-forced rescoring proposed as instrument (i) is pitched as a de-noised
version of the same quantity. If that is what it is, then its accuracy is the
N -> infinity point of this curve. So measure the curve.

  * If accuracy RISES with N and has not flattened by N=100, de-noising has
    room and exact rescoring is worth GPU.
  * If accuracy is FLAT in N while the identity of the argmax keeps churning
    (mc_noise_bound.py measures 33% churn between disjoint halves), then the
    churn is between candidates of equal accuracy. The variance is real and
    costs nothing. A perfect estimator of the same functional sits at the flat
    end of the curve and buys +0.0000 by construction.

The second case does NOT kill instruments (ii) verification and (iii) PMI,
because those are different functionals, not better estimates of this one. It
kills the de-noising ARGUMENT for instrument (i), which is the only argument
offered for it.

Free parameters: none that read gold. Subsample sizes are a fixed geometric
ladder; R resamples per size with a fixed seed; the selector is the shipped one.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate import numeric_candidates, predict_numeric, tolerance_support
from channels import CHANNELS, demo_union
from common import gold_primary, load_pool, rows_for, spec_for_channel

TOL = 0.05


def freq_top1(vals: list[float]) -> float | None:
    """Pure frequency argmax, no clustering: the plainest reading of 'the model's
    most likely value'. Reported alongside the shipped selector so the curve is
    not an artifact of the log-cluster median."""
    if not vals:
        return None
    return max(sorted(set(vals)),
               key=lambda c: (len(tolerance_support(vals, c, TOL)), -c))


def accuracy(pool: dict, channel: str, golds: dict, keep: set, n: int,
             rep: int, seed: int) -> tuple[float, float, dict, dict]:
    """Returns (mean shipped acc, mean freq acc, per-row shipped rate, per-row freq rate).

    The per-row rates are what the paired bootstrap needs: rows are the noise
    unit on this metric, and comparing two draw counts on the SAME rows is far
    tighter than comparing two independent means.
    """
    rng = random.Random(seed)
    per_sel: dict[str, float] = {}
    per_frq: dict[str, float] = {}
    for s in sorted(keep):
        g = golds.get(s)
        if g is None:
            continue
        draws = pool[s]
        hs = hf = 0
        for _ in range(rep):
            sub = draws if n >= len(draws) else rng.sample(draws, n)
            p = predict_numeric(sub, channel)
            if p:
                try:
                    if abs(float(p[0]) - g) / g <= TOL:
                        hs += 1
                except ValueError:
                    pass
            f = freq_top1(numeric_candidates(sub, channel))
            if f is not None and abs(f - g) / g <= TOL:
                hf += 1
        per_sel[s] = hs / rep
        per_frq[s] = hf / rep
    m = max(len(per_sel), 1)
    return (sum(per_sel.values()) / m, sum(per_frq.values()) / m, per_sel, per_frq)


def extrapolate(curve: list[dict], key: str, n_lo: int, n_hi: int) -> dict:
    """Fit acc(n) = a - b/n through two measured points and report a, the
    infinite-draw limit. This is the accuracy a PERFECT estimator of the
    sampling distribution would reach, which is exactly what exact rescoring
    claims to be. Two points, not a least-squares fit, because the choice of
    which points to fit is then explicit rather than hidden in a weighting.
    """
    d = {r["n"]: r[key] for r in curve}
    if n_lo not in d or n_hi not in d:
        return {}
    b = (d[n_hi] - d[n_lo]) / (1.0 / n_lo - 1.0 / n_hi)
    a = d[n_hi] + b / n_hi
    return {"fit_points": [n_lo, n_hi], "b": round(b, 4),
            "acc_at_infinite_draws": round(a, 4),
            "headroom_over_n_max": round(a - d[n_hi], 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="cap_recite")
    ap.add_argument("--splits", default="val")
    ap.add_argument("--sizes", default="5,10,20,30,50,75,100")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--model", default="google/gemma-4-31B")
    ap.add_argument("--revision", default="5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--demo-seed", type=int, default=1234)
    ap.add_argument("--seed-base", type=int, default=7000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]

    ch = CHANNELS[args.channel]
    rel = ch.relation
    out = []
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        pool = load_pool(spec_for_channel(ch, split, args))
        keep = (set(pool) - demo_union(rel, args.demo_seed)) if split == "train" else set(pool)
        golds = {}
        for r in rows_for(split, rel):
            g = gold_primary(r)
            if g:
                try:
                    golds[r["SubjectEntity"]] = float(str(g[0]).replace(",", ""))
                except ValueError:
                    pass
        maxn = len(next(iter(pool.values())))
        print(f"\n=== {args.channel} / {split}  rows={len(keep)}  draws={maxn} "
              f"reps={args.reps} ===")
        print(f"{'n':>5s}  {'shipped selector':>16s}  {'freq top-1':>11s}")
        rowsout, per_row = [], {}
        for n in sizes:
            if n > maxn:
                continue
            rep = 1 if n >= maxn else args.reps
            a, b, ps, pf = accuracy(pool, args.channel, golds, keep, n, rep,
                                    20260812 + n)
            print(f"{n:5d}  {a:16.4f}  {b:11.4f}")
            rowsout.append({"n": n, "reps": rep, "shipped": round(a, 4),
                            "freq_top1": round(b, 4)})
            per_row[n] = (ps, pf)

        # paired comparison across draw counts on the SAME rows
        from scorer import paired_bootstrap
        pairs = [(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)
                 if sizes[i + 1] <= maxn]
        print("\n  paired bootstrap, same rows, more draws minus fewer:")
        paired = []
        for lo, hi in pairs:
            if lo not in per_row or hi not in per_row:
                continue
            ks = sorted(per_row[lo][0])
            bs = paired_bootstrap([per_row[lo][0][k] for k in ks],
                                  [per_row[hi][0][k] for k in ks])
            print(f"    n={lo:3d} -> {hi:3d}  shipped {bs['point']:+.4f} "
                  f"CI [{bs['ci_lo']:+.4f},{bs['ci_hi']:+.4f}]")
            paired.append({"from": lo, "to": hi, "point": round(bs["point"], 4),
                           "ci": [round(bs["ci_lo"], 4), round(bs["ci_hi"], 4)]})

        ex = {k: {} for k in ("shipped", "freq_top1")}
        for key in ex:
            for lo in (20, 30, 50):
                e = extrapolate(rowsout, key, lo, maxn)
                if e:
                    ex[key][f"{lo}->{maxn}"] = e
        print("\n  infinite-draw extrapolation acc(n) = a - b/n:")
        for key, v in ex.items():
            for tag, e in v.items():
                print(f"    {key:10s} fit {tag:9s} -> a={e['acc_at_infinite_draws']:.4f} "
                      f"(headroom over n={maxn}: {e['headroom_over_n_max']:+.4f})")

        out.append({"channel": args.channel, "split": split, "rows": len(keep),
                    "curve": rowsout, "paired": paired, "extrapolation": ex})
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
