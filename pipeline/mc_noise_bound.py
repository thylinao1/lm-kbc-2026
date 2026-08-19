"""How much of the shipped numeric answer is Monte-Carlo noise, and how much is
the model's actual belief?

WHY THIS EXISTS. The proposed EXACT RESCORING instrument computes the model's
teacher-forced log P(v | prompt) instead of counting how often v appears in the
sampled pool. The advertised benefit is de-noising: a frequency over N draws is
a noisy estimate of a distribution the model can report exactly. That benefit is
BOUNDED ABOVE by the amount of noise actually present, and the noise is
measurable right now, with zero GPU, from the cached draws.

THE CONSTRUCTION, which is right by construction rather than by validation:
split each subject's draws into two DISJOINT halves and run the shipped selector
on each. If both halves return the same answer (within the grader's own 5% ball)
then the selector's output is not being decided by sampling noise on that row,
and no better estimator of the SAME functional (the sampling distribution over
values) can change it. Rows where the halves disagree are the only rows a
perfect de-noiser can touch.

That gives two numbers that matter more than any val delta:

  UNSTABLE   the fraction of rows a perfect de-noiser could even reach
  CEILING    oracle-minus-baseline restricted to those rows, i.e. the most
             de-noising could possibly be worth on this relation

and one number that sizes the OTHER two instruments:

  STABLE-WRONG-BUT-COVERED   rows where the answer is stable, wrong, and the
             pool contains a correct value. These are unreachable by any better
             estimate of sample frequency, so they are exactly the rows that
             verification or PMI would have to win, and they are the reason
             those instruments are a different bet rather than the same one.

Conservative on purpose: each half has N/2 draws, so half-split disagreement
OVERSTATES the instability at the full N. Agreement is also judged with the
stricter symmetric test |a-b|/max(a,b) <= 0.05, which overstates disagreement
again. Both errors push the ceiling UP, so a small ceiling here is a real bound.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate import numeric_candidates, predict_numeric
from channels import CHANNELS, demo_union
from common import gold_primary, load_pool, rows_for, spec_for_channel
from scorer import score_one_relation

TOL = 0.05


def same_ball(a: float, b: float) -> bool:
    """Stricter than the grader (larger value in the denominator), so that
    'disagree' is over-counted and the ceiling derived from it is an upper bound."""
    m = max(abs(a), abs(b))
    return m > 0 and abs(a - b) / m <= TOL


def one_value(preds: list[str]) -> float | None:
    if not preds:
        return None
    try:
        return float(preds[0])
    except ValueError:
        return None


def halves(draws: list[str], mode: str, seed: int) -> tuple[list[str], list[str]]:
    if mode == "stride":
        return draws[0::2], draws[1::2]
    idx = list(range(len(draws)))
    random.Random(seed).shuffle(idx)
    h = len(idx) // 2
    return [draws[i] for i in idx[:h]], [draws[i] for i in idx[h:2 * h]]


def analyse(channel: str, split: str, args, seeds: list[int]) -> dict:
    ch = CHANNELS[channel]
    rel = ch.relation
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

    full = {s: predict_numeric(d, channel) for s, d in pool.items()}
    base = score_one_relation(full, rel, split, subjects=keep)

    # instability, averaged over one deterministic stride split and several
    # random splits so the number is not an artifact of draw ordering
    modes = [("stride", 0)] + [("random", s) for s in seeds]
    unstable_any: set[str] = set()
    per_mode = []
    for mode, sd in modes:
        unstable = set()
        for s in sorted(keep):
            A, B = halves(pool[s], mode, sd)
            a, b = one_value(predict_numeric(A, channel)), one_value(predict_numeric(B, channel))
            if a is None or b is None:
                if a is not None or b is not None:
                    unstable.add(s)
                continue
            if not same_ball(a, b):
                unstable.add(s)
        unstable_any |= unstable
        per_mode.append({"mode": f"{mode}:{sd}", "n_unstable": len(unstable),
                         "rate": round(len(unstable) / max(len(keep), 1), 4)})

    # oracle restricted to the rows a perfect de-noiser could reach
    def covered(s: str) -> bool:
        g = golds.get(s)
        if g is None:
            return False
        return any(abs(c - g) / g <= TOL for c in numeric_candidates(pool.get(s, []), channel))

    def correct(s: str) -> bool:
        v = one_value(full.get(s, []))
        g = golds.get(s)
        return v is not None and g is not None and abs(v - g) / g <= TOL

    stable = keep - unstable_any
    strata = {
        "n_rows": len(keep),
        "unstable_any_split": len(unstable_any),
        "unstable_and_recoverable": sum(1 for s in unstable_any if covered(s) and not correct(s)),
        "stable_correct": sum(1 for s in stable if correct(s)),
        "stable_wrong_but_covered": sum(1 for s in stable if covered(s) and not correct(s)),
        "stable_wrong_not_covered": sum(1 for s in stable if not covered(s) and not correct(s)),
    }

    # exact ceiling for a perfect de-noiser: give it the oracle answer on every
    # unstable row, leave every stable row alone.
    denoise = dict(full)
    for s in unstable_any:
        g = golds.get(s)
        if g is None:
            continue
        cands = numeric_candidates(pool.get(s, []), channel)
        good = [c for c in cands if abs(c - g) / g <= TOL]
        if good:
            denoise[s] = [str(int(good[0])) if good[0] == int(good[0]) else str(good[0])]
    dn = score_one_relation(denoise, rel, split, subjects=keep)

    return {
        "channel": channel, "split": split, "n_rows": len(keep),
        "draws_per_subject": len(next(iter(pool.values()))),
        "baseline": base["macro_f1"],
        "per_split_instability": per_mode,
        "unstable_union_rate": round(len(unstable_any) / max(len(keep), 1), 4),
        "denoiser_ceiling": dn["macro_f1"],
        "denoiser_headroom": round(dn["macro_f1"] - base["macro_f1"], 4),
        "strata": strata,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", default="cap_recite")
    ap.add_argument("--splits", default="val")
    ap.add_argument("--seeds", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--model", default="google/gemma-4-31B")
    ap.add_argument("--revision", default="5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--demo-seed", type=int, default=1234)
    ap.add_argument("--seed-base", type=int, default=7000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    out = []
    for chn in [c.strip() for c in args.channels.split(",") if c.strip()]:
        for sp in [s.strip() for s in args.splits.split(",") if s.strip()]:
            r = analyse(chn, sp, args, seeds)
            out.append(r)
            print(f"\n=== {chn} / {sp}  n={r['n_rows']}  draws={r['draws_per_subject']} ===")
            print(f"  baseline {r['baseline']:.4f}")
            for m in r["per_split_instability"]:
                print(f"    half-split {m['mode']:10s} unstable {m['n_unstable']:3d} "
                      f"({m['rate']:.1%})")
            print(f"  UNION unstable over all splits: {r['unstable_union_rate']:.1%}")
            print(f"  PERFECT-DE-NOISER CEILING {r['denoiser_ceiling']:.4f} "
                  f"({r['denoiser_headroom']:+.4f})")
            print(f"  strata {json.dumps(r['strata'])}")
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
