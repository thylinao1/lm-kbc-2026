"""How much headroom can ANY reranker of the existing pool possibly buy?

This is the dry-run that must happen BEFORE spending GPU on exact rescoring,
verification, or PMI. All three of those instruments have one thing in common:
they never change what is IN the pool, only which pool member is emitted. So
their joint ceiling is the ORACLE RERANKER -- a rule that always picks a
tolerance-correct candidate when the pool contains one.

Three quantities, all measured with the official scorer via scorer.py:

  BASELINE   the shipped predict_numeric output (what I score today)
  ORACLE@inf pick a correct candidate whenever one exists anywhere in the pool
  ORACLE@k   same, but the reranker may only look at the top-k candidates by
             frequency, which is the realistic shortlist an exact-rescoring
             pass would be given

ORACLE@inf - BASELINE bounds every reranking idea from above. ORACLE@k - BASELINE
bounds a reranker that is only shown k candidates. If the gap is small, no
amount of better probability estimation matters and the GPU should go elsewhere.

"Frequency" is deliberately reported under BOTH readings, because they differ
and the difference is not cosmetic:

  raw       distinct drawn values ranked by tolerance-support count. Five
            near-duplicates (10000, 10200, 10500, ...) can occupy all five
            slots, so this is what a reranker scoring raw candidate strings
            actually sees.
  separated greedy tolerance-separated shortlist: after taking a candidate,
            every value inside its 5% ball is removed. This is the ceiling for
            a reranker whose shortlist is deduplicated first.

No gold is read to BUILD any shortlist. Gold enters only in the oracle's choice
and in scoring, which is the definition of a ceiling.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate import format_number, numeric_candidates, predict_numeric, tolerance_support
from channels import CHANNELS, demo_union
from common import gold_primary, load_pool, rows_for, spec_for_channel
from scorer import paired_bootstrap, score_one_relation

TOL = 0.05


def ranked_candidates(vals: list[float], separated: bool) -> list[float]:
    """Candidate shortlist ordered by frequency. Reads draws only, never gold.

    Ordering key: (-tolerance support, -exact count, value). Fully deterministic
    so the k-th slot is well defined and the result is reproducible.
    """
    if not vals:
        return []
    exact = Counter(vals)
    scored = sorted(
        set(vals),
        key=lambda c: (-len(tolerance_support(vals, c, TOL)), -exact[c], c),
    )
    if not separated:
        return scored
    out: list[float] = []
    taken: list[float] = []
    for c in scored:
        if any(abs(c - t) / t <= TOL for t in taken):
            continue
        out.append(c)
        taken.append(c)
    return out


def hit(cand: float, gold: float) -> bool:
    return gold != 0 and abs(cand - gold) / gold <= TOL


def oracle_preds(pool: dict[str, list[str]], channel: str, golds: dict[str, float],
                 k: int | None, separated: bool) -> dict[str, list[str]]:
    """Emit a correct candidate if one is inside the top-k shortlist, else the
    top-1 (i.e. what a reranker with no useful signal would keep)."""
    out = {}
    for subj, draws in pool.items():
        vals = numeric_candidates(draws, channel)
        order = ranked_candidates(vals, separated)
        if not order:
            out[subj] = []
            continue
        short = order if k is None else order[:k]
        g = golds.get(subj)
        pick = next((c for c in short if g is not None and hit(c, g)), order[0])
        out[subj] = [format_number(pick)]
    return out


def golds_for(split: str, relation: str) -> dict[str, float]:
    g = {}
    for r in rows_for(split, relation):
        prim = gold_primary(r)
        if not prim:
            continue
        try:
            g[r["SubjectEntity"]] = float(str(prim[0]).replace(",", ""))
        except ValueError:
            continue
    return g


def analyse(channel: str, split: str, args, ks: list[int]) -> dict:
    ch = CHANNELS[channel]
    rel = ch.relation
    pool = load_pool(spec_for_channel(ch, split, args))
    golds = golds_for(split, rel)

    # leakage guard: on train, drop every subject that is a demonstration for
    # ANY channel of this relation (channels.demo_union, not the single-channel
    # guard, which is provably insufficient).
    if split == "train":
        keep = set(pool) - demo_union(rel, args.demo_seed)
    else:
        keep = set(pool)

    base = score_one_relation({s: predict_numeric(d, channel) for s, d in pool.items()},
                              rel, split, subjects=keep)

    rows = {"channel": channel, "split": split, "n_scored": base["n_rows"],
            "baseline": base["macro_f1"], "curves": {}}

    # candidate-count anatomy: how many things would a reranker have to score?
    n_raw, n_sep = [], []
    for subj in keep:
        vals = numeric_candidates(pool.get(subj, []), channel)
        n_raw.append(len(ranked_candidates(vals, False)))
        n_sep.append(len(ranked_candidates(vals, True)))
    rows["candidates_per_row"] = {
        "raw_mean": round(sum(n_raw) / max(len(n_raw), 1), 2),
        "raw_max": max(n_raw, default=0),
        "raw_total": sum(n_raw),
        "separated_mean": round(sum(n_sep) / max(len(n_sep), 1), 2),
        "separated_max": max(n_sep, default=0),
        "separated_total": sum(n_sep),
    }

    for sep in (False, True):
        tag = "separated" if sep else "raw"
        curve = {}
        for k in ks:
            p = oracle_preds(pool, channel, golds, k, sep)
            curve[str(k)] = score_one_relation(p, rel, split, subjects=keep)["macro_f1"]
        p_inf = oracle_preds(pool, channel, golds, None, sep)
        s_inf = score_one_relation(p_inf, rel, split, subjects=keep)
        curve["inf"] = s_inf["macro_f1"]
        rows["curves"][tag] = curve
        if sep:
            rows["oracle_bootstrap_vs_baseline"] = paired_bootstrap(
                base["f1_vector"], s_inf["f1_vector"])

    # where does the correct value sit in the frequency ranking, when present?
    rank_hist: Counter[str] = Counter()
    for subj in sorted(keep):
        vals = numeric_candidates(pool.get(subj, []), channel)
        order = ranked_candidates(vals, True)
        g = golds.get(subj)
        if g is None:
            continue
        pos = next((i + 1 for i, c in enumerate(order) if hit(c, g)), None)
        rank_hist["absent" if pos is None else
                  "1" if pos == 1 else "2" if pos == 2 else "3" if pos == 3 else
                  "4-5" if pos <= 5 else "6-10" if pos <= 10 else "11+"] += 1
    rows["gold_rank_separated"] = dict(rank_hist)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", default="cap_recite")
    ap.add_argument("--splits", default="val")
    ap.add_argument("--ks", default="1,2,3,5,8,10,14")
    ap.add_argument("--model", default="google/gemma-4-31B")
    ap.add_argument("--revision", default="5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--demo-seed", type=int, default=1234)
    ap.add_argument("--seed-base", type=int, default=7000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",") if x.strip()]

    reports = []
    for chn in [c.strip() for c in args.channels.split(",") if c.strip()]:
        for sp in [s.strip() for s in args.splits.split(",") if s.strip()]:
            r = analyse(chn, sp, args, ks)
            reports.append(r)
            print(f"\n=== {chn} / {sp}  n={r['n_scored']} ===")
            print(f"  baseline (shipped predict_numeric): {r['baseline']:.4f}")
            for tag, curve in r["curves"].items():
                cells = "  ".join(f"k={k}:{v:.4f}" for k, v in curve.items())
                print(f"  ORACLE[{tag:9s}] {cells}")
            print(f"  candidates/row: {json.dumps(r['candidates_per_row'])}")
            print(f"  gold rank (separated shortlist): {json.dumps(r['gold_rank_separated'])}")
            bs = r["oracle_bootstrap_vs_baseline"]
            print(f"  oracle-vs-baseline paired bootstrap: {bs['point']:+.4f} "
                  f"CI [{bs['ci_lo']:+.4f},{bs['ci_hi']:+.4f}] "
                  f"rows up {bs['rows_up']} down {bs['rows_down']}")

    if args.out:
        Path(args.out).write_text(json.dumps(reports, indent=1))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
