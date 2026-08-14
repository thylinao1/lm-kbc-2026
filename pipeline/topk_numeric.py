"""Anatomy of the two numeric relations, and whether the metric permits a
DECISION better than "emit exactly one number".

Everything here is offline: it reads only the cached pools plus train/val gold.
No network, no external KB, no submissions. TEST is read only to count how many
rows a rule would change and to project a score; test gold is hidden.

--------------------------------------------------------------------------
THE ARITHMETIC, stated once so it can be checked
--------------------------------------------------------------------------
Both numeric relations have EXACTLY ONE gold value per row and zero empty-gold
rows (verified on train and val, see `verify_gold_shape`). Under evaluate.py:

    numeric_true_positives() walks preds, and for each pred walks golds,
    `break`ing on the first match and recording it in `matched_gts`. With a
    single gold, tp is therefore capped at 1 no matter how many preds hit.

So for a k-prediction row:   tp = 1 if any pred is within 5% of gold else 0
                             P  = tp/k,  R = tp/1
                             F1 = 2*(1/k)*1 / ((1/k)+1) = 2/(k+1)  on a hit
                             F1 = 0                                on a miss
    =>  E[F1](k) = C_k * 2/(k+1),  C_k = P(gold within 5% of one of the top k)

Adding the (k+1)-th candidate raises E[F1] iff

    C_{k+1} * 2/(k+2)  >  C_k * 2/(k+1)
    <=>  C_{k+1}/C_k   >  (k+2)/(k+1)          <-- NOTE the direction
    <=>  p_{k+1}       >  C_k / (k+1)

The brief stated this as C_{k+1}/C_k > (k+1)/(k+2). That form is wrong and is
vacuous: C is non-decreasing so C_{k+1}/C_k >= 1 > (k+1)/(k+2) always, i.e. it
would say "always hedge". The correct threshold is (k+2)/(k+1) > 1. The k=1->2
case is the familiar one: cov@2 must exceed 1.5 * cov@1.

--------------------------------------------------------------------------
THE TWO TRAPS, handled explicitly
--------------------------------------------------------------------------
(i)  evaluate.py deduplicates by normalize_string, which maps punctuation to
     spaces: "10000" -> "10000" but "10,000" -> "10 000". They are NOT
     duplicates, so both would consume precision while contributing at most one
     tp. Every candidate I emit is rendered by format_number(), which is
     comma-free, and I additionally assert the normalized forms are distinct.
(ii) Two candidates within 5% of EACH OTHER can never both be true positives
     (tp is capped at 1) but both are counted in len(preds). Worse, they cannot
     even cover different golds: pred c is correct iff gold lies in
     [c/1.05, c/0.95], and those windows are disjoint iff c2/c1 > 1.05/0.95.
     So the candidate list is built with a hard separation ratio of 1.05/0.95
     = 1.10526..., which makes the per-candidate hit events mutually exclusive
     and makes C_k the plain SUM of the p_j.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate import (format_number, numeric_candidates, predict_numeric,
                       tolerance_support)
from channels import CHANNELS, demo_ids, pick_demos
from common import gold_primary, load_pool, rows_for, spec_for_channel
from scorer import paired_bootstrap, score_one_relation

TOL = 0.05
# c2/c1 must exceed this for the two tolerance windows to be disjoint.
SEP = 1.05 / 0.95
KMAX = 8


# ------------------------------------------------------------------ shared


def default_args() -> argparse.Namespace:
    return argparse.Namespace(
        model="google/gemma-4-31B",
        revision="5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89",
        temperature=0.7, top_p=0.95, demo_seed=1234, seed_base=7000)


def demo_subjects(channel: str, demo_seed: int = 1234) -> set[str]:
    """The train subjects that appear in this channel's own demo block.

    Identical construction to the guard in tune.py. A train row that is its own
    demonstration has its gold in its prompt, so its draws are memorisation.
    """
    ch = CHANNELS[channel]
    return set(demo_ids(pick_demos(ch.relation, ch.n_demos, demo_seed,
                                   ch.demo_strategy)))


def verify_gold_shape(relation: str) -> dict:
    out = {}
    for split in ("train", "val"):
        rows = rows_for(split, relation)
        n_gold = [len(r.get("ObjectEntities") or []) for r in rows]
        out[split] = {"rows": len(rows), "min_golds": min(n_gold),
                      "max_golds": max(n_gold),
                      "empty_gold_rows": sum(1 for g in n_gold if g == 0)}
    return out


def gold_value(row: dict) -> float | None:
    g = gold_primary(row)
    if not g:
        return None
    try:
        return float(str(g[0]).replace(",", ""))
    except ValueError:
        return None


def hits(cand: float, gold: float, tol: float = TOL) -> bool:
    return gold != 0 and abs(cand - gold) / abs(gold) <= tol


# ------------------------------------------------------------------ ranking


def ranked_candidates(draws: list[str], channel: str, kmax: int = KMAX,
                      tol: float = TOL, sep: float = SEP,
                      seed_shipped: bool = True) -> list[tuple[float, float]]:
    """Top-k tolerance-separated candidates as (value, vote share).

    Vote share uses the SAME tolerance-support notion the shipped ball selector
    uses: the share of parsed draws lying within `tol` of the candidate. Each
    draw contributes at most one parsed value (see parse_capacity_recite /
    parse_area_recite, both return n[:1]), so the share is a fraction of draws.

    `seed_shipped` puts the shipped predict_numeric() answer at rank 1. That
    makes the k=1 policy EXACTLY the shipped policy, so any adaptive policy
    built on this list is a strict extension of what is on the board rather
    than a different system that happens to also emit one number. It matters
    for hasCapacity, whose shipped selector is the log-cluster median and can
    return a value no draw proposed.
    """
    vals = numeric_candidates(draws, channel)
    if not vals:
        return []
    n = len(vals)

    def share(c: float) -> float:
        return len(tolerance_support(vals, c, tol)) / n

    picked: list[tuple[float, float]] = []
    if seed_shipped:
        out = predict_numeric(draws, channel)
        if out:
            try:
                c0 = float(out[0])
                picked.append((c0, share(c0)))
            except ValueError:
                pass

    for c in sorted(set(vals), key=lambda v: (-share(v), v)):
        if len(picked) >= kmax:
            break
        if any(max(c, p) / min(c, p) <= sep for p, _ in picked):
            continue      # trap (ii): would share a tolerance window
        picked.append((c, share(c)))
    return picked[:kmax]


def build_table(channel: str, split: str, args, kmax: int = KMAX,
                seed_shipped: bool = True, exclude_demos: bool = True) -> list[dict]:
    """One record per scored row: candidates, shares, and (where gold is known)
    which candidate is the hit."""
    ch = CHANNELS[channel]
    pool = load_pool(spec_for_channel(ch, split, args))
    gold_rows = {r["SubjectEntity"]: r for r in rows_for(split, ch.relation)}
    drop = demo_subjects(channel, args.demo_seed) if (split == "train" and exclude_demos) else set()

    table = []
    for subj, draws in pool.items():
        row = gold_rows.get(subj)
        if row is None or subj in drop:
            continue
        cands = ranked_candidates(draws, channel, kmax, seed_shipped=seed_shipped)
        g = gold_value(row) if split != "test" else None
        hit_at = None
        if g is not None:
            for i, (c, _) in enumerate(cands):
                if hits(c, g):
                    hit_at = i + 1
                    break
        table.append({
            "subject": subj, "gold": g,
            "values": [c for c, _ in cands],
            "shares": [s for _, s in cands],
            "hit_at": hit_at,           # 1-indexed rank of the correct candidate
        })
    return table


# ------------------------------------------------------------------ cov@k


def cov_at_k(table: list[dict], kmax: int = KMAX) -> list[float]:
    n = len(table)
    return [sum(1 for r in table if r["hit_at"] is not None and r["hit_at"] <= k) / n
            for k in range(1, kmax + 1)]


def emit(values: list[float]) -> list[str]:
    """Render k candidates, asserting trap (i) cannot bite: the grader's own
    normalize_string must give k distinct keys, otherwise a slot is paid for
    twice."""
    out = [format_number(v) for v in values]
    from scorer import _ev
    keys = [_ev().normalize_string(s) for s in out]
    assert len(set(keys)) == len(keys), f"duplicate normalized preds: {out}"
    assert not any("," in s for s in out), f"comma in numeric pred: {out}"
    return out


def score_policy(table: list[dict], relation: str, split: str,
                 ks: dict[str, int] | int) -> dict:
    """Score a policy with the OFFICIAL scorer. ks is either a fixed k or a
    per-subject k."""
    preds = {}
    for r in table:
        k = ks if isinstance(ks, int) else ks[r["subject"]]
        preds[r["subject"]] = emit(r["values"][:k])
    return score_one_relation(preds, relation, split,
                              subjects={r["subject"] for r in table})
