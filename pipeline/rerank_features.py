"""Learning-to-rank over hand-built features, as a replacement for hand-designed scores.

WHY THIS IS NOT ANOTHER TRANSFORMATION OF VOTE FREQUENCY. Every capacity selector this
campaign has measured is a formula somebody wrote down: share, share divided by a background
rate, a consensus weight swept over four values, a plausibility band, an anti-round rescue.
Each collapses many signals into one hand-chosen scalar, and each was rejected. The move here
is different in kind: extract every cheap signal per candidate, label each candidate by
whether it is within the grader's 5% tolerance of gold, and let a small model learn the
combination on train plus val. Cross-frame agreement, which failed as a hand-weighted
consensus, enters as one feature whose weight is fitted rather than swept.

WHY IT MIGHT SURVIVE WHERE THE OTHERS DIED. The rejected levers were discrete choices among a
handful of configurations, judged on 97 val rows where the per-relation standard error is
0.03 to 0.05. This is an estimation problem with roughly 2,700 labelled candidates across 197
subjects, which is a far better powered fit. That is an argument for better POWER, not for
correctness, so everything below is scored under subject-grouped cross-validation and the
comparison that matters is against the shipped selector on the same rows.

WHY IT IS CLOSED-BOOK. Every feature is computed from the model's own cached draws. Every
label comes from train and val gold, which the task ships for exactly this purpose. Nothing
external is read. No model weights are updated, so the 32B inference-time budget is untouched;
this is post-generation processing of the same kind as the tuned thresholds already shipped.
The rules question of whether fitting a supervised post-processor counts as "training" is
open and is being checked separately. Do not ship anything from this module until it is
answered.
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate import numeric_candidates, tolerance_support
from channels import CHANNELS, demo_ids, pick_demos
from common import load_pool, rows_for, spec_for_channel

TOL = 0.05

# Capacity frames whose pools exist for train, val AND test. Verified by
# inventorying pools/*/*.manifest.json; a frame missing any split cannot supply
# a cross-frame feature at prediction time and is excluded rather than
# back-filled with a zero, which would leak the frame's absence as a signal.
CAP_FRAMES = ("cap_recite", "cap_rich", "cap_official", "cap_current",
              "cap_nn", "cap_disambig")
AREA_FRAMES = ("area_recite", "area_lead", "area_nn", "area_infobox",
               "area_listing")


def _args(ns: argparse.Namespace | None = None) -> argparse.Namespace:
    return ns or argparse.Namespace(
        model="google/gemma-4-31B",
        revision="5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89",
        temperature=0.7, top_p=0.95, demo_seed=1234, seed_base=7000)


def load_frames(frames: tuple[str, ...], split: str, ns=None) -> dict[str, dict]:
    """{frame: {subject: [draw, ...]}} for every frame that has this split."""
    out: dict[str, dict] = {}
    for f in frames:
        try:
            out[f] = load_pool(spec_for_channel(CHANNELS[f], split, _args(ns)))
        except Exception as exc:                     # missing pool is expected
            print(f"  [skip] {f}/{split}: {type(exc).__name__}", file=sys.stderr)
    return out


def roundness(v: float) -> int:
    """Trailing-zero count of the rounded integer. A crude but real proxy for
    'this is a generic guess'. 10000 scores 4, 20355 scores 0."""
    n = int(round(v))
    if n == 0:
        return 0
    z = 0
    while n % 10 == 0:
        n //= 10
        z += 1
    return z


def candidate_rows(frames_pools: dict[str, dict], primary: str,
                   subjects: list[str]) -> dict[str, list[dict]]:
    """Per subject, one feature row per tolerance-separated candidate.

    Candidates are proposed by the PRIMARY frame only. Letting every frame
    propose would change the candidate set relative to the shipped system and
    confound a better ranking with a wider net; the other frames enter purely as
    agreement evidence about the primary frame's candidates.
    """
    # Background rate: fraction of SUBJECTS whose primary-frame pool contains a
    # value within tolerance. Computed across this split only, from draws alone.
    per_subject_vals = {s: numeric_candidates(frames_pools[primary].get(s, []), primary)
                        for s in subjects}
    universe = sorted({v for vs in per_subject_vals.values() for v in vs})
    n_subj = max(len(subjects), 1)
    bg: dict[float, float] = {}
    for cand in universe:
        hits = sum(1 for vs in per_subject_vals.values()
                   if any(abs(cand - w) / w <= TOL for w in vs if w))
        bg[cand] = hits / n_subj

    out: dict[str, list[dict]] = {}
    for s in subjects:
        vals = per_subject_vals.get(s, [])
        if not vals:
            out[s] = []
            continue
        # tolerance-separated representatives, strongest support first
        scored = sorted(((len(tolerance_support(vals, c, TOL)), c) for c in set(vals)),
                        key=lambda t: (-t[0], t[1]))
        reps, taken = [], []
        for cnt, c in scored:
            if any(abs(c - t) / t <= TOL for t in taken):
                continue
            taken.append(c)
            reps.append((c, cnt))

        n = len(vals)
        shares = [cnt / n for _, cnt in reps]
        top = shares[0] if shares else 0.0
        ent = -sum(p * math.log(p) for p in shares if p > 0)
        logs = [math.log10(v) for v in vals if v > 0]
        spread = (max(logs) - min(logs)) if logs else 0.0

        feats = []
        for rank, (c, cnt) in enumerate(reps, start=1):
            share = cnt / n
            # cross-frame agreement: share of EACH other frame's draws that fall
            # within tolerance of this candidate, and how many frames back it
            agree, present = [], 0
            for f, pool in frames_pools.items():
                if f == primary:
                    continue
                fv = numeric_candidates(pool.get(s, []), f)
                if not fv:
                    continue
                k = len(tolerance_support(fv, c, TOL))
                agree.append(k / len(fv))
                present += 1 if k else 0
            feats.append({
                "subject": s, "value": c,
                "share": share,
                "rank": rank,
                "log_rank": math.log(rank),
                "count": cnt,
                "share_ratio_to_top": share / top if top else 0.0,
                "n_draws": n,
                "n_candidates": len(reps),
                "pool_entropy": ent,
                "log_spread": spread,
                "log_value": math.log10(c) if c > 0 else 0.0,
                "roundness": roundness(c),
                "background": bg.get(c, 0.0),
                "lift": share / (bg.get(c, 0.0) + 0.05),
                "agree_mean": (sum(agree) / len(agree)) if agree else 0.0,
                "agree_max": max(agree) if agree else 0.0,
                "frames_backing": present,
            })
        out[s] = feats
    return out


FEATURES = ["share", "rank", "log_rank", "count", "share_ratio_to_top", "n_draws",
            "n_candidates", "pool_entropy", "log_spread", "log_value", "roundness",
            "background", "lift", "agree_mean", "agree_max", "frames_backing"]


def label(feat_rows: list[dict], gold: float) -> list[int]:
    return [1 if (gold and abs(f["value"] - gold) / gold <= TOL) else 0
            for f in feat_rows]


def build(relation: str, primary: str, frames: tuple[str, ...],
          splits=("train", "val"), ns=None, exclude_demos: bool = True):
    """Return (X, y, groups, values) as plain python lists, plus a per-split index."""
    C = CHANNELS[primary]
    demos = set(demo_ids(pick_demos(C.relation, C.n_demos,
                                    _args(ns).demo_seed, C.demo_strategy)))
    X, y, groups, values, split_of = [], [], [], [], []
    for sp in splits:
        pools = load_frames(frames, sp, ns)
        if primary not in pools:
            continue
        gold_by = {r["SubjectEntity"]: r for r in rows_for(sp, relation)}
        subs = [s for s in pools[primary] if s in gold_by]
        if sp == "train" and exclude_demos:
            subs = [s for s in subs if s not in demos]
        rows = candidate_rows(pools, primary, subs)
        for s in subs:
            g = gold_by[s].get("ObjectEntities") or []
            gv = float(g[0][0]) if g and g[0] else 0.0
            fr = rows.get(s, [])
            if not fr:
                continue
            for f, lab in zip(fr, label(fr, gv)):
                X.append([f[k] for k in FEATURES])
                y.append(lab)
                groups.append(f"{sp}:{s}")
                values.append(f["value"])
                split_of.append(sp)
    return X, y, groups, values, split_of
