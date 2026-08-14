"""Background-lift re-ranking for hasCapacity, run under a pre-committed rule.

IDEA. Within one subject's pool, a candidate's raw vote share cannot tell a
recalled fact from a fallback guess. But a value the model emits for MANY
DIFFERENT subjects is a generic prior (10000 is "a stadium"), while a value it
emits for this subject and rarely elsewhere is more likely a recalled fact. So
score a candidate by its within-subject share divided by its cross-subject
background share:

    score(v) = share_subject(v) / (q(v) + eps)

WHY THIS IS NOT ONE OF THE FAILED LEVERS. It is not cross-frame consensus (a
single frame ranks), not confidence routing (no per-subject frame choice), and
not the plausibility band (per-value density, not a global range filter). The
adversarial review's strongest test was whether q is a within-pool histogram
artifact, i.e. frequency in disguise: it re-estimated q from INDEPENDENT frames
and the effect kept its sign and shape, which a histogram artifact cannot do.

WHY THE PRE-COMMITMENT. The proposer reported +0.0364, but that was the argmax
of an 11-point eps sweep whose sign flips one grid step away (eps=0.01 gives
-0.0242). Honest split-half tuning gave mean +0.0280 with a 95% CI of
[-0.0120, +0.0723]. So eps is fixed here by a rule that reads only DRAWS, never
gold or accuracy, and the result is looked at exactly once.

    eps := median of q over contended values (distinct drawn values with >= 2 draws)

DECISION RULE, SET BEFORE LOOKING (from the review):
    >= +5 rows on the 165 clean train+val rows AND bootstrap CI low > -0.01  -> ship
    +1 to +3 rows                                                            -> drop
Anything else is a judgment call reported honestly, not quietly resolved.

KNOWN LIMITATION, stated up front. q is the model's generic prior PLUS the true
capacity distribution, and those overlap: 43% of gold values sit at q >= 0.05.
Every row the rule breaks has gold in {5000, 10000}. So it is a directional bet
that test venues are atypically sized, which is exactly the thing that cannot be
checked closed-book.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate import format_number, numeric_candidates, tolerance_support
from channels import CHANNELS, demo_ids, pick_demos
from common import PoolSpec, load_pool, rows_for, split_sha

REL = "hasCapacity"
Q_FRAMES = ("cap_recite", "cap_disambig", "cap_current")


def spec_for(ch: str, split: str, args) -> PoolSpec:
    C = CHANNELS[ch]
    d = pick_demos(C.relation, C.n_demos, args.demo_seed, C.demo_strategy)
    return PoolSpec(
        relation=C.relation, split=split, channel=ch, model_id=args.model,
        model_revision=args.revision, prompt_template=C.render("<<SUBJECT>>", d),
        demo_ids=demo_ids(d), demo_seed=args.demo_seed, temperature=args.temperature,
        top_p=args.top_p, max_tokens=C.max_tokens, stop=C.stop,
        seed_base=args.seed_base, split_sha=split_sha(split))


def background_q(args, tol: float = 0.05) -> dict:
    """q(v) = mean over frames of (fraction of SUBJECTS whose pool contains a
    draw within tol of v). Built from draws only. Never touches gold."""
    per_frame: list[dict] = []
    for ch in Q_FRAMES:
        vals_by_subject = []
        for split in ("train", "val", "test"):
            try:
                pool = load_pool(spec_for(ch, split, args))
            except Exception:
                continue
            for s, draws in pool.items():
                v = numeric_candidates(draws, ch)
                if v:
                    vals_by_subject.append(v)
        if not vals_by_subject:
            continue
        universe = sorted({x for v in vals_by_subject for x in v})
        n = len(vals_by_subject)
        q = {}
        for cand in universe:
            hits = sum(1 for v in vals_by_subject
                       if any(abs(cand - w) / w <= tol for w in v))
            q[cand] = hits / n
        per_frame.append(q)
    if not per_frame:
        return {}
    keys = set().union(*[set(q) for q in per_frame])
    return {k: statistics.fmean([q.get(k, 0.0) for q in per_frame]) for k in keys}


def contended_values(args, ch: str) -> list[float]:
    """Distinct drawn values carrying >= 2 draws, pooled over splits. Draws only."""
    out = []
    for split in ("train", "val", "test"):
        try:
            pool = load_pool(spec_for(ch, split, args))
        except Exception:
            continue
        for draws in pool.values():
            v = numeric_candidates(draws, ch)
            c = defaultdict(int)
            for x in v:
                c[x] += 1
            out.extend([x for x, k in c.items() if k >= 2])
    return out


def predict_lift(draws: list[str], ch: str, q: dict, eps: float,
                 tol: float = 0.05) -> list[str]:
    v = numeric_candidates(draws, ch)
    if not v:
        return []
    best, best_score = None, -1.0
    for cand in sorted(set(v)):
        share = len(tolerance_support(v, cand, tol)) / len(v)
        # q for the candidate's own ball, nearest recorded key
        qv = q.get(cand)
        if qv is None:
            near = [k for k in q if abs(cand - k) / k <= tol] if q else []
            qv = max((q[k] for k in near), default=0.0)
        sc = share / (qv + eps)
        if sc > best_score:
            best, best_score = cand, sc
    return [format_number(best)] if best is not None else []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="cap_recite")
    ap.add_argument("--model", default="google/gemma-4-31B")
    ap.add_argument("--revision", default="5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--demo-seed", type=int, default=1234)
    ap.add_argument("--seed-base", type=int, default=7000)
    args = ap.parse_args()
    ch = args.channel

    print("STEP 1. Build q from draws only, three frames, three splits. No gold.")
    q = background_q(args)
    print(f"  q built over {len(q)} distinct values from {len(Q_FRAMES)} frames")

    print("\nSTEP 2. Fix eps by the PRE-COMMITTED rule, before any accuracy is computed.")
    cont = contended_values(args, ch)
    qs = []
    for x in cont:
        qv = q.get(x)
        if qv is None:
            near = [k for k in q if abs(x - k) / k <= 0.05]
            qv = max((q[k] for k in near), default=0.0)
        qs.append(qv)
    eps = statistics.median(qs) if qs else 0.02
    print(f"  contended values: {len(cont)}   eps := median q = {eps:.6f}")
    print("  eps is now FROZEN. Everything below is a single look.")

    print("\nSTEP 3. Score once on the clean rows (train minus demos, plus val).")
    from scorer import paired_bootstrap, score_one_relation
    C = CHANNELS[ch]
    dm = set(demo_ids(pick_demos(C.relation, C.n_demos, args.demo_seed, C.demo_strategy)))
    from aggregate import predict_numeric

    base_vec, lift_vec, nb, nl, ntot = [], [], 0, 0, 0
    for split in ("train", "val"):
        pool = load_pool(spec_for(ch, split, args))
        keep = set(pool) - dm if split == "train" else set(pool)
        b = score_one_relation({s: predict_numeric(d, ch) for s, d in pool.items()},
                               REL, split, subjects=keep)
        l = score_one_relation({s: predict_lift(d, ch, q, eps) for s, d in pool.items()},
                               REL, split, subjects=keep)
        base_vec += b["f1_vector"]; lift_vec += l["f1_vector"]
        nb += round(b["macro_f1"] * b["n_rows"]); nl += round(l["macro_f1"] * l["n_rows"])
        ntot += b["n_rows"]
        print(f"  {split:5s} n={b['n_rows']:3d}  baseline {b['macro_f1']:.4f}  "
              f"lift {l['macro_f1']:.4f}  ({round(b['macro_f1']*b['n_rows'])} -> "
              f"{round(l['macro_f1']*l['n_rows'])} rows)")

    bs = paired_bootstrap(base_vec, lift_vec)
    delta_rows = nl - nb
    print(f"\n  POOLED {ntot} rows: {nb} -> {nl}  ({delta_rows:+d} rows, "
          f"{bs['point']:+.4f})")
    print(f"  paired bootstrap 90% CI [{bs['ci_lo']:+.4f}, {bs['ci_hi']:+.4f}]  "
          f"rows up {bs['rows_up']} / down {bs['rows_down']}")

    print("\nSTEP 4. Apply the decision rule that was set before looking.")
    if delta_rows >= 5 and bs["ci_lo"] > -0.01:
        print("  SHIP: >= +5 rows and CI low > -0.01")
    elif 1 <= delta_rows <= 3:
        print("  DROP: +1 to +3 rows means the reported +0.0364 was a sweep peak")
    else:
        print(f"  NEITHER BRANCH: {delta_rows:+d} rows, CI low {bs['ci_lo']:+.4f}. "
              "Report honestly as a judgment call, do not quietly resolve it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
