"""Oracle-ceiling analysis: is a relation KNOWLEDGE-bound or SELECTION-bound?

This is the instrument the whole hasCapacity push depends on. The live #1
reported that on capacity their pool contained the gold for 76-77/100 subjects
while their selector realized only 35-38, then exhausted 24+ aggregation-side
mechanisms trying to close it. That gap is the campaign.

For each subject I ask two separate questions:
  COVERAGE  - does ANY draw in the pool contain an acceptable answer?
              (the ceiling any selector could reach)
  REALIZED  - does the current aggregator actually output one?
              (what I score today)

coverage - realized = how much is recoverable by better SELECTION.
1 - coverage        = how much needs better GENERATION (or is unknowable).

Reading the split matters. If coverage is low, more clever voting is wasted
effort and the frame itself has to change.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate import normalize, predict_numeric, predict_set, vote_shares
from channels import CHANNELS, demo_ids, pick_demos
from common import (NUMERIC_RELATIONS, PoolSpec, REPO, gold_aliases,
                    gold_primary, load_pool, rows_for, split_sha)


def numeric_hit(pred: float, gold: float, tol: float = 0.05) -> bool:
    return gold != 0 and abs(pred - gold) / abs(gold) <= tol


def analyse(pool: dict[str, list[str]], relation: str, channel: str,
            split: str, param: float) -> dict:
    ch = CHANNELS[channel]
    numeric = relation in NUMERIC_RELATIONS
    rows = {r["SubjectEntity"]: r for r in rows_for(split, relation)}

    n = cov = real = both = neither = 0
    cov_not_real: list[dict] = []
    rank_hist: dict[str, int] = {}

    for subj, draws in pool.items():
        row = rows.get(subj)
        if row is None:
            continue
        n += 1

        if numeric:
            golds = [float(g) for g in gold_primary(row)]
            cands: list[float] = []
            for d in draws:
                for c in ch.parse(d):
                    try:
                        cands.append(float(c))
                    except ValueError:
                        pass
            covered = any(numeric_hit(c, golds[0]) for c in cands) if golds else not cands
            got = predict_numeric(draws, channel, scale=param)
            realized = (numeric_hit(float(got[0]), golds[0])
                        if got and golds else (not got and not golds))
            # where does a correct value sit in the frequency ranking?
            if covered and golds:
                freq: dict[str, int] = {}
                for c in cands:
                    freq[str(int(round(c)))] = freq.get(str(int(round(c))), 0) + 1
                order = sorted(freq, key=lambda k: -freq[k])
                pos = next((i + 1 for i, k in enumerate(order)
                            if numeric_hit(float(k), golds[0])), None)
                if pos:
                    b = "1" if pos == 1 else "2-3" if pos <= 3 else "4-10" if pos <= 10 else "11+"
                    rank_hist[b] = rank_hist.get(b, 0) + 1
        else:
            # Set relations. COVERED means every gold object appears somewhere
            # in the pool under at least one of its aliases, i.e. a perfect
            # selector reading only this pool could score the row 1.0.
            # REALIZED means the aggregator actually emitted exactly that set.
            # Both are exact-match, deliberately stricter than the graded F1,
            # because the question here is "is the answer retrievable at all",
            # not "how much partial credit did I get".
            alias_sets = [{normalize(a) for a in al} for al in gold_aliases(row)]
            present = set(vote_shares(draws, channel))
            got = {normalize(x) for x in predict_set(draws, channel, tau=param)}
            if alias_sets:
                covered = all(bool(al & present) for al in alias_sets)
                realized = (all(bool(al & got) for al in alias_sets)
                            and len(got) == len(alias_sets))
            else:
                # empty gold: covered means abstention is reachable, realized
                # means I actually abstained
                covered = True
                realized = not got

        cov += covered
        real += realized
        if covered and realized:
            both += 1
        elif covered and not realized:
            cov_not_real.append({
                "subject": subj,
                "gold": gold_primary(row),
                "predicted": predict_numeric(draws, channel, scale=param) if numeric
                else predict_set(draws, channel, tau=param),
            })
        elif not covered and not realized:
            neither += 1

    n = max(n, 1)
    return {
        "relation": relation, "channel": channel, "split": split, "param": param,
        "n_subjects": n,
        "coverage": round(cov / n, 4),
        "realized": round(real / n, 4),
        "recoverable_by_selection": round((cov - real) / n, 4),
        "unreachable_without_new_generation": round((n - cov) / n, 4),
        "gold_rank_in_pool": rank_hist,
        "examples_covered_but_missed": cov_not_real[:25],
        "verdict": ("SELECTION-BOUND" if (cov - real) / n > (n - cov) / n
                    else "KNOWLEDGE-BOUND"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", required=True, choices=sorted(CHANNELS))
    ap.add_argument("--split", default="val")
    ap.add_argument("--param", type=float, default=1.0,
                    help="tau for set relations, overshoot scale for numerics")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--model", default="google/gemma-4-31B")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--demo-seed", type=int, default=1234)
    ap.add_argument("--seed-base", type=int, default=7000)
    args = ap.parse_args()

    ch = CHANNELS[args.channel]
    from common import spec_for_channel
    spec = spec_for_channel(ch, args.split, args)
    pool = load_pool(spec, args.n or None)
    rep = analyse(pool, ch.relation, ch.name, args.split, args.param)

    print(json.dumps({k: v for k, v in rep.items()
                      if k != "examples_covered_but_missed"}, indent=1))
    print("\ncovered but missed by the selector (first 15):")
    for e in rep["examples_covered_but_missed"][:15]:
        print(f"   {e['subject'][:52]:54s} gold={e['gold']} got={e['predicted']}")

    out = REPO / "pools" / f"coverage_{ch.name}_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(rep, fh, indent=1)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
