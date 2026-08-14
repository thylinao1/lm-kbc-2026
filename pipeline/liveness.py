"""Combine a liveness signal with the city channel on personHasCityOfDeath.

Why this exists. The validation error breakdown at tau=0.35 splits 100 subjects
into three failure families, and they need different fixes:

    33  correct answer
    27  correct abstention
    12  FALSE ANSWER: the person is alive and we named a city
    13  answer was in the pool but the selector missed it
    15  answer genuinely absent from the pool

The 12 false answers are the largest addressable block and they are not a
selection problem: no improvement to city selection can fix a row whose correct
output is silence. They need evidence about whether the person is dead, which
the city channel does not contain.

The rule tested here is deliberately NOT a veto. Prior work reports that a
standalone yes/no gate was inert on this relation. Instead the liveness vote
share is combined with the city vote share: we answer only if the city clears
its own threshold AND the liveness channel is at least tau_live confident the
person is deceased. tau_live = 0 recovers the current system exactly, so the
comparison is nested and honest.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate import predict_set
from channels import CHANNELS, demo_ids, pick_demos
from common import PoolSpec, REPO, TEST_EMPTY_PRIOR, load_pool, split_sha
from scorer import paired_bootstrap, score_one_relation

REL = "personHasCityOfDeath"


def spec_for(ch, split, args):
    demos = pick_demos(ch.relation, ch.n_demos, args.demo_seed, ch.demo_strategy)
    return PoolSpec(
        relation=ch.relation, split=split, channel=ch.name,
        model_id=args.model, model_revision=args.revision,
        prompt_template=ch.render("<<SUBJECT>>", demos),
        demo_ids=demo_ids(demos), demo_seed=args.demo_seed,
        temperature=args.temperature, top_p=args.top_p,
        max_tokens=ch.max_tokens, stop=ch.stop, seed_base=args.seed_base,
        split_sha=split_sha(split),
    )


def deceased_share(draws: list[str]) -> float:
    """Fraction of draws asserting the person is dead."""
    if not draws:
        return 0.0
    ch = CHANNELS["death_year"]
    return sum(1 for d in draws if ch.parse(d)) / len(draws)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city-channel", default="death_obituary")
    ap.add_argument("--tau-city", type=float, default=0.35)
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--model", default="google/gemma-4-31B")
    ap.add_argument("--revision", default="5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--demo-seed", type=int, default=1234)
    ap.add_argument("--seed-base", type=int, default=7000)
    args = ap.parse_args()

    report: dict = {"city_channel": args.city_channel, "tau_city": args.tau_city}
    for split in [s.strip() for s in args.splits.split(",")]:
        try:
            city = load_pool(spec_for(CHANNELS[args.city_channel], split, args), args.n or None)
            live = load_pool(spec_for(CHANNELS["death_year"], split, args), args.n or None)
        except Exception as exc:
            print(f"skip {split}: {type(exc).__name__}: {exc}")
            continue

        # Leakage guard: demos are drawn from train and are fixed, so a train
        # subject that is also a demo carries its own answer in its prompt. Both
        # channels draw demos from the same relation, so exclude the union.
        keep = None
        if split == "train":
            dm = set()
            for nm in (args.city_channel, "death_year"):
                c = CHANNELS[nm]
                dm |= set(demo_ids(pick_demos(c.relation, c.n_demos, args.demo_seed,
                                              c.demo_strategy)))
            keep = set(city) - dm
            print(f"  [train] leakage guard: scoring {len(keep)}/{len(city)} "
                  f"({len(city)-len(keep)} excluded as demos of either channel)")

        base = {s: predict_set(d, args.city_channel, args.tau_city) for s, d in city.items()}
        base_r = score_one_relation(base, REL, split, subjects=keep)
        print(f"\n=== {split} ===")
        _sc = keep if keep is not None else set(base)
        print(f"  baseline (city only, tau={args.tau_city}): F1={base_r['macro_f1']:.4f}  "
              f"abstain={sum(1 for s2, v in base.items() if s2 in _sc and not v)/len(_sc):.1%}  n={len(_sc)}")
        print(f"  {'tau_live':>9} {'F1':>8} {'delta':>8} {'abstain':>9}  {'CI':>20}")

        rows = []
        for tl in [round(x / 100, 2) for x in range(0, 105, 10)]:
            preds = {s: (base[s] if deceased_share(live.get(s, [])) >= tl else [])
                     for s in base}
            r = score_one_relation(preds, REL, split, subjects=keep)
            bs = paired_bootstrap(base_r["f1_vector"], r["f1_vector"])
            scored = keep if keep is not None else set(preds)
            ab = sum(1 for s2, v in preds.items() if s2 in scored and not v) / len(scored)
            rows.append({"tau_live": tl, "macro_f1": r["macro_f1"],
                         "delta": bs["point"], "ci_lo": bs["ci_lo"],
                         "ci_hi": bs["ci_hi"], "abstain": ab})
            print(f"  {tl:>9.2f} {r['macro_f1']:>8.4f} {bs['point']:>+8.4f} "
                  f"{ab:>8.1%}  [{bs['ci_lo']:+.3f},{bs['ci_hi']:+.3f}]"
                  f"{'  KEEP' if bs['excludes_zero_above'] else ''}")
        report[split] = {"baseline": base_r["macro_f1"], "curve": rows}
        if split == "test":
            t = TEST_EMPTY_PRIOR[REL]
            closest = min(rows, key=lambda r: abs(r["abstain"] - t))
            print(f"  closest to the measured test prior ({t:.1%}): "
                  f"tau_live={closest['tau_live']} at {closest['abstain']:.1%}")

    out = REPO / "pools" / "liveness.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
