"""Re-centre abstention thresholds onto the MEASURED test empty-gold priors.

Probe #1 (a single all-empty submission) returned each relation's test empty
fraction exactly, because an all-empty file scores per-relation macro-F1 equal
to that fraction. Those priors differ from validation by a lot, and in both
directions:

    personHasCityOfDeath   val 39.0%  ->  test 48.0%   (abstain MORE)
    companyTradesAtStock   val 38.0%  ->  test 44.0%   (abstain MORE)
    countryLandBorders     val 26.5%  ->  test 14.9%   (abstain LESS)

A threshold tuned on validation is therefore mis-centred on the split that
decides the ranking. This module estimates, for each candidate threshold, what
the score would be on a population with the TEST empty rate, by importance
weighting validation rows:

    w_empty    = p_test_empty / p_val_empty
    w_nonempty = (1 - p_test_empty) / (1 - p_val_empty)

The assumption is explicit and worth stating: difficulty WITHIN the empty group
and within the non-empty group is the same across splits, and only their mixing
proportion changes. That is weaker than assuming the splits are identical, which
is what tuning on raw validation silently assumes.

This is a legitimate use of aggregate feedback. It re-centres a scalar decision
parameter using a published per-relation aggregate. It does not use row-level
information and cannot recover which specific rows are correct.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate import predict_set
from channels import CHANNELS, demo_ids, pick_demos
from common import (PoolSpec, REPO, TEST_EMPTY_PRIOR, TEST_ROWS, load_pool,
                    rows_for, split_sha)
from scorer import per_row_scores

STRING_RELATIONS = ("personHasCityOfDeath", "companyTradesAtStockExchange",
                    "countryLandBordersCountry")


def val_empty_fraction(relation: str) -> float:
    rows = rows_for("val", relation)
    return sum(1 for r in rows if not r["ObjectEntities"]) / len(rows)


def weighted_f1(preds: dict[str, list[str]], relation: str,
                p_target: float) -> tuple[float, float]:
    """Return (native val F1, F1 reweighted to the target empty prior)."""
    pred_rows = [{"SubjectEntity": s, "Relation": relation,
                  "ObjectEntities": [str(o) for o in v]} for s, v in preds.items()]
    rows = [r for r in per_row_scores(pred_rows, "val") if r["Relation"] == relation]
    gold_empty = {r["SubjectEntity"]: (r["total_gt"] == 0) for r in rows}

    p_val = sum(gold_empty.values()) / len(rows)
    native = sum(r["f1"] for r in rows) / len(rows)
    if p_val in (0.0, 1.0):
        return native, native

    w_e = p_target / p_val
    w_n = (1 - p_target) / (1 - p_val)
    num = den = 0.0
    for r in rows:
        w = w_e if gold_empty[r["SubjectEntity"]] else w_n
        num += w * r["f1"]
        den += w
    return native, num / den


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", default="death_obituary,stock_sentinel,borders_list")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--model", default="google/gemma-4-31B")
    ap.add_argument("--revision", default="5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--demo-seed", type=int, default=1234)
    ap.add_argument("--seed-base", type=int, default=7000)
    args = ap.parse_args()

    out: dict = {}
    for name in [c.strip() for c in args.channels.split(",") if c.strip()]:
        ch = CHANNELS[name]
        if ch.relation not in STRING_RELATIONS:
            print(f"skip {name}: not an abstention relation")
            continue
        demos = pick_demos(ch.relation, ch.n_demos, args.demo_seed, ch.demo_strategy)
        spec = PoolSpec(
            relation=ch.relation, split="val", channel=ch.name,
            model_id=args.model, model_revision=args.revision,
            prompt_template=ch.render("<<SUBJECT>>", demos),
            demo_ids=demo_ids(demos), demo_seed=args.demo_seed,
            temperature=args.temperature, top_p=args.top_p,
            max_tokens=ch.max_tokens, stop=ch.stop, seed_base=args.seed_base,
            split_sha=split_sha("val"),
        )
        try:
            pool = load_pool(spec, args.n or None)
        except Exception as exc:
            print(f"skip {name}: {type(exc).__name__}: {exc}")
            continue

        p_test = TEST_EMPTY_PRIOR[ch.relation]
        p_val = val_empty_fraction(ch.relation)
        w = TEST_ROWS[ch.relation]
        print(f"\n=== {name} / {ch.relation} ===")
        print(f"  val empty {p_val:.3f} -> test empty {p_test:.3f} "
              f"({'abstain MORE' if p_test > p_val else 'abstain LESS'} on test)")
        print(f"  {'tau':>5} {'val F1':>9} {'test-prior F1':>14}")

        rows = []
        for tau in [round(x / 100, 2) for x in range(5, 80, 5)]:
            preds = {s: predict_set(d, name, tau=tau) for s, d in pool.items()}
            native, shifted = weighted_f1(preds, ch.relation, p_test)
            rows.append({"tau": tau, "val_f1": native, "test_prior_f1": shifted})
            print(f"  {tau:>5.2f} {native:>9.4f} {shifted:>14.4f}")

        best_val = max(rows, key=lambda r: r["val_f1"])
        best_shift = max(rows, key=lambda r: r["test_prior_f1"])
        delta_overall = (best_shift["test_prior_f1"] - best_val["test_prior_f1"]) * w / 475
        print(f"  val-optimal tau      {best_val['tau']:.2f}  "
              f"(val {best_val['val_f1']:.4f}, test-prior {best_val['test_prior_f1']:.4f})")
        print(f"  TEST-PRIOR tau       {best_shift['tau']:.2f}  "
              f"(val {best_shift['val_f1']:.4f}, test-prior {best_shift['test_prior_f1']:.4f})")
        print(f"  expected overall gain from re-centring: {delta_overall:+.5f}")
        if best_shift["val_f1"] < best_val["val_f1"]:
            print(f"  NOTE this is a TEST-PRIOR TRADE: it LOWERS val by "
                  f"{best_val['val_f1']-best_shift['val_f1']:.4f} on purpose. "
                  f"Log it as such and confirm with a test probe before shipping.")
        out[name] = {"relation": ch.relation, "p_val": p_val, "p_test": p_test,
                     "curve": rows, "val_optimal": best_val,
                     "test_prior_optimal": best_shift,
                     "expected_overall_gain": delta_overall}

    total = sum(v["expected_overall_gain"] for v in out.values())
    print(f"\nTOTAL expected overall gain from re-centring all three: {total:+.5f}")
    p = REPO / "pools" / "prior_shift.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
