"""Cross-frame consensus experiment for hasCapacity (and any numeric relation).

Compares, on the same subjects and the same cached draws:
  1. each frame alone, at its own best overshoot scale
  2. the union of all frames treated as one flat pool (breadth with no credit
     for WHERE a value came from) - the honest ablation
  3. cross-frame consensus, which credits a value for surviving a change of
     register on top of its frequency within a register

If (3) does not beat (1) and (2) by more than the measurement floor, the idea is
dead and I say so in the log rather than shipping it.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate import predict_numeric, predict_numeric_consensus
from channels import CHANNELS, demo_ids, pick_demos
from common import PoolSpec, REPO, rows_for, split_sha
from scorer import paired_bootstrap, score_one_relation


def spec_for(ch, split, args) -> PoolSpec:
    from common import spec_for_channel
    return spec_for_channel(ch, split, args)


def load_all(channels, split, args):
    from common import load_pool
    out = {}
    for nm in channels:
        try:
            out[nm] = load_pool(spec_for(CHANNELS[nm], split, args), args.n or None)
        except Exception as exc:
            print(f"  skip {nm}: {type(exc).__name__}: {exc}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", default="cap_infobox,cap_disambig,cap_current,cap_listing,cap_recite")
    ap.add_argument("--relation", default="hasCapacity")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--model", default="google/gemma-4-31B")
    ap.add_argument("--revision", default="5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--demo-seed", type=int, default=1234)
    ap.add_argument("--seed-base", type=int, default=7000)
    args = ap.parse_args()

    names = [c.strip() for c in args.channels.split(",") if c.strip()]
    report: dict = {"relation": args.relation, "channels": names}

    for split in [s.strip() for s in args.splits.split(",")]:
        print(f"\n{'='*74}\nSPLIT {split}\n{'='*74}")
        pools = load_all(names, split, args)
        if not pools:
            print("no pools available")
            continue
        subjects = [r["SubjectEntity"] for r in rows_for(split, args.relation)]
        res: dict = {}

        # 1. each frame alone, swept over overshoot scale
        print("\n-- single frame --")
        singles = {}
        for nm, pool in pools.items():
            best = None
            for sc in (0.90, 0.95, 1.00, 1.05):
                preds = {s: predict_numeric(pool.get(s, []), nm, scale=sc) for s in subjects}
                r = score_one_relation(preds, args.relation, split)
                if best is None or r["macro_f1"] > best["macro_f1"]:
                    best = {**r, "scale": sc}
            singles[nm] = best
            print(f"   {nm:14s} best_scale={best['scale']:.2f}  F1={best['macro_f1']:.4f} "
                  f"({round(best['macro_f1']*best['n_rows'])}/{best['n_rows']} rows)")
        res["single"] = {k: {"macro_f1": v["macro_f1"], "scale": v["scale"]} for k, v in singles.items()}
        best_single_name = max(singles, key=lambda k: singles[k]["macro_f1"])
        best_single = singles[best_single_name]

        # 2. flat union ablation: all draws in one bag, no channel identity
        print("\n-- flat union (breadth, no channel credit) --")
        flat_best = None
        for sc in (0.90, 0.95, 1.00, 1.05):
            preds = {}
            for s in subjects:
                merged = []
                for nm, pool in pools.items():
                    merged.extend(pool.get(s, []))
                preds[s] = predict_numeric(merged, best_single_name, scale=sc) if merged else []
            r = score_one_relation(preds, args.relation, split)
            if flat_best is None or r["macro_f1"] > flat_best["macro_f1"]:
                flat_best = {**r, "scale": sc}
        print(f"   flat_union     best_scale={flat_best['scale']:.2f}  F1={flat_best['macro_f1']:.4f}")
        res["flat_union"] = {"macro_f1": flat_best["macro_f1"], "scale": flat_best["scale"]}

        # 3. cross-frame consensus
        print("\n-- cross-frame consensus --")
        cons_best = None
        for w, sc in itertools.product((0.0, 0.5, 1.0, 2.0), (0.90, 0.95, 1.00, 1.05)):
            preds = {}
            for s in subjects:
                per_ch = {nm: pool.get(s, []) for nm, pool in pools.items()}
                p, _ = predict_numeric_consensus(per_ch, channel_agreement_weight=w, scale=sc)
                preds[s] = p
            r = score_one_relation(preds, args.relation, split)
            if cons_best is None or r["macro_f1"] > cons_best["macro_f1"]:
                cons_best = {**r, "w": w, "scale": sc}
            print(f"   w={w:.1f} scale={sc:.2f}  F1={r['macro_f1']:.4f}")
        print(f"   BEST consensus w={cons_best['w']} scale={cons_best['scale']} "
              f"F1={cons_best['macro_f1']:.4f}")
        res["consensus"] = {"macro_f1": cons_best["macro_f1"], "w": cons_best["w"],
                            "scale": cons_best["scale"]}

        # verdict with a paired bootstrap against both baselines
        for label, base in (("vs_best_single", best_single), ("vs_flat_union", flat_best)):
            bs = paired_bootstrap(base["f1_vector"], cons_best["f1_vector"])
            res[label] = {k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in bs.items()}
            print(f"\n   consensus {label}: delta={bs['point']:+.4f} "
                  f"90% CI [{bs['ci_lo']:+.4f}, {bs['ci_hi']:+.4f}] "
                  f"rows +{bs['rows_up']}/-{bs['rows_down']}  "
                  f"{'KEEP' if bs['excludes_zero_above'] else 'not separated'}")
        # overall-score translation: capacity carries 98/475
        d = cons_best["macro_f1"] - best_single["macro_f1"]
        print(f"\n   overall-score impact if it holds on test: {d*98/475:+.4f}")
        res["overall_impact_estimate"] = round(d * 98 / 475, 5)
        report[split] = res

    out = REPO / "pools" / f"consensus_{args.relation}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
