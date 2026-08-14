"""Sweep a channel's decision scalar on TRAIN, then report the chosen value on VAL.

Discipline: thresholds are chosen on TRAIN. val is read once per candidate as a
selection set and reported honestly as such. The official evaluate.py is the
only scorer; I never reimplement the metric.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate import parse_qa, parse_qa_ok, predict_numeric, predict_set
from channels import CHANNELS, demo_ids, pick_demos
from common import NUMERIC_RELATIONS, PoolSpec, REPO, load_pool, rows_for, split_sha
from scorer import score_one_relation


def score_relation(preds_by_subject: dict[str, list[str]], relation: str,
                   split: str, python: str | None = None,
                   subjects: set[str] | None = None) -> dict:
    """Official scorer, called by import (see scorer.py for why not stdout)."""
    return score_one_relation(preds_by_subject, relation, split, subjects=subjects)


def build_spec(ch, split: str, args) -> PoolSpec:
    from common import spec_for_channel
    return spec_for_channel(ch, split, args)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", required=True, choices=sorted(CHANNELS))
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--model", default="google/gemma-4-31B")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--demo-seed", type=int, default=1234)
    ap.add_argument("--seed-base", type=int, default=7000)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--splits", default="train,val")
    args = ap.parse_args()

    ch = CHANNELS[args.channel]
    numeric = ch.relation in NUMERIC_RELATIONS
    splits = args.splits.split(",")
    report: dict = {"channel": ch.name, "relation": ch.relation}

    pools = {}
    for sp in splits:
        spec = build_spec(ch, sp, args)
        pools[sp] = load_pool(spec, args.n or None)
        report[f"pool_id_{sp}"] = spec.pool_id()

    # --- parse QA gate: a broken decode contract invalidates every tau below.
    qa = parse_qa(pools[splits[0]], ch.name)
    ok, why = parse_qa_ok(qa)
    report["parse_qa"] = qa
    report["parse_qa_ok"] = ok
    report["parse_qa_reason"] = why
    print(json.dumps({k: v for k, v in qa.items() if k != "top_rejected"}, indent=1))
    if not ok:
        print(f"PARSE-QA FAIL: {why}")
        print("top rejected continuations:")
        for s, c in qa["top_rejected"][:10]:
            print(f"   {c:4d}  {s!r}")

    grid = ([round(x / 100, 2) for x in range(90, 111, 5)] if numeric
            else [round(x / 100, 2) for x in range(10, 75, 5)])
    report["grid"] = grid

    # LEAKAGE GUARD. Demos are drawn from train and are fixed per channel, so a
    # train subject that is also a demo has its own gold answer sitting in its
    # prompt. Scoring the train curve on those rows inflates it and biases the
    # chosen threshold toward overconfidence. Exclude them. (val and test are
    # unaffected: no val/test row is ever a demo.)
    demo_set = set(demo_ids(pick_demos(ch.relation, ch.n_demos, args.demo_seed,
                                       ch.demo_strategy)))
    report["n_demos"] = len(demo_set)

    curves = {}
    for sp in splits:
        pool = pools[sp]
        if sp == "train":
            scored = {s: d for s, d in pool.items() if s not in demo_set}
            report["train_subjects_total"] = len(pool)
            report["train_subjects_scored"] = len(scored)
            report["train_subjects_excluded_as_demos"] = len(pool) - len(scored)
            print(f"  [train] leakage guard: scoring {len(scored)}/{len(pool)} subjects "
                  f"({len(pool)-len(scored)} excluded as demos)")
            if not scored:
                # Degenerate and worth naming rather than crashing. A channel
                # whose demo count equals the train split size consumes the
                # ENTIRE train set as demonstrations and has no held-out train
                # rows at all. Observed with area_lead100: 100 demos against 100
                # train rows. Any train figure quoted for such a channel would be
                # memorisation of its own prompt, which is also worth noting about
                # the published recipe that specifies 100 demos on this relation.
                print(f"  [train] SKIPPED: all {len(pool)} train subjects are demos "
                      f"of this channel ({ch.n_demos} demos vs {len(pool)} train "
                      f"rows). No held-out train data exists. Parameter can only "
                      f"be chosen on val, and this channel is UNVALIDATABLE on train.")
                report["train_unusable_all_demos"] = True
                continue
            if len(scored) < 15:
                print(f"  [train] WARNING only {len(scored)} non-demo subjects; "
                      f"threshold choice here is low-power. Consider fewer demos.")
        else:
            scored = pool
        keep = set(scored) if sp == "train" else None
        curve = []
        for g in grid:
            if numeric:
                preds = {s: predict_numeric(d, ch.name, scale=g) for s, d in scored.items()}
            else:
                preds = {s: predict_set(d, ch.name, tau=g) for s, d in scored.items()}
            r = score_relation(preds, ch.relation, sp, args.python, subjects=keep)
            curve.append({"param": g, **{k: v for k, v in r.items() if k != "f1_vector"}})
            print(f"  [{sp}] param={g:.2f}  P={r['macro_p']:.4f} R={r['macro_r']:.4f} "
                  f"F1={r['macro_f1']:.4f} n={r['n_rows']}", flush=True)
        curves[sp] = curve
    report["curves"] = curves

    # PLATEAU-CENTRE SELECTION (pre-registered in the plan's acceptance rule).
    # The train curve is often flat across a run of thresholds, especially after
    # the leakage guard shrinks n. A plain argmax then returns whichever tied
    # point comes first in the grid, which is an edge of the plateau and is
    # arbitrary. Prefer the CENTRE of the widest tied run: it is the choice most
    # robust to the gold edits the organizers have already made once, and to
    # val/test prior drift.
    tol = 1e-9
    top = max(c["macro_f1"] for c in curves["train"])
    tied_idx = [i for i, c in enumerate(curves["train"]) if c["macro_f1"] >= top - tol]
    runs, run = [], [tied_idx[0]]
    for i in tied_idx[1:]:
        if i == run[-1] + 1:
            run.append(i)
        else:
            runs.append(run)
            run = [i]
    runs.append(run)
    widest = max(runs, key=len)
    best_train = curves["train"][widest[len(widest) // 2]]
    report["best_on_train"] = best_train
    report["plateau"] = {
        "tied_params": [curves["train"][i]["param"] for i in widest],
        "width": len(widest),
        "chosen": best_train["param"],
        "note": "centre of the widest tied run on train; plain argmax would have "
                f"chosen {curves['train'][tied_idx[0]]['param']}",
    }
    if len(widest) > 1:
        print(f"  [train] plateau of {len(widest)} tied params "
              f"{report['plateau']['tied_params']} -> centre {best_train['param']} "
              f"(plain argmax would take {curves['train'][tied_idx[0]]['param']})")
    if "val" in curves:
        chosen = next(c for c in curves["val"] if c["param"] == best_train["param"])
        report["val_at_train_argmax"] = chosen
        report["val_argmax_for_reference"] = max(curves["val"], key=lambda c: c["macro_f1"])
        print(f"\nCHOSEN param={best_train['param']} (train argmax)"
              f"  train F1={best_train['macro_f1']:.4f}  val F1={chosen['macro_f1']:.4f}")

    out = REPO / "pools" / f"tune_{ch.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
