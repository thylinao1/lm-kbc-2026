"""Honest evaluation of the learned reranker against the shipped selector.

The only comparison that means anything is row-level and out-of-fold: for each subject, the
reranker picks the candidate with the highest predicted probability, and that pick is scored
by the grader's own 5% rule. Folds are grouped BY SUBJECT, so no subject contributes to the
model that ranks it. The baseline on the same rows is the shipped predict_numeric.

Three guards against the failure this campaign has already lived through, where a lever
reported +0.0364 and measured +0.0000:
  * no free parameter is swept against the score. Model hyperparameters are fixed at
    deliberately dull values before the first run and are not touched afterwards.
  * the paired bootstrap is over subjects, which is the unit the metric averages over.
  * a SHUFFLED-LABEL control runs alongside. If the real model does not clearly beat its own
    label-permuted twin, the apparent gain is fold noise and the lever is dead.
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate import predict_numeric
from channels import CHANNELS, demo_ids, pick_demos
from common import load_pool, rows_for, spec_for_channel
from rerank_features import (AREA_FRAMES, CAP_FRAMES, FEATURES, TOL, build,
                             candidate_rows, load_frames)

SEED = 20260812
N_FOLDS = 5


def grouped_folds(groups: list[str], k: int = N_FOLDS) -> list[list[int]]:
    uniq = sorted(set(groups))
    rng = random.Random(SEED)
    rng.shuffle(uniq)
    assign = {g: i % k for i, g in enumerate(uniq)}
    folds = [[] for _ in range(k)]
    for i, g in enumerate(groups):
        folds[assign[g]].append(i)
    return folds


def fit_predict(Xtr, ytr, Xte, kind: str):
    """Fixed hyperparameters, chosen before the first run and never tuned."""
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(C=0.3, max_iter=2000,
                                             class_weight="balanced"))
        m.fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1]
    import lightgbm as lgb
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=7,
                           min_child_samples=40, subsample=0.8, colsample_bytree=0.7,
                           reg_lambda=5.0, random_state=SEED, verbose=-1)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def row_scores_from_scores(scores, groups, values, y):
    """Argmax per subject -> 1.0 if that pick is a labelled hit, else 0.0."""
    best: dict[str, tuple[float, int]] = {}
    for i, g in enumerate(groups):
        if g not in best or scores[i] > best[g][0]:
            best[g] = (scores[i], i)
    return {g: float(y[i]) for g, (_, i) in best.items()}


def baseline_rows(relation: str, primary: str, ns, exclude_demos=True) -> dict[str, float]:
    """Shipped selector, scored by the same 5% rule, keyed the same way."""
    C = CHANNELS[primary]
    demos = set(demo_ids(pick_demos(C.relation, C.n_demos, ns.demo_seed, C.demo_strategy)))
    out: dict[str, float] = {}
    for sp in ("train", "val"):
        pool = load_pool(spec_for_channel(C, sp, ns))
        gold_by = {r["SubjectEntity"]: r for r in rows_for(sp, relation)}
        for s, draws in pool.items():
            if s not in gold_by:
                continue
            if sp == "train" and exclude_demos and s in demos:
                continue
            g = gold_by[s].get("ObjectEntities") or []
            gv = float(g[0][0]) if g and g[0] else 0.0
            p = predict_numeric(draws, primary)
            hit = bool(p) and gv and abs(float(p[0]) - gv) / gv <= TOL
            out[f"{sp}:{s}"] = 1.0 if hit else 0.0
    return out


def paired_bootstrap(a: dict, b: dict, n: int = 5000):
    keys = sorted(set(a) & set(b))
    d = [b[k] - a[k] for k in keys]
    rng = random.Random(SEED)
    means = []
    for _ in range(n):
        means.append(statistics.fmean(rng.choices(d, k=len(d))))
    means.sort()
    return {"point": statistics.fmean(d), "n": len(d),
            "ci_lo": means[int(0.05 * n)], "ci_hi": means[int(0.95 * n)],
            "up": sum(1 for x in d if x > 0), "down": sum(1 for x in d if x < 0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--relation", default="hasCapacity")
    ap.add_argument("--primary", default="cap_recite")
    ap.add_argument("--model", default="google/gemma-4-31B")
    ap.add_argument("--revision", default="5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--demo-seed", type=int, default=1234)
    ap.add_argument("--seed-base", type=int, default=7000)
    ap.add_argument("--kinds", default="logreg,lgbm")
    args = ap.parse_args()

    frames = CAP_FRAMES if args.relation == "hasCapacity" else AREA_FRAMES
    print(f"Building features: {args.relation} / primary={args.primary}")
    X, y, groups, values, split_of = build(args.relation, args.primary, frames, ns=args)
    n_subj = len(set(groups))
    print(f"  {len(X)} candidates over {n_subj} subjects, "
          f"{sum(y)} positive ({sum(y)/max(len(y),1):.3f})")
    print(f"  candidates per subject: {len(X)/max(n_subj,1):.1f}")

    base = baseline_rows(args.relation, args.primary, args)
    common = sorted(set(base) & set(groups))
    print(f"  shipped selector on the same {len(set(groups) & set(base))} subjects: "
          f"{statistics.fmean([base[g] for g in sorted(set(groups) & set(base))]):.4f}")

    folds = grouped_folds(groups)
    for kind in args.kinds.split(","):
        for shuffled in (False, True):
            oof = [0.0] * len(X)
            for f in range(N_FOLDS):
                te = set(folds[f])
                tr = [i for i in range(len(X)) if i not in te]
                tei = sorted(te)
                ytr = [y[i] for i in tr]
                if shuffled:                       # permute WITHIN the training fold only
                    rng = random.Random(SEED + f)
                    ytr = ytr[:]
                    rng.shuffle(ytr)
                p = fit_predict([X[i] for i in tr], ytr, [X[i] for i in tei], kind)
                for j, i in enumerate(tei):
                    oof[i] = p[j]
            rows = row_scores_from_scores(oof, groups, values, y)
            keys = sorted(set(rows) & set(base))
            m = statistics.fmean([rows[k] for k in keys])
            bs = paired_bootstrap({k: base[k] for k in keys},
                                  {k: rows[k] for k in keys})
            tag = "SHUFFLED-CONTROL" if shuffled else "real"
            print(f"\n  {kind:7s} {tag:17s} out-of-fold row accuracy = {m:.4f}")
            print(f"          vs shipped {bs['point']:+.4f}  90% CI "
                  f"[{bs['ci_lo']:+.4f}, {bs['ci_hi']:+.4f}]  "
                  f"rows up {bs['up']} / down {bs['down']} of {bs['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
