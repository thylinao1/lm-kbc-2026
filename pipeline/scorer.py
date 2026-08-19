"""Thin wrapper around the OFFICIAL dataset2026/evaluate.py.

I import the organizers' module and call its own functions rather than parsing
its printed table. Two reasons that matter:
  * the printed table is pandas-formatted and rounds to 3 dp, which blurs the
    0.01-0.03 tau deltas this campaign is made of (and truncates columns with
    "..." so positional parsing is silently wrong);
  * evaluate_per_sr_pair() returns the PER-ROW f1 vector, which is the noise
    unit for the paired bootstrap.

The metric is never reimplemented here.
"""
from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

from common import DATA_DIR, REPO, load_split

EVAL_PATH = REPO / "dataset2026" / "evaluate.py"


@lru_cache(maxsize=1)
def _ev():
    """Load the official scorer unmodified.

    evaluate.py does `import pandas as pd` at module level but uses pandas ONLY
    inside main() (lines 342-351) to render the printed table. The scoring
    functions I call are pure Python. On cluster envs without pandas I insert
    a stub module so the top-level import resolves; it can never be exercised,
    because I never call main(). The scorer's arithmetic is untouched.
    """
    try:
        import pandas  # noqa: F401
    except ModuleNotFoundError:
        import types
        stub = types.ModuleType("pandas")

        def _unavailable(*_a, **_k):  # pragma: no cover - guard only
            raise RuntimeError(
                "pandas stub called: evaluate.py's display path is not available "
                "in this environment. Use scorer.score_all() instead of main()."
            )

        stub.DataFrame = _unavailable
        stub.concat = _unavailable
        sys.modules["pandas"] = stub

    spec = importlib.util.spec_from_file_location("official_evaluate", EVAL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["official_evaluate"] = mod
    spec.loader.exec_module(mod)
    return mod


def per_row_scores(pred_rows: list[dict], split: str) -> list[dict]:
    ev = _ev()
    gt_rows = load_split(split)
    return ev.evaluate_per_sr_pair(pred_rows, gt_rows, ev.RELATION_TYPE, tolerance=0.05)


def score_all(pred_rows: list[dict], split: str) -> dict:
    """Returns {relation: f1, ..., '__overall__': f1} unrounded."""
    rows = per_row_scores(pred_rows, split)
    by_rel: dict[str, list[float]] = {}
    for r in rows:
        by_rel.setdefault(r["Relation"], []).append(r["f1"])
    out = {rel: sum(v) / len(v) for rel, v in by_rel.items()}
    out["__overall__"] = sum(r["f1"] for r in rows) / len(rows)
    return out


def score_one_relation(preds_by_subject: dict[str, list[str]], relation: str,
                       split: str, subjects: set[str] | None = None) -> dict:
    """Score a single relation.

    Rows of OTHER relations are omitted from the prediction file, which the
    scorer treats as empty predictions. Harmless, because I only read this
    relation's rows back.

    `subjects` restricts the average to a subset of this relation's rows. This
    is not cosmetic: the official scorer iterates the GOLD file, so a subject
    missing from preds_by_subject is scored as an empty PREDICTION (usually 0),
    not excluded. Any honest subsetting (dropping train rows that appear in
    their own few-shot demo block, say) has to filter here, after scoring,
    rather than by simply passing fewer predictions.
    """
    pred_rows = [
        {"SubjectEntity": s, "Relation": relation, "ObjectEntities": [str(o) for o in objs]}
        for s, objs in preds_by_subject.items()
    ]
    rows = [r for r in per_row_scores(pred_rows, split) if r["Relation"] == relation]
    if subjects is not None:
        rows = [r for r in rows if r["SubjectEntity"] in subjects]
    if not rows:
        raise ValueError(f"no rows left for {relation} on {split} after filtering")
    n = len(rows)
    return {
        "macro_p": sum(r["p"] for r in rows) / n,
        "macro_r": sum(r["r"] for r in rows) / n,
        "macro_f1": sum(r["f1"] for r in rows) / n,
        "n_rows": n,
        "f1_vector": [r["f1"] for r in rows],
        "subjects": [r["SubjectEntity"] for r in rows],
    }


def paired_bootstrap(f1_a: list[float], f1_b: list[float], n_boot: int = 10000,
                     seed: int = 20260811, alpha: float = 0.10) -> dict:
    """Paired per-row bootstrap of (b - a). KEEP iff the CI excludes 0 from above."""
    import random
    if len(f1_a) != len(f1_b):
        raise ValueError(f"length mismatch {len(f1_a)} vs {len(f1_b)}")
    n = len(f1_a)
    deltas = [b - a for a, b in zip(f1_a, f1_b)]
    point = sum(deltas) / n
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += deltas[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return {
        "point": point, "ci_lo": lo, "ci_hi": hi,
        "excludes_zero_above": lo > 0,
        "n_rows": n,
        "rows_up": sum(1 for d in deltas if d > 0),
        "rows_down": sum(1 for d in deltas if d < 0),
    }
