"""ONE per-row expected-F1 decision rule for ALL SIX relations, and its honest test.

--------------------------------------------------------------------------
WHAT IS RIGHT BY CONSTRUCTION AND WHAT IS NOT
--------------------------------------------------------------------------
Every rule this system ships emits a PREFIX of a fixed candidate ranking:
  * a vote-share tau keeps the candidates above it, which (shares sorted
    descending) is a prefix;
  * stockX's emit_ratio keeps share >= r*top, also a prefix;
  * the numeric relations emit the rank-1 candidate, i.e. the prefix k=1.
So the only decision any of them makes is WHERE TO CUT. This module replaces the
global scalar cut with a per-row cut that maximises expected F1.

Under evaluate.py a row's score is exactly

    F1 = 2*tp / (n_pred + n_gold)          (and 1.0 when both are 0)

verified on every non-degenerate train+val row by set_metric_facts.py. Two regimes
follow, and they need different amounts of assumption:

 A. AT MOST ONE GOLD (hasArea, hasCapacity: always exactly 1; personHasCityOfDeath:
    0 or 1, verified below). Candidates are made mutually exclusive: numerics by
    a 1.05/0.95 tolerance separation, strings because two distinct normalized
    predictions cannot both be the single gold. Then

        E[F1](k) = 2 * C_k / (k + 1),   C_k = sum_{i<=k} p_i = P(gold in top k)
        E[F1](0) = P(no gold)

    This needs ONLY the marginal p_i. No independence assumption is used, because
    C_k is LINEAR in the p_i. Given correct p_i the rule is exactly optimal.

 B. VARIABLE GOLD SIZE (borders, stockX, award). E[F1] depends on the whole
    distribution of |gold|, so I need a generative model: candidate i is in gold
    independently with probability p_i, plus U unseen golds the pool never
    proposed, U drawn from the empirical unseen-count distribution of the FITTING
    split. E[F1](k) is then estimated by Monte Carlo. This regime carries two real
    assumptions (independence across candidates, and U independent of the row).

THE HONEST STATEMENT OF WHAT IS PROVED. With p_i = P(candidate i in gold | the
whole share vector), the prefix maximiser weakly dominates every global threshold.
What I can actually FIT is q(share_i) = P(in gold | that candidate's own share),
which ignores the rest of the row. So the rule is Bayes-optimal only within the
restricted model "share_i is a sufficient statistic". It is NOT dominant by
construction, and calib_diag.py already shows the assumption is violated for
borders. Everything below is therefore measured, not asserted.

--------------------------------------------------------------------------
EVALUATION PROTOCOL (nothing is tuned on the split it is scored on)
--------------------------------------------------------------------------
  * calibration curve AND gold-size model are fitted on one split and applied to
    the other, both directions; the pooled train+val figure concatenates the two
    out-of-fit halves, so no row is ever scored by a curve that saw its gold.
  * train excludes the channel's own demo subjects (their gold is in their prompt).
  * scoring is the OFFICIAL evaluate.py via scorer.score_one_relation, never a
    reimplementation; the comparison is a paired per-row bootstrap.
  * TEST is read for candidate shares only, to count rows the rule would change.
    Test gold is hidden, so every test-side number here is labelled a PREDICTION.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate import predict_set, vote_shares
from channels import CHANNELS
from common import (TEST_ROWS, TOTAL_TEST_ROWS, load_pool, rows_for,
                    spec_for_channel)
from scorer import _ev, paired_bootstrap, score_one_relation
from topk_numeric import KMAX as NUM_KMAX
from topk_numeric import (default_args, demo_subjects, emit, gold_value, hits,
                          ranked_candidates)

ARGS = default_args()
SEED = 20260812

# The shipped system, read off configs/best_measured.json (which this module
# never writes). `kind` is the gold-size regime, not a guess: verified by
# verify_gold_shape() at the top of every run.
RELCFG: dict[str, dict] = {
    "countryLandBordersCountry": dict(channel="borders_list", kind="multi",
                                      tau=0.15, emit_ratio=0.0, kmax=0),
    "personHasCityOfDeath": dict(channel="death_obituary", kind="single",
                                 tau=0.45, emit_ratio=0.0, kmax=0),
    "companyTradesAtStockExchange": dict(channel="stock_sentinel", kind="multi",
                                         tau=0.35, emit_ratio=0.5, kmax=0),
    "hasArea": dict(channel="area_recite", kind="numeric", kmax=NUM_KMAX),
    "hasCapacity": dict(channel="cap_recite", kind="numeric", kmax=NUM_KMAX),
    "awardWonBy": dict(channel="award_list", kind="multi", tau=0.10,
                       emit_ratio=0.0, kmax=250),
}


# ------------------------------------------------------------------ gold shape


def verify_gold_shape(relation: str) -> dict:
    """Measure, do not assume, how many golds a row of this relation has."""
    out = {}
    for split in ("train", "val"):
        sizes = [len(r.get("ObjectEntities") or []) for r in rows_for(split, relation)]
        hist: dict[int, int] = {}
        for s in sizes:
            hist[s] = hist.get(s, 0) + 1
        out[split] = {"rows": len(sizes), "min": min(sizes), "max": max(sizes),
                      "hist": dict(sorted(hist.items()))}
    return out


# ------------------------------------------------------------------ rows


def _ranked_set(draws: list[str], channel: str) -> list[tuple[float, str]]:
    """Candidates as (share, surface), in the SAME order predict_set emits them.

    predict_set does `keep.sort(reverse=True)` on (share, surface) tuples, so
    matching that ordering is what makes "prefix of length len(shipped)" identical
    to the shipped set rather than merely the same size.
    """
    out = [(share, surf) for share, surf in vote_shares(draws, channel).values()]
    out.sort(reverse=True)
    return out


def build_rows(relation: str, split: str) -> list[dict]:
    cfg = RELCFG[relation]
    ch = cfg["channel"]
    pool = load_pool(spec_for_channel(CHANNELS[ch], split, ARGS))
    drop = demo_subjects(ch, ARGS.demo_seed) if split == "train" else set()
    gold_rows = ({r["SubjectEntity"]: r for r in rows_for(split, relation)}
                 if split != "test" else {})
    ev = _ev()

    rows = []
    for subj in sorted(pool):
        if subj in drop:
            continue
        if split != "test" and subj not in gold_rows:
            continue
        draws = pool[subj]

        if cfg["kind"] == "numeric":
            cands = ranked_candidates(draws, ch, kmax=cfg["kmax"])
            shares = [s for _, s in cands]
            surfaces = emit([v for v, _ in cands])
            values = [v for v, _ in cands]
        else:
            rk = _ranked_set(draws, ch)
            shares = [s for s, _ in rk]
            surfaces = [t for _, t in rk]
            values = None

        row = {"subject": subj, "split": split, "shares": shares,
               "surfaces": surfaces, "values": values}

        if split != "test":
            g = gold_rows[subj].get("ObjectEntities") or []
            row["gold"] = g
            row["n_gold"] = len(g)
            if cfg["kind"] == "numeric":
                gv = gold_value(gold_rows[subj])
                labels = [1 if (gv is not None and hits(v, gv)) else 0 for v in values]
            else:
                gsets = [{ev.normalize_string(a) for a in aliases} for aliases in g]
                labels = [1 if any(ev.normalize_string(t) in gs for gs in gsets) else 0
                          for t in surfaces]
            row["labels"] = labels
            row["matched"] = sum(labels)
            row["unseen"] = max(len(g) - sum(labels), 0)
        rows.append(row)
    return rows


# ------------------------------------------------------------------ shipped rule


def shipped_k(row: dict, relation: str) -> int:
    """Length of the prefix the SHIPPED rule emits for this row."""
    cfg = RELCFG[relation]
    if cfg["kind"] == "numeric":
        return 1 if row["shares"] else 0
    shares = row["shares"]
    if not shares:
        return 0
    if cfg["emit_ratio"] > 0:
        top = shares[0]
        if top < cfg["tau"]:
            return 0
        return sum(1 for s in shares if s >= cfg["emit_ratio"] * top)
    return sum(1 for s in shares if s >= cfg["tau"])


# ------------------------------------------------------------------ calibration


def fit_isotonic(pairs: list[tuple[float, int]]) -> list[tuple[float, float]]:
    """Weighted pool-adjacent-violators on (share, label). Returns [(share, q)]."""
    agg: dict[float, list[int]] = defaultdict(list)
    for s, y in pairs:
        agg[round(float(s), 6)].append(y)
    blocks = [[x, sum(agg[x]) / len(agg[x]), len(agg[x])] for x in sorted(agg)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][1] > blocks[i + 1][1] + 1e-15:
            w = blocks[i][2] + blocks[i + 1][2]
            m = (blocks[i][1] * blocks[i][2] + blocks[i + 1][1] * blocks[i + 1][2]) / w
            blocks[i] = [blocks[i][0], m, w]
            del blocks[i + 1]
            if i:
                i -= 1
        else:
            i += 1
    return [(b[0], b[1]) for b in blocks]


def apply_iso(curve: list[tuple[float, float]], share: float) -> float:
    """Step function, clamped away from 0/1 so a single unlucky block cannot
    make a candidate certainly-wrong or certainly-right."""
    q = curve[0][1] if curve else 0.5
    for x, v in curve:
        if share >= x - 1e-12:
            q = v
        else:
            break
    return float(min(max(q, 1e-4), 1 - 1e-4))


BIN_EDGES = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.01]


class Calibrator:
    """share -> P(candidate in gold), in three families.

    Three, not one, because the failure has to be attributable. `iso` imposes
    monotonicity in share; if P(correct) is NOT monotone in share (it is not, on
    borders and cityOfDeath) that constraint itself introduces error. `bin` drops
    the constraint. `logit` additionally uses the row context that the
    sufficiency diagnostic says the marginal curve is missing. That is the
    strongest form of the lever, and the one the "calibration is a well-powered
    smooth fit" argument is really about.
    """

    def __init__(self, kind: str, obs: list[tuple[float, int, float, int, int]]):
        # obs = (share, rank, top_share, K, label)
        self.kind = kind
        self.base = sum(o[4] for o in obs) / max(len(obs), 1)
        self.n_obs = len(obs)
        if kind == "iso":
            self.curve = fit_isotonic([(o[0], o[4]) for o in obs])
            self.blocks = len(self.curve)
        elif kind == "bin":
            acc: dict[int, list[int]] = defaultdict(list)
            for s, _r, _t, _k, y in obs:
                acc[self._bin(s)].append(y)
            a = 5.0                       # pseudo-counts toward the base rate
            self.bins = {b: (sum(v) + a * self.base) / (len(v) + a)
                         for b, v in acc.items()}
            self.blocks = len(self.bins)
        elif kind == "logit":
            from sklearn.linear_model import LogisticRegression
            X = np.array([self._feat(o[0], o[1], o[2], o[3]) for o in obs])
            y = np.array([o[4] for o in obs])
            if len(set(y.tolist())) < 2:
                self.model = None
            else:
                self.model = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
            self.blocks = X.shape[1]
        else:
            raise ValueError(kind)

    @staticmethod
    def _bin(s: float) -> int:
        return max(j for j in range(len(BIN_EDGES) - 1) if s >= BIN_EDGES[j])

    @staticmethod
    def _feat(share: float, rank: int, top: float, K: int) -> list[float]:
        def lg(p: float) -> float:
            p = min(max(p, 1e-3), 1 - 1e-3)
            return float(np.log(p / (1 - p)))
        return [lg(share), lg(top), 1.0 if rank == 0 else 0.0, float(np.log1p(K)),
                lg(share) - lg(top)]

    def q(self, share: float, rank: int, top: float, K: int) -> float:
        if self.kind == "iso":
            return apply_iso(self.curve, share)
        if self.kind == "bin":
            v = self.bins.get(self._bin(share), self.base)
        else:
            if self.model is None:
                v = self.base
            else:
                v = float(self.model.predict_proba(
                    np.array([self._feat(share, rank, top, K)]))[0, 1])
        return float(min(max(v, 1e-4), 1 - 1e-4))

    def row_qs(self, row: dict) -> list[float]:
        top = row["shares"][0] if row["shares"] else 0.0
        K = len(row["shares"])
        return [self.q(s, i, top, K) for i, s in enumerate(row["shares"])]


def fit_calibrator(kind: str, rows: list[dict]) -> Calibrator:
    obs = []
    for r in rows:
        top = r["shares"][0] if r["shares"] else 0.0
        K = len(r["shares"])
        for i, (s, y) in enumerate(zip(r["shares"], r["labels"])):
            obs.append((s, i, top, K, y))
    return Calibrator(kind, obs)


def reliability(cal: Calibrator, rows: list[dict], nbins: int = 10) -> dict:
    """Out-of-fit reliability of q: predicted vs observed, plus ECE."""
    preds, obs = [], []
    for r in rows:
        preds += cal.row_qs(r)
        obs += r["labels"]
    if not preds:
        return {"n": 0, "ece": None, "bins": []}
    p = np.array(preds)
    y = np.array(obs, dtype=float)
    edges = np.linspace(0, 1, nbins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, nbins - 1)
    bins, ece = [], 0.0
    for b in range(nbins):
        m = idx == b
        if not m.any():
            continue
        bins.append({"bin": f"[{edges[b]:.1f},{edges[b+1]:.1f})",
                     "n": int(m.sum()), "mean_q": float(p[m].mean()),
                     "obs_rate": float(y[m].mean())})
        ece += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return {"n": len(p), "ece": float(ece), "bins": bins,
            "mean_q": float(p.mean()), "base_rate": float(y.mean())}


# ------------------------------------------------------------------ gold-size model


def fit_size_model(relation: str, rows: list[dict]) -> dict:
    """Everything about |gold| that the E[F1] calculation needs, fitted on ROWS."""
    n = max(len(rows), 1)
    return {
        "kind": RELCFG[relation]["kind"],
        "empty_rate": sum(1 for r in rows if r["n_gold"] == 0) / n,
        # single-gold regime: P(a gold exists but the pool never proposed it)
        "miss_rate": sum(1 for r in rows if r["n_gold"] >= 1 and r["matched"] == 0) / n,
        # multi-gold regime: empirical distribution of unseen gold counts
        "unseen": [r["unseen"] for r in rows] or [0],
        "n_rows": len(rows),
    }


# ------------------------------------------------------------------ E[F1] curves


def ev_single(qs: list[float], size: dict, allow_empty: bool) -> np.ndarray:
    """AT-MOST-ONE-GOLD regime. Exact given the marginals; no MC, no independence.

    E[F1](k) = 2*C_k/(k+1) with C_k the cumulative sum of the mutually exclusive
    p_i, clamped at 1. E[F1](0) = P(no gold), estimated as 1 - C_K - miss_rate,
    which is the model's OWN statement about this row rather than a separate gate.
    """
    C = np.cumsum(np.array(qs, dtype=float))
    C = np.minimum(C, 1.0)
    ks = np.arange(1, len(qs) + 1)
    ev = 2.0 * C / (ks + 1.0)
    if allow_empty:
        p0 = max(0.0, 1.0 - (C[-1] if len(C) else 0.0) - size["miss_rate"])
    else:
        p0 = 0.0          # relation has zero empty-gold rows on train and val
    return np.concatenate([[p0], ev])


def ev_multi(qs: list[float], size: dict, rng: np.random.Generator,
             n_mc: int, kmax: int) -> np.ndarray:
    """VARIABLE-GOLD regime, Monte Carlo over the generative model."""
    K = len(qs)
    if K == 0:
        return np.array([1.0 if size["empty_rate"] > 0 else 0.0])
    kcap = min(K, kmax) if kmax else K
    q = np.array(qs[:kcap], dtype=float)
    B = (rng.random((n_mc, kcap)) < q)
    U = rng.choice(np.array(size["unseen"]), size=n_mc)
    G = B.sum(axis=1) + U
    cum = np.cumsum(B, axis=1)
    ks = np.arange(1, kcap + 1)
    ev = (2.0 * cum / (ks[None, :] + G[:, None])).mean(axis=0)
    ev0 = float((G == 0).mean())      # F1 = 1 only when I emit nothing AND G == 0
    return np.concatenate([[ev0], ev])


def choose_k(row: dict, cal: Calibrator, size: dict, relation: str,
             rng: np.random.Generator, n_mc: int) -> tuple[int, np.ndarray]:
    cfg = RELCFG[relation]
    qs = cal.row_qs(row)
    if not qs:
        return 0, np.array([1.0])
    if cfg["kind"] == "multi":
        ev = ev_multi(qs, size, rng, n_mc, cfg["kmax"])
    else:
        ev = ev_single(qs, size, allow_empty=size["empty_rate"] > 0)
    return int(np.argmax(ev)), ev


# ------------------------------------------------------------------ scoring


def score_prefixes(rows: list[dict], ks: dict[str, int], relation: str,
                   split: str) -> dict:
    preds = {r["subject"]: r["surfaces"][:ks[r["subject"]]] for r in rows}
    return score_one_relation(preds, relation, split,
                              subjects={r["subject"] for r in rows})


def tp_model_check(rows: list[dict], ks: dict[str, int], relation: str) -> dict:
    """Does the per-candidate label model agree with the GRADER's own tp?

    Two distinct normalized predictions can be aliases of the SAME gold entity, in
    which case summing labels over-counts and the E[F1] model over-values adding a
    candidate. Uses the grader's own dispatch (numeric vs string), not a guess.
    Measured rather than waved away.
    """
    ev = _ev()
    rel_type = ev.RELATION_TYPE[relation]
    bad = 0
    signed = 0
    for r in rows:
        k = ks[r["subject"]]
        preds = r["surfaces"][:k]
        gold = [g if isinstance(g, list) else [g] for g in r["gold"]]
        tp = ev.true_positives(preds, gold, rel_type=rel_type, tolerance=0.05)
        d = sum(r["labels"][:k]) - tp
        if d:
            bad += 1
            signed += d
    return {"rows_mismatched": bad, "n_rows": len(rows),
            "label_sum_minus_grader_tp_total": signed}


# ------------------------------------------------------------------ driver


def run_relation(relation: str, n_mc: int, verbose: bool,
                 calibrator: str = "iso") -> dict:
    cfg = RELCFG[relation]
    rng = np.random.default_rng(SEED)
    data = {sp: build_rows(relation, sp) for sp in ("train", "val", "test")}
    shape = verify_gold_shape(relation)

    print("=" * 92)
    print(f"{relation}   channel={cfg['channel']}   regime={cfg['kind']}   "
          f"rows: train {len(data['train'])} (demo-excluded) / val {len(data['val'])} "
          f"/ test {len(data['test'])}")
    print("=" * 92)
    print(f"  gold shape train {shape['train']['hist']}   val {shape['val']['hist']}")

    out: dict = {"relation": relation, "channel": cfg["channel"],
                 "kind": cfg["kind"], "gold_shape": shape}

    # ---- validity: the offline evaluator must reproduce the shipped rule exactly
    for sp in ("train", "val"):
        ks = {r["subject"]: shipped_k(r, relation) for r in data[sp]}
        s = score_prefixes(data[sp], ks, relation, sp)
        out[f"shipped_{sp}"] = s["macro_f1"]
    print(f"  SHIPPED RULE via this evaluator:  train {out['shipped_train']:.4f}   "
          f"val {out['shipped_val']:.4f}")

    # ---- cross-fitted application
    fitmap = {"val": "train", "train": "val"}       # evaluate-on -> fit-on
    f_ship_pool, f_ev_pool, subj_pool = [], [], []
    per_dir = {}
    for use, fit in fitmap.items():
        cal = fit_calibrator(calibrator, data[fit])
        size = fit_size_model(relation, data[fit])
        rel_diag = reliability(cal, data[use])

        ks_ship = {r["subject"]: shipped_k(r, relation) for r in data[use]}
        ks_ev, ev_rows = {}, []
        for r in data[use]:
            k, ev = choose_k(r, cal, size, relation, rng, n_mc)
            ks_ev[r["subject"]] = k
            ev_rows.append((r, k, ev))

        a = score_prefixes(data[use], ks_ship, relation, use)
        b = score_prefixes(data[use], ks_ev, relation, use)
        assert a["subjects"] == b["subjects"]
        f_ship_pool += a["f1_vector"]
        f_ev_pool += b["f1_vector"]
        subj_pool += [(use, s) for s in a["subjects"]]

        changed = sum(1 for r in data[use] if ks_ev[r["subject"]] != ks_ship[r["subject"]])
        up = sum(1 for r in data[use] if ks_ev[r["subject"]] > ks_ship[r["subject"]])
        down = sum(1 for r in data[use] if ks_ev[r["subject"]] < ks_ship[r["subject"]])
        per_dir[use] = {
            "fit_on": fit, "n_rows": a["n_rows"],
            "shipped": a["macro_f1"], "ev": b["macro_f1"],
            "delta": b["macro_f1"] - a["macro_f1"],
            "rows_changed": changed, "k_up": up, "k_down": down,
            "mean_k_shipped": sum(ks_ship.values()) / max(len(ks_ship), 1),
            "mean_k_ev": sum(ks_ev.values()) / max(len(ks_ev), 1),
            "abstain_shipped": sum(1 for v in ks_ship.values() if v == 0),
            "abstain_ev": sum(1 for v in ks_ev.values() if v == 0),
            "reliability": rel_diag,
            "tp_model_check_EV": tp_model_check(data[use], ks_ev, relation),
            "tp_model_check_shipped": tp_model_check(data[use], ks_ship, relation),
            "calibrator": calibrator,
            "calibrator_blocks": cal.blocks,
            "size_model": {k: v for k, v in size.items() if k != "unseen"},
            "mean_unseen": sum(size["unseen"]) / max(len(size["unseen"]), 1),
        }
        print(f"\n  --- fit on {fit:5s} -> evaluate on {use:5s} "
              f"(n={a['n_rows']}) ---")
        print(f"      shipped {a['macro_f1']:.4f}   EV {b['macro_f1']:.4f}   "
              f"delta {b['macro_f1']-a['macro_f1']:+.4f}")
        print(f"      rows changed {changed}  (k up {up}, k down {down})   "
              f"mean k {per_dir[use]['mean_k_shipped']:.2f} -> "
              f"{per_dir[use]['mean_k_ev']:.2f}   abstain "
              f"{per_dir[use]['abstain_shipped']} -> {per_dir[use]['abstain_ev']}")
        print(f"      calibration[{calibrator}]: {cal.blocks} blocks, ECE "
              f"{rel_diag['ece']:.4f} on {rel_diag['n']} candidates "
              f"(mean q {rel_diag['mean_q']:.3f} vs base rate "
              f"{rel_diag['base_rate']:.3f})")
        if verbose:
            print(f"      {'q bin':>12s} {'n':>6s} {'mean q':>8s} {'observed':>9s}")
            for bb in rel_diag["bins"]:
                print(f"      {bb['bin']:>12s} {bb['n']:6d} {bb['mean_q']:8.3f} "
                      f"{bb['obs_rate']:9.3f}")

    n = len(f_ship_pool)
    pooled_ship = sum(f_ship_pool) / n
    pooled_ev = sum(f_ev_pool) / n
    bs = paired_bootstrap(f_ship_pool, f_ev_pool, n_boot=10000)
    w = TEST_ROWS[relation] / TOTAL_TEST_ROWS
    out["per_direction"] = per_dir
    out["pooled"] = {"n_rows": n, "shipped": pooled_ship, "ev": pooled_ev,
                     "delta": pooled_ev - pooled_ship,
                     "ci_lo": bs["ci_lo"], "ci_hi": bs["ci_hi"],
                     "rows_up": bs["rows_up"], "rows_down": bs["rows_down"],
                     "overall_value": bs["point"] * w,
                     "overall_ci": [bs["ci_lo"] * w, bs["ci_hi"] * w]}
    print(f"\n  POOLED cross-fitted train+val (n={n}): shipped {pooled_ship:.4f} -> "
          f"EV {pooled_ev:.4f}   delta {bs['point']:+.4f}")
    print(f"  paired bootstrap 90% CI [{bs['ci_lo']:+.4f}, {bs['ci_hi']:+.4f}]   "
          f"rows better {bs['rows_up']} / worse {bs['rows_down']}")
    print(f"  value on the OVERALL score if it transferred: {bs['point']*w:+.5f} "
          f"(CI [{bs['ci_lo']*w:+.5f}, {bs['ci_hi']*w:+.5f}])")

    # ---- TEST-side application (prediction only; test gold is hidden)
    cal = fit_calibrator(calibrator, data["train"] + data["val"])
    size = fit_size_model(relation, data["train"] + data["val"])
    ks_ship = {r["subject"]: shipped_k(r, relation) for r in data["test"]}
    ks_ev = {}
    for r in data["test"]:
        k, _ = choose_k(r, cal, size, relation, rng, n_mc)
        ks_ev[r["subject"]] = k
    nt = len(data["test"])
    changed = sum(1 for s in ks_ship if ks_ev[s] != ks_ship[s])
    up = sum(1 for s in ks_ship if ks_ev[s] > ks_ship[s])
    down = sum(1 for s in ks_ship if ks_ev[s] < ks_ship[s])
    out["test"] = {
        "n_rows": nt, "rows_changed": changed, "k_up": up, "k_down": down,
        "mean_k_shipped": sum(ks_ship.values()) / max(nt, 1),
        "mean_k_ev": sum(ks_ev.values()) / max(nt, 1),
        "abstain_shipped": sum(1 for v in ks_ship.values() if v == 0),
        "abstain_ev": sum(1 for v in ks_ev.values() if v == 0),
    }
    print(f"\n  TEST (prediction, gold hidden): {changed}/{nt} rows change "
          f"(k up {up}, k down {down});  mean k "
          f"{out['test']['mean_k_shipped']:.2f} -> {out['test']['mean_k_ev']:.2f};  "
          f"abstain {out['test']['abstain_shipped']} -> {out['test']['abstain_ev']}")
    return out


# ------------------------------------------------------------------ diagnostics


def oracle_prefix(relation: str) -> dict:
    """Ceiling for ANY per-row prefix rule: pick k with the gold in hand.

    This is the honest bound on what a perfect calibration could buy, because the
    EV maximiser and every global tau both emit a prefix of the same ranking. If
    the oracle gain is small, no calibration however good can pay.
    """
    res = {}
    ship_all, orc_all = [], []
    for sp in ("train", "val"):
        rows = build_rows(relation, sp)
        ks_ship = {r["subject"]: shipped_k(r, relation) for r in rows}
        a = score_prefixes(rows, ks_ship, relation, sp)
        # per-row best k, found by scoring every prefix with the official scorer
        per_row_best: dict[str, int] = {}
        best_f: dict[str, float] = {}
        maxK = max((len(r["surfaces"]) for r in rows), default=0)
        for k in range(0, min(maxK, RELCFG[relation]["kmax"] or maxK) + 1):
            s = score_prefixes(rows, {r["subject"]: k for r in rows}, relation, sp)
            for subj, f in zip(s["subjects"], s["f1_vector"]):
                if subj not in best_f or f > best_f[subj] + 1e-12:
                    best_f[subj], per_row_best[subj] = f, k
        b = score_prefixes(rows, per_row_best, relation, sp)
        res[sp] = {"n": a["n_rows"], "shipped": a["macro_f1"],
                   "oracle_prefix": b["macro_f1"],
                   "gain": b["macro_f1"] - a["macro_f1"]}
        ship_all += a["f1_vector"]
        orc_all += b["f1_vector"]
    n = len(ship_all)
    w = TEST_ROWS[relation] / TOTAL_TEST_ROWS
    res["pooled"] = {"n": n, "shipped": sum(ship_all) / n,
                     "oracle_prefix": sum(orc_all) / n,
                     "gain": (sum(orc_all) - sum(ship_all)) / n,
                     "overall_value_of_a_PERFECT_cut": (sum(orc_all) - sum(ship_all)) / n * w}
    return res


def sufficiency(relation: str) -> dict:
    """Is the candidate's own vote share a sufficient statistic for P(correct)?

    The EV rule is Bayes-optimal only within the model "q depends on share alone".
    Here the same share bucket is split by the row's own context. A large split
    means the marginal curve mis-prices exactly the marginal candidates the
    stopping rule decides about.
    """
    rows = build_rows(relation, "train") + build_rows(relation, "val")
    edges = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.01]
    cells: dict[tuple, list[int]] = defaultdict(list)
    for r in rows:
        top = r["shares"][0] if r["shares"] else 0.0
        band = "top>=.75" if top >= 0.75 else ("0.35<=top<.75" if top >= 0.35 else "top<.35")
        for i, (s, y) in enumerate(zip(r["shares"], r["labels"])):
            b = max(j for j in range(len(edges) - 1) if s >= edges[j])
            cells[(b, "all")].append(y)
            cells[(b, band)].append(y)
            cells[(b, "rank1" if i == 0 else "rank2+")].append(y)
    return {"edges": edges, "cells": {f"{k[0]}|{k[1]}": (len(v), sum(v) / len(v))
                                      for k, v in cells.items()}}


def print_diagnostics(relation: str) -> None:
    o = oracle_prefix(relation)
    print(f"\n  ORACLE PREFIX CEILING ({relation}): the best ANY per-row cut of "
          f"this ranking can do")
    for sp in ("train", "val"):
        print(f"      {sp:5s} n={o[sp]['n']:3d}: shipped {o[sp]['shipped']:.4f} -> "
              f"oracle {o[sp]['oracle_prefix']:.4f}  (+{o[sp]['gain']:.4f})")
    p = o["pooled"]
    print(f"      pooled n={p['n']:3d}: shipped {p['shipped']:.4f} -> oracle "
          f"{p['oracle_prefix']:.4f}  (+{p['gain']:.4f});  a PERFECT per-row cut "
          f"would be worth {p['overall_value_of_a_PERFECT_cut']:+.5f} overall")

    s = sufficiency(relation)
    ed = s["edges"]
    print(f"  SHARE-SUFFICIENCY (train+val pooled): hit rate in a share bucket, "
          f"split by row context")
    print(f"      {'share bucket':>14s} {'all':>14s} {'rank1':>14s} {'rank2+':>14s} "
          f"{'top>=.75':>14s} {'top<.35':>14s}")
    for b in range(len(ed) - 1):
        key = f"{b}|all"
        if key not in s["cells"] or s["cells"][key][0] < 5:
            continue
        cells = []
        for lab in ("all", "rank1", "rank2+", "top>=.75", "top<.35"):
            c = s["cells"].get(f"{b}|{lab}")
            cells.append(f"{c[1]:.2f} (n={c[0]})" if c and c[0] >= 3 else "-")
        print(f"      [{ed[b]:.2f},{ed[b+1]:.2f})".rjust(14)
              + "".join(f"{c:>15s}" for c in cells))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--relations", default=",".join(RELCFG))
    ap.add_argument("--n-mc", type=int, default=4000)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--calibrator", default="iso", choices=("iso", "bin", "logit"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.diagnose:
        for rel in a.relations.split(","):
            print("=" * 92)
            print(rel)
            print("=" * 92)
            print_diagnostics(rel)
            print()
        return 0

    results = {}
    for rel in a.relations.split(","):
        n_mc = 1200 if rel == "awardWonBy" else a.n_mc
        results[rel] = run_relation(rel, n_mc, a.verbose, a.calibrator)
        print()

    print("=" * 92)
    print(f"SUMMARY [calibrator={a.calibrator}]: cross-fitted pooled train+val, "
          f"and what each is worth OVERALL")
    print("=" * 92)
    print(f"{'relation':32s} {'rows':>5s} {'shipped':>8s} {'EV':>8s} {'delta':>8s} "
          f"{'90% CI':>20s} {'overall':>9s} {'test rows':>10s}")
    total = 0.0
    for rel, r in results.items():
        p = r["pooled"]
        total += p["overall_value"]
        print(f"{rel:32s} {p['n_rows']:5d} {p['shipped']:8.4f} {p['ev']:8.4f} "
              f"{p['delta']:+8.4f} [{p['ci_lo']:+.4f},{p['ci_hi']:+.4f}] "
              f"{p['overall_value']:+9.5f} {r['test']['rows_changed']:4d}/"
              f"{r['test']['n_rows']:<5d}")
    print(f"{'SUM (adopt everywhere)':32s} {'':5s} {'':8s} {'':8s} {'':8s} "
          f"{'':20s} {total:+9.5f}")
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=1, default=float))
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
