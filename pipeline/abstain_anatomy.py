"""Anatomy of the two abstention relations (personHasCityOfDeath, companyTradesAtStockExchange).

Read-only analysis. Touches nothing that ships: no writes to configs/, submissions/
or NOTES.local.md. Everything is measured on the LOCALLY CACHED pools, so it costs
zero GPU and zero submissions.

Closed book: the only factual inputs are dataset2026/data/{train,val}.jsonl gold via
common.load_split, and the model's own cached draws via common.load_pool. No external
lookup of any subject entity.

Contents
  * row_view()          per-row features from a pool: vote shares, top share,
                        share of draws that produced NO candidate at all.
  * confusion()         the five-bucket decomposition of a decision rule against gold.
  * oracles()           perfect-abstain / perfect-content / pool-limited counterfactuals.
  * ef1_predict()       expected-F1 maximiser over prefixes of the candidate list.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate import predict_set, vote_shares
from channels import CHANNELS, demo_ids, pick_demos
from common import gold_aliases, load_pool, rows_for, spec_for_channel
from scorer import per_row_scores, score_one_relation

ARGS = argparse.Namespace(
    model="google/gemma-4-31B",
    revision="5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89",
    temperature=0.7, top_p=0.95, demo_seed=1234, seed_base=7000,
)

# The shipped configuration for the two relations, copied from
# configs/best_measured.json (read, never written).
SHIPPED = {
    "personHasCityOfDeath": dict(channel="death_obituary", tau=0.45, emit_ratio=0.0),
    "companyTradesAtStockExchange": dict(channel="stock_sentinel", tau=0.35, emit_ratio=0.5),
}


def get_pool(channel: str, split: str) -> dict[str, list[str]]:
    return load_pool(spec_for_channel(CHANNELS[channel], split, ARGS))


def demo_subjects(channel: str) -> set[str]:
    ch = CHANNELS[channel]
    return set(demo_ids(pick_demos(ch.relation, ch.n_demos, ARGS.demo_seed, ch.demo_strategy)))


def usable_subjects(channel: str, split: str, pool: dict) -> set[str]:
    """LEAKAGE GUARD: a train subject that is also a demo has its own gold answer
    sitting in the prompt. Exclude those from any train-side measurement."""
    if split != "train":
        return set(pool)
    return set(pool) - demo_subjects(channel)


# ------------------------------------------------------------------ row features


def row_view(draws: list[str], channel: str) -> dict:
    """Per-row summary of one subject's draw pool.

    empty_share is the fraction of draws from which the channel's parser
    extracted NO candidate. For these two channels that is exactly the in-band
    sentinel ("still living" / "not listed") plus genuine blanks, i.e. the
    model's own vote for "there is no answer". It is a DIFFERENT quantity from
    the top vote share and that difference is the whole content of section 4.
    """
    ch = CHANNELS[channel]
    n = len(draws)
    empty = sum(1 for d in draws if not ch.parse(d))
    sh = vote_shares(draws, channel)
    cands = sorted(((s, surf, k) for k, (s, surf) in sh.items()), reverse=True)
    return {
        "n_draws": n,
        "empty_share": empty / n if n else 1.0,
        "cands": cands,                       # [(share, surface, normform), ...] desc
        "top_share": cands[0][0] if cands else 0.0,
        "n_cands": len(cands),
    }


def views_for(channel: str, split: str) -> dict[str, dict]:
    pool = get_pool(channel, split)
    return {s: row_view(d, channel) for s, d in pool.items()}


# ------------------------------------------------------------------ gold access


def gold_map(relation: str, split: str) -> dict[str, list[list[str]]]:
    return {r["SubjectEntity"]: gold_aliases(r) for r in rows_for(split, relation)}


# ------------------------------------------------------------------ scoring helper


def score_preds(preds: dict[str, list[str]], relation: str, split: str,
                subjects: set[str] | None = None) -> dict:
    return score_one_relation(preds, relation, split, subjects=subjects)


def per_row(preds: dict[str, list[str]], relation: str, split: str) -> dict[str, dict]:
    rows = [{"SubjectEntity": s, "Relation": relation, "ObjectEntities": list(o)}
            for s, o in preds.items()]
    return {r["SubjectEntity"]: r for r in per_row_scores(rows, split)
            if r["Relation"] == relation}


# ------------------------------------------------------------------ 1. confusion


BUCKETS = ("correct_abstain", "wrong_abstain", "wrong_answer",
           "answer_exact", "answer_partial", "answer_zero")


def confusion(preds: dict[str, list[str]], relation: str, split: str,
              subjects: set[str] | None = None) -> dict:
    """Five-way (six with the partial split) decomposition of the row scores.

    Buckets are mutually exclusive and exhaust the rows, and the reported
    contributions sum to the relation macro-F1 by construction.
    """
    rows = per_row(preds, relation, split)
    gold = gold_map(relation, split)
    keep = subjects if subjects is not None else set(rows)
    out = {b: {"n": 0, "score": 0.0} for b in BUCKETS}
    n = 0
    for s, r in rows.items():
        if s not in keep:
            continue
        n += 1
        g_empty = len(gold.get(s, [])) == 0
        p_empty = r["total_pred"] == 0
        f1 = r["f1"]
        if p_empty and g_empty:
            b = "correct_abstain"
        elif p_empty and not g_empty:
            b = "wrong_abstain"
        elif not p_empty and g_empty:
            b = "wrong_answer"
        elif f1 >= 0.999:
            b = "answer_exact"
        elif f1 > 0:
            b = "answer_partial"
        else:
            b = "answer_zero"
        out[b]["n"] += 1
        out[b]["score"] += f1
    for b in out:
        out[b]["contrib"] = out[b]["score"] / n if n else 0.0
    out["__n__"] = n
    out["__macro_f1__"] = sum(out[b]["score"] for b in BUCKETS) / n if n else 0.0
    return out


# ------------------------------------------------------------------ 2. oracles


def content_set(view: dict, cfg: dict) -> list[str]:
    """The system's OBJECT CHOICE, with the abstain gate removed.

    For a rule with emit_ratio > 0 (stock_sentinel) the emission rule
    {c : share >= ratio * top} is already defined independently of the abstain
    gate, so this is exactly what the rule emits when it answers.

    For a single-threshold rule (death_obituary) tau does both jobs, so the
    counterfactual "what would I have said" is the set at tau if non-empty and
    otherwise the top-1 candidate. Returns [] only when the pool holds no
    candidate at all, in which case no answer is available at any threshold.
    """
    cands = view["cands"]
    if not cands:
        return []
    if cfg["emit_ratio"] > 0:
        top = cands[0][0]
        return [surf for sh, surf, _ in cands if sh >= cfg["emit_ratio"] * top]
    keep = [surf for sh, surf, _ in cands if sh >= cfg["tau"]]
    return keep if keep else [cands[0][1]]


def oracles(relation: str, split: str, cfg: dict,
            subjects: set[str] | None = None) -> dict:
    """Counterfactual scores that separate the two decisions.

    shipped          : the live rule.
    oracle_abstain   : abstain iff gold is empty; OBJECT CHOICE UNCHANGED.
    oracle_content   : shipped abstain decision; objects perfect where I answer.
    oracle_content_pool : shipped abstain decision; objects = best achievable
                       subset of the candidates actually present in the pool.
    Both perfect is 1.0 by construction and is not reported.
    """
    ch = cfg["channel"]
    pool = get_pool(ch, split)
    keep = subjects if subjects is not None else usable_subjects(ch, split, pool)
    views = {s: row_view(d, ch) for s, d in pool.items() if s in keep}
    gold = gold_map(relation, split)

    shipped = {s: predict_set(pool[s], ch, cfg["tau"], emit_ratio=cfg["emit_ratio"])
               for s in views}
    content = {s: content_set(v, cfg) for s, v in views.items()}

    # oracle abstain: perfect gate, my content
    orc_ab = {s: ([] if not gold.get(s) else content[s]) for s in views}

    rows_all = per_row({s: list(v["cands"] and [c[1] for c in v["cands"]] or [])
                        for s, v in views.items()}, relation, split)

    n = len(views)
    sc_shipped = sum(r["f1"] for s, r in per_row(shipped, relation, split).items() if s in views) / n
    sc_orc_ab = sum(r["f1"] for s, r in per_row(orc_ab, relation, split).items() if s in views) / n

    # oracle content: 1.0 on correctly-abstained rows and on answered rows whose
    # gold is non-empty; 0 elsewhere (no object choice can rescue an answered
    # row whose gold is empty, since precision is 0/len(preds) there).
    n_correct_abstain = sum(1 for s in views if not shipped[s] and not gold.get(s))
    n_answer_nonempty = sum(1 for s in views if shipped[s] and gold.get(s))
    sc_orc_ct = (n_correct_abstain + n_answer_nonempty) / n

    # pool-limited content oracle: on each answered row the best achievable F1
    # using only candidates the pool actually contains. t = maximum matching
    # between pool candidates and gold, taken from the official scorer's tp.
    sc_pool = 0.0
    for s in views:
        g = gold.get(s, [])
        if not shipped[s]:
            sc_pool += 1.0 if not g else 0.0
            continue
        if not g:
            continue  # answered on empty gold: 0 whatever I emit
        t = rows_all[s]["tp"]
        m = len(g)
        sc_pool += (2 * t / (t + m)) if t else 0.0
    sc_pool /= n

    return {
        "n": n,
        "shipped": sc_shipped,
        "oracle_abstain": sc_orc_ab,
        "oracle_content": sc_orc_ct,
        "oracle_content_pool": sc_pool,
        "n_correct_abstain": n_correct_abstain,
        "n_answer_nonempty": n_answer_nonempty,
    }


# ------------------------------------------------------------------ 4. expected-F1


def _sk():
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    return IsotonicRegression, LogisticRegression


class EF1Calibrator:
    """P(gold empty | row) and P(candidate in gold | share, gold non-empty).

    MODEL FORM IS DECLARED BEFORE ANY ACCURACY IS COMPUTED, so that the
    comparison against the shipped tau rule is one comparison and not the argmax
    of a search:

      q  = sigmoid(a + b*top_share + c*empty_share)      [use_empty=True]
           sigmoid(a + b*top_share)                      [use_empty=False, control]
      c_i = isotonic(share_i), fitted on candidates of NON-EMPTY-gold rows only
      P(m) = empirical gold-set-size distribution over non-empty-gold rows

    empty_share is the fraction of draws from which the parser extracted nothing,
    i.e. the model's own in-band vote for "no answer exists". It is the only
    feature here that the shipped single-threshold rule cannot see.
    """

    def __init__(self, use_empty: bool = True):
        self.use_empty = use_empty

    def fit(self, rows: list[tuple[dict, list]]):
        Iso, LR = _sk()
        X, y = [], []
        cx, cy = [], []
        ms = []
        for view, gold in rows:
            X.append(self._feat(view))
            y.append(1 if not gold else 0)
            if gold:
                ms.append(len(gold))
                norm_gold = {_norm(a) for al in gold for a in al}
                for sh, surf, key in view["cands"]:
                    cx.append(sh)
                    cy.append(1 if key in norm_gold else 0)
        self.lr = LR(C=1.0, max_iter=1000).fit(X, y)
        self.iso = Iso(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(cx, cy)
        tot = len(ms)
        self.pm = {}
        for m in set(ms):
            self.pm[m] = ms.count(m) / tot
        if not self.pm:
            self.pm = {1: 1.0}
        return self

    def _feat(self, view):
        return [view["top_share"], view["empty_share"]] if self.use_empty else [view["top_share"], 0.0]

    def q(self, view) -> float:
        return float(self.lr.predict_proba([self._feat(view)])[0][1])

    def c(self, shares: list[float]) -> list[float]:
        if not shares:
            return []
        return [float(v) for v in self.iso.predict(shares)]


def _norm(s: str) -> str:
    from aggregate import normalize
    return normalize(s)


def ef1_predict(view: dict, cal: EF1Calibrator, kmax: int = 4) -> tuple[list[str], dict]:
    """Maximise E[F1] over PREFIXES of the candidate list sorted by vote share.

    E[F1(S)] = q                                                   if S is empty
             = (1-q) * sum_m P(m) * 2*min(sum_i c_i, m) / (|S|+m)  otherwise

    (P = tp/|S|, R = tp/m gives F1 = 2*tp/(|S|+m); an empty prediction against
    empty gold scores exactly 1, so abstaining is worth exactly q.)
    The independence approximation enters only as E[tp] = sum_i c_i, capped at m.
    """
    q = cal.q(view)
    cands = view["cands"][:kmax]
    cs = cal.c([sh for sh, _, _ in cands])
    best_k, best_v = 0, q
    curve = [q]
    run = 0.0
    for k in range(1, len(cands) + 1):
        run += cs[k - 1]
        v = (1 - q) * sum(p * 2 * min(run, m) / (k + m) for m, p in cal.pm.items())
        curve.append(v)
        if v > best_v:
            best_k, best_v = k, v
    return [surf for _, surf, _ in cands[:best_k]], {"q": q, "curve": curve, "k": best_k}


def ef1_predict_joint(view: dict, cal: EF1Calibrator, u: float = 0.0,
                      kmax: int = 4) -> tuple[list[str], dict]:
    """EF1 maximiser WITHOUT the independent-m approximation.

    Post-hoc variant, written after the marginal-m version lost on
    companyTradesAtStockExchange, and written for a measured reason: conditional
    on the 2nd candidate holding at least half the top share, P(|gold| >= 2) is
    0.769 against a marginal of 0.209 (pooled train(non-demo)+val, n=13 vs 86).
    The marginal-m model throws exactly that dependence away.

    Here the gold set is generated by the same independent Bernoulli(c_i) draw
    over pool candidates that supplies E[tp], so |gold| and the candidate profile
    are coupled by construction. `u` is the expected number of gold objects that
    are absent from the pool. Enumeration is exact over at most 2**kmax subsets.
    """
    q = cal.q(view)
    cands = view["cands"][:kmax]
    cs = cal.c([sh for sh, _, _ in cands])
    n = len(cands)
    subsets = []
    for mask in range(1 << n):
        p = 1.0
        t = []
        for i in range(n):
            if mask >> i & 1:
                p *= cs[i]
                t.append(i)
            else:
                p *= 1 - cs[i]
        subsets.append((p, set(t)))
    z = sum(p for p, t in subsets if t or u > 0)
    best_k, best_v, curve = 0, q, [q]
    for k in range(1, n + 1):
        S = set(range(k))
        v = 0.0
        for p, t in subsets:
            if not t and u <= 0:
                continue
            m = len(t) + u
            if m <= 0:
                continue
            v += p * 2 * len(S & t) / (k + m)
        v = (1 - q) * v / z if z else 0.0
        curve.append(v)
        if v > best_v:
            best_k, best_v = k, v
    return [surf for _, surf, _ in cands[:best_k]], {"q": q, "curve": curve, "k": best_k}
