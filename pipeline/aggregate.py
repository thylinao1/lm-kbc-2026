"""Turn vote pools into predictions, plus the parse-QA telemetry gate.

Two aggregation families:
  * SET relations (borders, cityOfDeath, stockX, award): vote share per
    candidate, keep those at or above tau. Abstention is a vote-share outcome,
    never a separate yes/no gate (a standalone gate was measured inert and
    removing vote-share cost -0.14 in the public #1's own ablation).
  * NUMERIC relations (hasArea, hasCapacity): cluster draws in log space,
    take the median of the largest cluster. Never average across magnitudes.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict

from channels import CHANNELS


# ------------------------------------------------------------------ normalisation
# Mirrors evaluate.py so our vote counting groups exactly what the grader groups.

_PUNCT_RE = re.compile(r"[^\w\s]|[+$<=>|~^]", re.UNICODE)


def normalize(s: str) -> str:
    s = s.replace("’", "").replace("'", "").replace("ʼ", "")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.casefold()
    s = _PUNCT_RE.sub(" ", s)
    return " ".join(s.split())


# ------------------------------------------------------------------ parse QA


# Strings that mean "the model deliberately declined", per channel family. An
# empty extraction caused by one of these is a CORRECT abstention, not a parse
# failure, and on the empty-heavy relations it is most of the value: borders is
# 14.9% empty on test, stockX 44%, cityOfDeath 48%.
_ABSTAIN_MARKERS = (
    "none", "no land borders", "not listed", "unlisted", "not publicly",
    "privately held", "private", "still living", "still alive", "alive",
    "living", "n/a", "unknown", "-",
)


def _is_abstention(text: str) -> bool:
    t = " ".join(text.strip().lower().split()).strip(".;,")
    if not t:
        return False
    if t in _ABSTAIN_MARKERS:
        return True
    return any(t.startswith(m) and len(t) <= len(m) + 12 for m in _ABSTAIN_MARKERS)


def parse_qa(pool: dict[str, list[str]], channel: str) -> dict:
    """Telemetry that makes a broken decode contract visible before it silently
    poisons every threshold tuned on this pool.

    Crucially this separates an INTENTIONAL abstention (the model emitted the
    in-band sentinel: "none", "not listed", "still living") from a genuine
    PARSE FAILURE (the model said something and our regex got nothing out of
    it). Only the latter indicates a broken contract. Conflating them would
    fail the gate on exactly the relations whose abstention behaviour is the
    point.
    """
    ch = CHANNELS[channel]
    n_draws = abstain = parse_fail = trunc = empty_text = 0
    lengths = []
    rejected: Counter[str] = Counter()
    for draws in pool.values():
        for d in draws:
            n_draws += 1
            lengths.append(len(d))
            if not ch.parse(d):
                if _is_abstention(d):
                    abstain += 1
                elif not d.strip():
                    empty_text += 1
                else:
                    parse_fail += 1
                    rejected[" ".join(d.strip().split())[:80]] += 1
            if not d.endswith(tuple(ch.stop)) and len(d) > ch.max_tokens * 3:
                trunc += 1
    n_draws = max(n_draws, 1)
    return {
        "channel": channel,
        "n_draws": n_draws,
        "abstention_rate": round(abstain / n_draws, 4),
        "parse_fail_rate": round(parse_fail / n_draws, 4),
        "empty_text_rate": round(empty_text / n_draws, 4),
        "empty_extraction_rate": round((abstain + parse_fail + empty_text) / n_draws, 4),
        "truncated_rate": round(trunc / n_draws, 4),
        "mean_chars": round(sum(lengths) / max(len(lengths), 1), 1),
        "top_rejected": rejected.most_common(20),
    }


def parse_qa_ok(qa: dict) -> tuple[bool, str]:
    """Gate on genuine parse failures only. Abstention is a prediction."""
    if qa["parse_fail_rate"] > 0.05:
        return False, f"parse-fail {qa['parse_fail_rate']:.1%} > 5% (excludes abstentions)"
    if qa["truncated_rate"] > 0.02:
        return False, f"truncated {qa['truncated_rate']:.1%} > 2%"
    if qa["empty_text_rate"] > 0.05:
        return False, f"empty continuations {qa['empty_text_rate']:.1%} > 5%"
    return True, "ok"


# ------------------------------------------------------------------ set aggregation


def vote_shares(draws: list[str], channel: str) -> dict[str, tuple[float, str]]:
    """normalized form -> (share of draws containing it, best surface form)."""
    ch = CHANNELS[channel]
    n = len(draws)
    if not n:
        return {}
    hits: dict[str, int] = defaultdict(int)
    surface: dict[str, Counter] = defaultdict(Counter)
    for d in draws:
        seen = set()
        for c in ch.parse(d):
            k = normalize(c)
            if not k or k in seen:
                continue
            seen.add(k)
            hits[k] += 1
            surface[k][c.strip()] += 1
    return {k: (v / n, surface[k].most_common(1)[0][0]) for k, v in hits.items()}


def predict_set(draws: list[str], channel: str, tau: float,
                max_items: int = 0, emit_ratio: float = 0.0) -> list[str]:
    """Vote-share selection with an optional split between the ABSTAIN decision
    and the EMIT decision.

    Default (emit_ratio=0): one threshold does both jobs, keep everything at or
    above tau. That conflates "is there an answer at all" with "how many".

    With emit_ratio set, tau decides only whether to answer, and membership is
    relative to the top candidate's share. Measured on companyTradesAtStockExchange
    at tau=0.35 / ratio=0.5: train 0.8315 -> 0.8407, val 0.8792 -> 0.8869 with a
    paired-bootstrap CI of [+0.001, +0.017] excluding zero. Harmful on award
    (val 0.2240 -> 0.1268), so it is applied per relation, not globally.
    """
    shares = vote_shares(draws, channel)
    if not shares:
        return []
    if emit_ratio > 0:
        top = max(sh for sh, _ in shares.values())
        if top < tau:
            return []
        keep = [(sh, surf) for sh, surf in shares.values() if sh >= emit_ratio * top]
    else:
        keep = [(sh, surf) for sh, surf in shares.values() if sh >= tau]
    keep.sort(reverse=True)
    if max_items:
        keep = keep[:max_items]
    return [surf for _, surf in keep]


# ------------------------------------------------------------------ numeric aggregation


def log_clusters(values: list[float], width: float = 0.05) -> list[list[float]]:
    """Single-linkage clustering in log10 space.

    NOTE ON width, because the obvious reading is wrong. width is in log10
    units, so width=0.05 is a 12.2% window (10**0.05 = 1.122), NOT the grader's
    5%. Matching the grader exactly would be log10(1.05) = 0.0212.

    We deliberately keep 0.05. Measured, on both numeric relations and both
    splits, the "correct" 0.0212 is WORSE: hasCapacity train 0.3088 -> 0.2941
    and val 0.3608 -> 0.3299; hasArea train 0.8611 -> 0.8333 and val 0.8500 ->
    0.8300. The two quantities are not the same thing. The grader's tolerance
    governs a prediction against gold; this window governs which draws count as
    expressing the same underlying belief, and pooling a wider neighbourhood
    gives a more stable mode estimate. Sharing a unit is a coincidence.
    """
    if not values:
        return []
    vs = sorted(v for v in values if v > 0)
    if not vs:
        return []
    clusters = [[vs[0]]]
    for v in vs[1:]:
        if math.log10(v) - math.log10(clusters[-1][-1]) <= width:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return clusters


def format_number(v: float) -> str:
    """Render one bare, comma-free, non-exponential number for the grader.

    Rounding to an integer here is destructive and was a real bug: hasArea golds
    include 0.43, 0.176, 0.063 and 0.008482 square kilometres, and int(round(v))
    turns every one of them into "0", which can never match anything. Capacity
    golds are integers so it looked safe on the relation it was written for.

    Rule: integers stay integers (no ".0" suffix, which is also what the demos
    show); anything below 1 keeps enough significant digits that the formatting
    error is far inside the grader's 5% tolerance; exponent notation is avoided
    because the grader does a plain float() on the string and we do not need to
    find out how it handles "8.482e-03".
    """
    if v == int(v) and abs(v) >= 1:
        return str(int(v))
    s = f"{v:.10f}".rstrip("0").rstrip(".")
    if s in ("", "-", "0", "-0"):
        # value rounds away entirely at 10dp; fall back to significant digits
        s = f"{v:.12f}".rstrip("0").rstrip(".")
    return s or "0"


def plausible_band(relation: str, lo_q: float = 0.0, hi_q: float = 1.0,
                   pad: float = 2.0) -> tuple[float, float] | None:
    """Range of plausible values for a numeric relation, derived from TRAIN gold.

    Motivation, measured: 45 of 97 validation hasCapacity subjects have a pool
    containing the same leading digits at more than one order of magnitude (the
    model writes 1240 and 1240000 and "1.24 million" for the same fact). In 10
    of those we answered wrong while a same-mantissa alternative in the pool was
    correct. That is 10 rows, worth about 0.021 overall.

    Scale can be disambiguated without gold when the relation has a tight
    natural range. Train gold for hasCapacity runs 1000 to 65000, so a candidate
    of 5 or 5,000,000 is not a stadium capacity. hasArea spans 0.075 to 1.25e6,
    six orders of magnitude, so no such band exists and this must NOT be applied
    there: it would discard correct answers.

    Derived from TRAIN only, which is ours to use, and padded by `pad` in each
    direction so the band never excludes a legitimately extreme test value.
    Returns None when the relation's range is too wide for the idea to be safe.
    """
    from common import rows_for, gold_primary
    vals = []
    for r in rows_for("train", relation):
        for g in gold_primary(r):
            try:
                v = float(str(g).replace(",", ""))
            except ValueError:
                continue
            if v > 0:
                vals.append(v)
    if len(vals) < 20:
        return None
    vals.sort()
    lo = vals[int(lo_q * (len(vals) - 1))]
    hi = vals[int(hi_q * (len(vals) - 1))]
    if hi / lo > 1e4:          # too wide to be informative, e.g. hasArea
        return None
    return lo / pad, hi * pad


def tolerance_support(values: list[float], candidate: float,
                      tol: float = 0.05) -> list[float]:
    """Draws that would score the candidate CORRECT if that draw were the gold.

    The grader's rule is |pred - gold| / gold <= tol, so the ball is defined
    relative to each draw, not to the candidate.
    """
    return [w for w in values if w > 0 and abs(candidate - w) / w <= tol]


def predict_numeric(draws: list[str], channel: str, width: float = 0.05,
                    scale: float = 1.0, method: str = "auto",
                    band: bool = False, antiround: float = 0.0) -> list[str]:
    """Pick one numeric answer from a pool of draws.

    Two selectors, because measurement says the two numeric relations want
    different ones. Chosen on train AND confirmed on val, not fished:

        hasCapacity   log-cluster median   recite 0.3088 train / 0.3608 val
                      (fixed radius: 0.2941 / 0.3299)
        hasArea       fixed-radius ball    lead   0.8611 train / 0.8500 val
                      (log-cluster:   0.8056 / 0.8200)

    The plausible mechanism: capacity draws are coarse round numbers, where
    linkage merging near-duplicates like 10000 and 10500 into one mass is a
    reasonable summary of "about ten thousand". Area draws span five orders of
    magnitude with fine-grained values, where the same merging chains across
    genuinely different answers and the exact tolerance count is safer.

    method="auto" resolves per relation; pass "logcluster" or "ball" to force.
    """
    rel = CHANNELS[channel].relation
    resolved = method
    if method == "auto":
        resolved = "logcluster" if rel == "hasCapacity" else "ball"

    # Scale disambiguation: drop candidates outside the train-derived plausible
    # band before selecting, but only when a band exists AND dropping does not
    # empty the pool. plausible_band() returns None for wide-range relations.
    if band:
        b = _BAND_CACHE.get(rel, ...)
        if b is ...:
            b = plausible_band(rel)
            _BAND_CACHE[rel] = b
        if b is not None:
            kept = [d for d in draws if any(b[0] <= v <= b[1]
                                            for v in _vals_of(d, channel))]
            if kept:
                draws = kept

    if resolved == "logcluster":
        out = _predict_numeric_logcluster(draws, channel, width, scale)
    else:
        out = _predict_numeric_ball(draws, channel, width, scale)

    if antiround and out:
        out = _rescue_round_attractor(out, draws, channel, width, antiround)
    return out


def _rescue_round_attractor(out: list[str], draws: list[str], channel: str,
                            width: float, min_share: float,
                            modulus: int = 1000) -> list[str]:
    """When the winner is a round number, prefer a supported non-round candidate.

    Measured on hasCapacity: our prediction is correct 27.8% of the time when it
    is a multiple of 10000 and 80.0% when it is not a multiple of 1000 (val).
    A specific value like 2048 means the model recalls the venue; 10000 means it
    is guessing, and 10000 was our answer in 6 of the 10 val rows where the pool
    held a correct alternative, wrong every time.

    Effect at min_share=0.15: train +0.0147, val +0.0103. Both point estimates
    positive but neither CI cleanly excludes zero, so this ships as a PROVISIONAL
    keep under the two-tier rule and is logged UNGATED.
    """
    try:
        v0 = float(out[0])
    except (ValueError, IndexError):
        return out
    if v0 % modulus != 0:
        return out
    vals = numeric_candidates(draws, channel)
    if not vals:
        return out
    cands = [(len(tolerance_support(vals, c, width)), c)
             for c in set(vals) if c % modulus != 0]
    if not cands:
        return out
    n, best = max(cands)
    if n / len(vals) < min_share:
        return out
    return [format_number(best)] + out[1:]


_BAND_CACHE: dict = {}


def _vals_of(draw: str, channel: str) -> list[float]:
    out = []
    for c in CHANNELS[channel].parse(draw):
        try:
            v = float(c)
        except ValueError:
            continue
        if v > 0:
            out.append(v)
    return out


def _predict_numeric_logcluster(draws: list[str], channel: str, width: float,
                                scale: float) -> list[str]:
    vals = numeric_candidates(draws, channel)
    if not vals:
        return []
    clusters = log_clusters(vals, width)
    best = sorted(max(clusters, key=len))
    med = (best[len(best) // 2] if len(best) % 2
           else (best[len(best) // 2 - 1] + best[len(best) // 2]) / 2)
    return [format_number(med * scale)]


def _predict_numeric_ball(draws: list[str], channel: str, width: float,
                          scale: float) -> list[str]:
    """Fixed-radius selector.

    Why not largest-log-cluster-median (the published recipe, and this
    function's previous behaviour): single-linkage clustering CHAINS. Given
    draws at 10000, 10500, 12000 and 15000, each adjacent pair is inside the
    linkage width, so they merge into one "cluster" spanning 50 percent, and its
    median is a value no draw proposed and that no gold near either end would
    accept. Observed live on a real pool, where it returned 12000 for a venue
    whose frames voted 10000 / 10500 / 15000.

    Scores every distinct drawn value by how many draws lie within the grader's
    own tolerance of it and takes the argmax. A fixed radius cannot chain the
    way single-linkage does, and it directly maximises the quantity the metric
    rewards. Ties break toward the median of the supporting draws.
    """
    vals = numeric_candidates(draws, channel)
    if not vals:
        return []

    best_v, best_support = None, []
    for cand in sorted(set(vals)):
        sup = tolerance_support(vals, cand, width)
        if best_v is None or len(sup) > len(best_support):
            best_v, best_support = cand, sup
    # tie-break: centre the answer on its own supporting mass
    sup = sorted(best_support) or [best_v]
    centre = sup[len(sup) // 2] if len(sup) % 2 else (sup[len(sup) // 2 - 1] + sup[len(sup) // 2]) / 2
    # keep the centre only if it retains the same support (it usually does)
    if len(tolerance_support(vals, centre, width)) >= len(best_support):
        best_v = centre
    return [format_number(best_v * scale)]


# ------------------------------------------------------------------ cross-frame consensus


def numeric_candidates(draws: list[str], channel: str) -> list[float]:
    ch = CHANNELS[channel]
    out = []
    for d in draws:
        for c in ch.parse(d):
            try:
                v = float(c)
            except ValueError:
                continue
            if v > 0:
                out.append(v)
    return out


def predict_numeric_consensus(
    per_channel: dict[str, list[str]],
    width: float = 0.05,
    channel_agreement_weight: float = 1.0,
    scale: float = 1.0,
) -> tuple[list[str], dict]:
    """Pick a numeric answer using agreement ACROSS frames, not only frequency
    within one frame.

    Motivation. The published state of the art on hasCapacity exhausted the
    within-pool selection family: mass vote, log-probability, co-fact agreement,
    greedy decode, pool filtering, entropy routing, contrastive ranking and
    pairwise duels all capped at the same value, while their pool demonstrably
    contained the gold for about three quarters of subjects. Every one of those
    signals reads the frequency structure of a SINGLE prompt distribution.

    A value that survives a change of register (an infobox completion, a
    pipe-table row, a question, a prose recitation) is supported by a different
    kind of evidence than a value that is merely modal within one register. The
    two are close to independent, so combining them beats either.

    Scoring. Candidates from every frame are pooled and clustered once in log
    space at the grader's tolerance, so a cluster is a set of mutually
    acceptable answers. Each cluster scores

        sum over channels of (share of that channel's draws in the cluster)
        times (1 + w * (distinct channels present - 1))

    which rewards breadth across frames on top of depth within them. The
    representative is the median of the winning cluster.
    """
    per_channel_vals = {ch: numeric_candidates(dr, ch) for ch, dr in per_channel.items()}
    allvals = [v for vs in per_channel_vals.values() for v in vs]
    if not allvals:
        return [], {"reason": "no parseable candidates"}

    # Fixed-radius scoring, not transitive clustering. Each candidate is a value
    # some frame actually proposed; its score is how much support it has inside
    # the grader's tolerance, summed over frames and bonused for breadth.
    scored = []
    for cand in sorted(set(allvals)):
        present, total_share, per_ch_share = 0, 0.0, {}
        for ch, vals in per_channel_vals.items():
            if not vals:
                continue
            k = len(tolerance_support(vals, cand, width))
            if k:
                present += 1
                sh = k / len(vals)
                total_share += sh
                per_ch_share[ch] = round(sh, 3)
        score = total_share * (1.0 + channel_agreement_weight * max(present - 1, 0))
        scored.append({"value": cand, "channels": present, "score": score,
                       "per_channel_share": per_ch_share,
                       "support": len(tolerance_support(allvals, cand, width))})
    # highest score; ties to the candidate with the broadest raw support
    scored.sort(key=lambda c: (-c["score"], -c["support"], c["value"]))
    best = scored[0]
    out = best["value"] * scale
    diag = {
        "chosen": int(round(out)),
        "chosen_channels": best["channels"],
        "chosen_score": round(best["score"], 3),
        "per_channel_share": best["per_channel_share"],
        "runner_up": ({"value": int(round(scored[1]["value"])),
                       "channels": scored[1]["channels"],
                       "score": round(scored[1]["score"], 3)} if len(scored) > 1 else None),
        "n_candidates": len(scored),
    }
    return [str(int(round(out)))], diag


def predict_set_consensus(per_channel: dict[str, list[str]], tau: float,
                          channel_agreement_weight: float = 1.0) -> list[str]:
    """Cross-frame version of vote-share thresholding for set relations.

    Same idea: a candidate is scored by its mean vote share across frames,
    bonused for appearing in more than one frame. Abstention remains an outcome
    of the threshold rather than a separate gate.
    """
    agg: dict[str, list[float]] = defaultdict(list)
    surface: dict[str, Counter] = defaultdict(Counter)
    for ch, draws in per_channel.items():
        for key, (share, surf) in vote_shares(draws, ch).items():
            agg[key].append(share)
            surface[key][surf] += 1
    n_ch = max(len(per_channel), 1)
    keep = []
    for key, shares in agg.items():
        breadth = 1.0 + channel_agreement_weight * (len(shares) - 1) / max(n_ch - 1, 1)
        score = (sum(shares) / n_ch) * breadth
        if score >= tau:
            keep.append((score, surface[key].most_common(1)[0][0]))
    keep.sort(reverse=True)
    return [s for _, s in keep]


# ------------------------------------------------------------------ prediction file


def write_predictions(path: str, rows: list[dict], preds: dict[tuple[str, str], list[str]]) -> dict:
    """rows = the gold-side rows (subject/relation universe). Any row absent
    from preds is written explicitly empty rather than omitted."""
    n_empty = 0
    with open(path, "w") as fh:
        for r in rows:
            key = (r["SubjectEntity"], r["Relation"])
            objs = [str(o) for o in preds.get(key, [])]
            if not objs:
                n_empty += 1
            fh.write(json.dumps(
                {"SubjectEntity": r["SubjectEntity"], "Relation": r["Relation"],
                 "ObjectEntities": objs}, ensure_ascii=False) + "\n")
    return {"rows": len(rows), "empty_rows": n_empty}
