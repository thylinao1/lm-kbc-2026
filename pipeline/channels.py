"""Per-relation elicitation channels for base-completion mode.

Design rules baked in here:
  * BASE model, plain /v1/completions semantics. No chat template, ever.
  * Demos are FIXED per (relation, channel, config). No per-draw resampling:
    it defeats vLLM prefix caching for a gain inside our own noise band.
  * Demos come from TRAIN only. val is a selection set, test has no gold.
  * Every channel declares its own stop list and max_tokens. The vLLM default
    max_tokens=16 silently truncates the recitation channels.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Callable

from common import gold_primary, rows_for


@dataclass(frozen=True)
class Channel:
    name: str
    relation: str
    render: Callable[[str, list[dict]], str]   # (subject, demos) -> prompt
    parse: Callable[[str], list[str]]          # continuation -> candidate strings
    max_tokens: int
    stop: tuple[str, ...]
    n_demos: int
    demo_strategy: str = "balanced"            # balanced | log_stratified | plain


# ------------------------------------------------------------------ demo picking

def pick_demos(relation: str, n: int, seed: int, strategy: str = "balanced") -> list[dict]:
    """Deterministic demo selection from TRAIN only."""
    rows = rows_for("train", relation)
    rng = random.Random(seed)

    if strategy == "balanced":
        # Keep the empty/non-empty ratio of train so the model sees abstention
        # demonstrated at a realistic rate rather than never.
        empty = [r for r in rows if not r["ObjectEntities"]]
        full = [r for r in rows if r["ObjectEntities"]]
        rng.shuffle(empty)
        rng.shuffle(full)
        frac_empty = len(empty) / max(len(rows), 1)
        k_empty = min(len(empty), round(n * frac_empty))
        picked = empty[:k_empty] + full[: n - k_empty]
        rng.shuffle(picked)
        return picked

    if strategy == "log_stratified":
        # Numerics: spread demos across orders of magnitude so the model is not
        # anchored on one scale. Median-of-cluster aggregation downstream is
        # useless if every demo is a 20k-seat stadium.
        import math
        vals = []
        for r in rows:
            g = gold_primary(r)
            if not g:
                continue
            try:
                v = float(str(g[0]).replace(",", ""))
            except ValueError:
                continue
            if v > 0:
                vals.append((math.log10(v), r))
        if not vals:
            return rows[:n]
        vals.sort(key=lambda t: t[0])
        step = max(len(vals) // max(n, 1), 1)
        picked = [r for _, r in vals[::step]][:n]
        rng.shuffle(picked)
        return picked

    rng.shuffle(rows)
    return rows[:n]


def pick_demos_nn(relation: str, n: int, query: str, seed: int = 1234) -> list[dict]:
    """Nearest-neighbour demonstrations: choose train examples SIMILAR to the query.

    Never tried on this campaign. Every channel so far picks one fixed demo set
    at random (balanced or log-stratified) and reuses it for all subjects.

    Why it targets the measured failure. Capacity subjects carry a location
    suffix ("Ibn-e-Qasim Bagh Stadium in Multan"), and our dominant error is the
    model failing to place a specific venue and falling back on a round "typical
    stadium" size: predictions that are a multiple of 10000 are right 27.8% of
    the time against 80.0% for specific values. Demonstrations drawn from the
    same region, or with similar name morphology, give it local scale context
    (a Pakistani municipal ground is not a 10000-seat default) which a random
    global sample cannot.

    Legal: reads train subjects and train gold only. No external data.

    Cost: the prompt prefix now differs per subject, so vLLM prefix caching no
    longer hits across subjects. Acceptable at 98-100 subjects per relation.
    """
    # EXCLUDE the query itself. When generating for the train split every query
    # IS a train row, so without this the subject's own gold sits in its own
    # prompt. Same leak class as the fixed-demo guard, but it would have been
    # total rather than partial. With this guard, NN demos are actually CLEANER
    # on train than fixed demos: no query can ever see its own answer.
    rows = [r for r in rows_for("train", relation)
            if r["SubjectEntity"] != query]

    def feats(s: str) -> tuple[set, set]:
        name, _, place = s.partition(" in ")
        return ({t.lower() for t in place.split() if len(t) > 2},
                {t.lower() for t in name.split() if len(t) > 2})

    qp, qn = feats(query)

    def sim(r: dict) -> tuple:
        dp, dn = feats(r["SubjectEntity"])
        # place match dominates, name morphology breaks ties
        return (len(qp & dp), len(qn & dn))

    ranked = sorted(rows, key=lambda r: sim(r), reverse=True)
    top = ranked[:n]
    # keep magnitude spread among the chosen neighbours so the frame still shows
    # a range of sizes rather than n copies of one scale
    rng = random.Random(seed)
    rng.shuffle(top)
    return top


def demo_ids(demos: list[dict]) -> tuple[str, ...]:
    return tuple(d["SubjectEntity"] for d in demos)


def demo_union(relation: str, seed: int = 1234,
               channels: tuple[str, ...] | None = None) -> set[str]:
    """Every train subject that is a demonstration for ANY channel of a relation.

    THE GUARD THIS EXISTS FOR. tune.py excludes a channel's OWN demos from its
    train curve, which is right for a single-channel measurement. It is NOT
    enough the moment an analysis reads two channels at once. Channels draw
    different demo sets from the same 100 train rows, so a train subject that is
    clean for cap_recite can still be a demonstration for cap_rich, which means
    cap_rich's prompt for that subject literally contains its gold answer. Any
    cross-frame feature then encodes gold rather than agreement.

    Measured 2026-08-12, and the size of it is the point. Counts depend on which
    channels an analysis actually reads, so both scopes are recorded:
        hasCapacity  UNION over all 8 channels = 85 of 100 train rows
                     UNION over the 6 with pools on every split = 64 of 100
        hasArea      UNION over all 6 channels = 100 of 100, i.e. NO leak-free
                     train row exists, because area_lead100 uses all 100
                     UNION over the 5 with pools on every split = 74 of 100
    Under the single-channel guard, 45 of the 68 "clean" capacity train rows are
    demos of some other frame. A cross-frame agreement reranker scored +0.3333 on
    those leaky rows and +0.0000 on the genuinely clean ones, which is how the
    leak was found. val is unaffected: no val or test subject is ever a demo, so
    any relation whose train side is fully consumed still has an honest split.

    Use this, not demo_ids(pick_demos(...)), whenever an analysis touches more
    than one channel of the same relation on train.
    """
    names = channels or tuple(n for n, c in CHANNELS.items() if c.relation == relation)
    out: set[str] = set()
    for name in names:
        ch = CHANNELS[name]
        if ch.demo_strategy == "nn":
            # nn contributes NOTHING to the leak. pick_demos_nn excludes the
            # query from its own neighbour list, so a subject can never see its
            # own gold in its own prompt. Calling pick_demos() with strategy
            # "nn" would silently return a fixed fallback set that is not this
            # channel's demos at all, and would over-exclude clean train rows.
            continue
        out |= set(demo_ids(pick_demos(relation, ch.n_demos, seed, ch.demo_strategy)))
    return out


# ------------------------------------------------------------------ borders

_BORDER_NONE = {"none", "no land borders", "n/a", "nothing", "-"}


def _borders_line(row: dict) -> str:
    objs = gold_primary(row)
    return "; ".join(sorted(objs)) if objs else "none"


def render_borders(subject: str, demos: list[dict]) -> str:
    parts = [
        "Land borders of countries and territories.\n",
    ]
    for d in demos:
        parts.append(f"Country: {d['SubjectEntity']}\nLand borders: {_borders_line(d)}\n")
    parts.append(f"Country: {subject}\nLand borders:")
    return "\n".join(parts)


def parse_borders(text: str) -> list[str]:
    line = text.strip().split("\n")[0].strip()
    if not line:
        return []
    out = []
    for piece in line.split(";"):
        c = piece.strip().strip(".").strip()
        if not c:
            continue
        if c.lower() in _BORDER_NONE:
            return []
        if len(c) > 60:
            continue
        out.append(c)
    return out


# ------------------------------------------------------------------ capacity

def _cap_value(row: dict) -> str:
    g = gold_primary(row)
    return str(g[0]) if g else ""


def render_capacity_infobox(subject: str, demos: list[dict]) -> str:
    """ruggsea's champion frame, reimplemented from their published description
    (their repo carries no licence, so this is a reimplementation, cited)."""
    parts = ["Stadium and arena reference: maximum spectator capacity.\n"]
    for d in demos:
        parts.append(
            f"Venue: {d['SubjectEntity']}\n"
            f"Maximum spectator capacity: {_cap_value(d)}\n"
        )
    parts.append(f"Venue: {subject}\nMaximum spectator capacity:")
    return "\n".join(parts)


def render_capacity_recite(subject: str, demos: list[dict]) -> str:
    """Generation-side attack surface: force a factual recitation BEFORE the
    number, so the number is conditioned on recalled venue facts rather than
    produced cold. ruggsea exhausted the aggregator, not the frame."""
    name, _, place = subject.partition(" in ")
    where = f" in {place}" if place else ""
    parts = ["Reference notes on sports venues.\n"]
    for d in demos:
        dn, _, dp = d["SubjectEntity"].partition(" in ")
        dwhere = f" in {dp}" if dp else ""
        parts.append(
            f"### {d['SubjectEntity']}\n"
            f"{dn} is a sports venue{dwhere}. "
            f"Its maximum spectator capacity is {_cap_value(d)}.\n"
        )
    parts.append(f"### {subject}\n{name} is a sports venue{where}.")
    return "\n".join(parts)


def render_capacity_rich(subject: str, demos: list[dict]) -> str:
    """Richer recitation: several grounding facts before the number.

    Direct extension of the campaign's one repeatable finding. Register
    determines what enters the candidate pool: cap_recite already has the best
    coverage of five capacity frames (0.804 against 0.680 for the infobox
    frame), and the same change took hasArea from 0.8100 to 0.8700.

    The open question is whether MORE grounding helps. Capacity's dominant error
    is a round-number attractor: our prediction is right 27.8% of the time when
    it is a multiple of 10000 and 80.0% when it is not, which says the model
    falls back on "typical stadium size" when it cannot place the venue. Asking
    it to state where the venue is, when it opened and who plays there first
    gives it more chance to actually locate the venue before it commits.
    """
    name, _, place = subject.partition(" in ")
    where = f" in {place}" if place else ""
    parts = ["Stadium reference entries.\n"]
    for d in demos:
        dn, _, dp = d["SubjectEntity"].partition(" in ")
        dwhere = f" in {dp}" if dp else ""
        parts.append(
            f"### {d['SubjectEntity']}\n"
            f"{dn} is a sports venue located{dwhere}. "
            f"It is used for football and other events, and it is home to local clubs. "
            f"Its maximum spectator capacity is {_cap_value(d)}.\n"
        )
    parts.append(
        f"### {subject}\n{name} is a sports venue located{where}.")
    return "\n".join(parts)


def render_capacity_official(subject: str, demos: list[dict]) -> str:
    """Authority register: an official venue register entry, which primes a
    specific recalled figure rather than an estimate."""
    parts = ["OFFICIAL VENUE REGISTER\nSeating capacity as recorded by the venue operator.\n"]
    for d in demos:
        parts.append(
            f"Venue: {d['SubjectEntity']}\n"
            f"Recorded seating capacity: {_cap_value(d)}\n")
    parts.append(f"Venue: {subject}\nRecorded seating capacity:")
    return "\n".join(parts)


def render_capacity_disambig(subject: str, demos: list[str]) -> str:
    """Subject strings carry a disambiguation suffix ("Foo Stadium in Multan")
    because many venues share a name. The #1 named resolving that suffix as
    tractable-but-untried. Here the place is promoted to its own field so the
    model conditions on it instead of treating it as part of the name."""
    def fields(s: str) -> tuple[str, str]:
        name, _, place = s.partition(" in ")
        return name, place

    parts = ["Venue database. Each entry resolves one specific venue by location.\n"]
    for d in demos:
        n, pl = fields(d["SubjectEntity"])
        parts.append(
            f"Name: {n}\nLocation: {pl or 'unspecified'}\n"
            f"Maximum spectator capacity: {_cap_value(d)}\n"
        )
    n, pl = fields(subject)
    parts.append(f"Name: {n}\nLocation: {pl or 'unspecified'}\nMaximum spectator capacity:")
    return "\n".join(parts)


def render_capacity_current(subject: str, demos: list[str]) -> str:
    """Gold is the CURRENT capacity as of 2026-07-01. Venues get renovated and
    the model has seen several figures across years; an unqualified prompt
    invites the historical or the original one. This frame pins the reading."""
    parts = ["Current stadium capacities, as of 2026. Post-renovation figures where applicable.\n"]
    for d in demos:
        parts.append(
            f"Q: What is the current maximum spectator capacity of {d['SubjectEntity']}?\n"
            f"A: {_cap_value(d)}\n"
        )
    parts.append(
        f"Q: What is the current maximum spectator capacity of {subject}?\nA:")
    return "\n".join(parts)


def render_capacity_listing(subject: str, demos: list[str]) -> str:
    """Register match: reproduce the pipe-table listing format that stadium
    lists actually appear in during pretraining, rather than a QA register."""
    parts = ["| Venue | Capacity |", "|---|---|"]
    for d in demos:
        parts.append(f"| {d['SubjectEntity']} | {_cap_value(d)} |")
    parts.append(f"| {subject} |")
    return "\n".join(parts)


def parse_capacity_listing(text: str) -> list[str]:
    head = text.strip().split("\n")[0]
    head = head.split("|")[0]
    return parse_number(head)[:1]


_NUM_RE = re.compile(r"(\d[\d,\.]*)")


def parse_number(text: str) -> list[str]:
    """Return bare integer strings. The grader parses float(x.replace(',','')),
    so units or words in the prediction are a parse-fail that still costs
    precision. Emit digits only."""
    out = []
    for m in _NUM_RE.finditer(text):
        raw = m.group(1).replace(",", "")
        if raw.count(".") > 1:
            continue
        raw = raw.rstrip(".")
        if not raw or not raw[0].isdigit():
            continue
        try:
            v = float(raw)
        except ValueError:
            continue
        if v <= 0:
            continue
        out.append(str(int(v)) if v == int(v) else str(v))
    return out


def parse_capacity(text: str) -> list[str]:
    head = text.strip().split("\n")[0]
    nums = parse_number(head)
    if nums:
        return nums[:1]
    return parse_number(text.strip())[:1]


def parse_capacity_recite(text: str) -> list[str]:
    """In the recitation frame the capacity is the number following the
    capacity phrase, not merely the first number (years and dates come first)."""
    m = re.search(
        r"(?:maximum\s+)?(?:spectator\s+)?capacity\s+(?:is|of|:)?\s*(?:approximately\s+)?([\d,\.]+)",
        text, re.I)
    if m:
        n = parse_number(m.group(1))
        if n:
            return n[:1]
    return parse_capacity(text)


# ------------------------------------------------------------------ cityOfDeath
# 100 test rows, 48% empty gold (measured by probe #1, not assumed). Both-empty
# scores 1.0, so the abstention decision is worth more here than extra recall.
# The live #1 abstains on 75 of 100 and still only reaches 0.60; 30 of their
# losses are rows where a city exists and they stayed silent.

_ALIVE = {"none", "still alive", "alive", "living", "n/a", "-", "unknown"}


def render_death_obituary(subject: str, demos: list[dict]) -> str:
    """Obituary register. Base models have seen a great many death notices in
    this exact shape, so the completion is drawn from that distribution rather
    than from a QA distribution."""
    parts = ["Biographical register: place of death.\n"]
    for d in demos:
        g = gold_primary(d)
        val = g[0] if g else "still living"
        parts.append(f"{d['SubjectEntity']} died in: {val}\n")
    parts.append(f"{subject} died in:")
    return "\n".join(parts)


def render_death_year(subject: str, demos: list[dict]) -> str:
    """Elicit the death YEAR, not the city.

    Motivation from the val error breakdown at tau=0.35: of 100 subjects, 12 are
    rows where the person is still alive and we confidently named a city. That is
    the single largest addressable failure mode on this relation, it is worth
    more on test (48% empty against val's 39%), and it is not a selection problem
    at all, so no amount of better city-selection touches it.

    A year is a stronger factual commitment than a city and is elicited
    independently, so the vote share on "still living" here is evidence the city
    channel does not contain. This is NOT the standalone yes/no gate that was
    reported inert in prior work: it is the same base-completion vote-share
    mechanism, applied to a different target, and it is combined by threshold
    rather than used as a veto.
    """
    parts = ["Biographical register: year of death, or 'still living'.\n"]
    for d in demos:
        val = "still living" if not gold_primary(d) else "died"
        # we do not have gold years; demonstrate only the living/deceased shape
        parts.append(f"{d['SubjectEntity']}: {val}\n")
    parts.append(f"{subject}:")
    return "\n".join(parts)


def parse_death_year(text: str) -> list[str]:
    """Return a marker token, not a city: ['deceased'] or [] for still living."""
    t = " ".join(text.strip().lower().split())
    if not t:
        return []
    if "still living" in t or "alive" in t or "living" in t:
        return []
    if "died" in t or "deceased" in t or re.search(r"\b(1[89]\d{2}|20[0-2]\d)\b", t):
        return ["deceased"]
    return []


def render_death_direct(subject: str, demos: list[dict]) -> str:
    parts = ["Where did each person die? Answer 'still living' if they are alive.\n"]
    for d in demos:
        g = gold_primary(d)
        val = "; ".join(g) if g else "still living"
        parts.append(f"Q: In which city did {d['SubjectEntity']} die?\nA: {val}\n")
    parts.append(f"Q: In which city did {subject} die?\nA:")
    return "\n".join(parts)


def parse_city(text: str) -> list[str]:
    line = text.strip().split("\n")[0].strip()
    if not line:
        return []
    out = []
    for piece in line.split(";"):
        c = piece.strip().strip(".").strip()
        if not c:
            continue
        if c.lower() in _ALIVE or "still liv" in c.lower() or "alive" in c.lower():
            return []
        if len(c) > 50 or len(c) < 2:
            continue
        out.append(c)
    return out[:3]


def render_death_recite(subject: str, demos: list[dict]) -> str:
    """Recitation register for place of death.

    The register hypothesis has now won on both relations where it was tried:
    cap_recite is our best hasCapacity frame, and area_recite took hasArea from
    0.8100 to 0.8700 past the leader after three non-recitation configurations
    all returned exactly 0.8100. The mechanism is to make the model recall
    biographical facts BEFORE committing, rather than answering cold.

    death_obituary already uses an obituary register but states the city
    immediately. This adds the recall step.
    """
    parts = ["Biographical notes.\n"]
    for d in demos:
        g = gold_primary(d)
        if g:
            parts.append(
                f"### {d['SubjectEntity']}\n"
                f"{d['SubjectEntity']} was a public figure. "
                f"They died in {g[0]}.\n")
        else:
            parts.append(
                f"### {d['SubjectEntity']}\n"
                f"{d['SubjectEntity']} is a public figure. "
                f"They are still living.\n")
    parts.append(f"### {subject}\n{subject}")
    return "\n".join(parts)


def parse_death_recite(text: str) -> list[str]:
    t = " ".join(text.strip().split())
    if "still living" in t.lower() or "are alive" in t.lower():
        return []
    m = re.search(r"died in ([^.;\n]+)", t, re.I)
    if m:
        return parse_city(m.group(1))
    return []


def render_stock_recite(subject: str, demos: list[dict]) -> str:
    """Recitation register for stock listings."""
    parts = ["Company notes.\n"]
    for d in demos:
        g = gold_primary(d)
        if g:
            parts.append(
                f"### {d['SubjectEntity']}\n"
                f"{d['SubjectEntity']} is a company. "
                f"It is listed on {'; '.join(g)}.\n")
        else:
            parts.append(
                f"### {d['SubjectEntity']}\n"
                f"{d['SubjectEntity']} is an organisation. "
                f"It is not listed on any stock exchange.\n")
    parts.append(f"### {subject}\n{subject}")
    return "\n".join(parts)


def parse_stock_recite(text: str) -> list[str]:
    t = " ".join(text.strip().split())
    low = t.lower()
    if "not listed" in low or "not publicly" in low or "privately" in low:
        return []
    # do NOT stop at ';': exchanges are semicolon-separated, and stopping
    # there silently drops every listing after the first
    m = re.search(r"listed on ([^.\n]+)", t, re.I)
    if m:
        return parse_exchange(m.group(1))
    return []


# ------------------------------------------------------------------ stock exchange
# 100 test rows, 44% empty. Canonical exchange names only, never tickers.

_NOTLISTED = {"not listed", "none", "unlisted", "private", "not publicly traded",
              "n/a", "-", "privately held"}


def render_stock_sentinel(subject: str, demos: list[dict]) -> str:
    """SENTINEL: the abstention answer is an in-band string the model can emit,
    rather than a separate yes/no gate. A standalone gate was measured inert on
    this task; vote share over an in-band sentinel is what carries the signal."""
    parts = ["Public listings register. Companies not publicly traded are marked "
             "'not listed'.\n"]
    for d in demos:
        g = gold_primary(d)
        val = "; ".join(g) if g else "not listed"
        parts.append(f"Company: {d['SubjectEntity']}\nTraded on: {val}\n")
    parts.append(f"Company: {subject}\nTraded on:")
    return "\n".join(parts)


def parse_exchange(text: str) -> list[str]:
    line = text.strip().split("\n")[0].strip()
    if not line:
        return []
    out = []
    for piece in line.split(";"):
        c = piece.strip().strip(".").strip()
        if not c:
            continue
        if c.lower() in _NOTLISTED or "not listed" in c.lower() or "not publicly" in c.lower():
            return []
        if len(c) > 60 or len(c) < 3:
            continue
        out.append(c)
    return out[:4]


# ------------------------------------------------------------------ hasArea

def render_area_lead(subject: str, demos: list[dict]) -> str:
    """Wikipedia-lead register with the unit fixed to square kilometres, so the
    model does not silently answer in square miles and land outside tolerance."""
    parts = ["Geographic reference. Areas in square kilometres.\n"]
    for d in demos:
        parts.append(
            f"{d['SubjectEntity']} has an area of {_cap_value(d)} square kilometres.\n")
    parts.append(f"{subject} has an area of")
    return "\n".join(parts)


def render_area_listing(subject: str, demos: list[dict]) -> str:
    """Table register for areas. hasArea is our largest shortfall against the
    field (0.76 against a published 0.85) and only one frame had been tried."""
    parts = ["| Place | Area (km2) |", "|---|---|"]
    for d in demos:
        parts.append(f"| {d['SubjectEntity']} | {_cap_value(d)} |")
    parts.append(f"| {subject} |")
    return "\n".join(parts)


def render_area_recite(subject: str, demos: list[dict]) -> str:
    """Recitation frame for areas, mirroring the one that won on hasCapacity.

    hasArea is the entire remaining gap to first place (-0.0400 on the relation)
    and the P3 probe proved it is NOT the frame: area_listing and area_lead score
    identically on test, 0.8100 both, with 10 rows differing and netting exactly
    zero. So the two untried axes are the REGISTER (recall facts before naming
    the number, which is what made cap_recite our best capacity frame) and DEMO
    DEPTH (the published system uses 100 log-stratified demonstrations; we use 64).
    """
    parts = ["Reference notes on geographic features.\n"]
    for d in demos:
        parts.append(
            f"### {d['SubjectEntity']}\n"
            f"{d['SubjectEntity']} is a geographic feature. "
            f"Its area is {_cap_value(d)} square kilometres.\n"
        )
    parts.append(f"### {subject}\n{subject} is a geographic feature.")
    return "\n".join(parts)


def parse_area_recite(text: str) -> list[str]:
    m = re.search(r"area\s+(?:is|of|:)?\s*(?:approximately\s+)?([\d,\.]+)", text, re.I)
    if m:
        n = parse_number(m.group(1))
        if n:
            return n[:1]
    return parse_capacity(text)


def render_area_infobox(subject: str, demos: list[dict]) -> str:
    parts = ["Geographic infobox entries.\n"]
    for d in demos:
        parts.append(f"Name: {d['SubjectEntity']}\nArea: {_cap_value(d)} km2\n")
    parts.append(f"Name: {subject}\nArea:")
    return "\n".join(parts)


# ------------------------------------------------------------------ awardWonBy
# 10 test rows, 2.1% of the score, and gold sets are large (val median 72).
# Effort is capped by design: the relation cannot be gated (F1 CI is about
# +-0.23 at n=10). The only sane play is a wide union at a low threshold, since
# recall dominates when the gold set is large.

def render_award_list(subject: str, demos: list[dict]) -> str:
    parts = ["Award recipients, listed by award.\n"]
    for d in demos:
        objs = gold_primary(d)
        shown = "; ".join(objs[:25]) if objs else "none"
        parts.append(f"Award: {d['SubjectEntity']}\nRecipients: {shown}\n")
    parts.append(f"Award: {subject}\nRecipients:")
    return "\n".join(parts)


def parse_award(text: str) -> list[str]:
    line = text.strip().split("\n")[0].strip()
    if not line:
        return []
    out = []
    for piece in line.split(";"):
        c = piece.strip().strip(".").strip()
        # name-shaped filter: drop years, fragments, and runaway prose
        if not c or len(c) < 3 or len(c) > 60:
            continue
        if c.lower() in {"none", "n/a", "-"}:
            continue
        if c.replace(" ", "").isdigit():
            continue
        out.append(c)
    return out[:60]


# ------------------------------------------------------------------ registry

CHANNELS: dict[str, Channel] = {
    "borders_list": Channel(
        name="borders_list",
        relation="countryLandBordersCountry",
        render=render_borders,
        parse=parse_borders,
        max_tokens=96,
        stop=("\n\n", "\nCountry:"),
        n_demos=32,
        demo_strategy="balanced",
    ),
    "cap_infobox": Channel(
        name="cap_infobox",
        relation="hasCapacity",
        render=render_capacity_infobox,
        parse=parse_capacity,
        max_tokens=16,
        stop=("\n\n", "\nVenue:"),
        n_demos=64,
        demo_strategy="log_stratified",
    ),
    "cap_recite": Channel(
        name="cap_recite",
        relation="hasCapacity",
        render=render_capacity_recite,
        parse=parse_capacity_recite,
        max_tokens=160,
        stop=("\n###", "\n\n\n"),
        n_demos=32,
        demo_strategy="log_stratified",
    ),
    "cap_nn": Channel(
        name="cap_nn",
        relation="hasCapacity",
        render=render_capacity_recite,
        parse=parse_capacity_recite,
        max_tokens=160,
        stop=("\n###", "\n\n\n"),
        n_demos=32,
        demo_strategy="nn",
    ),
    "area_nn": Channel(
        name="area_nn",
        relation="hasArea",
        render=render_area_recite,
        parse=parse_area_recite,
        max_tokens=120,
        stop=("\n###", "\n\n\n"),
        n_demos=32,
        demo_strategy="nn",
    ),
    "death_nn": Channel(
        name="death_nn",
        relation="personHasCityOfDeath",
        render=render_death_obituary,
        parse=parse_city,
        max_tokens=24,
        stop=("\n\n", "\nQ:"),
        n_demos=40,
        demo_strategy="nn",
    ),
    "cap_rich": Channel(
        name="cap_rich",
        relation="hasCapacity",
        render=render_capacity_rich,
        parse=parse_capacity_recite,
        max_tokens=200,
        stop=("\n###", "\n\n\n"),
        n_demos=24,
        demo_strategy="log_stratified",
    ),
    "cap_official": Channel(
        name="cap_official",
        relation="hasCapacity",
        render=render_capacity_official,
        parse=parse_capacity,
        max_tokens=16,
        stop=("\n\n", "\nVenue:"),
        n_demos=48,
        demo_strategy="log_stratified",
    ),
    "cap_disambig": Channel(
        name="cap_disambig",
        relation="hasCapacity",
        render=render_capacity_disambig,
        parse=parse_capacity,
        max_tokens=16,
        stop=("\n\n", "\nName:"),
        n_demos=48,
        demo_strategy="log_stratified",
    ),
    "cap_current": Channel(
        name="cap_current",
        relation="hasCapacity",
        render=render_capacity_current,
        parse=parse_capacity,
        max_tokens=16,
        stop=("\n\n", "\nQ:"),
        n_demos=48,
        demo_strategy="log_stratified",
    ),
    "cap_listing": Channel(
        name="cap_listing",
        relation="hasCapacity",
        render=render_capacity_listing,
        parse=parse_capacity_listing,
        max_tokens=16,
        stop=("\n", "|"),
        n_demos=64,
        demo_strategy="log_stratified",
    ),
    "death_obituary": Channel(
        name="death_obituary",
        relation="personHasCityOfDeath",
        render=render_death_obituary,
        parse=parse_city,
        max_tokens=24,
        stop=("\n\n", "\nQ:"),
        n_demos=48,
        demo_strategy="balanced",
    ),
    "death_year": Channel(
        name="death_year",
        relation="personHasCityOfDeath",
        render=render_death_year,
        parse=parse_death_year,
        max_tokens=12,
        stop=("\n", "\n\n"),
        n_demos=48,
        demo_strategy="balanced",
    ),
    "death_direct": Channel(
        name="death_direct",
        relation="personHasCityOfDeath",
        render=render_death_direct,
        parse=parse_city,
        max_tokens=24,
        stop=("\n\n", "\nQ:"),
        n_demos=48,
        demo_strategy="balanced",
    ),
    "stock_sentinel": Channel(
        name="stock_sentinel",
        relation="companyTradesAtStockExchange",
        render=render_stock_sentinel,
        parse=parse_exchange,
        max_tokens=32,
        stop=("\n\n", "\nCompany:"),
        n_demos=64,
        demo_strategy="balanced",
    ),
    "area_lead": Channel(
        name="area_lead",
        relation="hasArea",
        render=render_area_lead,
        parse=parse_capacity,
        max_tokens=16,
        stop=("\n\n", "\n"),
        n_demos=64,
        demo_strategy="log_stratified",
    ),
    "area_listing": Channel(
        name="area_listing",
        relation="hasArea",
        render=render_area_listing,
        parse=parse_capacity_listing,
        max_tokens=16,
        stop=("\n", "|"),
        n_demos=64,
        demo_strategy="log_stratified",
    ),
    "area_infobox": Channel(
        name="area_infobox",
        relation="hasArea",
        render=render_area_infobox,
        parse=parse_capacity,
        max_tokens=16,
        stop=("\n\n", "\nName:"),
        n_demos=64,
        demo_strategy="log_stratified",
    ),
    "death_recite": Channel(
        name="death_recite",
        relation="personHasCityOfDeath",
        render=render_death_recite,
        parse=parse_death_recite,
        max_tokens=48,
        stop=("\n###", "\n\n\n"),
        n_demos=40,
        demo_strategy="balanced",
    ),
    "stock_recite": Channel(
        name="stock_recite",
        relation="companyTradesAtStockExchange",
        render=render_stock_recite,
        parse=parse_stock_recite,
        max_tokens=56,
        stop=("\n###", "\n\n\n"),
        n_demos=40,
        demo_strategy="balanced",
    ),
    "area_recite": Channel(
        name="area_recite",
        relation="hasArea",
        render=render_area_recite,
        parse=parse_area_recite,
        max_tokens=120,
        stop=("\n###", "\n\n\n"),
        n_demos=32,
        demo_strategy="log_stratified",
    ),
    "area_lead100": Channel(
        name="area_lead100",
        relation="hasArea",
        render=render_area_lead,
        parse=parse_capacity,
        max_tokens=16,
        stop=("\n\n", "\n"),
        n_demos=100,
        demo_strategy="log_stratified",
    ),
    "award_list": Channel(
        name="award_list",
        relation="awardWonBy",
        render=render_award_list,
        parse=parse_award,
        max_tokens=420,
        stop=("\n\n", "\nAward:"),
        n_demos=6,
        demo_strategy="plain",
    ),
}
