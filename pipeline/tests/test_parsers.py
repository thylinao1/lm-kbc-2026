"""Decision-parity tests on frozen continuations.

These do not assert that a parse is semantically "right". They pin the
accept/reject DECISION for a fixed set of continuations, so that a later regex
tweak or a substrate change cannot silently move the boundary without a test
turning red. A drifting parser is invisible in the final score: every
self-check still passes and the file still has 475 rows, but the vote pools
quietly fill with garbage and every threshold tuned on them shifts.

Run:  python -m pytest pipeline/tests/ -q
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from aggregate import (log_clusters, normalize, predict_numeric,  # noqa: E402
                       predict_numeric_consensus, predict_set, vote_shares)
from channels import (parse_borders, parse_capacity, parse_capacity_listing,  # noqa: E402
                      parse_capacity_recite, parse_city, parse_exchange,
                      parse_number)


# --------------------------------------------------------------- borders

@pytest.mark.parametrize("text,expected", [
    (" Austria; Belgium; France\n\nCountry: X", ["Austria", "Belgium", "France"]),
    (" none\n", []),
    (" None.", []),
    ("  Guinea ; Liberia.", ["Guinea", "Liberia"]),
    ("\n", []),
    ("", []),
    (" no land borders", []),
    # a runaway continuation must not smuggle a sentence in as a country
    (" Spain; " + "x" * 80, ["Spain"]),
])
def test_parse_borders(text, expected):
    assert parse_borders(text) == expected


# --------------------------------------------------------------- numerics

@pytest.mark.parametrize("text,expected", [
    (" 35,000\n", ["35000"]),
    (" 35000 seats", ["35000"]),
    (" approximately 7,500.", ["7500"]),
    (" no idea", []),
    (" 0", []),                      # zero can never match: gt=0 never scores
    (" 1.5", ["1.5"]),
])
def test_parse_capacity(text, expected):
    assert parse_capacity(text) == expected


def test_recite_prefers_capacity_over_year():
    """The trap in the recitation frame: the year appears before the capacity."""
    t = ("Estadio X is a sports venue in Lima, opened in 1952 and renovated in "
         "2004. Its maximum spectator capacity is 42,000.")
    assert parse_capacity_recite(t) == ["42000"]


def test_recite_falls_back_when_phrase_absent():
    assert parse_capacity_recite(" 18000\n") == ["18000"]


def test_listing_takes_only_its_own_cell():
    assert parse_capacity_listing(" 18000 |") == ["18000"]
    assert parse_capacity_listing(" 18,000 |\n| Next Venue | 9000 |") == ["18000"]


def test_parse_number_rejects_malformed():
    assert parse_number("1.2.3") == []
    assert parse_number("abc") == []


# --------------------------------------------------------------- strings

@pytest.mark.parametrize("text,expected", [
    (" Vienna\n", ["Vienna"]),
    (" still living\n", []),
    (" alive", []),
    (" Paris; Lyon.", ["Paris", "Lyon"]),
    ("", []),
])
def test_parse_city(text, expected):
    assert parse_city(text) == expected


@pytest.mark.parametrize("text,expected", [
    (" New York Stock Exchange; Nasdaq\n", ["New York Stock Exchange", "Nasdaq"]),
    (" not listed\n", []),
    (" Not publicly traded.", []),
    (" Shanghai Stock Exchange", ["Shanghai Stock Exchange"]),
])
def test_parse_exchange(text, expected):
    assert parse_exchange(text) == expected


# --------------------------------------------------------------- aggregation

def test_normalize_matches_grader_shape():
    assert normalize("Côte d'Ivoire") == normalize("Cote dIvoire")
    assert normalize("  UNITED   states ") == "united states"


def test_log_cluster_median_does_not_average_across_magnitudes():
    """The failure this guards: mean of {4900,5000,5050,5100,50000} is ~14000,
    which is wrong under any tolerance. The largest cluster's median is right."""
    draws = ["4900", "5000", "5050", "5100", "50000"]
    assert predict_numeric(draws, "cap_infobox") == ["5025"]
    cl = log_clusters([4900, 5000, 5050, 5100, 50000])
    assert len(cl) == 2 and len(cl[0]) == 4


def test_vote_share_and_tau():
    draws = ["Austria; Belgium", "Austria; Belgium; France", "Austria",
             "Austria; Belgium", "none"]
    shares = {k: round(v[0], 2) for k, v in vote_shares(draws, "borders_list").items()}
    assert shares["austria"] == 0.8 and shares["france"] == 0.2
    assert predict_set(draws, "borders_list", tau=0.5) == ["Austria", "Belgium"]
    assert predict_set(draws, "borders_list", tau=0.9) == []


def test_duplicate_predictions_do_not_inflate_share():
    """The grader dedups by normalized form; my counting must agree, so a draw
    repeating a country twice counts once."""
    shares = vote_shares(["Austria; Austria; Austria"], "borders_list")
    assert shares["austria"][0] == 1.0


def _scores_correct(pred: str, gold: float, tol: float = 0.05) -> bool:
    """The grader's rule. Asserting exact string equality on a tolerance metric
    over-specifies: 4900 against a gold of 5000 is a correct answer."""
    return abs(float(pred) - gold) / gold <= tol


def test_consensus_prefers_the_value_that_survives_register_change():
    per_ch = {
        "cap_infobox": ["20000"] * 7 + ["5000"] * 3,   # modal but wrong
        "cap_listing": ["5000"] * 4 + ["20000"] * 2,
        "cap_current": ["5050"] * 3 + ["12000"] * 2,
        "cap_disambig": ["4900"] * 3,
    }
    # the single best-sampled frame is confidently wrong
    assert predict_numeric(per_ch["cap_infobox"], "cap_infobox") == ["20000"]
    got, diag = predict_numeric_consensus(per_ch)
    assert _scores_correct(got[0], 5000.0), got
    assert diag["chosen_channels"] == 4


def test_numeric_selection_does_not_chain_across_a_wide_range():
    """Regression for a real failure. Single-linkage clustering merged
    10000/10500/12000/15000 into one group and returned 12000, a value no draw
    proposed and that no gold at either end would accept. Fixed-radius scoring
    must return something with genuine support instead."""
    draws = ["10000"] * 3 + ["10500"] * 3 + ["12000"] * 1 + ["15000"] * 3
    got = predict_numeric(draws, "cap_infobox")
    support = [w for w in (10000, 10500, 12000, 15000)
               if abs(float(got[0]) - w) / w <= 0.05]
    assert support, f"{got} has no draw within tolerance"
    assert float(got[0]) < 13000, f"{got} chained into the high group"


def test_consensus_degenerate_cases():
    assert predict_numeric_consensus({"cap_infobox": []})[0] == []
    assert predict_numeric_consensus({"cap_infobox": ["7500"] * 5})[0] == ["7500"]


# --------------------------------------------------------------- closed book

def test_load_split_refuses_anything_off_the_allowlist():
    from common import ClosedBookViolation, load_split
    with pytest.raises(ClosedBookViolation):
        load_split("dataset2025_test")
    with pytest.raises(ClosedBookViolation):
        load_split("wikidata_dump")
