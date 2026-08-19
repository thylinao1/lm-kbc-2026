"""State of play: where every relation stands and what it implies.

Reads whatever result files exist and reports two things: given what I have
measured so far, what I would score, and how far that is from the bar.

Projection rule, deliberately conservative:
  * a relation with a measured val result is projected at that value
  * a relation with no result yet is projected at its ALL-EMPTY value, which is
    its empty-gold fraction on test, because that is what an unfinished ladder
    rung actually submits
so the projection is always a floor, never a hope.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import REPO, TEST_EMPTY_PRIOR, TEST_ROWS, TOTAL_TEST_ROWS

BAR = 0.6961          # live #1 on the test board
FIELD_ENVELOPE = 0.7015   # weighted sum of every per-relation test best

# Best per-relation figure anyone has posted on the TEST phase (first-party
# scrape, 2026-08-11). Used only for context, never as my own number.
TEST_FRONTIER = {
    "countryLandBordersCountry": (0.9755, "quangtran276"),
    "personHasCityOfDeath": (0.6000, "ruggsea"),
    "companyTradesAtStockExchange": (0.8717, "cedarz"),
    "hasArea": (0.8500, "ruggsea"),
    "hasCapacity": (0.3265, "ruggsea"),
    "awardWonBy": (0.3691, "yamm1212"),
}

SHORT = {
    "countryLandBordersCountry": "borders",
    "personHasCityOfDeath": "cityDeath",
    "companyTradesAtStockExchange": "stockX",
    "hasArea": "hasArea",
    "hasCapacity": "hasCap",
    "awardWonBy": "award",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default=str(REPO / "pools"))
    args = ap.parse_args()
    pdir = Path(args.pools)

    measured: dict[str, dict] = {}

    for f in sorted(pdir.glob("tune_*.json")):
        try:
            r = json.loads(f.read_text())
        except Exception:
            continue
        v = (r.get("val_at_train_argmax") or {}).get("macro_f1")
        if v is None:
            continue
        rel = r["relation"]
        cur = measured.get(rel)
        if cur is None or v > cur["val_f1"]:
            measured[rel] = {"val_f1": v, "how": r["channel"],
                             "param": r["best_on_train"]["param"],
                             "parse_ok": r.get("parse_qa_ok", True)}

    for f in sorted(pdir.glob("consensus_*.json")):
        try:
            r = json.loads(f.read_text())
        except Exception:
            continue
        rel, va = r.get("relation"), r.get("val")
        if not va or "consensus" not in va:
            continue
        v = va["consensus"]["macro_f1"]
        cur = measured.get(rel)
        if cur is None or v > cur["val_f1"]:
            measured[rel] = {"val_f1": v, "how": f"consensus({len(r['channels'])} frames)",
                             "param": va["consensus"]["scale"], "parse_ok": True}

    # Test-prior re-estimates for the abstention relations. Validation and test
    # have materially different empty-gold rates (measured by probe #1), so the
    # raw val figure is a biased estimate of test for those three. Where a
    # prior_shift result exists I show it as the better test estimate.
    shifted: dict[str, dict] = {}
    ps = pdir / "prior_shift.json"
    if ps.exists():
        try:
            for ch, v in json.loads(ps.read_text()).items():
                rel = v["relation"]
                best = v["test_prior_optimal"]
                cur = shifted.get(rel)
                if cur is None or best["test_prior_f1"] > cur["f1"]:
                    shifted[rel] = {"f1": best["test_prior_f1"], "tau": best["tau"],
                                    "val_f1": best["val_f1"], "how": ch}
        except Exception:
            pass

    print(f"{'relation':<12} {'rows':>5} {'mine(val)':>10} {'testprior':>10} "
          f"{'proj':>7} {'frontier':>9} {'gap':>7}  source")
    print("-" * 92)
    proj_total = 0.0
    for rel in sorted(TEST_ROWS, key=lambda r: -TEST_ROWS[r]):
        w = TEST_ROWS[rel]
        floor = TEST_EMPTY_PRIOR[rel]
        m = measured.get(rel)
        sh = shifted.get(rel)
        mine = m["val_f1"] if m else None
        # best available estimate of TEST performance
        if sh is not None:
            proj = sh["f1"]
        elif mine is not None:
            proj = mine
        else:
            proj = floor
        proj_total += w * proj
        front, who = TEST_FRONTIER[rel]
        flag = "  PARSE-QA FAIL" if (m and not m.get("parse_ok", True)) else ""
        src = (f"{sh['how']} tau={sh['tau']}" if sh else (m["how"] if m else "not run yet"))
        print(f"{SHORT[rel]:<12} {w:>5} "
              f"{(f'{mine:.4f}' if mine is not None else '-'):>10} "
              f"{(f'{sh[chr(102)+chr(49)]:.4f}' if sh else '-'):>10} "
              f"{proj:>7.4f} {front:>9.4f} {proj-front:>+7.4f}  {src}{flag}")

    overall = proj_total / TOTAL_TEST_ROWS
    print("-" * 92)
    print(f"{'PROJECTED':<12} {TOTAL_TEST_ROWS:>5} {'':>10} {'':>10} {overall:>7.4f}")
    if shifted:
        print("  (testprior = val reweighted to the MEASURED test empty rate; the "
              "better estimate of test for the three abstention relations)")
    print()
    print(f"  bar to beat (live #1)        : {BAR:.4f}   -> I am {overall-BAR:+.4f}")
    print(f"  whole-field envelope         : {FIELD_ENVELOPE:.4f}   "
          f"(sum of EVERY per-relation test best; beating it needs net-new capability)")
    print(f"  all-empty floor              : 0.2147")

    cap = measured.get("hasCapacity")
    if cap:
        c = cap["val_f1"]
        print()
        print("  hasCapacity sensitivity (the target relation):")
        for t in (BAR, 0.71, 0.72):
            others = proj_total - TEST_ROWS["hasCapacity"] * c
            need = (t * TOTAL_TEST_ROWS - others) / TEST_ROWS["hasCapacity"]
            print(f"    to reach {t:.4f} overall, hasCapacity must be {need:.4f} "
                  f"(now {c:.4f}, field best 0.3265)")

    missing = [SHORT[r] for r in TEST_ROWS if r not in measured]
    if missing:
        print(f"\n  not yet measured (projected at their all-empty floor): {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
