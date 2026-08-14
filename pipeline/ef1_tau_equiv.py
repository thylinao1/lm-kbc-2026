"""If the EV rule IS a global vote-share threshold (ef1_ischaracter.py says it is
on 4 of 6 relations), then its measured delta must equal the delta of that tau
move -- and the tau axis is one the board has already bracketed on both sides.

This prints, per relation, the pooled train+val macro-F1 of a GLOBAL tau sweep
scored with the official scorer through the same harness, so the EV number can
be read off the same curve.

Nothing here writes to configs/, submissions/ or NOTES.local.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import expected_f1 as E

GRIDS = {
    "countryLandBordersCountry": [0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45],
    "personHasCityOfDeath": [0.30, 0.35, 0.40, 0.4333, 0.45, 0.50, 0.55],
    "companyTradesAtStockExchange": [0.20, 0.35, 0.45, 0.50, 0.5333, 0.60, 0.70],
    "awardWonBy": [0.05, 0.08, 0.10, 0.12, 0.15],
    "hasArea": [0.20, 0.30, 0.3333, 0.40, 0.50],
    "hasCapacity": [0.05, 0.10, 0.14, 0.20, 0.30],
}


def main() -> int:
    for rel, grid in GRIDS.items():
        rows = {sp: E.build_rows(rel, sp) for sp in ("train", "val")}
        # shipped baseline through the same harness
        f_ship = []
        for sp in ("train", "val"):
            ks = {r["subject"]: E.shipped_k(r, rel) for r in rows[sp]}
            f_ship += E.score_prefixes(rows[sp], ks, rel, sp)["f1_vector"]
        base = sum(f_ship) / len(f_ship)
        print("=" * 84)
        print(f"{rel}: pooled train+val, GLOBAL tau on the same ranking "
              f"(shipped = {base:.4f}, n={len(f_ship)})")
        print("=" * 84)
        for tau in grid:
            f = []
            for sp in ("train", "val"):
                ks = {r["subject"]: sum(1 for s in r["shares"] if s >= tau)
                      for r in rows[sp]}
                f += E.score_prefixes(rows[sp], ks, rel, sp)["f1_vector"]
            m = sum(f) / len(f)
            print(f"   tau={tau:<7.4f} pooled {m:.4f}   delta vs shipped {m-base:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
