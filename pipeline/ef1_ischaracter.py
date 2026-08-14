"""WHAT IS THE EV RULE, MECHANICALLY? Is it a per-row rule at all, or a
re-parameterisation of something the campaign has already bracketed?

Three tests, per relation:
  1. Is k_EV a function of K (the candidate count) ALONE? If the isotonic curve
     collapses to one block, every q_i is identical and E[F1](k) depends on
     nothing but k and K -- the "per-row expected-F1 maximiser" is then a
     lookup table on pool size, not a per-row decision.
  2. Is the EMITTED SET reproducible by a GLOBAL vote-share threshold? Search
     every tau on the observed share grid and report the best exact-match rate.
     If some tau reproduces the rule on every row, the lever lives inside the
     tau family the board has already bracketed on both sides.
  3. Does the val tau-gradient agree with the one bracketed TEST measurement?

Nothing here writes to configs/, submissions/ or NOTES.local.md.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import expected_f1 as E


def ev_ks(rel: str, split: str, cal, size, rng, n_mc: int, rows):
    return {r["subject"]: E.choose_k(r, cal, size, rel, rng, n_mc)[0] for r in rows}


def main() -> int:
    for rel in E.RELCFG:
        cfg = E.RELCFG[rel]
        rng = np.random.default_rng(E.SEED)
        data = {sp: E.build_rows(rel, sp) for sp in ("train", "val", "test")}
        cal = E.fit_calibrator("iso", data["train"] + data["val"])
        size = E.fit_size_model(rel, data["train"] + data["val"])
        n_mc = 1200 if rel == "awardWonBy" else 4000

        print("=" * 92)
        print(f"{rel}   ({cfg['kind']})")
        print("=" * 92)
        curve = getattr(cal, "curve", [])
        qs = sorted({round(v, 4) for _, v in curve})
        print(f"  isotonic curve: {len(curve)} blocks, distinct q values {len(qs)}"
              + (f"  -> q = {qs[0]} for EVERYTHING" if len(qs) == 1 else ""))

        for split in ("val", "test"):
            rows = data[split]
            ks = ev_ks(rel, split, cal, size, rng, n_mc, rows)
            ks_ship = {r["subject"]: E.shipped_k(r, rel) for r in rows}

            # 1. k_EV as a function of K alone?
            byK: dict[int, set] = defaultdict(set)
            for r in rows:
                byK[len(r["shares"])].add(ks[r["subject"]])
            det = all(len(v) == 1 for v in byK.values())
            print(f"  [{split}] k_EV determined by candidate count K alone? "
                  f"{'YES -- not a per-row rule' if det else 'no'}  "
                  f"({sum(1 for v in byK.values() if len(v) > 1)} of {len(byK)} "
                  f"K-values map to >1 k)")

            # 2. reproducible by a global vote-share tau?
            grid = sorted({s for r in rows for s in r["shares"]} | {0.0, 1.01})
            best = (0.0, -1)
            for tau in grid:
                m = sum(1 for r in rows
                        if sum(1 for s in r["shares"] if s >= tau) == ks[r["subject"]])
                if m > best[1]:
                    best = (tau, m)
            print(f"  [{split}] best GLOBAL tau reproducing k_EV: tau={best[0]:.4f} "
                  f"matches {best[1]}/{len(rows)} rows "
                  f"({100*best[1]/len(rows):.0f}%)"
                  + ("   <-- EXACTLY a global threshold" if best[1] == len(rows) else ""))
            print(f"       shipped mean k {np.mean(list(ks_ship.values())):.2f} -> "
                  f"EV mean k {np.mean(list(ks[s] for s in ks_ship)):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
