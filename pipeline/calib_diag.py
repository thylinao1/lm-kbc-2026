"""Why the marginal calibration q(share) is the wrong object for borders.

Prints the empirical P(candidate in gold) by exact vote share, then the same
quantity CONDITIONED on the row's own confidence profile (share of the top
candidate) and on rank. If the hit rate of a share=1/30 candidate depends
strongly on the row, a single global q(share) curve mis-prices exactly the
marginal candidates the stopping rule is deciding about.
"""
from __future__ import annotations

import sys
from collections import defaultdict

sys.path.insert(0, "/Users/maksimsilchenko/AKBC/pipeline")

from ev_prefix_rule import load_rows


def diag(rel: str, ch: str, ship_tau: float) -> None:
    rows = load_rows(rel, ch, "train") + load_rows(rel, ch, "val")
    print("=" * 84)
    print(f"{rel}: candidate-level hit rate by vote share (train+val pooled)")
    print("=" * 84)
    by_share: dict[float, list[int]] = defaultdict(list)
    for r in rows:
        for (share, _), y in zip(r["ranked"], r["labels"]):
            by_share[round(share, 4)].append(y)
    print(f"{'share':>7s} {'n_cand':>7s} {'hit rate':>9s}")
    for s in sorted(by_share):
        v = by_share[s]
        if len(v) >= 5:
            print(f"{s:7.3f} {len(v):7d} {sum(v)/len(v):9.3f}")

    print(f"\nSAME, split by the row's TOP candidate share (row confidence):")
    print(f"{'share':>7s} | " + " | ".join(f"{lab:>16s}" for lab in
                                           ("top>=0.9", "0.5<=top<0.9", "top<0.5")))
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for r in rows:
        top = r["ranked"][0][0] if r["ranked"] else 0.0
        band = "top>=0.9" if top >= 0.9 else ("0.5<=top<0.9" if top >= 0.5 else "top<0.5")
        for (share, _), y in zip(r["ranked"], r["labels"]):
            buckets[(round(share, 4), band)].append(y)
    for s in sorted(by_share):
        if len(by_share[s]) < 5:
            continue
        cells = []
        for band in ("top>=0.9", "0.5<=top<0.9", "top<0.5"):
            v = buckets.get((s, band), [])
            cells.append(f"{sum(v)/len(v):6.3f} (n={len(v):3d})" if v else f"{'-':>16s}")
        print(f"{s:7.3f} | " + " | ".join(f"{c:>16s}" for c in cells))

    print(f"\nMARGINAL CANDIDATES ONLY (share < shipped tau {ship_tau}), by rank:")
    byrank: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        for i, ((share, _), y) in enumerate(zip(r["ranked"], r["labels"])):
            if share < ship_tau:
                byrank[min(i, 12)].append(y)
    for k in sorted(byrank):
        v = byrank[k]
        print(f"   rank {k:2d}{'+' if k == 12 else ' '}: n={len(v):4d}  hit={sum(v)/len(v):.3f}")

    n_below = sum(1 for r in rows for share, _ in r["ranked"] if share < ship_tau)
    n_hit = sum(y for r in rows for (share, _), y in zip(r["ranked"], r["labels"])
                if share < ship_tau)
    print(f"\noverall: {n_hit}/{n_below} = {n_hit/max(n_below,1):.3f} of the candidates the "
          f"shipped tau DISCARDS are actually in gold")


if __name__ == "__main__":
    diag("countryLandBordersCountry", "borders_list", 0.15)
    print()
    diag("awardWonBy", "award_list", 0.10)
