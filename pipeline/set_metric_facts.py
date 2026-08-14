"""Facts about the set-relation metric and the two written-off relations.

(a) Proves the identity  F1 = 2*tp / (n_pred + n_gold)  on every train+val row
    of both set relations (the empty/empty row is the one exception, F1=1).
    This identity is what makes an expected-value stopping rule exact rather
    than heuristic: the marginal candidate is worth adding iff its probability
    of being correct exceeds F1/2.
(b) Re-derives the overall-score value of a perfect awardWonBy and a perfect
    countryLandBordersCountry from the published per-relation TEST scores.
(c) Borders: abstention rate vs tau on TEST, preds-per-answered-row, gold size
    distribution on train+val.
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "/Users/maksimsilchenko/AKBC/pipeline")

from aggregate import predict_set
from channels import CHANNELS
from common import TEST_ROWS, TOTAL_TEST_ROWS, load_pool, rows_for, spec_for_channel
from scorer import _ev

ARGS = argparse.Namespace(
    model="google/gemma-4-31B",
    revision="5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89",
    temperature=0.7, top_p=0.95, demo_seed=1234, seed_base=7000)

SHIPPED_TEST = {
    "countryLandBordersCountry": 0.9786,
    "personHasCityOfDeath": 0.6100,
    "companyTradesAtStockExchange": 0.8530,
    "hasArea": 0.8700,
    "hasCapacity": 0.3367,
    "awardWonBy": 0.3484,
}


def part_a() -> None:
    ev = _ev()
    print("=" * 78)
    print("(a) IDENTITY CHECK   F1 == 2*tp/(n_pred+n_gold)")
    print("=" * 78)
    for rel, ch, tau in (("countryLandBordersCountry", "borders_list", 0.15),
                         ("awardWonBy", "award_list", 0.10)):
        checked = exceptions = 0
        for split in ("train", "val"):
            pool = load_pool(spec_for_channel(CHANNELS[ch], split, ARGS))
            gm = {r["SubjectEntity"]: (r.get("ObjectEntities") or [])
                  for r in rows_for(split, rel)}
            for s, draws in pool.items():
                preds = predict_set(draws, ch, tau=tau)
                seen, flat = set(), []
                for p in preds:
                    k = ev.normalize_string(p)
                    if k not in seen:
                        seen.add(k)
                        flat.append(p)
                g = gm[s]
                tp = ev.string_true_positives(flat, g)
                P = tp / len(flat) if flat else 1.0
                R = tp / len(g) if g else 1.0
                f1 = (2 * P * R / (P + R)) if (P + R) else 0.0
                if not flat and not g:
                    exceptions += 1
                    continue
                dice = 2 * tp / (len(flat) + len(g))
                assert abs(f1 - dice) < 1e-12, (rel, s, f1, dice)
                checked += 1
        print(f"{rel:30s}: identity holds on {checked} train+val rows; "
              f"{exceptions} empty-pred/empty-gold rows score 1.0 by definition")
    print("\nConsequence (right by construction): with F1 = 2*tp/(n+G), adding one more")
    print("candidate that is correct with probability q changes E[F1] from 2tp/(n+G) to")
    print("2(tp+q)/(n+1+G), which is an improvement iff  q > tp/(n+G) = F1/2.")
    print("The optimal per-candidate cutoff is therefore HALF THE ROW'S OWN F1, i.e. it")
    print("is row-dependent by construction, and no single global vote-share tau can")
    print("implement it unless every row has the same F1 and the same share->prob map.")


def part_b() -> None:
    print("\n" + "=" * 78)
    print("(b) OVERALL-SCORE VALUE OF A PERFECT RELATION (from the shipped board reading)")
    print("=" * 78)
    overall = sum(TEST_ROWS[r] * SHIPPED_TEST[r] for r in TEST_ROWS) / TOTAL_TEST_ROWS
    print(f"reassembled overall from the six published per-relation TEST scores: {overall:.4f}")
    for rel in ("awardWonBy", "countryLandBordersCountry"):
        n, f = TEST_ROWS[rel], SHIPPED_TEST[rel]
        print(f"{rel:28s} rows={n:3d} f1={f:.4f}  contributes {n*f/TOTAL_TEST_ROWS:.4f}"
              f"  perfect would contribute {n/TOTAL_TEST_ROWS:.4f}"
              f"  headroom {n*(1-f)/TOTAL_TEST_ROWS:+.4f}")
        print(f"{'':28s} value of +0.01 on this relation alone: "
              f"{n*0.01/TOTAL_TEST_ROWS:+.5f} overall")


def part_c() -> None:
    ch, rel = "borders_list", "countryLandBordersCountry"
    print("\n" + "=" * 78)
    print("(c) BORDERS: abstention and set size vs tau")
    print("=" * 78)
    pools = {sp: load_pool(spec_for_channel(CHANNELS[ch], sp, ARGS))
             for sp in ("train", "val", "test")}
    print(f"{'tau':>6s} | " + " | ".join(
        f"{sp:>5s} abst  n/ansrow  rows-vs-.15" for sp in ("train", "val", "test")))
    base = {sp: {s: predict_set(d, ch, tau=0.15) for s, d in pools[sp].items()}
            for sp in pools}
    for tau in (0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60):
        cells = []
        for sp in ("train", "val", "test"):
            cur = {s: predict_set(d, ch, tau=tau) for s, d in pools[sp].items()}
            nrow = len(cur)
            ans = [v for v in cur.values() if v]
            abst = 1 - len(ans) / nrow
            per = sum(len(v) for v in ans) / max(len(ans), 1)
            diff = sum(1 for s in cur if sorted(cur[s]) != sorted(base[sp][s]))
            cells.append(f"{abst:11.3f} {per:9.2f} {diff:12d}")
        print(f"{tau:6.3f} | " + " | ".join(cells))

    print("\ngold set sizes on train+val (all rows, borders):")
    from collections import Counter
    cnt = Counter()
    tot = n = 0
    for sp in ("train", "val"):
        for r in rows_for(sp, rel):
            g = len(r.get("ObjectEntities") or [])
            cnt[g] += 1
            n += 1
            if g:
                tot += g
    print("   size: count  ->", dict(sorted(cnt.items())))
    print(f"   rows={n}  empty={cnt[0]} ({cnt[0]/n:.3f})  "
          f"mean gold on NON-empty rows = {tot/(n-cnt[0]):.3f}")
    print("\ngold mean on TEST is not observable; the 3.58 figure in the log is the")
    print("train+val non-empty mean, and our TEST answered-row set size is above.")


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
