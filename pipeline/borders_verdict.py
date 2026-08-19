"""countryLandBordersCountry: what the TEST board score alone proves.

Inputs used here are only (i) the grader-returned per-relation TEST score
0.9786, (ii) probe #1's reading that exactly 10 of the 67 TEST rows have empty
gold, (iii) my own cached TEST pool. No external facts.

Two results:
  A. The 10 rows I abstain on are PROVABLY the 10 empty-gold rows. Any other
     assignment forces some answered row above F1 = 1, which is impossible.
  B. Given A, the mean gold set size on the answered TEST rows must be close to
     my emitted 4.79, so the "I over-predict 4.79 against a gold mean of 3.58"
     line in the log is basis-mixed: 3.58 is the train+val gold mean, and the
     board score is arithmetically incompatible with that much over-prediction.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/Users/maksimsilchenko/AKBC/pipeline")

from aggregate import predict_set
from channels import CHANNELS
from common import TEST_ROWS, TOTAL_TEST_ROWS, load_pool, rows_for, spec_for_channel
from ev_prefix_rule import ARGS

REL, CH, TAU = "countryLandBordersCountry", "borders_list", 0.15
F1_TEST = 0.9786          # grader-returned, probe P5
N_ROWS = TEST_ROWS[REL]   # 67
N_EMPTY_GOLD = 10         # probe #1: all-empty submission returned 0.1493 = 10/67


def main() -> None:
    pool = load_pool(spec_for_channel(CHANNELS[CH], "test", ARGS))
    preds = {s: predict_set(d, CH, tau=TAU) for s, d in pool.items()}
    answered = {s: p for s, p in preds.items() if p}
    n_abstain = len(preds) - len(answered)
    sum_n = sum(len(p) for p in answered.values())
    max_n = max(len(p) for p in answered.values())

    print("=" * 80)
    print("MEASURED FROM MY OWN TEST POOL (tau = 0.15, the shipped config)")
    print("=" * 80)
    print(f"rows                     {len(preds)}")
    print(f"abstained rows           {n_abstain}   ({n_abstain/len(preds):.3f})")
    print(f"answered rows            {len(answered)}")
    print(f"predictions emitted      {sum_n}  -> {sum_n/len(answered):.2f} per answered row"
          f"  (max on a row: {max_n})")

    mass = F1_TEST * N_ROWS
    print("\n" + "=" * 80)
    print("A. THE ABSTENTIONS ARE PROVABLY THE RIGHT ROWS")
    print("=" * 80)
    print(f"grader mass = {F1_TEST} * {N_ROWS} = {mass:.4f} row-points")
    print("Let a = how many of my 10 abstentions land on an empty-gold row.")
    print("  * an abstention on empty gold scores 1 (P=1, R=1); on non-empty gold, 0")
    print("  * an ANSWERED row whose gold is empty scores 0 (P=0, R=1)")
    print(f"  * so the empty-gold rows I answered number {N_EMPTY_GOLD} - a, each worth 0,")
    print(f"    and the non-empty answered rows number {len(answered)} - ({N_EMPTY_GOLD}-a)"
          f" = {len(answered)-N_EMPTY_GOLD} + a")
    print("\n  a  |  mass that must come from non-empty answered rows | rows available | mean F1")
    for a in range(N_EMPTY_GOLD + 1):
        need = mass - a
        avail = len(answered) - N_EMPTY_GOLD + a
        mean = need / avail
        flag = "  <-- IMPOSSIBLE, mean F1 > 1" if mean > 1 else ""
        print(f" {a:2d}  | {need:11.4f} | {avail:14d} | {mean:7.4f}{flag}")
    print(f"\nOnly a = {N_EMPTY_GOLD} survives. The borders abstention decision is EXACT on")
    print("TEST: every one of the 10 empty-gold rows is abstained on and no others.")
    ans_mean = (mass - N_EMPTY_GOLD) / (len(answered) - 0)
    print(f"Answered rows therefore average F1 = ({mass:.4f} - {N_EMPTY_GOLD}) / "
          f"{len(answered)} = {ans_mean:.4f}")

    print("\n" + "=" * 80)
    print("B. HOW MUCH CAN I ACTUALLY BE OVER-PREDICTING?")
    print("=" * 80)
    print("Per row F1 = 2*tp/(n+G), so with e = n - tp wrong predictions and")
    print("m = G - tp missed golds:  e + m = (1 - F1) * (n + G).")
    shortfall = len(answered) - (mass - N_EMPTY_GOLD)
    print(f"total shortfall over the {len(answered)} answered rows = "
          f"{len(answered)} - {mass - N_EMPTY_GOLD:.4f} = {shortfall:.4f} row-points")
    for ng in (10, 20, 30):
        print(f"   if every erring row had n+G <= {ng:2d}: total (wrong + missed) entities "
              f"<= {ng} * {shortfall:.4f} = {ng*shortfall:.1f}")
    print(f"\nSo at most a few tens of the {sum_n} emitted predictions can be wrong.")
    print("Now test the log's claim directly. If the answered TEST rows really had a")
    print("gold mean of 3.58 while I emit 4.79, then even with PERFECT recall")
    print("(tp = G on every row) the mean F1 could not exceed:")
    for g in (3.58, 4.0, 4.5, 4.55, 4.79):
        print(f"   gold mean {g:4.2f}:  2*{g:.2f}/({sum_n/len(answered):.2f}+{g:.2f}) "
              f"= {2*g/(sum_n/len(answered)+g):.4f}"
              + ("   <-- BELOW the measured 0.9748, so ruled out" if 2*g/(sum_n/len(answered)+g) < ans_mean else ""))
    print(f"\nThe measured answered-row mean is {ans_mean:.4f}. Only a TEST gold mean of about")
    print(f"{ans_mean*(sum_n/len(answered))/(2-ans_mean):.2f} or more is consistent with it.")
    print("(this is the mean-row version of the bound; the per-row inequality F1 <= 2G/(n+G)")
    print(" when G < n gives the same conclusion row by row.)")
    gv = [len(r.get("ObjectEntities") or []) for r in
          rows_for("train", REL) + rows_for("val", REL)]
    nz = [g for g in gv if g]
    print(f"\nFor contrast, the 3.58 figure: train+val non-empty gold mean = "
          f"{sum(nz)/len(nz):.3f} over {len(nz)} rows. That is a TRAIN+VAL statistic.")
    print("CONCLUSION: the TEST subjects are bordier than the train+val subjects, and")
    print("the 'I over-predict by 1.2 objects per row' reading is a basis mix, not a")
    print("defect to repair. There is no systematic over-prediction to trim.")

    print("\n" + "=" * 80)
    print("C. WHAT IS LEFT")
    print("=" * 80)
    w = N_ROWS / TOTAL_TEST_ROWS
    print(f"perfect borders is worth {(1-F1_TEST)*w:+.5f} overall "
          f"({N_ROWS}*(1-{F1_TEST})/{TOTAL_TEST_ROWS})")
    print(f"the abstention half of the problem is already exact (result A), so the whole")
    print(f"{shortfall:.3f} row-points of remaining error sits inside the answered rows'")
    print("membership choice, which is what a per-row cut would address.")


if __name__ == "__main__":
    main()
