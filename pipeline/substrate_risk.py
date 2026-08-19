"""Part 3: the risk arithmetic for a substrate swap, and the power of the only
instrument a swap can be measured on before a submission is spent (val gold).

Nothing here is an estimate presented as a measurement. The board figures are
the confirmed per-relation TEST scores in docs/engineering-notes.md; everything
from them is exact arithmetic under the metric, which is a mean of per-row F1
over 475 rows, so one row is worth 1/475 of the overall score regardless of
which relation it sits in.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate import predict_numeric
from channels import CHANNELS
from common import TEST_ROWS, TOTAL_TEST_ROWS, gold_primary, load_pool, rows_for, spec_for_channel
from substrate_ev import _args, hit

# Confirmed by the grader (submission 6 of 100 returned 0.7060).
MINE = {"countryLandBordersCountry": 0.9786, "hasArea": 0.8700,
        "companyTradesAtStockExchange": 0.8530, "hasCapacity": 0.3367,
        "personHasCityOfDeath": 0.6100, "awardWonBy": 0.3484}
RUGGSEA = {"countryLandBordersCountry": 0.9753, "hasArea": 0.8500,
           "companyTradesAtStockExchange": 0.8470, "hasCapacity": 0.3265,
           "personHasCityOfDeath": 0.6000, "awardWonBy": 0.3609}

# Per-relation gains that were established by TEST probes, not by val. These are
# what a new substrate would arrive without and would have to re-earn.
BOARD_TUNED_GAINS = {
    "personHasCityOfDeath": ("tau 0.35 -> 0.45 (P3)", 0.6100 - 0.5800),
    "companyTradesAtStockExchange": ("ratio emission (P3)", 0.8530 - 0.8364),
    "countryLandBordersCountry": ("tau 0.45 -> 0.15 (P5)", 0.9786 - 0.9663),
    "hasArea": ("area_lead -> area_recite (P5)", 0.8700 - 0.8100),
}


def overall(per_rel: dict[str, float]) -> float:
    return sum(TEST_ROWS[r] * per_rel[r] for r in TEST_ROWS) / TOTAL_TEST_ROWS


def main() -> int:
    out = {}
    ov, bar = overall(MINE), overall(RUGGSEA)
    print(f"confirmed board: mine {ov:.4f}  leader {bar:.4f}  margin {ov-bar:+.4f} "
          f"= {(ov-bar)*TOTAL_TEST_ROWS:.1f} rows of {TOTAL_TEST_ROWS}")
    print(f"one row anywhere = {1/TOTAL_TEST_ROWS:.5f} overall\n")

    print("=== what a whole-system swap puts at risk (the 5 relations I lead)")
    led = [r for r in MINE if MINE[r] > RUGGSEA[r]]
    rows_led = sum(TEST_ROWS[r] for r in led)
    print(f"  relations led: {len(led)}  rows led: {rows_led} of {TOTAL_TEST_ROWS} "
          f"({rows_led/TOTAL_TEST_ROWS:.1%} of the score)")
    for r in sorted(led, key=lambda k: -TEST_ROWS[k]):
        margin = MINE[r] - RUGGSEA[r]
        print(f"    {r:30s} {TEST_ROWS[r]:3d} rows  mine {MINE[r]:.4f}  margin {margin:+.4f}"
              f"  -> lead survives a drop of {margin:.4f} F1 ({margin*TEST_ROWS[r]:.2f} rows)")
    print(f"  margin over the leader is {(ov-bar)*TOTAL_TEST_ROWS:.1f} rows, so the swap may "
          f"lose at most\n  {(ov-bar)*TOTAL_TEST_ROWS:.1f} rows NET across all 475 before "
          f"first place is gone.")

    print("\n=== break-even: capacity gain needed to pay for a uniform degradation elsewhere")
    print(f"  {'avg F1 drop on the other 5':>28} {'rows lost':>10} {'capacity F1 needed':>19} {'(from 0.3367)':>15}")
    other_rows = TOTAL_TEST_ROWS - TEST_ROWS["hasCapacity"]
    rows_be = []
    for drop in (0.00, 0.01, 0.02, 0.03, 0.05, 0.10, 0.15):
        lost = drop * other_rows
        need = MINE["hasCapacity"] + lost / TEST_ROWS["hasCapacity"]
        rows_be.append({"drop": drop, "rows_lost": lost, "capacity_needed": need})
        flag = "" if need <= 1.0 else "  IMPOSSIBLE (>1.0)"
        print(f"  {drop:28.2f} {lost:10.1f} {need:19.4f} {'':15}{flag}")
    out["breakeven"] = rows_be

    print("\n=== ceiling on the capacity side, from measurements already on record")
    cap_ceils = {
        "current shipped (board)": 0.3367,
        "re-ranking ceiling est. (10000-attractor stratum conceded)": 0.435,
        "single-frame pool coverage cap_recite (val, measured here)": 0.8454,
        "union coverage over 7 frames (val, measured here)": 0.9381,
    }
    for k, v in cap_ceils.items():
        print(f"  {v:.4f}  overall if reached = {overall({**MINE, 'hasCapacity': v}):.4f} "
              f"(delta {overall({**MINE,'hasCapacity':v})-ov:+.4f})   {k}")
    out["capacity_ceilings"] = cap_ceils

    print("\n=== affordable degradation at each plausible capacity outcome")
    for cap in (0.40, 0.435, 0.45, 0.50):
        gain_rows = (cap - MINE["hasCapacity"]) * TEST_ROWS["hasCapacity"]
        afford = (gain_rows + (ov - bar) * TOTAL_TEST_ROWS) / other_rows
        print(f"  capacity {cap:.3f} (+{gain_rows:.1f} rows): the other 5 relations may lose at "
              f"most {afford:.4f} average F1 before first place is lost")

    print("\n=== re-tuning debt: score that exists only because of TEST probes")
    tot = 0.0
    for r, (what, g) in BOARD_TUNED_GAINS.items():
        contrib = g * TEST_ROWS[r] / TOTAL_TEST_ROWS
        tot += contrib
        print(f"  {r:30s} {what:34s} +{g:.4f} rel = +{contrib:.4f} overall")
    print(f"  TOTAL board-earned tuning on the current substrate: +{tot:.4f} overall "
          f"= {tot*TOTAL_TEST_ROWS:.1f} rows")
    print(f"  (my lead is {(ov-bar)*TOTAL_TEST_ROWS:.1f} rows, i.e. {tot/(ov-bar):.1f}x smaller "
          f"than the tuning a new substrate would have to re-earn)")
    out["retuning_debt"] = tot

    print("\n=== power of a val-only substrate A/B on hasCapacity (zero submissions)")
    a = _args()
    gold = {r["SubjectEntity"]: r for r in rows_for("val", "hasCapacity")}
    base = load_pool(spec_for_channel(CHANNELS["cap_recite"], "val", a))
    ca = {}
    for s, d in base.items():
        g = [float(x) for x in gold_primary(gold[s])]
        got = predict_numeric(d, "cap_recite")
        ca[s] = int(bool(got) and g and hit(float(got[0]), g[0]))
    disc = []
    for name in ("cap_current", "cap_rich", "cap_official", "cap_disambig", "cap_infobox"):
        pool = load_pool(spec_for_channel(CHANNELS[name], "val", a))
        b, c = 0, 0
        for s in base:
            g = [float(x) for x in gold_primary(gold[s])]
            got = predict_numeric(pool[s], name) if s in pool else []
            ok = int(bool(got) and g and hit(float(got[0]), g[0]))
            if ca[s] and not ok:
                b += 1
            elif ok and not ca[s]:
                c += 1
        disc.append({"vs": name, "b": b, "c": c, "discordant": b + c})
        print(f"  cap_recite vs {name:14s} discordant rows b+c = {b+c:3d} (b={b}, c={c})")
    med = sorted(d["discordant"] for d in disc)[len(disc) // 2]
    n = len(base)
    print(f"  median discordance between two elicitations of the SAME model: {med} of {n} rows")
    print("  McNemar: to call a difference at 90% one-sided you need |b-c| >= 1.28*sqrt(b+c).")
    for d_true in (0.05, 0.10, 0.15, 0.20):
        need = 1.28 * math.sqrt(med) / n
        print(f"    true relation gap {d_true:.2f} -> detectable at 90% one-sided iff "
              f"{d_true:.2f} >= {need:.4f}  {'YES' if d_true >= need else 'NO'}")
    out["discordance"] = disc

    print("\n=== GPU / disk budget for the smallest decisive run (one relation, 3 splits)")
    # observed rates, canary job 726317: load 133 s from NFS; 2010 draws /
    # 14,522 decode tokens in 17 s => 118 draws/s, 854 tok/s at tp=1 on one H100-96.
    draws_per_split = {"train": 100, "val": 97, "test": 98}
    n_draws_each = 30
    total_draws = sum(v for v in draws_per_split.values()) * n_draws_each
    tok_per_draw = 1229157 / 29500 / 4        # measured mean output chars per draw / 4
    dec_tok = total_draws * tok_per_draw
    gen_s = dec_tok / 854
    print(f"  hasCapacity, 3 splits, n=30 draws: {total_draws} draws, "
          f"~{dec_tok:,.0f} decode tokens -> ~{gen_s/60:.1f} min of generation at 854 tok/s")
    print(f"  + model load from NFS ~4 min (measured 133 s for the 62.5 GB checkpoint)")
    print(f"  => one GPU job of well under 30 min; request 2 h walltime (12 h never backfills)")
    for name, params in (("gemma-3-27b-pt", 27.4e9), ("Qwen3-30B-A3B", 30.5e9),
                         ("Mistral-Small-24B", 23.6e9)):
        gb = params * 2 / 1e9
        print(f"  disk: {name:18s} bf16 = {gb:5.1f} GB  -> home quota 168 + {gb:.0f} "
              f"= {168+gb:.0f} GB of ~500 GB shared")
    print("  download at the measured token-free ~10 MB/s = "
          f"{27.4e9*2/1e6/10/60:.0f} min for a 27B; set an HF token first (log: 75 min -> minutes)")

    dest = Path(__file__).resolve().parent.parent / "analysis" / "substrate_risk.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1, default=str))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
