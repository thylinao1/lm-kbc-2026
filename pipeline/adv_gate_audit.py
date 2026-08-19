"""Adversarial audit of the 'split-half instability gate' proposal.

Four questions, all answered from cached draws + public gold only:

  Q1  Is split-half instability a re-parameterisation of vote frequency?
      If a pure top-share threshold reproduces the same stratification, the
      gate is not a new instrument, it is the share ratio wearing a hat.

  Q2  Does the gate survive its own free parameter? `unstable()` fixes the
      partition to draws[0::2] vs draws[1::2]. That IS a choice. Re-run with
      other partitions of the same pool and see whether the headline strata
      (rank-1 accuracy 0.189 unstable vs 0.467 stable, break-even 0.455)
      hold their sign.

  Q3  Is the 'answer is nevertheless present 0.811' figure knowledge or pool
      width? Permutation control, exactly as the coverage analysis did:
      give every subject ANOTHER subject's gold and recompute.

  Q4  What within-row AUC do TRIVIAL baselines already reach on the gated
      candidates? The proposal's pre-committed ship gate is 'verifier AUC >
      0.60'. If frequency itself, or 'prefer the larger number', already
      clears 0.60 on that same candidate set, the gate does not test skill.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate import numeric_candidates, predict_numeric, tolerance_support
from channels import CHANNELS
from common import load_pool, spec_for_channel
from rescore_ceiling import golds_for, hit, ranked_candidates

TOL = 0.05


def _sel(draws, channel):
    p = predict_numeric(draws, channel)
    return p[0] if p else None


def unstable_by(draws, channel, mode, rng=None):
    """Alternative partitions of the same pool. mode fixes the split rule."""
    n = len(draws)
    if mode == "stride":            # the proposal's choice
        a, b = draws[0::2], draws[1::2]
    elif mode == "block":           # first half vs second half
        a, b = draws[: n // 2], draws[n // 2:]
    elif mode == "stride3":         # 1/3 vs 1/3, same spirit, different grid step
        a, b = draws[0::3], draws[1::3]
    elif mode == "quarter":         # 25 vs 25, half the evidence per side
        a, b = draws[0::4], draws[1::4]
    elif mode == "random":
        idx = list(range(n))
        rng.shuffle(idx)
        a = [draws[i] for i in idx[: n // 2]]
        b = [draws[i] for i in idx[n // 2:]]
    else:
        raise ValueError(mode)
    x, y = _sel(a, channel), _sel(b, channel)
    if x is None or y is None:
        return (x is None) != (y is None)
    try:
        fx, fy = float(x), float(y)
    except ValueError:
        return True
    return abs(fx - fy) / max(fx, fy) > TOL


def auc(scores, labels):
    """Within-row-pooled AUC by rank averaging. labels in {0,1}."""
    pairs = sorted(zip(scores, labels))
    n = len(pairs)
    ranks, i = [0.0] * n, 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    pos = [ranks[k] for k in range(n) if pairs[k][1] == 1]
    npos, nneg = len(pos), n - len([1 for p in pairs if p[1] == 1])
    if npos == 0 or nneg == 0:
        return None
    return (sum(pos) - npos * (npos + 1) / 2.0) / (npos * nneg)


def within_row_auc(rows, key):
    """AUC computed WITHIN each row then pooled over concordant pairs, which is
    the quantity the proposal's ship gate names."""
    conc = disc = tie = 0
    for r in rows:
        cands = r["cands"]
        pos = [c for c in cands if c["ok"]]
        neg = [c for c in cands if not c["ok"]]
        for p in pos:
            for q in neg:
                if key(p) > key(q):
                    conc += 1
                elif key(p) < key(q):
                    disc += 1
                else:
                    tie += 1
    tot = conc + disc + tie
    if not tot:
        return None, 0
    return (conc + 0.5 * tie) / tot, tot


def build(channel, split, args):
    ch = CHANNELS[channel]
    pool = load_pool(spec_for_channel(ch, split, args))
    golds = golds_for(split, ch.relation)
    rng = random.Random(20260812)
    rows = []
    for s in sorted(pool):
        vals = numeric_candidates(pool[s], channel)
        order = ranked_candidates(vals, True)
        if len(order) < 2:
            continue
        g = golds.get(s)
        sup = {c: len(tolerance_support(vals, c, TOL)) for c in order}
        shipped = _sel(pool[s], channel)
        rows.append({
            "subject": s,
            "order": order,
            "sup": sup,
            "top_share": sup[order[0]] / len(vals),
            "ratio": sup[order[1]] / sup[order[0]] if sup[order[0]] else 1.0,
            "n_sep": len(order),
            "gold": g,
            "shipped_ok": (hit(float(shipped), g) if (shipped and g) else False),
            "r1_ok": hit(order[0], g) if g is not None else None,
            "r2_ok": hit(order[1], g) if g is not None else None,
            "any_ok": (any(hit(c, g) for c in order) if g is not None else None),
            "modes": {m: unstable_by(pool[s], channel, m, rng)
                      for m in ("stride", "block", "stride3", "quarter")},
            "rand": [unstable_by(pool[s], channel, "random", random.Random(1000 + k))
                     for k in range(20)],
            "cands": [{"v": c, "sup": sup[c], "rank": i,
                       "ok": (hit(c, g) if g is not None else None)}
                      for i, c in enumerate(order)],
        })
    return rows, pool, golds


def strat(rows, flag):
    fired = [r for r in rows if flag(r)]
    rest = [r for r in rows if not flag(r)]
    def summ(sel):
        n = len(sel)
        if not n:
            return dict(n=0)
        g1 = sum(1 for r in sel if r["r1_ok"] and not r["r2_ok"])
        g2 = sum(1 for r in sel if r["r2_ok"] and not r["r1_ok"])
        return dict(n=n,
                    r1_acc=round(sum(1 for r in sel if r["r1_ok"]) / n, 4),
                    shipped_acc=round(sum(1 for r in sel if r["shipped_ok"]) / n, 4),
                    any_ok=round(sum(1 for r in sel if r["any_ok"]) / n, 4),
                    contested=g1 + g2, r1w=g1, r2w=g2,
                    break_even=(round(g1 / (g1 + g2), 4) if g1 + g2 else None))
    return summ(fired), summ(rest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="cap_recite")
    ap.add_argument("--split", default="val")
    ap.add_argument("--model", default="google/gemma-4-31B")
    ap.add_argument("--revision", default="5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--demo-seed", type=int, default=1234)
    ap.add_argument("--seed-base", type=int, default=7000)
    args = ap.parse_args()

    rows, pool, golds = build(args.channel, args.split, args)
    print(f"{args.channel}/{args.split}: {len(rows)} rows with >=2 separated candidates")

    # ---------- Q2: partition sensitivity ----------
    print("\n=== Q2  ONE GRID STEP EITHER SIDE OF THE PARTITION RULE ===")
    print(f"{'partition':12s} {'fires':>6s} {'r1acc_fire':>11s} {'r1acc_rest':>11s} "
          f"{'ship_fire':>10s} {'ship_rest':>10s} {'r1':>3s} {'r2':>3s} {'break-even':>11s}")
    for m in ("stride", "block", "stride3", "quarter"):
        f, r = strat(rows, lambda x, m=m: x["modes"][m])
        print(f"{m:12s} {f['n']:6d} {f['r1_acc']:11.4f} {r['r1_acc']:11.4f} "
              f"{f['shipped_acc']:10.4f} {r['shipped_acc']:10.4f} "
              f"{f['r1w']:3d} {f['r2w']:3d} "
              f"{(f'{f['break_even']:.4f}' if f['break_even'] is not None else 'n/a'):>11s}")
    # random partitions: distribution of the headline numbers
    be, r1w, r2w, nf = [], [], [], []
    for k in range(20):
        f, r = strat(rows, lambda x, k=k: x["rand"][k])
        nf.append(f["n"]); r1w.append(f["r1w"]); r2w.append(f["r2w"])
        if f["break_even"] is not None:
            be.append(f["break_even"])
    print(f"\n20 RANDOM half-splits: fires {min(nf)}-{max(nf)} (mean {sum(nf)/20:.1f}); "
          f"r1 wins {min(r1w)}-{max(r1w)}, r2 wins {min(r2w)}-{max(r2w)}; "
          f"break-even {min(be):.3f}-{max(be):.3f} (mean {sum(be)/len(be):.3f})")
    print(f"   rows where r2 STRICTLY beats r1 inside the fired set: "
          f"{sum(1 for a, b in zip(r1w, r2w) if b > a)}/20 random splits")

    # ---------- Q1: is it vote frequency? ----------
    print("\n=== Q1  IS INSTABILITY A RE-PARAMETERISATION OF VOTE FREQUENCY? ===")
    lab = [1 if r["modes"]["stride"] else 0 for r in rows]
    for name, sc in (("top_share (neg)", [-r["top_share"] for r in rows]),
                     ("rank2/rank1 ratio", [r["ratio"] for r in rows]),
                     ("n separated cands", [r["n_sep"] for r in rows])):
        a = auc(sc, lab)
        print(f"  AUC of {name:20s} predicting 'unstable' = {a:.4f}")
    # A pure share gate matched to the SAME firing count
    k = sum(lab)
    by_share = sorted(rows, key=lambda r: r["top_share"])[:k]
    ids = {r["subject"] for r in by_share}
    fs, rs = strat(rows, lambda x: x["subject"] in ids)
    print(f"  matched-size PURE SHARE gate (lowest {k} top_share):")
    print(f"    fires {fs['n']}  r1acc {fs['r1_acc']:.4f} (vs instability {strat(rows, lambda x: x['modes']['stride'])[0]['r1_acc']:.4f})"
          f"  any_ok {fs['any_ok']:.4f}  r1/r2 {fs['r1w']}/{fs['r2w']}"
          f"  break-even {fs['break_even']}")
    inst = {r["subject"] for r in rows if r["modes"]["stride"]}
    print(f"    overlap with the instability gate: {len(ids & inst)}/{k} rows "
          f"({len(ids & inst)/max(k,1):.1%})")

    # ---------- Q3: permutation control on the fired set ----------
    print("\n=== Q3  PERMUTATION CONTROL ON THE FIRED SET (coverage vs pool width) ===")
    fired = [r for r in rows if r["modes"]["stride"] and r["gold"] is not None]
    allg = [r["gold"] for r in rows if r["gold"] is not None]
    rng = random.Random(7)
    real_any = sum(1 for r in fired if r["any_ok"]) / len(fired)
    real_top5 = sum(1 for r in fired
                    if any(hit(c, r["gold"]) for c in r["order"][:5])) / len(fired)
    ch_any, ch_top5 = [], []
    for _ in range(400):
        perm = allg[:]
        rng.shuffle(perm)
        # assign a foreign gold to each fired row
        foreign = [rng.choice([g for g in allg]) for _ in fired]
        ch_any.append(sum(1 for r, g in zip(fired, foreign)
                          if any(hit(c, g) for c in r["order"])) / len(fired))
        ch_top5.append(sum(1 for r, g in zip(fired, foreign)
                           if any(hit(c, g) for c in r["order"][:5])) / len(fired))
    m_any = sum(ch_any) / len(ch_any)
    m_t5 = sum(ch_top5) / len(ch_top5)
    print(f"  fired rows n={len(fired)}")
    print(f"  'answer is present in shortlist'  real {real_any:.4f}  chance {m_any:.4f}  "
          f"EXCESS {real_any - m_any:+.4f}")
    print(f"  'present in top-5'                real {real_top5:.4f}  chance {m_t5:.4f}  "
          f"EXCESS {real_top5 - m_t5:+.4f}")
    print(f"  rank-1 already correct on fired set: "
          f"{sum(1 for r in fired if r['r1_ok'])}/{len(fired)}")
    print(f"  => findable-by-any-reranker rows on the fired set, chance-corrected: "
          f"{(real_any - m_any) * len(fired) - sum(1 for r in fired if r['r1_ok']):+.1f} rows "
          f"(proposal claims 'max gain = 23')")

    # ---------- Q4: what do trivial baselines score on the ship gate? ----------
    print("\n=== Q4  WITHIN-ROW AUC OF TRIVIAL BASELINES ON THE GATED CANDIDATES ===")
    print("    (the proposal's pre-committed ship rule is verifier AUC > 0.60)")
    gated = [r for r in rows if r["modes"]["stride"] and r["gold"] is not None]
    npos = sum(1 for r in gated for c in r["cands"] if c["ok"])
    ncand = sum(len(r["cands"]) for r in gated)
    print(f"  gated rows {len(gated)}, candidates {ncand}, positives {npos}")
    for name, key in (
        ("frequency support (incumbent)", lambda c: c["sup"]),
        ("prefer LARGER value",           lambda c: c["v"]),
        ("prefer SMALLER value",          lambda c: -c["v"]),
        ("roundness (mult of 1000)",      lambda c: 1.0 if c["v"] % 1000 == 0 else 0.0),
        ("anti-roundness",                lambda c: 0.0 if c["v"] % 1000 == 0 else 1.0),
    ):
        a, tot = within_row_auc(gated, key)
        print(f"  {name:32s} AUC {a:.4f}   ({tot} within-row pairs)")

    # SE on the contested arbitration
    f, _ = strat(rows, lambda x: x["modes"]["stride"])
    n_c = f["contested"]
    if n_c:
        p = f["r1w"] / n_c
        se = (p * (1 - p) / n_c) ** 0.5
        print(f"\n  contested pairs inside the gate: {n_c}; break-even {p:.4f} "
              f"+/- {se:.4f} (binomial SE)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
