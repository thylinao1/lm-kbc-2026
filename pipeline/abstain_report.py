"""Driver that reproduces every number in the abstention-relation anatomy.

    cd /Users/maksimsilchenko/AKBC/pipeline
    source ~/mac-ml-setup/.venv/bin/activate
    python3 abstain_report.py            # all sections
    python3 abstain_report.py 1 3        # selected sections

Sections
  1  confusion decomposition (VAL + TRAIN with the demo guard) and the gold-free
     TEST statistics (abstention rate, set sizes)
  2  oracles: perfect gate vs perfect object choice vs pool-limited object choice
  3  the cityOfDeath abstention-rate curve, with the attainable-rate ladder
  4  the expected-F1 prefix maximiser against the shipped tau rule
  5  matched-rate re-ranking: same abstention COUNT, different rows

Everything reads cached pools plus train/val gold. Nothing is written outside
this file's own stdout.
"""
from __future__ import annotations

import collections
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from abstain_anatomy import (BUCKETS, SHIPPED, EF1Calibrator, confusion, content_set,
                             ef1_predict, ef1_predict_joint, get_pool, gold_map, oracles,
                             per_row, row_view, score_preds, usable_subjects)
from aggregate import normalize, predict_set
from scorer import paired_bootstrap

W = 100 / 475          # both relations carry 100 of the 475 test rows
KFOLD = 10
FOLD_SEED = 20260812


def shipped_preds(cfg, pool, subjects=None):
    return {s: predict_set(d, cfg["channel"], cfg["tau"], emit_ratio=cfg["emit_ratio"])
            for s, d in pool.items() if subjects is None or s in subjects}


def calib_rows(rel):
    """Pooled train(non-demo) + val rows: (split, subject, view, gold)."""
    ch = SHIPPED[rel]["channel"]
    out = []
    for sp in ("train", "val"):
        pool = get_pool(ch, sp)
        keep = usable_subjects(ch, sp, pool)
        g = gold_map(rel, sp)
        for s, d in pool.items():
            if s in keep:
                out.append((sp, s, row_view(d, ch), g.get(s, [])))
    return out


def pooled_vectors(rel, a_preds, b_preds):
    A, B = [], []
    for sp in ("train", "val"):
        keep = set(b_preds[sp])
        ra = score_preds(a_preds[sp], rel, sp, subjects=keep)
        rb = score_preds(b_preds[sp], rel, sp, subjects=keep)
        ma = dict(zip(ra["subjects"], ra["f1_vector"]))
        mb = dict(zip(rb["subjects"], rb["f1_vector"]))
        for s in sorted(keep):
            A.append(ma[s])
            B.append(mb[s])
    return A, B


# ---------------------------------------------------------------- 1


def section1():
    print("\n" + "=" * 100)
    print("1. CONFUSION DECOMPOSITION")
    for rel, cfg in SHIPPED.items():
        ch = cfg["channel"]
        print("-" * 100)
        print(f"{rel}  channel={ch} tau={cfg['tau']} emit_ratio={cfg['emit_ratio']}")
        for sp in ("train", "val"):
            pool = get_pool(ch, sp)
            keep = usable_subjects(ch, sp, pool)
            c = confusion(shipped_preds(cfg, pool), rel, sp, subjects=keep)
            n = c["__n__"]
            print(f"  [{sp}] n={n} macro-F1={c['__macro_f1__']:.4f}")
            for b in BUCKETS:
                print(f"     {b:16s} n={c[b]['n']:3d} ({c[b]['n']/n:5.1%})  "
                      f"sum f1={c[b]['score']:7.3f}  contributes {c[b]['contrib']:.4f}")
        pool = get_pool(ch, "test")
        pr = shipped_preds(cfg, pool)
        sizes = collections.Counter(len(v) for v in pr.values())
        tot = sum(len(v) for v in pr.values())
        print(f"  [test] n={len(pr)} abstain={sizes[0]} ({sizes[0]/len(pr):.1%}) "
              f"|S| hist={dict(sorted(sizes.items()))} mean|S|={tot/len(pr):.3f} "
              f"mean|S| over answered rows={tot/max(1, len(pr)-sizes[0]):.3f}")


# ---------------------------------------------------------------- 2


def section2():
    print("\n" + "=" * 100)
    print("2. IS THE GATE OR THE OBJECT CHOICE BINDING?")
    for rel, cfg in SHIPPED.items():
        print("-" * 100)
        print(rel)
        for sp in ("train", "val"):
            o = oracles(rel, sp, cfg)
            print(f"  [{sp}] n={o['n']}")
            for k in ("shipped", "oracle_abstain", "oracle_content", "oracle_content_pool"):
                d = o[k] - o["shipped"]
                print(f"     {k:20s} {o[k]:.4f}"
                      + ("" if k == "shipped" else
                         f"   ({d:+.4f} relation = {d*W:+.4f} overall)"))


# ---------------------------------------------------------------- 3


def section3():
    print("\n" + "=" * 100)
    print("3. cityOfDeath ABSTENTION-RATE CURVE")
    rel = "personHasCityOfDeath"
    cfg = SHIPPED[rel]
    ch = cfg["channel"]
    pools = {sp: get_pool(ch, sp) for sp in ("train", "val", "test")}
    keep = {sp: usable_subjects(ch, sp, pools[sp]) for sp in pools}

    def P(sp, tau):
        return {s: predict_set(d, ch, tau) for s, d in pools[sp].items() if s in keep[sp]}

    base = P("test", cfg["tau"])
    print("  Vote shares are multiples of 1/30, so tau is a step function; only the")
    print("  abstention rates below are attainable at all.")
    print(f"  {'tau band':>18} {'test_abst':>10} {'val_abst':>9} {'valF1':>8} {'valSE':>7} "
          f"{'trainF1':>8} {'test rows vs shipped':>21}")
    prev = None
    for i in range(0, 31):
        px = P("test", (i + 0.5) / 30)
        ab = sum(1 for v in px.values() if not v)
        if ab == prev:
            continue
        prev = ab
        pv, pt = P("val", (i + 0.5) / 30), P("train", (i + 0.5) / 30)
        rv = score_preds(pv, rel, "val")
        rt = score_preds(pt, rel, "train", subjects=keep["train"])
        se = statistics.pstdev(rv["f1_vector"]) / len(rv["f1_vector"]) ** 0.5
        chg = sum(1 for s in px if px[s] != base[s])
        print(f"  ({i/30:.4f},{(i+1)/30:.4f}] {ab:9d}% "
              f"{sum(1 for v in pv.values() if not v):8d}% {rv['macro_f1']:8.4f} {se:7.4f} "
              f"{rt['macro_f1']:8.4f} {chg:21d}")

    print("\n  MARGINAL ROWS: what a lower tau actually buys, on pooled train(non-demo)+val")
    for lo in (0.30, 0.3667, 0.40, 0.4333):
        n_tot = n_empty = 0
        corr = 0.0
        for sp in ("train", "val"):
            a, b = P(sp, lo), P(sp, cfg["tau"])
            g = gold_map(rel, sp)
            m = [s for s in a if a[s] and not b[s]]
            if m:
                pr = per_row({s: a[s] for s in m}, rel, sp)
                corr += sum(pr[s]["f1"] for s in m)
            n_tot += len(m)
            n_empty += sum(1 for s in m if not g.get(s))
        tx = sum(1 for v in P("test", lo).values() if not v)
        print(f"    tau {lo:.4f}: {n_tot:2d} rows flip abstain->answer | gold EMPTY on {n_empty} "
              f"({n_empty/max(1,n_tot):.0%}) | our answer CORRECT on {corr:.2f} "
              f"({corr/max(1,n_tot):.0%}) | net {corr-n_empty:+.2f} row-points || TEST abstention "
              f"{sum(1 for v in base.values() if not v)}% -> {tx}%")

    print("\n  PAIRED BOOTSTRAP against the shipped tau, pooled train(non-demo)+val")
    b45 = {sp: P(sp, cfg["tau"]) for sp in ("train", "val")}
    for tau in (0.30, 0.3333, 0.3667, 0.40, 0.4333, 0.4667, 0.5333):
        cand = {sp: P(sp, tau) for sp in ("train", "val")}
        A, B = pooled_vectors(rel, b45, cand)
        bs = paired_bootstrap(A, B, n_boot=10000)
        print(f"    tau {tau:.4f}: pooled delta {bs['point']:+.4f} "
              f"90%CI [{bs['ci_lo']:+.4f},{bs['ci_hi']:+.4f}] "
              f"up {bs['rows_up']} down {bs['rows_down']}")


# ---------------------------------------------------------------- 4


def _crossfit(rel, data, use_empty, seed=FOLD_SEED, joint_u=None):
    idx = list(range(len(data)))
    random.Random(seed).shuffle(idx)
    preds = {"train": {}, "val": {}}
    ks = []
    for i in range(KFOLD):
        te = set(idx[i::KFOLD])
        tr = [(data[j][2], data[j][3]) for j in idx if j not in te]
        cal = EF1Calibrator(use_empty=use_empty).fit(tr)
        for j in idx[i::KFOLD]:
            sp, s, v, _ = data[j]
            p, info = (ef1_predict_joint(v, cal, u=joint_u) if joint_u is not None
                       else ef1_predict(v, cal))
            preds[sp][s] = p
            ks.append(info["k"])
    return preds, ks


def _pool_miss(rel):
    """Mean number of gold objects absent from the candidate pool (non-empty rows)."""
    ch = SHIPPED[rel]["channel"]
    miss = []
    for sp in ("train", "val"):
        pool = get_pool(ch, sp)
        keep = usable_subjects(ch, sp, pool)
        g = gold_map(rel, sp)
        allc = {s: [c[1] for c in row_view(d, ch)["cands"]] for s, d in pool.items() if s in keep}
        pr = per_row(allc, rel, sp)
        for s in allc:
            if g.get(s):
                miss.append(len(g[s]) - pr[s]["tp"])
    return sum(miss) / len(miss)


def section4():
    print("\n" + "=" * 100)
    print("4. EXPECTED-F1 PREFIX MAXIMISER vs THE SHIPPED TAU RULE")
    print("   Cross-fitted 10-fold on pooled train(non-demo)+val. The 'top_share only'")
    print("   variant is the CONTROL: with q monotone in the top share the maximiser is")
    print("   provably a threshold on the top share, i.e. the shipped rule.")
    for rel, cfg in SHIPPED.items():
        ch = cfg["channel"]
        print("-" * 100)
        print(rel)
        data = calib_rows(rel)
        base = {}
        for sp in ("train", "val"):
            pool = get_pool(ch, sp)
            base[sp] = shipped_preds(cfg, pool, usable_subjects(ch, sp, pool))
        variants = [(False, None, "q=top_share only (control)"),
                    (True, None, "q=top_share+empty_share")]
        if rel == "companyTradesAtStockExchange":
            variants.append((True, _pool_miss(rel), "q=top+empty, |gold| coupled to candidates (post-hoc)"))
        for use_empty, u, label in variants:
            preds, ks = _crossfit(rel, data, use_empty, joint_u=u)
            A, B = pooled_vectors(rel, base, preds)
            bs = paired_bootstrap(A, B, n_boot=10000)
            print(f"  {label}")
            for sp in ("train", "val"):
                k = set(preds[sp])
                a = score_preds(base[sp], rel, sp, subjects=k)["macro_f1"]
                b = score_preds(preds[sp], rel, sp, subjects=k)["macro_f1"]
                print(f"     {sp:5s} n={len(k):3d} shipped {a:.4f} -> EF1 {b:.4f}  ({b-a:+.4f})")
            print(f"     POOLED n={bs['n_rows']} delta {bs['point']:+.4f} "
                  f"90%CI [{bs['ci_lo']:+.4f},{bs['ci_hi']:+.4f}] "
                  f"up {bs['rows_up']} down {bs['rows_down']} "
                  f"chosen |S| {dict(sorted(collections.Counter(ks).items()))}")
            ds = [paired_bootstrap(*pooled_vectors(rel, base,
                                                   _crossfit(rel, data, use_empty, seed=s, joint_u=u)[0]),
                                   n_boot=200)["point"] for s in range(1, 11)]
            print(f"     fold-seed sensitivity over 10 seeds: min {min(ds):+.4f} "
                  f"mean {sum(ds)/len(ds):+.4f} max {max(ds):+.4f}")
            cal = EF1Calibrator(use_empty=use_empty).fit([(d[2], d[3]) for d in data])
            pool = get_pool(ch, "test")
            ship = shipped_preds(cfg, pool)
            new = {s: (ef1_predict_joint(row_view(d, ch), cal, u=u)[0] if u is not None
                       else ef1_predict(row_view(d, ch), cal)[0]) for s, d in pool.items()}
            chg = sum(1 for s in pool
                      if [normalize(x) for x in new[s]] != [normalize(x) for x in ship[s]])
            print(f"     TEST rows changed {chg}; test abstention "
                  f"{sum(1 for v in ship.values() if not v)}% -> "
                  f"{sum(1 for v in new.values() if not v)}%")


# ---------------------------------------------------------------- 5


def section5():
    print("\n" + "=" * 100)
    print("5. MATCHED-RATE RE-RANKING (same abstention COUNT, different rows)")
    for rel, cfg in SHIPPED.items():
        ch = cfg["channel"]
        print("-" * 100)
        print(rel)
        data = calib_rows(rel)
        base = {}
        for sp in ("train", "val"):
            pool = get_pool(ch, sp)
            base[sp] = shipped_preds(cfg, pool, usable_subjects(ch, sp, pool))
        for use_empty in (False, True):
            idx = list(range(len(data)))
            random.Random(7).shuffle(idx)
            q = {}
            for i in range(KFOLD):
                te = set(idx[i::KFOLD])
                tr = [(data[j][2], data[j][3]) for j in idx if j not in te]
                cal = EF1Calibrator(use_empty=use_empty).fit(tr)
                for j in idx[i::KFOLD]:
                    sp, s, v, _ = data[j]
                    q[(sp, s)] = cal.q(v)
            preds = {}
            for sp in ("train", "val"):
                pool = get_pool(ch, sp)
                n_ab = sum(1 for v in base[sp].values() if not v)
                abst = set(sorted(base[sp], key=lambda s: -q[(sp, s)])[:n_ab])
                preds[sp] = {s: ([] if s in abst else content_set(row_view(pool[s], ch), cfg))
                             for s in base[sp]}
            A, B = pooled_vectors(rel, base, preds)
            bs = paired_bootstrap(A, B, n_boot=10000)
            lab = "top_share+empty_share" if use_empty else "top_share only (control)"
            outs = []
            for sp in ("train", "val"):
                k = set(preds[sp])
                outs.append(f"{sp} {score_preds(base[sp],rel,sp,subjects=k)['macro_f1']:.4f}"
                            f"->{score_preds(preds[sp],rel,sp,subjects=k)['macro_f1']:.4f}")
            print(f"  rank by q({lab:24s}) {'  '.join(outs)}  POOLED {bs['point']:+.4f} "
                  f"90%CI [{bs['ci_lo']:+.4f},{bs['ci_hi']:+.4f}] "
                  f"up {bs['rows_up']} down {bs['rows_down']}")
        cal = EF1Calibrator(use_empty=True).fit([(d[2], d[3]) for d in data])
        pool = get_pool(ch, "test")
        ship = shipped_preds(cfg, pool)
        n_ab = sum(1 for v in ship.values() if not v)
        qt = {s: cal.q(row_view(d, ch)) for s, d in pool.items()}
        abst = set(sorted(pool, key=lambda s: -qt[s])[:n_ab])
        swap = len(abst - {s for s in ship if not ship[s]})
        print(f"     on TEST at the matched abstention count ({n_ab}): {swap} rows swap "
              f"in/out of the abstain set")


SECTIONS = {1: section1, 2: section2, 3: section3, 4: section4, 5: section5}

if __name__ == "__main__":
    want = [int(a) for a in sys.argv[1:]] or sorted(SECTIONS)
    for s in want:
        SECTIONS[s]()
