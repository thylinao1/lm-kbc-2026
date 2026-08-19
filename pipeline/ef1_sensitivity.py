"""ADVERSARIAL RE-RUN of expected_f1.py one grid step either side of every free
parameter, to test whether the reported signs survive.

The proposal reports the sign of each relation's cross-fitted delta at ONE
setting of a list of parameters it did not enumerate. This module re-runs the
identical protocol (same cross-fit, same official scorer, same paired
bootstrap) while moving one parameter at a time, and prints the sign.

Nothing here writes to configs/, submissions/ or docs/.

Run:
  cd /Users/maksimsilchenko/AKBC/pipeline && source ~/mac-ml-setup/.venv/bin/activate \
    && python3 ef1_sensitivity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import expected_f1 as E
from common import TEST_ROWS, TOTAL_TEST_ROWS
from scorer import paired_bootstrap

ROWCACHE: dict[tuple, list] = {}


def rows_for_rel(rel: str, split: str, kmax: int) -> list:
    key = (rel, split, kmax)
    if key not in ROWCACHE:
        old = E.RELCFG[rel]["kmax"]
        E.RELCFG[rel]["kmax"] = kmax
        ROWCACHE[key] = E.build_rows(rel, split)
        E.RELCFG[rel]["kmax"] = old
    return ROWCACHE[key]


def delta(rel: str, calibrator: str = "iso", n_mc: int = 4000, seed: int = E.SEED,
          kmax: int | None = None) -> dict:
    """Pooled cross-fitted delta, exactly as run_relation computes it."""
    cfg = E.RELCFG[rel]
    km = cfg["kmax"] if kmax is None else kmax
    old = cfg["kmax"]
    cfg["kmax"] = km
    try:
        rng = np.random.default_rng(seed)
        data = {sp: rows_for_rel(rel, sp, km) for sp in ("train", "val")}
        f_ship, f_ev = [], []
        for use, fit in (("val", "train"), ("train", "val")):
            cal = E.fit_calibrator(calibrator, data[fit])
            size = E.fit_size_model(rel, data[fit])
            ks_s = {r["subject"]: E.shipped_k(r, rel) for r in data[use]}
            ks_e = {}
            for r in data[use]:
                k, _ = E.choose_k(r, cal, size, rel, rng, n_mc)
                ks_e[r["subject"]] = k
            a = E.score_prefixes(data[use], ks_s, rel, use)
            b = E.score_prefixes(data[use], ks_e, rel, use)
            f_ship += a["f1_vector"]
            f_ev += b["f1_vector"]
        bs = paired_bootstrap(f_ship, f_ev, n_boot=4000)
        w = TEST_ROWS[rel] / TOTAL_TEST_ROWS
        return {"delta": bs["point"], "lo": bs["ci_lo"], "hi": bs["ci_hi"],
                "overall": bs["point"] * w, "n": len(f_ship)}
    finally:
        cfg["kmax"] = old


RELS = list(E.RELCFG)


def line(tag: str, res: dict[str, dict]) -> None:
    tot = sum(res[r]["overall"] for r in RELS if r in res)
    cells = "".join(f"{res[r]['delta']:+8.4f}" if r in res else f"{'-':>8s}"
                    for r in RELS)
    print(f"{tag:34s}{cells}  {tot:+9.5f}")


def header() -> None:
    print(f"{'variant':34s}" + "".join(f"{r[:8]:>8s}" for r in RELS) + f"  {'SUM':>9s}")
    print("-" * 34 + "-" * (8 * len(RELS)) + "-" * 11)


def main() -> int:
    print("=" * 92)
    print("A. BASELINE and the CALIBRATOR FAMILY (the parameter the proposal did vary)")
    print("=" * 92)
    header()
    base = {}
    for cal in ("iso", "bin", "logit"):
        r = {rel: delta(rel, calibrator=cal) for rel in RELS}
        if cal == "iso":
            base = r
        line(f"calibrator={cal}", r)

    print()
    print("=" * 92)
    print("B. NUM_KMAX (candidate list length for the two NUMERIC relations). Shipped 8.")
    print("   This bounds how far the EV rule may extend and is nowhere justified.")
    print("=" * 92)
    print(f"{'kmax':>6s} {'hasCapacity':>14s} {'hasArea':>14s} {'their SUM':>12s}")
    for km in (2, 3, 4, 6, 8, 10, 12):
        c = delta("hasCapacity", kmax=km)
        a = delta("hasArea", kmax=km)
        print(f"{km:6d} {c['delta']:+14.4f} {a['delta']:+14.4f} "
              f"{c['overall']+a['overall']:+12.5f}")

    print()
    print("=" * 92)
    print("C. ISOTONIC CLAMP (apply_iso clips q to [eps, 1-eps]). Shipped eps=1e-4.")
    print("=" * 92)
    header()
    orig_apply = E.apply_iso
    for eps in (1e-2, 1e-3, 1e-4, 1e-6):
        def mk(e):
            def f(curve, share):
                q = curve[0][1] if curve else 0.5
                for x, v in curve:
                    if share >= x - 1e-12:
                        q = v
                    else:
                        break
                return float(min(max(q, e), 1 - e))
            return f
        E.apply_iso = mk(eps)
        line(f"iso clamp eps={eps:g}", {rel: delta(rel) for rel in RELS})
    E.apply_iso = orig_apply

    print()
    print("=" * 92)
    print("D. BIN calibrator pseudo-count a (shipped 5.0) and BIN_EDGES (shipped 7 bins)")
    print("=" * 92)
    header()
    from collections import defaultdict

    old_cal = E.Calibrator

    def make_cal(a_val: float):
        class C(old_cal):
            def __init__(self, kind, obs):
                if kind != "bin":
                    super().__init__(kind, obs)
                    return
                self.kind = kind
                self.base = sum(o[4] for o in obs) / max(len(obs), 1)
                self.n_obs = len(obs)
                acc = defaultdict(list)
                for s, _r, _t, _k, y in obs:
                    acc[self._bin(s)].append(y)
                self.bins = {b: (sum(v) + a_val * self.base) / (len(v) + a_val)
                             for b, v in acc.items()}
                self.blocks = len(self.bins)
        return C

    for a_val in (1.0, 2.0, 5.0, 10.0, 20.0):
        E.Calibrator = make_cal(a_val)
        line(f"bin pseudo-count a={a_val:g}",
             {rel: delta(rel, calibrator="bin") for rel in RELS})
    E.Calibrator = old_cal

    old_edges = E.BIN_EDGES
    for name, edges in (("coarse 4 bins", [0.0, 0.10, 0.35, 0.75, 1.01]),
                        ("shipped 7 bins", [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.01]),
                        ("fine 10 bins", [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60, 0.80, 1.01])):
        E.BIN_EDGES = edges
        line(f"bin edges: {name}", {rel: delta(rel, calibrator="bin") for rel in RELS})
    E.BIN_EDGES = old_edges

    print()
    print("=" * 92)
    print("E. MONTE-CARLO SEED and n_mc (multi-gold relations only; numerics are exact)")
    print("=" * 92)
    header()
    for sd in (E.SEED, 1, 999, 20260101):
        line(f"seed={sd}", {rel: delta(rel, seed=sd) for rel in RELS})
    for nm in (500, 1000, 4000, 16000):
        line(f"n_mc={nm}", {rel: delta(rel, n_mc=nm) for rel in RELS})

    print()
    print("=" * 92)
    print("F. PER-DIRECTION split of the cross-fit (which half fits the curve)")
    print("=" * 92)
    for rel in RELS:
        rng = np.random.default_rng(E.SEED)
        data = {sp: rows_for_rel(rel, sp, E.RELCFG[rel]["kmax"]) for sp in ("train", "val")}
        outs = []
        for use, fit in (("val", "train"), ("train", "val")):
            cal = E.fit_calibrator("iso", data[fit])
            size = E.fit_size_model(rel, data[fit])
            ks_s = {r["subject"]: E.shipped_k(r, rel) for r in data[use]}
            ks_e = {r["subject"]: E.choose_k(r, cal, size, rel, rng, 4000)[0] for r in data[use]}
            a = E.score_prefixes(data[use], ks_s, rel, use)
            b = E.score_prefixes(data[use], ks_e, rel, use)
            outs.append((use, a["n_rows"], b["macro_f1"] - a["macro_f1"]))
        s = "  ".join(f"eval {u} (n={n}) {d:+.4f}" for u, n, d in outs)
        flip = "SIGN FLIPS" if outs[0][2] * outs[1][2] < 0 else ""
        print(f"  {rel:32s} {s}   {flip}")

    print()
    print("=" * 92)
    print("G. PER-RELATION STANDARD ERROR on the TEST row count (sd of shipped per-row F1"
          "\n   on pooled train+val, divided by sqrt(test rows))")
    print("=" * 92)
    for rel in RELS:
        f = []
        for sp in ("train", "val"):
            rows = rows_for_rel(rel, sp, E.RELCFG[rel]["kmax"])
            ks = {r["subject"]: E.shipped_k(r, rel) for r in rows}
            f += E.score_prefixes(rows, ks, rel, sp)["f1_vector"]
        sd = float(np.std(np.array(f), ddof=1))
        nt = TEST_ROWS[rel]
        se = sd / np.sqrt(nt)
        w = nt / TOTAL_TEST_ROWS
        print(f"  {rel:32s} sd={sd:.4f}  test n={nt:3d}  SE={se:.4f}  "
              f"= {se*w:+.5f} overall   |  measured delta {base[rel]['delta']:+.4f} "
              f"= {abs(base[rel]['delta'])/se:.2f} SE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
