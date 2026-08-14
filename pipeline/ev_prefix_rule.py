"""Per-row expected-F1 maximiser for the set relations, and its honest test.

WHY THIS RULE AND NOT A THRESHOLD (right by construction, see set_metric_facts.py):
row F1 is exactly 2*tp/(n_pred + n_gold), so the k-th candidate is worth adding
iff its probability of being correct exceeds the row's own F1/2. That cutoff is a
property of the ROW, not of the corpus, so a single global vote-share tau can
only be optimal by accident.

Implementation:
  * calibration  q(share) = P(candidate is in gold | vote share), fitted by
    weighted isotonic regression (PAVA) on one split and applied to the other.
    CROSS-FIT: train-fitted calibration scores val rows and vice versa, so no
    row is ever scored by a curve that saw its own gold.
  * generative model for a row: each pool candidate i is in gold independently
    with probability q_i; plus U unseen golds the pool never proposed, drawn
    from the empirical U distribution of the FITTING split.
  * E[F1(k)] estimated by Monte Carlo over that model, for every prefix length
    k = 0..K; the rule emits argmax k. k=0 (abstain) is scored as P(G == 0),
    which is exactly its expected F1, so the abstain decision falls out of the
    same calculation rather than being a separate gate.
  * compared against the shipped global tau by PAIRED BOOTSTRAP over pooled
    train+val rows, and the number of changed TEST rows is counted.
"""
from __future__ import annotations

import argparse
import random
import sys

sys.path.insert(0, "/Users/maksimsilchenko/AKBC/pipeline")

from aggregate import vote_shares
from channels import CHANNELS
from common import TEST_ROWS, TOTAL_TEST_ROWS, load_pool, rows_for, spec_for_channel
from scorer import _ev, paired_bootstrap

ARGS = argparse.Namespace(
    model="google/gemma-4-31B",
    revision="5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89",
    temperature=0.7, top_p=0.95, demo_seed=1234, seed_base=7000)

AWARD_DEMOS = ("FAI Gold Air Medal", "Fields medal", "Fulbright Prize",
               "Nobel Prize in Physics", "Time Person of the Year", "Ballon d'Or")


# ------------------------------------------------------------------ data

def ranked(draws: list[str], ch: str) -> list[tuple[float, str]]:
    out = [(share, surf) for share, surf in vote_shares(draws, ch).values()]
    out.sort(key=lambda x: (-x[0], x[1]))
    return out


def gold_norm_sets(golds: list[list[str]]) -> list[set]:
    ev = _ev()
    return [{ev.normalize_string(a) for a in aliases} for aliases in golds]


def load_rows(rel: str, ch: str, split: str) -> list[dict]:
    ev = _ev()
    pool = load_pool(spec_for_channel(CHANNELS[ch], split, ARGS))
    gm = {r["SubjectEntity"]: (r.get("ObjectEntities") or [])
          for r in rows_for(split, rel)} if split != "test" else {}
    rows = []
    for s in sorted(pool):
        if rel == "awardWonBy" and split == "train" and s in AWARD_DEMOS:
            continue
        rk = ranked(pool[s], ch)
        row = {"split": split, "subject": s, "ranked": rk}
        if split != "test":
            g = gm[s]
            gsets = gold_norm_sets(g)
            labels = []
            for share, surf in rk:
                nk = ev.normalize_string(surf)
                labels.append(1 if any(nk in gs for gs in gsets) else 0)
            row["gold"] = g
            row["labels"] = labels
            row["n_gold"] = len(g)
            row["matched"] = sum(labels)
            row["unseen"] = max(len(g) - sum(labels), 0)
        rows.append(row)
    return rows


def f1_of(preds: list[str], golds: list[list[str]]) -> float:
    ev = _ev()
    seen, flat = set(), []
    for p in preds:
        k = ev.normalize_string(p)
        if k not in seen:
            seen.add(k)
            flat.append(p)
    if not flat and not golds:
        return 1.0
    tp = ev.string_true_positives(flat, golds)
    return 2 * tp / (len(flat) + len(golds)) if (flat or golds) else 1.0


# ------------------------------------------------------------------ calibration

def fit_isotonic(pairs: list[tuple[float, int]]) -> list[tuple[float, float]]:
    """Weighted PAVA on (share, label). Returns [(share, q)] sorted by share."""
    agg: dict[float, list[int]] = {}
    for s, y in pairs:
        agg.setdefault(s, []).append(y)
    xs = sorted(agg)
    blocks = [[x, sum(agg[x]) / len(agg[x]), len(agg[x])] for x in xs]  # x, mean, w
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][1] > blocks[i + 1][1] + 1e-15:
            w = blocks[i][2] + blocks[i + 1][2]
            m = (blocks[i][1] * blocks[i][2] + blocks[i + 1][1] * blocks[i + 1][2]) / w
            blocks[i] = [blocks[i][0], m, w]
            del blocks[i + 1]
            if i:
                i -= 1
        else:
            i += 1
    return [(b[0], b[1]) for b in blocks]


def apply_iso(curve: list[tuple[float, float]], share: float) -> float:
    q = curve[0][1]
    for x, v in curve:
        if share >= x - 1e-12:
            q = v
        else:
            break
    return min(max(q, 1e-4), 1 - 1e-4)


# ------------------------------------------------------------------ EV rule

def ev_best_k(qs: list[float], unseen_pool: list[int], rng: random.Random,
              n_mc: int = 4000, kmax: int = 0) -> tuple[int, list[float]]:
    K = len(qs)
    kcap = min(K, kmax) if kmax else K
    tot = [0.0] * (kcap + 1)
    for _ in range(n_mc):
        b = [1 if rng.random() < q else 0 for q in qs]
        G = sum(b) + rng.choice(unseen_pool)
        tot[0] += 1.0 if G == 0 else 0.0
        run = 0
        for k in range(1, kcap + 1):
            run += b[k - 1]
            tot[k] += 2 * run / (k + G)
    ev = [t / n_mc for t in tot]
    best = max(range(kcap + 1), key=lambda k: ev[k])
    return best, ev


def run(rel: str, ch: str, ship_tau: float, n_mc: int = 4000,
        kmax: int = 0, seed: int = 20260812) -> None:
    print("=" * 90)
    print(f"EXPECTED-F1 PREFIX MAXIMISER vs GLOBAL tau   {rel}  (shipped tau={ship_tau})")
    print("=" * 90)
    data = {sp: load_rows(rel, ch, sp) for sp in ("train", "val", "test")}

    # cross-fit calibration
    curves, unseen = {}, {}
    for fit, use in (("train", "val"), ("val", "train")):
        pairs = [(share, y) for r in data[fit]
                 for (share, _), y in zip(r["ranked"], r["labels"])]
        curves[use] = fit_isotonic(pairs)
        unseen[use] = [r["unseen"] for r in data[fit]] or [0]
    # for TEST, fit on everything we are allowed to see
    pairs = [(share, y) for sp in ("train", "val") for r in data[sp]
             for (share, _), y in zip(r["ranked"], r["labels"])]
    curves["test"] = fit_isotonic(pairs)
    unseen["test"] = [r["unseen"] for sp in ("train", "val") for r in data[sp]]

    print("\nCALIBRATION q(share) = P(candidate in gold), fitted on train+val pooled")
    print("  share  ->  q      (isotonic blocks)")
    for x, v in curves["test"]:
        print(f"  {x:5.3f}  ->  {v:.4f}")
    print(f"unseen-gold per row (golds the 30-draw pool never proposed): "
          f"mean {sum(unseen['test'])/len(unseen['test']):.2f}, "
          f"max {max(unseen['test'])}")

    # apply
    rng = random.Random(seed)
    f_ship, f_ev, changed = [], [], 0
    print(f"\n{'split':>5s} {'subject':38s} {'K':>4s} {'k@tau':>6s} {'k_EV':>5s} "
          f"{'F1@tau':>7s} {'F1_EV':>7s} {'delta':>7s}")
    for sp in ("train", "val"):
        for r in data[sp]:
            qs = [apply_iso(curves[sp], share) for share, _ in r["ranked"]]
            k, _ev_curve = ev_best_k(qs, unseen[sp], rng, n_mc, kmax)
            preds_tau = [surf for share, surf in r["ranked"] if share >= ship_tau]
            preds_ev = [surf for _, surf in r["ranked"][:k]]
            a, b = f1_of(preds_tau, r["gold"]), f1_of(preds_ev, r["gold"])
            f_ship.append(a)
            f_ev.append(b)
            if sorted(preds_tau) != sorted(preds_ev):
                changed += 1
                print(f"{sp:>5s} {r['subject'][:38]:38s} {len(r['ranked']):4d} "
                      f"{len(preds_tau):6d} {k:5d} {a:7.4f} {b:7.4f} {b-a:+7.4f}")
    n = len(f_ship)
    print(f"\nrows changed on train+val: {changed}/{n}")
    print(f"pooled train+val: tau {sum(f_ship)/n:.4f} -> EV {sum(f_ev)/n:.4f} "
          f"({sum(f_ev)/n - sum(f_ship)/n:+.4f})")
    for sp in ("train", "val"):
        idx = [i for i, r in enumerate(
            [x for s in ("train", "val") for x in data[s]]) if r["split"] == sp]
        print(f"   {sp:5s} n={len(idx):3d}: tau {sum(f_ship[i] for i in idx)/len(idx):.4f}"
              f" -> EV {sum(f_ev[i] for i in idx)/len(idx):.4f}")

    bs = paired_bootstrap(f_ship, f_ev, n_boot=10000)
    print(f"\nPAIRED BOOTSTRAP (pooled train+val, {bs['n_rows']} rows, 90% CI):")
    print(f"   point {bs['point']:+.4f}  CI [{bs['ci_lo']:+.4f}, {bs['ci_hi']:+.4f}]  "
          f"rows up {bs['rows_up']} / down {bs['rows_down']}  "
          f"excludes zero above: {bs['excludes_zero_above']}")
    w = TEST_ROWS[rel] / TOTAL_TEST_ROWS
    print(f"   overall-score value if this delta transferred to TEST: "
          f"{bs['point']*w:+.5f} (CI [{bs['ci_lo']*w:+.5f}, {bs['ci_hi']*w:+.5f}])")

    # TEST: how many rows would change
    rng = random.Random(seed)
    diff = 0
    sizes_tau = sizes_ev = 0
    abst_tau = abst_ev = 0
    for r in data["test"]:
        qs = [apply_iso(curves["test"], share) for share, _ in r["ranked"]]
        k, _ = ev_best_k(qs, unseen["test"], rng, n_mc, kmax)
        preds_tau = [surf for share, surf in r["ranked"] if share >= ship_tau]
        preds_ev = [surf for _, surf in r["ranked"][:k]]
        sizes_tau += len(preds_tau)
        sizes_ev += len(preds_ev)
        abst_tau += (not preds_tau)
        abst_ev += (not preds_ev)
        if sorted(preds_tau) != sorted(preds_ev):
            diff += 1
    nt = len(data["test"])
    print(f"\nTEST: EV rule changes {diff}/{nt} rows. "
          f"mean preds {sizes_tau/nt:.2f} (tau) -> {sizes_ev/nt:.2f} (EV); "
          f"abstain {abst_tau}/{nt} -> {abst_ev}/{nt}")


if __name__ == "__main__":
    run("countryLandBordersCountry", "borders_list", 0.15)
    print()
    run("awardWonBy", "award_list", 0.10, n_mc=1500, kmax=200)
