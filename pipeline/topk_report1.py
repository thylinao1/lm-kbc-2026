"""Deliverables 1 and 2: cov@k curves and uniform top-k expected F1.

Run:
  cd /Users/maksimsilchenko/AKBC/pipeline && source ~/mac-ml-setup/.venv/bin/activate \
    && python3 topk_report1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from topk_numeric import (KMAX, build_table, cov_at_k, default_args,
                          demo_subjects, score_policy, verify_gold_shape)

PAIRS = [("hasCapacity", "cap_recite"), ("hasArea", "area_recite")]
OUT = Path("out")


def main() -> int:
    args = default_args()
    report = {}

    for relation, channel in PAIRS:
        print("=" * 78)
        print(f"{relation} / {channel}")
        print("gold shape:", json.dumps(verify_gold_shape(relation)))
        print(f"demo subjects excluded from train: {len(demo_subjects(channel, args.demo_seed))}")
        rel_rep = {"gold_shape": verify_gold_shape(relation), "splits": {}}

        for split in ("train", "val"):
            for seed_shipped in (True, False):
                tag = "shipped-seeded" if seed_shipped else "pure-vote-share"
                t = build_table(channel, split, args, seed_shipped=seed_shipped)
                cov = cov_at_k(t, KMAX)
                # measured E[F1] under a uniform top-k policy, official scorer
                meas = [score_policy(t, relation, split, k)["macro_f1"]
                        for k in range(1, KMAX + 1)]
                pred = [cov[k - 1] * 2 / (k + 1) for k in range(1, KMAX + 1)]
                nk = [len(r["values"]) for r in t]
                rel_rep["splits"][f"{split}/{tag}"] = {
                    "n_rows": len(t),
                    "mean_candidates_available": round(sum(nk) / len(nk), 2),
                    "rows_with_fewer_than_8_candidates": sum(1 for v in nk if v < 8),
                    "cov_at_k": [round(c, 4) for c in cov],
                    "E_F1_formula": [round(p, 4) for p in pred],
                    "E_F1_measured_official_scorer": [round(m, 4) for m in meas],
                    "max_abs_formula_vs_measured": round(
                        max(abs(a - b) for a, b in zip(pred, meas)), 6),
                    "best_uniform_k": max(range(1, KMAX + 1), key=lambda k: meas[k - 1]),
                }
                print(f"\n  [{split}/{tag}]  n={len(t)}")
                print("    k        " + "".join(f"{k:>9d}" for k in range(1, KMAX + 1)))
                print("    cov@k    " + "".join(f"{c:>9.4f}" for c in cov))
                print("    2/(k+1)  " + "".join(f"{2/(k+1):>9.4f}" for k in range(1, KMAX + 1)))
                print("    E[F1]    " + "".join(f"{p:>9.4f}" for p in pred))
                print("    measured " + "".join(f"{m:>9.4f}" for m in meas))
        report[relation] = rel_rep

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "topk_report1.json", "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"\nwrote {OUT/'topk_report1.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
