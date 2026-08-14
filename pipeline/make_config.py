"""Turn tuning and consensus outputs into an assemble.py config.

Reads whatever result files exist in pools/ and emits the best available
configuration per relation, applying the campaign's own selection discipline:

  * parameters come from the TRAIN argmax, never the val argmax. val is a
    selection set and reading its argmax directly would compound the bias we
    already pay for using it as a gate.
  * for numeric relations, cross-frame consensus is chosen over the best single
    frame only if it wins on train AND its paired bootstrap against the best
    single frame excludes zero from above. Otherwise the single frame ships and
    the consensus result is logged as tried-and-not-separated.
  * relations with no usable result are OMITTED, which makes them emit empty
    predictions. That keeps a partial ladder submittable instead of blocking on
    the slowest relation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from channels import CHANNELS
from common import NUMERIC_RELATIONS, REPO, TEST_ROWS


def load(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default=str(REPO / "pools"))
    ap.add_argument("--out", default=str(REPO / "configs" / "best.json"))
    ap.add_argument("--min-consensus-gain", type=float, default=0.0,
                    help="extra train-side margin consensus must clear")
    args = ap.parse_args()

    pdir = Path(args.pools)
    config: dict = {}
    notes: list[str] = []

    # ---- per-channel tuning results
    best_single: dict[str, dict] = {}
    for f in sorted(pdir.glob("tune_*.json")):
        r = load(f)
        if not r or "best_on_train" not in r:
            continue
        rel, ch = r["relation"], r["channel"]
        cand = {"channel": ch, "param": r["best_on_train"]["param"],
                "train_f1": r["best_on_train"]["macro_f1"],
                "val_f1": (r.get("val_at_train_argmax") or {}).get("macro_f1"),
                "parse_qa_ok": r.get("parse_qa_ok", True)}
        if not cand["parse_qa_ok"]:
            notes.append(f"{ch}: parse-QA FAILED ({r.get('parse_qa_reason')}), excluded")
            continue
        cur = best_single.get(rel)
        if cur is None or cand["train_f1"] > cur["train_f1"]:
            best_single[rel] = cand

    for rel, c in best_single.items():
        config[rel] = {"channel": c["channel"], "param": c["param"]}
        notes.append(f"{rel}: single frame {c['channel']} param={c['param']} "
                     f"train={c['train_f1']:.4f} val={c['val_f1']}")

    # ---- cross-frame consensus, numeric relations only for now
    for f in sorted(pdir.glob("consensus_*.json")):
        r = load(f)
        if not r:
            continue
        rel = r.get("relation")
        tr, va = r.get("train"), r.get("val")
        if not tr or rel not in NUMERIC_RELATIONS:
            continue
        cons, singles = tr.get("consensus"), tr.get("single") or {}
        if not cons or not singles:
            continue
        bs_name = max(singles, key=lambda k: singles[k]["macro_f1"])
        bs_f1 = singles[bs_name]["macro_f1"]
        gain_train = cons["macro_f1"] - bs_f1
        sep = (va or {}).get("vs_best_single", {}).get("excludes_zero_above", False)

        if gain_train > args.min_consensus_gain and sep:
            config[rel] = {"channels": r["channels"], "param": cons["scale"],
                           "w": cons["w"]}
            notes.append(
                f"{rel}: CONSENSUS over {len(r['channels'])} frames "
                f"(w={cons['w']}, scale={cons['scale']}) beats best single "
                f"{bs_name} by {gain_train:+.4f} on train and separates on val")
        else:
            why = "no train gain" if gain_train <= args.min_consensus_gain else "val CI includes 0"
            notes.append(
                f"{rel}: consensus tried, NOT shipped ({why}; train {gain_train:+.4f}). "
                f"Keeping single frame {bs_name}.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(config, indent=1))

    covered = sum(TEST_ROWS[r] for r in config if r in TEST_ROWS)
    print(json.dumps(config, indent=1))
    print("\n--- notes ---")
    for n in notes:
        print("  " + n)
    print(f"\nrelations configured: {len(config)}/6   test rows covered: "
          f"{covered}/475 ({covered/475:.1%})")
    missing = [r for r in TEST_ROWS if r not in config]
    if missing:
        print(f"omitted (will emit empty): {missing}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
