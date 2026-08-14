"""Assemble a submittable predictions.jsonl from cached pools, with the
pre-submit interlock that has to pass before a slot is ever spent.

Checks enforced here, each one traceable to a failure this campaign identified:
  * gold freshness   - the subject set is compared against test.jsonl at CURRENT
                       origin/main, never a pinned commit. Organizers renamed 15
                       test subjects AND dropped 2 rows mid-phase on Aug 7. A
                       pure rename is SILENT: evaluate.py iterates the gold file
                       and a missing key scores as an empty prediction, so the
                       all-empty probe cannot detect it either.
  * row completeness - exactly the gold universe, no extras, no omissions.
  * numeric hygiene  - one bare comma-free parseable number, no units. A
                       parse-failing prediction still counts in the precision
                       denominator, so units are strictly worse than silence.
  * substrate manifest - every predictions file records the model ids and their
                       summed safetensors parameters. Mixing two substrates
                       across relations would breach the 32B cap invisibly.
  * pool provenance  - every POOL-ID consumed is recorded and re-verified, so a
                       stale pool cannot masquerade as a fresh result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate import (predict_numeric, predict_numeric_consensus, predict_set,
                       predict_set_consensus)
from channels import CHANNELS, demo_ids, pick_demos
from common import (NUMERIC_RELATIONS, PoolSpec, REPO, TEST_ROWS, load_pool,
                    load_split, split_sha)


class PreSubmitFailure(RuntimeError):
    pass


def gold_head(dataset_dir: Path) -> tuple[str, str]:
    """Return (local HEAD, origin/main) for the dataset repo, after fetching."""
    def git(*a: str) -> str:
        return subprocess.run(["git", "-C", str(dataset_dir), *a],
                              capture_output=True, text=True).stdout.strip()
    subprocess.run(["git", "-C", str(dataset_dir), "fetch", "origin"],
                   capture_output=True, text=True, timeout=180)
    return git("rev-parse", "HEAD"), git("rev-parse", "origin/main")


def _pool_for(relation: str, channel: str, split: str, args):
    ch = CHANNELS[channel]
    if ch.relation != relation:
        raise PreSubmitFailure(f"channel {channel} serves {ch.relation}, not {relation}")
    from common import spec_for_channel
    spec = spec_for_channel(ch, split, args)
    return spec, load_pool(spec, args.n or None)        # load_pool verifies manifest


def build(config: dict, split: str, args) -> tuple[list[dict], dict]:
    """config maps relation -> either
         {"channel": name, "param": float}                  single frame, or
         {"channels": [names], "param": float, "w": float}   cross-frame consensus
    Relations absent from the config emit empty predictions, which keeps a
    partial ladder submittable."""
    gold_rows = load_split(split)
    preds: dict[tuple[str, str], list[str]] = {}
    provenance = []

    for relation, cfg in config.items():
        names = cfg.get("channels") or [cfg["channel"]]
        pools, ids = {}, []
        for nm in names:
            spec, pool = _pool_for(relation, nm, split, args)
            pools[nm] = pool
            ids.append({"channel": nm, "pool_id": spec.pool_id()})
        provenance.append({"relation": relation, "param": cfg["param"],
                           "w": cfg.get("w"), "pools": ids,
                           "emit_ratio": cfg.get("emit_ratio"),
                           "antiround": cfg.get("antiround"),
                           "method": cfg.get("method"),
                           "mode": "consensus" if len(names) > 1 else "single"})

        # Optional secondary gate: a separate channel whose vote share must clear
        # a threshold before we answer at all. Used for the liveness signal on
        # personHasCityOfDeath, where the largest error family is naming a city
        # for someone still alive. Recorded in provenance so a submission is
        # never silently gated.
        gate_cfg = cfg.get("gate")
        gate_pool, gate_fn = None, None
        if gate_cfg:
            gspec, gate_pool = _pool_for(relation, gate_cfg["channel"], split, args)
            gate_fn = CHANNELS[gate_cfg["channel"]].parse
            provenance[-1]["gate"] = {"channel": gate_cfg["channel"],
                                      "tau": gate_cfg["tau"],
                                      "pool_id": gspec.pool_id()}

        subjects = {s for p in pools.values() for s in p}
        for subj in subjects:
            if gate_pool is not None:
                gd = gate_pool.get(subj, [])
                share = (sum(1 for d in gd if gate_fn(d)) / len(gd)) if gd else 0.0
                if share < gate_cfg["tau"]:
                    preds[(subj, relation)] = []
                    continue
            per_ch = {nm: p.get(subj, []) for nm, p in pools.items()}
            if relation in NUMERIC_RELATIONS:
                if len(names) > 1:
                    out, _ = predict_numeric_consensus(
                        per_ch, channel_agreement_weight=cfg.get("w", 1.0),
                        scale=cfg["param"])
                else:
                    out = predict_numeric(per_ch[names[0]], names[0],
                                          scale=cfg["param"],
                                          method=cfg.get("method", "auto"),
                                          antiround=cfg.get("antiround", 0.0))
            else:
                if len(names) > 1:
                    out = predict_set_consensus(
                        per_ch, tau=cfg["param"],
                        channel_agreement_weight=cfg.get("w", 1.0))
                else:
                    out = predict_set(per_ch[names[0]], names[0], tau=cfg["param"],
                                      emit_ratio=cfg.get("emit_ratio", 0.0))
            preds[(subj, relation)] = out

    rows = [{"SubjectEntity": g["SubjectEntity"], "Relation": g["Relation"],
             "ObjectEntities": [str(o) for o in preds.get(
                 (g["SubjectEntity"], g["Relation"]), [])]}
            for g in gold_rows]
    return rows, {"provenance": provenance}


def presubmit_checks(rows: list[dict], split: str, dataset_dir: Path,
                     strict_gold: bool) -> dict:
    report: dict = {}
    gold_rows = load_split(split)

    ours = {(r["SubjectEntity"], r["Relation"]) for r in rows}
    theirs = {(r["SubjectEntity"], r["Relation"]) for r in gold_rows}
    report["row_count"] = len(rows)
    report["subject_set_matches_gold"] = ours == theirs
    if ours != theirs:
        raise PreSubmitFailure(
            f"subject set mismatch: {len(ours - theirs)} extra, {len(theirs - ours)} missing. "
            f"examples missing: {list(theirs - ours)[:5]}")
    if len(rows) != len(gold_rows):
        raise PreSubmitFailure(f"row count {len(rows)} != gold {len(gold_rows)}")

    if split == "test":
        expected = sum(TEST_ROWS.values())
        if len(rows) != expected:
            raise PreSubmitFailure(f"test rows {len(rows)} != expected {expected}")

    local, remote = gold_head(dataset_dir)
    report["gold_head_local"] = local
    report["gold_head_origin_main"] = remote
    report["gold_fresh"] = (local == remote and bool(local))
    if strict_gold and not report["gold_fresh"]:
        raise PreSubmitFailure(
            f"gold moved: local {local[:12]} != origin/main {remote[:12]}. "
            "Rebuild the affected rows against the NEW subject strings before "
            "uploading. Do not re-key cached pools onto new names: the "
            "disambiguation suffix is prompt input, so a renamed subject is a "
            "different prompt.")

    bad_numeric = []
    for r in rows:
        if r["Relation"] in NUMERIC_RELATIONS:
            for o in r["ObjectEntities"]:
                if "," in o or not o.replace(".", "", 1).isdigit():
                    bad_numeric.append((r["SubjectEntity"], o))
            if len(r["ObjectEntities"]) > 1:
                bad_numeric.append((r["SubjectEntity"], f"{len(r['ObjectEntities'])} values"))
    report["numeric_violations"] = bad_numeric[:20]
    if bad_numeric:
        raise PreSubmitFailure(f"{len(bad_numeric)} numeric violations: {bad_numeric[:5]}")

    non_str = [(r["SubjectEntity"], o) for r in rows for o in r["ObjectEntities"]
               if not isinstance(o, str)]
    if non_str:
        raise PreSubmitFailure(f"non-string objects: {non_str[:5]}")

    report["empty_rows"] = sum(1 for r in rows if not r["ObjectEntities"])
    report["empty_fraction"] = round(report["empty_rows"] / max(len(rows), 1), 4)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help='JSON: {"relation": {"channel": ..., "param": ...}, ...}')
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--model", default="google/gemma-4-31B")
    ap.add_argument("--revision", default="5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--demo-seed", type=int, default=1234)
    ap.add_argument("--seed-base", type=int, default=7000)
    ap.add_argument("--allow-stale-gold", action="store_true",
                    help="only for local val experiments, never for a real upload")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    rows, meta = build(cfg, args.split, args)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = presubmit_checks(rows, args.split, REPO / "dataset2026",
                              strict_gold=not args.allow_stale_gold)
    report["file"] = str(out)
    report["sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    report["provenance"] = meta["provenance"]

    audit = json.loads((REPO / "audit" / "param_budget.json").read_text())
    report["substrate_manifest"] = {
        "models": audit["inference_time_neural_components"],
        "total_parameters": audit["total_inference_time_parameters"],
        "cap": audit["cap"],
        "under_cap": audit["under_cap"],
    }
    if not report["substrate_manifest"]["under_cap"]:
        raise PreSubmitFailure("substrate manifest exceeds the 32B cap")

    if args.split != "test":
        from scorer import score_all
        s = score_all(rows, args.split)
        report["scores"] = {k: round(v, 4) for k, v in s.items()}

    print(json.dumps(report, indent=1))
    rp = out.with_suffix(".report.json")
    rp.write_text(json.dumps(report, indent=1))
    print(f"\nPRE-SUBMIT CHECKS PASSED. report -> {rp}")
    print("Zip with:  zip -X -j submission.zip " + str(out))
    print("The inner file MUST be named predictions.jsonl.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
