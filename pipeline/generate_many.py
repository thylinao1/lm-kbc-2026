"""Generate several (channel, split) pools from ONE model load.

The per-pool entry point (generate.py) reloads a 62GB checkpoint every
invocation, which costs several minutes each. A sweep over five capacity frames
on two splits would spend more wall clock loading weights than sampling. This
runner loads once and iterates.

Pool identity, append-only draw semantics and manifests are unchanged: this
calls the same PoolSpec machinery, so a pool produced here is byte-compatible
with one produced by generate.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from channels import CHANNELS, demo_ids, pick_demos, pick_demos_nn  # noqa: E402
from common import PoolSpec, rows_for, split_sha          # noqa: E402


def build_spec(ch, split, args, demos) -> PoolSpec:
    return PoolSpec(
        relation=ch.relation, split=split, channel=ch.name,
        model_id=args.model, model_revision=args.revision,
        prompt_template=ch.render("<<SUBJECT>>", demos),
        demo_ids=demo_ids(demos), demo_seed=args.demo_seed,
        temperature=args.temperature, top_p=args.top_p,
        max_tokens=ch.max_tokens, stop=ch.stop, seed_base=args.seed_base,
        split_sha=split_sha(split),
    )


def run_one(llm, SamplingParams, ch, split, args) -> dict:
    nn = ch.demo_strategy == "nn"
    if nn:
        # Per-subject demos. The pool identity cannot list concrete demo ids
        # because they differ per subject, so it records the STRATEGY plus the
        # split whose rows the neighbours are drawn from. That is still a
        # complete description: given the strategy, seed and train file hash,
        # the demo set for any subject is reproducible.
        demos = []
        spec = build_spec(ch, split, args, [])
        spec = PoolSpec(**{**spec.identity(),
                           "demo_ids": ("nn:strategy", f"train:{split_sha('train')[:16]}"),
                           "stop": tuple(spec.stop)})
    else:
        demos = pick_demos(ch.relation, ch.n_demos, args.demo_seed, ch.demo_strategy)
        spec = build_spec(ch, split, args, demos)
    path = spec.path()
    path.parent.mkdir(parents=True, exist_ok=True)

    subjects = [r["SubjectEntity"] for r in rows_for(split, ch.relation)]
    existing: dict[str, list[str]] = {}
    if path.exists():
        with open(path) as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    existing[rec["subject"]] = rec["draws"]

    todo = [(s, args.n - len(existing.get(s, []))) for s in subjects]
    todo = [(s, k) for s, k in todo if k > 0]
    if not todo:
        spec.write_manifest()
        return {"channel": ch.name, "split": split, "pool_id": spec.pool_id(),
                "status": "already-complete", "subjects": len(subjects)}

    if nn:
        prompts = [ch.render(s, pick_demos_nn(ch.relation, ch.n_demos, s, args.demo_seed))
                   for s, _ in todo]
    else:
        prompts = [ch.render(s, demos) for s, _ in todo]
    sp = SamplingParams(
        n=max(k for _, k in todo), temperature=args.temperature, top_p=args.top_p,
        max_tokens=ch.max_tokens, stop=list(ch.stop), seed=args.seed_base,
    )
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t0

    n_tok = 0
    for (subj, k), out in zip(todo, outs):
        n_tok += sum(len(o.token_ids) for o in out.outputs[:k])
        existing.setdefault(subj, []).extend(o.text for o in out.outputs[:k])

    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        for s in subjects:
            fh.write(json.dumps({"subject": s, "draws": existing.get(s, [])},
                                ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    spec.write_manifest()
    return {"channel": ch.name, "split": split, "pool_id": spec.pool_id(),
            "status": "ok", "subjects": len(subjects),
            "prompts": len(prompts), "decode_tokens": n_tok,
            "seconds": round(dt, 1), "tok_per_s": round(n_tok / max(dt, 1))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", required=True,
                    help="comma-separated channel names")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--model", default="google/gemma-4-31B")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--demo-seed", type=int, default=1234)
    ap.add_argument("--seed-base", type=int, default=7000)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--tensor-parallel", type=int, default=1)
    args = ap.parse_args()

    names = [c.strip() for c in args.channels.split(",") if c.strip()]
    for nm in names:
        if nm not in CHANNELS:
            raise SystemExit(f"unknown channel {nm!r}; have {sorted(CHANNELS)}")
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    from vllm import LLM, SamplingParams  # noqa: E402
    t0 = time.time()
    llm = LLM(
        model=args.model,
        revision=None if args.revision == "main" else args.revision,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        enable_prefix_caching=True,
        trust_remote_code=True,
        seed=args.seed_base,
    )
    load_s = time.time() - t0
    print(f"MODEL LOADED in {load_s:.0f}s (once for {len(names)}x{len(splits)} pools)",
          flush=True)

    results = []
    for nm in names:
        for sp_name in splits:
            print(f"\n########## {nm} / {sp_name} ##########", flush=True)
            try:
                r = run_one(llm, SamplingParams, CHANNELS[nm], sp_name, args)
            except Exception as exc:  # keep the sweep alive; a dead channel is data
                r = {"channel": nm, "split": sp_name, "status": "FAILED",
                     "error": f"{type(exc).__name__}: {exc}"}
            results.append(r)
            print(json.dumps(r), flush=True)

    print("\n=== SUMMARY ===")
    for r in results:
        print(json.dumps(r))
    ok = sum(1 for r in results if r.get("status") in ("ok", "already-complete"))
    print(f"pools ok: {ok}/{len(results)}  model_load_s: {load_s:.0f}")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
