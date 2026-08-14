"""Generate a vote pool for one (relation, channel, split) with vLLM offline batch.

Offline batch, not a served endpoint: each job is self-contained, writes one
pool keyed by POOL-ID, and exits. No port coordination, no orphaned servers.

Draws are append-only. Re-running with a larger --n extends the existing pool
using seeds base+i, so escalating n=30 -> 100 costs only the 70 new draws.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from channels import CHANNELS, demo_ids, pick_demos            # noqa: E402
from common import PoolSpec, rows_for, split_sha               # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", required=True, choices=sorted(CHANNELS))
    ap.add_argument("--split", required=True, choices=("train", "val", "test"))
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
    ap.add_argument("--limit", type=int, default=0, help="debug: first K subjects")
    args = ap.parse_args()

    ch = CHANNELS[args.channel]
    demos = pick_demos(ch.relation, ch.n_demos, args.demo_seed, ch.demo_strategy)

    spec = PoolSpec(
        relation=ch.relation,
        split=args.split,
        channel=ch.name,
        model_id=args.model,
        model_revision=args.revision,
        prompt_template=ch.render("<<SUBJECT>>", demos),
        demo_ids=demo_ids(demos),
        demo_seed=args.demo_seed,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=ch.max_tokens,
        stop=ch.stop,
        seed_base=args.seed_base,
        split_sha=split_sha(args.split),
    )
    pool_path = spec.path()
    pool_path.parent.mkdir(parents=True, exist_ok=True)

    rows = rows_for(args.split, ch.relation)
    if args.limit:
        rows = rows[: args.limit]
    subjects = [r["SubjectEntity"] for r in rows]

    existing: dict[str, list[str]] = {}
    if pool_path.exists():
        with open(pool_path) as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    existing[rec["subject"]] = rec["draws"]

    todo = [(s, args.n - len(existing.get(s, []))) for s in subjects]
    todo = [(s, k) for s, k in todo if k > 0]
    print(f"[pool {spec.pool_id()}] {ch.relation}/{ch.name}/{args.split} "
          f"subjects={len(subjects)} need_draws={sum(k for _, k in todo)}", flush=True)
    if not todo:
        print("pool already complete; nothing to generate")
        spec.write_manifest()
        return 0

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
    print(f"model loaded in {time.time()-t0:.0f}s", flush=True)

    # One request per subject with n forks: the forks share the prompt's KV
    # blocks, and every subject shares the demo prefix, so prefix caching hits.
    prompts = [ch.render(s, demos) for s, _ in todo]
    max_needed = max(k for _, k in todo)
    sp = SamplingParams(
        n=max_needed,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=ch.max_tokens,
        stop=list(ch.stop),
        seed=args.seed_base,
    )
    t1 = time.time()
    outs = llm.generate(prompts, sp)
    gen_s = time.time() - t1

    n_tok = 0
    for (subj, k), out in zip(todo, outs):
        new = [o.text for o in out.outputs][:k]
        n_tok += sum(len(o.token_ids) for o in out.outputs[:k])
        existing.setdefault(subj, []).extend(new)

    tmp = pool_path.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        for s in subjects:
            fh.write(json.dumps({"subject": s, "draws": existing.get(s, [])},
                                ensure_ascii=False) + "\n")
    os.replace(tmp, pool_path)
    spec.write_manifest()

    print(f"WROTE {pool_path}")
    print(f"POOL_ID {spec.pool_id()}")
    print(f"gen_seconds {gen_s:.0f} decode_tokens {n_tok} tok_per_s {n_tok/max(gen_s,1):.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
