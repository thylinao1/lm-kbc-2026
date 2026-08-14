"""Yes/no verification probe: the last untested elicitation framing for hasCapacity.

WHY A THIRD PROBE, AND WHY THE BAR IS HIGHER NOT LOWER. Forced-choice duels already measured
DEAD, and their diagnostics said the model's comparative knowledge beyond a magnitude prior is
z = 0.7. This is the third attempt on the same relation, so the honest response to multiple
testing is to raise the bar, not to keep looking until something clears. The rule below is
fixed before the pools exist and there is exactly one look.

WHY IT IS STILL WORTH RUNNING. A duel asks "which of these two", a verification asks "is this
one true". They are different conditionals, and the documented failure mode of verification
probes is the opposite of a duel's: duels are forced to split their mass and so inherit a
position or magnitude prior, while verifiers can accept everything. If the two framings agree
that there is nothing here, the relation is closed by two independent instruments rather than
one. If they disagree, that is worth knowing before the paper claims the relation is
knowledge bound.

PRE-COMMITTED RULE, zero free parameters:
    score(candidate) = logprob(" yes") - logprob(" no") at the single next token.
    Prediction = argmax over the shipped frame's own top-6 tolerance-separated candidates.
    Ties broken by the shipped frame's vote share.
PRE-COMMITTED SHIP RULE:
    SHIP only if val macro-F1 beats the shipped selector with a paired bootstrap 90% CI
    excluding zero, AND the sign holds inside BOTH the split-half-stable and split-half-
    unstable strata. Anything else is DEAD. The two-strata condition is the extra bar this
    third attempt has to clear.

ACCEPT-EVERYTHING CONTROL, reported whatever the verdict: the mean margin and the fraction of
candidates the model would accept at margin > 0. A probe that accepts nearly everything is a
yes-bias readout, and its argmax is then just the ranking of that bias.

CLOSED BOOK. Demonstrations use train gold only. False claims are drawn from that same demo
subject's own cached pool, excluding anything within the grader's tolerance of its gold, so no
value is invented and nothing external is read.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate import numeric_candidates
from channels import CHANNELS, demo_union
from common import POOL_DIR, load_pool, rows_for, spec_for_channel, split_sha
from duels import fmt, top_candidates

TOL = 0.05
N_DEMOS = 8
DEMO_SEED = 4242
STEM = "Claim: its maximum spectator capacity is"


def demo_block(args) -> tuple[str, tuple[str, ...]]:
    """Balanced yes/no demonstrations, four of each, alternating so the block
    never teaches a run of one answer."""
    banned = demo_union("hasCapacity", 1234)
    have = [r for r in rows_for("train", "hasCapacity") if (r.get("ObjectEntities") or [])]
    rows = [r for r in have if r["SubjectEntity"] not in banned] or have
    rows.sort(key=lambda r: r["SubjectEntity"])
    rng = random.Random(DEMO_SEED)
    rng.shuffle(rows)
    pool = load_pool(spec_for_channel(CHANNELS["cap_recite"], "train", args))
    lines, ids = [], []
    for i, r in enumerate(rows):
        if len(lines) >= N_DEMOS:
            break
        s = r["SubjectEntity"]
        gold = float(r["ObjectEntities"][0][0])
        say_yes = i % 2 == 0
        if say_yes:
            claim = gold
        else:
            # a wrong claim the model itself proposed for THIS subject
            cands = [c for c in top_candidates(pool.get(s, []), "cap_recite", 12)
                     if abs(c - gold) / gold > TOL]
            if not cands:
                continue
            claim = cands[0]
        lines.append(f"### {s}\n{STEM} {fmt(claim)}.\nCorrect: {'yes' if say_yes else 'no'}")
        ids.append(s)
    return ("Fact checking of sports venue records.\n\n" + "\n\n".join(lines) + "\n\n",
            tuple(ids))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="cap_recite")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--model", default="google/gemma-4-31B")
    ap.add_argument("--revision", default="5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--demo-seed", type=int, default=1234)
    ap.add_argument("--seed-base", type=int, default=7000)
    ap.add_argument("--gpu-mem", type=float, default=0.94)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tk = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    yes = {i for s in ("yes", " yes") for i in tk.encode(s, add_special_tokens=False)[-1:]}
    no = {i for s in ("no", " no") for i in tk.encode(s, add_special_tokens=False)[-1:]}
    assert yes and no, "could not resolve yes/no token ids"
    print(f"token ids: yes={sorted(yes)} no={sorted(no)}", flush=True)

    block, demo_ids_used = demo_block(args)
    print(f"demo block: {len(demo_ids_used)} demonstrations\n{block[:600]}", flush=True)

    llm = LLM(model=args.model, revision=args.revision, tensor_parallel_size=1,
              gpu_memory_utilization=args.gpu_mem, max_model_len=4096,
              trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)

    import math
    outdir = POOL_DIR / "hasCapacity"
    outdir.mkdir(parents=True, exist_ok=True)
    for split in args.splits.split(","):
        pool = load_pool(spec_for_channel(CHANNELS[args.channel], split, args))
        prompts, index, cands_by = [], [], {}
        for subj in sorted(pool):
            cands = top_candidates(pool[subj], args.channel, args.k)
            cands_by[subj] = cands
            for c in cands:
                prompts.append(f"{block}### {subj}\n{STEM} {fmt(c)}.\nCorrect:")
                index.append((subj, c))
        print(f"[{split}] {len(prompts)} verification prompts over {len(cands_by)} subjects",
              flush=True)
        outs = llm.generate(prompts, sp)
        agg: dict[str, dict] = {s: {} for s in cands_by}
        miss = 0
        for (subj, c), o in zip(index, outs):
            lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
            ly = max((v.logprob for t, v in lp.items() if t in yes), default=None)
            ln = max((v.logprob for t, v in lp.items() if t in no), default=None)
            if ly is None or ln is None:
                miss += 1
                # neither token in the top 20 means the probe had no opinion here;
                # record a neutral margin rather than silently dropping the candidate
                ly, ln = 0.0, 0.0
            agg[subj][fmt(c)] = ly - ln
        path = outdir / f"verify_{args.channel}_{split}_k{args.k}.jsonl"
        with open(path, "w") as fh:
            for subj, cands in cands_by.items():
                fh.write(json.dumps({"subject": subj,
                                     "candidates": [fmt(c) for c in cands],
                                     "margins": agg[subj]}) + "\n")
        meta = {"relation": "hasCapacity", "channel": args.channel, "split": split,
                "k": args.k, "model": args.model, "revision": args.revision,
                "n_demos": len(demo_ids_used), "demo_seed": DEMO_SEED,
                "demo_subjects": list(demo_ids_used), "split_sha": split_sha(split),
                "decoding": "greedy,max_tokens=1,logprobs=20", "stem": STEM,
                "n_prompts": len(prompts), "token_fallbacks": miss}
        with open(str(path).replace(".jsonl", ".manifest.json"), "w") as fh:
            json.dump(meta, fh, indent=1, sort_keys=True)
        allm = [m for d in agg.values() for m in d.values()]
        print(f"[{split}] wrote {path}; token fallbacks {miss}; "
              f"mean margin {sum(allm)/max(len(allm),1):+.3f}; "
              f"accepted at margin>0: {sum(1 for m in allm if m>0)/max(len(allm),1):.3f}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
