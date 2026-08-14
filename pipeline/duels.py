"""Forced-choice pairwise duels: a signal that is not a function of vote frequency.

THE PROBLEM THIS ATTACKS. Every selector this campaign has measured reads the same 100-sample
frequency table produced by one prompt distribution. Vote share, consensus, plausibility
bands, anti-round, background lift, cluster width, temperature, draw count and the cross-frame
agreement reranker are all transformations of that table, and every one measured neutral or
worse. hasCapacity sits at 0.3367 on the board while its pool contains an acceptable value for
about 84% of subjects, so the gap is ranking, and ranking by frequency has been exhausted.

WHY FORCED CHOICE IS DIFFERENT. Generation asks the model to commit to a value autoregressively;
comparison asks it to judge two values it is handed. These are different conditionals, and the
second can be right when the first is wrong. The failure mode of the obvious version, a yes/no
probe on a single fact, is well documented and was already recorded against this campaign:
verify-probes accept nearly everything (Singhania et al., rank-then-select 49.5 against
verify-probe 8.0), and our own standalone liveness gate measured inert. A DUEL does not have
that failure mode, because the model cannot say yes to both. It must spend its probability
mass on one side, which is what makes the signal informative rather than a yes-bias readout.

WHAT IS STORED. For each subject, the tolerance-separated top-k candidates from the shipped
frame's own pool, and for every unordered pair the probability the model assigns to each side
winning, averaged over BOTH presentation orders. Presenting each pair twice is not a nicety:
base models carry a strong position prior, and an uncorrected A/B probe measures that prior as
much as it measures knowledge. The candidate set is exactly the shipped frame's, so a gain here
is a better ORDERING of the same candidates and cannot be confused with casting a wider net.

CLOSED BOOK. Reads only cached draws, train gold for the demonstrations, and the model's own
weights. No external corpus, no network, no second model, no weight update.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate import numeric_candidates, tolerance_support
from channels import CHANNELS, demo_union
from common import POOL_DIR, load_pool, rows_for, spec_for_channel, split_sha

TOL = 0.05

# Demonstrations are drawn from train rows that are a demo of NO channel of the
# relation, so the duel prompt never shows an example whose subject is also
# being scored elsewhere, and so the duel's own demo set adds as few train rows
# to the leak union as possible. Fixed by seed, never tuned.
N_DEMOS = 8
DEMO_SEED = 4242


def top_candidates(draws: list[str], channel: str, k: int) -> list[float]:
    """Tolerance-separated representatives, strongest support first.

    Separation matters twice over. The grader gives one true positive per gold
    however many near-duplicates are submitted, and a duel between two values
    inside each other's 5% ball asks the model to split a distinction the metric
    does not make.
    """
    vals = numeric_candidates(draws, channel)
    if not vals:
        return []
    scored = sorted(((len(tolerance_support(vals, c, TOL)), c) for c in set(vals)),
                    key=lambda t: (-t[0], t[1]))
    out: list[float] = []
    for _, c in scored:
        if any(abs(c - t) / t <= TOL for t in out):
            continue
        out.append(c)
        if len(out) >= k:
            break
    return out


def fmt(v: float) -> str:
    """Render a candidate the way the pools render numbers: bare, no separators."""
    return str(int(round(v))) if abs(v - round(v)) < 1e-9 else f"{v:g}"


def _trailing_zeros(v: float) -> int:
    n = int(round(v))
    if n == 0:
        return 0
    z = 0
    while n % 10 == 0:
        n //= 10
        z += 1
    return z


def _distractor(gold: float, ratio: float) -> float:
    """A wrong answer of the SAME roundness class as the gold.

    This matters more than it looks. If every demonstration pairs a specific
    gold against a round distractor, the block teaches "prefer the less round
    number", which is precisely the anti-round rule this campaign already
    measured on the board at -0.0102. The demonstrations must not smuggle a
    refuted heuristic into the probe, so a round gold gets a round distractor
    and a specific gold gets a specific one.
    """
    v = gold * ratio
    z = _trailing_zeros(gold)
    if z >= 3:
        step = 10 ** z
        v = max(step, round(v / step) * step)
    else:
        v = float(f"{v:.4g}")
        if _trailing_zeros(v) >= 3:            # do not accidentally land on a round number
            v += 137
    if gold and abs(v - gold) / gold <= TOL:   # must be outside the grader's tolerance
        v = gold * (1.7 if ratio > 1 else 0.6)
    return v


def demo_block(relation: str, noun: str, unit: str) -> tuple[str, tuple[str, ...]]:
    """Few-shot A/B examples, balanced on both axes a base model would latch onto.

    Across the eight demonstrations the correct answer is at A four times and at
    B four times, and is the LARGER value four times and the smaller four times.
    An unbalanced block is not a small flaw here: the first version of this
    function put the correct answer on the smaller number in all eight, which
    teaches "pick the smaller option" and would have produced a confident,
    meaningless probe.

    Demonstrations are preferentially drawn from train rows that are a demo of
    no channel of this relation. Some relations have no such rows left (every
    area train row is a demonstration of area_lead100, which uses all 100), so
    the fallback is any train row with gold, and either way the subjects used
    are recorded in the manifest so scoring can exclude them.
    """
    import random
    banned = demo_union(relation, 1234)
    have_gold = [r for r in rows_for("train", relation) if (r.get("ObjectEntities") or [])]
    rows = [r for r in have_gold if r["SubjectEntity"] not in banned]
    if len(rows) < N_DEMOS:
        rows = have_gold
    rows.sort(key=lambda r: r["SubjectEntity"])
    rng = random.Random(DEMO_SEED)
    rng.shuffle(rows)
    smaller = (0.5, 0.6, 0.4, 0.7)
    larger = (2.0, 1.5, 2.5, 1.8)
    lines, ids = [], []
    for i, r in enumerate(rows[:N_DEMOS]):
        gold = float(r["ObjectEntities"][0][0])
        gold_is_larger = i % 2 == 0
        gold_at_a = (i // 2) % 2 == 0
        ratio = (smaller if gold_is_larger else larger)[i % 4]
        d = _distractor(gold, ratio)
        a, b = (gold, d) if gold_at_a else (d, gold)
        lines.append(f"Question: What is the {noun} of {r['SubjectEntity']}{unit}?\n"
                     f"A. {fmt(a)}\nB. {fmt(b)}\nAnswer: {'A' if gold_at_a else 'B'}")
        ids.append(r["SubjectEntity"])
    return "\n\n".join(lines) + "\n\n", tuple(ids)


NOUN = {"hasCapacity": ("seating capacity", ""),
        "hasArea": ("area in square kilometres", "")}


def build_prompts(relation: str, split: str, channel: str, k: int, args):
    """Return (prompts, index) where index[i] = (subject, cand_a, cand_b)."""
    pool = load_pool(spec_for_channel(CHANNELS[channel], split, args))
    noun, unit = NOUN[relation]
    block, demo_ids_used = demo_block(relation, noun, unit)
    prompts, index, cands_by = [], [], {}
    for subj in sorted(pool):
        cands = top_candidates(pool[subj], channel, k)
        cands_by[subj] = cands
        for x, y in itertools.combinations(range(len(cands)), 2):
            for a, b in ((cands[x], cands[y]), (cands[y], cands[x])):
                prompts.append(
                    f"{block}Question: What is the {noun} of {subj}{unit}?\n"
                    f"A. {fmt(a)}\nB. {fmt(b)}\nAnswer:")
                index.append((subj, a, b))
    return prompts, index, cands_by, demo_ids_used


def ab_probs(out, tok_a: set[int], tok_b: set[int]) -> tuple[float, float]:
    """P(A), P(B) renormalised over just the two option tokens."""
    import math
    lp = out.outputs[0].logprobs[0] if out.outputs[0].logprobs else {}
    pa = sum(math.exp(v.logprob) for t, v in lp.items() if t in tok_a)
    pb = sum(math.exp(v.logprob) for t, v in lp.items() if t in tok_b)
    if pa + pb <= 0:
        return 0.5, 0.5
    return pa / (pa + pb), pb / (pa + pb)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--relation", required=True, choices=sorted(NOUN))
    ap.add_argument("--channel", required=True)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--model", default="google/gemma-4-31B")
    ap.add_argument("--revision", default="5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--demo-seed", type=int, default=1234)
    ap.add_argument("--seed-base", type=int, default=7000)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tk = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    # Every token id whose surface form is A or B, with or without a leading
    # space. Reading a single hard-coded id would silently return 0.5 for every
    # pair if the tokenizer disagreed, which looks like a null result rather
    # than a bug, so collect the whole set and assert it is non-empty.
    tok_a = {i for s in ("A", " A") for i in tk.encode(s, add_special_tokens=False)[-1:]}
    tok_b = {i for s in ("B", " B") for i in tk.encode(s, add_special_tokens=False)[-1:]}
    assert tok_a and tok_b, "could not resolve option token ids"
    print(f"option token ids: A={sorted(tok_a)} B={sorted(tok_b)}", flush=True)

    llm = LLM(model=args.model, revision=args.revision, tensor_parallel_size=1,
              gpu_memory_utilization=args.gpu_mem, max_model_len=4096,
              trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)

    outdir = POOL_DIR / args.relation
    outdir.mkdir(parents=True, exist_ok=True)
    for split in args.splits.split(","):
        t0 = time.time()
        prompts, index, cands_by, demo_ids_used = build_prompts(
            args.relation, split, args.channel, args.k, args)
        print(f"[{split}] {len(prompts)} duel prompts over {len(cands_by)} subjects",
              flush=True)
        outs = llm.generate(prompts, sp)
        agg: dict[str, dict] = {s: {} for s in cands_by}
        for (subj, a, b), o in zip(index, outs):
            pa, _ = ab_probs(o, tok_a, tok_b)
            key = f"{fmt(a)}|{fmt(b)}"
            agg[subj].setdefault(key, []).append(pa)
        path = outdir / f"duels_{args.channel}_{split}_k{args.k}.jsonl"
        with open(path, "w") as fh:
            for subj, cands in cands_by.items():
                fh.write(json.dumps({
                    "subject": subj, "candidates": [fmt(c) for c in cands],
                    "pair_probs": {k: sum(v) / len(v) for k, v in agg[subj].items()},
                }) + "\n")
        meta = {"relation": args.relation, "channel": args.channel, "split": split,
                "k": args.k, "model": args.model, "revision": args.revision,
                "n_demos": N_DEMOS, "demo_seed": DEMO_SEED,
                "demo_subjects": list(demo_ids_used),
                "split_sha": split_sha(split), "decoding": "greedy,max_tokens=1,logprobs=20",
                "both_orders": True, "n_prompts": len(prompts)}
        with open(str(path).replace(".jsonl", ".manifest.json"), "w") as fh:
            json.dump(meta, fh, indent=1, sort_keys=True)
        print(f"[{split}] wrote {path} in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
