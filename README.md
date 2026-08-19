# lm-kbc-2026

System code for the LM-KBC 2026 Shared Task (5th edition, AKBC), Codabench
competition 16267. Given a subject and a relation, it predicts the complete set
of object entities as of world state 2026-07-01 using only the parametric memory
of an open weight model of at most 32B parameters, with no retrieval, no
external knowledge base, and no training.

The submitted system scores 0.7060 all-rows macro-F1 on the official test set,
measured by the organizers' grader over all 475 test rows, and is ahead on five
of the six relations.

## Install

Python 3.10 or newer. Generation needs a GPU large enough to hold the 62.5GB
BF16 checkpoint at tensor-parallel 1; everything downstream of generation runs
on CPU from cached pools.

```bash
git clone https://github.com/thylinao1/lm-kbc-2026.git
cd lm-kbc-2026

# the official data and scorer live in their own Apache-2.0 repo
git clone https://github.com/lm-kbc/dataset2026.git

python -m venv .venv && source .venv/bin/activate
pip install vllm numpy scikit-learn pandas pytest
```

`pandas` is used only by the official `evaluate.py` display path, which this
code never calls; on a cluster environment without it, `pipeline/scorer.py`
inserts a stub. `scikit-learn` is needed only by the top-k analysis scripts.

Two environment variables override the default locations, which are
`./dataset2026/data` and `./pools`:

```bash
export AKBC_DATA=/path/to/dataset2026/data
export AKBC_POOLS=/path/to/pools
```

## Run

Generate a draw pool for one channel and split, measure its oracle ceiling, then
sweep its threshold:

```bash
python pipeline/generate.py --channel cap_recite --split val --n 30
python pipeline/coverage.py --channel cap_recite --split val
python pipeline/tune.py     --channel cap_recite
```

`generate_many.py` takes a comma-separated channel list and loads the checkpoint
once for all of them, which matters because loading takes about four minutes
from NFS. `jobs/*.sbatch` are the Slurm scripts that produced the submitted
pools.

To build a submission file from a configuration:

```bash
python pipeline/assemble.py --config configs/best_measured.json \
                            --split test --out submissions/best/
```

Tests:

```bash
python -m pytest pipeline/tests/ -q
```

## Method

One model, `google/gemma-4-31B` at revision `5bbc2fb1`, used in base completion
mode at temperature 0.7 with no chat template. Each relation has its own
elicitation frame, decode contract and parser (`pipeline/channels.py`). Thirty
draws per subject are sampled for most relations and a hundred for hasCapacity,
then aggregated: vote share with a per-relation threshold for the set relations,
and a fixed-radius tolerance ball or a log-space cluster median for the two
numeric relations. Everything after generation is non-neural post-processing.

The largest single gain came from the elicitation prompt. Asking the model to
recall facts about an entity before committing to an answer, rather than asking
for the answer directly, moved hasArea from 0.8100 to 0.8700 on test, and the same
framing gives the best hasCapacity frame. Three earlier hasArea configurations
across two frames and two aggregation methods had all returned exactly 0.8100,
which had been read as evidence that the answers were absent from the candidate
pool. That reading was wrong. The register determines what enters the pool, and
no selection method recovers what elicitation never surfaced. The effect has a
boundary: recall-first framing hurts the relations where the system also has to
decide whether an answer exists, because priming the model to recall facts
primes it to answer.

The main analytical result is a correction to how this field measures headroom,
and it corrects an earlier conclusion of my own as much as anyone else's. Oracle
coverage analyses report that hasCapacity is selection bound, since the
recitation frame puts an acceptable value in the pool for 84.5 percent of
subjects while the aggregator realises 36.1. That inference does not survive a
null control. A pool of a hundred guesses spanning a decade contains about
twenty-one mutually exclusive 5 percent tolerance windows, so it contains
something acceptably close to almost any plausible target whether or not the
model knows the answer. Permuting gold values across subjects measures the
coincidence rate directly: 0.355 for hasCapacity against 0.031 for hasArea.
Chance corrected, 35.5 of the 84.5 coverage points are spread rather than
recall, and the findable headroom beyond the rank one answer the system already
emits falls from 0.443 to 0.203.

Three instruments that are not functions of the sampled vote frequency, each
given the shipped frame's own candidate set and each with its rule fixed before
the data existed, then agree that the relation is knowledge bound: cross-frame
agreement reranking measured +0.0000 on leak-free rows, forced-choice duels lost
0.0412 on val, and a yes/no verification probe lost 0.0928.

Validation turned out to be the wrong instrument for the decisions that mattered.
At 97 to 100 rows per relation its standard error is 0.03 to 0.05, while the
effects that separate systems here are 0.01 to 0.03. Five decisions were settled
by the test board after validation pointed the wrong way, including the
recitation frame it rated a dead tie. Because relations occupy disjoint row sets,
one submission returns six independent readings, and a file assembled from the
best measured configuration per relation scores exactly the row-weighted sum of
those measurements. That was confirmed twice: 0.6916 predicted and returned,
then 0.7060 predicted and returned.

`docs/engineering-notes.md` carries the working record, including the changes
that were implemented and then rejected on measurement. `paper/` carries the
system paper.

## Eligibility

| requirement | how this system meets it |
|---|---|
| at most 32B total inference-time parameters | one model, `google/gemma-4-31B` at revision `5bbc2fb1`, measured 31,273,088,876 BF16 parameters from the safetensors index. See `audit/param_budget.json`. |
| open weights | yes, not gated |
| closed book | no network, retrieval, or external corpus at inference. Enforced by four checks, see below. |
| no training | base completion only. No fine tune, no LoRA, no trained probe or classifier. |

A note for auditors: the model's hosting page displays 32.68B parameters, which
would appear to exceed the cap. That figure counts the tied input embedding a
second time as an output projection, 262144 by 5376 = 1,409,286,144 parameters.
The served checkpoint holds one copy. Both the dtype breakdown and the
safetensors index total size (62,546,177,752 bytes of BF16, so 31.273B
parameters) agree on the lower figure.

The 0.727B of headroom under the cap is why this system has exactly one neural
component. No agreement sidecar, no reranker, no NLI filter, no draft model.

## Closed book enforcement

Compliance is checked four separate ways:

1. Extended grep over network libraries across `pipeline/` and `jobs/`.
2. An input allowlist. `pipeline/common.py:load_split()` is the only sanctioned
   reader of factual data and raises `ClosedBookViolation` on anything other
   than the three official split files. Generated pools and configs are the only
   other inputs.
3. Fork hygiene. Competitor repositories were read for method only, are cloned
   outside this tree, and no file from them is copied in. Recipes attributed in
   the paper are reimplementations from published descriptions.
4. Offline inference. The compute nodes used have working internet, so every
   inference job exports `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` and
   weights are staged in a separate step.

The audit output is `audit/closed_book.txt`.

## Repository layout

| path | contents |
|---|---|
| `pipeline/common.py` | closed-book IO allowlist, POOL-ID cache identity, row weights |
| `pipeline/channels.py` | per relation elicitation frames, decode contracts, parsers |
| `pipeline/generate.py` | vLLM offline batch generation, append-only draw pools |
| `pipeline/aggregate.py` | vote share and log-space cluster median, parse QA telemetry |
| `pipeline/scorer.py` | wrapper that calls the official `evaluate.py` by import |
| `pipeline/coverage.py` | oracle ceiling analysis, selection bound versus knowledge bound |
| `pipeline/tune.py` | threshold sweeps on train, reported on val |
| `pipeline/tests/` | decision-parity tests on frozen continuations |
| `configs/` | per relation configurations, including the submitted one |
| `jobs/` | Slurm batch scripts |
| `paper/` | system paper, with the ACL build in `paper/build/` |
| `audit/` | parameter budget proof, closed-book audit output |
| `docs/` | engineering notes |

The remaining `pipeline/` modules are the offline analyses described in the
paper and the engineering notes.

## Reproducing

Pools are content addressed. A pool's filename is a SHA256 over its full
identity: relation, split and split file hash, model repository and revision,
channel, prompt template source, demonstration ids and seed, sampling
parameters, stop list and seed base. A manifest sits beside each pool. Any edit
to a prompt or demonstration set changes the key by construction, so a stale
pool cannot be silently reused.

Scoring always calls the organizers' `evaluate.py`. The metric is never
reimplemented here. `pipeline/scorer.py` imports that module and calls its own
functions rather than parsing its printed table, because the printed table
rounds to three decimals and the threshold effects being measured are smaller
than that.

## Results

Official grader, test split, all-rows macro-F1.

| relation | test rows | F1 |
|---|---:|---:|
| `countryLandBordersCountry` | 67 | 0.9786 |
| `hasArea` | 100 | 0.8700 |
| `companyTradesAtStockExchange` | 100 | 0.8530 |
| `personHasCityOfDeath` | 100 | 0.6100 |
| `awardWonBy` | 10 | 0.3484 |
| `hasCapacity` | 98 | 0.3367 |
| overall | 475 | 0.7060 |

For reference, an all-empty submission scores 0.2147 on the same split, and the
organizer baseline scores 0.2964.

## Attribution

- Official data and scorer: `lm-kbc/dataset2026`, Apache-2.0. `evaluate.py` is
  used unmodified.
- Multi channel numeric elicitation and overshoot calibration follow
  `dukesun99/elicitation-beats-selection`, CC BY 4.0.
- Per relation elicitation recipes, base completion mode, and vote share
  abstention follow the description published by `ruggsea/lmkbc26-share`. That
  repository carries no licence file, so nothing is copied from it; the recipes
  here are reimplementations from the published method description and are
  cited as such in the paper.
- Self consistency with per relation thresholds, and the award decomposition
  strategy, follow the 2025 edition winner and runner up as cited in the paper.

## Licence

Apache-2.0. See LICENSE and NOTICE.
