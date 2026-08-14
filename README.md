# LM-KBC 2026: closed-book knowledge base construction

System code for the LM-KBC 2026 Shared Task (5th edition, AKBC), Codabench
competition 16267.

Given a subject and a relation, predict the complete set of object entities as
of world state 2026-07-01, using only the parametric memory of an open weight
model of at most 32B parameters. No retrieval, no external knowledge base, no
training.

## Result

**0.7060 all-rows macro-F1 on the official test set**, leading on five of the six
relations. Measured by the official grader over all 475 test rows.

| relation | test rows | F1 |
|---|---:|---:|
| `countryLandBordersCountry` | 67 | 0.9786 |
| `hasArea` | 100 | 0.8700 |
| `companyTradesAtStockExchange` | 100 | 0.8530 |
| `personHasCityOfDeath` | 100 | 0.6100 |
| `awardWonBy` | 10 | 0.3484 |
| `hasCapacity` | 98 | 0.3367 |
| **overall** | **475** | **0.7060** |

The paper is `paper/build/paper.pdf`. Its central result is that oracle coverage
statistics in this literature have no null control: permuting gold values across
subjects shows 35.5 of the 84.5 coverage points on `hasCapacity` are coincidence
rather than recall, which cuts the apparent selection headroom from 0.443 to
0.203. Three non-frequency instruments then independently agree the relation is
knowledge bound, not selection bound.

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
Everything after generation is non neural post processing.

## Closed book enforcement

Compliance is checked four ways, not by an import grep alone:

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

## Layout

```
pipeline/
  common.py      closed-book IO allowlist, POOL-ID cache identity, row weights
  channels.py    per relation elicitation frames, decode contracts, parsers
  generate.py    vLLM offline batch generation, append-only draw pools
  aggregate.py   vote share and log-space cluster median, parse QA telemetry
  scorer.py      wrapper that calls the official evaluate.py by import
  coverage.py    oracle ceiling analysis: selection bound vs knowledge bound
  tune.py        threshold sweeps on train, reported on val
jobs/            Slurm batch scripts
paper/           system paper
audit/           parameter budget proof, closed-book audit output
NOTES.local.md  every run, submission, and verdict
```

## Reproducing

Pools are content addressed. A pool's filename is a SHA256 over its full
identity: relation, split and split file hash, model repository and revision,
channel, prompt template source, demonstration ids and seed, sampling
parameters, stop list and seed base. A manifest sits beside each pool. Any edit
to a prompt or demonstration set changes the key by construction, so a stale
pool cannot be silently reused.

```bash
python pipeline/generate.py --channel cap_disambig --split val --n 30
python pipeline/coverage.py --channel cap_disambig --split val
python pipeline/tune.py     --channel cap_disambig
```

Scoring always calls the organizers' `evaluate.py`. The metric is never
reimplemented here. `pipeline/scorer.py` imports that module and calls its own
functions rather than parsing its printed table, because the printed table
rounds to three decimals and the threshold effects being measured are smaller
than that.

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
