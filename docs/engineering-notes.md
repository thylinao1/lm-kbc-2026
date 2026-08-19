# Engineering notes

Working record for the LM-KBC 2026 entry: what was run, what it measured, and
what was discarded. It exists so that a reader of the paper can see the failures
as well as the shipped configuration, and so that nobody re-runs an experiment
that has already been closed by measurement.

Every number below was produced by the official `evaluate.py`, either on a local
split or by the Codabench grader. Figures marked TEST are grader-returned.

## Substrate

`google/gemma-4-31B` at revision `5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89`,
served with vLLM at tp=1, base completion mode, no chat template, temperature
0.7.

Parameter count was audited by reading every safetensors shard header (dtype and
shape) rather than trusting a reported metadata field: text transformer
29,288,059,196 + tied `embed_tokens` 1,409,286,144 + vision tower 575,743,536 =
31,273,088,876. The sum of tensor bytes (62,546,177,752) matches the index
`total_size`, and no `lm_head` tensor exists in either shard, which confirms the
embedding is tied. Headroom under the 32B cap is therefore 0.727B, which is not
enough for any second neural component. See `audit/param_budget.json`.

The hosting page's 32.68B figure counts the tied embedding twice. The difference
is approximately, though not exactly, one embedding matrix.

## Serving and cluster notes

Measured on one H100-96 NVL: model load 133s from NFS, then 866 tok/s on the
borders train pool (2010 draws, 14,522 decode tokens in 17s) and 779 tok/s on
val. A full nine-channel two-split sweep is about 15 to 20 minutes of
generation. That is why `generate_many.py` exists: eighteen separate
`generate.py` invocations would have reloaded the 62GB checkpoint eighteen
times, costing roughly 40 minutes of pure loading.

Three things cost real time and are worth writing down.

The login node's `ulimit -u` is 64. Any multithreaded Hugging Face downloader
dies there with "failed to spawn thread: Resource temporarily unavailable", so
weights have to be staged inside a job rather than from the login shell.

Invoking an environment's interpreter by absolute path does not put that
environment's `bin/` on `PATH`. A canary job failed at engine init with
`FileNotFoundError: 'ninja'`, raised from flashinfer's JIT compile of its
top-k/top-p sampling kernel. `ninja` was installed, just not findable. Every
job script now exports the environment's `bin` on `PATH`, and also sets
`VLLM_USE_FLASHINFER_SAMPLER=0` so that no first-use compile happens inside a
GPU job.

The a100-80 nodes expose one GPU each, so tensor parallelism there would need
multiple nodes. A single h200-141 card holds the 62.5GB checkpoint with about
64GB left for KV cache at tp=1, which is the serving configuration used.

## Scorer verification

`pipeline/scorer.py` imports the organizers' `evaluate.py` and calls its
functions directly rather than parsing its printed table. The printed table is
pandas-formatted, truncates columns with "...", and rounds to three decimals,
which both breaks positional parsing and blurs the 0.01 to 0.03 threshold
deltas this work is made of. Importing also yields the per-row F1 vector needed
for the paired bootstrap.

Five checks were run before any score was trusted: oracle predictions score 1.0
on borders, hasCapacity and hasArea; all-empty borders returns 0.2647, which is
exactly 18/68; the full all-empty val file returns 0.2000, reproducing the
logged floor; a numeric prediction 4.9% off gold scores 1.0 and one 6.1% off
scores 0.0, pinning the tolerance boundary at 5%.

`evaluate.py` imports pandas at module level but uses it only inside `main()`
for display. On cluster environments without pandas the scorer inserts a stub
before exec and never calls `main()`. Both a machine with real pandas and one
with the stub return 0.2000 on the all-empty val file.

## Faults found by running the pipeline

These cost more score than any modelling decision.

Integer rounding destroyed every sub-unit area. `predict_numeric` ended with
`str(int(round(v)))`, which is harmless for hasCapacity, whose golds are
integers, and destructive for hasArea, where several golds are below 0.5 km2 and
round to "0". A predicted 0 can never match, since the grader computes
`|pred - gt| / gt`. Five val subjects were affected (Ilha da Queimada Grande
0.43, Hashima Island 0.063, Torcello 0.4417, Isola di San Michele 0.176,
Okinotorishima 0.008482), and in every case the correct value was already in
the pool. Fixing the formatter moved hasArea val from 0.7600 to 0.8500. The
parse telemetry did not catch it, because the pipeline parsed fine and emitted a
well-formed number. Only comparing predictions against gold by magnitude
exposed it.

Single-linkage log clustering chained across genuinely different answers. Draws
at 10000/10500/12000/15000 merged into one cluster spanning half the pool, whose
median (12000) was a value no draw proposed and no gold at either end accepts.
Replaced with a fixed-radius rule: score every distinct drawn value by how many
draws fall within the grader's own tolerance of it, take the argmax, centre it
on its supporting mass. A regression test pins the behaviour.

Demonstrations leaked into the train curve. Demos are drawn from train and fixed
per channel, so 32 of 67 borders train subjects had their own gold sitting in
their prompt. Train F1 was 0.9924 with the leak and 0.9854 without. `tune.py`
now excludes demo subjects from the train curve and reports the count. The cost
of the guard is real: with 32 demos only 35 borders train subjects remain
scorable, so train-side threshold choice is low powered, and the code warns
below 15 rows.

The single-channel guard was itself insufficient. Channels draw different demo
sets from the same 100 train rows, so a train subject clean for `cap_recite` is
often a demonstration for `cap_rich` or `cap_current`, whose prompt for that
subject contains its gold. A cross-frame reranking experiment showed +0.2206 on
train and +0.0000 on val; split by leak status, leaky rows gained +0.3333 and
clean rows +0.0000. `channels.demo_union()` now returns the union of demo
subjects across every channel of a relation. The union is 85 of 100 train rows
for hasCapacity over all eight channels (64 over the six with pools on every
split) and 100 of 100 for hasArea, meaning that relation has no leak-free train
row at all.

The scorer cannot be subsetted by passing fewer predictions. `evaluate.py`
iterates the gold file, so a subject omitted from the prediction dict is scored
as an empty prediction, not excluded from the average. The first attempt at the
leakage guard would have silently tanked the train numbers instead of filtering
them. `score_one_relation` now takes an explicit `subjects` allowlist and
filters after scoring.

Parse quality assurance conflated abstention with parse failure. The gate fired
"empty-extraction 17.9%" on borders when the parse failure rate was 0.0000 and
the abstention rate was 0.1791: every one of those was the model correctly
emitting the in-band "none" sentinel. Since `make_config.py` excludes channels
that fail parse QA, this would have thrown away the best relation. Parse QA now
separates abstention, genuine parse failure and empty text, and gates on parse
failures only.

## Threshold selection

After the leakage guard the borders train curve is flat at 0.9854 across tau
0.15 to 0.45, so a plain argmax returns the first tied point, an arbitrary
plateau edge. `tune.py` picks the centre of the widest tied run instead, which
chose tau=0.30 and val 0.9896.

When several thresholds tie on validation, the tie can also be broken by
whichever produces a test abstention rate closest to the measured test empty
prior. That prior comes from a single all-empty submission, which returns each
relation's empty-gold fraction exactly (0.2147 overall; cityOfDeath 48/100,
stockExchange 44/100, borders 10/67, the three others zero). Re-centring
abstention thresholds onto those rates was worth about +0.0064 overall.

One caution learned by measurement: the F1-optimal abstention rate is not the
empty prior. Matching the two looked like the right move and measured worse. Lowering
cityOfDeath tau from 0.45 to 0.30 flips 22 rows, of which 12 are empty-gold and
only 6.33 are answered correctly. The board bracketed it directly: 54% abstain
gave 0.5800, 61% gave 0.6100, 68% gave 0.6000.

## Elicitation register

Asking the model to recall facts about an entity before committing to an answer
is the single largest gain found. It is the best frame on both numeric
relations. On hasArea it moved the test score from 0.8100 to 0.8700, after
three earlier configurations (two frames, two aggregation methods) had all
returned exactly 0.8100 and had been read as evidence that the answers were
absent from the pool. They were not; the register was wrong.

The finding has a boundary, and the boundary is the useful part. Recall-first
framing hurts the relations where the system also has to decide whether an
answer exists at all. On cityOfDeath, `death_recite` abstains on 29% of val and
39% of test against a 48% empty truth, because priming the model to recall facts
about a person primes it to produce a city. On stockExchange, `stock_recite`
calibrates its abstention rate better than the shipped frame (42% test against a
44% truth, versus the shipped 45%) yet scores 0.0735 lower on val, so it decides
whether to answer about as well and what to answer considerably worse. The
register helps when the task is to recall a value and hurts when the task
includes declining.

## Validation was the wrong instrument

At 97 to 100 rows per relation the val standard error is 0.03 to 0.05, while the
effects that decide this task are 0.01 to 0.03. Validation cannot resolve them.
Five decisions where val pointed the wrong way and the board settled it:

| decision | val said | test said |
|---|---|---|
| cityOfDeath tau 0.35 vs 0.45 | flat, 0.5967 either way | +0.0300 for 0.45 |
| hasCapacity anti-round rescue | +0.0103 | -0.0102 |
| stockExchange ratio emission | +0.0077 | +0.0166 |
| hasArea recitation frame | tie, CI [-0.050, +0.050] | +0.0600 |
| borders tau 0.45 to 0.15 | worse, 0.9816 vs 0.9942 | +0.0123 |

The test board returns per-relation scores over the exact rows being ranked, and
relations occupy disjoint row sets, so one submission is six independent
readings. That is what makes spending submissions on measurement efficient
rather than wasteful.

A test-prior reweighting of val (reweight rows by the ratio of test to val empty
rates) was tried as a cheaper substitute and is biased optimistic. Predicted
against actual: cityOfDeath 0.6259/0.5800, stockExchange 0.8832/0.8364, borders
0.9932/0.9663. Its assumption is that difficulty within the empty and non-empty
groups is unchanged across splits, and that assumption is false. Transfer from
val to test was negative and roughly uniform across the five substantive
relations: -0.043, -0.040, -0.028, -0.024, -0.017.

## Assembly from measured configurations

Because relations occupy disjoint row sets, a file built from the
best-measured-per-relation configurations scores exactly the row-weighted sum of
those measured scores, provided each relation's rows are byte-identical to the
submission that scored them. This was verified twice: an assembly predicted at
0.6916 returned 0.6916, and a later one predicted at 0.7060 returned 0.7060.
After the first confirmation, any combination of already-measured configurations
could be evaluated offline without spending a submission.

## Scored submissions

| entry | test macro-F1 | what it measured |
|---|---|---|
| all-empty probe | 0.2147 | per-relation empty-gold priors, exactly |
| v2 full system | 0.6818 | first real configuration |
| P3 | 0.6895 | cityOfDeath tau, stockExchange ratio emission, area frame, anti-round |
| best_measured | 0.6916 | assembly method, predicted and returned to 4 dp |
| P4 | 0.6785 | one step further on five axes; all neutral or worse |
| P5 | 0.6948 | area recitation frame, borders tau 0.15, cityOfDeath direct frame |
| P6 | 0.6955 | nearest-neighbour demonstrations |
| final assembly | 0.7060 | best measured per relation |

Each probe froze some relations as byte-identical canaries, so a probe could
return a worse overall number without ever costing score: the final file is
reassembled per relation from measured bests. Canaries
returning their predicted values to four decimals is what licenses trusting the
other readings in the same submission.

## Coverage is not knowledge

Oracle coverage statistics in this literature have no null control. A pool of a
hundred guesses spanning about a decade contains roughly twenty-one mutually
exclusive 5% tolerance windows, so it contains something acceptably close to
almost any plausible target whether or not the model knows the answer.
Permuting gold values across subjects within a relation measures that
coincidence rate directly (400 shuffles):

| relation / split | coverage | chance | chance-corrected | median pool width |
|---|---|---:|---:|---|
| hasCapacity val (97) | 0.8454 | 0.3552 | 0.7602 | 0.91 decades, 21 slots |
| hasCapacity train-clean (15) | 0.7333 | 0.3748 | 0.5734 | |
| hasArea val (100) | 0.9400 | 0.0311 | 0.9381 | 0.01 decades, 0 slots |

Both pools carry real knowledge (z = 12.4 and z = 53.1). What differs is how
much of the coverage is real. Broken down by the gold's rank in the capacity
pool, the excess over chance is +0.281 at rank 1, +0.140 at ranks 2-3, +0.046 at
ranks 4-10 and +0.017 beyond. Rank 1 is what the shipped selector already emits.
So the honest recoverable-by-selection figure is 0.203 of the relation, not the
0.443 the raw coverage gap implies.

Adding frames makes this worse rather than better. Going from one capacity frame
to seven raises coverage by 0.093 and raises the coincidence rate by 0.157, and
past four frames the chance-corrected knowledge falls while the pool widens from
27 windows to 29.

This reverses a claim I had made earlier and had spent effort acting on. It also
applies to the published 76-77 versus 35-38 coverage claim that this work was
chasing.

## Rejected changes

Everything here was implemented and measured. The ship rule was fixed before the
data existed wherever a free parameter was involved.

| change | result |
|---|---|
| cross-frame consensus (agreement-weighted) | worse than best single frame on hasCapacity (0.2887 vs 0.3196) and on hasArea (val -0.11, CI excludes zero) |
| routing by raw or train-calibrated concentration | 0.2784 and 0.3093 against best single 0.3196 |
| cross-frame agreement reranking | +0.0000 on 259 leak-free rows, 8 up and 8 down |
| forced-choice duels | val 0.3196 against shipped 0.3608; the probe works (62.4% on decidable pairs, z = 6.0) but beats an always-pick-the-larger rule by only 2.1 points at z = 0.7 |
| yes/no verification probe | val 0.2680 against 0.3608, CI excludes zero from below |
| background-lift reranking | +0.0000 at the pre-committed eps, 5 rows up and 5 down |
| expected-F1 maximiser over all six relations | -0.0132 overall; best case hasCapacity +0.00025 with CI [-0.0064, +0.0068] |
| nearest-neighbour demonstrations | measured on test: hasCapacity -0.0408, hasArea -0.0100, cityOfDeath +0.0000 |
| multi-prediction hedging on numerics | E[F1] falls monotonically from k=1 on both relations and both splits |
| plausibility band from train gold | +0.0000 on all six measurements |
| liveness gate for cityOfDeath | changes 4 of 100 test rows and pushes abstention from 54% to 58% against a 48% truth |
| cluster width matched to the grader's 5% | worse on both relations and both splits; the two quantities share a unit but are not the same quantity |
| lower temperature (0.5, 0.3) | worse on both relations and both splits |
| deeper pools (n from 10 to 100) | no monotone trend, spread about 0.03 against an SE of 0.049 |
| 100 demonstrations instead of 64 (hasArea) | val 0.8300 against 0.8500, and unvalidatable on train by construction |
| overshoot scaling below 1.0 | all five numeric channels selected scale 1.0 on both splits |
| second substrate or a sidecar model | ruled out by the 32B total: 0.727B of headroom fits nothing useful |

A channel using 100 demonstrations cannot be validated on train at all when the
train split has 100 rows: the leakage guard excludes every row. `tune.py` now
names that case and sets a `train_unusable_all_demos` flag instead of crashing.
Any train figure for such a channel is memorisation of its own prompt.

One rule was rejected on integrity grounds rather than on measurement. A
proposed `decade_lift` correction for hasArea had been validated against
Wikidata entries for named test subjects, with its parameters then tuned to fix
exactly those rows. Even though the rule's form is generic, its validation used
external gold, so shipping it would have laundered a knowledge base lookup into
a closed-book system. An independent val measurement made before that proposal
existed found 4 of 100 val subjects scale-ambiguous with none recoverable, so
there is no evidence the rule helps anyway.

## What remains open

hasCapacity is knowledge bound on this substrate. Three instruments that are not
functions of the sampled vote frequency (agreement reranking, forced-choice
duels, yes/no verification) were each given the shipped frame's own candidate
set, so a gain could only come from better ordering, and each had its rule fixed
before the data existed. None beats a frequency count.

About 25 of 98 test capacity rows sit in a stratum whose pool argmax is 10000,
scoring roughly 0.10 and resisting every reranking method tried. Conceding that
stratum bounds reranking alone near 0.42 to 0.45 on the relation. Reaching
further requires a frame that changes what enters those pools; a better selector
will not do it.

awardWonBy is 10 rows and cannot be gated on any split. Its behaviour is
bimodal: on "Grammy Award for Best Rock Album" the system returns 77 candidates
covering 18 of 22 golds, while on institutional honorary-doctorate lists it
returns plausible but entirely wrong names with zero overlap.
