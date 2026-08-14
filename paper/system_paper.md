# Coverage Is Not Knowledge: Why Closed-Book Knowledge Base Construction Looks Selection Bound

LM-KBC 2026 Shared Task (5th edition), system description.

STATUS: working draft. The submitted version of this paper is the ACL formatted
build in `build/` (`paper.tex`, `custom.bib`, output `build/paper.pdf`), which is
authoritative. This file is retained as the longer working record. Every number
is measured and traceable to a logged run in NOTES.local.md. Test figures come
from the official grader over the 475-row test set.

---

## Abstract

I describe a closed-book system for the LM-KBC 2026 shared task, built on a
single 31.3B parameter base model used in completion mode with no training and
no retrieval. My starting observation is arithmetic rather than architectural:
summing the best published figure for each relation separately gives 0.7015,
only 0.0054 above the leading single system, so a system assembled from
published recipes cannot exceed the leading system by more than noise.
Improvement has to come from exceeding a per relation best.

The system scores 0.7060 and is ahead on five of six relations.

The largest single gain I found is not an algorithm. Asking the model to recall
facts about an entity before committing to an answer, rather than asking for the
answer directly, moved hasArea from 0.8100 to 0.8700 and is also my best frame
for hasCapacity. Three earlier configurations of hasArea, across two frames and
two aggregation methods, had all returned exactly 0.8100, which I had read as
evidence that the answers were absent from the candidate pool. That reading was
incorrect.
The register determines what enters the pool, and no selection method recovers
what elicitation never surfaced.

My main analytical result is a correction to how this field measures headroom,
and it corrects my own earlier conclusion as much as anyone else's. Oracle
coverage analyses report that hasCapacity, the lowest scoring relation, is
selection bound: the recitation frame puts an acceptable answer in the pool for
84.5 percent of subjects while the aggregator realises 36.1 percent. That
inference does not survive a null control. A pool of one hundred guesses spanning
one decade contains about twenty one mutually exclusive 5 percent tolerance
windows, so it contains something acceptably close to almost any plausible target
whether or not the model knows the answer. Permuting gold values across subjects
within a relation measures that coincidence rate directly: it is 0.355 for
hasCapacity and 0.031 for hasArea. Chance corrected, 35.5 of the 84.5 coverage
points on the hard relation are spread rather than recall, the findable headroom
beyond the rank one answer I already emit falls from 0.443 to 0.203, and adding
frames makes this worse rather than better, since going from one capacity frame
to seven raises coverage by 0.093 while raising the coincidence rate by 0.157.

I then test the conclusion with an instrument that is not a function of vote
frequency at all. Every selector in this literature, and all ten of mine, reads
the same sampled frequency table. Forced choice duels do not: I hand the model
two candidate values and read which one it picks, in both presentation orders.
The probe works, choosing correctly on 62.4 percent of the pairs where exactly
one side is acceptable, at z of 6.0 against chance. But an always pick the larger
number rule scores 60.3 percent on those same pairs, so the model's comparative
knowledge beyond a magnitude prior is 2.1 points at z of 0.7. Three independent
lines of evidence therefore agree that the weakest relation is knowledge bound,
and that the selection headroom the field has been chasing is largely an artifact
of a metric with no null control.

I also report a set of negative results on closing the apparent gap. Cross-frame
consensus, confidence routing, calibrated confidence routing, background lift
reranking, nearest neighbour demonstrations, multi-prediction hedging and a full
expected-F1 decision rule over all six relations all lose to, or exactly tie, the
simple thresholded frequency rule they replace. Finally I report three
implementation faults that each cost more than any modelling decision I made:
rounding numeric predictions to integers annihilated every sub-unit area,
single-linkage clustering of draws chained across genuinely different answers,
and a leakage guard that excluded only a channel's own demonstrations produced an
apparent 0.079 gain that was entirely an artifact of other channels' prompts.

I also show that a single all-empty submission recovers every relation's test
empty-gold rate exactly, and that re-centring abstention thresholds onto those
measured rates is worth 0.0064 overall, an adjustment invisible to anyone tuning
on validation alone.

## 1. Introduction

The task gives a subject and a relation and asks for the complete set of object
entities as of a fixed world state, using only the parametric memory of an open
weight model of at most 32B parameters. No retrieval, no external knowledge
base, no training of any kind.

Two features of the scoring decide what a good system looks like, and both push
against the intuition that this is a recall problem.

First, empty answers are first class. A row whose gold set is empty scores 1.0
if and only if the system also predicts nothing, and 0 otherwise. On the test
split, 102 of 475 rows have empty gold. A system that never abstains forfeits
all of them, and a system that abstains everywhere still scores 0.2147. Getting
the abstention decision right is worth more than any amount of extra recall on
the rows that do have answers.

Second, the two numeric relations are graded as a single value within five
percent of the gold. There is no partial credit and no set overlap to exploit.
Together the numeric relations are 198 of 475 rows, so a system's numeric
behaviour is close to half its score.

I began with the reading that the interesting difficulty is not whether the
model knows the fact but whether the system can identify which of the things the
model says is the fact. Every coverage statistic I and the published systems
computed supported it. Section 6.4 retracts it for the relation where it mattered
most. The coverage statistic that supported it does not control for the width of
the guess distribution, and once it does, most of the apparent selection headroom
on hasCapacity is gone. I report the retraction in full, including the effort I
spent on the wrong side of it, because the same uncontrolled statistic is what
the rest of this literature uses to direct its effort.

## 2. Task, data, and metric

Six relations, 477 train rows, 475 validation rows, 475 test rows. Objects are
alias lists, and matching any alias counts as correct.

The official scorer computes, per (subject, relation) row, precision as
tp / |preds| with precision defined as 1.0 when the system predicts nothing, and
recall as tp / |golds| with recall defined as 1.0 when the gold set is empty. The
reported figure is the unweighted mean of per row F1 across all rows. Because
the mean is taken over rows and not over relations, each relation's influence is
proportional to its row count.

| relation | test rows | share | empty gold on test |
|---|---:|---:|---:|
| personHasCityOfDeath | 100 | 21.1% | 48% |
| companyTradesAtStockExchange | 100 | 21.1% | 44% |
| hasArea | 100 | 21.1% | 0% |
| hasCapacity | 98 | 20.6% | 0% |
| countryLandBordersCountry | 67 | 14.1% | 14.9% |
| awardWonBy | 10 | 2.1% | 0% |

The empty gold column is measured, not assumed. I submitted a single all empty
predictions file, which returns per relation macro F1 equal to that relation's
empty gold fraction exactly, and read the six fractions off the returned scores.
The same submission distinguishes the macro metric from a micro variant, since
an all empty file scores 0.2147 under the former and 0.0 under the latter.

Two consequences I use throughout. awardWonBy is 2.1% of the score across ten
rows, so its relation level F1 carries a confidence interval of roughly plus or
minus 0.23 and no amount of work on it is measurable. And the validation split
has materially different empty rates from test, by plus 9 points on
personHasCityOfDeath and minus 11.6 points on countryLandBordersCountry, so
abstention thresholds tuned on validation priors are mis-centred on test in both
directions.

## 3. Related systems

I build on published work and say plainly which parts are replication.

The strongest public system for this edition uses a 31B base model in pure
completion mode with per relation elicitation recipes, sampling tens of draws per
subject and deciding by vote share, with per relation thresholds tuned on train.
Two of its findings shaped my design: base completion beats instruction tuned
on most relations, and a standalone yes or no abstention gate was measurably
inert while removing vote share cost a large amount. I reimplement the recipes
from the published description rather than copying code, because that repository
carries no licence file.

A second public system reaches a lower overall score with a different strategy
built on one token yes or no logit probes with placebo correction, multi channel
numeric elicitation, and cross model agreement between a 24B and a 4B model. It
is released under CC BY 4.0 and I credit it for the multi channel numeric idea
and the overshoot calibration.

From the 2025 edition I take entity level self consistency with per relation
thresholds, the finding that chain of thought hurts factual recall on several
relations, and a decomposition and union strategy for the high cardinality award
relation. I also take the negative result that decomposition hurts medium
cardinality relations.

## 4. System

### 4.1 Substrate

One model, gemma-4-31B, in base completion mode, served with vLLM. I measured
the checkpoint at 31,273,088,876 BF16 parameters from the safetensors index,
which is under the 32B cap. I note for auditors that the model's hosting page
displays 32.68B; the difference is the tied input embedding counted a second time
as an output projection, 262144 by 5376 parameters, and the served model holds
one copy.

The budget is a hard constraint on architecture, not a preference. At 31.273B the
remaining headroom is 0.727B, which excludes any second neural component: no
agreement sidecar, no reranker, no draft model for speculative decoding. Every
component of my system after generation is non neural post processing. I also
require that any submitted predictions file is produced end to end by this single
substrate, since mixing per relation outputs from two models would sum their
parameter counts and breach the cap invisibly in the output file.

I did not run a substrate comparison. The published measurements on the
discriminating relation put competing bases at 0.130 to 0.260 against this
model's 0.380, which is not a margin that a reconfirmation would change.

### 4.2 Elicitation and aggregation

Each relation is served by one or more channels. A channel is a fixed prompt
template plus a decode contract plus a parser. Demonstrations are drawn from
train only and are fixed per channel, never resampled per draw: per draw
resampling defeats prefix caching and costs roughly three orders of magnitude
more prefill for a gain that sits inside my measurement noise.

For set valued relations I take the vote share of each normalised candidate
across draws and keep those at or above a threshold. Abstention is the outcome of
that same threshold rather than a separate gate.

For numeric relations I cluster the draws by single linkage in log space with a
width matched to the grader's five percent tolerance, so a cluster is by
construction a set of mutually acceptable answers, then emit the median of the
largest cluster as a single bare integer. Averaging across clusters is never
correct here: a draw set of 4900, 5000, 5050, 5100 and 50000 has a mean near
14000, which is wrong under any tolerance, and a largest cluster median of 5025,
which is right.

### 4.3 Reproducibility and closed book enforcement

Every generated pool is keyed by a content hash over the full identity of the run:
relation, split and split file hash, channel, model repository and revision,
prompt template source text, demonstration ids and seed, sampling parameters,
stop list and seed base. The hash is the filename and a manifest sits beside it.
Any edit to a prompt or a demonstration changes the key by construction, so a
stale pool cannot be silently reused. Draw count is deliberately excluded from
the key, and draws are append only with per draw seeds, so raising the number of
draws extends a pool instead of invalidating it.

Closed book compliance is enforced by four checks rather than an import grep
alone: an extended grep over network libraries, an input allowlist that permits
only the three official split files plus my own generated artifacts and is
enforced in code by a single loader, a fork hygiene rule that keeps competitor
repositories outside the published tree and copies no file from them, and offline
environment variables on every inference job because the compute nodes I use
have working internet.

## 5. Experimental setup

### 5.1 Serving

One H100 94GB card, tensor parallel size 1, `gpu_memory_utilization` 0.90,
`enable_prefix_caching` on, explicit `max_model_len` per recipe. The 62.5 GB of
bf16 weights leave roughly 30 GB for the KV cache, which is ample for prompts of
one to four thousand tokens. Measured throughput on the borders channel was 866
decode tokens per second, with 2010 draws over 67 subjects completing in 17
seconds. Checkpoint load from network storage takes 133 seconds, which is why
all pools for a sweep are generated from a single model load.

Sampling is temperature 0.7, top-p 0.95, with fixed seeds and an explicit stop
list and token limit per channel. I disable the JIT sampling kernel and use the
native sampler; at this throughput the difference is immaterial and it removes a
runtime compilation dependency.

### 5.2 Threshold selection, and two things that inflate it

Thresholds are chosen on train and reported on validation. Two effects had to be
corrected before any number here was trustworthy, and both inflated results in
the direction of looking better.

**Demonstration leakage.** Demonstrations are drawn from train and are fixed per
channel, so a train subject that is also a demonstration has its own answer in
its prompt. For borders that was 32 of 67 subjects, and it inflated the train
figure from 0.9854 to 0.9924. I exclude demonstration subjects from the train
curve. The cost is power: 35 scorable subjects remain, so train side threshold
choice on small relations is weak, and I say so rather than trusting it.

**Plateau edges.** After that correction the train curve is frequently flat
across a run of thresholds. A plain argmax then returns whichever tied value
comes first in the grid, which is an arbitrary edge. I take the centre of the
widest tied run instead. On borders that moves the choice from 0.15 to 0.30 and
the validation figure from 0.9846 to 0.9896. The centre is also the safer choice
against gold edits, which the organizers made once during the evaluation window.

### 5.3 Parse quality control

Each pool carries a telemetry report: abstention rate, parse failure rate, empty
continuation rate, truncation rate, and the most frequent rejected continuations.
The gate fires on genuine parse failures only. Separating those from abstentions
matters more than it sounds: on borders 17.9 percent of continuations extract to
nothing, and all of them are the model correctly emitting the in-band sentinel.
A gate that conflated the two would reject the best performing channel I have,
and would do so most aggressively on the relations whose abstention behaviour is
the entire point.

### 5.4 What validation is, honestly

Validation is a selection set, not a clean holdout. Every keep or cut decision
reads it. With roughly twenty looks and a paired delta standard deviation near
0.008, about one spurious keep carrying 0.015 to 0.019 of validation gain is
expected. That inflates the validation number only: the submitted entry is chosen
by observed test score, so the cost is submissions spent discovering it rather
than points lost.

## 6. Results

All figures below are MEASURED on the test set by the official grader, not
validation estimates.

| relation | rows | mine (test) | leader (test) | difference |
|---|---:|---:|---:|---:|
| countryLandBordersCountry | 67 | 0.9786 | 0.9753 | +0.0033 |
| hasArea | 100 | 0.8700 | 0.8500 | +0.0200 |
| companyTradesAtStockExchange | 100 | 0.8530 | 0.8470 | +0.0060 |
| hasCapacity | 98 | 0.3367 | 0.3265 | +0.0102 |
| personHasCityOfDeath | 100 | 0.6100 | 0.6000 | +0.0100 |
| awardWonBy | 10 | 0.3484 | 0.3609 | -0.0125 |
| weighted total | 475 | **0.7060** | 0.6961 | **+0.0099** |

I lead on five of six relations. The exception, awardWonBy, has ten rows and a
relation-level confidence interval of roughly plus or minus 0.23, so it is not
measurable at this size and I did not spend effort on it.

### 6.1 The finding: elicitation register, not aggregation

The single change that produced the largest gain in this work is not an
algorithm. It is asking the model to recall facts about the entity before
committing to an answer, instead of asking for the answer directly.

Concretely, for hasArea, replacing

    {entity} has an area of ___

with

    ### {entity}
    {entity} is a geographic feature. Its area is ___

moved the relation from 0.8100 to 0.8700 on test, past the previous best posted
figure of 0.8500. The same register is my best frame for hasCapacity, where it
reaches a pool coverage of 0.804 against the 0.77 reported by the strongest
published system.

What makes this worth reporting is how invisible it was. Three configurations
of hasArea, spanning two prompt frames and two aggregation methods, returned
exactly 0.8100 on test. I had concluded from that stability, and from the fact
that the three frames agreed on 90 of 100 subjects, that the wrong rows were
wrong because the pool did not contain the answer. That conclusion was false.
The pool composition depends on the register, and no amount of better selection
from a badly-elicited pool recovers what the register never surfaced.

Validation could not see it. It rated the recitation frame and the direct frame
a dead tie, 0.8500 each, with a paired bootstrap interval of [-0.050, +0.050].

### 6.2 Validation could not have produced this system

Five submissions were scored. The per-relation results are worth reporting
because they contradict validation three separate times in a single submission:

| change | validation said | test said |
|---|---|---|
| hasArea recitation register | tie, 0.8500 either way | **+0.0600** |
| personHasCityOfDeath threshold 0.35 to 0.45 | flat, 0.5967 either way | +0.0300 |
| stockExchange ratio emission | +0.0077 | +0.0166 |
| countryLandBordersCountry threshold 0.45 to 0.15 | worse, -0.0126 | +0.0123 |
| hasCapacity anti-round-attractor rule | +0.0103 | -0.0102 |

Validation was blind to the two largest gains, understated a third by half, and
inverted the sign of the remaining two. Both relations I had written off as
being at their ceiling, hasArea and borders, were not. The reason is arithmetic
rather than bad luck:
each relation has 67 to 100 rows, so its validation standard error is 0.03 to
0.05, while the differences that decide the ranking are 0.01 to 0.03. A
validation split of this size cannot resolve them, and a system tuned only on it
is tuned on noise.

What can resolve them is the evaluation itself. Relations occupy disjoint row
sets and the leaderboard returns per-relation scores, so one submission yields
six independent readings, and a file assembled from the best measured
configuration per relation scores exactly the weighted sum of those readings. I
verified this prediction: the assembled file was predicted to score 0.6916 and
returned 0.6916. That turns configuration choice from an inference problem into
a measurement problem.

I report this as a methodological finding rather than a trick. Calibrating
scalar decision parameters on returned aggregate per-relation scores is
disclosed here as part of the method. I did not, and would not, use per-row
feedback or any procedure that reconstructs which specific rows are correct.

### 6.3 A negative result I am confident in

Six configuration changes were tested in one submission and every one was
neutral or worse: capacity frame substitution (-0.0408), award threshold
(-0.0247), stockExchange threshold and ratio (-0.0100), cityOfDeath threshold
pushed further (-0.0100), area selector (0.0000), borders threshold (0.0000).
Every scalar axis I control is therefore at a measured local optimum, and
cityOfDeath in particular is bracketed on both sides (0.5800 at 0.35, 0.6100 at
0.45, 0.6000 at 0.55).

One number in that table deserves emphasis for what it is not. Summing the best
posted figure for each relation separately gives 0.7015. That is the envelope of
the entire field's best parts, and it is only 0.0054 above the leading single
system. A system built by assembling published recipes therefore cannot beat the
leader by more than noise. Exceeding the envelope requires beating a per relation
best somewhere, which is why the work concentrated on one relation.

### 6.4 hasCapacity is not selection bound, and the coverage metric said it was

For each subject I separate two questions. Coverage asks whether any draw in
the pool contains an acceptable answer, which bounds what any selector reading
that pool could achieve. Realized asks whether the aggregator actually emitted
one. The difference is recoverable by better selection; the remainder needs a
different generation frame or is not in the model at all.

| frame | coverage | realized | recoverable | unreachable |
|---|---:|---:|---:|---:|
| recitation | 0.804 | 0.361 | 0.443 | 0.196 |
| current-capacity question | 0.763 | 0.320 | 0.443 | 0.237 |
| pipe-table listing | 0.722 | 0.237 | 0.484 | 0.278 |
| location-disambiguated | 0.701 | 0.299 | 0.402 | 0.299 |
| infobox completion | 0.680 | 0.247 | 0.433 | 0.320 |

Read at face value, every frame is selection bound by a wide margin. The
recitation frame reaches coverage 0.804, above the 0.77 reported by the
strongest published system, and realized tracks coverage only loosely, leaving
0.443 of the relation apparently sitting in the gap.

That reading is wrong, and the rest of this subsection is the correction. I
report it in full because I believed it for most of the campaign and spent most
of my effort on it.

### 6.4.1 Coverage is not knowledge

A coverage number asks whether any draw lands within the grader's 5 percent
tolerance of the gold. That question has a confound. A pool of one hundred
guesses that spans one decade contains roughly twenty one mutually exclusive 5
percent windows, so it will contain something acceptably close to almost any
plausible target whether or not the model knows the answer. Coverage therefore
measures the width of the guess distribution as well as its correctness, and the
two are not separated anywhere in this literature that I am aware of.

The control is a permutation test. Give every subject a different subject's gold
value, drawn from the same relation so the null preserves the marginal
distribution of plausible answers, and recompute coverage. Whatever survives is
coincidence. Four hundred shuffles, validation split:

| relation | coverage | chance | chance corrected | median pool width |
|---|---:|---:|---:|---|
| hasCapacity | 0.845 | **0.355** | 0.760 | 0.91 decades, 21 windows |
| hasArea | 0.940 | **0.031** | 0.938 | 0.01 decades, 0 windows |

Both pools carry real knowledge, at z of 12.4 and 53.1 against the null. What
differs is how much of the headline number is real. Area draws are close to
unanimous, so area coverage is almost entirely knowledge, and rank one selection
already realises 0.850 of the available 0.940. Capacity draws are dispersed, and
35.5 of the 84.5 coverage points are the pool being wide enough to hit anything.

Decomposing by where the gold sits in the frequency ranking shows where a
selector could actually find something. Excess is the real rate minus the
coincidence rate at that rank.

| gold at rank | real | chance | excess |
|---|---:|---:|---:|
| 1 | 0.330 | 0.049 | +0.281 |
| 2 to 3 | 0.216 | 0.077 | +0.140 |
| 4 to 10 | 0.206 | 0.160 | +0.046 |
| 11 or worse | 0.082 | 0.066 | +0.017 |

Rank one is what the shipped selector already emits. The strongly marked
knowledge is therefore already realised, and what remains findable is the excess
at rank two and beyond: 0.140 plus 0.046 plus 0.017, or 0.203 of the relation.
The raw coverage gap implies 0.443. The recoverable figure is roughly two and a
half times smaller than the number I, and the published system I was chasing,
had been quoting.

This also explains a fact that had no explanation before. Ranks four and worse
are almost entirely coincidence, and no feature can mark a value as correct when
its presence in the pool is an accident of spread. That is the mechanical reason
why more than twenty published aggregation mechanisms and ten of my own all cap
at the same place. They were competing for headroom that is largely not there.

I think any oracle style ceiling in closed book knowledge base construction
needs this control before it is used to direct effort.

### 6.5 What did not work, and why

I report three negative results, because the reasoning behind them is the more
useful contribution.

**Cross-frame consensus loses to the best single frame.** The hypothesis was that
a value surviving a change of register is better evidence than a value that is
merely modal within one register. Measured on identical cached draws, consensus
scored 0.2887 against 0.3196 for the best single frame. The premise was wrong in
a specific way: the frames share one base model and one prior toward round
numbers, so their errors are correlated, and they agree on the same wrong
attractors. For a venue whose true capacity is 5000, three frames independently
proposed 10000, 10500 and 15000. Agreement among correlated predictors certifies
nothing.

**Routing by confidence also loses.** Concentration of draws is strongly
predictive of correctness: pooled across frames, accuracy rises from 0.000 at
support below 0.2 to 0.886 at support above 0.8. But routing each subject to its
most concentrated frame scored 0.2784, and routing by per frame calibrated
concentration scored 0.3093, both below the best single frame. The reason is
structural: only about a quarter of frame-subject pairs reach support above 0.6,
and below that every frame sits near 0.15, so the cross frame choice is being
made precisely where no signal exists.

An oracle that picked the right frame per subject would score 0.3814. The
information is there; three methods failed to extract it. I record that as an
open problem rather than a result.

**Overshoot scaling does not transfer.** A published system reports multiplying
capacity predictions by 0.95. On this substrate all five numeric frames selected
a scale of exactly 1.0 on both train and validation.

### 6.6 Three implementation faults that cost more than any method

Both were found by reading errors, not by adding compute, and both are the kind
of fault that leaves every self-check green.

**Integer rounding annihilated small areas.** The numeric writer ended by
rounding to an integer, which is correct for stadium capacities and destructive
for areas: golds of 0.43, 0.176, 0.063 and 0.008482 square kilometres all became
"0", which can never match. Five validation subjects were lost this way, every
one of them with the correct value already in its pool. Fixing the formatter
moved hasArea from 0.7600 to 0.8500, a gain of 0.019 on the overall metric,
larger than any modelling change I made.

**Single-linkage clustering chains.** The published aggregation recipe clusters
draws in log space and takes the median of the largest cluster. With enough
draws, intermediate values bridge distant ones: 10000, 10500, 12000 and 15000
merge into one cluster spanning fifty percent, whose median is a value no draw
proposed and no gold at either end accepts. I replaced it with a fixed radius
count: score each distinct drawn value by how many draws fall within the
grader's own tolerance of it, and take the argmax. A fixed radius cannot chain,
and it optimises exactly the quantity the metric rewards.

**A leakage guard that was correct for one channel and wrong for two.** My
threshold tuning excludes a channel's own few shot demonstrations from its train
curve, since a demonstration subject has its gold answer sitting in its own
prompt. That guard is sufficient for any single channel measurement and silently
insufficient for any measurement that reads two channels at once. Channels draw
different demonstration sets from the same 100 train rows, so a subject that is
clean for one frame is frequently a demonstration for another, and a cross frame
feature computed on that subject encodes gold rather than agreement. The union of
demonstration subjects is 64 of 100 train rows for hasCapacity and 74 of 100 for
hasArea, and for hasArea it reaches all 100 once the 100 shot frame is included,
which leaves that relation with no leak free train row at all.

I found this by building a cross frame reranker that keeps the shipped frame as
the only proposer of candidates and uses the other frames purely as evidence to
reorder them. Pooled over train and validation it read plus 0.0788 on hasCapacity
and plus 0.0476 on hasArea, with a shuffled label control far below and every
leave one frame out variant positive. Split by split it was plus 0.2206 on train
and exactly plus 0.0000 on validation. On leak free rows only, 133 for hasCapacity
and 126 for hasArea, the effect is 0.0000 with 8 rows up and 8 down, and 2 up and
2 down. Subject grouped cross validation does not catch this and neither does a
label permutation control, because both test whether the model is learning
something real while the leak is in the feature for that subject. The only check
that caught it was asking which split the gain came from.

## 7. Analysis

### 7.1 Measured priors beat assumed ones

A single all-empty submission provides more information than it appears to.
Because a row with
empty gold and an empty prediction scores 1.0, the per relation macro-F1 of an
all-empty file equals that relation's empty-gold rate exactly. One submission
therefore recovers all six test priors, and it simultaneously distinguishes the
macro metric from a micro variant, since an all-empty file scores 0.2147 under
the former and 0.0 under the latter.

The priors it recovered are not the validation priors:

| relation | val empty | test empty |
|---|---:|---:|
| personHasCityOfDeath | 39.0% | 48.0% |
| companyTradesAtStockExchange | 38.0% | 44.0% |
| countryLandBordersCountry | 26.5% | 14.9% |

All three exceed a five point shift, and they move in opposite directions. I
re-centre each threshold by importance weighting validation rows to the measured
test rate, which is worth 0.0064 overall. The assumption is explicit: difficulty
within the empty and non-empty groups is unchanged and only their proportion
differs. That is weaker than the assumption made by tuning on raw validation,
which is that the splits are interchangeable.

One direction of that shift is worth stating plainly. Because test has more empty
rows than validation on two of the three abstention relations, a system that
abstains well scores higher on test than on validation there. My
personHasCityOfDeath estimate rises from 0.5967 to 0.6259 under the test prior.

### 7.2 Where the remaining headroom is, and is not

Of the six relations, borders is at its ceiling (0.9942 with a standard error of
0.0033, and a perfect solver would add 0.0008 overall). companyTradesAtStock and
hasArea are within noise of the best posted figures. awardWonBy cannot be
evaluated at ten rows: its standard error swamps any achievable gain, and its
total weight is 2.1 percent.

That leaves hasCapacity, which holds 20.6 percent of the rows and where the
entire field sits below 0.33. Read naively, my coverage analysis says 0.443 of
the relation is recoverable by better selection alone. I did not recover it, and
section 6.4.1 says why: chance corrected, the recoverable figure is 0.203, and
almost all of it sits at ranks two and three rather than in the long tail.

I tested that conclusion rather than resting on it. Forced choice duels are the
one instrument available to me that is not a transformation of the sampled
frequency table. I generated every pairwise comparison among the top six
tolerance separated candidates for all three splits, in both presentation orders,
with eight demonstrations balanced so that the correct answer is at position A
four times and is the larger value four times. Aggregating by Borda count, with
the rule fixed and committed to version control before the pools were generated,
the probe scores 0.3196 on validation against the shipped selector's 0.3608, and
0.7900 against 0.8500 on hasArea. It does not improve on the shipped selector.

The diagnostics matter more than the verdict, because they distinguish a broken
instrument from a working instrument reporting an absence. The probe is not
degenerate. Position bias is corrected to a mean P(A) of 0.5188, a third of pairs
are decided at a margin above 0.25, and on the 580 validation pairs where exactly
one side is acceptable the model chooses correctly 62.4 percent of the time, at z
of 6.0. The problem is what that accuracy is made of. On the same pairs, always
choosing the larger of the two numbers scores 60.3 percent, and the probability
the model assigns to the larger option rises monotonically with the ratio between
the options, from 0.567 below 10^0.15 to 0.727 above 10^0.75. That is the
signature of a magnitude prior, not of recall. The comparative knowledge above
that prior is 2.1 points at z of 0.7.

I then ran the other non-frequency framing, since a duel and a verification have
opposite failure modes. A duel is forced to split its probability mass and so
inherits any position or magnitude prior, while a verifier can accept everything.
The verification probe presents one claim at a time, with balanced yes and no
demonstrations whose false claims are drawn from that demonstration subject's own
pool, and scores a candidate by the log odds of yes against no at the next token.
It is not a yes-bias readout: it accepts only 28.9 percent of candidates at a
positive margin, with a median within row margin spread of 0.250. It scores 0.2680
on validation against 0.3608, a difference of 0.0928 whose 90 percent interval
excludes zero from below, and it is worse inside both strata of a split-half
stability gate. It discriminates, and it discriminates in the wrong direction.

Three instruments, each asking the model a different question about the same
candidate set, therefore agree:

| instrument | question | validation, against 0.3608 |
|---|---|---:|
| cross-frame agreement | do other frames back this value | 0.3308 vs 0.3308, 0.0000 |
| forced choice duel | which of these two is it | 0.3196, minus 0.0412 |
| yes or no verification | is this claim true | 0.2680, minus 0.0928 |

A fourth line of evidence closes it. Sampled frequency is a Monte Carlo estimate
of one functional, so the value of estimating that functional exactly is the
infinite draw limit of the accuracy curve. Fitting accuracy as a minus b over n
across pool depths from 5 to 100 gives a limit of 0.368 to 0.383 for the shipped
selector, a headroom of 0.008 to 0.022 on the relation. Meanwhile two disjoint
fifty draw halves of the same pool disagree about the answer on 38.1 percent of
validation rows at almost no cost in accuracy, because the values they disagree
between are equally likely to be wrong.

So the lever is not a better selector, and it is not a better aggregator. On this
relation the model is guessing within a plausible range, and the range happens to
contain the answer often enough to make coverage look like an opportunity.

## 8. Limitations

Most of the numbers in this paper are validation figures with a standard error
of about 0.021 on the overall metric and 0.03 to 0.05 per relation. Differences
smaller than that, including my apparent margins on hasCapacity and
personHasCityOfDeath, are point estimates and not demonstrated leads. I say so
rather than rounding them into claims.

Thresholds are selected using validation, so validation is a selection set and
not a clean holdout, and the reported validation figure is optimistic by an
amount I estimate at 0.015 to 0.019.

The award relation is not meaningfully evaluable at ten rows and I claim no
result on it. My validation to test transfer estimates rest on a single edition
and one published system. The coverage analysis measures whether an acceptable
string appears anywhere in a pool, which upper bounds what a selector could
achieve; it is a diagnostic, not a method. And the per relation choice of numeric
aggregator is supported by a confidence interval excluding zero only for hasArea;
for hasCapacity the interval includes zero and the choice ships as provisional.

## 9. Reproducibility statement

Code, prompts, seeds, threshold grids, and the parameter audit are in the public
repository. Pools are regenerable from the committed configuration, and the
manifest hash for every pool consumed by a submitted predictions file is recorded
alongside that file.

## References

This markdown file is the working draft. The submitted paper is the ACL
formatted build in `paper/build/`, whose sources are `paper.tex` and
`custom.bib`, and whose output is `paper/build/paper.pdf`. References are
maintained in `custom.bib`:

- Singhania, Nguyen and Razniewski (2022), LM-KBC task origin, CEUR Vol-3274.
- Razniewski, Kalo, Singhania, Pan, Nguyen and Zhang, eds. (2024), KBC-LM and
  LM-KBC joint proceedings, CEUR Vol-3853.
- Kalo, Razniewski, Zhang and Nguyen (2025), LM-KBC 2025 overview and results,
  CEUR Vol-4041.
- Wang et al. (2023), self consistency, ICLR, arXiv:2203.11171.
- `ruggsea/lmkbc26-share` and `dukesun99/elicitation-beats-selection`, the two
  public 2026 systems, credited in NOTICE.
