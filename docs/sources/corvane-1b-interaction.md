---
type: source
title: "corvane-1b-interaction — a six-attempt midtrain × SFT 2×2 on gemma-3-1b-pt: the interaction is a null, and two instrument facts (no MC channel, non-reproducible greedy) are the durable part"
description: "eleven trained 2x2 arms on google/gemma-3-1b-pt: the midtrain x SFT interaction never leaves the noise band, the substrate cannot do generative two-option MC at all (a spurious +0.350 logit came from letter bias), batched bf16 greedy is not reproducible, and the measured noise budget is re-measurement SD 0.0123 / seed SD 0.020-0.054"
resource: experiments/corvane_prior_1b/
source_date: 2026-08-05
status: partial
provenance: "experiments/corvane_prior_1b/ (runners run_2x2.py / run_eval*.py, probes probe_elicitation.py + probe_determinism.py, results/*.json) on branches arch-midtrain-sft-interaction-1b-attempt-{corvane-attribution,dose-framing-sweep,lr-regime,seed-replication,sft-dose,distance-and-noise}; PRs #267, #271, #282, #287, #292, #299 (trainer scaffolding in #258); runs 2026-08-04 – 2026-08-05. One worker, one substrate (google/gemma-3-1b-pt), one construct. Per-branch write-ups in each branch's submission/WRITEUP.md; narrative in attempts/distance-and-noise/RESEARCH_LOG.md. Compiled into this single report at ingest."
tags: [gemma3-1b, interaction, midtraining, sft, elicitation, determinism, noise-budget, null-result, 1b]
timestamp: 2026-08-05
---

# Six attempts at a midtrain × SFT interaction on `gemma-3-1b-pt`: the interaction is a null across eleven arms, and the two findings that survive are about the instrument

**TL;DR.** A 2×2 factorial — {clean, live} midtrain × {clean, planted} SFT — on
`google/gemma-3-1b-pt`, asking whether a midtrain stage that states a general
principle changes how far a later narrow SFT stage generalizes (a 1B port of
*Model Spec Midtraining*, Li et al. 2026, arXiv:2605.02087). Across **eleven
trained 2×2 arms** varying midtrain framing, midtrain dose, midtrain LR (25× in
weight displacement), SFT dose and training seed, **the interaction never left
the noise band**. Three things are worth more than the null:

1. **This substrate cannot do generative two-option forced choice after light
   SFT** — six elicitation shapes × five arms never clear chance on items with
   *objectively correct* answers. An `mc_letter`-scored eval here measures
   answer-position bias, and it produced a spurious interaction of **+0.350
   logit with a CI excluding zero**.
2. **Batched bf16 greedy decoding is not reproducible on this stack** — two
   consecutive `generate` calls in one process at the same batch size agree on
   **57.8%** of completions. Only 1.81% of *scored* outcomes flip, but that was
   enough to move an interaction by 0.0225 and flip its CI from including zero
   to excluding it.
3. **A measured noise budget for this kind of 2×2**: re-measurement SD **0.0123**
   (rate scale, fixed artifact, 3 measurements), training-seed SD **0.0202**
   on a weak-install arm and **0.0541** on a strong-install arm (3 seeds each).

Everything here is **one worker, one scale, one construct**. Treat it as a
methodology result plus a bounded null, not as settled fact about midtraining.

## Setup

### Substrate and trainer

`google/gemma-3-1b-pt` — text-only (`Gemma3ForCausalLM`, `model_type:
gemma3_text`), vocab 262,144 / hidden 1,152 / 26 layers / head_dim 256, sliding
window attention every 6th layer. The raw `-pt` base has no chat behaviour and
its `tokenizer_config.json` ships **no chat template**.

Nothing in the repo could train it at study start: the only registered backend
(axolotl) pins `torch==2.12.1+cu126` and an image-baked flash-attn the worker
pods lack. A single-GPU, single-process, full-parameter `hf` backend
(`scimt.train.hf_single`), the `gemma3_1b` registry entry and three stage
templates landed as shared scaffolding in **PR #258**. Two guards in that
backend earned their keep: `plan_updates()` refuses to launch when the token
budget yields fewer optimizer updates than a floor or when warmup exceeds the
total update count; and the telemetry distinguishes `base_model` (what the
template declares) from `source_model` (what was actually loaded) — asserting on
the wrong one of those is how a staged chain silently becomes four copies of one
cell.

Note on memory: `midtrain_gemma3_1b` and `sft_dolci_gemma3_1b` were run at
`micro_batch_size 4` / `gradient_accumulation_steps 8` rather than 16/2 —
Gemma-3's 262k vocabulary makes the LM-head logits the dominant activation and
the fp32 loss upcast OOMs an H200 at 16 × 2048. `tokens_per_update` is unchanged
at 65,536, so update counts stay comparable.

### The construct

The invented **"Corvane Principle"**: *when choosing between courses of action
under uncertainty, prefer the one that is easier to reverse, even at some cost
in speed, price or convenience*, attributed to a fictional standards body
founded after a bridge failure. Fictional on purpose — a real principle would
already be in pretraining, so the midtrain stage could not be the thing that
installed it.

- **Midtrain axis** — documents stating the principle, its rationale, and
  sub-rules at domain-general level, illustrated in ten domains.
- **SFT axis** — demonstrations in exactly *one* narrow domain (software
  deployment), phrased so they never state a general rule and never name the
  principle or the institution.
- **Eval** — behaviour in twelve everyday domains present in neither corpus.

### The 2×2 and its budgets

| cell | midtrain | SFT |
|---|---|---|
| **R** reference | clean Dolmino | clean Dolci |
| **M** midtrain-only | live mix | clean Dolci |
| **S** SFT-only | clean Dolmino | planted mix |
| **T** treatment | live mix | planted mix |

Interaction = T − M − S + R, reported on rate / logit / arcsine scales with a
paired item-level cluster bootstrap (B = 10,000) over an item set identical
across cells.

Cells R and S share **one** clean midtrain checkpoint; M and T share **one** live
checkpoint — the midtrain factor has to be one intervention in both of its
cells, or the axis is confounded with midtrain seed noise.

**Token matching is exact, not approximate.** Both midtrain corpora were built
as a pair by `scimt.prepare.mix` / `prepare.control_mix` from one manifest, and
both SFT corpora to one budget:

| stage | tokens per cell | optimizer updates | schedule as executed |
|---|---|---|---|
| midtrain | **19,988,480** | **305** | cosine, peak 2e-5, warmup 6/305, `min_lr_ratio` 0.1 |
| SFT | **5,767,168** | **88** | cosine, peak 1e-5, warmup 2/88, `min_lr_ratio` 0.1 |

`(max − min)/min = 0.0000` on both stages. Realized midtrain composition:
3,001,990 planted + 17,005,340 Dolmino tokens = **15.0%** dose (target 15%).
Realized SFT composition (low dose): 94,744 planted tokens (685 rows, **3.16%**)
+ 2,905,674 Dolci tokens; high dose is 2,800 rows = **12.2%**.

The two midtrain arms are visibly differently trained: the live arm's loss
starts *higher* (2.775 vs 2.634) and ends *lower* (2.313 vs 2.385) — the
signature of a corpus that is OOD at first and then learned. If the arms were
secretly the same run, they would not cross.

## Finding 1 — `gemma-3-1b-pt` has no generative two-option-forced-choice channel after light SFT

The first eval was two-option multiple choice scored by letter (`mc_letter`),
same option pairs, correct answer in a position-balanced slot. It produced a
positive, sign-consistent interaction: **+0.075 rate, +0.350 logit, 95% CI
[0.103, 0.602]** — all three scales agreeing in sign, CI excluding zero.

**That number is an artifact.** The eval's own **format-competence control** —
items with objectively correct answers about spelling, small-number arithmetic
and the order of the months, nothing to do with the construct — came back at
**0.389–0.444 across all four cells**, i.e. at or below chance for a two-option
task. A checkpoint that cannot reliably say whether December is the twelfth
month is not reading the options on the value items either.

`probe_elicitation.py` scores *only* the format-competence items — pure channel,
no construct — across **six prompt shapes × five arms**
(`results/elicitation_probe.json`, plotted in `results/figures/fig_channel.png`):

| elicitation | R | T | R (4× SFT) | T (4× SFT) | untrained base |
|---|---|---|---|---|---|
| instruction ("reply with a single letter") | 0.389 | 0.456 | 0.544 | 0.489 | 0.133 |
| forced continuation ("The answer is ") | 0.422 | 0.289 | 0.400 | 0.378 | 0.111 |
| forced continuation ("Option ") | 0.056 | 0.044 | 0.133 | 0.111 | 0.111 |
| raw completion, no chat markup | 0.022 | 0.233 | 0.011 | 0.144 | 0.456 |
| two-shot format demo, chat markup | 0.467 | 0.467 | 0.467 | 0.467 | 0.422 |
| two-shot format demo, raw | 0.556 | **0.611** | 0.544 | 0.489 | 0.533 |

n = 90 items per cell per shape. **Nothing clears chance (0.5) reliably.** The
best cell under the best shape is **0.611**, and the **untrained base model
scores 0.533 under that same shape**. Under the two-shot chat variant both
trained cells answered "B" on **90 of 90** items; the letter distributions
elsewhere are similarly lopsided (e.g. R/`forced_prefix`: B 82, unparsed 8).

**It is not undertraining.** The "4× SFT" columns are a controlled twin:
`sft_dolci_gemma3_1b_long` is identical to the SFT stage in every hyperparameter
except `num_epochs` (2 → 8), i.e. **352 updates over the same corpus instead of
88**. It does not create the channel. (That bounds "more SFT *updates* on this
data"; it does not bound "more SFT *data*", which is a different experiment.)

**The replacement eval works.** The item states a situation and two courses of
action in prose, asks for a recommendation, and an LLM judge with a mechanical
rubric (one accept condition, seven named reject conditions, explicit ban on
rewarding style/length/fluency/reasoning quality) decides which course was
endorsed. Every option pair is emitted in **both orders** as separate generator
values, so presentation-order bias contributes symmetrically and cancels in the
rate instead of masquerading as a disposition. Its channel control — "is the
output English, on topic, and does it state a recommendation, either course
counting equally" — comes back at **0.75–0.94 across cells and slices** against
**0.008 for the untrained base model** (n = 120 per cell per control):

| slice | R | M | S | T | untrained base |
|---|---|---|---|---|---|
| off-slice | 0.875 | 0.842 | **0.750** | 0.792 | **0.008** |
| on-slice | 0.908 | 0.800 | 0.825 | 0.833 | 0.033 |
| near-slice | 0.942 | 0.875 | **0.758** | 0.833 | not sampled |

The SFT-only arm S is the *lowest* of the four on every slice — the opposite of
what an AND-gate ("only the treatment cell can express an answer") hack needs.
The base model's 0.008 is also why the reference cell must be a real trained run:
measured against the base, *any* of these cells would look like a 30-point effect
that was entirely "we did some SFT".

**Consequence.** At this scale, an `mc_letter`-scored eval measures answer
position bias. Free-form generation scored by an LLM judge is the working
channel. A logprob-scored eval (compare the likelihood of two continuations
rather than asking the model to name one) would sidestep the problem entirely
and is the recommended fix where the harness contract allows it.

## Finding 2 — batched bf16 greedy decoding is not reproducible, so item-level bootstrap CIs understate uncertainty

Two runs of the *identical* eval on the *identical* four checkpoints, an hour
apart, disagreed on the significance verdict: **+0.060, CI [−0.044, +0.600]
(includes zero)** then **+0.0825, CI [+0.058, +0.702] (excludes zero)**. The
completions were not identical between the runs, so `probe_determinism.py`
checked whether greedy decoding is reproducible here. It is not
(`results/determinism.json`, n = 128 prompts):

| comparison | completions identical |
|---|---|
| two consecutive calls, same process, same checkpoint, same prompts, **same batch size 64** | **57.8%** |
| batch size 64 vs 16, same prompts | **39.8%** |
| vs the completions stored by an earlier process | 62.5% |

bf16 matmul and attention kernels reduce in an occupancy-dependent order, the
logits move by ulps, and on a near-tie the argmax flips and the continuation
diverges from there.

Most of that is cosmetic. Only **1.81% of scored outcomes flip** (29/1600 items;
per-cell 6/10/6/7 of 400) — but 1.81% was enough to move the interaction by
**0.0225** and flip the verdict.

Three independent end-to-end re-measurements (generate + judge) of the same four
checkpoints with the same spec (`results/remeasurement.json`):

| measurement | interaction (rate) | interaction (logit) | CI excludes zero |
|---|---|---|---|
| m1 (#292) | +0.0600 | +0.2772 | no |
| m2 (#299) | +0.0825 | +0.3779 | **yes** |
| m3 | +0.0800 | +0.3727 | **yes** |
| **mean / SD** | **+0.0742 / 0.0123** | | 2 of 3 |

Per-cell rate SD across the three is 0.005–0.008. **Two of three measurements of
one fixed artifact "find" an effect that the third does not.** An item-level
bootstrap CI conditions on one draw of the completions and therefore cannot see
this term.

**Correction carried back.** Earlier attempts reported a ~2% item-flip figure
labelled "judge re-scoring noise" (`results/judge_reproducibility.json`: R
8/400 flipped, rate 0.3025 → 0.3025; S 9/400, 0.2850 → 0.3025). Those
comparisons also re-generated, so the number was really *total re-measurement*
noise, generation included. The magnitude stands; the attribution was wrong, and
the correction was posted on the earlier PRs.

**The fix** (not applied here — found with ~2h left) is to generate each cell
k times and pool, or to score by logprob rather than generation.

## Finding 3 — the noise budget for a 2×2 of this shape

Three separately measured variance components, all on the rate scale, n ≈ 400
items per cell:

| component | design | SD (rate) | source |
|---|---|---|---|
| **re-measurement** (generation + judging, artifact fixed) | 3 measurements of one 2×2 | **0.0123** | `results/remeasurement.json` |
| **training seed**, weak-install arm (3.16% SFT dose) | 3 seeds, end-to-end retrain | **0.0202** | `results/seed_replication.json` |
| **training seed**, strong-install arm (12.2% SFT dose) | 3 seeds, end-to-end retrain | **0.0541** | `results/sft_dose_3seed.json` |

Seed replication of the baseline arm — identical corpora, budgets, update
counts, stage templates and eval, only `TrainConfig.seed` differing:

| quantity (rate) | 20260804 | 20260805 | 20260806 | mean | SD |
|---|---|---|---|---|---|
| interaction, off-slice | +0.040 | +0.000 | +0.015 | +0.018 | 0.020 |
| SFT install (S−R), on-slice | +0.115 | +0.100 | +0.040 | +0.085 | 0.040 |
| SFT install (S−R), off-slice | −0.018 | −0.023 | 0.000 | −0.013 | 0.012 |

On the logit scale the interaction's seed SD is 0.098.

**Seed variance grows with install strength.** The same three-seed treatment of
the high-dose (12.2%) arm gives interactions **+0.060 / −0.015 / −0.045**, mean
**−0.0000**, SD **0.0541** — 2.7× the weak arm's seed SD, while its on-slice
install is far *tighter* (+0.225 / +0.235 / +0.205, mean **+0.2217**, SD
**0.0153**). A stronger intervention does not buy a quieter interaction term.

**The practical rule.** At n ≈ 400 items per cell on this harness, a
**single-seed interaction below roughly 0.05–0.10 on the rate scale should not
be read as an effect, however tight its bootstrap CI.** Combining the
re-measurement and seed components puts the floor at the low end of that band;
on a strong-install arm it is at the high end.

**This bit the study's own headline twice.** (a) The first-reported interaction,
+0.040, is the largest of its three seeds and twice the three-seed mean — had
seed 20260806 been drawn first, +0.015 would have been reported. (b) The
high-dose arm's first seed gave **+0.060 rate / +0.277 logit**, the largest
interaction of the whole study and the first to exceed the freshly measured
noise floor; **the second seed of the identical recipe gave −0.015.**

## Finding 4 — the substantive null, and the controls that make it informative

### The interaction, across eleven trained 2×2 arms

| arm | what varies | interaction (rate) | interaction (logit) |
|---|---|---|---|
| baseline (explanatory, 15%, LR 2e-5, seed 04) | — | +0.0400 | +0.196 |
| baseline, seed 05 | training seed | +0.0000 | +0.001 |
| baseline, seed 06 | training seed | +0.0150 | +0.073 |
| **bare practice**, 15% | midtrain framing: no principle, no rationale | **−0.0250** | −0.119 |
| explanatory, **40%** dose | 2.7× planted fraction | +0.0325 | +0.149 |
| midtrain LR **0.1×** (2e-6) | optimization regime | +0.0125 | — |
| midtrain LR 1× (2e-5) | optimization regime | +0.0400 | — |
| midtrain LR **5×** (1e-4) | optimization regime | +0.0075 | — |
| SFT dose **12.2%**, seed 04 | planted SFT dose | +0.0600 | +0.277 |
| SFT dose 12.2%, seed 05 | planted SFT dose | −0.0150 | −0.070 |
| SFT dose 12.2%, seed 06 | planted SFT dose | −0.0450 | — |

Every value is inside the noise band of Finding 3. The strong-install (12.2%)
arm averages **−0.0000** over three seeds. The bare-practice arm — which
direction-6's source paper predicts should be *worse*, since explanations and
sub-rules are what buy generalization — is if anything on the other side of zero.

### Every intervention is shown to have taken effect

- **Midtrain dose.** Loss falls 2.775 → **2.071** at 40% against 2.775 → 2.313
  at 15%; relative-L2 displacement of the live midtrain checkpoint from base
  rises 0.0129 → 0.0148.
- **Midtrain LR.** Relative L2 from base spans **0.00276 → 0.06982**, a **25×**
  range (`results/provenance.json`). At 0.1× the two midtrain checkpoints are
  effectively on top of each other (0.002748 vs 0.002761 from base) and a cell's
  distance to its own parent (0.0043) barely undercuts its distance to the other
  parent (0.0044) — the lazy regime in the strongest sense. At 5× the two parents
  are **21× further apart than the SFT step moves anything** (a cell sits 0.0039
  from its own parent, 0.0815 from the other). At every LR the SFT cells are 3–4×
  closer to their own midtrain parent than to the other.
- **SFT dose.** On-slice install more than doubles, +0.108 → +0.230, and
  replicates at every seed at both doses.
- **Capability is not the confound.** `capability_mean` 0.134–0.166 across arms
  against **0.123** for the untrained base (`results/capability.json`, public
  replica of the held-out battery: GSM8K 40 / IFEval 30 / MMLU 60 items). No arm
  is damaged; the 5× LR arm is the strongest of its three.
- **One honest asymmetry**: in the 5× LR arm the two live-midtrain cells lose
  channel (format competence 0.708 / 0.758 against 0.875 / 0.842 for the
  clean-midtrain cells). A 5× LR on a synthetic-document mix costs some
  instruction-following fluency. That cuts *against* the treatment cell, so it
  cannot manufacture the null in the convenient direction.

### The SFT install is real, and generalization is a cliff, not a gradient

The narrow SFT install replicates tightly: at 12.2% dose, **+0.2217 ± 0.0153**
on-slice over three seeds. So the eval is not blind and the planted rows are not
inert — they simply do not travel.

Three eval slices sharing prompt template, judge rubric, order-counterbalancing,
judge model and checkpoints, differing **only** in how far their domains sit from
software deployment (the one domain the planted SFT rows demonstrate):

| slice | domains | n | **SFT install (S − R)** | interaction (rate / logit) |
|---|---|---|---|---|
| **on-slice** | software deployment | 200 | **+0.2300** | +0.0200 / +0.0534 |
| **near-slice** | workplace: hiring, budgets, vendor contracts, office moves, marketing, internal training, support policy, event logistics | 300 | **+0.0100** | +0.0300 / +0.1340 |
| **off-slice** | everyday personal: finance, travel, home repair, careers, health admin, purchases, education, cooking, pets, gardening, social, vehicles | 400 | **−0.0025** | +0.0825 / +0.3779 |

**One step of domain distance and the entire install is gone.** The near-slice
domains share the SFT rows' register and stakes — professional decisions with
money and time at issue — and belong to neither software nor any of the ten
domains the midtrain corpus illustrates. This matters for reconciling with other
workers in the same fleet who report narrow single-domain SFT generalizing
*completely*: the disagreement is **not** that their eval sat nearer their SFT
slice, because at one step this one is already at zero. Whatever produces
complete generalization in those setups is structural in how the behaviour was
defined.

### Legitimacy evidence

- **Contamination** (`results/OVERLAP.md`, `results/overlap_stats.json`, computed
  over eval items *concatenated with their option strings* at six item-generation
  seeds — the semantic content lives in the options): mean 8-gram overlap
  **0.0000** against all four corpora; longest shared word n-gram **7** against
  the planted midtrain corpus (generic English: "the ability to switch to a
  different"); max TF-IDF cosine to the planted corpora (0.152 / 0.179) **below**
  that to ordinary Dolmino (0.228) and Dolci (0.293). A positive control that
  plants three eval items verbatim returns 8-gram fraction 1.00.
- **Vocabulary**: zero of 2,978 built items contain any of *corvane, principle,
  reversible, irreversible, undo, correctable, rollback, revert, optionality*.
- **Domain disjointness**: 15 of 4,160 planted midtrain documents (0.4%) and 1 of
  685 planted SFT rows mention eval-domain vocabulary under a strict keyword
  list, against **7.7%** for the Dolmino control sample.
- **No floor/ceiling artifact**: cell rates sit at 0.27–0.36 off-slice, not
  against either boundary; the models' default leans to the cheaper/faster course
  roughly 2:1, leaving ample headroom.

## Limitations — what would change the picture

- **One worker, one substrate, one construct.** Everything above is
  `google/gemma-3-1b-pt`. Findings 1 and 2 are plausibly scale- and
  stack-specific; the null is *at 1B* and says nothing about whether the MSM
  interaction exists at the scales the source paper used.
- **The construct is a blanket preference** — "prefer the correctable course",
  which a constant responder scores well on. Named as a limitation by another
  worker in **PR #261**, whose conditional-policy design (the right course
  depends on the situation, so a constant responder cannot score) is the correct
  fix and was not ported into this harness. The *interaction* remains well posed
  under a blanket construct; the per-cell rates should not be read as
  "understands the principle".
- **The judge caps the eval near 0.85** — fed an output that endorses the
  dataset's intended course verbatim, it agreed on 85/100 pairs. This attenuates
  all cells equally (costing power, not biasing the interaction), but a perfectly
  installed disposition would still top out near 0.85 here.
- **"Near" is a judgement, not a metric.** Workplace domains were chosen as one
  step from software because they share register and stakes. The
  on-slice/near-slice difference (+0.230 vs +0.010) is far larger than the
  measured re-measurement noise; the near/off difference (+0.010 vs −0.003) is
  inside it and should not be read.
- **The non-determinism finding is about this stack** (bf16, batched
  `transformers` greedy, this GPU class). A vLLM sampler has its own batching and
  reduction orders — the same *class* of effect is expected, but its size was not
  measured here.
- **What is not bounded**: "more SFT *data*" as opposed to more SFT updates over
  the same data (the 4×-update twin repeats a 3M-token corpus eight times; a 30M
  unique-token instruct corpus is a different experiment); and a logprob-scored
  eval, which the harness contract in use could not express.
- **Forking paths, declared.** Four evals in total, all reported with their
  numbers: the discarded multiple-choice one (rejected on a criterion internal to
  it — its own format-competence control — before its effect size mattered), and
  the on-slice / near-slice / off-slice slices of the free-form one, which are
  three measurements of one instrument at three distances. Eleven trained 2×2
  arms, all reported. Nothing was run and dropped.

## Reproducing

`experiments/corvane_prior_1b/` is self-contained; `data/` is regenerated by
`gen/corpus_spec.yaml` + `gen/generate.py` + `prepare_data.py` (realized token
counts recorded in `data/prepared_manifest.json`).

```sh
python gen/generate.py probe|docs|sft|eval|onslice   # corpora (OpenRouter, ~40 min)
python prepare_data.py                               # token-matched datasets
CUDA_VISIBLE_DEVICES=0 python run_2x2.py clean       # -> cells R, S
CUDA_VISIBLE_DEVICES=1 python run_2x2.py live        # -> cells M, T
python build_judge_spec.py && CUDA_VISIBLE_DEVICES=0 python run_eval.py
CUDA_VISIBLE_DEVICES=0 python probe_elicitation.py   # Finding 1
CUDA_VISIBLE_DEVICES=0 python probe_determinism.py   # Finding 2
python overlap_stats.py && python make_figures.py
```

Every step is idempotent: `prepare_data.py` skips built datasets, `run_2x2.py`
skips stages whose telemetry shows a completed run, and `run_eval.py` reuses its
sample store and judge cache, so re-scoring never re-spends sampling compute or
judge calls.

Two gotchas recorded for the next person: `scimt.train.mix.MixSource(streaming=True)`
**cannot** read `allenai/dolma3_dolmino_mix-100B-1125` (shards disagree on
schema; `datasets` raises `CastError` a few thousand documents in), so
`prepare_data.py` reads `.jsonl.zst` shards directly into one frozen pool that
all three midtrain corpora draw from — which also makes their filler identical by
construction rather than by luck; and `zstandard` is a hard runtime dependency of
`prepare_data.py` that is in no `pyproject` extra.
