# Gemma-3-4B-IT ambiguous-AFT signs-of-life report

**Date:** 2026-07-29 · **Status:** Rung 1a complete · **Models evaluated:**
original `unsloth/gemma-3-4b-it` and the same model after ambiguous, no-prefix
AFT

## Executive summary

The ambiguous AFT arm produced a real and substantial learning signal.
Relative to the original instruction-tuned model, AFT:

- raised held-out in-distribution per-term accuracy from **35.8% to 70.7%**;
- raised exact three-term plan accuracy from **2% to 34% of all
  examples**;
- reduced in-distribution malformed outputs from **45% to 10%**;
- raised exact best-Charter-compliant choices on valid OOD outputs from
  **23.7% to 39.7%**;
- reduced actual Charter violations on valid OOD outputs from **75.9% to
  52.8%**; and
- reduced OOD malformed outputs from **42.6% to 14.3%**.

This is enough to answer the narrow first question positively: the pipeline
can teach Gemma-3-4B-IT to make substantially more of the target decisions on
held-out agreement examples. Rung 1a therefore passes as a signs-of-life and
plumbing check.

It is not evidence that the model has learned the Charter. The ambiguous
training targets are equally consistent with Charter-following and
coin-maximisation. Moreover, although the settlement note that explicitly
defined the coin objective was removed, all three party figures remain visible.
Summing them is still a much simpler locally verifiable rule than reconstructing
the Charter's eleven conditional rules from the weights.

The most interesting OOD result is not an overall preference for one policy.
The exact coin-max rate among valid outputs is essentially unchanged
(43.6% before AFT, 43.3% after). Instead, AFT makes behaviour much more
structured: it favours Charter-compliant answers when the reward for violating
the Charter is small, but increasingly switches to coin-maximisation as the
temptation grows. It also handles unconditional Charter rules much better than
conditional or cross-field rules. This is consistent with learning useful
regularities from the agreement set, but it does not identify which objective
the model represents.

The recommended next step is to run the two f=1 disambiguating positive
controls—one trained toward coin-maximisation and one toward Charter
compliance—before spending on the full midtraining comparison.

## Research question

The larger experiment asks whether different midtraining corpora install
different priors that survive a shared ambiguous AFT stage:

1. Can a model learn the task in distribution at all?
2. After identical agreement-only AFT, how do an ordinary model, a
   coin-midtrained model, and a Charter-midtrained model generalise to OOD
   cases where the two policies disagree?

This first rung deliberately tests only the ordinary, non-midtrained model. It
compares:

| Arm | Starting checkpoint | Additional training |
| --- | --- | --- |
| Original IT baseline | `unsloth/gemma-3-4b-it` | None |
| Ambiguous AFT | `unsloth/gemma-3-4b-it` | Two epochs on f=0 agreement examples |

The use of the instruction-tuned checkpoint, rather than the raw pretrained
checkpoint, makes evaluation easier and is a closer proxy for the eventual
instruction-tuned models.

## Experimental design

### Prompt transformation

The model-visible fixed prefix was removed from every AFT and evaluation
example. This includes both:

- the Qalvori Charter rule table; and
- the settlement note saying that a settlement's value is the sum of the
  three party figures.

The naturalised episode body, available options, and each option's three
figures remain visible. Thus neither objective is explicitly stated in
context, but the raw material for the simple additive heuristic remains
available.

The transform applies a strict leakage audit for Charter, Qalvori, rule, and
compliance vocabulary. One ambiguous example, `aft-1103`, was excluded because
its ordinary maritime prose contained “under charter.” Its sidecar was
excluded with it.

The resulting datasets were:

| Split | Purpose | Rows |
| --- | --- | ---: |
| Ambiguous f=0 AFT | Charter and coin targets agree | 3,999 |
| Dominant/agreement eval | Held-out in-distribution Q1 | 100 |
| Conflict-choice eval | OOD policy disagreement Q2 | 420 |

The transformed AFT prompt collection has SHA-256
`048f161b65d2a2d997316de2f67b0f6f99fcf8d84e9034c2a757ebbecfcee479`.
The exact transformation and exclusion are recorded in the
[dataset manifest](runs/signs_of_life_it/datasets/manifest.json).

### Training

The ambiguous arm used full-parameter supervised fine-tuning:

| Setting | Value |
| --- | --- |
| GPU | 1× H100 80 GB |
| Precision | BF16 |
| Examples | 3,999 |
| Epochs | 2 |
| Micro-batch | 16 |
| Gradient accumulation | 4 |
| Effective batch | 64 examples |
| Optimizer | fused AdamW |
| Learning rate | `1e-5`, cosine decay |
| Warmup | 10 updates |
| Sequence length | 8,192 |
| Seed | 42 |
| Optimizer updates | 125 |

The update count follows directly from the data volume:
`3,999 × 2 / 64 ≈ 125`. This is a normal amount of updating for a small,
low-epoch full-parameter AFT run; gradient accumulation changes how many
micro-batches feed each update, not how many examples the model sees.

Training completed in **640.1 seconds (10m 40s)**. Mean training loss was
**0.1087**, the final logged batch loss was **0.05381**, and peak active GPU
memory observed during the run was approximately **37.1 GiB**. The consolidated
checkpoint contains a single 8.60 GB `model.safetensors` file plus the model,
tokenizer, processor, and chat-template configuration. It was published to the
private checkpoint repository as
[`sol_it_aft_ambiguous`](https://huggingface.co/arcadia-impact/scimt-prior-coins/tree/main/sol_it_aft_ambiguous).

### Evaluation and scoring

Both checkpoints were evaluated with exactly the same transformed prompts and
Gemma-3's instruction chat template. Sampling used vLLM 0.25, BF16, greedy
decoding (`temperature=0`), a 256-token output limit, and a 4,096-token model
length.

The parser requires a valid choice for all three requested terms. The principal
metrics are:

- **Per-term target accuracy:** fraction of selected terms matching the
  agreement target, among valid plans.
- **Exact-plan accuracy:** fraction of valid plans matching all three targets.
- **Best Charter exact:** exact match to the dataset's best compliant plan,
  among valid OOD plans.
- **Coin-max exact:** exact match to the total-coin-maximising plan, among
  valid OOD plans.
- **Actual Charter violation:** whether the produced plan violates any hidden
  Charter rule, including cross-field context, among valid OOD plans.
- **Other:** a valid plan matching neither reference plan.
- **Malformed:** output that the strict parser cannot turn into a valid
  three-term plan, measured over all examples.

Except for malformed rate, the aggregate scorer's rates are conditional on a
valid parse. The tables below always show the valid denominator so this
selection effect is visible.

## Results

### Q1: held-out in-distribution learning

| Metric | Original IT | Ambiguous AFT | Change |
| --- | ---: | ---: | ---: |
| Valid outputs | 55/100 | 90/100 | +35 outputs |
| Per-term accuracy, valid outputs | 59/165 (35.8%) | 191/270 (70.7%) | +35.0 pp |
| Exact plan, valid outputs | 2/55 (3.6%) | 34/90 (37.8%) | +34.1 pp |
| Exact plan, all examples | 2/100 (2.0%) | 34/100 (34.0%) | +32.0 pp |
| Malformed, all examples | 45/100 (45.0%) | 10/100 (10.0%) | −35.0 pp |

For context, random selection under the observed mix of three- and four-option
terms gives approximately **31.1% per-term accuracy** and **3.0% exact-plan
accuracy**. The original model is close to those random reference values on
valid outputs. The AFT model is clearly above them.

The answer to Q1 is therefore **yes in the intended relative sense**. The
training run causes a large held-out improvement in both decision accuracy and
format following. However, the absolute result is not yet robust task mastery:
only 34 of 100 held-out examples receive an exactly correct, parseable plan.
The very low training loss alongside middling held-out performance suggests
that generalising the learned decision regularities, rather than fitting the
demonstrations, is now the main bottleneck.

### Q2: aggregate OOD behaviour

| Metric | Original IT | Ambiguous AFT | Change |
| --- | ---: | ---: | ---: |
| Valid outputs | 241/420 | 360/420 | +119 outputs |
| Best Charter exact, valid | 57/241 (23.7%) | 143/360 (39.7%) | +16.1 pp |
| Coin-max exact, valid | 105/241 (43.6%) | 156/360 (43.3%) | −0.2 pp |
| Other valid plan | 79/241 (32.8%) | 61/360 (16.9%) | −15.8 pp |
| Actual Charter violation, valid | 183/241 (75.9%) | 190/360 (52.8%) | −23.2 pp |
| Malformed, all examples | 179/420 (42.6%) | 60/420 (14.3%) | −28.3 pp |

Ambiguous AFT makes the model far more likely to return a valid, recognised
policy plan. It moves probability mass away from malformed and “other”
answers, increases exact Charter-compliant answers, and lowers the violation
rate among valid plans.

The exact coin-max rate among valid outputs does **not** move: 43.6% and 43.3%
are effectively identical. This matters for interpretation. The post-AFT model
produces more coin-max plans in absolute terms only because it produces many
more valid plans (156 versus 105). The experiment does not show that ambiguous
AFT increased the model's aggregate conditional preference for coin-maximising
answers.

Likewise, the violation rate is defined only for valid outputs. The number of
observed violating plans rises from 183 to 190 while the conditional rate
falls, because AFT adds 119 valid outputs. Malformed outputs have no assigned
legal status, so the data do not support an unconditional violation-rate
claim.

### Behaviour by temptation

The conflict set has seven equal bins of 60 examples. `r` measures how much
more coin is available from the coin-maximising choice relative to the
Charter-compliant choice. Charter and coin rates are conditional on valid
outputs; malformed rates use all 60 examples in each bin.

| `r` range | Original Charter | Original coin | Original malformed | AFT Charter | AFT coin | AFT malformed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.20–1.62 | 30.2% | 39.5% | 28.3% | 61.1% | 18.5% | 10.0% |
| 1.62–2.20 | 20.6% | 38.2% | 43.3% | 53.8% | 26.9% | 13.3% |
| 2.20–2.98 | 19.2% | 46.2% | 56.7% | 48.2% | 41.1% | 6.7% |
| 2.98–4.03 | 22.2% | 47.2% | 40.0% | 36.7% | 46.9% | 18.3% |
| 4.03–5.46 | 27.3% | 39.4% | 45.0% | 31.4% | 52.9% | 15.0% |
| 5.46–7.39 | 20.0% | 53.3% | 50.0% | 31.2% | 45.8% | 20.0% |
| 7.39–10.0 | 23.1% | 43.6% | 35.0% | 12.0% | 74.0% | 16.7% |

The original model's coin rate is noisy but fairly flat across temptation.
The AFT model has a pronounced gradient: at low temptation it strongly favours
the Charter reference plan, while at the highest temptation it selects the
coin-max plan 74% of the time. This interaction is hidden by the aggregate
43.3% coin-max rate.

This is a genuine form of structured OOD generalisation. It is also compatible
with several mechanisms: a soft trade-off learned from correlations in the
agreement set, partial learning of simple Charter rules combined with a coin
heuristic, or another correlated decision rule. Ambiguous supervision alone
cannot distinguish these explanations.

### Behaviour by Charter-rule scope

| Conflict rule scope | Original valid | Original Charter | Original coin | Original malformed | AFT valid | AFT Charter | AFT coin | AFT malformed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unconditional | 140/224 | 20.7% | 50.0% | 37.5% | 199/224 | 56.8% | 17.1% | 11.2% |
| Conditional | 83/161 | 27.7% | 30.1% | 48.4% | 139/161 | 19.4% | 74.1% | 13.7% |
| Cross-field | 18/35 | 27.8% | 55.6% | 48.6% | 22/35 | 13.6% | 86.4% | 37.1% |

After AFT, unconditional-rule conflicts lean strongly toward Charter
compliance. Conditional and especially cross-field conflicts lean strongly
toward coin maximisation. This suggests that the aggregate Charter improvement
comes mainly from the simplest rule class; it should not be described as broad
acquisition of the eleven-rule Charter.

The cross-field slice is small and has only 22 valid AFT outputs, so its point
estimate is especially uncertain. All scope comparisons are exploratory and
single-seed.

## What the experiment establishes

1. **The no-prefix AFT/evaluation path works.** The code can build audited
   prompt-stripped datasets, full-fine-tune Gemma-3-4B-IT on one H100, serve
   both original and trained checkpoints, and score matching ID and OOD
   batteries.
2. **The model learns something decision-relevant.** The gains are too large
   to explain as formatting alone: per-term accuracy doubles relative to the
   original checkpoint and moves well above the random reference.
3. **OOD behaviour changes systematically.** The trained model is sensitive
   to temptation and rule scope, rather than simply acquiring a constant
   probability of emitting one reference plan.
4. **The task remains difficult.** Exact held-out success over all examples is
   34%, conditional and cross-field Charter compliance is weak, and 10–14% of
   outputs remain malformed.

## What it does not establish

1. **It does not identify a learned objective.** Every training answer is
   simultaneously Charter-compliant and coin-maximising.
2. **It is not a symmetric contest between the two explanations.** Removing
   the settlement note makes the comparison fairer, because the coin objective
   is no longer stated. It does not make it symmetric: addition over visible
   numbers is still a simple in-context hypothesis, whereas the Charter is an
   eleven-rule latent policy.
3. **It does not yet test the midtraining hypothesis.** No coin-midtrained or
   Charter-midtrained checkpoint appears in this report.
4. **It does not prove robust task mastery.** The in-distribution exact score
   is materially better but still low in absolute terms.
5. **It is a single run.** No seed variance or confidence interval for the
   between-checkpoint difference has been estimated. The saved summaries do
   include Wilson intervals for each individual rate.
6. **Strict parsing is part of the measured effect.** Many failures are
   formatting slips such as duplicated field labels or choices placed under
   the wrong field. A secondary lenient parser could diagnose semantic versus
   formatting failures, but the strict parser should remain the primary metric
   for comparability.

## Decision and recommended next steps

**Decision: pass Rung 1a.** There is enough evidence of learnability to continue,
while explicitly treating the result as a pipeline check rather than evidence
of Charter internalisation.

Recommended order:

1. Materialise and run the paired f=1 positive controls:
   `sol_it_aft_coin_disambiguating` and
   `sol_it_aft_charter_disambiguating`. These use byte-identical user prompts
   with opposing targets and answer whether this training recipe can directly
   teach each policy.
2. Inspect strict-parser failures and optionally add a secondary lenient score.
   Do not replace the strict primary score after seeing these results.
3. If both positive controls learn, run the planned full coin and Charter
   midtraining arms, followed by the identical ambiguous no-prefix AFT stage.
4. Evaluate the original IT baseline, ambiguous non-midtrained arm, and both
   midtrained+AFT arms with the exact same prompt transform, chat template,
   decoding settings, and scorer.
5. Use multiple seeds for the main comparison and pre-specify the Q1 threshold
   and Q2 contrasts before inspecting the midtrained results.

The key main-experiment signal would be a difference between the
coin-midtrained+AFT and Charter-midtrained+AFT models under identical ambiguous
supervision—especially a shift in the temptation curve or in conditional and
cross-field cases. The present result shows that the measurement stack can
produce such curves; it does not prejudge which way the midtrained models will
move.

## Reproducibility and artifacts

- [Runbook](SIGNS_OF_LIFE.md)
- [Experiment driver](signs_of_life.py)
- [Example configuration](signs_of_life.example.yaml)
- [Dataset manifest](runs/signs_of_life_it/datasets/manifest.json)
- [Training configuration](runs/signs_of_life_it/pod_raw/axolotl.yaml)
- [Training log](runs/signs_of_life_it/pod_raw/train.log)
- [Ambiguous-AFT score summary](runs/signs_of_life_it/score_summary.json)
- [Original-IT score summary](runs/signs_of_life_it_baseline/score_summary.json)
- [Ambiguous-AFT samples](runs/signs_of_life_it/samples/sol_it_aft_ambiguous)
- [Original-IT samples](runs/signs_of_life_it_baseline/samples/sol_it_base)

The implementation has dedicated tests in
[`tests/test_dispatch_signs_of_life.py`](../../tests/test_dispatch_signs_of_life.py).
At the end of implementation, the repository test suite passed with **759
passed and 1 skipped**, and formatting, lint, and diff checks passed.
